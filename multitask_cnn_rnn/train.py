import os
import pickle
import statistics
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from data.custom_dataset_data_loader import MultitaskDatasetDataLoader
from models.models import ModelsFactory
from options.train_options import TrainOptions
from utils.logging_utils import save_plots

# RuntimeError: received 0 items of ancdata ###########################
from utils.model_utils import EarlyStop

torch.multiprocessing.set_sharing_strategy("file_system")
#########################################################################
torchaudio.set_audio_backend("sox_io")


class Trainer:
    def __init__(self):
        self._opt = TrainOptions().parse()

        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        train_transforms = self._model.resnet50_GRU.backbone.backbone.augment_transforms
        val_transforms = self._model.resnet50_GRU.backbone.backbone.compose_transforms
        self.training_dataloaders = MultitaskDatasetDataLoader(self._opt, train_mode='Train',
                                                               transform=train_transforms)
        self.training_dataloaders = self.training_dataloaders.load_multitask_train_data()
        self.validation_dataloaders = MultitaskDatasetDataLoader(self._opt, train_mode='Validation',
                                                                 transform=val_transforms)
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        print("Traning Tasks:{}".format(self._opt.tasks))
        actual_bs = self._opt.batch_size * len(self._opt.tasks)
        print("The actual batch size is {}*{}={}".format(self._opt.batch_size, len(self._opt.tasks), actual_bs))
        print("Training sets: {} images ({} images per task)".format(
            len(self.training_dataloaders) * actual_bs * self._opt.seq_len,
            len(self.training_dataloaders) * self._opt.batch_size * self._opt.seq_len))
        print("Validation sets")
        for task in self._opt.tasks:
            data_loader = self.validation_dataloaders[task]
            print("{}: {} images".format(task, len(data_loader) * self._opt.batch_size * self._opt.seq_len))

        self.visual_dict = {'training': pd.DataFrame(), 'validation': pd.DataFrame()}
        self.train_losses = {}
        self.early_stop = EarlyStop(self._opt.early_stop)
        self._train()

    def _train(self):
        self._total_steps = self._opt.load_epoch * len(self.training_dataloaders) * self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()
        self._current_val_acc = 0.
        if len(self._opt.pretrained_teacher_model) == 0:
            for i_epoch in range(self._opt.load_epoch + 1, self._opt.teacher_nepochs + 1):
                epoch_start_time = time.time()
                self._model.get_current_LR()
                # train epoch
                self._train_epoch(i_epoch)
                self.training_dataloaders.reset()
                val_acc = self._validate(i_epoch)
                if val_acc > self._current_val_acc:
                    print("validation acc improved, from {:.4f} to {:.4f}".format(self._current_val_acc, val_acc))
                    print('saving the model at the end of epoch %d, steps %d' % (i_epoch, self._total_steps))
                    self._model.save('teacher')
                    self._current_val_acc = val_acc
                self.save_visual_dict('teacher')
                self.save_logging_image('teacher')
                # print epoch info
                time_epoch = time.time() - epoch_start_time
                print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                      (i_epoch, self._opt.teacher_nepochs, time_epoch,
                       time_epoch / 60, time_epoch / 3600))

                if self.early_stop.should_stop(len(self._opt.tasks) - val_acc):
                    print(f'Early stop at epoch {i_epoch}')
                    break
            return
        else:
            self._model.resnet50_GRU.load_state_dict(torch.load(self._opt.pretrained_teacher_model))
        # record the teacher_model
        self._teacher_model = deepcopy(self._model)
        del self._model
        self._model = None
        self._teacher_model.set_eval()
        for i_student in range(self._opt.n_students):
            self._current_val_acc = 0.
            self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)  # re-initialize
            self.visual_dict = {'training': pd.DataFrame(), 'validation': pd.DataFrame()}
            for i_epoch in range(1, self._opt.student_nepochs + 1):
                epoch_start_time = time.time()
                self._model.get_current_LR()
                self._train_epoch_kd(i_epoch)
                self.training_dataloaders.reset()
                val_acc = self._validate(i_epoch)
                if val_acc > self._current_val_acc:
                    print("validation acc improved, from {:.4f} to {:.4f}".format(self._current_val_acc, val_acc))
                    print('saving the model at the end of epoch %d, steps %d' % (i_epoch, self._total_steps))
                    self._model.save('student_{}'.format(i_student))
                    self._current_val_acc = val_acc
                self.save_visual_dict('student_{}'.format(i_student))
                self.save_logging_image('student_{}'.format(i_student))
                # print epoch info
                time_epoch = time.time() - epoch_start_time
                print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                      (i_epoch, self._opt.student_nepochs, time_epoch,
                       time_epoch / 60, time_epoch / 3600))

                if self.early_stop.should_stop(len(self._opt.tasks) - val_acc):
                    print(f'Early stop at epoch {i_epoch}')
                    break

    def _train_epoch(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self.training_dataloaders):
            iter_start_time = time.time()
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s

            # train model
            self._model.set_input(train_batch)
            self._model.optimize_parameters()
            # torch.cuda.empty_cache()

            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            self.save_training_loss_to_visual_dict()

            # display terminal
            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.training_dataloaders))
                self._last_print_time = time.time()
            # if i_train_batch > 30:
            #     break

    def _train_epoch_kd(self, i_epoch):
        epoch_iter = 0
        self._model.set_train()
        for i_train_batch, train_batch in enumerate(self.training_dataloaders):
            iter_start_time = time.time()
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s

            # train model
            self._model.set_input(train_batch)

            if self._opt.use_shared_annotations:
                self._model.optimize_parameters_kd_v2(self._teacher_model)
            else:
                self._model.optimize_parameters_kd(self._teacher_model)
            # torch.cuda.empty_cache()

            # update epoch info
            self._total_steps += self._opt.batch_size
            epoch_iter += self._opt.batch_size

            self.save_training_loss_to_visual_dict()

            # display terminal
            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.training_dataloaders))
                self._last_print_time = time.time()
            # if i_train_batch > 30:
            #     break

    def update_train_losses(self, loss_dict):
        for key in loss_dict.keys():
            if key not in self.train_losses:
                self.train_losses[key] = []
            self.train_losses[key].append(loss_dict[key])

    def get_train_losses(self):
        return {key: statistics.mean(self.train_losses[key]) for key in self.train_losses.keys()}

    def clear_train_losses(self):
        self.train_losses = {}

    def save_training_loss_to_visual_dict(self):
        loss_dict = self._model.get_current_errors()
        self.update_train_losses(loss_dict)
        df = self.visual_dict['training']
        data = loss_dict
        self.visual_dict['training'] = df.append(
            pd.DataFrame(data, columns=list(data.keys()), index=[self._total_steps // self._opt.batch_size]))

    def save_validation_res_to_visual_dict(self, eval_res):
        df = self.visual_dict['validation']
        data = eval_res
        self.visual_dict['validation'] = df.append(
            pd.DataFrame(data, columns=list(data.keys()), index=[self._total_steps // self._opt.batch_size]))

    def save_visual_dict(self, save_name):
        save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, '{}.pkl'.format(save_name))
        pickle.dump(self.visual_dict, open(save_path, 'wb'))

    def save_logging_image(self, save_name):
        load_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, '{}.pkl'.format(save_name))
        visual_dict = pickle.load(open(load_path, 'rb'))
        train_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, save_name + '_train.png')
        val_path = os.path.join(self._opt.checkpoints_dir, self._opt.name, save_name + '_val.png')
        save_plots(visual_dict, train_path, val_path)

    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch, num_batches):
        errors = self.get_train_losses()
        t = (time.time() - iter_start_time)
        start_time = time.strftime("%H:%M", time.localtime(iter_start_time))
        output = "Time {}\tBatch Time {:.2f}\t Epoch [{}]([{}/{}])\t loss {:.4f}\t".format(
            start_time, t,
            i_epoch, i_train_batch, num_batches,
            errors['loss'])
        for task in errors.keys():
            output += '{} {:.4f}\t'.format(task, errors['{}'.format(task)])
        print(output)
        self.clear_train_losses()

    def _validate(self, i_epoch):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        eval_per_task = {}
        for task in self._opt.tasks:
            track_val_preds = {'preds': []}
            track_val_labels = {'labels': []}
            val_errors = OrderedDict()
            data_loader = self.validation_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                # evaluate model
                wrapped_v_batch = {task: val_batch}
                self._model.set_input(wrapped_v_batch, input_tasks=[task])
                outputs, errors = self._model.forward(return_estimates=True, input_tasks=[task])

                # store current batch errors
                for k, v in errors.items():
                    if k in val_errors:
                        val_errors[k] += v  # accmulate over iters
                    else:
                        val_errors[k] = v
                # store the predictions and labels
                track_val_preds['preds'].append(outputs[task][task])
                track_val_labels['labels'].append(wrapped_v_batch[task]['label'])
                # if i_val_batch > 30:
                #     break
            # normalize errors
            for k in val_errors.keys():
                val_errors[k] /= len(data_loader)
            # calculate metric
            preds = np.concatenate(track_val_preds['preds'], axis=0)
            labels = np.concatenate(track_val_labels['labels'], axis=0)
            B, N = preds.shape[:2]
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(preds.reshape(B * N, -1).squeeze(), labels.reshape(B * N, -1).squeeze())
            now_time = time.strftime("%H:%M", time.localtime(val_start_time))
            output = "{} Validation {}: Epoch [{}] Step [{}] loss {:.4f} Eval_0 {:.4f} Eval_1 {:.4f}".format(
                task, now_time, i_epoch, self._total_steps, val_errors['loss'], eval_items[0], eval_items[1])
            print(output)
            if task != 'VA':
                eval_per_task[task] = eval_res
            else:
                eval_per_task['valence'] = eval_items[0]
                eval_per_task['arousal'] = eval_items[1]

        print("Validation Performance:")
        output = ""
        for task in eval_per_task.keys():
            output += '{} Metric: {:.4f}   '.format(task, eval_per_task[task])
        print(output)
        # set model back to train
        self._model.set_train()

        self.save_validation_res_to_visual_dict(eval_per_task)
        return sum([eval_per_task[k] for k in eval_per_task])

    def _test(self):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        eval_per_task = {}
        for task in self._opt.tasks:
            track_val_preds = {'preds': []}
            track_val_labels = {'labels': []}
            val_errors = OrderedDict()
            data_loader = self.test_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                # evaluate model
                wrapped_v_batch = {task: val_batch}
                self._model.set_input(wrapped_v_batch, input_tasks=[task])
                outputs, errors = self._model.forward(return_estimates=True, input_tasks=[task])

                # store current batch errors
                for k, v in errors.items():
                    if k in val_errors:
                        val_errors[k] += v  # accmulate over iters
                    else:
                        val_errors[k] = v
                # store the predictions and labels
                track_val_preds['preds'].append(outputs[task][task])
                track_val_labels['labels'].append(wrapped_v_batch[task]['label'])
                # if i_val_batch > 30:
                #     break
            # normalize errors
            for k in val_errors.keys():
                val_errors[k] /= len(data_loader)
            # calculate metric
            preds = np.concatenate(track_val_preds['preds'], axis=0)
            labels = np.concatenate(track_val_labels['labels'], axis=0)
            B, N = preds.shape[:2]
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(preds.reshape(B * N, -1).squeeze(), labels.reshape(B * N, -1).squeeze())
            now_time = time.strftime("%H:%M", time.localtime(val_start_time))
            output = "{} Test {}: loss {:.4f} Eval_0 {:.4f} Eval_1 {:.4f}".format(
                task, now_time, val_errors['loss'], eval_items[0], eval_items[1])
            print(output)
            if task != 'VA':
                eval_per_task[task] = eval_res
            else:
                eval_per_task['valence'] = eval_items[0]
                eval_per_task['arousal'] = eval_items[1]

        print("Test Performance:")
        output = ""
        for task in eval_per_task.keys():
            output += '{} Metric: {:.4f}   '.format(task, eval_per_task[task])
        print(output)
        # set model back to train
        self._model.set_train()

        self.save_validation_res_to_visual_dict(eval_per_task)
        return sum([eval_per_task[k] for k in eval_per_task])


if __name__ == "__main__":
    trainer = Trainer()
