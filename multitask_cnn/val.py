import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.custom_dataset_data_loader import MultitaskDatasetDataLoader
from models.models import ModelsFactory
from options.train_options import TrainOptions

# RuntimeError: received 0 items of ancdata ###########################
torch.multiprocessing.set_sharing_strategy("file_system")
#########################################################################


class Trainer:
    def __init__(self):
        self._opt = TrainOptions().parse()
        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        val_transforms = self._model.resnet50.backbone.compose_transforms
        self.validation_dataloaders = MultitaskDatasetDataLoader(self._opt, train_mode ='Validation', transform = val_transforms)
        self.validation_dataloaders = self.validation_dataloaders.load_multitask_val_test_data()
        self.test_dataloaders = MultitaskDatasetDataLoader(self._opt, train_mode ='Test', transform = val_transforms)
        self.test_dataloaders = self.test_dataloaders.load_multitask_val_test_data()

        print("Validation sets")
        val_test_tasks = [task for task in self._opt.tasks if task != 'EXPR_VA']
        for task in val_test_tasks:
            data_loader = self.validation_dataloaders[task]
            print("{}: {} images".format(task, len(data_loader.dataset)))
        print("Test sets")
        for task in val_test_tasks:
            data_loader = self.test_dataloaders[task]
            print("{}: {} images".format(task, len(data_loader.dataset)))
        self.visual_dict = {'training': pd.DataFrame(), 'validation': pd.DataFrame()}
        self.train_losses = {}
        self._model.resnet50.load_state_dict(torch.load(self._opt.pretrained_teacher_model))
        val_score = self._validate(0)
        print('Validation score: {:.4f}'.format(val_score))
        test_res = self._test()
        print('Test score: {:.4f}'.format(test_res))

    def _validate(self, i_epoch):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        eval_per_task = {}
        for task in self._opt.tasks:
            if task == 'EXPR_VA':
                continue
            track_val_preds = {'preds': []}
            track_val_labels = {'labels': []}
            val_errors = OrderedDict()
            data_loader = self.validation_dataloaders[task]
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                wrapped_v_batch = {task: val_batch}
                self._model.set_input(wrapped_v_batch, input_tasks = [task])
                outputs, errors = self._model.forward(return_estimates=True, input_tasks = [task])

                # store current batch errors
                for k, v in errors.items():
                    if k in val_errors:
                        val_errors[k] += v  # accumulate over iters
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
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(preds, labels)
            now_time = time.strftime("%H:%M", time.localtime(val_start_time))
            output = "{} Validation {}:  loss {:.4f} Eval_0 {:.4f} Eval_1 {:.4f}".format(
                task, now_time, val_errors['loss'], eval_items[0], eval_items[1])
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

        return sum([eval_per_task[k] for k in eval_per_task])

    def _test(self):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        eval_per_task = {}
        for task in self._opt.tasks:
            if task == 'EXPR_VA':
                continue
            track_test_preds = {'preds': []}
            track_test_labels = {'labels': []}
            val_errors = OrderedDict()
            data_loader = self.test_dataloaders[task]
            for i_val_batch, test_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                wrapped_v_batch = {task: test_batch}
                self._model.set_input(wrapped_v_batch, input_tasks = [task])
                outputs, errors = self._model.forward(return_estimates=True, input_tasks = [task])

                # store current batch errors
                for k, v in errors.items():
                    if k in val_errors:
                        val_errors[k] += v  # accumulate over iters
                    else:
                        val_errors[k] = v
                # store the predictions and labels
                track_test_preds['preds'].append(outputs[task][task])
                track_test_labels['labels'].append(wrapped_v_batch[task]['label'])

            # normalize errors
            for k in val_errors.keys():
                val_errors[k] /= len(data_loader)

            # calculate metric
            preds = np.concatenate(track_test_preds['preds'], axis=0)
            labels = np.concatenate(track_test_labels['labels'], axis=0)
            metric_func = self._model.get_metrics_per_task()[task]
            eval_items, eval_res = metric_func(preds, labels)
            output = "{} Test: loss {:.4f} Eval_0 {:.4f} Eval_1 {:.4f}".format(task, val_errors['loss'],
                                                                               eval_items[0], eval_items[1])
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

        return sum([eval_per_task[k] for k in eval_per_task])


if __name__ == "__main__":
    trainer = Trainer()
