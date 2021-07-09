import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from path import PATH
from utils.model_utils import EXPR_Losses, VA_Losses, BackBone, Head, Model, EXPR_metric, \
    VA_metric, EXPR_VA_Loss
from .models import BaseModel

PRESET_VARS = PATH()
MODEL_DIR = PRESET_VARS.MODEL_DIR


def mean(data: list):
    return sum(data) / len(data)


class ResNet50(BaseModel):
    def __init__(self, opt):
        super(ResNet50, self).__init__(opt)
        self._name = 'ResNet50'
        self._output_size_per_task = {'EXPR': self._opt.EXPR_label_size,
                                      'VA': self._opt.VA_label_size * self._opt.digitize_num,
                                      'EXPR_VA': self._opt.EXPR_label_size + self._opt.VA_label_size * self._opt.digitize_num}
        self._criterions_per_task = {'EXPR': self._opt.EXPR_criterion, 'VA': self._opt.VA_criterion}
        self.lambdas_per_task = {'EXPR': self._opt.lambda_EXPR, 'VA': [self._opt.lambda_V, self._opt.lambda_A]}
        # create networks
        self._init_create_networks()
        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def _init_create_networks(self):
        """
        init current model according to sofar tasks
        """
        backbone = BackBone(self._opt)
        tasks = [x for x in self._opt.tasks if x != 'EXPR_VA']
        output_sizes = [self._output_size_per_task[x] for x in tasks]
        output_feature_dim = backbone.output_feature_dim
        hidden_size = self._opt.hidden_size
        classifiers = [Head(output_feature_dim, hidden_size, output_sizes[i]) for i in range(len(tasks))]
        classifiers = nn.ModuleList(classifiers)
        self.resnet50 = Model(backbone, classifiers, tasks)
        if len(self._gpu_ids) > 1:
            self.resnet50 = torch.nn.DataParallel(self.resnet50, device_ids=self._gpu_ids)
        if len(self._opt.pretrained_resnet50_model) > 0:
            if os.path.exists(self._opt.pretrained_resnet50_model):
                self.resnet50.load_state_dict(torch.load(self._opt.pretrained_resnet50_model))
                print("resnet50 model loaded from {}".format(self._opt.pretrained_resnet50_model))
            else:
                raise ValueError("path {} does not exist".format(self._opt.pretrained_resnet50_model))
        self.resnet50.cuda()

    def load(self):
        load_epoch = self._opt.load_epoch
        # load feature extractor
        self._load_network(self.resnet50, 'resnet50', load_epoch)
        self._load_optimizer(self._optimizer_F, 'F', load_epoch)

    def save(self, label):
        """
        save network, the filename is specified with the sofar tasks and iteration
        """
        self._save_network(self.resnet50, 'resnet50', label)
        # save optimizers
        self._save_optimizer(self._optimizer_F, 'F', label)

    def load_label(self, label):
        self._load_network(self.resnet50, 'resnet50', label)
        self._load_optimizer(self._optimizer_F, 'F', label)

    def _init_train_vars(self):
        self._current_lr_F = self._opt.lr_F  # feature extractor
        params_to_update = []
        for name, param in self.resnet50.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        if self._opt.optimizer == 'Adam':
            self._optimizer_F = torch.optim.Adam(params_to_update, lr=self._current_lr_F,
                                                 betas=(self._opt.F_adam_b1, self._opt.F_adam_b2))
        elif self._opt.optimizer == 'SGD':
            self._optimizer_F = torch.optim.SGD(params_to_update, lr=self._current_lr_F)
        self._LR_scheduler = self._get_scheduler(self._optimizer_F, self._opt)

    def _format_label_tensor(self, task):
        _Tensor_Long = torch.cuda.LongTensor if self._gpu_ids else torch.LongTensor
        _Tensor_Float = torch.cuda.FloatTensor if self._gpu_ids else torch.FloatTensor
        if task == 'EXPR':
            return _Tensor_Long(self._opt.batch_size)
        elif task == 'VA':
            return _Tensor_Float(self._opt.batch_size, self._output_size_per_task[task])
        elif task == 'EXPR_VA':
            return _Tensor_Float(self._opt.batch_size, self._output_size_per_task[task])

    def _format_sub_tensor(self, task):
        if task == 'EXPR':
            # if task == EXPR, sub will be VA
            return self._format_label_tensor('VA')
        elif task == 'VA':
            # if task == VA, sub will be EXPR
            return self._format_label_tensor('EXPR')
        elif task == 'EXPR_VA':
            # if task == EXPR_VA, sub will be the same
            return self._format_label_tensor(task)

    def _init_prefetch_inputs(self):
        self._input_image = OrderedDict(
            [(task, self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)) for task in
             self._opt.tasks])
        self._label = OrderedDict([(task, self._format_label_tensor(task)) for task in self._opt.tasks])
        self._sub = OrderedDict([(task, self._format_sub_tensor(task)) for task in self._opt.tasks])

    def _init_losses(self):
        # get the training loss
        criterions = {}
        criterions['EXPR'] = EXPR_Losses(self._opt)
        criterions['VA'] = VA_Losses(self._opt)
        criterions['EXPR_VA'] = EXPR_VA_Loss(self._opt)
        self._criterions_per_task = criterions
        self._loss = Variable(self._Tensor([0]))

    def set_input(self, input, input_tasks=None):
        """
        During training, the input will be only related to the current task
        During validation, because the current model needs to be evaluated on all sofar tasks, the task needs to be specified
        """
        tasks = self._opt.tasks if input_tasks is None else input_tasks
        for t in tasks:
            self._input_image[t].resize_(input[t]['image'].size()).copy_(input[t]['image'])
            self._label[t].resize_(input[t]['label'].size()).copy_(input[t]['label'])
            if 'sub' in input[t]:
                self._sub[t].resize_(input[t]['sub'].size()).copy_(input[t]['sub'])
            if len(self._gpu_ids) > 0:
                self._input_image[t] = self._input_image[t].cuda(self._gpu_ids[0], non_blocking=True)
                self._label[t] = self._label[t].cuda(self._gpu_ids[0], non_blocking=True)
                self._sub[t] = self._sub[t].cuda(self._gpu_ids[0], non_blocking=True)

    def set_train(self):
        self.resnet50.train()
        self._is_train = True

    def set_eval(self):
        self.resnet50.eval()
        self._is_train = False

    def forward(self, return_estimates=False, input_tasks=None):
        # validation the eval_task
        val_dict = dict()
        out_dict = dict()
        loss = 0.
        if not self._is_train:
            tasks = self._opt.tasks if input_tasks is None else input_tasks
            for t in tasks:
                with torch.no_grad():
                    input_image = Variable(self._input_image[t])
                    label = Variable(self._label[t])
                    output = self.resnet50(input_image)
                criterion_task = self._criterions_per_task[t].get_task_loss()
                loss_task = criterion_task(output['output'][t], label)
                if t != 'VA':
                    val_dict['loss_' + t] = loss_task.item()
                    loss += self.lambdas_per_task[t] * loss_task
                else:
                    loss_v, loss_a = loss_task
                    val_dict['loss_valence'] = loss_v.item()
                    val_dict['loss_arousal'] = loss_a.item()
                    l_v, l_a = self.lambdas_per_task[t]
                    loss += l_v * loss_v + l_a * loss_a
                if return_estimates:
                    out_dict[t] = self._format_estimates(output['output'])
                else:
                    out_dict[t] = dict([(key, output['output'][key].cpu().numpy()) for key in output['output'].keys()])
            val_dict['loss'] = loss.item()
        else:
            raise ValueError("Do not call forward function in training mode. USE optimize_parameters() INSTEAD.")

        return out_dict, val_dict

    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = self._opt.digitize_num
                v = F.softmax(output['VA'][:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis=1)

        return estimates

    def optimize_parameters(self):
        train_dict = dict()
        loss = {}
        if self._is_train:
            for t in self._opt.tasks:
                if t not in loss and t != 'EXPR_VA':
                    loss[t] = []
                input_image = Variable(self._input_image[t])
                label = Variable(self._label[t])
                output = self.resnet50(input_image)

                if t == 'EXPR_VA':
                    criterion_expr = self._criterions_per_task['EXPR'].get_task_loss()
                    loss_expr = criterion_expr(output['output']['EXPR'], label[:, :1].view(-1).type(torch.long))
                    criterion_va = self._criterions_per_task['VA'].get_task_loss()
                    loss_v, loss_a = criterion_va(output['output']['VA'], label[:, 1:])
                    l_v, l_a = self.lambdas_per_task['VA']
                    l_expr = self.lambdas_per_task['EXPR']
                    loss['EXPR'] += [l_expr * loss_expr]
                    loss['VA'] += [l_v * loss_v + l_a * loss_a]

                    train_dict['loss_' + t] = (loss_expr.item() + loss_v.item() + loss_a.item()) / 3.
                elif t != 'VA':
                    criterion_task = self._criterions_per_task[t].get_task_loss()
                    loss_task = criterion_task(output['output'][t], label)
                    train_dict['loss_' + t] = loss_task.item()
                    loss[t] += [self.lambdas_per_task[t] * loss_task]
                else:
                    criterion_task = self._criterions_per_task[t].get_task_loss()
                    loss_task = criterion_task(output['output'][t], label)
                    loss_v, loss_a = loss_task
                    train_dict['loss_valence'] = loss_v.item()
                    train_dict['loss_arousal'] = loss_a.item()
                    l_v, l_a = self.lambdas_per_task[t]
                    loss[t] += [l_v * loss_v + l_a * loss_a]
            self._optimizer_F.zero_grad()
            loss = sum([mean(loss[x]) for x in loss.keys()])
            loss.backward()
            train_dict['loss'] = loss.item()
            self._optimizer_F.step()
            self.loss_dict = train_dict
        else:
            raise ValueError("Do not call optimize_parameters function in test mode. USE forward() INSTEAD.")

    def optimize_parameters_kd_v2(self, teacher_model):
        train_dict = dict()
        loss = {}

        for t in self._opt.tasks:
            if t not in loss and t != 'EXPR_VA':
                loss[t] = []

        if self._is_train:
            for t in self._opt.tasks:
                input_image = Variable(self._input_image[t])
                label = Variable(self._label[t])
                sub = Variable(self._sub[t])
                output = self.resnet50(input_image)
                l_t, l_s = self._opt.lambda_teacher, 1 - self._opt.lambda_teacher

                with torch.no_grad():
                    teacher_preds = teacher_model.resnet50(input_image)

                if t == 'EXPR_VA':
                    criterion_expr = self._criterions_per_task['EXPR'].get_task_loss()
                    loss_expr = criterion_expr(output['output']['EXPR'], label[:, :1].view(-1).type(torch.long))
                    distillation_expr = self._criterions_per_task['EXPR'].get_distillation_loss()
                    loss_kd_expr = distillation_expr(output['output']['EXPR'], teacher_preds['output']['EXPR'])
                    loss['EXPR'] += [l_s * loss_expr + l_t * loss_kd_expr]

                    criterion_va = self._criterions_per_task['VA'].get_task_loss()
                    loss_v, loss_a = criterion_va(output['output']['VA'], label[:, 1:])
                    distillation_va = self._criterions_per_task['VA'].get_distillation_loss()
                    loss_kd_v, loss_kd_a = distillation_va(output['output']['VA'], teacher_preds['output']['VA'])
                    loss['VA'] += [l_s * loss_v + l_t * loss_kd_v + l_s * loss_a + l_t * loss_kd_a]
                    train_dict['loss_' + t] = loss['EXPR'][-1].item() + loss['VA'][-1].item()

                if t == 'EXPR':
                    criterion_expr = self._criterions_per_task['EXPR'].get_task_loss()
                    loss_expr = criterion_expr(output['output']['EXPR'], label)
                    distillation_expr = self._criterions_per_task['EXPR'].get_distillation_loss()
                    loss_kd_expr = distillation_expr(output['output']['EXPR'], teacher_preds['output']['EXPR'])
                    loss['EXPR'] += [l_s * loss_expr + l_t * loss_kd_expr]

                    # The sub task of EXPR will be VA
                    gt_flag = sub[:, 0] != -2.
                    va_gt = sub[gt_flag]
                    distillation_va = self._criterions_per_task['VA'].get_distillation_loss()
                    all_loss_kd_v, all_loss_kd_a = [], []

                    if len(va_gt) != len(sub):
                        # Compute distillation loss for the samples have no VA annotations
                        loss_kd_v, loss_kd_a = distillation_va(output['output']['VA'][~ gt_flag],
                                                               teacher_preds['output']['VA'][~ gt_flag])
                        all_loss_kd_a.append(loss_kd_a)
                        all_loss_kd_v.append(loss_kd_v)
                    if len(va_gt) > 0:
                        # Compute GT loss + distillation loss for the samples have VA annotations
                        criterion_va = self._criterions_per_task['VA'].get_task_loss()
                        loss_v, loss_a = criterion_va(output['output']['VA'][gt_flag], va_gt)
                        loss_kd_gt_v, loss_kd_gt_a = distillation_va(output['output']['VA'][gt_flag],
                                                                     teacher_preds['output']['VA'][gt_flag])
                        all_loss_kd_v.append(l_s * loss_v + l_t * loss_kd_gt_v)
                        all_loss_kd_a.append(l_s * loss_a + l_t * loss_kd_gt_a)

                    loss['VA'] += [mean(all_loss_kd_v) + mean(all_loss_kd_a)]
                    train_dict['loss_' + t] = loss['EXPR'][-1].item() + loss['VA'][-1].item()

                if t == 'VA':
                    criterion_va = self._criterions_per_task['VA'].get_task_loss()
                    loss_v, loss_a = criterion_va(output['output']['VA'], label)
                    distillation_va = self._criterions_per_task['VA'].get_distillation_loss()
                    loss_kd_v, loss_kd_a = distillation_va(output['output']['VA'], teacher_preds['output']['VA'])
                    loss['VA'] += [l_s * loss_v + l_t * loss_kd_v + l_s * loss_a + l_t * loss_kd_a]

                    # The sub task of VA will be EXPR
                    gt_flag = sub != -2.
                    expr_gt = sub[gt_flag]
                    distillation_expr = self._criterions_per_task['EXPR'].get_distillation_loss()
                    all_loss_kd_expr = []

                    if len(expr_gt) != len(sub):
                        loss_kd_expr = distillation_expr(output['output']['EXPR'][~ gt_flag],
                                                         teacher_preds['output']['EXPR'][~ gt_flag])
                        all_loss_kd_expr.append(loss_kd_expr)
                    if len(expr_gt) > 0:
                        criterion_expr = self._criterions_per_task['EXPR'].get_task_loss()
                        loss_expr = criterion_expr(output['output']['EXPR'][gt_flag], expr_gt)
                        loss_kd_gt_expr = distillation_expr(output['output']['EXPR'][gt_flag],
                                                            teacher_preds['output']['EXPR'][gt_flag])
                        all_loss_kd_expr.append(l_s * loss_expr + l_t * loss_kd_gt_expr)

                    loss['EXPR'] += [mean(all_loss_kd_expr)]
                    train_dict['loss_' + t] = loss['EXPR'][-1].item() + loss['VA'][-1].item()

            self._optimizer_F.zero_grad()
            loss = sum([mean(loss[x]) for x in loss.keys()])
            loss.backward()
            train_dict['loss'] = loss.item()
            self._optimizer_F.step()
            self.loss_dict = train_dict
        else:
            raise ValueError("Do not call optimize_parameters function in test mode. USE forward() INSTEAD.")

    def optimize_parameters_kd(self, teacher_model):
        train_dict = dict()
        loss = 0.
        loss_per_task = {'EXPR': 0, 'valence': 0, 'arousal': 0}
        if self._is_train:
            for t in self._opt.tasks:
                input_image = Variable(self._input_image[t])
                label = Variable(self._label[t])
                output = self.resnet50(input_image)
                with torch.no_grad():
                    teacher_preds = teacher_model.resnet50(input_image)
                for task in self._opt.tasks:
                    distillation_task = self._criterions_per_task[task].get_distillation_loss()
                    loss_task = distillation_task(output['output'][task], teacher_preds['output'][task])
                    if task == t:
                        if task != 'VA':
                            criterion_task = self._criterions_per_task[t].get_task_loss()
                            loss_task = self._opt.lambda_teacher * loss_task + (
                                        1 - self._opt.lambda_teacher) * criterion_task(output['output'][t], label)
                        else:
                            criterion_task = self._criterions_per_task[t].get_task_loss()
                            loss_v, loss_a = criterion_task(output['output'][t], label)
                            loss_task = [
                                self._opt.lambda_teacher * loss_task[0] + (1 - self._opt.lambda_teacher) * loss_v,
                                self._opt.lambda_teacher * loss_task[1] + (1 - self._opt.lambda_teacher) * loss_a, ]
                    if task != 'VA':
                        loss_per_task[task] += loss_task.item()
                        loss += loss_task
                    else:
                        loss_v, loss_a = loss_task
                        loss_per_task['valence'] += loss_v.item()
                        loss_per_task['arousal'] += loss_a.item()
                        loss += loss_v + loss_a
            loss = loss / len(self._opt.tasks)
            for key in loss_per_task.keys():
                loss_task = loss_per_task[key]
                train_dict['loss_' + key] = loss_task / len(self._opt.tasks)
            train_dict['loss'] = loss.item()
            self._optimizer_F.zero_grad()
            loss.backward()
            self._optimizer_F.step()
            self.loss_dict = train_dict
        else:
            raise ValueError("Do not call optimize_parameters function in test mode. USE forward() INSTEAD.")

    def get_current_errors(self):
        return self.loss_dict

    def get_metrics_per_task(self):
        return {"EXPR": EXPR_metric, "VA": VA_metric}

    def get_current_LR(self):
        LR = []
        for param_group in self._optimizer_F.param_groups:
            LR.append(param_group['lr'])
        print('current learning rate: {}'.format(np.unique(LR)))
