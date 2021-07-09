#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:43:02 2019
utils
@author: ddeng
"""
import os
import random
import sys
from os.path import join as pjoin

import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFilter
from sklearn.metrics import f1_score
from torch.autograd import Variable
from utils.data_utils import RandomHorizontalFlip, RandomCrop

from path import PATH

PRESET_VARS = PATH()
MODEL_DIR = PRESET_VARS.MODEL_DIR


def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(target[:, i], input[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s


def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]


def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs


def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc


def VA_metric(x, y):
    items = [CCC_score(x[:, 0], y[:, 0]), CCC_score(x[:, 1], y[:, 1])]
    return items, sum(items)


def EXPR_metric(x, y): 
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = accuracy(x, y)
    return [f1, acc], 0.67*f1 + 0.33*acc


class CCCLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1]):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        if self.digitize_num !=0:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = Variable(torch.as_tensor(bins, dtype = torch.float32).cuda()).view((1, -1))

    def forward(self, x, y): 
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)
        vx = x - torch.mean(x) 
        vy = y - torch.mean(y) 
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1-ccc


class Custom_CrossEntropyLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1]):
        super(Custom_CrossEntropyLoss, self).__init__() 
        self.digitize_num = digitize_num
        self.range = range
        assert self.digitize_num != 1
        self.edges = np.linspace(*self.range, num=self.digitize_num + 1)

    def forward(self, x, y): 
        # the target y is continuous value (BS, )
        # the input x is  probability output(digitized)
        y = y.view(-1)
        y_numpy = y.data.cpu().numpy()
        y_dig = np.digitize(y_numpy, self.edges) - 1
        y_dig[y_dig == self.digitize_num] = self.digitize_num -1
        y = Variable(torch.cuda.LongTensor(y_dig))
        return F.cross_entropy(x, y)


class EXPR_Losses(object):
    def __init__(self, opt):
        self._opt = opt
        self.class_num = self._opt.EXPR_label_size
        self.criterion = self._opt.EXPR_criterion
        self.temperature = self._opt.temperature

    def get_task_loss(self):
        if self.criterion == 'CE':
            task_loss = nn.CrossEntropyLoss().cuda()
        return task_loss

    def get_distillation_loss(self):
        def distill(y, teacher_pred, T=self.temperature):
            loss = nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(y/T, dim=-1), F.softmax(teacher_pred/T, dim=-1))
            return loss * T * T
        distillation_loss = distill
        return distillation_loss


class VA_Losses(object):
    def __init__(self, opt):
        self._opt = opt
        self.class_num = self._opt.VA_label_size
        self.criterion = self._opt.VA_criterion
        self.digitize_num = self._opt.digitize_num
        self.temperature = self._opt.temperature
        self.v_onehot = torch.cuda.FloatTensor(self._opt.batch_size, self.class_num).zero_()
        self.a_onehot = torch.cuda.FloatTensor(self._opt.batch_size, self.class_num).zero_()
        self.v_index = torch.cuda.LongTensor(self._opt.batch_size).zero_()
        self.a_index = torch.cuda.LongTensor(self._opt.batch_size).zero_()

    def get_task_loss(self):
        ccc_loss = CCCLoss(self.digitize_num)
        if self.digitize_num != 1:
            if self.criterion == 'CCC_CE':
                classification_loss = Custom_CrossEntropyLoss(self.digitize_num)

            def criterion_task(x, y):
                N = self.digitize_num 
                loss_v = (self._opt.lambda_ccc * ccc_loss(x[:, :N], y[:, :1]) + classification_loss(x[:, :N], y[:, :1]))
                loss_a = (self._opt.lambda_ccc * ccc_loss(x[:, N:], y[:, 1:]) + classification_loss(x[:, N:], y[:, 1:]))
                return loss_v, loss_a
        else:
            def criterion_task(x, y):
                N = self.digitize_num 
                return self._opt.lambda_V* ccc_loss(x[:, :N], y[:, :1]) + self._opt.lambda_A * ccc_loss(x[:, N:], y[:, 1:])
        task_loss = criterion_task
        return task_loss

    def get_distillation_loss(self):
        def distill(y, teacher_pred, T=self.temperature):
            N = self.digitize_num 
            loss_v = nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(y[:, :N]/T, dim=-1), F.softmax(teacher_pred[:, :N]/T, dim=-1))
            loss_a = nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(y[:, N:]/T, dim=-1), F.softmax(teacher_pred[:, N:]/T, dim=-1))
            return loss_v * T * T, loss_a * T * T
        distillation_loss = distill
        return distillation_loss


class EXPR_VA_Loss:
    def __init__(self, opt):
        self._opt = opt
        self.expr_loss = EXPR_Losses(self._opt)
        self.va_loss = VA_Losses(self._opt)
        self.temperature = self._opt.temperature

    def get_task_loss(self):
        def criterion_task(model_output, y):
            expr_loss = self.expr_loss.get_task_loss()(model_output['EXPR'], y[:, :1].view(-1).type(torch.long))
            loss_v, loss_a = self.va_loss.get_task_loss()(model_output['VA'], y[:, 1:])
            return expr_loss, loss_v, loss_a
        return criterion_task

    def get_distillation_loss(self):
        def distill(y, teacher_pred, T=self.temperature):
            raise Exception('Not implemented')

        distillation_loss = distill
        return distillation_loss


"""
pytorch benchmark utils: to load pretrained pytorch model , and their transformations
"""
def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3

    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition

    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod


def load_model(model_name, MODEL_DIR):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = pjoin(MODEL_DIR, model_name + '.py')
    weights_path = pjoin(MODEL_DIR, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net


def compose_transforms(meta, center_crop=True, new_imageSize = None,
                      override_meta_imsize=False):
    """Compose preprocessing transforms for model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `new_meta`
           to select the image input size, 
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    if override_meta_imsize:
        im_size = new_imageSize
    assert im_size[0] == im_size[1], 'expected square image size'
    if 'USE_BLUR' in os.environ and os.environ['USE_BLUR'] == 'True':
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    elif 'USE_BCH' in os.environ and os.environ['USE_BCH'] == 'True':
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    elif 'NO_AUG' in os.environ and os.environ['NO_AUG'] == 'True':
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    elif center_crop:
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)


def gaussian_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=1.5))


def augment_transforms(meta, random_crop=True, new_imageSize = None, override_meta_imsize=False):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    if override_meta_imsize:
        im_size = new_imageSize
    assert im_size[0] == im_size[1], 'expected square image size'
    if 'USE_OLD_METHOD' in os.environ and os.environ['USE_OLD_METHOD'] == 'True':
        print('Warning: Use old method for image transforms')
        # For testing the original transformations
        v = random.random()
        transform_list = [transforms.Resize(int(im_size[0] * 1.2)),
                          RandomCrop(im_size[0], v),
                          RandomHorizontalFlip(v)]
        print('Warning: using the old method for image transformations')
    elif 'USE_BLUR' in os.environ and os.environ['USE_BLUR'] == 'True':
        print('Warning: Use Random Blur')
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1])),
                          transforms.RandomApply([gaussian_blur])]
    elif 'NO_AUG' in os.environ and os.environ['NO_AUG'] == 'True':
        print('Warning: Disabled all augmentations')
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]

    elif 'USE_BCH' in os.environ and os.environ['USE_BCH'] == 'True':
        print('Warning: Use Random Blur, Random Crop, Random Flip')
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          transforms.RandomCrop((im_size[0], im_size[0])),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomApply([gaussian_blur])]
    elif random_crop:
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          transforms.RandomCrop((im_size[0], im_size[0])),
                          transforms.RandomHorizontalFlip()]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class = 8):
        super(Head, self).__init__()
        self._name = 'Head'
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc_0 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = self.bn0(x)
        f0 = self.bn1(F.relu(self.fc_0(x)))
        output = self.fc_1(self.dropout(f0))
        return {'output':output, 'feature':f0}


class BackBone(nn.Module):
    def __init__(self, opt):
        super(BackBone, self).__init__()
        self._name = 'BackBone'
        self._opt = opt
        self.model = self._init_create_networks()

    def _init_create_networks(self):
        # the feature extractor
        # different models have different input sizes, different mean and std
        if self._opt.pretrained_dataset == 'ferplus' or self._opt.pretrained_dataset == 'sfew':
            if self._opt.pretrained_dataset == 'ferplus':
                model_name = 'resnet50_ferplus_dag'
                model_dir = os.path.join(MODEL_DIR, 'fer+')
            else:
                model_name = 'resnet50_face_sfew_dag'
                model_dir = os.path.join(MODEL_DIR, 'sfew')
            feature_extractor = load_model(model_name, model_dir)
            meta = feature_extractor.meta
            if not meta['imageSize'][0] == self._opt.image_size:
                new_imageSize = [self._opt.image_size, self._opt.image_size, 3]
                override_meta_imsize = True
            else:
                new_imageSize = None
                override_meta_imsize = False

            setattr(self, 'augment_transforms', augment_transforms(meta, new_imageSize=new_imageSize,
                                                                   override_meta_imsize=override_meta_imsize))
            setattr(self, 'compose_transforms', compose_transforms(meta, new_imageSize=new_imageSize,
                                                                   override_meta_imsize=override_meta_imsize))
        elif self._opt.pretrained_dataset == 'imagenet':
            import torchvision.models as models
            model_name = 'resnet50_imagenet'
            feature_extractor = models.resnext50_32x4d(pretrained=True)
            im_size = self._opt.image_size
            transform_list = transforms.Compose([
                            transforms.Resize(int(im_size*1.2)),
                            transforms.CenterCrop(im_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
            setattr(self, 'compose_transforms', transform_list)
            transform_list = transforms.Compose([
                            transforms.Resize(int(im_size*1.2)),
                            transforms.RandomCrop(im_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
            setattr(self, 'augment_transforms', transform_list)
        else:
            raise ValueError("Pretrained dataset %s not recognized." % self._opt.pretrained_dataset)
        setattr(feature_extractor, 'name', model_name)
        # reform the final layer of feature extrator, turn it into a Identity module
        last_layer_name, last_module = list(feature_extractor.named_modules())[-1]
        try:
            in_channels, out_channels = last_module.in_features, last_module.out_features
            last_linear = True
        except:
            in_channels, out_channels = last_module.in_channels, last_module.out_channels
            last_linear = False
        setattr(feature_extractor, '{}'.format(last_layer_name), Identity()) # the second last layer has 512 dimensions
        setattr(self, 'output_feature_dim', in_channels)

        if not meta['imageSize'][0] == self._opt.image_size:
            pool_layer_name, pool_layer = list(feature_extractor.named_modules())[-2]
            setattr(feature_extractor, '{}'.format(pool_layer_name), nn.AdaptiveAvgPool2d((1, 1)))
        return feature_extractor

    def forward(self, x):
        return self.model(x) 


class Model(nn.Module):
    def __init__(self, backbone, classifier, sofar_task):
        super(Model, self).__init__()
        self._name = 'Model'
        self.backbone = backbone
        self.classifier = classifier
        self.sofar_task = sofar_task

    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        features = {'cross_task': f}
        outputs = {}
        for i, m in enumerate(self.classifier):
            task = self.sofar_task[i] 
            o = m(f)
            outputs[task] = o['output']
            features[task] = o['feature']
        return {'output': outputs, 'feature': features}


class EarlyStop:

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.losses = []
        self.best_loss = 99999999

    def should_stop(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
        if len(self.losses) <= self.n_epochs:
            return False
        best_loss_pos = self.losses.index(self.best_loss)
        if len(self.losses) - best_loss_pos <= self.n_epochs:
            return False
        return True
