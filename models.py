import numpy as np
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from numba import jit
import matplotlib.pyplot as plt
import uuid
from utils import get_top_grad_indices
import random





def get_model_names(num_clients, res_only=False):
    """

    @param num_clients: Total number of clients with the shared dataset
    @param res_only: If use ResNet only
    @return: list of names of models
    """
    pool = ['resnet18', 'resnet34', 'resnet50',
            'efficientnet_b0', 'efficientnet_b1',
            'mobilenetv3_large_100', 'mobilenetv3_small_075', 'mobilenetv3_small_100']
    if res_only:
        pool = ['resnet18', 'resnet34', 'resnet50']
    return random.choices(pool, k=num_clients -1)

def get_models(num_clients, n_classes, res_only=False, pretrained=False,
               norm_features=False, feature_excitation=False,
               random_feature_crop=False):
    """

    @param num_clients: Total number of clients with the shared dataset
    @param n_classes: Total number of classes
    @param res_only: If use ResNet only
    @param pretrained: If use the weights from ImageNet
    @param norm_features: If normalize the features after the extraction
    @param feature_excitation: Feature boosting
    @param random_feature_crop: If use random crop instead of the extraction module
    @return: List of models for the fed scheme
    """
    new_num_clients = num_clients - 1
    model_names = get_model_names(new_num_clients, res_only)
    models = []
    for name in model_names:
        if 'resnet' in name:
            models.append(ResNetWithHooksForGradients(model_name=name,
                                                      n_classes=n_classes,
                                                      pretrained=pretrained,
                                                      norm_features=norm_features,
                                                      feature_excitation=feature_excitation,
                                                      random_feature_crop=random_feature_crop))
        elif 'efficientnet' in name:
            models.append(EfficientNetHooksForGradients(model_name=name,
                                                        n_classes=n_classes,
                                                        pretrained=pretrained,
                                                        norm_features=norm_features,
                                                        feature_excitation=feature_excitation,
                                                        random_feature_crop=random_feature_crop))
        else:
            models.append(MobileNetV3HookForGradients(model_name=name,
                                                      n_classes=n_classes,
                                                      pretrained=pretrained,
                                                      norm_features=norm_features,
                                                      feature_excitation=feature_excitation,
                                                      random_feature_crop=random_feature_crop))



class ResNetWithHooksForGradients(nn.Module):
    def __init__(self, model_name, n_classes,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(ResNetWithHooksForGradients, self).__init__()
        self._model_name = model_name
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop

        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.fc.in_features
        self.model.fc = nn.Linear(self.feature_space, self.n_classes)
        self.flat = nn.Flatten(1, -1)
        self.pooling_layer = nn.AdaptiveMaxPool2d(2)

    def initialize_reset_gradients(self):
        self.gradient_1 = []
        self.gradient_2 = []
        self.gradient_3 = []
        self.gradient_4 = []

    def set_hooks(self):
        self.hook1 = self.model.layer1.register_backward_hook(lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_1))
        self.hook2 = self.model.layer2.register_backward_hook(lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_2))
        self.hook3 = self.model.layer3.register_backward_hook(lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_3))
        self.hook4 = self.model.layer4.register_backward_hook(lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_4))

    def save_gradients(self, grad, where: list):
        where.append(grad.cpu())

    def reset_hooks(self):
        self.hook1.remove()
        self.hook2.remove()
        self.hook3.remove()
        self.hook4.remove()

    def forward(self, x, use_only_flatten=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.act1(features)
        features = self.model.maxpool(features)
        layer1 = self.model.layer1(features)
        layer2 = self.model.layer2(layer1)
        layer3 = self.model.layer3(layer2)
        layer4 = self.model.layer4(layer3)
        logits = self.model.global_pool(layer4)
        preds = self.model.fc(logits)
        return preds, layer1, layer2, layer3, layer4

class EfficientNetHooksForGradients(nn.Module):
    def __init__(self, model_name, n_classes,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(EfficientNetHooksForGradients, self).__init__()
        self._model_name = model_name
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.feature_space, self.n_classes)
        self.flat = nn.Flatten(1, -1)
        self.pooling_layer = nn.AdaptiveMaxPool2d(2)

    def initialize_reset_gradients(self):
        self.gradient_1 = []
        self.gradient_2 = []
        self.gradient_3 = []
        self.gradient_4 = []
        self.gradient_5 = []
        self.gradient_6 = []
        self.gradient_7 = []

    def set_hooks(self):
        self.hook1 = self.model.blocks[0].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_1))
        self.hook2 = self.model.blocks[1].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_2))
        self.hook3 = self.model.blocks[2].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_3))
        self.hook4 = self.model.blocks[3].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_4))
        self.hook5 = self.model.blocks[4].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_5))
        self.hook6 = self.model.blocks[5].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_6))
        self.hook7 = self.model.blocks[6].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_7))

    def save_gradients(self, grad, where: list):
        where.append(grad.cpu())

    def reset_hooks(self):
        self.hook1.remove()
        self.hook2.remove()
        self.hook3.remove()
        self.hook4.remove()
        self.hook5.remove()
        self.hook6.remove()
        self.hook7.remove()

    def forward(self, x, use_only_flatten=False):
        features = self.model.conv_stem(x)
        features = self.model.bn1(features)
        layer1 = self.model.blocks[0](features)
        layer2 = self.model.blocks[1](layer1)
        layer3 = self.model.blocks[2](layer2)
        layer4 = self.model.blocks[3](layer3)
        layer5 = self.model.blocks[4](layer4)
        layer6 = self.model.blocks[5](layer5)
        layer7 = self.model.blocks[6](layer6)
        features = self.model.conv_head(layer7)
        features = self.model.bn2(features)
        flatten = self.model.global_pool(features)
        preds = self.model.classifier(flatten)
        return preds, layer4, layer5, layer6, layer7

class MobileNetV3HookForGradients(nn.Module):
    def __init__(self, model_name, n_classes,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(MobileNetV3HookForGradients, self).__init__()
        self._model_name = model_name
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.feature_space, self.n_classes)
        self.flat = nn.Flatten(1, -1)
        self.pooling_layer = nn.AdaptiveMaxPool2d(2)

    def initialize_reset_gradients(self):
        self.gradient_1 = []
        self.gradient_2 = []
        self.gradient_3 = []
        self.gradient_4 = []
        self.gradient_5 = []
        self.gradient_6 = []
        self.gradient_7 = []

    def set_hooks(self):
        self.hook1 = self.model.blocks[0].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_1))
        self.hook2 = self.model.blocks[1].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_2))
        self.hook3 = self.model.blocks[2].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_3))
        self.hook4 = self.model.blocks[3].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_4))
        self.hook5 = self.model.blocks[4].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_5))
        self.hook6 = self.model.blocks[5].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_6))
        self.hook7 = self.model.blocks[6].register_backward_hook(
            lambda module, grad_input, grad_output: self.save_gradients(grad_output[0], self.gradient_7))

    def save_gradients(self, grad, where: list):
        where.append(grad.cpu())

    def reset_hooks(self):
        self.hook1.remove()
        self.hook2.remove()
        self.hook3.remove()
        self.hook4.remove()
        self.hook5.remove()
        self.hook6.remove()
        self.hook7.remove()

    def forward(self, x, use_only_flatten=False):
        features = self.model.conv_stem(x)
        features = self.model.bn1(features)
        layer1 = self.model.blocks[0](features)
        layer2 = self.model.blocks[1](layer1)
        layer3 = self.model.blocks[2](layer2)
        layer4 = self.model.blocks[3](layer3)
        layer5 = self.model.blocks[4](layer4)
        layer6 = self.model.blocks[5](layer5)
        layer7 = self.model.blocks[6](layer6)
        features = self.model.global_pool(layer7)
        features = self.model.conv_head(features)
        features = self.model.act2(features)
        flatten = self.model.flatten(features)
        preds = self.model.classifier(flatten)
        return preds, layer4, layer5, layer6, layer7






