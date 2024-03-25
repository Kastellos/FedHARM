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
from common_dataset_training import get_top_grad_indices

from common_dataset_training import get_indices_per_layer


class CentralNode():
    def __init__(self, n_clients):

        self.n_clients = n_clients

        self.class_attentive_vectors = {}


    def rearrange_main_dict(self, main_d):
        """
        Rearranging the dictionary given to a more suitable one for to access the features
        Prev Format->
            Client# = {Class# = {Layer# = features
                                {Layer# = features
                                ...
                      {Class# = {Layer# = features
                                {Layer# = features
                                ...
                      ...

        New Format ->
            Class# = {Client# = {Layer# = features
            ...
        @param main_d: Dictionary with previous format
        @return: New dictionary with the new format
        """
        rearranged_d = {}
        for client, classes in main_d.items():
            for class_name, values in classes.items():
                if class_name not in rearranged_d:
                    rearranged_d[class_name] = {}
                rearranged_d[class_name][client] = values
        return rearranged_d

    def aggregate_one_class_featurec(self, x: dict):
        """
        Aggregation of the features of one class
        @param x: Dictionary that contains the features from all the clients of all the blocks
        @return: Aggregated dictionary with one feature per block
        """

        l1 = []
        l2 = []
        l3 = []
        l4 = []
        for client in x.keys():
            l1.append(x[client][1])
            l2.append(x[client][2])
            l3.append(x[client][3])
            l4.append(x[client][4])

        l1 = torch.stack(l1)
        l2 = torch.stack(l2)
        l3 = torch.stack(l3)
        l4 = torch.stack(l4)

        temp_dict = {
            1: torch.mean(l1, 0),
            2: torch.mean(l2, 0),
            3: torch.mean(l3, 0),
            4: torch.mean(l4, 0)
        }
        return temp_dict

    def aggregate_class_attentive_vectors(self, class_att_vectors: dict):
        """
        Aggregation of all the class attentive vectors to get one vector per class per block
        @param class_att_vectors: Class attentive vector per client per class per block
        @return: Nothing (Aggregated feature vectors per class per block)
        """

        self.class_attentive_vectors = {}
        cav = self.rearrange_main_dict(class_att_vectors)

        for class_name, clients in cav.items():
            class_dict = self.aggregate_one_class_featurec(clients)
            self.class_attentive_vectors[class_name] = class_dict

    def average_models(self, models, aggregated_model, coef_of_each_model=0.1,
                       num_samples_per_client=[10000, 10000, 10000, 10000, 10000]):
        # Initialize a dictionary to store the sum of model parameters
        print('Averaging the models...')
        # sum_parameters = {}
        #
        # # Iterate over local models
        # for local_model in models:
        #     # Iterate over the parameters of the local model
        #     for name, parameter in local_model.model.named_parameters():
        #         if parameter.requires_grad:
        #             # If the parameter has not been added to sum_parameters yet, create an entry
        #             if name not in sum_parameters:
        #                 sum_parameters[name] = parameter.data.clone().detach()
        #             # If the parameter already exists in sum_parameters, add the current parameter
        #             else:
        #                 sum_parameters[name] += coef_of_each_model * parameter.data
        #
        # # Calculate the average of the parameters
        # num_local_models = len(models)
        # for name, sum_parameter in sum_parameters.items():
        #     sum_parameters[name] /= num_local_models
        #
        # # Update the central model with the averaged parameters
        # for central_parameter, sum_parameter in zip(aggregated_model.model.parameters(), sum_parameters.values()):
        #     central_parameter.data.copy_(sum_parameter)
        #
        # # Clear the gradients of the central model
        # aggregated_model.zero_grad()

        """
            Perform Federated Averaging on the local models.

            Args:
            - local_models (list of nn.Module): The local models from each client.
            - num_samples_per_client (list of int): The number of samples for each client.

            Returns:
            - nn.Module: The global model after aggregation.
            """

        global_model = models[0].state_dict()
        total_samples = sum(num_samples_per_client)

        # Initialize global model with zeros
        for key in global_model.keys():
            global_model[key] = torch.zeros_like(global_model[key])

        # Compute the weighted average
        for model, num_samples in zip(models, num_samples_per_client):
            weight = num_samples / total_samples
            local_state = model.state_dict()
            for key in global_model.keys():
                global_model[key] += local_state[key] * weight

        # Create a new model instance for the global model and load the averaged state_dict
        global_model_instance = type(models[0])()  # Assuming all local models are of the same type
        global_model_instance.load_state_dict(global_model)

        return global_model_instance


        # print('Averaging Done!')
        # return aggregated_model

class CentralNodeUpgraded():
    def __init__(self, num_clients):

        self.num_clients = num_clients
        self.aggregate_feature_maps = {}

    def first_aggregation(self, local_models_feature_maps: list):

        for block in local_models_feature_maps[0].keys():
            block_features = {}
            dummy_tensor = torch.zeros(local_models_feature_maps[0][block][0].shape[0],
                                       device=local_models_feature_maps[0][block][0].device)
            for cls in local_models_feature_maps[0][block].keys():
                for i in range(len(local_models_feature_maps)):
                    dummy_tensor = dummy_tensor + local_models_feature_maps[i][block][cls]

                block_features[cls] = dummy_tensor / len(local_models_feature_maps)
            self.aggregate_feature_maps[block] = block_features

    def update_global_feature_maps(self, local_models_feature_maps: list):

        for block in local_models_feature_maps[0].keys():
            block_features = {}
            dummy_tensor = torch.zeros(local_models_feature_maps[0][block][0].shape[0],
                                       device=local_models_feature_maps[0][block][0].device)
            for cls in local_models_feature_maps[0][block].keys():
                for i in range(len(local_models_feature_maps)):
                    dummy_tensor = dummy_tensor + local_models_feature_maps[i][block][cls]

                dummy_tensor = dummy_tensor + self.aggregate_feature_maps[block][cls]
                block_features[cls] = dummy_tensor / (len(local_models_feature_maps) + 1)
            self.aggregate_feature_maps[block] = block_features



class ResNetLearningBlockDiscriptors(nn.Module):
    def __init__(self, model_name, n_classes, indices_1, indices_2, indices_3, indices_4,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(ResNetLearningBlockDiscriptors, self).__init__()
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.indices_3 = indices_3
        self.indices_4 = indices_4
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.fc.in_features
        self.model.fc = nn.Linear(self.feature_space, self.n_classes)

        if ('18' in model_name) or ('34' in model_name):
            # self.out_channels_per_model = [64, 128, 256, 512]


            self.local_conv_layer1 = nn.Conv2d(self.model.layer1[-1].conv2.out_channels, 64, 3, 2)
            self.global_conv_layer1 = nn.Conv2d(self.model.layer1[-1].conv2.out_channels, 64, 7, 2)

            self.local_conv_layer2 = nn.Conv2d(self.model.layer2[-1].conv2.out_channels, 128, 3, 2)
            self.global_conv_layer2 = nn.Conv2d(self.model.layer2[-1].conv2.out_channels, 128, 7, 2)

            self.local_conv_layer3 = nn.Conv2d(self.model.layer3[-1].conv2.out_channels, 256, 3, 2)
            self.global_conv_layer3 = nn.Conv2d(self.model.layer3[-1].conv2.out_channels, 256, 7, 2)

            self.local_conv_layer4 = nn.Conv2d(self.model.layer4[-1].conv2.out_channels, 512, 3, 2)
            self.global_conv_layer4 = nn.Conv2d(self.model.layer4[-1].conv2.out_channels, 512, 7, 2)

            self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        else:

            self.local_conv_layer1 = nn.Conv2d(self.model.layer1[-1].conv3.out_channels, 64, 3, 2)
            self.global_conv_layer1 = nn.Conv2d(self.model.layer1[-1].conv3.out_channels, 64, 7, 2)

            self.local_conv_layer2 = nn.Conv2d(self.model.layer2[-1].conv3.out_channels, 128, 3, 2)
            self.global_conv_layer2 = nn.Conv2d(self.model.layer2[-1].conv3.out_channels, 128, 7, 2)

            self.local_conv_layer3 = nn.Conv2d(self.model.layer3[-1].conv3.out_channels, 256, 3, 2)
            self.global_conv_layer3 = nn.Conv2d(self.model.layer3[-1].conv3.out_channels, 256, 7, 2)

            self.local_conv_layer4 = nn.Conv2d(self.model.layer4[-1].conv3.out_channels, 512, 3, 2)
            self.global_conv_layer4 = nn.Conv2d(self.model.layer4[-1].conv3.out_channels, 512, 7, 2)

            self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))


    # def activations_hook(self, grad):
    #     self.gradients = grad
    #
    # def get_activations_gradient(self):
    #     return self.gradients

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

        """ Get intermidiate features """

        # Layer 1
        local_out_conv_layer1 = self.local_conv_layer1(layer1)
        global_out_conv_layer1 = self.global_conv_layer1(layer1)
        spatial_dims_layer1 = global_out_conv_layer1.shape[2:]
        local_out_conv_layer1 = F.interpolate(local_out_conv_layer1, size=spatial_dims_layer1, mode='bilinear',
                                              align_corners=False)
        layer1_features = self.average_pooling(torch.cat([local_out_conv_layer1, global_out_conv_layer1], dim=1))

        # Layer 2
        local_out_conv_layer2 = self.local_conv_layer2(layer2)
        global_out_conv_layer2 = self.global_conv_layer2(layer2)
        spatial_dims_layer2 = global_out_conv_layer2.shape[2:]
        local_out_conv_layer2 = F.interpolate(local_out_conv_layer2, size=spatial_dims_layer2, mode='bilinear',
                                              align_corners=False)
        layer2_features = self.average_pooling(torch.cat([local_out_conv_layer2, global_out_conv_layer2], dim=1))

        # Layer 3
        local_out_conv_layer3 = self.local_conv_layer3(layer3)
        global_out_conv_layer3 = self.global_conv_layer3(layer3)
        spatial_dims_layer3 = global_out_conv_layer3.shape[2:]
        local_out_conv_layer3 = F.interpolate(local_out_conv_layer3, size=spatial_dims_layer3, mode='bilinear',
                                              align_corners=False)
        layer3_features = self.average_pooling(torch.cat([local_out_conv_layer3, global_out_conv_layer3], dim=1))

        # Layer 4
        local_out_conv_layer4 = self.local_conv_layer4(layer4)
        global_out_conv_layer4 = self.global_conv_layer4(layer4)
        spatial_dims_layer4 = global_out_conv_layer4.shape[2:]
        local_out_conv_layer4 = F.interpolate(local_out_conv_layer4, size=spatial_dims_layer4, mode='bilinear',
                                              align_corners=False)
        layer4_features = self.average_pooling(torch.cat([local_out_conv_layer4, global_out_conv_layer4], dim=1))

        representation_features = torch.cat([layer1_features.view(layer1_features.shape[0], -1),
                                             layer2_features.view(layer2_features.shape[0], -1),
                                             layer3_features.view(layer3_features.shape[0], -1),
                                             layer4_features.view(layer4_features.shape[0], -1)], dim=1)

        return preds, representation_features



class ResNetRepresentationWorking(nn.Module):
    def __init__(self, model_name, n_classes, indices_1, indices_2, indices_3, indices_4,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(ResNetRepresentationWorking, self).__init__()
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.indices_3 = indices_3
        self.indices_4 = indices_4
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.fc.in_features
        self.model.fc = nn.Linear(self.feature_space, self.n_classes)
        self.flat = nn.Flatten(1, -1)
        self.pooling_layer = nn.AdaptiveMaxPool2d(2)

    # def activations_hook(self, grad):
    #     self.gradients = grad
    #
    # def get_activations_gradient(self):
    #     return self.gradients

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

        if use_only_flatten:
            return preds, logits

        else:

            # Pooling from each layer
            layer1 = self.pooling_layer(layer1)
            layer1 = self.flat(layer1)

            layer2 = self.pooling_layer(layer2)
            layer2 = self.flat(layer2)

            layer3 = self.pooling_layer(layer3)
            layer3 = self.flat(layer3)

            layer4 = self.pooling_layer(layer4)
            layer4 = self.flat(layer4)

            # Random Selection
            selected_features1 = layer1[:, self.indices_1]
            selected_features2 = layer2[:, self.indices_2]
            selected_features3 = layer3[:, self.indices_3]
            selected_features4 = layer4[:, self.indices_4]

            representation_features = torch.cat([selected_features1,
                                                 selected_features2,
                                                 selected_features3,
                                                 selected_features4], dim=1)

            return preds, representation_features

class EfficientNetFamily(nn.Module):
    def __init__(self, model_name, n_classes, indices_1, indices_2, indices_3, indices_4,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(EfficientNetFamily, self).__init__()
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.indices_3 = indices_3
        self.indices_4 = indices_4
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.feature_space, self.n_classes)
        self.flat = nn.Flatten(1, -1)
        self.pooling_layer = nn.AdaptiveMaxPool2d(2)

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
        features_ = self.model.global_pool(features)
        preds = self.model.classifier(features_)

        if use_only_flatten:

            return preds, features_

        else:


            layer4 = self.pooling_layer(layer4)
            layer4 = self.flat(layer4)

            layer5 = self.pooling_layer(layer5)
            layer5 = self.flat(layer5)

            layer6 = self.pooling_layer(layer6)
            layer6 = self.flat(layer6)

            layer7 = self.pooling_layer(layer7)
            layer7 = self.flat(layer7)

            selected_features1 = layer4[:, self.indices_1]
            selected_features2 = layer5[:, self.indices_2]
            selected_features3 = layer6[:, self.indices_3]
            selected_features4 = layer7[:, self.indices_4]

            representation_features = torch.cat([selected_features1,
                                                 selected_features2,
                                                 selected_features3,
                                                 selected_features4], dim=1)

            return preds, representation_features

class MobileNetV3Family(nn.Module):
    def __init__(self, model_name, n_classes, indices_1, indices_2, indices_3, indices_4,
                 pretrained=False, norm_features=False, feature_excitation=False,
                 random_feature_crop=False):
        super(MobileNetV3Family, self).__init__()
        self.n_classes = n_classes
        self.norm_features = norm_features
        self.feature_excitation = feature_excitation
        self.random_feature_crop = random_feature_crop
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.indices_3 = indices_3
        self.indices_4 = indices_4
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.feature_space = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.feature_space, self.n_classes)
        self.flat = nn.Flatten(1, -1)
        self.pooling_layer = nn.AdaptiveMaxPool2d(2)

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
        features_ = self.model.flatten(features)
        preds = self.model.classifier(features_)

        if use_only_flatten:

            return preds, features_
        else:

            layer4 = self.pooling_layer(layer4)
            layer4 = self.flat(layer4)

            layer5 = self.pooling_layer(layer5)
            layer5 = self.flat(layer5)

            layer6 = self.pooling_layer(layer6)
            layer6 = self.flat(layer6)

            layer7 = self.pooling_layer(layer7)
            layer7 = self.flat(layer7)

            selected_features1 = layer4[:, self.indices_1]
            selected_features2 = layer5[:, self.indices_2]
            selected_features3 = layer6[:, self.indices_3]
            selected_features4 = layer7[:, self.indices_4]

            representation_features = torch.cat([selected_features1,
                                                 selected_features2,
                                                 selected_features3,
                                                 selected_features4], dim=1)

            return preds, representation_features



# ChatGPT version
def aggregate_epoch_features(self, prev_features, current_features):
    if not prev_features:
        for block, cls_features in current_features.items():
            prev_features[block] = {}
            for cls, features in cls_features.items():
                prev_features[block][cls] = features.clone()
    else:
        for block, cls_features in current_features.items():
            for cls, features in cls_features.items():
                prev_features[block][cls].add_(features).div_(2)
    torch.cuda.empty_cache()
    return prev_features

class CentralNodeCommonDataset():
    def __init__(self, common_dataset, num_clients):
        self.num_clients = num_clients
        self.common_dataset = common_dataset

    def check_if_gpu(self, feature_maps: list):
        for i in range(len(feature_maps)):
            if feature_maps[i].is_cuda:
                feature_maps[i] = feature_maps[i].to('cpu')

        torch.cuda.empty_cache()
        return feature_maps

    def aggregation(self, feature_maps: list):
        stacked_feature_maps = torch.stack(feature_maps)
        agg_feature_maps = torch.mean(stacked_feature_maps, dim=0)
        return agg_feature_maps

class ResNetFed(nn.Module):
    """
    Note: This is what we use for benchmarking
    """
    def __init__(self, model_name, n_classes, pretrained=True):
        super(ResNetFed, self).__init__()

        self.model_name = model_name
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
        if 'resnet' in self.model_name:
            self.embedding_space = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.classification_head = nn.Linear(self.embedding_space, self.n_classes, dtype=torch.float32)
        elif 'vgg' in self.model_name:
            self.embedding_space = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
            self.classification_head = nn.Linear(self.embedding_space, self.n_classes, dtype=torch.float32)

    def forward(self, x):
        features = self.model(x)
        logits = self.classification_head(features)
        return logits, features

class CentralFed():
    def __init__(self, model_example: dict, samples_per_client: list):
        """
        Central server for the federated system

        @param model_example: Example of the State of a model to get the types of the layers and the names of layers
        @param samples_per_client: A list with the number of total images per client
        """
        self.model_example = model_example
        self.samples_per_client = samples_per_client
        self.all_samples = sum(self.samples_per_client)
        # self.weights = [torch.as_tensor(i / self.all_samples) for i in self.samples_per_client]
        self.weights = [torch.as_tensor(5) for i in self.samples_per_client]
        print(f'Length of weights for aggregation: {len(self.weights)}')
        self.agg_model = {}
        for key, value in self.model_example.items():
            self.agg_model[key] = torch.zeros_like(value, dtype=value.dtype)

    def reset_agg(self):
        """
        Reset the Aggregated model with it is distributed for to the local clients
        @return: Nothing
        """
        self.agg_model = {}
        for key, value in self.model_example.items():
            self.agg_model[key] = torch.zeros_like(value, dtype=value.dtype)

    def get_local_parameters(self, models: list):
        """
        Get the parameters of all the local clients
        @param models: List with the models of each client
        @return: A list with the parameters of each client's network
        """
        models_param_list = []
        for model in models:
            model_params = {}
            for key, param in model.named_parameters():
                model_params[key] = param

            models_param_list.append(model_params)

        return models_param_list

    def _fed_avg(self, models_param_list: list):
        """
        Perform the federated average and get the aggregated parameters from the local models

        @param models_param_list: List with the parameters from each client's network
        @return: The aggregated network
        """

        averaged_params = models_param_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(models_param_list)):
                local_model_params = models_param_list[i]
                local_sample_number = self.samples_per_client[i]
                if i == 0:
                    averaged_params[k] = (
                            local_model_params[k] * local_sample_number / self.all_samples
                    )
                else:
                    averaged_params[k] += (
                            local_model_params[k] * local_sample_number / self.all_samples
                    )
        return averaged_params

    def _update_local_models(self, aggregated_model: dict, local_models: list):
        """
        Update the local clients with the aggregated model

        @param aggregated_model: The parameters of the aggregated model from the local clients
        @param local_models: The list of the local models that the weights will be updated to
        @return: Nothing special
        """
        with torch.no_grad():
            for model in local_models:
                for key, value in model.named_parameters():
                    value.copy_(aggregated_model[key])

        return local_models

    def fed_avg(self, models: list, ):

        assert len(models) == len(self.weights), 'Number of client models and split of the dataset is not the same!'

        for key, value in models[0].items():

            type_of_weights = value.dtype
            break

        for i in range(len(models)):
            model = models[i]
            weight = self.weights[i].type(dtype=type_of_weights)
            for key, value in model.items():
                if 'bn' in key or value.dtype == torch.int64:
                    value_np = value.numpy()
                    value_np = value_np.astype(np.float32)
                    weight_np = weight.numpy()
                    value_to_add = value_np * weight_np
                    value_to_add = np.round(value_to_add)
                    value_to_add = value_to_add.astype(np.int64)
                    value_to_add = np.array(value_to_add)
                    value_to_add = torch.from_numpy(value_to_add)
                    self.agg_model[key] += value_to_add
                else:
                    self.agg_model[key] += value * weight

        return self.agg_model


def compute_layerwise_difference(central_model, local_model):
    """Compute layer-wise differences between central and local models."""
    differences = {}

    # Iterate over named parameters for the central model
    for (name_central, param_central), (_, param_local) in zip(central_model.named_parameters(),
                                                               local_model.named_parameters()):
        # Calculate the absolute difference and sum it up
        diff = (param_central - param_local).abs().sum().item()
        differences[name_central] = diff

    return differences


def plot_layerwise_difference(differences, unique):
    """Plot the layer-wise differences."""
    names = list(differences.keys())
    values = list(differences.values())

    plt.figure(figsize=(15, 7))
    plt.bar(names, values, alpha=0.7)
    plt.xlabel('Layer Name')
    plt.ylabel('Total Absolute Weight Difference')
    plt.title('Layer-wise Weight Difference between Central and Local Model')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(f'/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/models_diff/{uuid.uuid4()}_{unique}.png')


if __name__ == "__main__":

    model = ResNetWithHooksForGradients(model_name='resnet50', n_classes=10, indices_1=128, indices_2=256,
                                           indices_3=512, indices_4=1024)
    model.initialize_reset_gradients()
    model.set_hooks()
    img = torch.randn(128, 3, 224, 224)
    img2 = torch.randn(16, 3, 224, 224)

    labels = torch.randint(0, 9, (1, img.shape[0])).squeeze()
    criterion = nn.CrossEntropyLoss()

    # output = model(img)
    # loss = criterion(output, labels)
    # loss.backward()
    #
    #
    # grad1, grad2, grad3, grad4 = model.gradient_1[0], model.gradient_2[0], model.gradient_3[0], model.gradient_4[0]
    # model.reset_hooks()
    # model.initialize_reset_gradients()
    #
    # grad1_indices, time = get_top_grad_indices(torch.abs(grad1), 128)
    # extracted_values = grad1[grad1_indices[:, 0], grad1_indices[:, 1], grad1_indices[:, 2], grad1_indices[:, 3]]
    # grad1_features = extracted_values.view(grad1.shape[0], -1)
    #
    # grad2_indices, time = get_top_grad_indices(torch.abs(grad2), 256)
    # extracted_values = grad2[grad2_indices[:, 0], grad2_indices[:, 1], grad2_indices[:, 2], grad2_indices[:, 3]]
    # grad2_features = extracted_values.view(grad2.shape[0], -1)
    #
    # grad3_indices, time = get_top_grad_indices(torch.abs(grad3), 512)
    # extracted_values = grad3[grad3_indices[:, 0], grad3_indices[:, 1], grad3_indices[:, 2], grad3_indices[:, 3]]
    # grad3_features = extracted_values.view(grad3.shape[0], -1)
    #
    # grad4_indices, time = get_top_grad_indices(torch.abs(grad4), 1024)
    # extracted_values = grad4[grad4_indices[:, 0], grad4_indices[:, 1], grad4_indices[:, 2], grad4_indices[:, 3]]
    # grad4_features = extracted_values.view(grad4.shape[0], -1)

    x = torch.abs(torch.randn(2, 3, 2, 2))
    testing, time = get_top_grad_indices(x, 2)
    testing_indexing = x[testing[:, 0], testing[:, 1], testing[:, 2], testing[:, 3]]
    test = testing.view(x.shape[0], -1)

    # res_selected_indices1, res_selected_indices2, res_selected_indices3, res_selected_indices4 = get_indices_per_layer(
    #     'resnet')
    # # logits, block1_1, block2_1, block3_1, block4_1 = model(img)
    # # logits2, block1_2, block2_2, block3_2, block4_2 = model(img2)
    # flattened_size1 = 256
    # num_features_to_select1 = 128
    # selected_indices1 = torch.randperm(flattened_size1)[:num_features_to_select1]
    #
    # # 2nd block
    # flattened_size2 = 512
    # num_features_to_select2 = 256
    # selected_indices2 = torch.randperm(flattened_size2)[:num_features_to_select2]
    #
    # # 3rd block
    # flattened_size3 = 1024
    # num_features_to_select3 = 512
    # selected_indices3 = torch.randperm(flattened_size3)[:num_features_to_select3]
    #
    # # 4th block
    # flattened_size4 = 2048
    # num_features_to_select4 = 1024
    # selected_indices4 = torch.randperm(flattened_size4)[:num_features_to_select4]
    # # model = ResNetRepresentationWorking('resnet18', 200, selected_indices1,
    # #                            selected_indices2, selected_indices3, selected_indices4)
    # model = ResNetRepresentationWorking('resnet50', 10, res_selected_indices1,
    #                            res_selected_indices2, res_selected_indices3, res_selected_indices4)
    #
    # pred, kappa = model(img)
