from tqdm import tqdm
import torch
import os
import copy
import torch.nn as nn
import GPUtil
from losses import *
import time

def use_grad_to_get_features_from_indices(block_indices, layer1, layer2, layer3, layer4,
                                          split_sizes=[128, 256, 512, 1024]):
    # block_indices = block_indices.to(device)
    indices_grad1, indices_grad2, indices_grad3, indices_grad4 = torch.split(block_indices, split_sizes, dim=1)
    # indices_grad1 = torch.reshape(indices_grad1, (-1, 4))
    # indices_grad2 = torch.reshape(indices_grad2, (-1, 4))
    # indices_grad3 = torch.reshape(indices_grad3, (-1, 4))
    # indices_grad4 = torch.reshape(indices_grad4, (-1, 4))

    # features_from_indices_layer1 = layer1[
    #     indices_grad1[:, :, 0], indices_grad1[:, :, 1], indices_grad1[:, :, 2], indices_grad1[:, :, 3]]
    # features_from_indices_layer2 = layer2[
    #     indices_grad2[:, :, 0], indices_grad2[:, :, 1], indices_grad2[:, :, 2], indices_grad2[:, :, 3]]
    # features_from_indices_layer3 = layer3[
    #     indices_grad3[:, :, 0], indices_grad3[:, :, 1], indices_grad3[:, :, 2], indices_grad3[:, :, 3]]
    # features_from_indices_layer4 = layer4[
    #     indices_grad4[:, :, 0], indices_grad4[:, :, 1], indices_grad4[:, :, 2], indices_grad4[:, :, 3]]
    grad_layer1 = []
    grad_layer2 = []
    grad_layer3 = []
    grad_layer4 = []
    for i in range(layer1.shape[0]):
        grad_layer1.append(layer1[i, indices_grad1[i, :, 1], indices_grad1[i, :, 2], indices_grad1[i, :, 3]])
        grad_layer2.append(layer2[i, indices_grad2[i, :, 1], indices_grad2[i, :, 2], indices_grad2[i, :, 3]])
        grad_layer3.append(layer3[i, indices_grad3[i, :, 1], indices_grad3[i, :, 2], indices_grad3[i, :, 3]])
        grad_layer4.append(layer4[i, indices_grad4[i, :, 1], indices_grad4[i, :, 2], indices_grad4[i, :, 3]])

    grad_layer1 = torch.stack(grad_layer1)
    grad_layer2 = torch.stack(grad_layer2)
    grad_layer3 = torch.stack(grad_layer3)
    grad_layer4 = torch.stack(grad_layer4)

    # features_from_indices_layer1 = torch.reshape(features_from_indices_layer1, (layer1.shape[0], -1))
    # features_from_indices_layer2 = torch.reshape(features_from_indices_layer2, (layer2.shape[0], -1))
    # features_from_indices_layer3 = torch.reshape(features_from_indices_layer3, (layer3.shape[0], -1))
    # features_from_indices_layer4 = torch.reshape(features_from_indices_layer4, (layer4.shape[0], -1))

    # return torch.cat((features_from_indices_layer1, features_from_indices_layer2,
    #                   features_from_indices_layer3, features_from_indices_layer4), dim=1)
    return torch.cat((grad_layer1, grad_layer2, grad_layer3, grad_layer4), dim=1)


def get_top_grad_indices(features: torch.Tensor, top_k):
    # Flatten the tensor to 1D
    tensor_flat = features.view(features.shape[0], -1)
    # Get the top 10 values and their indices in the flattened tensor
    values, indices_flat = torch.topk(tensor_flat, top_k, dim=1)
    # Convert the flat indices to 4D indices
    # num_spatial_elements = features.size(2) * features.size(3)
    start_manual = time.time()

    num_spatial_elements = features.size(2) * features.size(3)

    # No need for batch_indices as each row in indices_flat corresponds to a batch
    channel_indices = indices_flat // num_spatial_elements
    height_indices = (indices_flat % num_spatial_elements) // features.size(3)
    width_indices = indices_flat % features.size(3)

    # Create a tensor for batch indices
    batch_size = features.size(0)
    batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, top_k).reshape(-1)

    # Reshape and stack to get the correct 4D indices
    channel_indices = channel_indices.reshape(-1)
    height_indices = height_indices.reshape(-1)
    width_indices = width_indices.reshape(-1)
    indices_4d = torch.stack((batch_indices, channel_indices, height_indices, width_indices), dim=1)
    # ...
    # indices_4d = torch.stack(torch.meshgrid([torch.arange(features.size(0)),
    #                                          torch.arange(features.size(1)),
    #                                          torch.arange(features.size(2)),
    #                                          torch.arange(features.size(3))]), dim=-1).reshape(-1, 4)[indices_flat]
    stop_manual = time.time()

    return indices_4d, stop_manual - start_manual

def get_indices_per_layer(model: str):
    if model == 'resnet':
        # 1st block
        flattened_size1 = 256
        num_features_to_select1 = 128
        selected_indices1 = torch.randperm(flattened_size1)[:num_features_to_select1]

        # 2nd block
        flattened_size2 = 512
        num_features_to_select2 = 256
        selected_indices2 = torch.randperm(flattened_size2)[:num_features_to_select2]

        # 3rd block
        flattened_size3 = 1024
        num_features_to_select3 = 512
        selected_indices3 = torch.randperm(flattened_size3)[:num_features_to_select3]

        # 4th block
        flattened_size4 = 2048
        num_features_to_select4 = 1024
        selected_indices4 = torch.randperm(flattened_size4)[:num_features_to_select4]
    elif model == 'efficientnet':
        # 1st block
        flattened_size1 = 320
        num_features_to_select1 = 128
        selected_indices1 = torch.randperm(flattened_size1)[:num_features_to_select1]

        # 2nd block
        flattened_size2 = 448
        num_features_to_select2 = 256
        selected_indices2 = torch.randperm(flattened_size2)[:num_features_to_select2]

        # 3rd block
        flattened_size3 = 768
        num_features_to_select3 = 512
        selected_indices3 = torch.randperm(flattened_size3)[:num_features_to_select3]

        # 4th block
        flattened_size4 = 1280
        num_features_to_select4 = 1024
        selected_indices4 = torch.randperm(flattened_size4)[:num_features_to_select4]
    elif model == 'mobilenetv3':
        # 1st block
        flattened_size1 = 256
        num_features_to_select1 = 128
        selected_indices1 = torch.randperm(flattened_size1)[:num_features_to_select1]

        # 2nd block
        flattened_size2 = 352
        num_features_to_select2 = 256
        selected_indices2 = torch.randperm(flattened_size2)[:num_features_to_select2]

        # 3rd block
        flattened_size3 = 480
        num_features_to_select3 = 256
        selected_indices3 = torch.randperm(flattened_size3)[:num_features_to_select3]

        # 4th block
        flattened_size4 = 1280
        num_features_to_select4 = 1408
        selected_indices4 = torch.randperm(flattened_size4)[:num_features_to_select4]

    return selected_indices1, selected_indices2, selected_indices3, selected_indices4


def save_model(model, client, name, main_path='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/weights'):
    """

    """
    folder_path = os.path.join(main_path, name)
    if os.path.exists(folder_path):
        pass
    else:
        os.mkdir(folder_path)
    weights_path = f'{main_path}/{name}/client_{client}_{name}_best.pt'
    torch.save(model, weights_path)

def load_models(weight_path: str):
    print('Loading models.')
    weights = sorted(os.listdir(weight_path))
    loaded_models = []
    for i in range(len(weights)):
        current_path = os.path.join(weight_path, weights[i])
        loaded_models.append(torch.load(current_path))
    print('Finished model loading!')
    return loaded_models


def one_epoch_train_private(model, loader, optimizer, criterion, scaler, device, scheduler=None):
    """
    Initial training of each local model to get representative feature maps for each image in the common dataset

    @param model: Model of use
    @param loader: Loader for the private dataset of each client
    @param optimizer: Optimizer of use
    @param criterion: Loss function of use (typically cross-entropy)
    @param scaler: Scaler for mixed precision
    @param device: Transfer the images and labels on the models device (cuda)
    @param scheduler: Scheduler of use
    @return:
        training loss
        training accuracy
    """
    if model.feature_excitation:
        model.feature_excitation = False
    model.to(device)
    model.train()

    criterion.to(device)
    running_loss = 0.0
    acc = 0.0
    for i, batch in tqdm(enumerate(loader)):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # print(f'Images in the batch: {images.shape[0]}')
        with torch.cuda.amp.autocast():
            preds, layer1_features, layer2_features, layer3_features, layer4_features = model(images)
            loss = criterion(preds, labels)

        acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        torch.cuda.empty_cache()
        running_loss += loss.item()

    return running_loss / (i + 1), acc / (i + 1)

def one_epoch_train_private_no_grad(model, loader, optimizer, criterion, scaler, device, scheduler=None):
    """
    Initial training of each local model to get representative feature maps for each image in the common dataset

    @param model: Model of use
    @param loader: Loader for the private dataset of each client
    @param optimizer: Optimizer of use
    @param criterion: Loss function of use (typically cross-entropy)
    @param scaler: Scaler for mixed precision
    @param device: Transfer the images and labels on the models device (cuda)
    @param scheduler: Scheduler of use
    @return:
        training loss
        training accuracy
    """
    if model.feature_excitation:
        model.feature_excitation = False
    model.to(device)
    model.train()

    criterion.to(device)
    running_loss = 0.0
    acc = 0.0
    for i, batch in tqdm(enumerate(loader)):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # print(f'Images in the batch: {images.shape[0]}')
        with torch.cuda.amp.autocast():
            returns = model(images)
            preds = returns[0]
            loss = criterion(preds, labels)

        acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        torch.cuda.empty_cache()
        running_loss += loss.item()

    return running_loss / (i + 1), acc / (i + 1)

def one_epoch_valid_private(model, loader, optimizer, criterion, device):
    model.to(device)
    model.train()

    criterion.to(device)
    running_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds, layer1_features, layer2_features, layer3_features, layer4_features = model(images)
                loss = criterion(preds, labels)

            acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))

            torch.cuda.empty_cache()
            running_loss += loss.item()

    return running_loss / (i + 1), acc / (i + 1)

def one_epoch_valid_private_no_grad(model, loader, optimizer, criterion, device):
    model.to(device)
    model.train()

    criterion.to(device)
    running_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds, _ = model(images)
                loss = criterion(preds, labels)

            acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))

            torch.cuda.empty_cache()
            running_loss += loss.item()

    return running_loss / (i + 1), acc / (i + 1)

def one_epoch_valid_private_no_grad(model, loader, optimizer, criterion, device):
    model.to(device)
    model.train()

    criterion.to(device)
    running_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                returns = model(images)
                preds = returns[0]
                loss = criterion(preds, labels)

            acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))

            torch.cuda.empty_cache()
            running_loss += loss.item()

    return running_loss / (i + 1), acc / (i + 1)

def first_inferance_for_feature_maps(model, loader, optimizer, device):
    """

    @param model: Model of use
    @param loader: Loader of the common dataset
    @param optimizer: Optimizer of use
    @param device: Device we want the calculations to run on
    @return:
        b1_feature_maps: Concated feature maps of the first block
        b2_feature_maps: Concated feature maps of the second block
        b3_feature_maps: Concated feature maps of the third block
        b4_feature_maps: Concated feature maps of the forth block
        images_: Images to have the correct order for the cosine comparison
        labels_: Labels to have the correct order for the cosine comparison
    """
    model.to(device)
    model.eval()
    b1_feature_maps = []
    b2_feature_maps = []
    b3_feature_maps = []
    b4_feature_maps = []
    labs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            images, labels = batch
            labs.append(labels)
            images = images.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds, block1, block2, block3, block4 = model(images)

            b1_feature_maps.append(block1.cpu())
            b2_feature_maps.append(block2.cpu())
            b3_feature_maps.append(block3.cpu())
            b4_feature_maps.append(block4.cpu())


    b1_feature_maps = torch.cat(b1_feature_maps, dim=0)
    b2_feature_maps = torch.cat(b2_feature_maps, dim=0)
    b3_feature_maps = torch.cat(b3_feature_maps, dim=0)
    b4_feature_maps = torch.cat(b4_feature_maps, dim=0)
    labs = torch.cat(labs, dim=0)

    return b1_feature_maps, b2_feature_maps, b3_feature_maps, b4_feature_maps, labs

def first_inferance_for_feature_maps_specific_layer(model, loader, optimizer, device, layer_number: int):
    """

    @param model: Model of use
    @param loader: Loader of the common dataset
    @param optimizer: Optimizer of use
    @param device: Device we want the calculations to run on
    @return:
        b1_feature_maps: Concated feature maps of the first block
        b2_feature_maps: Concated feature maps of the second block
        b3_feature_maps: Concated feature maps of the third block
        b4_feature_maps: Concated feature maps of the forth block
        images_: Images to have the correct order for the cosine comparison
        labels_: Labels to have the correct order for the cosine comparison
    """
    model.to(device)
    model.eval()
    block_feature_maps = []
    labs = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            images, labels = batch
            labs.append(labels)
            images = images.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if layer_number == 0:
                    _, block, _, _, _ = model(images)
                elif layer_number == 1:
                    _, _, block, _, _ = model(images)
                elif layer_number == 2:
                    _, _, _, block, _ = model(images)
                elif layer_number == 3:
                    _, _, _, _, block = model(images)

            block_feature_maps.append(block.cpu())


    block_feature_maps = torch.cat(block_feature_maps, dim=0)
    labs = torch.cat(labs, dim=0)

    return block_feature_maps, labs

def representation_with_extitation(model, loader, criterion, optimizer, device, layer_number: int):
    if model.feature_excitation:
        pass
    else:
        model.feature_excitation = True
    model.to(device)
    criterion.to(device)
    model.eval()
    block_feature_maps = []
    indices = []
    labs = []

    for i, batch in tqdm(enumerate(loader)):
        model.zero_grad()
        images, labels = batch
        batch_size = images.shape[0]
        labs.append(labels)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if layer_number == 0:
                preds, block, _, _, _ = model(images)
            elif layer_number == 1:
                preds, _, block, _, _ = model(images)
            elif layer_number == 2:
                preds, _, _, block, _ = model(images)
            elif layer_number == 3:
                preds, _, _, _, block = model(images)

            loss = criterion(preds, labels)
            loss.backward()

        block = block.flatten(1, -1)
        block_grads = model.get_activations_gradient()
        block_grads = block_grads.flatten(1, -1)
        _, top_grad_indices = torch.topk(block_grads.abs(), 1024, dim=1)
        indices.append(top_grad_indices)
        for idx in range(batch_size):
            block_feature_maps.append(torch.index_select(block[i], 0, top_grad_indices))




    return torch.stack(block_feature_maps, dim=0), torch.stack(labs, dim=0)

def sort_indices_embeddings_by_ids(id_tensors, embedding_tensors, block_indices):
    sorted_embedding_tensors = []
    sorted_indices_tensors = []
    sorted_id_lists = []
    for ids, embeddings, indices in zip(id_tensors, embedding_tensors, block_indices):
        # Flatten the IDs and get the sorted indices
        ids_flattened = ids.view(-1)
        sorted_indices = ids_flattened.argsort()
        sorted_ids = ids_flattened[sorted_indices]

        # Apply the sorted indices to the embeddings
        embeddings_flattened = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        indices_flatten = indices.view(indices.shape[0], indices.shape[1], -1)

        sorted_embeddings = embeddings_flattened[sorted_indices, :, :]
        sorted_feature_indices = indices_flatten[sorted_indices, :, :]

        # Reshape back to original dimensions if necessary
        sorted_embeddings = sorted_embeddings.view_as(embeddings)
        sorted_feature_indices = sorted_feature_indices.view_as(indices)

        sorted_embedding_tensors.append(sorted_embeddings)
        sorted_indices_tensors.append(sorted_feature_indices)
        sorted_id_lists.append(sorted_ids)

    return torch.stack(sorted_embedding_tensors), torch.stack(sorted_id_lists), torch.stack(sorted_indices_tensors)

def sort_embeddings_by_ids(id_tensors, embedding_tensors):
    sorted_embedding_tensors = []
    sorted_id_lists = []
    for ids, embeddings in zip(id_tensors, embedding_tensors):
        # Flatten the IDs and get the sorted indices
        ids_flattened = ids.view(-1)
        sorted_indices = ids_flattened.argsort()
        sorted_ids = ids_flattened[sorted_indices]
        # Apply the sorted indices to the embeddings
        embeddings_flattened = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        sorted_embeddings = embeddings_flattened[sorted_indices, :, :]
        # Reshape back to original dimensions if necessary
        sorted_embeddings = sorted_embeddings.view_as(embeddings)
        sorted_embedding_tensors.append(sorted_embeddings)
        sorted_id_lists.append(sorted_ids)
    return torch.stack(sorted_embedding_tensors), torch.stack(sorted_id_lists)

def random_crop_representations(model: torch.nn.Module, loader, criterion, optimizer, device,
                                random_indices, layer_number: int=3):
    if model.random_feature_crop:
        pass
    else:
        model.random_feature_crop = True
    model.to(device)
    criterion.to(device)
    model.eval()
    block_feature_maps = []
    block_indices_maps = []
    labs = []
    images_ids = []

    for i, batch in tqdm(enumerate(loader)):

        model.initialize_reset_gradients()
        model.set_hooks()

        model.zero_grad()
        images, labels, image_id = batch
        labs.append(labels)
        images_ids.append(image_id)
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
            # if layer_number == 0:
            #     preds, block, _, _, _ = model(images)
            # elif layer_number == 1:
            #     preds, _, block, _, _ = model(images)
            # elif layer_number == 2:
            #     preds, _, _, block, _ = model(images)
            # elif layer_number == 3:
            #     preds, _, _, _, block = model(images)



            # preds, block = model(images, use_only_flatten=False)
        preds, layer1_features, layer2_features, layer3_features, layer4_features = model(images, use_only_flatten=False)
        loss = criterion(preds, labels)
        loss.backward()

        if 'resnet' in model._model_name:

            grad1, grad2, grad3, grad4 = model.gradient_1[0], model.gradient_2[0], model.gradient_3[0], \
                model.gradient_4[0]

        else:
            grad1, grad2, grad3, grad4 = model.gradient_4[0], model.gradient_5[0], model.gradient_6[0], \
                model.gradient_7[0]


        model.reset_hooks()
        model.initialize_reset_gradients()

        grad1_indices, time = get_top_grad_indices(torch.abs(grad1), 128)
        extracted_values = layer1_features[grad1_indices[:, 0], grad1_indices[:, 1], grad1_indices[:, 2], grad1_indices[:, 3]]
        grad1_features = extracted_values.view(grad1.shape[0], -1).detach().cpu()
        grad1_indices = grad1_indices.view(grad1.shape[0], -1, 4)

        grad2_indices, time = get_top_grad_indices(torch.abs(grad2), 256)
        extracted_values = layer2_features[grad2_indices[:, 0], grad2_indices[:, 1], grad2_indices[:, 2], grad2_indices[:, 3]]
        grad2_features = extracted_values.view(grad2.shape[0], -1).detach().cpu()
        grad2_indices = grad2_indices.view(grad2.shape[0], -1, 4)

        grad3_indices, time = get_top_grad_indices(torch.abs(grad3), 512)
        extracted_values = layer3_features[grad3_indices[:, 0], grad3_indices[:, 1], grad3_indices[:, 2], grad3_indices[:, 3]]
        grad3_features = extracted_values.view(grad3.shape[0], -1).detach().cpu()
        grad3_indices = grad3_indices.view(grad3.shape[0], -1, 4)

        grad4_indices, time = get_top_grad_indices(torch.abs(grad4), 1024)
        extracted_values = layer4_features[grad4_indices[:, 0], grad4_indices[:, 1], grad4_indices[:, 2], grad4_indices[:, 3]]
        grad4_features = extracted_values.view(grad4.shape[0], -1).detach().cpu()
        grad4_indices = grad4_indices.view(grad4.shape[0], -1, 4)

        # if len(block.shape) > 2:
        #     block = block.flatten(1, -1)

        # block_feature_maps.append(block.detach().cpu())
        block_feature_maps.append(torch.cat([grad1_features,
                                            grad2_features,
                                            grad3_features,
                                            grad4_features], dim=1))
        block_indices_maps.append(torch.cat([grad1_indices,
                                            grad2_indices,
                                            grad3_indices,
                                            grad4_indices], dim=1))


    block_feature_maps = torch.cat(block_feature_maps, 0)
    block_indices_maps = torch.cat(block_indices_maps, 0)

    return block_feature_maps, torch.cat(labs), torch.cat(images_ids), block_indices_maps

def random_crop_representations_no_grad(model: torch.nn.Module, loader, criterion, optimizer, device,
                                        random_indices, layer_number: int=3):
    if model.random_feature_crop:
        pass
    else:
        model.random_feature_crop = True
    model.to(device)
    criterion.to(device)
    model.eval()
    block_feature_maps = []
    block_indices_maps = []
    labs = []
    images_ids = []

    for i, batch in tqdm(enumerate(loader)):

        model.zero_grad()
        images, labels, image_id = batch
        labs.append(labels)
        images_ids.append(image_id)
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # if layer_number == 0:
            #     preds, block, _, _, _ = model(images)
            # elif layer_number == 1:
            #     preds, _, block, _, _ = model(images)
            # elif layer_number == 2:
            #     preds, _, _, block, _ = model(images)
            # elif layer_number == 3:
            #     preds, _, _, _, block = model(images)



            preds, block = model(images, use_only_flatten=False)

        if len(block.shape) > 2:
            block = block.flatten(1, -1)

        block_feature_maps.append(block.detach().cpu())


    return block_feature_maps, torch.cat(labs)



def train_client_on_common(model, loader, optimizer, criterion, scaler, device,
                           agg_block1, agg_block2, agg_block3, agg_block4, batch_size,
                           scheduler=None, sophisticated_average=True, a=0.1, b=0.2, c=0.3, d=0.4,
                           cross_entropy=None):

    model.to(device)
    model.train()
    criterion.to(device)
    cel = nn.CrossEntropyLoss()
    cos = nn.CosineSimilarity(1)

    running_loss = 0.0
    acc = 0.0
    for i, batch in tqdm(enumerate(loader)):

        # model.initialize_reset_gradients()
        # model.set_hooks()

        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        batch_agg_block4 = agg_block4[i * batch_size: (i + 1) * batch_size]
        batch_agg_block4 = batch_agg_block4.to(device)
        optimizer.zero_grad()
        model.zero_grad()

        with torch.cuda.amp.autocast():
            preds, _, _, _, block4 = model(images)
            loss_cos4 = torch.mean(1 - cos(block4, batch_agg_block4))
            loss = loss_cos4


        acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()

    return running_loss / (i + 1), acc / (i + 1)

def train_client_on_common_single_block(model, loader, optimizer, criterion, scaler, device, agg_block, block_indices,
                                        images_ids, batch_size,
                                        layer_number, scheduler=None, labs=None, sophisticated_average=True, a=0.1,
                                        b=0.2, c=0.3, d=0.4, cross_entropy=None, use_cosine=False, use_contrast=False):

    model.to(device)
    model.train()
    criterion.to(device)
    cel = nn.CrossEntropyLoss()
    cos = SimCosLoss(device, 1)

    running_loss = 0.0
    acc = 0.0
    batch_counter = 0
    for i, batch in tqdm(enumerate(loader)):

        images, labels, ids = batch
        images, labels = images.to(device), labels.to(device)
        # batch_agg_block = agg_block[i * batch_size: (i + 1) * batch_size]

        batch_agg_block = agg_block[ids]
        batch_block_indices = block_indices[ids]

        batch_agg_block = batch_agg_block.to(device)
        batch_block_indices.to(device)

        prev_labs = labs[i * batch_size: (i + 1) * batch_size]
        prev_labs = prev_labs.to(device)
        if (prev_labs == labels).all():
            # print('yeah mfs')
            batch_counter += 1

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # preds, block = model(images, use_only_flatten=False)
            preds, layer1_features, layer2_features, layer3_features, layer4_features = model(images, use_only_flatten=False)



            block_feature_from_grad = use_grad_to_get_features_from_indices(
                block_indices=batch_block_indices,
                layer1=layer1_features,
                layer2=layer2_features,
                layer3=layer3_features,
                layer4=layer4_features
            )
            similarity_loss = cos(block_feature_from_grad, batch_agg_block)
            cross_entropy_loss = cel(preds, labels)
            loss = similarity_loss + cross_entropy_loss
            # loss = 0

        acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()
    if batch_counter != len(loader):
        print('TACTICAL NUKE INCOMING!!!!')

    return running_loss / (i + 1), acc / (i + 1)

def train_client_on_common_single_block_no_grad(model, loader, optimizer, criterion, scaler, device, agg_block, block_indices,
                                        images_ids, batch_size,
                                        layer_number, scheduler=None, labs=None, sophisticated_average=True, a=0.1,
                                        b=0.2, c=0.3, d=0.4, cross_entropy=None, use_cosine=False, use_contrast=False):

    model.to(device)
    model.train()
    criterion.to(device)
    cel = nn.CrossEntropyLoss()
    cos = SimCosLoss(device, 1)

    running_loss = 0.0
    acc = 0.0
    batch_counter = 0
    for i, batch in tqdm(enumerate(loader)):

        images, labels, ids = batch
        images, labels = images.to(device), labels.to(device)
        # batch_agg_block = agg_block[i * batch_size: (i + 1) * batch_size]

        batch_agg_block = agg_block[ids]
        batch_block_indices = block_indices[ids]

        batch_agg_block = batch_agg_block.to(device)
        batch_block_indices.to(device)

        prev_labs = labs[i * batch_size: (i + 1) * batch_size]
        prev_labs = prev_labs.to(device)
        if (prev_labs == labels).all():
            # print('yeah mfs')
            batch_counter += 1

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # preds, block = model(images, use_only_flatten=False)
            preds, block = model(images, use_only_flatten=False)




        similarity_loss = cos(block, batch_agg_block)
        cross_entropy_loss = cel(preds, labels)
        loss = similarity_loss + cross_entropy_loss
        # loss = 0

        acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()
    if batch_counter != len(loader):
        print('TACTICAL NUKE INCOMING!!!!')

    return running_loss / (i + 1), acc / (i + 1)

def train_client_on_common_multiple_block(model, loader, optimizer, criterion, scaler, device, agg_block, batch_size,
                                        layer_number, scheduler=None, labs=None, sophisticated_average=True, a=0.1,
                                        b=0.2, c=0.3, d=0.4, cross_entropy=None, use_cosine=False, use_contrast=False):

    model.to(device)
    model.train()
    criterion.to(device)
    cel = nn.CrossEntropyLoss()
    cos = SimCosLoss(device, 1)

    running_loss = 0.0
    acc = 0.0
    print('Starting for specific loader')
    batch_counter = 0
    for i, batch in tqdm(enumerate(loader)):

        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        batch_agg_block = agg_block[i * batch_size: (i + 1) * batch_size]
        batch_agg_block = batch_agg_block.to(device)
        prev_labs = labs[i * batch_size: (i + 1) * batch_size]
        prev_labs = prev_labs.to(device)
        if (prev_labs == labels).all():
            # print('yeah mfs')
            batch_counter += 1
        batch_agg_block = batch_agg_block.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            preds, block0, block1, block2, block3 = model(images)


            similarity_loss = cos(block, batch_agg_block)
            cross_entropy_loss = cel(preds, labels)
            loss = similarity_loss + cross_entropy_loss

        acc += torch.sum((torch.argmax(preds, 1).float() == labels) / len(labels))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()
    if batch_counter != len(loader):
        print('TACTICAL NUKE INCOMING!!!!')

    return running_loss / (i + 1), acc / (i + 1)
