import gc
import os
from typing import List
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, \
    RandomContrast, RandomSaturation, RandomBrightness
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

def get_cifar10_splited(num_clients, trans, root='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/data'):
    trainset = CIFAR10(root=root, train=True, download=True, transform=trans)

    # 2. Shuffle and split indices
    indices = list(range(len(trainset)))
    torch.manual_seed(42)  # For reproducibility
    torch.randperm(len(trainset), out=torch.tensor(indices))

    split_size = len(trainset) // num_clients
    client_indices = [indices[i * split_size: (i + 1) * split_size] for i in range(num_clients)]
    return client_indices, trainset

def get_cifar10_val(valid_transformations, root='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/data'):
    validset = CIFAR10(root=root, train=False, download=True, transform=valid_transformations)
    return validset

def get_cifar10_splited_big_common(num_clients, trans,
                                   root='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/data',
                                   special_client_size=8000):
    special_indices_per_class_from_total = int(special_client_size / 10)
    if num_clients < 2:
        raise ValueError("Number of clients must be at least 2.")

    # Load the CIFAR10 dataset
    trainset = CIFAR10(root=root, train=True, download=True, transform=trans)

    # Shuffle indices
    indices = torch.randperm(len(trainset)).tolist()

    # Organize indices by class
    class_indices = [[] for _ in range(10)]  # CIFAR10 has 10 classes
    for idx in indices:
        _, label = trainset[idx]
        class_indices[label].append(idx)

    # First subset (special client)
    special_client_indices = []
    for class_list in class_indices:
        special_client_indices.extend(class_list[:special_indices_per_class_from_total])
        del class_list[:special_indices_per_class_from_total]

    # Calculate the number of images per class for the remaining clients
    remaining_images_per_class = len(class_indices[0])
    images_per_class_per_client = remaining_images_per_class // (num_clients - 1)

    # Distribute remaining images among other clients
    client_indices = [special_client_indices]  # Start with the special client
    for _ in range(num_clients - 1):
        client_subset = []
        for class_list in class_indices:
            client_subset.extend(class_list[:images_per_class_per_client])
            del class_list[:images_per_class_per_client]
        client_indices.append(client_subset)

    return client_indices, trainset

def get_cifar100_splited(num_clients, trans, root='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/data'):
    """

    @param num_clients: Number of local clients
    @param trans: Transformations (Only resize of you use FFCV)
    @param root: Path of the dataset
    @return:
        client_indices: Indices of each client
        trainset: The dataset
    """
    trainset = CIFAR100(root=root, train=True, download=True, transform=trans)

    # 2. Shuffle and split indices
    indices = list(range(len(trainset)))
    torch.manual_seed(42)  # For reproducibility
    torch.randperm(len(trainset), out=torch.tensor(indices))

    split_size = len(trainset) // num_clients
    client_indices = [indices[i * split_size: (i + 1) * split_size] for i in range(num_clients)]
    return client_indices, trainset

def get_cifar100_val(valid_transformations, root='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/data'):
    validset = CIFAR100(root=root, train=False, download=True, transform=valid_transformations)
    return validset

class CustomDataset(Dataset):
    def __init__(self, base_dataset, indices):
        """

        @param base_dataset: Main dataset
        @param indices: Indices for the federated learning
        """
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base_dataset[self.indices[index]]

class CustomDatasetCommon(Dataset):
    def __init__(self, base_dataset, indices, transforms):
        """

        @param base_dataset: Main dataset
        @param indices: Indices for the federated learning
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transforms = transforms
        self.id = torch.arange(0, len(self.indices))
        self.image_ids = {self.indices[i]: self.id[i].item() for i in range(len(self.indices))}

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        image, label = self.base_dataset[self.indices[index]]
        return self.transforms(image), label, self.image_ids[self.indices[index]]

def ffcv_writer(path, trainsets, validset,):
    """
    Create the FFCV datasets

    @param path: Path to save beton datasets
    @param trainsets: Datasets with the train data
    @param validset: Validation dataset
    @return: Nothing
    """
    datasets = {}

    for i in range(len(trainsets)):
        datasets[f'train_{i}'] = trainsets[i]

    datasets['valid'] = validset
    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'{path}/cifar_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)
        print(f'Beton {name} done!')
        # del writer
        # gc.collect()

def ffcv_loaders(batch_size, cifar_mean, cifar_std, path_betson, device, num_workers=10, common_dataset=False):
    """
    Create the loaders for FFCV from the created beton

    @param batch_size: Batch size of the loader
    @param cifar_mean: Mean for the normalization
    @param cifar_std: STD for the normalization
    @param path_betson: Path that the betsons are saved
    @param num_workers: Workers of the loaders
    @param common_dataset: If we have a common split of the dataset over the clients
    @return:
        train_loaders: Train loaders for each local node
        valid_loader: Validation laoder with the valid set of the Dataset
    """
    betsons = os.listdir(path_betson)
    betsons_paths = []
    for i in range(len(betsons)):
        betsons_paths.append(os.path.join(path_betson, betsons[i]))

    train_loaders =[]
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device(device)), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomBrightness(0.1),
                RandomSaturation(0.1),
                RandomContrast(0.1),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, cifar_mean))),  # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(torch.device(device), non_blocking=False),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(cifar_mean, cifar_std),
        ])

        # Create loaders
        if name == 'train':
            for i in range(len(betsons)):
                if 'train' in betsons[i]:
                    if common_dataset:
                        if train_loaders == []:
                            train_loaders.append(Loader(betsons_paths[i],
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   order=OrderOption.SEQUENTIAL,
                                                   drop_last=(name == 'train'),
                                                   pipelines={'image': image_pipeline,
                                                              'label': label_pipeline}))
                        else:
                            train_loaders.append(Loader(betsons_paths[i],
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        order=OrderOption.RANDOM,
                                                        drop_last=(name == 'train'),
                                                        pipelines={'image': image_pipeline,
                                                                   'label': label_pipeline}))
                    else:
                        train_loaders.append(Loader(betsons_paths[i],
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    order=OrderOption.RANDOM,
                                                    drop_last=(name == 'train'),
                                                    pipelines={'image': image_pipeline,
                                                               'label': label_pipeline}))
        else:
            for i in range(len(betsons)):
                if 'valid' in betsons[i]:
                    valid_loader = (Loader(betsons_paths[i],
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           order=OrderOption.RANDOM,
                                           drop_last=(name == 'valid'),
                                           pipelines={'image': image_pipeline,
                                                      'label': label_pipeline}))

    return train_loaders, valid_loader

class CommonDataset(Dataset):
    def __init__(self, images, labels, block1, block2, block3, block4):
        super(CommonDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.block4 = block4

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item], self.block1[item], self.block2[item], self.block3[item], self.block4[item]

class CommonDatasetDataset(Dataset):
    def __init__(self, images, labels):
        super(CommonDatasetDataset, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]




