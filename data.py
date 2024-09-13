import gc
import os
from typing import List
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import numpy as np

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, \
    RandomContrast, RandomSaturation, RandomBrightness
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

def get_data_iid(dataset_name: str, num_clients: int, transforms, train=True, special_client_size=10000):
    """
    Get the dataset splitted to clients in IID setting

    @param dataset_name: Name of the dataset
    @param num_clients: Total number of clients plus one for the shared subset
    @param transforms: Transformation for the images, different for the train and validation
    @param train: If the dataset will be for training
    @param special_client_size: Shared dataset length
    @return: Indices of each client images and the subsets
    """
    if train:
        if dataset_name == 'cifar10':
            return get_cifar10_splited_big_common(num_clients=num_clients, trans=transforms,
                                                  special_client_size=special_client_size)

        elif dataset_name == 'cifar100':
            return get_cifar100_splited_big_common(num_clients=num_clients, trans=transforms,
                                                   special_client_size=special_client_size)
        elif dataset_name == 'mnist':
            return get_mnist_splited_big_common(num_clients=num_clients, trans=transforms,
                                                special_client_size=special_client_size)
        else:
            raise Exception("dataset_name must be one of the following 3: cifar10, cifar100, mnist")
    else:
        if dataset_name == 'cifar10':
            return get_cifar10_val(valid_transformations=transforms)

        elif dataset_name == 'cifar100':
            return get_cifar100_val(valid_transformations=transforms)
        elif dataset_name == 'mnist':
            return get_mnist_val(valid_transformations=transforms)
        else:
            raise Exception("dataset_name must be one of the following 3: cifar10, cifar100, mnist")

def get_data_non_iid(dataset_name: str, num_clients: int, transforms, train=True, special_client_size=10000):
    """
        Get the dataset splitted to clients in on-IID setting

        @param dataset_name: Name of the dataset
        @param num_clients: Total number of clients plus one for the shared subset
        @param transforms: Transformation for the images, different for the train and validation
        @param train: If the dataset will be for training
        @param special_client_size: Shared dataset length
        @return: Indices of each client images and the subsets
        """

    if train:
        if dataset_name == 'cifar10':
            return get_cifar10_splited_non_iid_big_common(num_clients=num_clients, trans=transforms,
                                                          special_client_size=special_client_size)

        elif dataset_name == 'cifar100':
            return get_cifar100_splited_non_iid_big_common(num_clients=num_clients, trans=transforms,
                                                           special_client_size=special_client_size)
        else:
            raise Exception("dataset_name must be one of the following 3: cifar10, cifar100")
    else:
        if dataset_name == 'cifar10':
            return get_cifar10_val(valid_transformations=transforms)

        elif dataset_name == 'cifar100':
            return get_cifar100_val(valid_transformations=transforms)
        else:
            raise Exception("dataset_name must be one of the following 3: cifar10, cifar100")

def get_cifar10_splited_big_common(num_clients, trans,
                                   root='/home/atpsaltis/Anestis/datasets',
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

def get_cifar100_splited_big_common(num_clients, trans,
                                   root='/home/atpsaltis/Anestis/datasets',
                                   special_client_size=8000):
    special_indices_per_class_from_total = int(special_client_size / 10)
    if num_clients < 2:
        raise ValueError("Number of clients must be at least 2.")

    # Load the CIFAR10 dataset
    trainset = CIFAR100(root=root, train=True, download=True, transform=trans)

    # Shuffle indices
    indices = torch.randperm(len(trainset)).tolist()

    # Organize indices by class
    class_indices = [[] for _ in range(100)]  # CIFAR10 has 10 classes
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

def get_mnist_val(valid_transformations, root='/home/atpsaltis/Anestis/datasets'):
    validset = MNIST(root=root, train=False, download=True, transform=valid_transformations)
    return validset

def get_mnist_splited_big_common(num_clients, trans,
                                   root='/home/atpsaltis/Anestis/datasets',
                                   special_client_size=8000):
    special_indices_per_class_from_total = int(special_client_size / 10)
    if num_clients < 2:
        raise ValueError("Number of clients must be at least 2.")

    # Load the CIFAR10 dataset
    trainset = MNIST(root=root, train=True, download=True, transform=trans)

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

def get_cifar100_val(valid_transformations, root='/home/atpsaltis/Anestis/datasets'):
    validset = CIFAR100(root=root, train=False, download=True, transform=valid_transformations)
    return validset

def get_cifar10_splited_non_iid_big_common(num_clients, trans, alpha, root='/home/atpsaltis/Anestis/datasets', special_client_size=10000):
    if num_clients < 2:
        raise ValueError("Number of clients must be exactly 6.")

    # Load the CIFAR100 dataset
    trainset = CIFAR10(root=root, train=True, download=True, transform=trans)

    # Shuffle indices
    indices = torch.randperm(len(trainset)).tolist()

    # Organize indices by class
    class_indices = [[] for _ in range(10)]
    for idx in indices:
        _, label = trainset[idx]
        class_indices[label].append(idx)

    # First subset (special client) -- using an IID approach
    special_client_indices = []
    for class_list in class_indices:
        special_client_indices.extend(class_list[:int(special_client_size / 10)])
        del class_list[:int(special_client_size / 10)]

    # Calculate the number of images for each of the remaining clients
    images_per_non_special_client = (50000 - special_client_size) // (num_clients - 1)

    client_indices = [special_client_indices]

    # Distribute remaining images among other clients non-IID
    for _ in range(num_clients - 1):
        client_subset = []
        while len(client_subset) < images_per_non_special_client:
            # Sample proportions for each class using Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * 10)
            # Calculate the number of images to assign based on proportions
            max_assignable = images_per_non_special_client - len(client_subset)
            images_distribution = np.floor(proportions * max_assignable).astype(int)

            # Adjust for any small shortfall due to rounding
            if images_distribution.sum() < max_assignable:
                shortfall = max_assignable - images_distribution.sum()
                top_classes = np.argsort(proportions)[-shortfall:]  # Get indices with highest proportions
                images_distribution[top_classes] += 1  # Distribute shortfall to top classes proportionally

            # Assign images based on distribution
            for class_idx, num_images in enumerate(images_distribution):
                if num_images > len(class_indices[class_idx]):
                    num_images = len(class_indices[class_idx])
                client_subset.extend(class_indices[class_idx][:num_images])
                del class_indices[class_idx][:num_images]

        client_indices.append(client_subset)

    return client_indices, trainset


def get_cifar100_splited_non_iid_big_common(num_clients, trans, alpha, root='/home/atpsaltis/Anestis/datasets',
                                            special_client_size=10000):
    if num_clients < 2:
        raise ValueError("Number of clients must be exactly 6.")

    # Load the CIFAR100 dataset
    trainset = CIFAR100(root=root, train=True, download=True, transform=trans)

    # Shuffle indices
    indices = torch.randperm(len(trainset)).tolist()

    # Organize indices by class
    class_indices = [[] for _ in range(100)]
    for idx in indices:
        _, label = trainset[idx]
        class_indices[label].append(idx)

    # First subset (special client) -- using an IID approach
    special_client_indices = []
    for class_list in class_indices:
        special_client_indices.extend(class_list[:int(special_client_size / 100)])
        del class_list[:int(special_client_size / 100)]

    # Calculate the number of images for each of the remaining clients
    images_per_non_special_client = (50000 - special_client_size) // (num_clients - 1)

    client_indices = [special_client_indices]

    # Distribute remaining images among other clients non-IID
    for _ in range(num_clients - 1):
        client_subset = []
        while len(client_subset) < images_per_non_special_client:
            # Sample proportions for each class using Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * 100)
            # Calculate the number of images to assign based on proportions
            max_assignable = images_per_non_special_client - len(client_subset)
            images_distribution = np.floor(proportions * max_assignable).astype(int)

            # Adjust for any small shortfall due to rounding
            if images_distribution.sum() < max_assignable:
                shortfall = max_assignable - images_distribution.sum()
                top_classes = np.argsort(proportions)[-shortfall:]  # Get indices with highest proportions
                images_distribution[top_classes] += 1  # Distribute shortfall to top classes proportionally

            # Assign images based on distribution
            for class_idx, num_images in enumerate(images_distribution):
                if num_images > len(class_indices[class_idx]):
                    num_images = len(class_indices[class_idx])
                client_subset.extend(class_indices[class_idx][:num_images])
                del class_indices[class_idx][:num_images]

        client_indices.append(client_subset)

    return client_indices, trainset


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

class CustomDatasetMNIST(Dataset):
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
        image, label = self.base_dataset[self.indices[index]]
        image = image.convert('RGB')
        return image, label

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

class CustomDatasetCommonMNIST(Dataset):
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
        image = image.convert('RGB')
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

def sanity_check_distribution(sets):
    common_subset = sets[0]
    subsets = sets[1:]
    common_check = {}
    subsets_check = []
    for _ in range(len(subsets)):
        subsets_check.append({})
    for batch in common_subset:
        if batch[1] not in common_check.keys():
            common_check[batch[1]] = 1
        else:
            common_check[batch[1]] += 1

    for i in range(len(subsets)):
        for batch in subsets[i]:
            if batch[1] not in subsets_check[i].keys():
                subsets_check[i][batch[1]] = 1
            else:
                subsets_check[i][batch[1]] += 1

    print(f'Common Dataset distribution: {common_check},\n')
    for i in range(len(subsets_check)):
        print(f'Subset {i + 1} distribution: {subsets_check[i]}')


