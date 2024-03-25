from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb
import random
import time

from models import HardCodedResNet, CentralNodeCommonDataset
from data import *
from losses import *
from common_dataset_training import *

CFG = {'name': 'CVPR_Fed',
       'dataset': 'Cifar10',
       'implementation': 'Pytorch',
       'objective': 'Classification',
       'pretrained': False,
       'n_classes': 10,
       'batch_size': 512,
       'initial_lr': 1e-03,
       'lr': 1e-03,
       'embedding_dim': 128,
       'seed': 42,
       'num_workers': 12,
       'n_views': 2,
       'N_CLIENTS': 6,
       'TRAIN_CIRCLES': -10,
       'EPOCHS_PER_ROUND': 40,
       'EPOCHS_PER_CIRCLE_CENTRAL': -10,
       'COMMON_DATASET_TRAINING': 100,
       'FED_ROUNDS': 20,
       'scheduler': 'Reduce on Platue',
       'scheduler patience': 5,
       'scheduler coeficient': 0.5,
       'coeficient for aggregation': 1,
       'layer_number': 3,
       'norm_features': True,
       'use_of_common_split': True,
       'wandb': True,
       'debug': False}

if CFG['wandb']:
    wandb.init(project='CommonSplitFL', entity='anestis_kastellos', config=CFG,
               name=f'ResNet18, pretrained {CFG["pretrained"]}, one block training, block used: {CFG["layer_number"]}'
                    f'private training w/ CEL, common training w/ similarity w\ normalized feature maps only torch '
                    f'operations, valid w/ CEL')

def import_random_seed(seed=42):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'GPU currently in use: {device}')

train_transformations = transforms.Compose([
    transforms.Resize(224),
    # transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_transformations = transforms.Compose([
    transforms.Resize(224),
    # transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

torch_train_transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

torch_valid_transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if __name__ == "__main__":

    import_random_seed()

    # CIFAR_MEAN = (0.485, 0.456, 0.406)
    # CIFAR_STD = (0.229, 0.224, 0.225)
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    num_clients = CFG['N_CLIENTS']
    manual_logger = num_clients * [None]

    client_indices, trainset = get_cifar10_splited(num_clients=num_clients, trans=train_transformations)
    validset = get_cifar10_val(valid_transformations=valid_transformations)



    # Create datasets for each client
    sets = [CustomDataset(trainset, client_indices[i]) for i in range(num_clients)]
    common_dataset = CustomDatasetCommon(trainset, client_indices[0], torch_train_transformations)
    all_labels = []
    for set in sets:
        print(f'Subsets lens: {len(set)}')
        all_labels.extend(set.base_dataset.targets)

    unique_labels, counts_per_class = np.unique(all_labels, return_counts=True)

    # ffcv loaders
    # ffcv_writer(path='/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/betson_loaders',
    #             trainsets=sets, validset=validset)
    loaders, valid_loader = ffcv_loaders(CFG['batch_size'], CIFAR_MEAN, CIFAR_STD,
                                         '/home/kastellosa/PycharmProjects/federated_learning/CVPR_nov_23/betson_loaders',
                                         device=device,
                                         num_workers=CFG['num_workers'],
                                         common_dataset=CFG['use_of_common_split'])
    # Creation of common dataloader with conventional torch

    if CFG['use_of_common_split']:
        # Get the common loader and the private loaders
        # common_loader = loaders[0]
        common_loader = DataLoader(common_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=10,
                                   pin_memory=True)
        loaders = loaders[1:]

    models = [HardCodedResNet(model_name='resnet18', n_classes=CFG['n_classes'], pretrained=CFG['pretrained'],
                              norm_features=CFG['norm_features'])
              for _ in range(num_clients)]

    central_node = CentralNodeCommonDataset(common_loader, CFG['N_CLIENTS'] - 1)

    optimizers = [torch.optim.Adam(model.parameters(), lr=CFG['initial_lr']) for model in models]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_clients)]
    common_dataset_criterion = [CommonCosineLoss() for _ in range(num_clients)]

    schedulers = [torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG['lr'],
                                                      final_div_factor=100,
                                                      steps_per_epoch=len(loaders[0]),
                                                      epochs=CFG['FED_ROUNDS'] * CFG['EPOCHS_PER_ROUND'] * num_clients)
                  for optimizer in optimizers]
    scaler = torch.cuda.amp.GradScaler()

    BEST_VAL_ACC = (num_clients - 1) * [-1]
    COMMON_BEST_VAL_ACC = (num_clients - 1) * [-1]

    total_epochs_passes = 1

    print('Initial Round of training to get the first representations of the common dataset')
    for round in range(CFG['FED_ROUNDS']):
        global_images_representations = []
        for client in range(CFG['N_CLIENTS'] - 1):
            model = models[client]
            optimizer = optimizers[client]
            criterion = criterions[client]
            loader = loaders[client]
            for local_epoch in range(CFG['EPOCHS_PER_ROUND']):
                start_time = time.time()
                train_loss, train_acc = one_epoch_train_private(model,
                                                                loader,
                                                                optimizer,
                                                                criterion,
                                                                scaler,
                                                                device)
                valid_loss, valid_acc = one_epoch_valid_private(model,
                                                                valid_loader,
                                                                optimizer,
                                                                criterion,
                                                                device)
                print(f'Client: {client + 1}, Epoch: {local_epoch + 1}\n Train loss: {train_loss}, train acc: {train_acc} '
                      f'\n Valid loss: {valid_loss}, valid acc: {valid_acc} '
                      f'\n Epoch time: {(time.time() - start_time) / 60} mins')
                if CFG['wandb']:
                    wandb.log({f'Client {client + 1} Train Loss': train_loss,
                               f'Client {client + 1} Train Accuracy': train_acc,
                               f'Client {client + 1} Valid Loss': valid_loss,
                               f'Client {client + 1} Valid Accuracy': valid_acc,
                               f'Client {client + 1} Epoch': local_epoch + 1,
                               f'Total Epochs' : total_epochs_passes})

                total_epochs_passes += 1

        print('Creating the first representations of the common dataset')

        global_images_representations = []
        labels = []
        for client in range(CFG['N_CLIENTS'] - 1):
            block, labs = first_inferance_for_feature_maps_specific_layer(
                models[client],
                common_loader,
                optimizers[client],
                device,
                layer_number=CFG['layer_number'])
            """
            The construction of the representations is:
                [[Client1 -> b1f, b2f, b3f, b4f],
                 [Client2 -> b1f, b2f, b3f, b4f],
                 .
                 .
                 .]
            """
            global_images_representations.append([block])
            labels.append(labs)

        # Check if the labels are the same in all the mini batches
        # print('Checking the labels! Hope they are ok')
        # for i in range(labels[0].shape[0]):
        #     if (labels[0][i] == labels[1][i]) and (labels[1][i] == labels[2][i]) and (labels[2][i] == labels[3][i]) and (labels[3][i] == labels[1][i]):
        #         print('Fuck Yeah')
        #     else:
        #         print('Nooooooooooooooooooooooooooooooooooooo')

        # Getting the average representations of each image
        # glob_reps = []
        # block1 = []
        # block2 = []
        # block3 = []
        # block4 = []
        block = []
        for i in range(len(global_images_representations)):
            # block1.append(global_images_representations[i][0])
            # block2.append(global_images_representations[i][1])
            # block3.append(global_images_representations[i][2])
            # block4.append(global_images_representations[i][3])

            # Single block usage
            block.append(global_images_representations[i][0])

        # block1 = torch.stack(block1)
        # block2 = torch.stack(block2)
        # block3 = torch.stack(block3)
        # block4 = torch.stack(block4)

        # Single block usage
        block = torch.stack(block)

        # global_images_representations = [torch.mean(block1, dim=0),
        #                                  torch.mean(block2, dim=0),
        #                                  torch.mean(block3, dim=0),
        #                                  torch.mean(block4, dim=0)]

        # Single block usage
        global_images_representations = [torch.mean(block, dim=0)]

        # del block1, block2, block3, block4

        # Single block usage
        del block

        print('Training with similarity representations learning')
        for client in range(CFG['N_CLIENTS'] - 1):
            model = models[client]
            optimizer = optimizers[client]
            criterion = common_dataset_criterion[client]
            valid_criterion = nn.CrossEntropyLoss()
            loader = loaders[client]
            for common_epoch in range(CFG['EPOCHS_PER_ROUND']):
                start_time = time.time()
                # train_loss, train_acc = train_client_on_common(model,
                #                                                common_loader,
                #                                                optimizer,
                #                                                criterion,
                #                                                scaler,
                #                                                device,
                #                                                agg_block1=global_images_representations[0],
                #                                                agg_block2=global_images_representations[1],
                #                                                agg_block3=global_images_representations[2],
                #                                                agg_block4=global_images_representations[3],
                #                                                batch_size=CFG['batch_size'],
                #                                                sophisticated_average=True,
                #                                                cross_entropy=False)
                train_loss, train_acc = train_client_on_common_single_block(model,
                                                                            common_loader,
                                                                            optimizer,
                                                                            criterion,
                                                                            scaler,
                                                                            device,
                                                                            labs=labs,
                                                                            agg_block=global_images_representations[0],
                                                                            batch_size=CFG['batch_size'],
                                                                            layer_number=CFG['layer_number'],
                                                                            use_contrast=True,
                                                                            )
                valid_loss, valid_acc = one_epoch_valid_private(model,
                                                                valid_loader,
                                                                optimizer,
                                                                valid_criterion,
                                                                device)
                total_epochs_passes += 1
                print(f'Similarity results'
                      f'\n Client: {client + 1}, Epoch: {local_epoch + 1}\n Train loss: {train_loss}, train acc: {train_acc} '
                      f'\n Valid loss: {valid_loss}, valid acc: {valid_acc} '
                      f'\n Epoch time: {(time.time() - start_time) / 60} mins')
                if CFG['wandb']:
                    wandb.log({f'Client {client + 1} Train Loss': train_loss,
                               f'Client {client + 1} Train Accuracy': train_acc,
                               f'Client {client + 1} Valid Loss': valid_loss,
                               f'Client {client + 1} Valid Accuracy': valid_acc,
                               f'Client {client + 1} Epoch': local_epoch + 1,
                               f'Total Epochs': total_epochs_passes})



