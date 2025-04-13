import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10


def load_datasets(num_clients: int, dataset, data_split):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if dataset == "CIFAR10":
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(
           "The dataset is not implemented"
        )

    if data_split == "iid":
        # Split training set into `num_clients` partitions to simulate different local datasets
        props = [1/num_clients]*num_clients
        datasets = random_split(trainset, props, torch.Generator().manual_seed(42))
    elif data_split == "non_iid_number":
        # TO IMPLEMENT
        pass
    elif data_split == "non_iid_class":
        num_classes = len(trainset.classes)
        class_per_client = num_classes/num_clients
        data_indices_each_client = {client: [] for client in range(num_clients)}
        for c in range(num_classes):
            indices = (torch.tensor(trainset.targets)[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
            client_belong = int(c/class_per_client)
            data_indices_each_client[client_belong].extend(list(indices))
        datasets = []
        for i in range(num_clients):
            datasets.append(Subset(trainset, data_indices_each_client[i]))
    else:
        raise NotImplementedError(
           "The data split is not implemented"
        )

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def get_data_loader(num_clients: int, cid: int, dataset = "CIFAR10", data_split = "iid"):
    trainloaders, valloaders, testloader = load_datasets(num_clients, dataset, data_split)
    return trainloaders[cid], valloaders[cid], testloader