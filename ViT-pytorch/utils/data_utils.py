import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import glob
import os
import numpy as np
logger = logging.getLogger(__name__)

class Word_Association(Dataset):
    
    def __init__(self, cue_root, assoc_root):
        self.assoc_list = glob.glob(assoc_root + '/*.pt')        
        self.cues_list = glob.glob(cue_root + '/*.pt')
        self.transforms = transforms

    def __len__(self):
        return len(self.cues_list)
        
    def __getitem__(self, idx):
        cue_path = self.cues_list[idx]
        assoc_path = self.assoc_list[idx]
        cue_list = torch.load(cue_path)
        assoc_list = torch.load(assoc_path)
        print(cue_list, assoc_list)
        label = cue_path.split('/')[-1].split('.')[0].split('_')[-1]
        label = int(label)
        # make cue and assoc of length 200 each
        total_len = 200
        cue_tensor = torch.tensor(cue_list + [23001]*(total_len - len(cue_list)), dtype=torch.int32)
        assoc_tensor = torch.tensor(assoc_list + [23001]*(total_len - len(assoc_list)), dtype=torch.int32)
        label = torch.tensor([label], dtype=torch.int32)
        print(cue_tensor, assoc_tensor, label)
        return cue_tensor, assoc_tensor, label       

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "hymenoptera":
        data_dir = "hymenoptera_data"
        trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_test) if args.local_rank in [-1, 0] else None

    # Directly returns train and test loaders for word association tastk instead only train and test set sunlike other if/else statements above
    elif args.dataset == "word_assoc":
        batch_size=4
        validation_split = .25
        shuffle_dataset = True
        random_seed= 42

        data = Word_Association('/home/brandon/univ-judge/word_assoc/cues', 
        '/home/brandon/univ-judge/word_assoc/associations'
        )          

        dataset_size = len(data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(data, 
                             batch_size=batch_size, 
                             sampler=train_sampler,
                             pin_memory=True)

        test_loader = DataLoader(data, 
                             batch_size=batch_size, 
                             sampler=valid_sampler,
                             pin_memory=True)   

        return train_loader, test_loader
        
    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    # test_sampler = SequentialSampler(testset)    #uncomment later when split into train and test

    train_loader = DataLoader(trainset,
                            sampler=train_sampler,
                            batch_size=args.train_batch_size,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True) if testset is not None else None

    return train_loader, test_loader


"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
"""