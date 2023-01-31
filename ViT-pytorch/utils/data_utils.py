import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
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

    elif args.dataset == "word_assoc":
        trainset = Word_Association('/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/pagrawal/klab/ViT_universaljudge/Datasets/word_assoc/cues', 
        '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/pagrawal/klab/ViT_universaljudge/Datasets/word_assoc/associations'
        )          
        # add testing_data also                  
        
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

    if(args.dataset=="word_assoc"):
        batch_size=4
        train_loader = DataLoader(trainset, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)
        test_loader = DataLoader(trainset, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)                       # change test_loader , have a split of training and test 

    else:
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
