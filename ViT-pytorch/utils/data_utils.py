import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import glob
import os
import cv2
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
        #print(cue_list, assoc_list)
        label = cue_path.split('/')[-1].split('.')[0].split('_')[-1]
        label = int(label)
        # make cue and assoc of length 200 each
        total_len = 200
        cue_tensor = torch.tensor(cue_list + [23001]*(total_len - len(cue_list)), dtype=torch.int32)
        assoc_tensor = torch.tensor(assoc_list + [23001]*(total_len - len(assoc_list)), dtype=torch.int32)
        label = torch.tensor([label], dtype=torch.int32)
        #print(cue_tensor, assoc_tensor, label)
        return cue_tensor, assoc_tensor, label       

class Image_Captioning(Dataset):
   
    def __init__(self, images_root, captions_root, transforms):
        self.images_list = glob.glob(images_root + '/*.pt')        
        self.captions_list = glob.glob(captions_root + '/*.pt')
        self.transforms = transforms

    def __len__(self):
        return len(self.images_list)
       
    def __getitem__(self, idx):
        #extract image based on image associated with caption, because there are several captions per image

        caption_path = self.captions_list[idx]

        # Label = 1 for human, Label = 0 for AI
        label = caption_path.split('/')[-1].split('.')[0].split('_')[-1]
        label = int(label)

        # change this to server folder 
        img_folder = '/home/brandon/univ-judge/image_caption_data/images/'

        # get image path from caption path/file name
        image_path = img_folder + caption_path.split('/')[-1].split('.')[0].split('_')[0]

        image_path_new = cv2.imread(image_path)
        image_path_new = cv2.cvtColor(image_path_new, cv2.COLOR_BGR2RGB)

        image_transformed = self.transforms(image_path_new)

        image_tensor = torch.load(image_transformed)
        caption_loaded = torch.load(caption_path)
        print(image_tensor, caption_loaded)

        # maximum sequence length for image is 64, given 1024 by 1024 images with 16 x 16 patches
        # maximum sequence length for caption is 200
        # maximum sequence length for image + separator + caption is 265
        # This is invariant for all tasks, since some inputs may be longer than others 
        # Word association, conversation, image captioning

        # Make sure image patch tokens from conv2d do not conflict with dummy tokens

        total_len = 200
        #image_tensor = torch.tensor(image_list + [23001]*(total_len - len(image_list)), dtype=torch.int32)
        caption_tensor = torch.tensor(caption_loaded + [23001]*(total_len - len(caption_list)), dtype=torch.int32)
        label = torch.tensor([label], dtype=torch.int32)
        print(image_tensor, caption_tensor, label)

        return image_tensor, caption_tensor, label    

def get_loader(args):
    # local_rank is distributed training
    #if args.local_rank not in [-1, 0]:
        #torch.distributed.barrier()

    transform_train = transforms.Compose([
        #transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.Resize(1024, 1024)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        #transforms.Resize((args.img_size, args.img_size)),
        transforms.Resize(1024, 1024)
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

    # converts image to tensor automatically
    elif args.dataset == "hymenoptera":
        data_dir = "hymenoptera_data"
        trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_test) #if args.local_rank in [-1, 0] else None

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
        
    elif args.dataset == "image_captioning":
        batch_size=4
        validation_split = .25
        shuffle_dataset = True
        random_seed= 42

        # change paths to be correct
        data = Image_Captioning('/home/brandon/univ-judge/image_caption_data/images', 
        '/home/brandon/univ-judge/image_caption_data/captions', transform_train
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