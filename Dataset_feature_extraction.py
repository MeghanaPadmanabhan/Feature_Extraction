import json, os
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
import pickle
import imageio
from PIL import Image
import PIL
import numpy as np
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Dataset_FE(Dataset):
    def __init__(self, mode, data_len=None):
        self.mode = mode
        self.images=[]
        self.labels=[]

        if self.mode == 'train':
            with open("/home/cougarnet.uh.edu/mpadmana/Documents/Patient_disease/image_train.pkl","rb") as f:
                self.img_list=pickle.load(f)
            self.images = [imageio.imread(i) for i in self.img_list]
            with open("/home/cougarnet.uh.edu/mpadmana/Documents/Patient_disease/target_train.pkl","rb") as f:
                self.labels=pickle.load(f)
            self.labels = np.float32(self.labels)
            print("Original train image length: "+ str(len(self.images)))
            print("Original number of train labels" + str(len(self.labels)))


            number_of_diseases = len(self.labels[0])
            print("Number of diseases: "+ str(number_of_diseases))
            ## The number of images for each label.
            distribution = sum(self.labels)
            print("Distribution versus disease: "+ str(distribution))
            highest = max(distribution) # Class with maximum representation.

            aug_factors = []
            for i in range(0, number_of_diseases):
                aug_factors.append(highest // distribution[i])
            print("Aug_factors are: "+ str(aug_factors))

            original_number_of_images = len(self.images)

            for idx in range(original_number_of_images):
                img = Image.fromarray(self.images[idx], mode='RGB')
                self.images[idx] = img
                label_locations = list(np.where(self.labels[idx])[0])
                for position in label_locations:
                    a = aug_factors[position]
                    self.images, self.labels = self.augment_and_append(img, a, idx)


        if self.mode=='val':
            with open("/home/cougarnet.uh.edu/mpadmana/Documents/Patient_disease/image_val.pkl","rb") as f:
                self.img_list = pickle.load(f)
            self.images = [imageio.imread(i) for i in self.img_list]
            with open("/home/cougarnet.uh.edu/mpadmana/Documents/Patient_disease/target_val.pkl","rb") as f:
                self.labels = pickle.load(f)
            self.labels = np.float32(self.labels)

    def augment_and_append(self,img, a, idx):
            ## Do a-1 aug_steps if the aug_factor is a.
        transformation_list = ['RandomHorizontalFlip', 'RandomVerticalFlip', 'ColorJitter', 'RandomResizedCrop1','RandomResizedCrop2', 'RandomResizedCrop3', 'RandomResizedCrop4', 'RandomResizedCrop5',
                                   'RandomRotation1', 'RandomRotation2','RandomRotation3','RandomRotation4', 'RandomRotation5']

        for i in range(1, int(a)):
            if (i-1<len(transformation_list)):
                aug_step = transformation_list[i - 1]
                if aug_step == 'RandomHorizontalFlip':
                    transformed_img = transforms.RandomHorizontalFlip()(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomVerticalFlip':
                    transformed_img = transforms.RandomVerticalFlip()(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'ColorJitter':
                    transformed_img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05)(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomResizedCrop1':
                    transformed_img = transforms.RandomResizedCrop(224, scale=(0.7, 1.))(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomResizedCrop2':
                    transformed_img = transforms.RandomResizedCrop(224, scale=(0.8, 1.))(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomResizedCrop3':
                    transformed_img = transforms.RandomResizedCrop(224, scale=(0.85, 1.))(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomResizedCrop4':
                    transformed_img = transforms.RandomResizedCrop(224, scale=(0.9, 1.))(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomResizedCrop5':
                    transformed_img = transforms.RandomResizedCrop(224, scale=(0.7, 1.))(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomRotation1':
                    transformed_img = transforms.RandomRotation((0,20), resample=PIL.Image.BICUBIC)(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomRotation2':
                    transformed_img = transforms.RandomRotation((20,40), resample=PIL.Image.BICUBIC)(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomRotation3':
                    transformed_img = transforms.RandomRotation((40,60), resample=PIL.Image.BICUBIC)(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomRotation4':
                    transformed_img = transforms.RandomRotation((60,80), resample=PIL.Image.BICUBIC)(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))
                if aug_step == 'RandomRotation5':
                    transformed_img = transforms.RandomRotation((80,90), resample=PIL.Image.BICUBIC)(img)
                    np.float32(list(self.images).append(transformed_img))
                    np.float32(list(self.labels).append(self.labels[idx]))

        return self.images, self.labels

    def __getitem__(self, index):

        # img = torch.tensor(self.images[index]).div(255.).float()
        #img = torch.tensor(self.images[index]).float()
        #img = (img - img.min()) / (img.max() - img.min())
        #print("The size of image is: "+ str(img.shape))
        target = torch.tensor(self.labels[index].astype(np.float32))
        # labels = torch.tensor(np.where(self.labels[index])[0])
        # labels = labels.unsqueeze(0)
        # target = torch.zeros(labels.size(0), options.num_classes).scatter_(1, labels, 1.)
        # target = torch.tensor(target)

        if self.mode == 'train':
            img = self.images[index]
            img = transforms.Resize((224, 224), Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = img.div(255.).float()

            return img, target,index

        if self.mode=='val':
            img = Image.fromarray(self.images[index], mode='RGB')
            img = transforms.Resize((224, 224), Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = img.div(255.).float()
        #img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            return img, target,index

    def __len__(self):
        return len(self.labels)





