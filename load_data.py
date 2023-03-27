import cv2
import torch
import pandas as pd
import os
import PIL
import matplotlib
import time


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None, device = "cpu"):
        self.df = pd.read_csv(csv_path)
        self.device = device
        self.images_folder = images_folder
        self.transform = transform
        self.img_tensor, self.label_tensor = self.load_data()
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return (self.img_tensor[index], self.label_tensor[index])
    
    def load_data(self):
        for i in range(len(self.df)):
            img_name = os.path.join(self.images_folder, self.df.iloc[i, 0])
            image = cv2.imread(img_name)
            label = torch.tensor(int(self.df.iloc[i, -1])).to(self.device)
            if self.transform:
                image = self.transform(image).to(self.device)
            if i == 0:
                img_tensor = image
                label_tensor = label
            else:
                img_tensor = torch.hstack((img_tensor, image))
                label_tensor = torch.hstack((label_tensor, label))
        return img_tensor, label_tensor
    
    def load_data1(self):
        f = open("./train.csv", "r")
        l = f.readlines()
        # img_path_list = glob(data_path)
        for i in range(1, len(l)):
            line = l[i].split(",")
            img_path = line[0]
            lab = torch.tensor(int(line[-1].strip()))
            # tranform the image to same size
            image = cv2.imread(os.path.join(self.images_folder, img_path))
            if self.transform:
                image = self.transform(image)
                
            if i == 1:
                img_tensor = image
                label = lab
            else:
                img_tensor = torch.hstack((img_tensor, image))
                label = torch.hstack((label, lab))
        return img_tensor, label
    


class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.iloc[index]["img_path"]
        image = cv2.imread(os.path.join(self.images_folder, filename))
        # image.show()
        if self.transform is not None:
            image = self.transform(image)
        return image
    
# train_dataset = CustomDataset("./train.csv", "./images")
# print(len(train_dataset))
# print(train_dataset[0])

# test_dataset = CustomTestDataset("./test.csv", "./images")
# print(len(test_dataset))



