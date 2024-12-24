import os
from PIL import Image
import numpy as np
from torchvision import transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
from collections import Counter


def split_train_valid(data_name="cwt"):
    if data_name == "stft":
        train_set = f"{data_name}/train"
    else:
        assert data_name == "cwt"
        train_set = f"{data_name}/train"
    name_list = os.listdir(train_set)
    np.random.seed(1)
    np.random.shuffle(name_list)
    size = int(0.8 * len(name_list))
    train_list = name_list[:size]
    valid_list = name_list[size:]
    print(f" {data_name} Train set:  {Counter([int(x.split('.')[0].split('_')[-1]) for x in train_list])}")
    print(f"{data_name} Valid set:  {Counter([int(x.split('.')[0].split('_')[-1]) for x in valid_list])}")
    return train_list, valid_list, train_set


class BtchLoadFftData(Dataset):
    def __init__(self, data_list, img_file):
        super(BtchLoadFftData, self).__init__()

        self.data = data_list
        self.img_file = img_file

        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(size=64),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        name = self.data[index]
        label = name.split('.')[0].split('_')[-1]
        mel_file = os.path.join(self.img_file, name)
        img = Image.open(mel_file)
        feature = img.convert('RGB')

        feature = self.transforms(feature)
        label = torch.tensor(int(label))
        return feature, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    trains, valid, path = split_train_valid(data_name="stft")

    data = BtchLoadFftData(data_list=trains, img_file=path)
    dataloader = DataLoader(data, batch_size=20, shuffle=True)
    for i, batch in enumerate(dataloader):
        features, target = batch
        print('features array shape:', features.shape)
        print('label:', target)
        print('\n')

    # Train set:  Counter({0: 376, 1: 344})
    # Valid set:  Counter({1: 94, 0: 87})
