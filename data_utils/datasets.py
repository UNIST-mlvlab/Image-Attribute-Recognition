import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T


def Get_Dataset(experiment, data_path, label_path):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = T.Compose([
        T.Resize(size=(256, 128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
        ])
    transform_test = T.Compose([
        T.Resize(size=(256, 128)),
        T.ToTensor(),
        normalize
        ])

    if experiment == 'rap':
        train_dataset = MultiLabelDataset(root=data_path,
                    label=label_path+'/rap/train.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root=data_path,
                    label=label_path+'/rap/test.txt', transform=transform_test)
        return train_dataset, val_dataset



def default_loader(path):
    return Image.open(path).convert('RGB')


class MultiLabelDataset(data.Dataset, ):
    def __init__(self, root, label, transform=None, loader=default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)
