# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from vit_pytorch import ViT

# def get_data_cifar10(batch_size_train, batch_size_test):

#     train_set = torchvision.datasets.CIFAR10('./files/', train=True, download=True,
#                             transform=torchvision.transforms.Compose([
#                                 torchvision.transforms.RandomCrop(32, padding=4),
#                                 torchvision.transforms.RandomHorizontalFlip(),
#                                 torchvision.transforms.ToTensor(),
#                                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ]))

#     train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
#     test_set = torchvision.datasets.CIFAR10('./files/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                              ]))

#     test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
#     return train_subset_loader, test_subset_loader

def get_data_cifar10(batch_size_train, batch_size_test, num_cores=0):

    train_set = torchvision.datasets.CIFAR10('../data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
    ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR10('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                               
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test, num_workers=num_cores)
    return train_subset_loader, test_subset_loader


# def get_data_cifar100(batch_size_train, batch_size_test):

#     train_set = torchvision.datasets.CIFAR100('./files/', train=True, download=True,
#                             transform=torchvision.transforms.Compose([
#                                 torchvision.transforms.RandomCrop(32, padding=4),
#                                 torchvision.transforms.RandomHorizontalFlip(),
#                                 torchvision.transforms.ToTensor(),
#                                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ]))

#     train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
#     test_set = torchvision.datasets.CIFAR100('./files/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                              ]))

#     test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
#     return train_subset_loader, test_subset_loader

def get_data_cifar100(batch_size_train, batch_size_test, num_cores=0):

    train_set = torchvision.datasets.CIFAR100('../data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
    ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR100('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test,num_workers=num_cores)
    return train_subset_loader, test_subset_loader

def log_loss(output,target):
    loss = torch.sum(-target*torch.nn.functional.log_softmax(output))
    loss = loss.sum()
    return loss

class CNN_6_2(nn.Module):

    def __init__(self, num_classes):
        super(CNN_6_2,self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3,64,(3,3))
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,(3,3))
        self.batch2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2,2),stride=(2,2))
        
        # Block 2
        self.conv3 = nn.Conv2d(64,128,(3,3))
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,(3,3))
        self.batch4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2,2),stride=(2,2))
        
        # Block 3
        self.conv5 = nn.Conv2d(128, 256, (3,3))
        self.batch5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256,(3,3))
        self.batch6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.linear1 = nn.Linear(256,128)
        self.batch7 = nn.BatchNorm1d(128)

        # Block 1
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batch6(x)
        x = F.relu(x)

        x = x.view(x.size()[0],-1)
        x = self.linear1(x)
        x = self.batch7(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

def get_model(name, num_classes):
    model = None
    if name == 'cnn_6_2':
        model = CNN_6_2(num_classes)
    elif name == 'RESNET18':
        model = torchvision.models.resnet18(weights = None)
        model.fc = nn.Linear(512, num_classes)
    elif name == 'RESNET50':
        model = torchvision.models.resnet50(weights = None)
        model.fc = nn.Linear(2048, num_classes)
    elif name == 'VGG16':
        model = torchvision.models.vgg16(weights = None)
    elif name == 'VIT':
        model = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
    return model
