# libraries
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import math
from utils import *
from tqdm import tqdm
from robustbench.data import load_cifar10c
from torchmetrics import Accuracy

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,inplanes,planes,residual_not,batch_norm_not,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,residual_not,batch_norm_not,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batch_norm_not:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual

        out = self.relu(out)

        return out

ALPHA_ = 1

class ResNet(nn.Module):

    def __init__(self,depth,residual_not=True,batch_norm_not=True,base_channel=16,num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        # block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock
        self.base_channel = int(base_channel)
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.inplanes = self.base_channel * ALPHA_
        self.conv1 = nn.Conv2d(3,self.base_channel * ALPHA_,kernel_size=3,padding=1,bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(self.base_channel * ALPHA_)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,self.base_channel * ALPHA_,n,self.residual_not,self.batch_norm_not)
        self.layer2 = self._make_layer(block,self.base_channel * 2 * ALPHA_,n,self.residual_not,self.batch_norm_not,stride=2)
        self.layer3 = self._make_layer(block,self.base_channel * 4 * ALPHA_,n,self.residual_not,self.batch_norm_not,stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.base_channel * 4 * ALPHA_ * block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks,residual_not,batch_norm_not,stride=1):
        downsample = None
        if (stride != 1 or
                self.inplanes != planes * block.expansion) and (residual_not):
            if batch_norm_not:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes * block.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes * block.expansion,kernel_size=1,stride=stride,bias=False),)

        layers = nn.ModuleList()
        layers.append(
            block(self.inplanes, planes, residual_not, batch_norm_not, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, residual_not, batch_norm_not))

        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        if self.batch_norm_not:
            x = self.bn1(x)
        x = self.relu(x)  # 32x32
        output_list.append(x.reshape(x.size(0), -1))

        for layer in self.layer1:
            x = layer(x)  # 32x32
            output_list.append(x.reshape(x.size(0), -1))
        for layer in self.layer2:
            x = layer(x)  # 16x16
            output_list.append(x.reshape(x.size(0), -1))
        for layer in self.layer3:
            x = layer(x)  # 8x8
            output_list.append(x.reshape(x.size(0), -1))

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        output_list.append(x.reshape(x.size(0), -1))

        # return output_list, x
        return x

def resnet(**kwargs):
    # Constructs a ResNet model.
    return ResNet(**kwargs)

def getData(name, train_bs=128, test_bs=256):
    # Get the dataloader
    if name == 'augmented':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',train=True,download=True,transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,batch_size=train_bs,shuffle=True)

        testset = datasets.CIFAR10(root='../data',train=False,download=False,transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,batch_size=test_bs,shuffle=False)

    if name == 'original':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',train=True,download=True,transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,batch_size=train_bs,shuffle=True)

        testset = datasets.CIFAR10(root='../data',train=False,download=False,transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,batch_size=test_bs,shuffle=False)

    return train_loader, test_loader

def test_augmentation(model, test_loader, cuda=True):
    # Get the test performance
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('Testing Correct: ', correct / total_num, '\n')

def test_corruption(model, x, target, cuda=True):
    # get the test performance
    model.eval()
    if cuda:
        x, target = x.cuda(), target.cuda()
    preds = model(x)
    # get the accuracy
    accuracy = Accuracy(subset_accuracy=True)
    print("Testing Accuracy:" + str(accuracy(preds, target)))

# Training settings
parser = argparse.ArgumentParser(description='Training on Cifar10')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='Batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='Batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='Learning rate (default: 0.01)')
parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate ratio')
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80, 120], help='Decrease learning rate at these epochs.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('--batch-norm', action='store_false', help='Use batch norm or not')
parser.add_argument('--residual', action='store_false', help='Use residula connect or not')
parser.add_argument('--cuda', action='store_false', default=False, help='Use GPU or not')
parser.add_argument('--saving-folder', type=str, default='../model/CIFAR10/RESNET/', help='Choose saving folder')
parser.add_argument('--depth', type=int, default=20, help='ResNet model depth')
parser.add_argument('--augmentation', default='True', help='Use data augmentation or not')
parser.add_argument('--corruption', default='fog', help='Corruption type')
parser.add_argument('--testing', default='augmentation', help='Testing on augmentation or corruption')
args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get training dataset
data_name = ''
if args.augmentation == 'True':
    data_name = 'augmented'
    print("Training with data augmentation")
elif args.augmentation == 'False':
    data_name = 'original'
    print("Training without data augmentation")

train_loader, test_loader = getData(name=data_name, train_bs=args.batch_size, test_bs=args.test_batch_size)

# get model and optimizer
model = resnet(num_classes=10,
               depth=args.depth,
               residual_not=args.residual,
               batch_norm_not=args.batch_norm)
if args.cuda:
    model = model.cuda()
model = torch.nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epoch, gamma=args.lr_decay)

if not os.path.isdir(args.saving_folder):
    os.makedirs(args.saving_folder)

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    with tqdm(total=len(train_loader.dataset)) as progressbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            optimizer.step()
            optimizer.zero_grad()
            progressbar.set_postfix(loss=train_loss / total_num, acc=100. * correct / total_num)
            progressbar.update(target.size(0))
    
    # prepare the corruption testing data
    corruptions = [args.corruption]
    x_corruption, target_corruption = load_cifar10c(n_examples=args.test_batch_size, data_dir='../data', corruptions=corruptions, severity=5)
    target_corruption = target_corruption.long()

    train_loader_augmentation, test_loader_augmentation = getData(name='augmented', train_bs=args.batch_size, test_bs=args.test_batch_size)

    # test the model
    if args.testing == 'augmentation':
        test_augmentation(model, test_loader_augmentation, args.cuda)
    elif args.testing == 'corruption':
        test_corruption(model, x_corruption, target_corruption, cuda=args.cuda)
    lr_scheduler.step()

torch.save(model.state_dict(), args.saving_folder + 'resnet' + str(args.depth) + '_cifar10_' + data_name + ' ' + '_model_final.pt')