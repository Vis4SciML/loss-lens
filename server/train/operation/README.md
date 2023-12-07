# Training Operations Instructions
This is an instruction to introduce all the packed operation functions in the training pipeline. Currently under development.

## Models Instructions

```python
def get_model(name, num_classes):
    model = None
    if name == 'cnn_6_2':
        model = CNN_6_2(num_classes)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained = False)
        model.fc = nn.Linear(2048, num_classes)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained = False)
        model.fc = nn.Linear(2048, num_classes)
    return model
```

## Loss Instructions

```python
def log_loss(output,target):
    loss = torch.sum(-target*torch.nn.functional.log_softmax(output))
    loss = loss.sum()
    return loss
```

## Datasets Instructions

```python
def get_data_cifar10(batch_size_train, batch_size_test):

    train_set = torchvision.datasets.CIFAR10('./../files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
    ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR10('./../files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                               
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
    return train_subset_loader, test_subset_loader
```

```python
def get_data_cifar100(batch_size_train, batch_size_test):

    train_set = torchvision.datasets.CIFAR100('./files/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
    ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR100('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
    return train_subset_loader, test_subset_loader
```