python3 train_resnet_cifar10.py --lr 1e-2 --max_epochs 25 --model 'RESNET18' --output '../model/CIFAR10/RESNET18/RESNET18_model_threshold_{00}.pt' --threshold 0.0 > experiment.out
python3 train_resnet_cifar10.py --lr 1e-2 --max_epochs 25 --model 'RESNET18' --output '../model/CIFAR10/RESNET18/RESNET18_model_threshold_{20}.pt' --threshold 0.2 > experiment.out
python3 train_resnet_cifar10.py --lr 1e-2 --max_epochs 25 --model 'RESNET18' --output '../model/CIFAR10/RESNET18/RESNET18_model_threshold_{40}.pt' --threshold 0.4 > experiment.out
python3 train_resnet_cifar10.py --lr 1e-2 --max_epochs 25 --model 'RESNET18' --output '../model/CIFAR10/RESNET18/RESNET18_model_threshold_{60}.pt' --threshold 0.6 > experiment.out
python3 train_resnet_cifar10.py --lr 1e-2 --max_epochs 25 --model 'RESNET18' --output '../model/CIFAR10/RESNET18/RESNET18_model_threshold_{80}.pt' --threshold 0.8 > experiment.out
python3 train_resnet_cifar10.py --lr 1e-2 --max_epochs 25 --model 'RESNET18' --output '../model/CIFAR10/RESNET18/RESNET18_model_threshold_{100}.pt' --threshold 1.0 > experiment.out