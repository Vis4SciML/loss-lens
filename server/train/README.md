# Training model Instructions

In order to compare the different effects of different corruption inputs with different models, the training and calculation processes are divided into two parts for visualization. This folder will provide several examples of training the models, which will show how the models we trained and it might help to understand how the models should be look like while using this visualization tool. The trained models will be saved under the `model` folder under the upper level directory.

## 2-Layers MLP with MNIST
This is one toy example for this system. It will train a 2-layers MLP model with [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) and [MNIST-C](https://github.com/google-research/mnist-c) datasets. The models are defined as follows.

```python
class MLPSmall(torch.nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)
```

The MNIST-C dataset has 16 different kinds of corruptions and if we add the original MNIST dataset and add one more example which contains all the MNIST and MNIST-C datasets, we will have 18 input training datasets in total, which will generate 18 trained models in this example. The training process can be excute as follows.

```shell
python3 train_mlp_mnist.py --dataset=<training input dataset>
```

The training input dataset options are `original`, `brightness`, `canny_edges`, `dotted_line`, `fog`, `glass_blur`, `identity`, `impulse_noise`, `motion_blur`, `rotate`, `scale`, `shear`, `shot_noise`, `spatter`, `stripe`, `translate`, `zigzag`, `all`. The default setting is `original`.

We do provide a script file which contains all the commands to train all the models in MNIST and MNIST-C. You could just run the following command to obtain all the trained MNIST and MNIST-C models.

```shell
sh run_train_mlp_mnist.sh
```

One `model_initial.pt` and one `model_final.pt` will be generated under the corresponding folder which should be under `model/MNIST` folder. For example, two models of `original` training option will be generated under `model/MNIST/original` folder.

## RESNET with CIFAR-10
We have trained several ResNet models with [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10) and [CIFAR10-C](https://zenodo.org/record/2535967#.Y5oWsezMJPs) datasets. The model structure is obtained from [pytorchcv.model_provider](https://pypi.org/project/pytorchcv/). The models are defined as follows.

```python
model = torchvision.models.resnet18(weights = None)
model.fc = nn.Linear(512, num_classes)
```

```python
model = torchvision.models.resnet50(weights = None)
model.fc = nn.Linear(2048, num_classes)
```

We do provide a script file which contains several commands to train some RESNET models such as RESNET18 and RESNET50. You could just run the following command to obtain those trained models with different percents of CIFAR10-C or trained your own models with your settings.

```shell
sh run_train_resnet18_cifar10.sh
sh run_train_resnet50_cifar10.sh
```

You could find more details of functions under `operation` folder. The trained model will be generated under the corresponding folder which should be under `model/CIFAR10/RESNET18` or `model/CIFAR10/RESNET50` folders.

## ViT with CIFAR-10
We have trained several ViT models with [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10) and [CIFAR10-C](https://zenodo.org/record/2535967#.Y5oWsezMJPs) datasets. The model structure is obtained from `vit_pytorch`. The models are defined as follows.

```python
from vit_pytorch import ViT
model = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
```

We do provide a script file which contains several commands to train some ViT models. You could just run the following command to obtain those trained models with different percents of CIFAR10-C or trained your own models with your settings.

```shell
sh run_train_vit_cifar10.sh
```

You could find more details of functions under `operation` folder. The trained model will be generated under the corresponding folder which should be under `model/CIFAR10/VIT` folder.