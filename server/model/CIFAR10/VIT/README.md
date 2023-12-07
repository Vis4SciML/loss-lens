# VIT models
The models under the current folder are trained `VIT` models with CIAF10 and CIFAR10-C defined by the follows.

```python
from vit_pytorch import ViT
model = ViT(image_size = 32,patch_size = 4,num_classes = 10,dim = 1024,depth = 6,heads = 16,mlp_dim = 2048,dropout = 0.1,emb_dropout = 0.1)
```

To get all the pre-trained models, you need to run the following commands to get models.

```shell
wget -O VIT_model_threshold_\{00\}.pt https://www.dropbox.com/s/5jba38c7jqf8k02/vitcifar_0.pt?dl=0
```

```shell
wget -O VIT_model_threshold_\{20\}.pt https://www.dropbox.com/s/lxnaagtu8ppf7ho/vitcifar_20.pt?dl=0
```

```shell
wget -O VIT_model_threshold_\{40\}.pt https://www.dropbox.com/s/c904edy0eon6ze5/vitcifar_40.pt?dl=0
```

```shell
wget -O VIT_model_threshold_\{60\}.pt https://www.dropbox.com/s/2i8m664fxj7ijvy/vitcifar_60.pt?dl=0
```

```shell
wget -O VIT_model_threshold_\{80\}.pt https://www.dropbox.com/s/dgppv9hx12nxe90/vitcifar_80.pt?dl=0
```

```shell
wget -O VIT_model_threshold_\{100\}.pt https://www.dropbox.com/s/wleo5oe01131ov9/vitcifar_100.pt?dl=0
```
