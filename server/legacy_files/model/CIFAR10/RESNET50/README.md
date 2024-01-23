# RESNET50 models
The models under the current folder are trained by `RESNET50` defined by the follows.

```python
model = torchvision.models.resnet50(weights = None)
model.fc = nn.Linear(2048, num_classes)
```

To get all the pre-trained models, you need to run the following commands to get models.

```shell
wget -O RESNET50_model_threshold_\{00\}.pt https://www.dropbox.com/s/crfebs0sdacewv5/RESNET50_model_threshold_%7B00%7D.pt?dl=0
```

```shell
wget -O RESNET50_model_threshold_\{20\}.pt https://www.dropbox.com/s/75vp5oea1jpiwyl/RESNET50_model_threshold_%7B20%7D.pt?dl=0
```

```shell
wget -O RESNET50_model_threshold_\{40\}.pt https://www.dropbox.com/s/kx8fdaiucsvrs25/RESNET50_model_threshold_%7B40%7D.pt?dl=0
```

```shell
wget -O RESNET50_model_threshold_\{60\}.pt https://www.dropbox.com/s/g4dpi7vcn7gjtzs/RESNET50_model_threshold_%7B60%7D.pt?dl=0
```

```shell
wget -O RESNET50_model_threshold_\{80\}.pt https://www.dropbox.com/s/xjqy6xm44g6tyv9/RESNET50_model_threshold_%7B80%7D.pt?dl=0
```

```shell
wget -O RESNET50_model_threshold_\{100\}.pt https://www.dropbox.com/s/ni79mmznwm2qe6c/RESNET50_model_threshold_%7B100%7D.pt?dl=0
```
