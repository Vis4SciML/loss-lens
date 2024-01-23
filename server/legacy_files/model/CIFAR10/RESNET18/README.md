# RESNET18 models
The models under the current folder are trained by `RESNET18` defined by the follows.

```python
model = torchvision.models.resnet18(weights = None)
model.fc = nn.Linear(512, num_classes)
```

To get all the pre-trained models, you need to run the following commands to get models.

```shell
wget -O RESNET18_model_threshold_\{00\}.pt https://www.dropbox.com/s/oscsvungvizvcym/RESNET18_model_threshold_%7B00%7D.pt?dl=0
```

```shell
wget -O RESNET18_model_threshold_\{20\}.pt https://www.dropbox.com/s/i9z6t746lp8guqe/RESNET18_model_threshold_%7B20%7D.pt?dl=0
```

```shell
wget -O RESNET18_model_threshold_\{40\}.pt https://www.dropbox.com/s/ga3qyfuoat7aj13/RESNET18_model_threshold_%7B40%7D.pt?dl=0
```

```shell
wget -O RESNET18_model_threshold_\{60\}.pt https://www.dropbox.com/s/h3jntiyl7b7fcm5/RESNET18_model_threshold_%7B60%7D.pt?dl=0
```

```shell
wget -O RESNET18_model_threshold_\{80\}.pt https://www.dropbox.com/s/jeg6zelh6ti7ifg/RESNET18_model_threshold_%7B80%7D.pt?dl=0
```

```shell
wget -O RESNET18_model_threshold_\{100\}.pt https://www.dropbox.com/s/qdynzejjrpr1mrb/RESNET18_model_threshold_%7B100%7D.pt?dl=0
```
