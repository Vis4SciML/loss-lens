python3 calculate_torchvision_cifar10.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=4 --device=local
python3 calculate_cnn_mnist.py --model=CNN --reshapex=397 --reshapey=64 --losssteps=40 --device=local
python3 calculate_vit_cifar10.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=4 --device=local
# python3 calculate_torchvision_cifar10.py --model=RESNET50 --reshapex=234 --reshapey=128 --losssteps=4 --device=local
# python3 calculate_torchvision_cifar10.py --model=VGG16 --reshapex=145969 --reshapey=704 --losssteps=4 --device=local