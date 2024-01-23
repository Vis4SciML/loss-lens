# python3 calculate_cnn_mnist.py --model=CNN --reshapex=397 --reshapey=64 --losssteps=40 --device=nersc
# python3 calculate_torchvision_cifar10.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=40 --device=nersc --number_of_workers=12
# python3 calculate_torchvision_cifar10.py --model=RESNET50 --reshapex=234 --reshapey=128 --losssteps=40 --device=nersc --number_of_workers=12
# python3 calculate_torchvision_cifar10.py --model=VGG16 --reshapex=145969 --reshapey=704 --losssteps=40 --device=nersc --number_of_workers=12
# python3 calculate_vit_cifar10.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=40 --device=nersc
# python3 calculate_cnn_mnist_layer_cka.py --model=CNN --reshapex=397 --reshapey=64 --losssteps=40 --device=nersc

# python3 calculate_torchvision_cifar10_training_loss.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=20 --device=nersc --number_of_workers=0
# echo "Finished Training Loss"

# python3 calculate_torchvision_cifar10_training_loss_3d.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=10 --device=nersc --number_of_workers=0
# echo "Finished Training Loss Landscapes 3D"

# python3 calculate_torchvision_cifar10_loss_landscapes.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=20 --device=nersc --number_of_workers=0
# echo "Finished Loss Landscapes"

# python3 calculate_torchvision_cifar10_loss_landscapes_3d.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=10 --device=nersc --number_of_workers=0
# echo "Finished Loss Landscapes 3D"

# python3 calculate_torchvision_cifar10_model_evaluation.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=10 --device=nersc --number_of_workers=0
# echo "Finished Model Evaluation"

# python3 calculate_torchvision_cifar10_prediction_distribuation.py --model=RESNET18 --reshapex=228 --reshapey=64 --losssteps=10 --device=nersc --number_of_workers=12
# echo "Finished Prediction Distribution"

# python3 calculate_vit_cifar10_similarity.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=20 --device=nersc
# echo "Finished Global Similarity"

# python3 calculate_vit_cifar10_layer_similarity.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=20 --device=nersc
# echo "Finished Layer Similarity"

# python3 calculate_vit_cifar10_model_evaluation.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=20 --device=nersc
# echo "Finished Model Evaluation"

# python3 calculate_vit_cifar10_prediction_distribuation.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=20 --device=nersc
# echo "Finished Prediction Distribution"

# python3 calculate_vit_cifar10_training_loss.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=15 --device=nersc
# echo "Finished Training Loss"

# python3 calculate_vit_cifar10_loss_landscapes.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=15 --device=nersc
# echo "Finished Loss Landscapes"

# python3 calculate_vit_cifar10_loss_landscapes_3d.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=5 --device=nersc
# echo "Finished Loss Landscapes 3D"

python3 calculate_vit_cifar10_training_loss_3d.py --model=VIT --reshapex=464 --reshapey=128 --losssteps=5 --device=nersc
echo "Finished Training Loss 3D"
