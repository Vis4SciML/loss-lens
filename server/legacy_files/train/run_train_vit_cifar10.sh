python3 train_vit_cifar10.py --seed=123456 --lr=0.0001 --max_epochs=60 --model='VIT' --output='../model/CIFAR10/VIT/VIT_model_seed_123456_threshold_{00}.pt' --threshold 0.0 > experiment.out
python3 train_vit_cifar10.py --seed=123456 --lr=0.0001 --max_epochs=60 --model='VIT' --output='../model/CIFAR10/VIT/VIT_model_seed_123456_threshold_{100}.pt' --threshold 1.0 > experiment.out

python3 train_vit_cifar10.py --seed=1234 --lr=0.0001 --max_epochs=60 --model='VIT' --output='../model/CIFAR10/VIT/VIT_model_seed_1234_threshold_{00}.pt' --threshold 0.0 > experiment.out
python3 train_vit_cifar10.py --seed=1234 --lr=0.0001 --max_epochs=60 --model='VIT' --output='../model/CIFAR10/VIT/VIT_model_seed_1234_threshold_{100}.pt' --threshold 1.0 > experiment.out

python3 train_vit_cifar10.py --seed=123 --lr=0.0001 --max_epochs=60 --model='VIT' --output='../model/CIFAR10/VIT/VIT_model_seed_123_threshold_{00}.pt' --threshold 0.0 > experiment.out
python3 train_vit_cifar10.py --seed=123 --lr=0.0001 --max_epochs=60 --model='VIT' --output='../model/CIFAR10/VIT/VIT_model_seed_123_threshold_{100}.pt' --threshold 1.0 > experiment.out