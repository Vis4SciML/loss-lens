'''Script developed by Tiankai et al.'''
# libraries
import numpy as np
import torch
import random
import numpy as np
import torch.optim as optim
import argparse
import time
import tqdm
import torch.nn as nn
from robustbench.data import load_cifar10c

import operation.functions as operation
import csv

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=123, help='random seed')
parser.add_argument('--lr', default=1e-3, help='learning rate')
parser.add_argument('--max_epochs', default=120, help='number of epochs')
parser.add_argument('--model', default='VIT', help='model name used')
parser.add_argument('--output', default='model.pt', help='output name')
parser.add_argument('--threshold', default=0.2, help='threshold')    
args = parser.parse_args()

# Variables
batch_size_train = 64
batch_size_test = 128
learning_rate = float(args.lr)
model_name = str(args.model)
max_epoch = int(args.max_epochs)
out = str(args.output)
threshold = float(args.threshold)
random_seed = int(args.seed)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seed
torch.backends.cudnn.deterministic = True
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
num_classes = 10

# Training
train_loader, test_loader = operation.get_data_cifar10(batch_size_train, batch_size_test)
# print(len(train_loader))
num_examples = len(train_loader)*batch_size_train
num_cifar10c = int(num_examples*threshold)
x,targets = load_cifar10c(n_examples=num_cifar10c, data_dir='./data/CIFAR10-C')
# print(x.size())
y1 = [x[batch_size_train*i:batch_size_train*i + batch_size_train,:,:,:] for i in range(int(x.size()[0]/batch_size_train))]
y2 = [targets[batch_size_train*i:batch_size_train*i + batch_size_train] for i in range(int(x.size()[0]/batch_size_train))]
# print(len(y1))
network = operation.get_model(model_name, num_classes).to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
epoch_train_loss = []

for epoch in tqdm.tqdm(range(max_epoch), desc='Training'):
    train_loss = []
    network.train()
    start = time.time()
    # Train on Clean
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.to(device)
        data = data.to(device)
        output = network(data)
        optimizer.zero_grad()
        loss = loss_fn(output,target)
        train_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        
    # Train on Corrupted
    for data2, target2 in zip(y1,y2):
        target = target2.long().to(device)
        data = data2.to(device)
        output = network(data)
        optimizer.zero_grad()
        loss = loss_fn(output,target)
        train_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        

    epoch_train_loss.append(np.mean(train_loss))
    with open(f'./log_{args.model}_thresh{args.threshold}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(epoch_train_loss) 
    
    torch.save(network.state_dict(), out)

print("One Training Model Done!")