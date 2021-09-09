
import pdb
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from dataset import load_cifar10
import torch.backends.cudnn as cudnn
import pickle
from temperature_scaling import ModelWithTemperature
import argparse
import random
import torchvision.models as models

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--train_size', default=0.9, type=float, help='train split ratio')
parser.add_argument('--valid_size', default=0.1, type=float, help='validation split ratio')
parser.add_argument('--epoch_num', default=100, type=int, help='Training epoch number')

args = parser.parse_args()

batch_size = args.batch_size
train_size = args.train_size #train valid set ratio
valid_size = args.valid_size
num_epoches = args.epoch_num

trainloader, validloader, testloader = load_cifar10(batch_size=batch_size, train_size=train_size, valid_size=valid_size)
net = models.resnet18(pretrained = False, num_classes = 10)

device = "cuda" if torch.cuda.is_available() else 'cpu'
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):

    train_loss = 0
    correct_train = 0
    num_train_samples = 0
    
    valid_loss = 0
    correct_valid = 0
    num_valid_samples = 0;
    
    data_features = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(targets).sum().item()
        num_train_samples += len(inputs)
        loss.backward()
        optimizer.step()

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    np_rng_state = np.random.get_state()
    scaled_model = ModelWithTemperature(net)
    scaled_model.set_temperature(validloader)
    torch.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    np.random.set_state(np_rng_state)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = scaled_model(inputs)
            loss = loss_func(outputs, targets)
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_valid += predicted.eq(targets).sum().item()
            num_valid_samples += len(inputs)             

    return train_loss/num_train_samples, correct_train/num_train_samples, valid_loss/num_valid_samples, correct_valid/num_valid_samples, scaled_model

def test(epoch, model):
    
    test_loss = 0
    correct = 0
    num_test_samples = 0

    data_features = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            num_test_samples += len(inputs)

    return test_loss/num_test_samples, correct/num_test_samples


def main():
    train_loss_history = []
    test_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    test_acc_history = []
    valid_acc_history = []

    for epoch in range(num_epoches):
        train_loss, train_acc, valid_loss, valid_acc, model = train(epoch)
        test_loss, test_acc = test(epoch, model)

        #scheduler.step()
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)
        test_acc_history.append(test_acc)
        print('Epoch: %d / %d, train acc: %.06f, train loss: %.06f, valid acc: %.06f, valid loss: %0.6f, test acc: %.06f, test loss: %.06f' % \
              (epoch, num_epoches, train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss))

    title_str = 'training # %d'%(50000*train_size)
    plt.figure()
    plt.plot(train_acc_history, 'r', label='training accuracy')
    plt.plot(valid_acc_history, 'b', label='valid accuracy')
    plt.plot(test_acc_history, 'g', label='testing accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(title_str)
    plt.savefig('ResNet18_CF10_accuracy_non_persistent_temperature_scaling_after_each_epoch_no_augmentation_no_lr_schedule.png')

    plt.figure()
    plt.plot(train_loss_history, 'r', label='training loss')
    plt.plot(valid_loss_history, 'b', label='valid loss')
    plt.plot(test_loss_history, 'g', label='testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title_str)
    plt.savefig('ResNet18_CF10_loss_non_persistent_temperature_scaling_after_each_epoch_no_augmentation_no_lr_schedule.png')
    
if __name__ == '__main__':
    main()

