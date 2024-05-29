import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.optim as optim

from train import Trainer
from test import Tester

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_methods')))
from methods.CMAL.builder_resnet import Network_Wrapper, Features
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root = "/home/zazza/Documents/ML/fgvc-aircraft/data/images" # change this to your data directory

trainset = torchvision.datasets.ImageFolder(root=root+'/train', transform=transform.Compose([
        transform.Resize((550, 550)),
        transform.RandomCrop(448, padding=8),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

trainset, valset = torch.utils.data.random_split(trainset, [5000, 1667])

testset = torchvision.datasets.ImageFolder(root=root+'/test', transform=transform.Compose([
        transform.Resize((550, 550)),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=10,
                                            shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=4)

data_loaders = {
    "train_loader": trainloader,
    "val_loader": valloader,
    "test_loader": testloader
}  

model = torchvision.models.resnet50()
state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
model.load_state_dict(state_dict)

net_layers = list(model.children())
net_layers = net_layers[0:8]
model = Network_Wrapper(net_layers, 100)

CELoss = nn.CrossEntropyLoss()
optimizer = optim.SGD([
    {'params': model.classifier_concat.parameters(), 'lr': 0.002},
    {'params': model.conv_block1.parameters(), 'lr': 0.002},
    {'params': model.classifier1.parameters(), 'lr': 0.002},
    {'params': model.conv_block2.parameters(), 'lr': 0.002},
    {'params': model.classifier2.parameters(), 'lr': 0.002},
    {'params': model.conv_block3.parameters(), 'lr': 0.002},
    {'params': model.classifier3.parameters(), 'lr': 0.002},
    {'params': model.Features.parameters(), 'lr': 0.0002}

],
    momentum=0.9, weight_decay=5e-4)

scheduler = StepLR(optimizer, step_size=1, gamma=0.01)

training = Trainer(
    data_loaders=data_loaders, 
    dataset_name = "FGVC_Aircraft",
    model=model,
    optimizer=optimizer,
    loss_fn=CELoss,
    device=device,
    seed=42,
    exp_path="/home/zazza/Documents/ML/fgvc-aircraft/",
    exp_name="test_1",
    use_early_stopping=True,
    scheduler=scheduler, 
    cmal=True)

training.main(epochs=3, log_interval = 100)