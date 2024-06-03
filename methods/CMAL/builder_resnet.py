from __future__ import print_function
import torch
import torch.nn as nn

from methods.CMAL.highlight_images import *
from methods.CMAL.map_generate import *
from methods.CMAL.basic_conv import BasicConv

# Extract features from an input tensor taking first 8 layers of the model 
# and separates them into sequential models, forward applies layers to input tensors
# returns output tensors of 5th to 7th layers
class Features(nn.Module):
    def __init__(self, net_layers):

        super(Features, self).__init__()
        self.net_layers = net_layers

        if isinstance(net_layers[0],nn.Conv2d):
            self.net_layer_0 = nn.Sequential(net_layers[0])
            self.net_layer_1 = nn.Sequential(net_layers[1])
            self.net_layer_2 = nn.Sequential(net_layers[2])
            self.net_layer_3 = nn.Sequential(net_layers[3])
            self.net_layer_4 = nn.Sequential(*net_layers[4])
            self.net_layer_5 = nn.Sequential(*net_layers[5])
            self.net_layer_6 = nn.Sequential(*net_layers[6])
            self.net_layer_7 = nn.Sequential(*net_layers[7])

        elif isinstance(net_layers[0],nn.Sequential):  # EfficientNet
            self.net_layer_0 = nn.Sequential(net_layers[0], net_layers[1])  # Stem conv and batch norm
            self.net_layer_1 = nn.Sequential(*net_layers[2][0:6])           # MBConv blocks 0-5 (stage 1-2)
            self.net_layer_2 = nn.Sequential(*net_layers[2][6:12])          # MBConv blocks 6-11 (stage 3-4)
            self.net_layer_3 = nn.Sequential(*net_layers[2][12:22])         # MBConv blocks 12-21 (stage 5-6)
            self.net_layer_4 = nn.Sequential(*net_layers[2][22:30])         # MBConv blocks 22-29 (stage 7)
            self.net_layer_5 = nn.Sequential(*net_layers[2][30:38])         # MBConv blocks 30-37 (stage 8)
        else:
            raise ValueError("Unsupported model type")

    def forward(self, x):
        if isinstance(self.net_layers[0],nn.Conv2d):
            x = self.net_layer_0(x)
            x = self.net_layer_1(x)
            x = self.net_layer_2(x)
            x = self.net_layer_3(x)
            x = self.net_layer_4(x)
            x1 = self.net_layer_5(x)
            x2 = self.net_layer_6(x1)
            x3 = self.net_layer_7(x2)
        elif isinstance(self.net_layers[0],nn.Sequential):
            x = self.net_layer_0(x)  # Basic features
            x = self.net_layer_1(x)  # Early features
            x = self.net_layer_2(x)  # Mid-level features
            x1 = self.net_layer_3(x)  # Mid-high level features
            x2 = self.net_layer_4(x1) # Higher-level features (output 1)
            x3 = self.net_layer_5(x2) # More abstract features (output 2)
        
        # return as output the feature maps of the 5th, 6th and 7th layers
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class):
        super().__init__()
        self.Features = Features(net_layers)

        # reducing spatial dimension of the feature maps
        self.max_pool1 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=14, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=7, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

    # applies Features and then convolutional blocks and pooling
    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        # perform image classification
        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        # produce final output of the network
        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3


def cmal_train(inputs, targets, net, loss, optimizer, scheduler, train_loss=0, train_loss1=0, train_loss2=0, train_loss3=0, train_loss4=0, train_loss5=0, correct=0, total=0):    

    CELoss = loss
    netp = torch.nn.DataParallel(net, device_ids=[0])
    # Train the experts from deep to shallow with data augmentation by multiple steps
    # e3 trains the classifier_3
    optimizer.zero_grad()  # reset the gradients
    inputs3 = inputs
    output_1, output_2, output_3, _, map1, map2, map3 = netp(inputs3)  # distribute across multiple GPUs
    loss3 = CELoss(output_3, targets) * 1  # cross entropy loss function
    loss3.backward()
    optimizer.step()

    # generate attention maps
    p1 = net.state_dict()['classifier3.1.weight']
    p2 = net.state_dict()['classifier3.4.weight']
    att_map_3 = map_generate(map3, output_3, p1, p2)
    inputs3_att = attention_im(inputs, att_map_3)

    p1 = net.state_dict()['classifier2.1.weight']
    p2 = net.state_dict()['classifier2.4.weight']
    att_map_2 = map_generate(map2, output_2, p1, p2)
    inputs2_att = attention_im(inputs, att_map_2)

    p1 = net.state_dict()['classifier1.1.weight']
    p2 = net.state_dict()['classifier1.4.weight']
    att_map_1 = map_generate(map1, output_1, p1, p2)
    inputs1_att = attention_im(inputs, att_map_1)
    inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)

    # e2
    optimizer.zero_grad()  # reset the gradients of the parameters
    flag = torch.rand(1)  # determine which input data to use
    if flag < (1 / 3):
        inputs2 = inputs3_att
    elif (1 / 3) <= flag < (2 / 3):
        inputs2 = inputs1_att
    elif flag >= (2 / 3):
        inputs2 = inputs

    _, output_2, _, _, _, map2, _ = netp(inputs2)
    loss2 = CELoss(output_2, targets) * 1
    loss2.backward()
    optimizer.step()

    # e1
    optimizer.zero_grad()
    flag = torch.rand(1)
    if flag < (1 / 3):
        inputs1 = inputs3_att
    elif (1 / 3) <= flag < (2 / 3):
        inputs1 = inputs2_att
    elif flag >= (2 / 3):
        inputs1 = inputs

    output_1, _, _, _, map1, _, _ = netp(inputs1)
    loss1 = CELoss(output_1, targets) * 1
    loss1.backward()
    optimizer.step()


    # Train the experts and their concatenation with the overall attention region in one go
    optimizer.zero_grad()
    output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = netp(inputs_ATT)
    concat_loss_ATT = CELoss(output_1_ATT, targets)+\
                    CELoss(output_2_ATT, targets)+\
                    CELoss(output_3_ATT, targets)+\
                    CELoss(output_concat_ATT, targets) * 2
    concat_loss_ATT.backward()
    optimizer.step()


    # Train the concatenation of the experts with the raw input
    optimizer.zero_grad()
    _, _, _, output_concat, _, _, _ = netp(inputs)
    concat_loss = CELoss(output_concat, targets) * 2
    concat_loss.backward()
    optimizer.step()

    if scheduler is not None:
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = scheduler.get_lr()[nlr]

    # the maximum value in the tensor is the predicted label for input data
    _, predicted = torch.max(output_concat.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
    train_loss1 += loss1.item()
    train_loss2 += loss2.item()
    train_loss3 += loss3.item()
    train_loss4 += concat_loss_ATT.item()
    train_loss5 += concat_loss.item()

    return train_loss, float(correct), predicted

def save_model(self, net, device, path_name):
    net.cpu()
    torch.save(net, './' + path_name + '/model.pth')
    net.to(device)