# coding: utf-8
# Team : IIPLAB Medical Group
# Author：Bro Yuan
# Date ：2020/10/1 下午10:48
# Tool ：


import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
from utils import  Dataset
from utils import DataLoader
from torch.utils.data import Dataset as BaseDataset
from metrics import SegmentationMetric
from tqdm import  tqdm
from torch import nn
import cv2
import numpy as np
import os
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from Net import Net
from torchsummary import summary


def train(train_loader, model, criterion, optimizer, scheduler):
    train_loss_sum, train_fwiou_sum, n = 0.0, 0.0, 0
    model.train()

    for input, target in tqdm(train_loader,desc="training"):
        optimizer.zero_grad()

        input = input.type(torch.cuda.FloatTensor).cuda()
        output = model(input)
        target = target.cuda()
        target = torch.argmax(target, dim=1)
        target = target.cuda()
        # output = torch.argmax(output, dim=1)
        output = output.cuda()

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        #print('scheduler: {}'.format(scheduler.get_lr()[0]))



        for i in range(output.shape[0]):
            pre = output[i, :, :, :].argmax(axis=0).cpu()
            label = target[i, :, :].cpu()
            metric.addBatch(pre, label)
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            train_fwiou_sum += FWIoU

        train_loss_sum += loss.sum().item()
        # train_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]
    print('optim: {}'.format(optimizer.param_groups[0]['lr']))
    scheduler.step(train_fwiou_sum / n)
    return train_loss_sum / n, train_fwiou_sum / n


def validate(val_loader, model, criterion):
    val_loss_sum, val_fwiou_sum, n = 0.0, 0.0, 0
    model.eval()
    for input, target in tqdm(val_loader):
        input = input.cuda().type(torch.cuda.FloatTensor)
        target = target.cuda()
        target = torch.argmax(target, dim=1)
        output = model(input)
        output = torch.argmax(output, dim=1)
        loss = criterion(output, target.long())


        for i in range(output.shape[0]):
            pre = output[i, :, :, :].argmax(axis=0).cpu()
            label = target[i, :, :].cpu()
            metric.addBatch(pre, label)
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            val_fwiou_sum += FWIoU

        val_loss_sum += loss.sum().item()
        # val_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return val_loss_sum / n, val_fwiou_sum / n


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 16
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=CLASSES,
        activation=ACTIVATION,
    )

    # model.load_state_dict(torch.load('/home/public/deng/deng/torch/pt/GID_pre_DeeplabV3_RestNet/0.48869898404964923.pt'))

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    premodel_dict = torch.load('/home/public/deng/deng/torch/pt/GID_pre_Unet/0.8547977419618668.pt')
    # model.load_state_dict(premodel_dict)
    # del model
    #summary(model, (3,512,512), device='cpu')    model1 = Net(model).cuda()
    #summary(model1, (3, 512, 512), device='cpu')
    model1 =  smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=9,
        activation=ACTIVATION,
    )

    model1_dict = model1.state_dict()
    # encoder_dict = model.encoder.state_dict()
    del premodel_dict['segmentation_head.0.bias']
    del premodel_dict['segmentation_head.0.weight']
    model1_dict.update(premodel_dict)
    # model1_dict.update(model.decoder.state_dict())
    model1.load_state_dict(model1_dict)
    model1 = model1.cuda()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)



    train_dataset = Dataset(
        np.load('trainx.npy'),
        np.load('trainhot.npy'),
        preprocessing=preprocessing_fn,

    )



    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=16)

    # loss = smp.utils.losses.CrossEntropyLoss()
    # metrics = [
    #     smp.utils.metrics.IoU(threshold=0.5),
    # ]
    #
    # optimizer = torch.optim.Adam([
    #     dict(params=model.parameters(), lr=0.0001),
    # ])

    metric = SegmentationMetric(CLASSES)






#================================================
    criterion = smp.utils.losses.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([
        dict(params=model1.parameters(), lr=0.2),
    ])
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=4, verbose=True, min_lr=0.0001)
# freeze
    best_loss = 2020.0
    epochs = 10000
    best_FWIoU = 0




    for epoch in range(epochs):
        train_loss, train_FWIoU = train(train_loader, model1, criterion, optimizer, scheduler)

        # scheduler.step()

        print('Epoch %d: train loss %.4f, train FWIoU %.3f'
              % (epoch, train_loss, train_FWIoU))
        if epoch % 25 == 0:
            torch.save(model1.state_dict(), './pt/BiSai_Unet/epoch%s_%s.pt' % (str(epoch), str(train_FWIoU)))
        if best_FWIoU < train_FWIoU:
            best_FWIoU  = train_FWIoU
            torch.save(model1.state_dict(), './pt/BiSai_Unet/%s.pt'%str(best_FWIoU))

