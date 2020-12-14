
import os
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm

from . import models as nnmodels
from . import netlosses as nloss

from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import graphic as gph
from pytvision import netlearningrate
from pytvision import utils as pytutils


class ClassNeuralNet(NeuralNetAbstract):
    """
    classification Neural Net like preactresnet

    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
            -seed (int)
            -print_freq (int)
            -gpu (int)
            -view_freq (in epochs)
        """

        super(ClassNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu)
        self.view_freq = view_freq

    def create(self,
        arch,
        num_output_channels,
        num_input_channels,
        loss,
        lr,
        optimizer,
        lrsch,
        momentum=0.9,
        weight_decay=5e-4,
        pretrained=False,
        size_input=388,
        num_classes=8,
        backbone='preactresnet',
        num_filters=32,
        breal='real'
        ):
        """
        Create
            -arch (string): architecture
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) :
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
        """
        cfg_opt={ 'momentum':momentum, 'weight_decay':weight_decay }
        #cfg_scheduler={ 'step_size':100, 'gamma':0.1  }
        cfg_scheduler={ 'mode':'min', 'patience':10  }
        cfg_model = {'num_filters': num_filters}

        self.num_classes = num_classes

        super(ClassNeuralNet, self).create(
            arch,
            num_output_channels,
            num_input_channels,
            loss,
            lr,
            optimizer,
            lrsch,
            pretrained,
            cfg_opt=cfg_opt,
            cfg_scheduler=cfg_scheduler,
            cfg_model=cfg_model,
        )
        self.size_input = size_input
        self.backbone = backbone
        self.num_filters = num_filters

        self.topk        = nloss.TopkAccuracy()

        self.logger_train = Logger( 'Train', ['loss', 'loss_bce' ], [ 'topk'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_bce' ], [ 'topk'], self.plotter )
        self.breal = breal

        # Set the graphic visualization
        # self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )

    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained, **kwargs):
        """
        Create model
            -arch (string): select architecture
            -num_classes (int)
            -num_channels (int)
            -pretrained (bool)
        """

        self.net = None

        #--------------------------------------------------------------------------------------------
        # select architecture
        #--------------------------------------------------------------------------------------------
        num_filters = kwargs.get("num_filters", 32)
        # num_classes=1000, num_channels=3, initial_channels=64 for preactresnet

        kw = {'num_classes': self.num_classes, 'num_channels': num_input_channels, 'pretrained': pretrained}
        print("kw", kw)
        self.net = nnmodels.__dict__[arch](**kw)

        self.s_arch = arch
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        if self.cuda:
            self.net.cuda()
        if self.parallel and self.cuda:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))


    def save(self, epoch, prec, is_best=False, filename='checkpoint.pth.tar'):
        """
        Save model
        """
        print('>> save model epoch {} ({}) in {}'.format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        pytutils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': self.s_arch,
                'imsize': self.size_input,
                'num_output_channels': self.num_output_channels,
                'num_input_channels': self.num_input_channels,
                'num_classes': self.num_classes,
                'state_dict': net.state_dict(),
                'prec': prec,
                'optimizer' : self.optimizer.state_dict(),
            },
            is_best,
            self.pathmodels,
            filename
            )

    def load(self, pathnamemodel):
        """
        load model from pretrained model
        :param pathnamemodel: model path
        :return:
        """
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                self.num_classes = checkpoint['num_classes']
                self._create_model(checkpoint['arch'],
                                   checkpoint['num_output_channels'],
                                   checkpoint['num_input_channels'],
                                   False
                                    )
                self.size_input = checkpoint['imsize']
                self.net.load_state_dict( checkpoint['state_dict'] )
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))
        return bload

    def training(self, data_loader, epoch=0):
        #reset logger
        self.logger_train.reset()
        data_time  = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            # if dataset is real
            if self.breal == 'real':
                x_img, y_lab = sample['image'], sample['label']
                y_lab = y_lab.argmax(dim=1)

                if self.cuda:
                    x_img = x_img.cuda()
                    y_lab = y_lab.cuda()

                x_org = x_img.clone().detach()

            else:
                # if dataset is synthetic
                x_org, x_img, y_mask, meta = sample

                y_lab = meta[:, 0]
                y_theta = meta[:, 1:].view(-1, 2, 3)

                if self.cuda:
                    x_org = x_org.cuda()
                    x_img = x_img.cuda()
                    y_mask = y_mask.cuda()
                    y_lab = y_lab.cuda()
                    y_theta = y_theta.cuda()

            # fit (forward)
            y_lab_hat = self.net(x_img)

            # calculate classification loss
            loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )
            loss      = loss_bce
            # accuracy of choosing top k predicted classes
            topk      = self.topk( y_lab_hat, y_lab.long() )

            batch_size = x_img.shape[0]

            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()

            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item() },
                {'topk': topk[0][0].cpu() },
                batch_size,
                )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )

    def evaluate(self, data_loader, epoch=0):
        """
        evaluate on validation dataset
        :param data_loader: which data_loader to use
        :param epoch: current epoch
        :return:
            acc: average accuracy on data_loader
        """
        # reset loader
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(data_loader):

                # get data (image, label)
                if self.breal == 'real':
                    x_img, y_lab = sample["image"], sample["label"]
                    y_lab = y_lab.argmax(dim=1)

                    if self.cuda:
                        x_img = x_img.cuda()
                        y_lab = y_lab.cuda()
                    x_org = x_img.clone().detach()
                else:
                    x_org, x_img, y_mask, meta = sample

                    y_lab = meta[:,0]
                    y_theta   = meta[:,1:].view(-1, 2, 3)

                    if self.cuda:
                        x_org   = x_org.cuda()
                        x_img   = x_img.cuda()
                        y_mask  = y_mask.cuda()
                        y_lab   = y_lab.cuda()
                        y_theta = y_theta.cuda()


                # fit (forward)
                # print("x_img size", x_img.size())
                y_lab_hat  = self.net( x_img )

                # measure accuracy and record loss
                loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
                loss      = loss_bce
                topk      = self.topk( y_lab_hat, y_lab.long() )

                batch_size = x_img.shape[0]

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                    {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item() },
                    {'topk': topk[0][0].cpu() },
                    batch_size,
                    )

                # print the result in certain print frequency when iterating batches
                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader),
                        batch_time,
                        bplotter=False,
                        bavg=True,
                        bsummary=False,
                        )

        #save validation loss and accuracy
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        # print the average loss and accuracy after one iteration
        self.logger_val.logger(
            epoch, epoch, i, len(data_loader),
            batch_time,
            bplotter=True,
            bavg=True,
            bsummary=True,
            )

        return acc

    def representation( self, dataloader, breal='real'):
        """
        :param dataloader:
        :param breal:'real' or 'synthetic'
        :return:
            Y_labs: true labels
            Y_lab_hats: predicted labels
        """
        Y_labs = []
        Y_lab_hats = []
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal == 'real':
                    x_img, y_lab = sample['image'], sample['label']
                    y_lab = y_lab.argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                if self.cuda:
                    x_img = x_img.cuda()
                y_lab_hat = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())

        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        return Y_labs, Y_lab_hats


    def __call__(self, image ):
        # when calling the class, switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image
            y_lab_hat, att, fmap, srf = self.net(x)
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
        return y_lab_hat, att, fmap, srf

    def _create_loss(self, loss):
        # private method
        # create cross entropy loss
        self.criterion_bce = nn.CrossEntropyLoss().cuda()
        self.s_loss = loss



