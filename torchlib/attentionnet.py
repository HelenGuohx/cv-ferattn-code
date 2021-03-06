
import os

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
from pytvision import utils as pytutils

#----------------------------------------------------------------------------------------------
# Neural Net for Attention

class AttentionNeuralNetAbstract(NeuralNetAbstract):
    """
    Attention Neural Net Abstract
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

        super(AttentionNeuralNetAbstract, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        self.view_freq = view_freq


    # this function leads to the call `createLoss`, which needs our parameters
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
        alpha=2,
        beta=2
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

        # this function leads to the call `createLoss`, which needs our parameters
        super(AttentionNeuralNetAbstract, self).create(
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
            alpha=alpha,
            beta=beta
        )

        self.size_input = size_input
        self.backbone = backbone
        self.num_filters = num_filters

        self.accuracy = nloss.Accuracy()
        self.topk     = nloss.TopkAccuracy() #topk=(1,3,6)
        self.gmm      = nloss.GMMAccuracy( classes=num_classes, cuda=self.cuda  )
        self.dice     = nloss.Dice()

        # Set the graphic visualization
        # self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )


    def representation( self, dataloader, breal=True ):
        Y_labs = []
        Y_lab_hats = []
        Zs = []
        self.net.eval()
        with torch.no_grad():
            for i_batch, sample in enumerate( tqdm(dataloader) ):

                if breal:
                    x_img, y_lab = sample['image'], sample['label'].argmax(dim=1)
                else:
                    x_org, x_img, y_mask, y_lab = sample
                    y_lab=y_lab[:,0]

                x_img = x_img.cuda() if self.cuda else x_img
                z, y_lab_hat, _,_,_,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                Zs.append(z.data.cpu())

        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        Zs = np.concatenate( Zs, axis=0 )
        return Y_labs, Y_lab_hats, Zs

    def __call__(self, image ):
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image
            z, y_lab_hat, att, theta, att_t, fmap, srf = self.net(x)
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
        return z, y_lab_hat, att, theta, att_t, fmap, srf

    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained, **kwargs):
        """
        Create model, private method
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
        kw = {'dim': num_output_channels, 'num_classes': self.num_classes, 'num_channels': num_input_channels, 'pretrained': pretrained,
              'num_filters': num_filters}
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
                'num_filters':self.num_filters,
            },
            is_best,
            self.pathmodels,
            filename
            )

    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = torch.load( pathnamemodel ) if self.cuda else torch.load( pathnamemodel, map_location=lambda storage, loc: storage )
                self.num_classes = checkpoint['num_classes']
                self._create_model(checkpoint['arch'],
                                   checkpoint['num_output_channels'],
                                   checkpoint['num_input_channels'],
                                   False,
                                   num_filters=checkpoint['num_filters'],
                                    )
                self.size_input = checkpoint['imsize']
                self.net.load_state_dict( checkpoint['state_dict'] )
                print("=> loaded checkpoint for {} arch!".format(checkpoint['arch']))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))
        return bload



class AttentionNeuralNet(AttentionNeuralNetAbstract):
    """
    Attention Neural Net
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
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
        super(AttentionNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )


    # The beta parameters in this line is the result of an earlier refactor, and may safely be ignored.
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
        breal=True,
        alpha=2,
        beta=2
        ):
        """
        Create
        Args:
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) :
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """
        super(AttentionNeuralNet, self).create(
            arch,
            num_output_channels,
            num_input_channels,
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,
            backbone,
            num_filters,
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_bce', 'loss_att' ], [ 'topk'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_bce', 'loss_att' ], [ 'topk'], self.plotter )
        self.breal = breal


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

            # real dataset
            if self.breal == 'real':
                x_img, y_lab = sample['image'], sample['label']
                y_lab = y_lab.argmax(dim=1)

                if self.cuda:
                    x_img = x_img.cuda()
                    y_lab = y_lab.cuda()

                x_org = x_img.clone().detach()

            else:
                # synthetic dataset
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
            y_lab_hat, att, fmap, srf = self.net(x_img)

            # measure accuracy and record loss
            loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )
            loss_att  = self.criterion_att( x_org, att )
            loss      = loss_bce + loss_att
            topk      = self.topk( y_lab_hat, y_lab.long() )

            batch_size = x_img.shape[0]

            # optimizer
            self.optimizer.zero_grad()
            (loss*batch_size).backward()
            self.optimizer.step()

            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                {'topk': topk[0][0].cpu() },
                batch_size,
                )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    # The beta parameters in this line is the result of an earlier refactor, and may safely be ignored.
    def evaluate(self, data_loader, epoch=0, alpha=2, beta=2):
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
                y_lab_hat, att, fmap, srf  = self.net( x_img )

                # measure accuracy and record loss
                loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
                loss_att  = self.criterion_att( x_org, att )
                loss      = loss_bce + loss_att
                topk      = self.topk( y_lab_hat, y_lab.long() )

                batch_size = x_img.shape[0]

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                    {'loss': loss.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                    {'topk': topk[0][0].cpu() },
                    batch_size,
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader),
                        batch_time,
                        bplotter=False,
                        bavg=True,
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader),
            batch_time,
            bplotter=True,
            bavg=True,
            bsummary=True,
            )
        return acc

    def representation( self, dataloader, breal):
        """
        :param dataloader:
        :param breal: 'real' or 'synthetic'
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
                y_lab_hat, _,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())

        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        return Y_labs, Y_lab_hats


    def __call__(self, image ):
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image
            y_lab_hat, att, fmap, srf = self.net(x)
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
        return y_lab_hat, att, fmap, srf

    # this is used for inheritance, so even though our beta parameters aren't used, they need to be here.
    def _create_loss(self, loss, alpha=2, beta=2):

        # create loss
        if loss == 'attloss':
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            self.criterion_att = nloss.Attloss()
        else:
            assert(False)

        self.s_loss = loss




class AttentionSTNNeuralNet(AttentionNeuralNet):
    """
    Attention Neural Net with STN
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
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
        super(AttentionSTNNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )






class AttentionGMMNeuralNet(AttentionNeuralNetAbstract):
    """
    Attention Neural Net and GMM representation
    Args:
        -patchproject (str): path project
        -nameproject (str):  name project
        -no_cuda (bool): system cuda (default is True)
        -parallel (bool)
        -seed (int)
        -print_freq (int)
        -gpu (int)
        -view_freq (in epochs)
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
        super(AttentionGMMNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, view_freq  )



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
        breal='real',
        num_filters=32,
        alpha=2,
        beta=2
        ):
        """
        Create
        Args:
            -arch (string): architecture
            -num_output_channels,
            -num_input_channels,
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) :
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            -
        """
        super(AttentionGMMNeuralNet, self).create(
            arch,
            num_output_channels,
            num_input_channels,
            loss,
            lr,
            optimizer,
            lrsch,
            momentum,
            weight_decay,
            pretrained,
            size_input,
            num_classes,
            backbone,
            num_filters,
        )

        self.logger_train = Logger( 'Train', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss', 'loss_gmm', 'loss_bce', 'loss_att' ], [ 'topk', 'gmm'], self.plotter )
        self.breal = breal


    def training(self, data_loader, epoch=0):

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if self.breal == 'real':
                x_img, y_lab = sample["image"], sample["label"]
                y_lab = y_lab.argmax(dim=1)

                if self.cuda:
                    x_img = x_img.cuda()
                    y_lab = y_lab.cuda()
                x_org = x_img.clone().detach()
            else:
                x_org, x_img, y_mask, meta = sample

                y_lab = meta[:, 0]
                y_theta = meta[:, 1:].view(-1, 2, 3)

                if self.cuda:
                    x_org = x_org.cuda()
                    x_img = x_img.cuda()
                    y_mask = y_mask.cuda()
                    y_lab = y_lab.cuda()
                    y_theta = y_theta.cuda()

            batch_size = x_img.shape[0]


            # fit (forward)
            z, y_lab_hat, att, _, _ = self.net( x_img)

            # measure accuracy and record loss
            loss_bce  = self.criterion_bce(  y_lab_hat, y_lab.long() )
            loss_gmm  = self.criterion_gmm(  z, y_lab )
            loss_att  = self.criterion_att(  x_org, att )
            loss      = loss_bce + loss_gmm  + loss_att + + 0.0*z.norm(2)

            topk      = self.topk( y_lab_hat, y_lab.long() )
            gmm       = self.gmm( z, y_lab )

            # optimizer
            self.optimizer.zero_grad()
            (loss).backward() #batch_size
            self.optimizer.step()

            # update
            self.logger_train.update(
                {'loss': loss.cpu().item(), 'loss_gmm': loss_gmm.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },
                batch_size,
                )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):

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

                    y_lab = meta[:, 0]
                    y_theta = meta[:, 1:].view(-1, 2, 3)

                    if self.cuda:
                        x_org = x_org.cuda()
                        x_img = x_img.cuda()
                        y_mask = y_mask.cuda()
                        y_lab = y_lab.cuda()
                        y_theta = y_theta.cuda()

                batch_size = x_img.shape[0]

                # fit (forward)
                z, y_lab_hat, att, fmap, srf  = self.net( x_img )

                # measure accuracy and record loss
                loss_bce  = self.criterion_bce( y_lab_hat, y_lab.long() )
                loss_gmm  = self.criterion_gmm( z, y_lab )
                loss_att  = self.criterion_att( x_org, att  )
                loss      = loss_bce + loss_gmm + loss_att + 0.0*z.norm(2)

                topk      = self.topk( y_lab_hat, y_lab.long() )
                gmm       = self.gmm( z, y_lab )

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                    {'loss': loss.cpu().item(), 'loss_gmm': loss_gmm.cpu().item(), 'loss_bce': loss_bce.cpu().item(), 'loss_att':loss_att.cpu().item() },
                    {'topk': topk[0][0].cpu(), 'gmm': gmm.cpu().item() },
                    batch_size,
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader),
                        batch_time,
                        bplotter=False,
                        bavg=True,
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['topk'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader),
            batch_time,
            bplotter=True,
            bavg=True,
            bsummary=True,
            )

        return acc

    def representation( self, dataloader, breal ):
        Y_labs = []
        Y_lab_hats = []
        Zs = []
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

                z, y_lab_hat, _,_,_ = self.net( x_img )
                Y_labs.append(y_lab)
                Y_lab_hats.append(y_lab_hat.data.cpu())
                Zs.append(z.data.cpu())

        Y_labs = np.concatenate( Y_labs, axis=0 )
        Y_lab_hats = np.concatenate( Y_lab_hats, axis=0 )
        Zs = np.concatenate( Zs, axis=0 )
        return Y_labs, Y_lab_hats, Zs


    def __call__(self, image ):
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image
            z, y_lab_hat, att, fmap, srf = self.net(x)
            y_lab_hat = F.softmax( y_lab_hat, dim=1 )
        return z, y_lab_hat, att, fmap, srf

    # pass the beta parameters to the loss function that uses them
    def _create_loss(self, loss, alpha=2, beta=2):

        # create loss
        if loss == 'attloss':
            # claasification loss
            self.criterion_bce = nn.CrossEntropyLoss().cuda()
            # attention loss
            self.criterion_att = nloss.Attloss()
            # representation loss
            self.criterion_gmm = nloss.DGMMLoss( self.num_classes, cuda=self.cuda, alpha=alpha, beta=beta)
        else:
            assert(False)

        self.s_loss = loss












