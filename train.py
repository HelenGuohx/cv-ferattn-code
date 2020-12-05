
import os
import numpy as np

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from torchlib.datasets.fersynthetic  import SecuencialSyntheticFaceDataset
from torchlib.datasets.factory import FactoryDataset
from torchlib.attentionnet import AttentionNeuralNet, AttentionSTNNeuralNet, AttentionGMMNeuralNet, AttentionGMMSTNNeuralNet

import datetime
from argparse import ArgumentParser
from aug import get_transforms_aug, get_transforms_det

def arg_parser():
    """Arg parser"""
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--databack', metavar='DIR',
                        help='path to background dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--kfold', default=0, type=int, metavar='N',
                        help='k fold')
    parser.add_argument('--nactor', default=0, type=int, metavar='N',
                        help='number of the actores')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--snapshot', '-sh', default=10, type=int, metavar='N',
                        help='snapshot (default: 10)')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='NAME',
                        help='name to latest checkpoint (default: none)')
    parser.add_argument('--arch', default='simplenet', type=str,
                        help='architecture')
    parser.add_argument('--finetuning', action='store_true', default=False,
                        help='Finetuning')
    parser.add_argument('--loss', default='cross', type=str,
                        help='loss function')
    parser.add_argument('--opt', default='adam', type=str,
                        help='optimize function')
    parser.add_argument('--scheduler', default='fixed', type=str,
                        help='scheduler function for learning rate')
    parser.add_argument('--image-size', default=388, type=int, metavar='N',
                        help='image size')
    parser.add_argument('--channels', default=1, type=int, metavar='N',
                        help='input channel (default: 1)')
    parser.add_argument('--dim', default=64, type=int, metavar='N',
                        help='code size (default: 64)')
    parser.add_argument('--num-classes', '-c', default=10, type=int, metavar='N',
                        help='num classes (default: 10)')
    parser.add_argument('--name-dataset', default='mnist', type=str,
                        help='name dataset')
    parser.add_argument('--name-method', default='attnet', type=str,
                        help='name method')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Parallel')
    parser.add_argument('--balance', action='store_true', default=False,
                        help='balance data')
    parser.add_argument('--backbone', default='preactresnet', type=str,
                        help='backbone for classification and representation')
    parser.add_argument('--trainiteration', default=1000, type=int, metavar='N',
                        help='train iteration')
    parser.add_argument('--testiteration', default=100, type=int, metavar='N',
                        help='train iteration')

    return parser



def main(params=None):

    # parameters
    parser = arg_parser()
    if params is not None:
        args = parser.parse_args(params.split())
    else:
        args = parser.parse_args()
    imsize = args.image_size
    parallel=args.parallel
    num_classes=args.num_classes
    num_channels=args.channels
    dim=args.dim
    view_freq=1
    trainiteration = args.trainiteration
    testiteration = args.testiteration

    fname = args.name_method
    fnet = {
        'attnet':AttentionNeuralNet,
        'attstnnet':AttentionSTNNeuralNet,
        'attgmmnet':AttentionGMMNeuralNet,
        'attgmmstnnet':AttentionGMMSTNNeuralNet
        }

    network = fnet[fname](
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        parallel=parallel,
        seed=args.seed,
        print_freq=args.print_freq,
        gpu=args.gpu,
        view_freq=view_freq,
        )

    network.create(
        arch=args.arch,
        num_output_channels=dim,
        num_input_channels=num_channels,
        loss=args.loss,
        lr=args.lr,
        momentum=args.momentum,
        optimizer=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning,
        size_input=imsize,
        num_classes=num_classes
        )

    # resume
    network.resume( os.path.join(network.pathmodels, args.resume ) )
    cudnn.benchmark = True

    kfold=args.kfold
    nactores=args.nactor
    idenselect = np.arange(nactores) + kfold*nactores

    # datasets
    # training dataset
    # SyntheticFaceDataset, SecuencialSyntheticFaceDataset
    data = FactoryDataset.factory(
        pathname=args.data,
        name=args.name_dataset,
        subset=FactoryDataset.training,
        idenselect=idenselect,
        download=True)

    train_data = SecuencialSyntheticFaceDataset(
        data=FactoryDataset.factory(
            pathname=args.data,
            name=args.name_dataset,
            subset=FactoryDataset.training,
            idenselect=idenselect,
            download=True
            ),
        pathnameback=args.databack,
        ext='jpg',
        count=trainiteration,
        num_channels=num_channels,
        iluminate=True, angle=30, translation=0.2, warp=0.1, factor=0.2,
        transform_data=get_transforms_aug( imsize ),
        transform_image=get_transforms_det( imsize ),
        )


    num_train = len(train_data)
    if args.balance:
        labels, counts = np.unique(train_data.labels, return_counts=True)
        weights = 1/(counts/counts.sum())
        samples_weights = np.array([ weights[x] for x in train_data.labels ])
        sampler = WeightedRandomSampler( weights=samples_weights, num_samples=len(samples_weights) , replacement=True )
    else:
        sampler = SubsetRandomSampler(np.random.permutation( num_train ) )


    train_loader = DataLoader(train_data, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=network.cuda, drop_last=True, sampler=sampler ) #shuffle=True,


    # validate dataset
    # SyntheticFaceDataset, SecuencialSyntheticFaceDataset
    val_data = SecuencialSyntheticFaceDataset(
        data=FactoryDataset.factory(
            pathname=args.data,
            name=args.name_dataset,
            idenselect=idenselect,
            subset=FactoryDataset.validation,
            download=True
            ),
        pathnameback=args.databack,
        ext='jpg',
        count=testiteration,
        num_channels=num_channels,
        iluminate=True, angle=30, translation=0.2, warp=0.1, factor=0.2,
        transform_data=get_transforms_aug( imsize ),
        transform_image=get_transforms_det( imsize ),
        )

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=network.cuda, drop_last=False)

    # print neural net class
    print('SEG-Torch: {}'.format(datetime.datetime.now()) )
    print(network)

    # training neural net
    network.fit( train_loader, val_loader, args.epochs, args.snapshot )

    print("Optimization Finished!")
    print("DONE!!!")



if __name__ == '__main__':
    DATABACK = '~/.datasets/coco'
    DATA = '~/.datasets'
    NAMEDATASET = 'ck'
    PROJECT = '../out/attnet'
    EPOCHS = 500
    TRAINITERATION = 288000
    TESTITERATION = 2880
    BATCHSIZE = 32  # 32, 64, 128, 160, 200, 240
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.5
    PRINT_FREQ = 100
    WORKERS = 4
    RESUME = 'model_best.pth.tar'  # chk000000, model_best
    GPU = 0
    NAMEMETHOD = 'attnet'  # attnet, attstnnet, attgmmnet, attgmmstnnet
    ARCH = 'ferattention'  # ferattention, ferattentiongmm, ferattentionstn
    LOSS = 'attloss'
    OPT = 'adam'
    SCHEDULER = 'fixed'
    NUMCLASS = 8  # 6, 7, 8
    NUMCHANNELS = 3
    DIM = 32
    SNAPSHOT = 10
    IMAGESIZE = 64
    KFOLD = 0
    NACTOR = 10
    BACKBONE = 'preactresnet'  # preactresnet, resnet, cvgg

    EXP_NAME = 'feratt_'+ NAMEMETHOD + \
    '_' + ARCH + \
    '_' + LOSS + \
    '_' + OPT + \
    '_' + NAMEDATASET + \
    '_dim'+ str(DIM) + \
    '_bb'+ BACKBONE + \
    '_fold'+ str(KFOLD) + '_000'

    CUDA_VISIBLE_DEVICES = 0

    args = f"{DATA} \
--databack={DATABACK} \
--name-dataset={NAMEDATASET} \
--project={PROJECT} \
--name={EXP_NAME} \
--epochs={EPOCHS} \
--trainiteration={TRAINITERATION} \
--testiteration={TESTITERATION} \
--kfold={KFOLD} \
--nactor={NACTOR} \
--batch-size={BATCHSIZE} \
--learning-rate={LEARNING_RATE} \
--momentum={MOMENTUM} \
--image-size={IMAGESIZE} \
--channels={NUMCHANNELS} \
--dim={DIM} \
--num-classes={NUMCLASS} \
--print-freq={PRINT_FREQ} \
--snapshot={SNAPSHOT} \
--workers={WORKERS} \
--resume={RESUME} \
--gpu={GPU} \
--loss={LOSS} \
--opt={OPT} \
--scheduler={SCHEDULER} \
--name-method={NAMEMETHOD} \
--arch={ARCH} \
--finetuning "

    main(params=args)