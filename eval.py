
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt 
from tqdm import tqdm

import torch
import torch.nn.functional as TF
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from pytvision.transforms.aumentation import  ObjectImageMetadataTransform
from pytvision.transforms import transforms as mtrans

#sys.path.append('../')
from torchlib.transforms import functional as F
from torchlib.datasets.factory  import FactoryDataset
from torchlib.datasets.datasets import Dataset
from torchlib.datasets.fersynthetic  import SyntheticFaceDataset

from torchlib.attentionnet import AttentionNeuralNet, AttentionGMMNeuralNet
from torchlib.classnet import ClassNeuralNet

from aug import get_transforms_aug, get_transforms_det


# METRICS
import sklearn.metrics as metrics
from argparse import ArgumentParser


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--project',     metavar='DIR',  help='path to projects')
    parser.add_argument('--projectname', metavar='DIR',  help='name projects')
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--namedataset', metavar='S',    help='name to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--filename',    metavar='S',    help='name of the file output')
    parser.add_argument('--model',       metavar='S',    help='filename model')
    parser.add_argument('--breal', type=str, default='real', help='dataset is real or synthetic')
    parser.add_argument('--name-method', type=str, default='attnet', help='which neural network')
    parser.add_argument("--iteration", type=int, default='2000', help="iteration for synthetic images")

    return parser


def main(params=None):
    # This model has a lot of variabilty, so it needs a lot of parameters.
    # We use an arg parser to get all the arguments we need.
    # See above for the default values, definitions and information on the datatypes.
    parser = arg_parser()
    if params:
        args = parser.parse_args(params)
    else:
        args = parser.parse_args()

    # Configuration
    project         = args.project
    projectname     = args.projectname
    pathnamedataset = args.pathdataset  
    pathnamemodel   = args.model
    pathproject     = os.path.join( project, projectname )
    namedataset     = args.namedataset
    breal           = args.breal
    name_method      = args.name_method
    iteration       = args.iteration

    fname = args.name_method
    fnet = {
        'attnet': AttentionNeuralNet,
        'attgmmnet': AttentionGMMNeuralNet,
        'classnet': ClassNeuralNet,

    }
    
    no_cuda=False
    parallel=False
    gpu=0
    seed=1
    brepresentation=True
    bclassification_test=True
    brecover_test=False
    
    imagesize=64
    kfold = 5
    nactores = 10
    idenselect = np.arange(nactores) + kfold * nactores

    # experiments
    experiments = [ 
        { 'name': namedataset,        'subset': FactoryDataset.training,   'status': breal },
        { 'name': namedataset,        'subset': FactoryDataset.validation, 'status': breal }
        ]
    
    # representation datasets
    if brepresentation: 
    
        # create an instance of a model to modify
        print('>> Load model ...')
        network = fnet[fname](
            patchproject=project,
            nameproject=projectname,
            no_cuda=no_cuda,
            parallel=parallel,
            seed=seed,
            gpu=gpu,
            )

        cudnn.benchmark = True

        # load model
        if network.load( pathnamemodel ) is not True:
            print('>>Error!!! load model')
            assert(False)


        size_input = network.size_input
        # Perform the experiments
        for i, experiment in enumerate(experiments):
            
            name_dataset = experiment['name']
            subset = experiment['subset']
            breal = experiment['status']
            dataset = []
                        
            # load dataset 
            if breal == 'real':
                
                # real dataset 
                dataset = Dataset(    
                    data=FactoryDataset.factory(
                        pathname=pathnamedataset, 
                        name=namedataset, 
                        subset=subset, 
                        idenselect=idenselect,
                        download=True 
                    ),
                    num_channels=3,
                    transform=get_transforms_det( imagesize ),
                    )
            
            else:
            
                # synthetic dataset 
                dataset = SyntheticFaceDataset(
                    data=FactoryDataset.factory(
                        pathname=pathnamedataset, 
                        name=namedataset, 
                        subset=subset, 
                        idenselect=idenselect,
                        download=True 
                        ),
                    pathnameback='~/.datasets/coco', 
                    ext='jpg',
                    count=iteration,
                    num_channels=3,
                    iluminate=True, angle=45, translation=0.3, warp=0.2, factor=0.2,
                    transform_data=get_transforms_aug( imagesize ),
                    transform_image=get_transforms_det( imagesize ),
                    )

            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=10 )
            
            print("\ndataset:", breal)
            print("Subset:", subset)
            print("Classes", dataloader.dataset.data.classes)
            print("size of data:", len(dataset))
            print("num of batches", len(dataloader))

            if name_method == 'attgmmnet':
                # representation
                Y_labs, Y_lab_hats, Zs = network.representation(dataloader, breal)
                print(Y_lab_hats.shape, Zs.shape, Y_labs.shape)

                reppathname = os.path.join(pathproject, 'rep_{}_{}_{}.pth'.format(namedataset, subset,
                                                                                     breal))
                torch.save({'Yh': Y_lab_hats, 'Z': Zs, 'Y': Y_labs}, reppathname)
                print('save representation ...', reppathname)

            else:
                Y_labs, Y_lab_hats= network.representation( dataloader, breal )
                print("Y_lab_hats shape: {}, y_labs shape: {}".format(Y_lab_hats.shape, Y_labs.shape))

                reppathname = os.path.join( pathproject, 'rep_{}_{}_{}.pth'.format(namedataset, subset, breal ) )
                torch.save( { 'Yh':Y_lab_hats, 'Y':Y_labs }, reppathname )
                print( 'save representation ...', reppathname )
            

    
    if bclassification_test:

        tuplas=[]
        print('|Num\t|Acc\t|Prec\t|Rec\t|F1\t|Set\t|Type\t|Accuracy_type\t')
        for  i, experiment in enumerate(experiments):

            name_dataset = experiment['name']
            subset = experiment['subset']
            breal = experiment['status']
            real = breal

            rep_pathname = os.path.join( pathproject, 'rep_{}_{}_{}.pth'.format(
                namedataset, subset, breal) )

            data_emb = torch.load(rep_pathname)
            Yto = data_emb['Y']
            Yho = data_emb['Yh']

            yhat = np.argmax( Yho, axis=1 )
            y    = Yto

            acc = metrics.accuracy_score(y, yhat)
            precision = metrics.precision_score(y, yhat, average='macro')
            recall = metrics.recall_score(y, yhat, average='macro')
            f1_score = 2*precision*recall/(precision+recall)

            
            print( '|{}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{}\t|{}\t|{}\t'.format(
                i, 
                acc, precision, recall, f1_score,
                subset, real, 'topk'
            ))

            print(f'save y and yhat to {real}_{subset}_y.npz')
            np.savez(os.path.join(pathproject, f'{real}_{subset}_y.npz'), name1=yhat, name2=y)

            if name_method == 'attgmmnet':
                y = data_emb['Y']
                x = data_emb['Z']

                classes = np.unique(y)
                numclasses = len(classes)

                num = x.shape[0]
                dim = x.shape[1]

                Xmu = np.zeros((num, numclasses))
                yh = np.zeros((num,)).astype(int)

                ## Ecuation
                for idx, c in enumerate(classes):
                    index = y == int(c)
                    if index.sum() == 0: continue
                    xc = x[index, ...]
                    muc = xc.mean(axis=0)
                    Xmu[:, idx] = np.sum((x - muc) ** 2, axis=1)
                    yh[index] = idx

                yhat = np.argmin(Xmu, 1).astype(int)

                acc = metrics.accuracy_score(y, yhat)
                precision = metrics.precision_score(y, yhat, average='macro')
                recall = metrics.recall_score(y, yhat, average='macro')
                f1_score = 2 * precision * recall / (precision + recall)

                print('|{}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{}\t|{}\t|{}\t'.format(
                    i,
                    acc, precision, recall, f1_score,
                    subset, real, 'gmm'
                ))


            #|Name|Dataset|Cls|Acc| ...
            tupla = { 
                'Name':projectname,  
                'Dataset': '{}({})_{}'.format(  name_dataset,  subset, real ),
                'Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'F1 score': f1_score,        
            }
            tuplas.append(tupla)

        # save
        df = pd.DataFrame(tuplas)
        df.to_csv( os.path.join( pathproject, 'experiments_cls.csv' ) , index=False, encoding='utf-8')
        print('save experiments class ...')
        print()
    print('DONE!!!')
        

if __name__ == '__main__':
    main()