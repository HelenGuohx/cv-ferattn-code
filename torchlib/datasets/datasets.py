
import os
import numpy as np
import random
from collections import namedtuple

import torch
from pytvision.datasets import utility 
from pytvision.datasets.imageutl import imageProvide
from pytvision.transforms.aumentation import ObjectImageAndLabelTransform, ObjectImageTransform

import warnings
warnings.filterwarnings("ignore")




class Dataset( object ):
    """
    Generic dataset, does not support resampling
    """

    def __init__(self, 
        data,
        num_channels=1,
        count=None,
        transform=None 
        ):
        """
        Initialization 
        Args:
            @data: dataprovide class
            @num_channels: num of channels in image data
            @tranform: transformation functions
        """             
        
        if count is None: count = len(data)
        self.count = count
        self.data = data
        self.num_channels = num_channels        
        self.transform = transform   
        self.labels = data.labels if hasattr(data, 'labels') else [0] * len(data)
        self.classes = np.unique(self.labels) 
        self.numclass = len(self.classes)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label = self.data[idx]
        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)        
        label = utility.to_one_hot(label, self.numclass) #no one-hot haixuanguo

        # parse image and label to tensor
        obj = ObjectImageAndLabelTransform( image, label )
        # transform data
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()
    

class ResampleDataset( object ):
    """
    Resample data for larger generic dataset
    """

    def __init__(self, 
        data,
        num_channels=1,
        count=200,
        transform=None  
        ):
        """
        Initialization   
        data: dataloader class
        tranform: tranform           
        """             
        
        self.num_channels=num_channels
        self.data = data        
        self.transform = transform   
        self.labels = data.labels 
        self.count=count
        
        #self.classes = np.unique(self.labels)
        self.classes, self.frecs = np.unique(self.labels, return_counts=True)
        self.numclass = len(self.classes)
        
        #self.weights = 1-(self.frecs/np.sum(self.frecs))
        self.weights = np.ones( (self.numclass,1) )        
        self.reset(self.weights)
        
        self.labels_index = list()
        for cl in range( self.numclass ):             
            indx = np.where(self.labels==cl)[0]
            self.labels_index.append(indx)            

    # shuffle teh distribution
    def reset(self, weights):        
        self.dist_of_classes = np.array(random.choices(self.classes, weights=weights, k=self.count ))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   
                
        idx = self.dist_of_classes[idx]
        class_index = self.labels_index[idx]
        n =  len(class_index)
        idx = class_index[ random.randint(0,n-1) ]

        image, label = self.data[idx]

        image = np.array(image) 
        image = utility.to_channels(image, self.num_channels)            
        label = utility.to_one_hot(label, self.numclass)

        obj = ObjectImageAndLabelTransform( image, label )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()




# The following are not used in our code, but are included to prevent import statements from crashing
class SecuencialSamplesDataset( object ):
    pass

class SecuencialExSamplesDataset( object ):
    pass

class TripletsDataset( object ):
    pass

class MitosisDataset( object ):
    pass

class MitosisSecuencialSamplesDataset( object ):
    pass