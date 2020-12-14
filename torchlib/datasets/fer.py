import os
import numpy as np
import h5py
import cv2
import math

from pytvision.datasets.imageutl import dataProvide
from pytvision.transforms import functional as F


# Define a polygon to decide where to paste a face for a synthetic dataset
def getmask( x, p ):
    mask = np.zeros( x.shape[:2] )
    p = [  [[int(e[0]),int(e[1]) ]]  for e in p ]
    hull = cv2.convexHull( np.array(p),  False  )
    cv2.fillPoly(mask, [hull], 1)   
    return mask

# Obtain the necessary parameters to rotate our image to augment our data
def param2theta(mat_r, w, h):
    H = np.concatenate( (mat_r,[[0,0,1]]),axis=0 )      
    param = H #np.linalg.inv(H)
    theta = np.zeros([2,3])
    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*h/w
    theta[0,2] = param[0,2]*2/w + param[0,0] + param[0,1] - 1
    theta[1,0] = param[1,0]*w/h
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*2/h + param[1,0] + param[1,1] - 1
    return theta

# Turn our angle into a matrix we can use to convert our image
def angle2mat( fi ):
    "To convert angle to rotation matrix"
    rx = fi[0]; ry = fi[1]; rz = fi[2];
    sx = np.sin(rx); cx = np.cos(rx)
    sy = np.sin(ry); cy = np.cos(ry)
    sz = np.sin(rz); cz = np.cos(rz)
    Rx = np.array([[ 1,   0,  0], [ 0, cx, -sx], [  0, sx, cx]])
    Ry = np.array([[cy,   0, sy], [ 0,  1,   0], [-sy,  0, cy]]) 
    Rz = np.array([[cz, -sz,  0], [sz, cz,   0], [  0,  0,  1]])
    return np.dot(Rz,np.dot(Ry,Rx))

def boundingbox(box):
    "Estimate boundingbox"
    xmin = np.min(box[:,0]); ymin = np.min(box[:,1])
    xmax = np.max(box[:,0]); ymax = np.max(box[:,1])
    bbox = np.array([[xmin, ymin],[xmax, ymax]])
    return bbox

def resize( image, height=128,  width=128, interpolate_mode=cv2.INTER_LANCZOS4 ):
    return F.resize_image(image, 
        height=height, width=width, 
        resize_mode='square', 
        padding_mode=cv2.BORDER_CONSTANT,
        interpolate_mode=interpolate_mode, 
        )

# Adjust our face mask so that we don't majorly change the image when we apply a face
def adjust( image, plm  ):
    
    mu = plm.mean(axis=0)
    plm_c = plm - mu
    p1 = [ *plm_c[39,:], 1 ] 
    p2 = [ *plm_c[42,:], 1 ]     
    l = np.cross(p1, p2)
    th =  math.atan2( l[1]/l[2], l[0]/l[2] )

    d = ( (p1[0] - p2[0] )**2 + (p1[1] - p2[1] )**2 ) ** 0.5    
    ang = 90 - th* 180 / math.pi
    imsize = image.shape     

    matR = cv2.getRotationMatrix2D( (imsize[1]//2, imsize[0]//2) , -ang ,1 ) 
    H = param2theta(matR, imsize[1], imsize[0]  )   
    image_rot = cv2.warpAffine(image, matR, (imsize[1], imsize[0]))
    plm_rot = np.dot( H, np.concatenate((plm_c,np.ones([plm_c.shape[0],1])),axis=1).T )   
    plm_rot = plm_rot.T
    plm_rot = plm_rot + mu
    
    bbox = boundingbox( plm_rot )
    br = d*0.2
    bbox[0,0] = bbox[0,0] - br if bbox[0,0] - br > 0 else 0    
    bbox[0,1] = bbox[0,1] - br if bbox[0,1] - br > 0 else 0
    bbox[1,0] = bbox[1,0] + br if bbox[1,0] + br < imsize[0] else imsize[1]
    bbox[1,1] = bbox[1,1] + br if bbox[1,1] + br < imsize[1] else imsize[0]
        
        
    image_rot = image_rot[  int(bbox[0,1]):int(bbox[1,1]), int(bbox[0,0]):int(bbox[1,0]) ]
    plm_rot[:,0] = plm_rot[:,0] - bbox[0,0]
    plm_rot[:,1] = plm_rot[:,1] - bbox[0,1]
    
    return image_rot, plm_rot

class FERClassicDataset( dataProvide ):
    """
    FER CLASSIC dataset
        CK, JAFFE, BU
    one-hot encode emotions
    Args:
        path: data path
        filename: file extention must be .h5
        idenselect: index selected for testing, use Leave-10-subject-out methods to get train and test
        transform (callable, optional): Optional transform to be applied on a sample.   
    """

    classes = ['Neutral - NE', 'Happiness - HA', 'Surprise - SU', 'Sadness - SA', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Contempt - CO']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    def __init__(self, 
        path,
        filename,
        idenselect=[],
        train=True,
        transform=None,
        ):
          
        
        if os.path.isdir(path) is not True:
            raise ValueError('Path {} is not directory'.format(path))

        # read data file
        self.path = path
        self.filename = filename
        dir = os.path.join(path, filename + '.h5' )
        f = h5py.File(dir)

        # image vectors
        self.data   = np.array(f["data"])
        # image dimention
        self.imsize = np.array(f["imsize"])[0,:].astype(int)
        # subject number in the image
        self.iactor = np.array(f["iactor"]).astype(int)
        # emotion label
        self.labels = np.array(f["iclass"]).astype(int)
        # total number of data
        self.num = len(f["iclass"])

        # Emotions class to index/numbers
        if filename == 'ck' or filename == 'ckp':       
            #classes = ['Neutral - NE', 'Anger - AN', 'Contempt - CO', 'Disgust - DI', 'Fear - FR', 'Happiness - HA', 'Sadness - SA', 'Surprise - SU']
            toferp = [0, 4, 7, 5, 6, 1, 3, 2 ]
        elif filename=='bu3dfe' or filename=='jaffe':
            #classes = ['Neutral - NE', 'Anger - AN', 'Disgust - DI', 'Fear - FR', 'Happiness - HA', 'Sadness - SA', 'Surprise - SU', 'Contempt - CO']
            toferp = [0, 4, 5, 6, 1, 3, 2, 7 ]
        else:
            assert(False)

        # parser emotions to numbers
        self.labels = np.array([ toferp[l] for l in self.labels ])
        self.numclass = len(np.unique(self.labels))

        # choose images containing the selected actors
        index = np.ones( (self.num,1) )
        actors = np.unique(self.iactor)
        for i in idenselect:
            index[self.iactor == actors[i]] = 0
        # choose the train or validation index
        self.indexs = np.where(index == train)[0]        
        self.transform = transform
        
        self.labels_org = self.labels
        self.labels = self.labels[ self.indexs ]
        self.classes = [self.classes[ i ] for i in  np.unique( self.labels ) ] 
        self.numclass = len(self.classes)
        self.index = 0
              

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, i):   

        if i<0 and i>len(self.index): raise ValueError('Index outside range')
        i = self.indexs[i]
        self.index = i        
        image = np.array( self.data[i].reshape(self.imsize), dtype=np.uint8 )
        label = self.labels_org[ i ]  ### -1  ##########            
        return image, label

    def iden(self, i):
        return self.iactor[i]

    def getladmarks(self):
        pass

    def getroi(self):
        pass


# This dataset is not used in our code. This is left here so as to not interfere with import statements elsewhere.
class FERDarkClassicDataset( FERClassicDataset ):
    pass