#### CS6680-Computer Vision Final Project
# Implementation of FERAtt: Facial Expression Recognition with Attention Net

### [Paper](http://openaccess.thecvf.com/content_CVPRW_2019/html/MBCCV/Fernandez_FERAtt_Facial_Expression_Recognition_With_Attention_Net_CVPRW_2019_paper.html) | [arXiv](https://arxiv.org/abs/1810.12121)

This code is based on the [original code](https://github.com/pedrodiamel/ferattention) provided by FERAttention's author. 
We cleaned the code and remove some unrelated files, and then trained preactresnet, FERAtt + cls, and FERAtt + cls + rep with two datasets, CK+ and FER+
Our experiment result can be found in ***FinalPresentation.pdf***


### Something New
- Train our model on FER+ dataset
- Include preactresnet in training model
- Try different kernel size to get feature attention map
- Apply the model on real-time facial expression recognition
 
### Key files we need
```
/books
    data_process.ipynb  # process ck+ dataset into h5 file 
    data_synthetic_analysis.ipynb  # display images, backgrounds, sythetic images and masks
    vis_confusion_matrix.ipynb # create confusion matrix of predicted emotion and true emotions
    vis_regonition.ipynb # displays successfully and wrong recognized emotions in proposed model and imrpoved model
/pytvision #third-party package, no comment
/runs #shells for running
/scripts 
    download_coco_dataset.sh # shell for download coco dataset
/torchlib
    /datasets # create datasets for neural network 
        datasets.py         # transform images in datasets
        factory.py          # contains a collection of datasets
        fer.py              # parse raw data into dataset
        ferp.py             # FERPlus dataset includes download, preprocess
        fersynthetic.py     # parse dataset into sythetic face dataset
    /models
        ferattentionnet.py  # ferattention architechture
        preactresnet.py     # preactresnet architecture
    /transforms
        ferender.py         #generator for synthetic images

    attentionnet.py #network for ferattention
    classnet.py     #network for preactresnet
    netlosses.py    #store all loss functions

train.py  #create dataset, build neural network, train model
eval.py   #use the trained model to predict on test data
```
### Prerequisites

- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch
- virtual environment(virtualenv or conda)

### Installation
```shell script
git clone https://github.com/HelenGuohx/cv-ferattn-code.git
cd ferattn_code_light
python setup.py install
pip install -r installation.txt
```

### How to train and evaluate models

1. Clone the Repo from GitHub
2. Download the CK+ dataset from kaggle into a directory called ~/.datasets
3. Download coco dataset using bash scripts/download_coco_dataset.sh
4. Open and execute the jupyter notebook scripts/data_process.ipynb
5. Execute bash scripts in the runs directory
The <MODEL_PATH> is the folder that contains the saved model. 
you can find path printed out in the last line on console after running training shell
or under /out/<NAMEMETHOD>/<MODEL_PATH>
One example is 
MODEL_PATH = feratt_attnet_ferattention_attloss_adam_ck_synthetic_filter32_pool_size2_dim32_bbpreactresnet_fold5_000
```shell script
# Train real ck+
cd runs 

bash train_ck.sh
bash eval_ck.sh <MODEL_PATH>

# train synthetic ck+
# change BREAL to  'synthetic' in train_ck.sh and eval_ck.sh
bash train_ck.sh
bash eval_ck.sh <MODEL_PATH>

#train real FERPlus 
bash train_ferp.sh
bash eval_ferp.sh <MODEL_PATH>

#train synthetic FERPlus 
#change BREAL to 'synthetic' in train_ferp.sh and eval_ferp.sh
bash train_ferp.sh
bash eval_ferp.sh <MODEL_PATH>


# tune number of filters
bash modify_num_filter.sh

# tune beta and alpha
bash beta_train_ck.sh

```
The approximate time to execute the code




### How to apply FERAttn on real-time facial recognition
```shell script
cd fervideo

# download haarcascade_frontalface_default.xml from 
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

python liveVideoFrameRead.py --fname <fname> --projectname <projectname>

# example
# python liveVideoFrameRead.py --fname classnet --projectname feratt_classnet_preactresnet18_attloss_adam_ferp_real_filter32_dim32_bbpreactresnet_fold5_000

```  
