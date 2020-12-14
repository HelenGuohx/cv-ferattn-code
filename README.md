#### CS6680-Computer Vision Final Project
# Implementation of FERAtt: Facial Expression Recognition with Attention Net

### [Paper](http://openaccess.thecvf.com/content_CVPRW_2019/html/MBCCV/Fernandez_FERAtt_Facial_Expression_Recognition_With_Attention_Net_CVPRW_2019_paper.html) | [arXiv](https://arxiv.org/abs/1810.12121)

Key files we need
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


### How to run

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

```
  
