#### CS6680-Computer Vision Final Project
# Implementation of FERAtt: Facial Expression Recognition with Attention Net



### [Paper](http://openaccess.thecvf.com/content_CVPRW_2019/html/MBCCV/Fernandez_FERAtt_Facial_Expression_Recognition_With_Attention_Net_CVPRW_2019_paper.html) | [arXiv](https://arxiv.org/abs/1810.12121)

### Phase II submission

1. Add data process components
     - Create file ```./scrripts/data_process.ipynb``` is used to create neutral data and sample
     sample original data with 33, neutral, 45 Angry, 18 Contempt, 58 Disgust, 25 Fear, 69 Happy, 28 Sadness and 82 Surprise
     - Modify FERClassicDataset and add split_train_test function
     
2.  Initial evaluation on CK+ dataset with epoch=5
     - see training result in feratt_attnet_ferattention_attloss_adam_ck_dim32_bbpreactresnet_fold0_000.log 
     - see evaluation result (The training process needs further modified to fit real data) 
     ```shell script
    |Num	|Acc	|Prec	|Rec	|F1	|Set	|Type	
    |0	|0,171	|0,021	|0,125	|0,037	|train	|real	
    |1	|0,065	|0,008	|0,125	|0,015	|val	|real	
    |2	|0,168	|0,021	|0,125	|0,036	|train	|no_real	
    |3	|0,065	|0,008	|0,125	|0,015	|val	|no_real
    ```
   

### Phase III
1. Get the classification result of CK+ dataset and its synthetic dataset
2. Implement representation in neural network and corresponding loss
3. Get classification and representation results on BU3DFE 
4. Add modification


### How to run
```
mkdir ~/.dataset/ck

#1. download ck+dataset online

#2.  create training data in ./scrripts/data_process.ipynb 

#3. train
sh runs/train_ck.sh

#4. evaluation
sh runs/eval_ck.sh


```
  
