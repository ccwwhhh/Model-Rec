# MoRec
is a Python framework for Modal recommendation which integrates commonly used datasets and metrics, and implements many state-of-the-art models. 
It is a framework that satisfies the pre-training and fine-tuning paradigms. It includes multiple sub-frameworks, such as UniTSR, a module applying uniform item embedding suitable for pure text sequence recommendation.



# An Example to Get UniTSR Started

## 1.Download dataset

 
You can get Office from 
链接：https://pan.baidu.com/s/1TGRyNanax1IlIaonjl04og 
提取码：data

## 2.Configure

An example:
>training.set=./dataset/Amazon-Office/train.txt  
test.set=./dataset/Amazon-Office/test.txt  
dataset=./dataset/Amazon-Office/  
model.name=BERT4Rec  
model.type=sequential  
item.ranking=-topN 10,20  
embedding.size=64  
num.max.epoch=100  
batch_size=512  
learnRate=0.001  
reg.lambda=0.0001  
max_len=50  
BERT4Rec=-n_blocks 2 -drop_rate 0.2 -n_heads 1 -mask_rate 0.5 -eps 0.1 -lambda 0.001 -cl 0 -cltype text -uni 1  
output.setup=-dir ./results/  
feature=text

**-uni 1, -uni 2, -uni 3**  
Respectively means Standardized Sampling, User Sequence Sampling, and Popularity Sampling strategies.
**feature=text:** 
Using pure text to create representation.


## 3.Training

>python main.py

>MoRec: A library for Modal Recommendation.
>Baseline Models:
>SASRec   BERT4Rec   CL4SRec   UnisRec
>Please enter the baselines you want to run:(Type model that you want)
>stages:
>Pre   Fine
>Please enter the stage you want:Pre

