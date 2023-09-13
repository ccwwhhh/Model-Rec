import csv
import re
import pandas as pd
import numpy as np
from re import split
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, GPT2Tokenizer
from transformers import BertModel,GPT2LMHeadModel
import nltk
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import json
import time

max_len = 150
tokenizer = BertTokenizer.from_pretrained('bert', do_lower_case=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def text_preprocessing(s):
    s = s.lower()
    # 把can‘t变成can not
    s = re.sub(r"\'t", " not", s)
    # 去除@name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # 去除除了问号之外的标点符号
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # 去除特殊字符
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # 除去停用词（常见的停用词有an，a，the等等）
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # 除去尾部空白字符
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # 处理句子
            add_special_tokens=True,  # 加上[CLS] 和[SEP]
            max_length=max_len ,  # 一个句子的最大长度等于max_len
            pad_to_max_length=True,  # padding
            # return_tensors='pt',           #返回 tensor
            return_attention_mask=True , #返回 attention mask
        )
        # Add the outputs to the lists
        
        # if(len(input_ids)==0):
        #     input_ids.append(encoded_sent.get('input_ids'))
        # if(len(attention_masks)==0):
        #     attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # [total sentence, 100]
    return input_ids, attention_masks


#下面的函数根据config里数据集名称自动处理
#在这里加载特征数据集，把序列里的物品ID号和特征对应起来，在这里只取了新闻标题，可以取更多
def MINDsmall(id2item):
    featureOriginal = {}
    featureNew=[]
    with open(r"./dataset/MINDsmall/news.tsv", 'r', encoding="gbk",
              errors="ignore") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            #row[0]是原始的未加-的新闻号
            featureOriginal[row[0]] = row[1] + ' ' + row[2] + row[3]
    for i in id2item:
            featureNew.append(featureOriginal[split('-',id2item[i])[0]])
            # print(featureOriginal[split('-',id2item[i])[0]])
            # print(featureNew)
    train_inputs, train_masks = preprocessing_for_bert(featureNew)
    return train_inputs,train_masks

def Amazon(id2item):
    featureOriginal = {}
    featureNew = []
    csv.field_size_limit(10000000)
    with open(r"./dataset/Amazon-Pantry/feature.json", 'r', encoding="gbk",
              errors="ignore") as tsv_file:

        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        #for row in tsv_file.readlines():

        for row in tsv_reader:
            y=1
            try:
                description = re.findall('"description": (.*?)], "fit"', row[0])[0]
                # print(description)
                asin=re.findall('"asin": "(.*?)",', row[0])[0]

                # print(asin)
                featureOriginal[asin]=description
            except:
                y+=1
                print(y)

            # if('asin' in row.keys()):
            #     featureOriginal[row["asin"]] =' '.join(row["description"])
            # elif('details' in row.keys()):
            #     featureOriginal[row["details"]["asin"]] = ' '.join(row["description"])
    
    for i in id2item:
        try:
            featureNew.append(featureOriginal[id2item[i]])
        except:

            print('None')
            featureNew.append('None')
        # print(featureOriginal[split('-',id2item[i])[0]])
        # print(featureNew)
    train_inputs, train_masks = preprocessing_for_bert(featureNew)
    return train_inputs, train_masks
