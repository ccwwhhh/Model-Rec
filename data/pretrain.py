import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from transformers import BertModel,GPT2LMHeadModel
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.structure import PointWiseFeedForward
from util.loss_torch import l2_reg_loss
from data import feature
import os
from data.sequence import Sequence
from util.conf import OptionConf,ModelConf
from data.loader import FileIO

# Paper: Self-Attentive Sequential Recommendation

class Pretrain(object):
    def __init__(self, data,datasetfile,mask):
        # super(pretrain, self).__init__(conf, training_set, test_set)
        self.datasetfile=datasetfile
        self.data=data
        self.bert=Bert().cuda()
        


        initializer = nn.init.xavier_uniform_
        #whole_tensor=nn.Parameter(initializer(torch.empty(1,768))).cuda()
        #for dataset in self.datasetfile.split(","):

        self.functionName=(datasetfile.split("/")[2]).split("-")[0]
        #eval('feature.'+self.functionName+'('+'self.data.id2item'+')')
        #self.train_inputs, self.train_masks = feature.AmazonProcess(self.data.id2item)
        self.train_inputs, self.train_masks=eval('feature.'+self.functionName+'('+'self.data.id2item'+')')
        
        print(self.train_inputs.shape)
        whole_list = []
        i = 0
        while len(self.train_inputs) > ((i+1) * 100):
            outputs = self.bert(self.train_inputs[i*100:(i+1)*100].cuda(), self.train_masks[i*100:(i+1)*100].cuda())[0][:, 0, :]
            whole_list.append(outputs)
            i = i + 1

        outputs = self.bert(self.train_inputs[i*100:len(self.train_inputs)].cuda(), self.train_masks[i*100:len(self.train_inputs)].cuda())[0][:, 0, :]
        #tensor_size = [outputs.shape[0], outputs.shape[1]]
        whole_list.append(outputs)

        whole_tensor = whole_list[0]
        for i in range(1, len(whole_list)):
            whole_tensor = torch.cat([whole_tensor, whole_list[i]], 0)
        # if(mask):
        #     mask_tensor=nn.Parameter(initializer(torch.empty(1, whole_tensor.shape[1]))).cuda()
        #     whole_tensor = torch.cat([whole_tensor,  mask_tensor], 0)
        #     torch.save(whole_tensor, self.datasetfile + "whole_tensor_mask.pt")
            
       #if len(self.datasetFile.split(","))==1:
        #print(whole_tensor.shape)
        # else:
        torch.save(whole_tensor, self.datasetfile+"whole_tensor.pt")
        # elif len(self.datasetFile.split(","))>=1:
        #     torch.save(whole_tensor,'./dataset/fuse_tensor'+ self.filename+'.pt')
    def execute(self):
        pass
        
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert')
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        return outputs