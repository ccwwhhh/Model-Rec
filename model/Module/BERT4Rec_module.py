import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
from data.pretrain import Pretrain
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()
class BERT_Encoder(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(BERT_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.datasetFile = datasetFile
        self.feature = feature
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_

        if (self.feature == 'text' or self.feature == 'id+text'):

            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 1
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")
                tensor = tensor.to(1)
                torch_mask = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor = torch.cat([self.bert_tensor, torch_mask], 0)
                self.mlps = MLPS(self.emb_size)
                self.bert_tensor.requires_grad_(True)
            elif (len(self.datasetFile.split(",")) > 1):
                torch_mask = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.datasetFile.split(","):
                    if not os.path.exists(dataset + "whole_tensor.pt"):
                        mask = 1
                        pre = Pretrain(self.data, dataset, mask)
                    tensor = torch.load(dataset + "whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor = torch.cat([self.bert_tensor, torch_mask], 0)
                
                # self.bert_tensor = torch.load('./dataset/fuse_tensor'+ filename+'.pt')
                self.mlps = MLPS(self.emb_size)
                self.bert_tensor.requires_grad_(True)

        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 2, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len + 2, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)
       
        #basically the same with SASRec
        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer = torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate, 'gelu')
            self.forward_layers.append(new_fwd_layer)

    def freeze(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def forward(self, seq, pos):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        if (self.feature == 'text'):
            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq.cuda()])
        # seq_emb = self.item_emb[seq]
        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        # tl = seq_emb.shape[1]
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=None)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb


# encoder
class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
        # Instantiate BERT model
        # self.bert = BertModel.from_pretrained('/usr/gao/cwh/bert')
        # self.bert = BertModel.from_pretrained('bert')
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert_tensor = bert_tensor
        # self.bert_tensor.requires_grad = True
        # print("self.bert_tensor", self.bert_tensor[0])
        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.H),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):
        # [batchsize,sequenceLen,large_item_embedding]->[batchsize,sequenceLen,small_item_embedding]
        logits = self.classifier(bert_tensor)

        # print("logits", logits.shape)
        # logits=torch.reshape(logits,(batch,m,self.H))
        return logits
