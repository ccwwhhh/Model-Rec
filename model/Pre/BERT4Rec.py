import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.loss_torch import l2_reg_loss
from util.structure import PointWiseFeedForward
from random import sample
from math import floor
import random
import math
import pandas as pd
from datetime import datetime
from data.pretrain import Pretrain
import os
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from model.Module.BERT4Rec_module import MLPS
from model.Module.BERT4Rec_module import BERT_Encoder
# #
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()

# Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM'19
class BERT4Rec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BERT4Rec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['BERT4Rec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        datasetFile = self.config['dataset']
        self.eps = float(args['-eps'])
        head_num = int(args['-n_heads'])
        self.cl_type = args['-cltype']
        self.cl_rate = float(args['-lambda'])
        self.cl = float(args['-cl'])
        self.uni=float(args['-uni'])
        self.aug_rate = float(args['-mask_rate'])
        self.model = BERT_Encoder(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,self.feature,datasetFile)
        with open("./count_office.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    data = line[1:-1].split(", ")
                    data = np.asfarray(data, float)
        file.close()
        self.data1=data
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        listUNI=[]
        for epoch in range(self.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, seq_len = batch

                aug_seq, masked, labels = self.item_mask_for_bert(seq, seq_len, self.aug_rate, self.data.item_num+1)
                seq_emb = model.forward(aug_seq, pos)
                if self.cl == 1:
                    cl_loss = self.cl_rate * self.cal_cl_loss(labels,seq_emb,masked)
                else:
                    cl_loss = 0
                #衡量uni-loss
                if self.uni == 1:
                   #Standardized Sampling
                   uni_item_loss=self.uniformity_loss(y,pos,neg_idx)
                elif self.uni == 2:
                    #User Sequence Sampling
                   uni_item_loss = self.uniformity_loss_designed(y, pos, neg_idx)
                elif self.uni == 3:
                   #Popularity Sampling
                   uni_item_loss = self.uniformity_loss_popularity(y, pos, neg_idx,self.data1)
                else:
                   uni_item_loss = 0
                UNI = self.uniformity_loss_index()
                rec_loss = self.calculate_loss(seq_emb,masked,labels)

                # batch_loss = cl_loss+rec_loss+0.03*uni_item_loss
                batch_loss = cl_loss+rec_loss+0.03*uni_item_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'uni_item_loss:',uni_item_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'uni_item_loss:',uni_item_loss.item(),"UNI",UNI.item())
                if n % 200==0:
                    listUNI.append(UNI.item())

            model.eval()
            self.fast_evaluation(epoch)
        # self.draw()
        torch.save(model.state_dict(), './model/checkpoint/'+self.feature+'/BERT4Rec.pt')
        # with open("./train_loss_1.txt", 'a') as train_los:
        #     train_los.write(str(listUNI)+'\n')
        train_los.close()
    def item_mask_for_bert(self,seq,seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i]*mask_ratio),1))
            masked[i, to_be_masked] = 1
            # print("masked",masked)
            labels =labels+ list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq, masked, np.array(labels)

    def calculate_loss(self, seq_emb, masked, labels):
        
        masked=torch.tensor(masked)
        seq_emb = seq_emb[masked>0]
        seq_emb=seq_emb.view(-1, self.emb_size)
        if self.feature == 'text':
            emb=self.model.mlps(self.model.bert_tensor)
        elif self.feature == 'id':
            emb= self.model.item_emb
        elif self.feature == 'id+text':
            emb = self.model.item_emb+self.model.mlps(self.model.bert_tensor)
        logits = torch.mm(seq_emb, emb.t())
        #F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())/labels.shape[0]
        loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())
        return loss

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            for i,length in enumerate(seq_len):
                if length == self.max_len:
                    seq[i,:length-1] = seq[i,1:]
                    pos[i,:length-1] = pos[i,1:]
                    pos[i, length-1] = length
                    seq[i, length-1] = self.data.item_num+1
                else:
                    pos[i, length] = length+1
                    seq[i,length] = self.data.item_num+1
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]

            item_emb=self.model.item_emb
            if self.feature == 'text':
                  item_emb=self.model.mlps(self.model.bert_tensor)
            if self.feature=='id+text':
                  item_emb=self.model.mlps(self.model.bert_tensor)+self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0),  item_emb.transpose(0, 1))

        return score.cpu().numpy()
    def cal_cl_loss(self,label,seq_emb,masked):
        label=torch.tensor(label)
        # label=torch.unique(label)
        user_view = seq_emb[masked > 0]
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)

        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb

        random_noise1 = torch.rand_like(item_view).cuda()
        item_view_1 =item_view+ torch.sign(item_view)* F.normalize(random_noise1, dim=-1) * self.eps
        random_noise2 = torch.rand_like(item_view).cuda()
        item_view_2 = item_view + torch.sign(item_view) * F.normalize(random_noise2, dim=-1) * self.eps


        random_noise3 = torch.rand_like(user_view).cuda()
        random_noise4 = torch.rand_like(user_view).cuda()
        user_view1=user_view+torch.sign(user_view)* F.normalize(random_noise3, dim=-1) * self.eps
        user_view2 = user_view + torch.sign(user_view) * F.normalize(random_noise4, dim=-1) * self.eps
        item_cl_loss_item = InfoNCE(item_view_1[label], item_view_2[label], 0.2)
        item_cl_loss_user=InfoNCE(user_view1, user_view2, 0.2)
        return  item_cl_loss_item
    def draw(self):
        ItemInd = [i for i in range(self.data.item_num)]
        ItemInd = random.sample(ItemInd, self.data.item_num)
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb

        item_view = item_view[1:]
        Pi=item_view.cpu().detach().numpy()
        import seaborn as sns
        sns.set_theme(style="white")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')
        # plt.figure(figsize=(12, 4))
        from sklearn.manifold import TSNE
        Pi=TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi)


        colors=[]
        for i in range(0,len(Pi)):
            k=math.sqrt(Pi[i][0]*Pi[i][0]+Pi[i][1]*Pi[i][1])
            Pi[i][0]=(Pi[i][0]/k)
            Pi[i][1] = (Pi[i][1]/k)
            # colors.append(Pi[i][0]+Pi[i][1])
        # print(tsne.view())
        x1 = np.array(Pi[ItemInd, 0])
        y1 = np.array(Pi[ItemInd, 1])
        # s1 = plt.scatter(x1, y1, c='lightsteelblue',alpha=0.01,s=170)
        # plt.xticks(fontsize=18, weight='normal')
        columns = [' ', '  ']
        Pi = pd.DataFrame(Pi, columns = columns)

        # plt.subplot(1, 2, 2)
        sns.jointplot(x=' ',y='  ', data=Pi,kind="kde",cmap="Blues", shade=True, shade_lowest=True)
        # plt.yticks(fontsize=18, weight='normal')
        plt.title("BERT4Rec+ID",y=-0.17,fontsize=20,weight='bold')
        plt.show()

        now = datetime.now()
        plt.savefig('./picture/BERT4Rec/fig'+str(datetime.now())+'.svg', dpi=300, bbox_inches='tight',format="svg")
        plt.close()
    pass
    def uniformity_loss(self, label,pos,neg,t=2):

        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        label = torch.tensor(label)
        label= label[np.where(pos != 0)]


        x=item_view[label]

        x=x.reshape([-1,64])
        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1,64])

        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        # x=neg_emb
        x = torch.cat([x, neg_emb], dim=0)

        x = F.normalize(x, dim=-1)



        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


    def uniformity_loss_designed(self, label,pos,neg,t=2):

        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        label = torch.tensor(label)
        labelforcount=label.clone().detach()
        non_zero_counts = np.count_nonzero(labelforcount, axis=1)
        cumulative_counts = np.cumsum(non_zero_counts)
        cumulative_counts.tolist()


        label = label[np.where(pos != 0)]
        x = item_view[label]
        x = x.reshape([-1, 64])

        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5* x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])

        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        x = torch.cat([x, neg_emb], dim=0)
        # list1=self.optimized_find_index_in_final_counts(final_counts,x.shape[0])
        x = F.normalize(x, dim=-1)
        dists = torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        n = x.size(0)

        i = 0
        for num in range(0,len(cumulative_counts)):
             j = cumulative_counts[num]
             # print("dist",dists)

             dist1=torch.pdist(x[i:j], p=2).pow(2).mul(-t).exp().mean().log()
             # print(dist1)
             if(torch.isnan(0.8*dist1/len(cumulative_counts))==0  ):
                dists = dists-0.8*dist1/len(cumulative_counts)

             i=j
        result = dists
        return result

    def uniformity_loss_popularity(self, label, pos, neg, data,t=2):
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        label = torch.tensor(label)

        data=torch.tensor(data)


        label = label[np.where(pos != 0)]
        Xdata = data[label].reshape(-1)

        x = item_view[label]
        x = x.reshape([-1, 64])
        # print(x.shape)
        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])
        Ydata=torch.ones(int(1.5 * x.shape[0]))
        '''
        Ydata = data[realneg]
        Ydata[Ydata == 0] = 1
        '''
        data=torch.cat([Xdata, Ydata], dim=0).cuda()

        # print(data.mean())
        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)
        # data=data.view(-1, 1)

        # data= multiply_tensor_elements(data)
        distance= torch.triu(torch.ger(data,data),  diagonal=1)
        distance=distance[distance != 0]

        distance=distance/15
        # print(distance.mean())
        return torch.div(torch.pdist(x, p=2).pow(2),distance).mul(-t).exp().mean().log()
    def uniformity_loss_index(self, t=2):
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        sample= random.sample(list(range(1, self.data.item_num + 1)),2000)
        emb = item_view[sample]
        emb = emb.reshape([-1, 64])
        emb = F.normalize(emb, dim=-1)
        return torch.pdist(emb, p=2).pow(2).mul(-t).exp().mean().log()


def multiply_tensor_elements(tensor):
    """
    Multiply each element of the tensor with each subsequent element of the same tensor.
    """
    result = []
    for i in range(len(tensor)):
        for j in range(i + 1, len(tensor)):
            result.append(tensor[i] * tensor[j])

    return torch.tensor(result)