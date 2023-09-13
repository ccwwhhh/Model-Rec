import numpy as np
from data.data import Data
from collections import defaultdict

class Sequence(Data):
    def __init__(self, conf, training, test):
        super(Sequence, self).__init__(conf, training, test)
        self.item = {}
        self.id2item = {}
        self.seq = {}
        self.id2seq = {}
        # self.training_set_seq = defaultdict(dict)
        # self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.original_seq = self.__generate_set()
        self.raw_seq_num = len(self.seq)
        self.item_num = len(self.item)

    #认为是把序列和item重新编号
    def __generate_set(self):
        #seq从1开始，但不知道这里有没有打乱
        for seq in self.training_data:
            if len(self.training_data[seq]) < 2:
                continue
            if seq not in self.seq:
                self.seq[seq] = len(self.seq)
                self.id2seq[self.seq[seq]] = seq
            for item in self.training_data[seq]:
                if item not in self.item:
                    #self.item里面的应该是重新编号过的物品号
                    self.item[item] = len(self.item)+1 # 0 as placeholder
                    self.id2item[self.item[item]] = item
            #
            # self.training_set_seq[seq][item] = 1
            # self.training_set_i[item][seq] = 1

        for seq in self.test_data:
            if seq not in self.seq:
                continue
            self.test_set[seq][self.test_data[seq][0]] = 1
            self.test_set_item.add(self.test_data[seq][0])

        #
        original_sequences = []
        for seq in self.training_data:
            if len(self.training_data[seq]) < 2:
                continue
            #我认为这是重新编码后的序列
            original_sequences.append((seq,[self.item[item] for item in self.training_data[seq]]))
        return original_sequences

    # def sequence_split(self):
    #     augmented_sequences = []
    #     original_sequences = {}
    #     max_len = 0
    #     for seq in self.training_data:
    #         for n in range(1,len(self.training_data[seq])):
    #             augmented_sequences.append([[self.item[item] for item in self.training_data[seq][0:n]],self.item[self.training_data[seq][n]]])
    #             if len(self.training_data[seq])>max_len:
    #                 max_len = len(self.training_data[seq])
    #         original_sequences[seq]=[self.item[item] for item in self.training_data[seq]]
    #     return augmented_sequences,original_sequences, max_len


    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def get_seq_id(self, i):
        if i in self.seq:
            return self.seq[i]

