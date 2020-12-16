import torch
import numpy as np
from glob import glob
import os
import math
from random import shuffle

class ActionInstanceLoader:
    attention_model = None
    def __init__(self, path, mental_attention_sufix=None):
        self.path = path
        self.instances = None
        self.length = 2
        self.mental_attention_sufix = mental_attention_sufix
    def __len__(self):
        return self.length
    def load_npz(self, path):
        retval = np.load(path)['data'][0]
        if len(retval.shape) > 1:
            print(retval.shape)
            retval = np.zeros(4096)
        return retval
    def load_npzs(self, path):
        files = sorted(glob(os.path.join(self.path,path)))
        num  = min(10,len(files)//2)
        if num == 0:
            return torch.zeros(4096).float(),torch.zeros(4096).float()
        pre_emb  = torch.tensor(sum([self.load_npz(x) for x in files[:num]])/num)
        post_emb = torch.tensor(sum([self.load_npz(x) for x in files[-num:]])/num)
        return pre_emb.float(), post_emb.float()
    def __getitem__(self, index):
        if self.instances is None:
            pre_rgb_emb, post_rgb_emb = self.load_npzs('*rgb_4096.npz')
            if not self.mental_attention_sufix is None:
                pre_rgb_ma_emb, post_rgb_ma_emb = self.load_npzs(f'*rgb_4096_{self.mental_attention_sufix}.npz')
            else:
                pre_rgb_ma_emb, post_rgb_ma_emb = self.load_npzs('*rgb_4096.npz')
            self.instances = [[pre_rgb_emb,  pre_rgb_ma_emb  ],
                              [post_rgb_emb, post_rgb_ma_emb ]]
        return self.instances[index]
class ActionDataLoader:
    def __init__(self, path_list, batch_size, mental_attention_sufix=None):
        path_parser = lambda f: (ActionInstanceLoader(f[0],mental_attention_sufix),int(f[1])-1)
        # print(path_list[0])
        self.instanceLoaders = [path_parser(f) for f in path_list]
        self.__reset()
        self.batch_size = batch_size
        self.length = math.ceil(len(path_list) / self.batch_size)
    def __reset(self):
        self.index = 0
        shuffle(self.instanceLoaders)
    def __len__(self):
        return self.length
    def __iter__(self):
        return self
    def __next__(self):
        if self.index >= self.length:
            self.__reset()
            raise StopIteration()
        start = self.index * self.batch_size
        end   =      start + self.batch_size
        if end - start == 1:
            self.__reset()
            raise StopIteration()
        result, labels = zip(*self.instanceLoaders[start:end])
        pre_rgb_embs   = torch.stack([r[0][0] for r in result])
        pre_rgb_ma_embs   = torch.stack([r[0][1] for r in result])
        post_rgb_embs  = torch.stack([r[1][0] for r in result])
        post_rgb_ma_embs  = torch.stack([r[1][1] for r in result])
        result = torch.stack([
            torch.stack([pre_rgb_embs, pre_rgb_ma_embs]),
            torch.stack([post_rgb_embs,post_rgb_ma_embs])
        ])
        self.index += 1
        return result, labels