from glob import glob
from random import shuffle
import os, cv2
import numpy as np
import torch

class RGBVideoInstanceLoader:
    def __init__(self, path):
        self.path = path
        self.frame_sequences = None
        self.length = 0
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        if self.frame_sequences is None:
            img_files = sorted(glob(os.path.join(self.path,'*.jpg')))
            num_imgs  = min(20,len(img_files)//2)
            step = max(1,num_imgs//2)
            # print(num_imgs, len(img_files), step)
            img_read_fun = lambda f: np.transpose(cv2.resize(cv2.imread(f)/255,(224,224)), (2,1,0))
            pre_imgs  = torch.tensor([img_read_fun(x) for x in img_files[:2*step+1:step]]).float()
            post_imgs = torch.tensor([img_read_fun(x) for x in img_files[-2*step-1::step]]).float()
            frame_sequences = [pre_imgs, post_imgs]
            # print('return')
            return frame_sequences
            self.frame_sequences = [pre_imgs, post_imgs]
            self.length = len(self.frame_sequences)
        return self.frame_sequences[index]
    
class RGBVideoSequencePair:
    def __init__(self, pair):
        self.pair = pair
        
    def __getitem__(self, index):
        return self.pair[index]
    
    def __len__(self):
        return len(self.pair)
    
    def to(self, device):
        for i in range(len(self.pair)):
            self.pair[i] = [self.pair[i][j].to(device) for j in range(len(self.pair[i]))]
        return self
    
class RGBVideoDataLoader:
    def __init__(self, path_list, batch_size):
        path_parser = lambda f: (RGBVideoInstanceLoader(f[0]),int(f[1])-1)
        self.instanceLoaders = [path_parser(f) for f in path_list]
        self.__reset()
        self.batch_size = batch_size
        self.length = len(path_list) // self.batch_size + 1
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
        try:
            result, labels = zip(*self.instanceLoaders[start:end])
        except:
            self.__reset()
            raise StopIteration()
        result = [r[0] for r in result]
        result = list(zip(*result))
        result = RGBVideoSequencePair(result)
        self.index += 1
        return result, labels