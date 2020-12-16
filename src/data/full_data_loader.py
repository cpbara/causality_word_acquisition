import torch
import numpy as np
from glob import glob
import os
from random import shuffle

class RGBFlowCaptionEmbeddingsInstanceLoader:
    attention_model = None
    def __init__(self, path, load_captions=False, untrained_bert=False, bbox_type=None, attention_model=None):
        self.path = path
        self.instances = None
        self.length = 2
        self.load_captions = load_captions
        self.untrained_bert = untrained_bert
        self.bbox_type = bbox_type
        RGBFlowCaptionEmbeddingsInstanceLoader.attention_model = attention_model
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
            if self.untrained_bert:
                search_str = '*amt_untrained_bert.npz'
            else:
                search_str = '*amt_pretrained_bert.npz'
            caption_files = sorted(glob(os.path.join(self.path,search_str)))
            # cap_loader = lambda f: torch.tensor(np.load(f)['data'][0])*int(self.load_captions)
            cap_loader = lambda f: torch.tensor(np.zeros(4567)).float()
            pre_cap_emb, post_cap_emb = [cap_loader(f) for f in caption_files]
            if self.bbox_type is None:
                pre_rgb_emb, post_rgb_emb = self.load_npzs('*rgb.npz')
            else:
                pre_rgb_emb, post_rgb_emb = self.load_npzs(f'*rgb_{self.bbox_type}.npz')
            if not RGBFlowCaptionEmbeddingsInstanceLoader.attention_model is None:
                RGBFlowCaptionEmbeddingsInstanceLoader.attention_model.cpu().eval()
                with torch.no_grad():
                    pre_cap_emb  = RGBFlowCaptionEmbeddingsInstanceLoader.attention_model(pre_rgb_emb.reshape(1,-1), pre_cap_emb.reshape(1,-1))[0]
                    post_cap_emb = RGBFlowCaptionEmbeddingsInstanceLoader.attention_model(post_rgb_emb.reshape(1,-1), post_cap_emb.reshape(1,-1))[0]
            else:
                pre_cap_emb *= 0
                post_cap_emb *= 0
            pre_flow_emb, post_flow_emb = self.load_npzs('*flow.npz')
            self.instances = [[pre_cap_emb,  pre_rgb_emb,  pre_flow_emb  ],
                              [post_cap_emb, post_rgb_emb, post_flow_emb ]]
        return self.instances[index]
class RGBFlowCaptionEmbeddingsDataLoader:
    def __init__(self, path_list, batch_size, load_captions=False, utrained_bert=False, attention_model=None, bbox_type=None):
        path_parser = lambda f: (RGBFlowCaptionEmbeddingsInstanceLoader(f[0],load_captions, utrained_bert,bbox_type,attention_model),int(f[1])-1)
        # print(path_list[0])
        self.instanceLoaders = [path_parser(f) for f in path_list]
        self.__reset()
        self.batch_size = batch_size
        self.attention_model = attention_model
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
        if end - start == 1:
            self.__reset()
            raise StopIteration()
        result, labels = zip(*self.instanceLoaders[start:end])
        pre_cap_embs   = torch.stack([r[0][0] for r in result]).cuda()
        pre_rgb_embs   = torch.stack([r[0][1] for r in result]).cuda()
        pre_flow_embs  = torch.stack([r[0][2] for r in result]).cuda()
        post_cap_embs  = torch.stack([r[1][0] for r in result]).cuda()
        post_rgb_embs  = torch.stack([r[1][1] for r in result]).cuda()
        post_flow_embs = torch.stack([r[1][2] for r in result]).cuda()
        if self.attention_model is None:
            pre_att_emb  = pre_rgb_embs * 0
            post_att_emb = post_rgb_embs * 0
        else:
            pre_att_emb  = pre_cap_embs
            post_att_emb = post_cap_embs
        result = torch.stack([
            torch.stack([pre_rgb_embs, pre_att_emb, pre_flow_embs]),
            torch.stack([post_rgb_embs,post_att_emb,post_flow_embs])
        ])
        self.index += 1
        return result, labels