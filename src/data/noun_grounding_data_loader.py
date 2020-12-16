from glob import glob
import torch
import json
import numpy as np
from nltk import word_tokenize, pos_tag
import spacy
import os
from random import shuffle
from gensim.models import Word2Vec

class FrameEmbeddingLoader:
    def __init__(self, img_file, emb_file, store=False):
        self.img_file = img_file
        self.emb_file = emb_file
        self.img = None
        self.length = 1
        self.store = store
    def __len__(self):
        return self.length
    def __getitem__(self,_):
        if self.img is None:
            img = torch.tensor(np.load(self.img_file)['data'][0]).float()
            emb = torch.tensor(np.load(self.emb_file)['data'][0]).float()
            if self.store:
                self.img = torch.cat([img,emb],axis=-1)
            else:
                return torch.cat([img,emb],axis=-1)
        return self.img
    
class CaptionEmbeddingLoader:
    def __init__(self, embedding_file, ignore_language_model= False):
        self.embedding_file = embedding_file
        # print(embedding_file)
        self.embedding = None
        self.length = 1
        self.ignore_language_model = ignore_language_model
    def __len__(self):
        return self.length
    def __getitem__(self,_):
        if self.ignore_language_model:
            return torch.zeros(4567).float()
        if self.embedding is None:
            self.embedding = torch.tensor(np.load(self.embedding_file)['data'][0]).float()

        return self.embedding

class FrameNounGroundingInstanceLoader:
    nlp = spacy.load('en_core_web_md')
    w2v = Word2Vec.load('models/myWord2Vec.bin')
    def __init__(self, image_loader, cap_emb, word_embedding, bbox):
        self.image_loader = image_loader
        self.word_embedding = word_embedding
        self.bbox = bbox
        self.frame_sequences = None
        self.cap_emb = cap_emb
        self.length = 1
    def __len__(self):
        return self.length
    @staticmethod
    def LoadNounGroundings(bbox_file, mental_attention_sufix=None):
        fun = lambda x,s: '_'.join([x.replace('_bbox2.json',''),s])
        if not mental_attention_sufix is None:
            img_loader = FrameEmbeddingLoader(fun(bbox_file,'rgb_4096.npz'),fun(bbox_file,f'rgb_4096_{mental_attention_sufix}.npz'), store=True)
        else:
            img_loader = FrameEmbeddingLoader(fun(bbox_file,'rgb_4096.npz'), fun(bbox_file,'rgb_4096.npz'), store=True)
        bbox_data = json.load(open(bbox_file))
        
        word_fix = {
            'countertop' : 'counter-top',
            'dumbbell' : 'dumb-bell',
            'dumbbells' : 'dumb-bells',
            'mne' : 'men',
            'abg' : 'bag',
            'chid' : 'child',
            'anotherman' : 'male',
            'vag' : 'bag',
            'ballon' : 'balloon',
            'sumbbells' : 'dumb-bells',
            'bug' : 'guy',
            'shooners' : 'schooners',
            'civer' : 'cover',
            'wals' : 'walsh'
        }
        
        retval = []
        fun = lambda w: torch.tensor(FrameNounGroundingInstanceLoader.w2v.wv[w]).float()
        cap_emb = sum(map(fun, bbox_data['caption'].split()[1:-1]))
        
        for word, bbox in bbox_data['word_bboxes'].items():
            try:
                if word in word_fix:
                    word = word_fix[word]
                w_emb = FrameNounGroundingInstanceLoader.w2v.wv[word.lower()]
            except Exception as e:
                continue
            w_emb = torch.tensor(w_emb).float()
            bbox = torch.tensor(bbox[0]).float()/224
            loader = FrameNounGroundingInstanceLoader(img_loader, cap_emb, w_emb, bbox)
            retval.append(loader)
        return retval
    def __getitem__(self,_):
        return self.image_loader[0], self.cap_emb, self.word_embedding, self.bbox
        
class NounGroundingTuple:
    def __init__(self, tpl):
        self.tpl = tpl
    def __getitem__(self, index):
        return self.tpl[index]
    def __len__(self):
        return len(self.tpl)
    def to(self, device):
        for i in range(len(self.tpl)):
            self.tpl[i] = self.tpl[i].to(device)
        return self

class NounGroundingDataLoader:
    def __init__(self, path_list, batchsize,mental_attention_sufix=None):
        self.batchsize = batchsize
        fun = lambda path: sorted(glob(os.path.join(path,'*bbox2.json')))
        self.instance_loaders = []
        for path, _ in path_list:
            bbox_files = fun(path)
            for bbox_file in bbox_files[0::max(1,len(bbox_files)-1)]:
                self.instance_loaders.append(FrameNounGroundingInstanceLoader.LoadNounGroundings(bbox_file,mental_attention_sufix))
        self.instance_loaders = sum(self.instance_loaders,[])
        self.index = 0
        shuffle(self.instance_loaders)
        self.length = len(self.instance_loaders) // self.batchsize + 1
    def __reset(self):
        self.index = 0
        shuffle(self.instance_loaders)
    def __len__(self):
        return self.length
    def __iter__(self):
        return self
    def __next__(self):
        if self.index >= self.length:
            self.__reset()
            raise StopIteration()
        start = self.index * self.batchsize
        end   = min(len(self.instance_loaders), start + self.batchsize)
        if end - start == 1:
            self.__reset()
            raise StopIteration()
        result = [torch.stack(y) for y in zip(*[x[0] for x in self.instance_loaders[start:end]])]
        self.index += 1
        return NounGroundingTuple(result[:-1]), result[-1]