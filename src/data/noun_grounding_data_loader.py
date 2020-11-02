from glob import glob
import torch
import json
import numpy as np
from nltk import word_tokenize, pos_tag
import spacy
import os
from random import shuffle

class FrameEmbeddingLoader:
    def __init__(self, img_file):
        self.img_file = img_file
        self.img = None
        self.length = 1
    def __len__(self):
        return self.length
    def __getitem__(self,_):
        if self.img is None:
            self.img = torch.tensor(np.load(self.img_file)['data'][0]).float()
        return self.img
    
class CaptionEmbeddingLoader:
    def __init__(self, embedding_file):
        self.embedding_file = embedding_file
        self.embedding = None
        self.length = 1
    def __len__(self):
        return self.length
    def __getitem__(self,_):
        if self.embedding is None:
            self.embedding = torch.tensor(np.load(self.embedding_file)['data'][0]).float()
        return self.embedding

class FrameNounGroundingInstanceLoader:
    nlp = spacy.load('en_core_web_md')
    def __init__(self, image_loader, embedding_loader, word_embedding, bbox, importance):
        self.image_loader = image_loader
        self.embedding_loader = embedding_loader
        self.word_embedding = word_embedding
        self.bbox = bbox
        self.frame_sequences = None
        self.importance = importance
        self.length = 1
    def __len__(self):
        return self.length
    @staticmethod
    def LoadNounGroundings(bbox_file, untrained_language_model = False):
        fun = lambda x,s: '_'.join([x.strip('_bbox.json'),s])
        img_loader = FrameEmbeddingLoader(fun(bbox_file,'rgb.npz'))
        if untrained_language_model:
            f = fun(bbox_file,'amt_untrained_bert.npz')
            if not os.path.isfile(f):
                return []
            emb_loader = CaptionEmbeddingLoader(f)
        else:
            f = fun(bbox_file,'amt_pretrained_bert.npz')
            if not os.path.isfile(f):
                return []
            emb_loader = CaptionEmbeddingLoader(f)
        bbox_data = json.load(open(bbox_file))
        is_noun = lambda pos: pos[:2] == 'NN'
        sentence = bbox_data['caption'].replace(" <end>", ".").partition(' ')[2].replace('..','.')
        tokenized = word_tokenize(sentence)
        words = [word.lower() for (word, pos) in pos_tag(tokenized)]
        temp = [is_noun(pos) for (word, pos) in pos_tag(tokenized)]
        ids = [i for i, x in enumerate(temp) if x][:6]
        embs = FrameNounGroundingInstanceLoader.nlp(sentence)
        w_embs = {}
        for id in ids:
            w_embs[words[id]] = embs[id].vector
        retval = []
        for word, emb in w_embs.items():
            if word in bbox_data['word_bboxes']:
                bbox = torch.tensor(bbox_data['word_bboxes'][word][0]).float()/224
                importance = torch.tensor([1, 0]).float()
            else:
                bbox = torch.tensor([0,0,0,0]).float()
                importance = torch.tensor([0, 1]).float()
            retval.append(FrameNounGroundingInstanceLoader(img_loader, emb_loader, torch.tensor(emb).float(), bbox, importance))
        return retval
    def __getitem__(self,_):
        return self.image_loader[0], self.embedding_loader[0], self.word_embedding, self.bbox, self.importance
        
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
    def __init__(self, path_list, batchsize, untrained_language_model = False):
        self.batchsize = batchsize
        fun = lambda path: sorted(glob(os.path.join(path,'*bbox.json')))
        self.instance_loaders = []
        for path, _ in path_list:
            bbox_files = fun(path)
            for bbox_file in bbox_files[0::max(1,len(bbox_files)-1)]:
                self.instance_loaders.append(FrameNounGroundingInstanceLoader.LoadNounGroundings(bbox_file, untrained_language_model))
            # break
        self.instance_loaders = sum(self.instance_loaders,[])
        self.index = 0
        shuffle(self.instance_loaders)
        self.length = len(self.instance_loaders) // self.batchsize + 1
        # print(self.length)
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
        end   =      start + self.batchsize
        result = [torch.stack(y) for y in zip(*[x[0] for x in self.instance_loaders[start:end]])]
        self.index += 1
        return NounGroundingTuple(result[:3]), NounGroundingTuple(result[3:])