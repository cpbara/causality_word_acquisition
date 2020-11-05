from glob import glob
from transformers import BertTokenizer
from random import shuffle
import os

class Tokenizer:
    def __init__(self, tokenizer = None) -> None:
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased', do_lower_case=True
                ) 
    def __call__(self, sentence):
        return self.tokenizer(
                sentence, 
                add_special_tokens=True, padding=True, return_tensors='pt',
            )
        

class CaptionInstanceLoader:
    def __init__(self, path):
        self.path = path
        self.captions = None
        self.length = 0
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        if self.captions is None:
            amt_files = sorted(glob(os.path.join(self.path,'*_amt.txt')))
            self.captions = [open(x).readline().strip() for x in amt_files]
            self.length = len(self.captions)
        return self.captions[index]    
    
class TokenizationPair:
    def __init__(self, pair):
        self.pair = pair
        
    def __getitem__(self, index):
        return self.pair[index]
    
    def to(self, device):
        for item in self.pair:
            item.to(device)
        return self
    
class CaptionDataLoader:
    def __init__(self, path_list, batchsize, tokenizer = Tokenizer()):
        path_parser = lambda f: (CaptionInstanceLoader(f[0]),int(f[1])-1)
        self.instanceLoaders = [path_parser(f) for f in path_list]
        self.__reset()
        self.batchsize = batchsize
        self.length = len(path_list) // self.batchsize + 1
        self.tokenize = tokenizer
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
        start = self.index * self.batchsize
        end   =      start + self.batchsize
        result, labels = zip(*self.instanceLoaders[start:end])
        result = tuple(map(self.tokenize, list(zip(*result))))
        # result = [(x['input_ids'], x['attention_mask']) for x in result]
        result = TokenizationPair(result)
        self.index += 1
        return result, labels