import os
from glob import glob
import json
from gensim.models import Word2Vec

try:
    corpus = json.load(open('corpus.json'))
except Exception as e:
    root_dir = '/local/cpbara/actions-transitions/action_frames_release/'
    caption_files = sorted(glob(os.path.join(root_dir,'*/*/*/*/*amt.txt')))

    print(caption_files[0])
    with open(caption_files[0]) as f:
        print(f.readlines()[0].replace('<start> ','').replace(' <end>',''))
    corpus = []

    for caption_file in caption_files:
        print(caption_file)
        with open(caption_file) as f:
            corpus.append(f.readlines()[0].replace('<start> ','').replace(' <end>',''))
    json.dump(corpus,open('corpus.json','w'))


tokenized_corpus = [x.split() for x in corpus]

myWord2Vec = Word2Vec(tokenized_corpus, min_count=1, size=300)

myWord2Vec.save('myWord2Vec.bin')
