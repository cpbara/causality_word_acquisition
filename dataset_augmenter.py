import argparse
import torch
from src.consts import Consts
from src.data.data_splitters import action_recognition_splits
import json

from glob import glob
import os, cv2
import numpy as np

from src.data.caption_loader import Tokenizer
from src.models.image_model import ImageModel
from src.models.noun_grounding_model import NoundGroundingModel
import spacy
from nltk import word_tokenize, pos_tag
import string

def main(args):
    files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list)
    
    cap_lens = []
    vocabulary = set()
    for _, lst in files.items():
        for path, _ in lst:
            for amt_file in sorted(glob(os.path.join(path,'*amt.txt'))):
                for line in open(amt_file):
                    words = line.strip('<start>').strip('<end').split()
                    cap_lens.append(len(words))
                    vocabulary |= (set(words))
    vocabulary -= set(string.punctuation)
    vocabulary = dict([(w,i) for i,w in enumerate(sorted(list(vocabulary)))])
    json.dump(vocabulary,open('vocabulary.json','w'),indent=4)
    for _, lst in files.items():
        for path, _ in lst:
            for amt_file in sorted(glob(os.path.join(path,'*amt.txt'))):
                emb = np.zeros(len(vocabulary))
                for line in open(amt_file):
                    for word in line.strip('<start>').strip('<end').split():
                        if word in vocabulary:
                            emb[vocabulary[word]] = 1
                out_file = amt_file.rsplit('.',1)[0]+'_naive.npz'
                np.savez_compressed(out_file,data=emb)
                print(out_file)

    from_pretrained_bert_lang_model = Consts.untrained_bert()
    from_pretrained_bert_lang_model.load_state_dict(torch.load('models/language_model_from_pretrained_bert.torch'))
    from_pretrained_bert_lang_model.eval()
    
    from_untrained_bert_lang_model = Consts.untrained_bert()
    from_untrained_bert_lang_model.load_state_dict(torch.load('models/language_model_from_untrained_bert.torch'))
    from_untrained_bert_lang_model.eval()
    
    image_model = ImageModel().cuda().eval()
    
    noun_grounding_model = NoundGroundingModel().cuda().eval()
    
    tokenize = Tokenizer()
    img_read_fun = lambda f: np.transpose(cv2.resize(cv2.imread(f)/255,(224,224)), (2,1,0))
    
    nlp = spacy.load('en_core_web_md')
    
    for _, lst in files.items():
        for path, _ in lst:
            caption_emb_files = sorted(glob(os.path.join(path,'*amt_pretrained_bert.npz')))
            bbox_file_groups = sorted(glob(os.path.join(path,'*_bbox.json')))
            split_fun = lambda x: [x[:len(x)//2],x[len(x)//2:]]
            bbox_file_groups = split_fun(bbox_file_groups)
            for bbox_files, caption_file in zip(bbox_file_groups,caption_emb_files):
                for bbox_file in bbox_files:
                    bbox_data = json.load(open(bbox_file))
                    
                    img_file = '.'.join([bbox_file.strip('_bbox.json'),'jpg'])
                    img = cv2.imread(img_file)/255
                    
                    img_emb_file = '_'.join([bbox_file.strip('_bbox.json'),'rgb.npz'])
                    img_emb = np.load(img_emb_file)['data'][0]
                    
                    cap_emb = np.load(caption_file)['data'][0]
                    
                    is_noun = lambda pos: pos[:2] == 'NN'
                    sentence = bbox_data['caption'].replace(" <end>", ".").partition(' ')[2].replace('..','.')
                    tokenized = word_tokenize(sentence)
                    tokenized = word_tokenize(sentence)
                    words = [word.lower() for (word, _) in pos_tag(tokenized)]
                    temp = [is_noun(pos) for (_, pos) in pos_tag(tokenized)]
                    ids = [i for i, x in enumerate(temp) if x][:6]
                    embs = nlp(sentence)
                    w_embs = {}
                    for id in ids:
                        w_embs[words[id]] = embs[id].vector
                        
                    pr_mask = np.zeros(img.shape)
                    for word, emb in w_embs.items():
                        if word in bbox_data['word_bboxes']:
                            x = torch.stack([torch.tensor(img_emb).float().cuda()]), \
                                torch.stack([torch.tensor(cap_emb).float().cuda()]), \
                                torch.stack([torch.tensor(emb).float().cuda()])
                            # print(x[0].shape, x[1].shape, x[2].shape)
                            bbox, rel = noun_grounding_model(x)
                            rel = rel[0]
                            rel = torch.argmax(rel).cpu().data.numpy()
                            # print(rel, rel == 0)
                            if rel == 0:
                                bbox = bbox.cpu().data.numpy()[0]
                                bbox = img.shape[0]*bbox[0], img.shape[1]*bbox[1], img.shape[0]*bbox[2], img.shape[1]*bbox[3]
                                bbox = [int(x) for x in bbox]
                                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]
                                gtm = np.zeros(img.shape)
                                gtm[ymin:ymax,xmin:xmax,:] = 1
                                pr_mask += gtm
                    pr_mask = np.clip(pr_mask,0,1)/2 + 0.5
                    
                    # exit()
                    out = image_model(torch.stack([torch.tensor(np.transpose(cv2.resize(img*pr_mask,(224,224)), (2,1,0))).float().cuda()])).cpu().data.numpy()
                    
                    out_file = '_'.join([bbox_file.strip('_bbox.json'),'rgb_pr_bbox.npz'])
                    np.savez_compressed(out_file,data=out)
                    print(out_file)
            for bbox_file in sorted(glob(os.path.join(path,'*_bbox.json'))):
                bbox_data = json.load(open(bbox_file))
                
                img_file = '.'.join([bbox_file.strip('_bbox.json'),'jpg'])
                img = cv2.imread(img_file)/255 
                
                gt_mask = np.zeros(img.shape)
                # pr_mask = np.zeros(img.shape)
                for b in bbox_data['phrase_bboxes'].values():
                    b = [int(x) for x in b]
                    # print(b)
                    xmin, ymin, xmax, ymax = b[0], b[1], b[2]+b[0], b[3]+b[1]
                    gtm = np.zeros(img.shape)
                    gtm[ymin:ymax,xmin:xmax,:] = 1
                    gt_mask += gtm
                gt_mask = np.clip(gt_mask,0,1)/2 + 0.5
                
                out = image_model(torch.stack([torch.tensor(np.transpose(cv2.resize(img*gt_mask,(224,224)), (2,1,0))).float().cuda()])).cpu().data.numpy()
                
                out_file = '_'.join([bbox_file.strip('_bbox.json'),'rgb_gt_bbox.npz'])
                np.savez_compressed(out_file,data=out)
                print(out_file)
            for caption_file in sorted(glob(os.path.join(path,'*_amt.txt'))):
                token = tokenize(caption_file)

                _, out = from_pretrained_bert_lang_model(token['input_ids'], token['attention_mask'])
                out_file = '_'.join([caption_file.rsplit('.',1)[0],'pretrained_bert.npz'])
                np.savez_compressed(out_file,data=out.cpu().data.numpy())
                print(out_file)

                _, out = from_untrained_bert_lang_model(token['input_ids'], token['attention_mask'])
                out_file = '_'.join([caption_file.rsplit('.',1)[0],'untrained_bert.npz'])
                np.savez_compressed(out_file,data=out.cpu().data.numpy())
                print(out_file)
            for img_path in sorted(glob(os.path.join(path,'*jpg'))):
                out = image_model(torch.stack([torch.tensor(img_read_fun(img_path)).float().cuda()])).cpu().data.numpy()
                out_file = '_'.join([img_path.rsplit('.',1)[0],'rgb.npz'])
                np.savez_compressed(out_file,data=out)
                print(out_file)
            for img_path in sorted(glob(os.path.join(path,'*png'))):
                out = image_model(torch.stack([torch.tensor(img_read_fun(img_path)).float().cuda()])).cpu().data.numpy()
                out_file = '.'.join([img_path.rsplit('.',1)[0],'npz'])
                np.savez_compressed(out_file,data=out)
                print(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Model Trainer')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--out_model_path', type=str, default='classification_model.torch',
                        help='path to where the trained classifier model should be saved')
    parser.add_argument('--language_model_path', type=str, default='language_model.torch',
                        help='path to where the trained language model should be saved')
    parser.add_argument('--learning_rate', type=float, default=Consts.learning_rate,
                        help='optimizer learning rate')
    parser.add_argument('--num_epochs', type=int, default=Consts.num_epochs,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=Consts.batch_size,
                        help='size of the mini batch')
    parser.add_argument('--train_list', type=str, default=Consts.Splits.ActionRecognition.train_list,
                        help='path to training instance list file')
    parser.add_argument('--test_list', type=str, default=Consts.Splits.ActionRecognition.test_list,
                        help='path to training instance list file')
    parser.add_argument('--data_root_path', type=str, default=Consts.data_root_path,
                        help='path to training instance list file')
    parser.add_argument('--device', type=str, default=Consts.DEVICE,
                        help='device to use cuda or cpu')
    main(parser.parse_args())
