import argparse
import torch
from src.consts import Consts
from src.data.data_splitters import action_recognition_splits

from glob import glob
import os, cv2
import numpy as np

from src.data.caption_loader import Tokenizer
from src.models.image_model import ImageModel
        

def main(args):
    from_pretrained_bert_lang_model = Consts.untrained_bert()
    from_pretrained_bert_lang_model.load_state_dict(torch.load('models/language_model_from_pretrained_bert.torch'))
    
    from_untrained_bert_lang_model = Consts.untrained_bert()
    from_untrained_bert_lang_model.load_state_dict(torch.load('models/language_model_from_untrained_bert.torch'))
    
    image_model = ImageModel().cuda()
    
    tokenize = Tokenizer()
    img_read_fun = lambda f: np.transpose(cv2.resize(cv2.imread(f)/255,(224,224)), (2,1,0))
    
    files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list)
    
    for _, lst in files.items():
        for path, _ in lst:
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
