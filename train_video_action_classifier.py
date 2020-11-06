import argparse
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from src.utils.stopwatch import Stopwatch
from src.utils.model_trainer import ModelTrainer
from src.consts import Consts
from src.data.data_splitters import action_recognition_splits
from statistics import mean
import random

from src.data.rgb_video_loader import RGBVideoDataLoader
from src.models.rgb_video_action_classifier import RGBVideoActionClassifier

random.seed(0)

SUPER_CLASSES = [0]*2 + [1]*3 + [2]*2 + [3]*4 + [4]*3 + [5]*2 + [6]*2 + \
                [7]*3 + [8]*2 + [9]*3 + [10]*3 + [11]*2 + [12]*3 + [13]*3 + \
                [14]*2 + [15]*4

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.1)
def main(args):    
    def batch_prepper(batch):
        # print('#',end='',flush=True)
        return batch[0], torch.stack([torch.eye(43)[l] for l in batch[1]])
        # return batch[0], torch.stack([torch.eye(16)[SUPER_CLASSES[l]] for l in batch[1]])
        # return batch[0], torch.tensor([SUPER_CLASSES[l] for l in batch[1]])
        # return batch[0], torch.tensor(batch[1])

    def batch_parser(*args):
        progress = 100*args[0]/args[1]
        print(f'{progress:6.2f} {args[2]:10.8f}',end='\r')
        # print(args[0].shape)
        # print(args[1].shape)
        # print('&',end='',flush=True)
        # return list(torch.argmax(args[0],dim=-1).cpu().data.numpy()), list(args[1].cpu().data.numpy())
        return args[2], [list(torch.argmax(x,dim=-1).cpu().data.numpy()) for x in args[-2:]]
        
    def epoch_parser(train_results, val_results):
        # print('',flush=True)
        train_losses,   train_pairs = zip(*train_results)
        val_losses,     val_pairs   = zip(*val_results)
        train_pred, train_labels = zip(*train_pairs)
        val_pred, val_labels = zip(*val_pairs)
        train_acc = accuracy_score(sum(train_pred,[]), sum(train_labels,[]))
        val_acc = accuracy_score(sum(val_pred,[]), sum(val_labels,[]))
        f = lambda y: [SUPER_CLASSES[x] for x in y]
        train_acc2 = accuracy_score(f(sum(train_pred,[])), f(sum(train_labels,[])))
        val_acc2 = accuracy_score(f(sum(val_pred,[])), f(sum(val_labels,[])))
        return mean(train_losses), mean(val_losses), 100*train_acc, 100*val_acc, 100*train_acc2, 100*val_acc2
    
    files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list)
    data_loader = {}
    for key, val in files.items():
        data_loader[key] = RGBVideoDataLoader(val, batch_size=args.batch_size)

    model = RGBVideoActionClassifier().to(args.device)

    if not args.pretrained_model is None:
        model.load_state_dict(torch.load(args.pretrained_model))
    else:
        model.apply(init_weights)

    trainer = ModelTrainer(
        model = model,
        criterion = nn.MSELoss(),
        # criterion = nn.CrossEntropyLoss(),
        optimizer = lambda model: optim.SGD (
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-4),
        # optimizer = lambda model: optim.Adam(
        #     model.parameters(), 
        #     lr=args.learning_rate),
        batch_preproc=batch_prepper,
        batch_postproc=batch_parser,
        epoch_postproc=epoch_parser,
        device = args.device,
        max_grad_norm=args.max_grad_norm
    )

    timer = Stopwatch()
    pa = 0
    for epoch in range(args.num_epochs):
        tl, vl, ta, va, ta2, va2 = trainer(data_loader)
        out_str = '; '.join([
            f'Epoch: {epoch+1:5d}',
            f'Loss:({tl:10.8f}, {vl:10.8f})',
            f'Acc: ({ta:5.1f}, {va:5.1f})',
            f'Acc: ({ta2:5.1f}, {va2:5.1f})',
            timer.lap_str()
        ])
        print(out_str, flush=True)
        if pa < va:
            if not args.language_model_path is None:
                torch.save(model.cpu().img_model.state_dict(), args.language_model_path)
            torch.save(model.cpu().state_dict(), args.out_model_path)
            model.to(args.device)
            pa = va


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Model Trainer')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--out_model_path', type=str, default='classification_model.torch',
                        help='path to where the trained classifier model should be saved')
    parser.add_argument('--language_model_path', type=str, default='image_model.torch',
                        help='path to where the trained language model should be saved')
    parser.add_argument('--learning_rate', type=float, default=Consts.learning_rate,
                        help='optimizer learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=Consts.max_grad_norm,
                        help='maximum gradient norm')
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
