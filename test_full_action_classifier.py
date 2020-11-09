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
from src.data.full_data_loader import RGBFlowCaptionEmbeddingsDataLoader
from src.models.full_action_classifier import RGBFlowCaptionActionClassifier
from src.models.noun_grounding_model import MentalAttentionModel

SUPER_CLASSES = [0]*2 + [1]*3 + [2]*2 + [3]*4 + [4]*3 + [5]*2 + [6]*2 + \
                [7]*3 + [8]*2 + [9]*3 + [10]*3 + [11]*2 + [12]*3 + [13]*3 + \
                [14]*2 + [15]*4

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.1)

def main(args):    
    def batch_prepper(batch):
        return batch[0], torch.tensor([l for l in batch[1]])

    def batch_parser(*args):
        return args[2], [list(torch.argmax(args[-2],dim=-1).cpu().data.numpy()), list(args[-1].cpu().data.numpy())]
        
    def epoch_parser(train_results, val_results):
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
    
    if args.attention_model is None:
        attention_model = None
        model = RGBFlowCaptionActionClassifier().to(args.device)
    else:
        attention_model = MentalAttentionModel()
        attention_model.load_state_dict(torch.load(args.attention_model))
        model = RGBFlowCaptionActionClassifier(use_attention=True).to(args.device)
    
    files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list, train_size=args.train_set_size)
    data_loader = {}
    for key, val in files.items():
        data_loader[key] = RGBFlowCaptionEmbeddingsDataLoader(val, batch_size=args.batch_size, attention_model=attention_model, bbox_type=args.bbox_type)

    if not args.pretrained_model is None:
        model.load_state_dict(torch.load(args.pretrained_model))
    else:
        model.apply(init_weights)

    trainer = ModelTrainer(
        model = model,
        criterion = nn.CrossEntropyLoss(),
        optimizer = lambda model: optim.Adam(
            model.parameters(), 
            lr=args.learning_rate),
        batch_preproc=batch_prepper,
        batch_postproc=batch_parser,
        epoch_postproc=epoch_parser,
        device = args.device,
        max_grad_norm=args.max_grad_norm
    )

    timer = Stopwatch()
    pa = 0
    tl, vl, ta, va, ta2, va2 = trainer.test(data_loader)
    out_str = '; '.join([
        f'Acc 43: {va:5.1f}',
        f'Acc 16: {va2:5.1f}',
        timer.lap_str()
    ])
    print(out_str, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Model Trainer')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--attention_model', type=str, default=None,
                        help='Mental Attention model to use')
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
    parser.add_argument('--max_grad_norm', type=float, default=Consts.max_grad_norm,
                        help='maximum gradient norm')
    parser.add_argument('--bbox_type', type=str, default=None,
                        help='Type of bbox to use <pr_bbox>, <gt_bbox> or leave unassigned')
    parser.add_argument('--train_set_size', type=int, default=None,
                        help='size in % of the trainning set [1-70]')
    main(parser.parse_args())
