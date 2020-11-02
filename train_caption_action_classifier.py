import argparse
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np

from src.utils.stopwatch import Stopwatch
from src.utils.model_trainer import ModelTrainer
from src.consts import Consts
from src.data.data_splitters import action_recognition_splits
from src.data.noun_grounding_data_loader import NounGroundingDataLoader
from src.models.noun_grounding_model import NoundGroundingModel

def compare_bboxes(y, gt):
    y = [a.cpu().data.numpy() for a in y]
    gt = [a.cpu().data.numpy() for a in gt]
    
    yb, yi = y
    gtb, gti = gt
    
    yb = yb * (1 - np.argmax(yi))
    gtb = gtb * gti[0]

    i = np.array([
        max(yb[0], gtb[0]), max(yb[1], gtb[1]),
        min(yb[0] + yb[2], gtb[0] + gtb[2]) - max(yb[0], gtb[0]),
        min(yb[1] + yb[3], gtb[1] + gtb[3]) - max(yb[1], gtb[1]),
    ])
    
    if (i[2] <= 0) or (i[3] <= 0):
        IoU = 0
    else:
        IoU = i[2]*i[3] / (yb[2]*yb[3] + gtb[2]*gtb[3] - i[2]*i[3])
        
    return [int((IoU > 0.5) and (np.argmax(yi) == 0)), int(gti[0])]

def main(args):
    def batch_prepper(batch):
        return batch[0], batch[1]

    def batch_parser(*args):
        y, gt  = args
        retval = [compare_bboxes(x[:2],x[2:]) for x in zip(y[0],y[1],gt[0], gt[1])]
        return [x for x in zip(*retval)]
        
    def epoch_parser(train_results, val_results):
        train_losses,   train_pairs = zip(*train_results)
        val_losses,     val_pairs   = zip(*val_results)
        train_pred, train_labels = zip(*train_pairs)
        val_pred, val_labels = zip(*val_pairs)
        train_acc = accuracy_score(sum(train_pred,()), sum(train_labels,()))
        val_acc = accuracy_score(sum(val_pred,()), sum(val_labels,()))
        return mean(train_losses), mean(val_losses), 100*train_acc, 100*val_acc
    
    files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list)
    data_loader = {}
    for key, val in files.items():
        data_loader[key] = NounGroundingDataLoader(val, batchsize=args.batch_size)

    model = NoundGroundingModel()

    if not args.pretrained_model is None:
        model.load_state_dict(torch.load(args.pretrained_model))

    mse = nn.MSELoss()
    criterion = lambda x, l: mse(x[0],l[0])/3 + 2*mse(x[1],l[1])/3

    trainer = ModelTrainer(
        model = model,
        criterion = criterion,
        optimizer = lambda model: optim.SGD (
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-4),
        batch_preproc=batch_prepper,
        batch_postproc=batch_parser,
        epoch_postproc=epoch_parser,
        device = args.device,
        max_grad_norm=2.0
    )

    timer = Stopwatch()
    pa = 0
    for epoch in range(args.num_epochs):
        tl, vl, ta, va = trainer(data_loader)
        out_str = '; '.join([
            f'Epoch: {epoch+1:5d}',
            f'Loss:({tl:8.5f},{vl:8.5f})',
            f'Acc: ({ta:5.1f}, {va:5.1f})',
            timer.lap_str()
        ])
        print(out_str, flush=True)
        if pa < va:
            if not args.mental_attention_model_path is None:
                torch.save(model.mental_attention.state_dict(), args.mental_attention_model_path)
            torch.save(model.state_dict(), args.out_model_path)
            pa = va


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language Model Trainer')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--out_model_path', type=str, default='grounding_model.torch',
                        help='path to where the trained classifier model should be saved')
    parser.add_argument('--mental_attention_model_path', type=str, default='mental_attention_model.torch',
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
