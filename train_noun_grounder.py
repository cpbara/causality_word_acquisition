import argparse
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np
import os

from src import Consts
from src.utils import Stopwatch
from src.utils import ModelTrainer
from src.data import action_recognition_splits_2 as splits_fun
from src.data import NounGroundingDataLoader as Loader
from src.models import NoundGroundingModel as Grounder

def BBoxIoU(a,b):
    i = np.array([
        max(a[0], b[0]), 
        max(a[1], b[1]),
        min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]),
        min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]),
    ])
    if (i[2] <= 0) or (i[3] <= 0):
        return 0
    else:
        return i[2]*i[3] / (a[2]*a[3] + b[2]*b[3] - i[2]*i[3])

def compare_bboxes(y, gt):    
    yb = y.cpu().data.numpy()
    gtb = gt.cpu().data.numpy()
    IoU = BBoxIoU(yb,gtb)
    return [int((IoU > 0.5)), 1, IoU]

def main(args):
    def batch_prepper(batch):
        return batch[-2], batch[-1]

    def batch_parser(*args):
        print(args[:-3], end='\r')
        y, gt  = args[-2:]
        retval = [compare_bboxes(x[0],x[1]) for x in zip(y,gt)]
        return args[2], [x for x in zip(*retval)]
        
    def epoch_parser(train_results, val_results):
        train_losses,   train_pairs = zip(*train_results)
        val_losses,     val_pairs   = zip(*val_results)
        train_pred, train_labels, train_IoU = zip(*train_pairs)
        val_pred, val_labels, val_IoU = zip(*val_pairs)
        train_IoU = sum(train_IoU,())
        train_IoU = sum(train_IoU)/len(train_IoU)
        val_IoU = sum(val_IoU,())
        val_IoU = sum(val_IoU)/len(val_IoU)
        train_acc = accuracy_score(sum(train_pred,()), sum(train_labels,()))
        val_acc = accuracy_score(sum(val_pred,()), sum(val_labels,()))
        train_acc2 = accuracy_score(*zip(*[(x,y) for x,y in zip(sum(train_pred,()), sum(train_labels,())) if y == 1]))
        val_acc2 = accuracy_score(*zip(*[(x,y) for x,y in zip(sum(val_pred,()), sum(val_labels,())) if y == 1]))
        return mean(train_losses), mean(val_losses), 100*train_acc, 100*val_acc, 100*train_acc2, 100*val_acc2, train_IoU, val_IoU

    files = splits_fun(args.data_root_path, args.train_list, args.test_list)
    data_loader = {}
    for key, val in files.items():
        data_loader[key] = Loader(val, batchsize=args.batch_size,mental_attention_sufix=args.ma_emb_sufix)

    model = Grounder()
    if not args.pretrained_model is None:
        model.load_state_dict(torch.load(args.pretrained_model))

    criterion = nn.MSELoss()
    trainer = ModelTrainer(
        model = model,
        criterion = criterion,
        optimizer = lambda model: optim.Adam(
            model.parameters(), 
            lr=args.learning_rate),
        batch_preproc=batch_prepper,
        batch_postproc=batch_parser,
        epoch_postproc=epoch_parser,
        device = args.device,
        max_grad_norm=2.0
    )
    timer = Stopwatch()
    pa = 0
    for epoch in range(args.num_epochs):
        tl, vl, ta, va, ta2, va2, tiou, viou = trainer(data_loader)
        if pa < va:
            torch.save(model.state_dict(), args.out_model_path)
            pa = va
            
        out_str = '; '.join([
            f'PID: {os.getpid():6d}',
            f'Epoch: {epoch+1:5d}',
            f'Loss:({tl:8.5f},{vl:8.5f})',
            f'Acc: ({ta:5.1f}, {va:5.1f})',
            f'Acc: ({ta2:5.1f}, {va2:5.1f})',
            f'IoU: ({tiou:5.3f}, {viou:5.3f})',
            timer.lap_str()
        ])
        print(out_str, flush=True)


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
    parser.add_argument('--ignore_captions', type=bool, default=False,
                        help='Chose to ingnore captions')
    parser.add_argument('--ma_emb_sufix', type=str, default=None,
                        help='Type of Mental Attention sufix to use, e.g. <MentAttFromNounGrounding> or leave unassigned')
    main(parser.parse_args())



# import argparse
# import torch
# from torch import nn, optim
# from sklearn.metrics import accuracy_score
# from statistics import mean
# import numpy as np

# from src.utils.stopwatch import Stopwatch
# from src.utils.model_trainer import ModelTrainer
# from src.consts import Consts
# from src.data.data_splitters import action_recognition_splits
# from src.data.noun_grounding_data_loader import NounGroundingDataLoader
# from src.models.noun_grounding_model import NoundGroundingModel

# def compare_bboxes(y, gt):
#     y = [a.cpu().data.numpy() for a in y]
#     gt = [a.cpu().data.numpy() for a in gt]
    
#     yb, yi = y
#     gtb, gti = gt
    
#     yb = yb * (1 - np.argmax(yi))
#     gtb = gtb * gti[0]

#     i = np.array([
#         max(yb[0], gtb[0]), max(yb[1], gtb[1]),
#         min(yb[0] + yb[2], gtb[0] + gtb[2]) - max(yb[0], gtb[0]),
#         min(yb[1] + yb[3], gtb[1] + gtb[3]) - max(yb[1], gtb[1]),
#     ])
    
#     if (np.argmax(yi) == 0) and (gti[0] == 1):
#         IoU = 1
#     else:
#         if (i[2] <= 0) or (i[3] <= 0):
#             IoU = 0
#         else:
#             IoU = i[2]*i[3] / (yb[2]*yb[3] + gtb[2]*gtb[3] - i[2]*i[3])
        
#     return [int((IoU > 0.5) and (np.argmax(yi) == 0)), int(gti[0])]
#     # return [int(np.argmax(yi) == 0), int(gti[0])]

# def main(args):
#     def batch_prepper(batch):
#         return batch[0], batch[1]

#     def batch_parser(*args):
#         y, gt  = args[-2:]
#         retval = [compare_bboxes(x[:2],x[2:]) for x in zip(y[0],y[1],gt[0], gt[1])]
#         return args[2], [x for x in zip(*retval)]
        
#     def epoch_parser(train_results, val_results):
#         train_losses,   train_pairs = zip(*train_results)
#         val_losses,     val_pairs   = zip(*val_results)
#         train_pred, train_labels = zip(*train_pairs)
#         val_pred, val_labels = zip(*val_pairs)
#         train_acc = accuracy_score(sum(train_pred,()), sum(train_labels,()))
#         val_acc = accuracy_score(sum(val_pred,()), sum(val_labels,()))
#         return mean(train_losses), mean(val_losses), 100*train_acc, 100*val_acc
    
#     files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list)
#     data_loader = {}
#     for key, val in files.items():
#         data_loader[key] = NounGroundingDataLoader(val, batchsize=args.batch_size)

#     model = NoundGroundingModel()

#     if not args.pretrained_model is None:
#         model.load_state_dict(torch.load(args.pretrained_model))

#     mse = nn.MSELoss()
#     criterion = lambda x, l: mse(x[0],l[0])/3 + 2*mse(x[1],l[1])/3
#     # criterion = lambda x, l: mse(x[1],l[1])

#     trainer = ModelTrainer(
#         model = model,
#         criterion = criterion,
#         optimizer = lambda model: optim.SGD (
#             model.parameters(), 
#             lr=args.learning_rate, 
#             momentum=0.9, 
#             weight_decay=1e-4),
#         batch_preproc=batch_prepper,
#         batch_postproc=batch_parser,
#         epoch_postproc=epoch_parser,
#         device = args.device,
#         max_grad_norm=2.0
#     )

#     timer = Stopwatch()
#     pa = 0
#     for epoch in range(args.num_epochs):
#         tl, vl, ta, va = trainer(data_loader)
#         out_str = '; '.join([
#             f'Epoch: {epoch+1:5d}',
#             f'Loss:({tl:8.5f},{vl:8.5f})',
#             f'Acc: ({ta:5.1f}, {va:5.1f})',
#             timer.lap_str()
#         ])
#         print(out_str, flush=True)
#         if pa < va:
#             if not args.mental_attention_model_path is None:
#                 torch.save(model.mental_attention.state_dict(), args.mental_attention_model_path)
#             torch.save(model.state_dict(), args.out_model_path)
#             pa = va


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Language Model Trainer')
#     parser.add_argument('--pretrained_model', type=str, default=None,
#                         help='path to pretrained model')
#     parser.add_argument('--out_model_path', type=str, default='grounding_model.torch',
#                         help='path to where the trained classifier model should be saved')
#     parser.add_argument('--mental_attention_model_path', type=str, default='mental_attention_model.torch',
#                         help='path to where the trained language model should be saved')
#     parser.add_argument('--learning_rate', type=float, default=Consts.learning_rate,
#                         help='optimizer learning rate')
#     parser.add_argument('--num_epochs', type=int, default=Consts.num_epochs,
#                         help='number of epochs to train')
#     parser.add_argument('--batch_size', type=int, default=Consts.batch_size,
#                         help='size of the mini batch')
#     parser.add_argument('--train_list', type=str, default=Consts.Splits.ActionRecognition.train_list,
#                         help='path to training instance list file')
#     parser.add_argument('--test_list', type=str, default=Consts.Splits.ActionRecognition.test_list,
#                         help='path to training instance list file')
#     parser.add_argument('--data_root_path', type=str, default=Consts.data_root_path,
#                         help='path to training instance list file')
#     parser.add_argument('--device', type=str, default=Consts.DEVICE,
#                         help='device to use cuda or cpu')
#     main(parser.parse_args())
