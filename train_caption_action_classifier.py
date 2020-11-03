import argparse
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from src.utils.stopwatch import Stopwatch
from src.data.caption_loader import CaptionDataLoader
from src.models.caption_action_classifier_model import CaptionActionClassifier
from src.utils.model_trainer import ModelTrainer
from src.consts import Consts
from src.data.data_splitters import action_recognition_splits
from statistics import mean

def main(args):    
    def batch_prepper(batch):
        return batch[0], torch.stack([torch.eye(43)[l] for l in batch[1]])

    def batch_parser(*args):
        return [list(torch.argmax(x,dim=-1).cpu().data.numpy()) for x in args]
        
    def epoch_parser(train_results, val_results):
        train_losses,   train_pairs = zip(*train_results)
        val_losses,     val_pairs   = zip(*val_results)
        train_pred, train_labels = zip(*train_pairs)
        val_pred, val_labels = zip(*val_pairs)
        train_acc = accuracy_score(sum(train_pred,[]), sum(train_labels,[]))
        val_acc = accuracy_score(sum(val_pred,[]), sum(val_labels,[]))
        return mean(train_losses), mean(val_losses), 100*train_acc, 100*val_acc
    
    files = action_recognition_splits(args.data_root_path, args.train_list, args.test_list)
    data_loader = {}
    for key, val in files.items():
        data_loader[key] = CaptionDataLoader(val, batchsize=args.batch_size)

    model = CaptionActionClassifier(Consts.pretrained_bert()).to(args.device)

    if not args.pretrained_model is None:
        model.load_state_dict(torch.load(args.pretrained_model))

    trainer = ModelTrainer(
        model = model,
        criterion = nn.MSELoss(),
        optimizer = lambda model: optim.SGD (
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-4),
        batch_preproc=batch_prepper,
        batch_postproc=batch_parser,
        epoch_postproc=epoch_parser,
        device = args.device
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
        print(out_str)
        if pa < va:
            if not args.language_model_path is None:
                torch.save(model.transformer.state_dict(), args.language_model_path)
            torch.save(model.state_dict(), args.out_model_path)
            pa = va


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
