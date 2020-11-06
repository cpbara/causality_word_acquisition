import torch
from transformers import BertModel, BertConfig

class Consts:
    class Splits:
        class ActionRecognition:
            train_list = '/local/cpbara/actions-transitions/labels/task1/trainlist.txt'
            test_list  = '/local/cpbara/actions-transitions/labels/task1/testlist.txt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_bert = lambda : BertModel.from_pretrained('bert-base-uncased')
    untrained_bert  = lambda : BertModel(BertConfig())
    data_root_path = '/local/cpbara/actions-transitions/action_frames_release/'
    learning_rate = 1e-3
    num_epochs = 1000
    batch_size = 32
    max_grad_norm = 1.0
