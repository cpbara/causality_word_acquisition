from torch import nn
import torch

class ModelTrainer:
    def __init__(   self, 
                    model, 
                    criterion, 
                    optimizer, 
                    batch_preproc = None, 
                    batch_postproc = None, 
                    epoch_postproc = None, 
                    device = 'cpu',
                    max_grad_norm = 1.0
                    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model)
        if batch_preproc is None:
            self.prep_batch = lambda *args: args if len(args) > 1 else args[0]
        else:
            self.prep_batch = batch_preproc
        if batch_postproc is None:
            self.parse_batch = lambda *args: args if len(args) > 1 else args[0]
        else:
            self.parse_batch = batch_postproc
        if epoch_postproc is None:
            self.parse_epoch = lambda *args: args if len(args) > 1 else args[0]
        else:
            self.parse_epoch = epoch_postproc
        self.device = device
        self.max_grad_norm = max_grad_norm
        
    def forward(self,batch):
        self.model.to(self.device)
        data, labels = self.prep_batch(batch)
        prediction = self.model(data.to(self.device))
        loss = self.criterion(prediction, labels.to(self.device))
        return loss, prediction, labels
    
    def train(self,i,batch,n):
        self.model.train()
        self.optimizer.zero_grad()
        loss, prediction, labels = self.forward(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return self.parse_batch(i,n,loss.item(), prediction, labels)
    
    def validate(self,i,batch,n):
        self.model.eval()
        loss, prediction, labels = self.forward(batch)
        return self.parse_batch(i,n,loss.item(), prediction, labels)
    
    def __call__(self, data_loader):
        fun = lambda f,l: [f(i,b,len(l)) for i,b in enumerate(l)]
        train_res = fun(self.train,data_loader['training'])
        torch.cuda.empty_cache()
        val_res   = fun(self.validate,data_loader['validation'])
        torch.cuda.empty_cache()
        return self.parse_epoch(train_res, val_res)
    
    def test(self, data_loader):
        fun = lambda f,l: [f(i,b,len(l)) for i,b in enumerate(l)]
        val_res   = fun(self.validate,data_loader['validation'])
        torch.cuda.empty_cache()
        test_res  = fun(self.validate,data_loader['testing'])
        torch.cuda.empty_cache()
        return self.parse_epoch(val_res, test_res)
