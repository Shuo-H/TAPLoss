import gc
import torch
import dataloader
from model import build_train_model, DEVICE
from tqdm.auto import tqdm




class Train:
    def __init__(self, train_config):
        self.epochs      = train_config['epochs']
        self.model       = build_train_model()
        self.dataloader  = dataloader.run(train_config['audio_paths'], train_config['batch_size'])
        self.criterion   = torch.nn.MSELoss()
        self.optimizer   = torch.optim.Adam(
            self.model.parameters(), lr = train_config['lr'], amsgrad= True, weight_decay= 5e-6
        )
        self.scaler      = torch.cuda.amp.GradScaler()
        
    def step(self):
        self.model.train()
        batch_bar  = tqdm(total = len(self.dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')
        train_loss = 0
        pred = []
        true = []
        for i, (x, y_true) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            x, y_true = x.to(DEVICE), y_true.to(DEVICE)
            with torch.cuda.amp.autocast():
                y_pred = self.model(x)
                loss = self.criterion(y_true, y_pred)
                train_loss += loss.item()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_bar.set_postfix(
                loss="{:.04f}".format(loss/(i+1)),
                lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr']))
                )
            batch_bar.update()
            del x, y_true
            torch.cuda.empty_cache()
        train_loss /= len(self.dataloader)
        batch_bar.close()
        return train_loss
    
    def val(self):
        self.model.eval()
        batch_bar = tqdm(total = len(self.dataloader), dynamic_ncols=True, leave=False, position=0, desc='Val')
        val_loss = 0
        for i, (x, y_true) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            x, y_true = x.to(DEVICE), y_true.to(DEVICE)
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.criterion(y_true, y_pred)
                val_loss += loss.item()
            batch_bar.set_postfix(loss="{:.04f}".format(loss/(i+1)))
            batch_bar.update()
            del x, y_true
            torch.cuda.empty_cache()
        val_loss /= len(self.dataloader)
        batch_bar.close()
        return val_loss
    
    def run(self):
        gc.collect()
        torch.cuda.empty_cache()
        for epoch in range(0, self.epochs):
            print("\nEpoch: {}/{}".format(epoch+1, self.epochs))
            train_loss = self.step()
            val_loss   = self.val()
            # scheduler.step(val_loss)
            print(f"train loss:{train_loss}  val loss:{val_loss}")
            # if best > val_loss:
            #     best = val_loss
            #     print('saving model')
            #     torch.save({'model_state_dict':model.state_dict(),
            #       'optimizer_state_dict':optimizer.state_dict(),
            #       'loss':val_loss,
            #       'epoch': epoch
            #        },
            #        './cp1.pth')