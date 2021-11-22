import numpy as np
import torch
import pytorch_lightning as pl

_default_workplace = '/content/drive/MyDrive/kaggle/codes/ventilator_pressure_prediction/'
class SequentialTrain(pl.LightningModule):
    def __init__(
                self,
                model,
                criterion,
                optimizers,
                train_loader,
                valid_loader,
                schedulers = None,
                eval_func = None,                
                out = f'{_default_workplace}logs/'):

        super().__init__()
        self.save_hyperparameters()

        self._model = model
        self._criterion = criterion
        self._eval_func = eval_func

        self._train_loader = train_loader
        self._valid_loader = valid_loader

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        if not isinstance(schedulers, list):
            schedulers = [] if schedulers is None else [schedulers]

        self._optimizers = optimizers
        self._schedulers = schedulers

        self._out = out

    def _sqdtnp(self, x):
        if self.on_gpu:
            x = x.cpu()
        x = torch.squeeze(x).detach().numpy()

        return x
    
    def forward(self, x):
        return self._model(x)
    
    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._valid_loader

    def configure_optimizers(self):
        return self._optimizers, self._schedulers

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self._criterion(y, t)
        
        self.log('training_loss', loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self._criterion(y, t)

        res = {'val_loss': loss}

        if self._eval_func is not None:
            y = self._sqdtnp(y.cpu())
            t = self._sqdtnp(t.cpu())

            val_acc = self._eval_func(y, t)
            res['val_acc'] = val_acc
        
        return res

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('avg_train_loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('avg_val_loss', avg_loss)

        if 'val_acc' in outputs[0]:
            avg_acc = np.mean([x['val_acc'] for x in outputs])
            print(f'avg_acc is {avg_acc}')
            self.log('avg_val_acc', avg_acc)
