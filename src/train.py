import pytorch_lightning as pl
from torchvision import transforms as T
from config import Config
from trainner import WireModel
from utils import load_checkpoint, save_model

def train_model(config: Config, train_loader, val_loader, test_loader):
    model = WireModel(
        config.architecture,
        config.encoder_type,
        in_channels=config.in_channels,
        out_classes=config.out_classes,
        tmax=config.epoch * len(train_loader),
        pipeline_name=config.pipeline_name
    )
    if config.load_checkpoint:
        load_checkpoint(config.load_model_path, model.model)
        print(10*'=','> checkpoint loaded')
        
    trainer = pl.Trainer(max_epochs=config.epoch, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    valid_metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)
    print(valid_metrics)
    model.save_metrics()
    # save_model(model.model)
    save_model(model.model, model_name=f"cable_seg_{config.architecture}", suffix=f"epoch{config.epoch:03d}")