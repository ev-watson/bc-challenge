import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar, EarlyStopping

import config
from data_init import BCDataModule
from model import BinaryClassifier
from mlb_requests import Server

datamodule = BCDataModule()
model = BinaryClassifier()

freq = 5
log_steps = max(1, int(config.SPLIT * config.NUM_SAMPLES / config.BATCH_SIZE / freq))

trainer = Trainer(
    max_epochs=config.MAX_EPOCHS,
    callbacks=[
        LearningRateMonitor(logging_interval='epoch'),
        TQDMProgressBar(refresh_rate=log_steps),
        EarlyStopping(monitor="val_loss", patience=config.PATIENCE, mode="min")],
    log_every_n_steps=log_steps,
    accelerator='auto',
    devices='auto',
    strategy='auto',
    sync_batchnorm=False,
    precision='32-true',
)

print("TRAINING...")
trainer.fit(model, datamodule=datamodule)

print("TRAINING COMPLETE, SAVING MODEL...")
model.eval()
torch.save(model.state_dict(), config.MODEL_PATH)

print("TESTING...")
trainer.test(model, datamodule=datamodule)

print("DONE")


