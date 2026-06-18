import torch
import torch.nn.functional as F
import torch.optim
from lightning.pytorch import LightningModule
from torch import nn

import config


class BinaryClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM) for _ in range(config.NUM_HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)

        # bytes live in 0-255, so num_embeddings=256
        self.embeddings = nn.Embedding(256, config.INPUT_DIM)

        self.dropouts = nn.ModuleList([
            nn.Dropout(config.DROPOUT_RATE) for _ in range(config.NUM_HIDDEN_LAYERS)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM) for _ in range(config.NUM_HIDDEN_LAYERS)
        ])

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(dim=1)  # mean pool for histogram
        x = self.input_layer(x)
        x = F.relu(x)
        for layer, dropout, layer_norm in zip(self.hidden_layers, self.dropouts, self.layer_norms):
            x = layer_norm(x)
            x = F.relu(layer(x))
            x = dropout(x)
        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)