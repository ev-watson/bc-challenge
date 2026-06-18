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
        self.embeddings = nn.Embedding(256, config.EMBEDDING_DIM)

        self.dropouts = nn.ModuleList([
            nn.Dropout(config.DROPOUT_RATE) for _ in range(config.NUM_HIDDEN_LAYERS)
        ])

        self.input_norm = nn.LayerNorm(config.INPUT_DIM)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM) for _ in range(config.NUM_HIDDEN_LAYERS)
        ])

    def forward(self, x):
        # [b, 64]
        e = self.embeddings(x)  # [b, 64, embedding_dim]
        
        # various pools for feature extraction
        feats = [e.mean(dim=1), e.max(dim=1).values, e.std(dim=1)]  # [b, 3 * embedding_dim]

        # modulates over sequence to get POOL_MOD distinct pools that encode positional information
        for r in range(config.POOL_MOD):
            feats.append(e[:, r::config.POOL_MOD, :].mean(dim=1))   # [b, embedding_dim]

        x = torch.cat(feats, dim=1)  # [b, input_dim]
        x = self.input_norm(x)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  factor=0.1,
                                                                  patience=config.PATIENCE - 2)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': 'val_loss'}}