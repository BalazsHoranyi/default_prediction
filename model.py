import pytorch_lightning as pl
from torch import nn
import torch
from sklearn.metrics import accuracy_score
from omegaconf import DictConfig


class MLPWithEmbeddings(pl.LightningModule):
    def __init__(
        self,
        input_feature_shape,
        embeddings_dims: int,
        cfg: DictConfig,
        num_classes: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.embeddings = nn.ModuleList()
        for embeddings_dim in embeddings_dims:
            self.embeddings.append(nn.Embedding(embeddings_dim, cfg.embedding_dim))
        self.dropout = nn.Dropout(cfg.dropout)
        self.dense_layers = nn.ModuleList()
        input_size = input_feature_shape + (cfg.embedding_dim * len(embeddings_dims))
        for _ in range(cfg.n_layers):
            self.dense_layers.append(nn.Linear(input_size, input_size))

        self.output_layer = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x_features, x_embeddings):
        # last row of x is embedding
        embeddings = []
        for i in range(len(self.embeddings)):
            embeddings.append(self.embeddings[i](x_embeddings[:, i].long()))
        embeddings = torch.cat(embeddings, axis=1)
        x = torch.cat([x_features, embeddings], axis=1).float()
        for layer in self.dense_layers:
            x = layer(x)
            x = self.dropout(x)
        x = self.output_layer(x)

        return x

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        x_features, x_embeddings, y = batch

        logits = self.forward(x_features, x_embeddings)
        loss = self.loss(logits, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_features, x_embeddings, y = batch
        logits = self.forward(x_features, x_embeddings)

        loss = self.loss(logits, y)
        self.log("valid/loss", loss)
        preds = torch.softmax(logits, 1).argmax(1)
        self.log("valid/accuracy", accuracy_score(y.cpu(), preds.cpu()))
        return loss

    def predict_step(self, batch, batch_idx):
        x_features, x_embeddings, y = batch
        logits = self.forward(x_features, x_embeddings)
        preds = torch.softmax(logits, 1).argmax(1)
        return preds.cpu(), y.cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
