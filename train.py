import pytorch_lightning as pl
import pandas as pd
from model import MLPWithEmbeddings
from sklearn.model_selection import train_test_split
from dataloader import TabularDataset
import json
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import get_original_cwd, to_absolute_path
import os
from sklearn.metrics import accuracy_score
import torch
import glob
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


@hydra.main(config_path="hydra", config_name="config")
def main(cfg: DictConfig):
    data = pd.read_csv(
        os.path.join(get_original_cwd(), "data/LoanData_Numeric.csv"), index_col=0
    )
    embedding_files = glob.glob(os.path.join(get_original_cwd(), "data/*map.json"))
    embedding_dims = []
    embedding_names = []
    for file in embedding_files:
        with open(file, "r") as f:
            embedding_name = file.split("/")[-1].replace("_map.json", "")
            if embedding_name == "loan_status":
                loan_map = json.load(f)
                continue
            map = json.load(f)
            embedding_dims.append(len(map))
            embedding_names.append(embedding_name)
    X = data.drop(["loan_status"], axis=1)
    X_feature_cols = [col for col in X.columns if col not in embedding_names]
    X_embedding_cols = embedding_names

    y = data["loan_status"].values
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_features_train = X_train[X_feature_cols].values
    X_features_train[
        np.isnan(X_features_train)
    ] = 0  # I don't know what's happening here. X-train has no nan, grabbing values does.
    X_embeddings_train = X_train[X_embedding_cols].values
    X_features_valid = X_valid[X_feature_cols].values
    X_features_valid[
        np.isnan(X_features_valid)
    ] = 0  # I don't know what's happening here. X-train has no nan, grabbing values does.

    X_embeddings_valid = X_valid[X_embedding_cols].values

    train_dataset = TabularDataset(X_features_train, X_embeddings_train, y_train)
    valid_dataset = TabularDataset(X_features_valid, X_embeddings_valid, y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=1000)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1000)

    early_stop_callback = EarlyStopping(
        monitor="valid/loss", min_delta=0.00, patience=3, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(monitor="valid/loss")
    model = MLPWithEmbeddings(
        X_features_train.shape[1], embedding_dims, cfg, np.unique(y).shape[0]
    )
    mlf_logger = MLFlowLogger(
        experiment_name="LoanStatus",
        tracking_uri="file://" + get_original_cwd() + "/mlruns",
    )

    mlf_logger.log_hyperparams(OmegaConf.to_object(cfg))
    trainer = pl.Trainer(
        logger=mlf_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=5,
        gpus=1
    )
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader
    )

    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    pred_tuple = trainer.predict(best_model, valid_dataloader)
    preds = torch.cat([x[0] for x in pred_tuple])
    y = torch.cat([x[1] for x in pred_tuple])
    final_accuracy = accuracy_score(y, preds)
    mlf_logger.log_metrics({"final_valid_accuracy": final_accuracy})

    # save confusion matrix
    df_cm = pd.DataFrame(
        confusion_matrix(y, preds), index=list(loan_map.values())[:-1], columns=list(loan_map.values())[:-1]
    )
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
    if not os.path.exists(f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}"):
        os.mkdir(f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}")
    plt.savefig(f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}/confusion.png")

    trainer.logger.experiment.log_artifacts(
        mlf_logger.run_id, f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}"
    )

    if cfg.save_graph:
        if not os.path.exists(f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}"):
            os.mkdir(f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}")
        torch.save(
            model.state_dict(),
            f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}/MLPWithEmbeddings.pth",
        )
        trainer.logger.experiment.log_artifacts(
            mlf_logger.run_id, f"{get_original_cwd()}/mlruns/{mlf_logger.run_id}"
        )

    return final_accuracy


if __name__ == "__main__":
    main()
