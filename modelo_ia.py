import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import warnings
warnings.filterwarnings("ignore")

def criar_dataloaders(caminho_csv: str, encoder_length: int, prediction_length: int, features: list):
    df = pd.read_csv(caminho_csv)

    # tudo em string bonitinho :D
    for col in ["Day_of_Week", "Month", "Ticker", "Setor"]:
        df[col] = df[col].astype(str)

    linha_de_corte = df["Time_Idx"].max() - 126  # 6 meses para teste
    df_treino = df[df["Time_Idx"] <= linha_de_corte]

    dataset_treino = TimeSeriesDataSet(
        df_treino,
        time_idx="Time_Idx",
        target="Log_Return",
        group_ids=["Ticker"],
        min_encoder_length=encoder_length // 2,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        static_categoricals=["Ticker", "Setor"],
        time_varying_known_categoricals=["Day_of_Week", "Month"],
        time_varying_unknown_reals=features,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataset_validacao = TimeSeriesDataSet.from_dataset(
        dataset_treino, df, predict=False, min_prediction_idx=linha_de_corte + 1, stop_randomization=True
    )

    train_dataloader = dataset_treino.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = dataset_validacao.to_dataloader(train=False, batch_size=64, num_workers=0)

    return dataset_treino, train_dataloader, val_dataloader


def instanciar_tft(dataset_treino, lr: float, hidden_size: int, attn_heads: int, dropout: float, hidden_continuous_size: int = 16):
    tft = TemporalFusionTransformer.from_dataset(
        dataset_treino,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attn_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return tft


def treinar_modelo(tft, train_dataloader, val_dataloader, epochs: int):

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,  #
        callbacks=[lr_logger, early_stop_callback],
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return trainer.checkpoint_callback.best_model_path