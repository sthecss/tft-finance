from graficos import *
from pre_processamento import *
from modelo_ia import *
from pytorch_forecasting import TemporalFusionTransformer

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ============================
# --- CONFIGURAÇÕES GERAIS ---
# ============================

# Ações
TICKERS = [
    'ITUB3.SA',  # Itaú
    'ITUB4.SA',  # Itaú
    'BBAS3.SA',  # Banco do Brasil
    'BBDC3.SA',  # Bradesco
    'BBDC4.SA',  # Bradesco
    'SANB11.SA', # Santander
    'BPAC11',    # Banco Pactual
    # 'AXIA3.SA',  # Eletrobras
    # 'CPFE3.SA',  # CPFL
    # 'ENGI3.SA',  # Energisa
    # 'CMIG4.SA',   # Cemig
    # 'SBSP3.SA',  # Sabesp
    # 'CSMG3.SA',  # Copasa
    # 'SAPR4.SA'   # Sanepar
]

# Setores
SETORES = [
    'Setor Financeiro',
    'Setor Financeiro',
    'Setor Financeiro',
    'Setor Financeiro',
    'Setor Financeiro',
    'Setor Financeiro',
    'Setor Financeiro',
    # 'Energia Elétrica',
    # 'Energia Elétrica',
    # 'Energia Elétrica',
    # 'Energia Elétrica',
    # 'Saneamento',
    # 'Saneamento',
    # 'Saneamento'
]


TICKER_PRINCIPAL = "ITUB4.SA" # Ação a ser prevista
NOME_CSV = "dataset_tft.csv"

DATA_INICIO = "2022-01-01"
DATA_FIM    = "2025-12-31"
FEATURES     = ["Volume", "Log_Return", "SMA_14", "SMA_50", "Volatility_21", "RSI_14"]

ENCODER_LENGTH    = 30
PREDICTION_LENGTH = 5
EPOCHS            = 50
LR                = 0.001

# ============================
# --- EXPERIMENTOS -----------
# ============================
experiments = {
    'EXP_01': {'hidden_size':  16, 'attention_head_size': 1, 'dropout': 0.1, 'hidden_continuous_size':   8},
    'EXP_02': {'hidden_size':  32, 'attention_head_size': 2, 'dropout': 0.1, 'hidden_continuous_size':  16},
    'EXP_03': {'hidden_size':  64, 'attention_head_size': 4, 'dropout': 0.1, 'hidden_continuous_size':  32},
    'EXP_04': {'hidden_size':  64, 'attention_head_size': 4, 'dropout': 0.3, 'hidden_continuous_size':  32},
    'EXP_05': {'hidden_size': 128, 'attention_head_size': 4, 'dropout': 0.1, 'hidden_continuous_size':  64},
    'EXP_06': {'hidden_size': 128, 'attention_head_size': 8, 'dropout': 0.1, 'hidden_continuous_size':  64},
    'EXP_07': {'hidden_size': 128, 'attention_head_size': 8, 'dropout': 0.3, 'hidden_continuous_size':  64},
    'EXP_08': {'hidden_size': 256, 'attention_head_size': 8, 'dropout': 0.3, 'hidden_continuous_size': 128},
}


# ============================
# --- FUNÇÕES AUXILIARES -----
# ============================
def calcular_assertividade(modelo, val_loader) -> float:
    """Retorna % de acerto de direção (subiu/desceu) no conjunto de validação."""
    raw          = modelo.predict(val_loader, mode="raw", return_x=True)
    output       = raw.output[0]
    decoder_tgt  = raw.x["decoder_target"]

    reais     = decoder_tgt.cpu().numpy().flatten()
    previstos = output[:, :, 3].cpu().numpy().flatten()   # quantil 0.5

    return float((np.sign(reais) == np.sign(previstos)).mean()) * 100

# ============================
# --- PIPELINE PRINCIPAL -----
# ============================

def principal():
    print("=" * 60)
    print(" PIPELINE DE EXPERIMENTOS TFT")
    print("=" * 60)

    # 1. Dados (pula se CSV já existe)
    arquivo_csv = baixar_dados(
        tickers=TICKERS,
        setores=SETORES,
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM,
        nome_saida=NOME_CSV
    )
    if not arquivo_csv:
        print("Falha ao gerar arquivo de dados.")
        return

    # 2. Dólar (pula se já existe no CSV)
    arquivo_csv = adicionar_dolar(
        caminho_csv=arquivo_csv,
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM
    )

    # 3. Normalização z-score
    df = pd.read_csv(arquivo_csv)
    df = normalizar_features(df)
    df.to_csv(arquivo_csv, index=False)

    # 4. Dataloaders (únicos para todos os experimentos)
    dataset_treino, train_loader, val_loader = criar_dataloaders(
        caminho_csv=arquivo_csv,
        encoder_length=ENCODER_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        features=FEATURES
    )

    resultados = {}

    # 5. Loop de experimentos
    for nome_exp, cfg in experiments.items():
        print(f"\n{'='*60}")
        print(f"  {nome_exp}  |  {cfg}")
        print(f"{'='*60}")

        tft = instanciar_tft(
            dataset_treino=dataset_treino,
            lr=LR,
            hidden_size=cfg["hidden_size"],
            attn_heads=cfg["attention_head_size"],
            dropout=cfg["dropout"],
            hidden_continuous_size=cfg["hidden_continuous_size"],
        )

        caminho_ckpt = treinar_modelo(tft, train_loader, val_loader, epochs=EPOCHS)

        if not os.path.exists(caminho_ckpt):
            print(f"[ERRO] Checkpoint não encontrado para {nome_exp}. Pulando.")
            resultados[nome_exp] = {"assertividade": 0.0, "checkpoint": None}
            continue

        modelo_avaliado = TemporalFusionTransformer.load_from_checkpoint(caminho_ckpt)
        modelo_avaliado.eval()

        assertividade = calcular_assertividade(modelo_avaliado, val_loader)
        resultados[nome_exp] = {"assertividade": assertividade, "checkpoint": caminho_ckpt}

        print(f"[{nome_exp}] Assertividade: {assertividade:.1f}%")

    # 6. Gráficos finais
    print("\n\nGerando gráficos finais...")

    # Barra de assertividade + tabela
    grafico_assertividade(experiments, resultados)

    # Um gráfico individual por experimento + painel comparativo
    grafico_todos_experimentos(
        experimentos_cfg=experiments,
        resultados=resultados,
        val_loader=val_loader,
        caminho_csv=arquivo_csv,
        ticker=TICKER_PRINCIPAL,
    )


if __name__ == "__main__":
    principal()