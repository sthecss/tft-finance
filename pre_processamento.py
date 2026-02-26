import os
import pandas as pd
import numpy as np
import yfinance as yf

def normalizar_features(df: pd.DataFrame) -> pd.DataFrame:
    colunas_normalizar = [
        "Volume", "RSI_14", "USD_Close", "Volatility_21", "USD_Volatility_21"
    ]

    colunas_presentes = [c for c in colunas_normalizar if c in df.columns]

    for col in colunas_presentes:
        media  = df.groupby("Ticker")[col].transform("mean")
        std    = df.groupby("Ticker")[col].transform("std")
        df[col] = (df[col] - media) / (std + 1e-8)

    print(f"[NORMALIZAÇÃO] Z-score aplicado em: {colunas_presentes}")
    return df

def _processar_ticker(ticker: str, setor: str, data_inicio: str, data_fim: str) -> pd.DataFrame | None:
    """Baixa e processa um único ticker. Uso interno."""
    ativo = yf.Ticker(ticker)
    df = ativo.history(start=data_inicio, end=data_fim)

    if df.empty:
        print(f"[AVISO] Nenhum dado encontrado para {ticker}. Pulando.")
        return None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['Open', 'High', 'Low', 'Adj_Close', 'Volume']
    df.index = df.index.tz_localize(None)

    # Estacionariedade
    df['Log_Return'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1))
    df['Volume']     = np.log1p(df['Volume'])

    df['Open'] = df['Open'] / df['Adj_Close']
    df['High'] = df['High'] / df['Adj_Close']
    df['Low']  = df['Low']  / df['Adj_Close']

    # SMAs como ratio (já normalizados em torno de 1.0)
    df['SMA_14'] = df['Adj_Close'].rolling(14).mean()
    df['SMA_50'] = df['Adj_Close'].rolling(50).mean()
    df['SMA_14'] = df['Adj_Close'] / df['SMA_14']
    df['SMA_50'] = df['Adj_Close'] / df['SMA_50']

    # Volatilidade
    df['Volatility_21'] = df['Log_Return'].rolling(window=21).std()

    # RSI
    delta        = df['Adj_Close'].diff()
    ganho        = delta.where(delta > 0, 0.0).rolling(14).mean()
    perda        = -delta.where(delta < 0, 0.0).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (ganho / perda)))

    # Calendário
    df['Day_of_Week'] = df.index.dayofweek.astype(str)
    df['Month']       = df.index.month.astype(str)

    # Metadados
    df['Ticker'] = ticker
    df['Setor']  = setor

    df = df.dropna().reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Time_Idx'] = np.arange(len(df))

    return df


# ==============================================================================
# DOWNLOAD — MÚLTIPLOS TICKERS
# ==============================================================================

def baixar_dados(tickers: list, setores: list, data_inicio: str, data_fim: str,
                 nome_saida: str = "dataset_tft.csv") -> str | None:
    # ── Verificação: arquivo já existe? ──────────────────────────────────────
    if os.path.exists(nome_saida):
        print(f"[DADOS] Dataset '{nome_saida}' já existe. Pulando download.")
        print(f"        → Delete o arquivo para forçar novo download.")
        return nome_saida

    if len(tickers) != len(setores):
        print("[ERRO] As listas 'tickers' e 'setores' precisam ter o mesmo tamanho.")
        return None

    frames = []
    for ticker, setor in zip(tickers, setores):
        print(f"[DADOS] Baixando {ticker}...")
        df = _processar_ticker(ticker, setor, data_inicio, data_fim)
        if df is not None:
            frames.append(df)

    if not frames:
        print("[ERRO] Nenhum ticker retornou dados.")
        return None

    df_final = pd.concat(frames, ignore_index=True)

    # Time_Idx sequencial por Ticker
    df_final = df_final.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df_final["Time_Idx"] = df_final.groupby("Ticker").cumcount()

    df_final.to_csv(nome_saida, index=False)
    print(f"[DADOS] Concluído! {len(frames)} ticker(s) → '{nome_saida}' ({len(df_final)} linhas).")
    return nome_saida


# ==============================================================================
# DÓLAR
# ==============================================================================

def adicionar_dolar(caminho_csv: str, data_inicio: str, data_fim: str) -> str:
    """
    Baixa a cotação diária do dólar (USDBRL=X) e adiciona como features
    em cada linha do CSV, alinhando pela data.

    Colunas adicionadas: USD_Close, USD_Log_Return, USD_Volatility_21

    Se as colunas já existirem no CSV, pula o processo.
    Retorna o mesmo caminho_csv (arquivo sobrescrito).
    """
    df = pd.read_csv(caminho_csv, parse_dates=['Date'])

    # ── Verificação: dólar já foi adicionado? ────────────────────────────────
    if 'USD_Close' in df.columns:
        print("[DÓLAR] Features USD já existem no dataset. Pulando.")
        return caminho_csv

    print("[DÓLAR] Baixando USDBRL=X...")
    usd    = yf.Ticker("USDBRL=X")
    df_usd = usd.history(start=data_inicio, end=data_fim)

    if df_usd.empty:
        print("[AVISO] Não foi possível baixar o dólar. Pulando.")
        return caminho_csv

    df_usd.index = df_usd.index.tz_localize(None)
    df_usd = df_usd[['Close']].copy()
    df_usd.columns = ['USD_Close']
    df_usd['USD_Log_Return']    = np.log(df_usd['USD_Close'] / df_usd['USD_Close'].shift(1))
    df_usd['USD_Volatility_21'] = df_usd['USD_Log_Return'].rolling(21).std()
    df_usd = df_usd.dropna().reset_index()
    df_usd.rename(columns={'index': 'Date'}, inplace=True)
    df_usd['Date'] = pd.to_datetime(df_usd['Date']).dt.normalize()

    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.merge(
        df_usd[['Date', 'USD_Close', 'USD_Log_Return', 'USD_Volatility_21']],
        on='Date', how='left'
    )

    # Forward fill para dias sem cotação (feriados EUA)
    df[['USD_Close', 'USD_Log_Return', 'USD_Volatility_21']] = (
        df[['USD_Close', 'USD_Log_Return', 'USD_Volatility_21']].ffill()
    )
    df = df.dropna(subset=['USD_Close', 'USD_Log_Return', 'USD_Volatility_21'])

    # Recalcula Time_Idx após possível perda de linhas
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Time_Idx"] = df.groupby("Ticker").cumcount()

    df.to_csv(caminho_csv, index=False)
    print(f"[DÓLAR] Features USD adicionadas → '{caminho_csv}' atualizado.")
    return caminho_csv