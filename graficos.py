import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pytorch_forecasting import TemporalFusionTransformer


# ==============================================================================
# GRÁFICO: Preço Real vs Previsto — lógica exata do código de referência
# ==============================================================================

def grafico_real_vs_previsto(modelo, val_loader, caminho_csv: str, ticker: str,
                              nome_arquivo: str = "real_vs_previsto.png"):
    """
    Gera e salva o gráfico Preço Real vs Previsto para um modelo/ticker.
    Segue exatamente a lógica do script de referência.
    """
    df = pd.read_csv(caminho_csv, parse_dates=["Date"])

    raw         = modelo.predict(val_loader, mode="raw", return_x=True)
    output      = raw.output[0]
    decoder_tgt = raw.x["decoder_target"]
    time_idx    = raw.x["decoder_time_idx"]

    rows = []
    for i in range(output.shape[0]):
        for h in range(output.shape[1]):
            rows.append({
                "time_idx": int(time_idx[i, h].cpu().item()),
                "real":     float(decoder_tgt[i, h].cpu().item()),
                "p50":      float(output[i, h, 3].cpu().item()),
            })

    df_prev = (pd.DataFrame(rows)
                 .groupby("time_idx")[["real", "p50"]]
                 .mean()
                 .reset_index()
                 .sort_values("time_idx"))

    datas = df[df["Ticker"] == ticker][["Time_Idx", "Date", "Adj_Close"]].rename(
        columns={"Time_Idx": "time_idx"}
    )
    df_prev = df_prev.merge(datas, on="time_idx", how="left").dropna(subset=["Date"])

    idx_base   = df_prev["time_idx"].min() - 1
    preco_base = df[df["Time_Idx"] == idx_base]["Adj_Close"].values[0]

    df_prev["preco_previsto"] = preco_base * np.exp(df_prev["p50"].cumsum())

    plt.figure(figsize=(12, 6))
    plt.plot(df_prev["Date"], df_prev["Adj_Close"],
             color="#1f77b4", linewidth=2, label="Preço Real")
    plt.plot(df_prev["Date"], df_prev["preco_previsto"],
             color="#ff7f0e", linewidth=2, linestyle="--", label="Preço Previsto")
    plt.title(f"{ticker} — Preço Real vs Previsto", fontsize=14, pad=10)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Preço (R$)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()
    print(f"[GRÁFICO] Salvo: '{nome_arquivo}'")


# ==============================================================================
# GRÁFICO: Um Real vs Previsto por experimento (painel completo)
# ==============================================================================

def grafico_todos_experimentos(experimentos_cfg: dict, resultados: dict,
                                val_loader, caminho_csv: str, ticker: str,
                                nome_arquivo: str = "comparativo_todos_experimentos.png"):

    df = pd.read_csv(caminho_csv, parse_dates=["Date"])
    nomes  = list(resultados.keys())
    n_exps = len(nomes)
    n_cols = 2
    n_rows = (n_exps + n_cols - 1) // n_cols   # linhas para os subplots de preço

    fig = plt.figure(figsize=(14, 5 * (n_rows + 1)))
    gs  = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.55, wspace=0.3)

    for i, nome in enumerate(nomes):
        row = i // n_cols
        col = i % n_cols
        ax  = fig.add_subplot(gs[row, col])

        ckpt = resultados[nome].get("checkpoint")
        if not ckpt:
            ax.text(0.5, 0.5, "Sem checkpoint", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(nome)
            continue

        try:
            modelo = TemporalFusionTransformer.load_from_checkpoint(ckpt)
            modelo.eval()

            # ── mesma lógica do script de referência ──────────────────────
            raw         = modelo.predict(val_loader, mode="raw", return_x=True)
            output      = raw.output[0]
            decoder_tgt = raw.x["decoder_target"]
            time_idx    = raw.x["decoder_time_idx"]

            rows = []
            for ii in range(output.shape[0]):
                for h in range(output.shape[1]):
                    rows.append({
                        "time_idx": int(time_idx[ii, h].cpu().item()),
                        "real":     float(decoder_tgt[ii, h].cpu().item()),
                        "p50":      float(output[ii, h, 3].cpu().item()),
                    })

            df_prev = (pd.DataFrame(rows)
                         .groupby("time_idx")[["real", "p50"]]
                         .mean()
                         .reset_index()
                         .sort_values("time_idx"))

            datas = df[df["Ticker"] == ticker][["Time_Idx", "Date", "Adj_Close"]].rename(
                columns={"Time_Idx": "time_idx"}
            )
            df_prev = df_prev.merge(datas, on="time_idx", how="left").dropna(subset=["Date"])

            df_ticker = df[df["Ticker"] == ticker]
            idx_base = df_prev["time_idx"].min() - 1
            matching = df_ticker[df_ticker["Time_Idx"] == idx_base]["Adj_Close"].values

            if len(matching) == 0:
                # Fallback: usa o primeiro preço disponível do período
                preco_base = df_ticker["Adj_Close"].iloc[0]
            else:
                preco_base = matching[0]


            df_prev["preco_previsto"] = preco_base * np.exp(df_prev["p50"].cumsum())

            # ── gráfico individual ─────────────────────────────────────────
            nome_individual = f"{nome}_real_vs_previsto.png"
            fig_ind, ax_ind = plt.subplots(figsize=(12, 6))
            ax_ind.plot(df_prev["Date"], df_prev["Adj_Close"],
                        color="#1f77b4", linewidth=2, label="Preço Real")
            ax_ind.plot(df_prev["Date"], df_prev["preco_previsto"],
                        color="#ff7f0e", linewidth=2, linestyle="--", label="Preço Previsto")
            ax_ind.set_title(f"{ticker} — {nome} | Preço Real vs Previsto", fontsize=13, pad=10)
            ax_ind.set_xlabel("Data", fontsize=12)
            ax_ind.set_ylabel("Preço (R$)", fontsize=12)
            ax_ind.legend(fontsize=11)
            ax_ind.grid(True, alpha=0.3, linestyle="--")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(nome_individual, dpi=150)
            plt.close(fig_ind)
            print(f"[GRÁFICO] {nome_individual} salvo.")

            # ── subplot no painel ──────────────────────────────────────────
            ax.plot(df_prev["Date"], df_prev["Adj_Close"],
                    color="#1f77b4", linewidth=1.5, label="Real")
            ax.plot(df_prev["Date"], df_prev["preco_previsto"],
                    color="#ff7f0e", linewidth=1.5, linestyle="--", label="Previsto")

            cfg = experimentos_cfg[nome]
            titulo = (f"{nome}  hs={cfg['hidden_size']} heads={cfg['attention_head_size']} "
                      f"drop={cfg['dropout']}\n"
                      f"Assertividade: {resultados[nome]['assertividade']:.1f}%")
            ax.set_title(titulo, fontsize=9)
            ax.set_xlabel("Data", fontsize=8)
            ax.set_ylabel("Preço (R$)", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(axis='x', rotation=30, labelsize=7)

        except Exception as e:
            ax.text(0.5, 0.5, f"Erro:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            ax.set_title(nome)

    # ── Barra de assertividade (última linha, largura total) ─────────────────
    ax_bar = fig.add_subplot(gs[n_rows, :])
    scores = [resultados[k]["assertividade"] for k in nomes]
    cores  = ["#2ecc71" if s >= 55 else "#e74c3c" if s < 50 else "#f39c12" for s in scores]

    bars = ax_bar.bar(nomes, scores, color=cores, edgecolor="white", linewidth=0.8)
    ax_bar.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="Base aleatória (50%)")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Assertividade (%)", fontsize=11)
    ax_bar.set_title("Comparativo de Assertividade", fontsize=12)
    ax_bar.legend(fontsize=10)
    for bar, score in zip(bars, scores):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{score:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle(f"{ticker} — Comparativo de Experimentos TFT", fontsize=14, y=1.01)
    plt.savefig(nome_arquivo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[GRÁFICO] Painel completo salvo: '{nome_arquivo}'")


# ==============================================================================
# GRÁFICO: Só barra de assertividade + tabela
# ==============================================================================

def grafico_assertividade(experimentos_cfg: dict, resultados: dict,
                           nome_arquivo: str = "assertividade_experimentos.png"):
    """Barra de assertividade + tabela no terminal + CSV."""
    nomes  = list(resultados.keys())
    scores = [resultados[k]["assertividade"] for k in nomes]
    cores  = ["#2ecc71" if s >= 55 else "#e74c3c" if s < 50 else "#f39c12" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(nomes, scores, color=cores, edgecolor="white", linewidth=0.8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="Base aleatória (50%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Assertividade de Direção (%)", fontsize=12)
    ax.set_title("Comparativo de Experimentos TFT — Acerto de Direção", fontsize=14, pad=12)
    ax.legend(fontsize=10)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{score:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()
    print(f"[GRÁFICO] Assertividade salva: '{nome_arquivo}'")

    col_labels = ["Experimento", "hidden_size", "attn_heads", "dropout", "hidden_cont", "Assertividade (%)"]
    rows = []
    for nome in nomes:
        cfg = experimentos_cfg[nome]
        rows.append([nome, cfg["hidden_size"], cfg["attention_head_size"],
                     cfg["dropout"], cfg["hidden_continuous_size"],
                     f"{resultados[nome]['assertividade']:.1f}%"])

    df_tabela = pd.DataFrame(rows, columns=col_labels)
    print("\n" + "=" * 70)
    print("TABELA FINAL DE RESULTADOS")
    print("=" * 70)
    print(df_tabela.to_string(index=False))
    print("=" * 70)
    df_tabela.to_csv("resultados_experimentos.csv", index=False)
    print("[INFO] Tabela salva em 'resultados_experimentos.csv'")

def gerar_grafico_comparativo(resultados: dict, experiments: dict, nome_arquivo: str = "comparativo_experimentos.png"):
    nomes  = list(resultados.keys())
    scores = [resultados[k]["assertividade"] for k in nomes]

    cores = ["#2ecc71" if s >= 55 else "#e74c3c" if s < 50 else "#f39c12" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(nomes, scores, color=cores, edgecolor="white", linewidth=0.8)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, label="Base aleatória (50%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Assertividade de Direção (%)", fontsize=12)
    ax.set_title("Comparativo de Experimentos TFT — Acerto de Direção", fontsize=14, pad=12)
    ax.legend(fontsize=10)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{score:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Tabela abaixo do gráfico
    col_labels = ["Experimento", "hidden_size", "attn_heads", "dropout", "hidden_cont", "Assertividade (%)"]
    table_data = []
    for nome in nomes:
        cfg = experiments[nome]
        table_data.append([
            nome,
            cfg["hidden_size"],
            cfg["attention_head_size"],
            cfg["dropout"],
            cfg["hidden_continuous_size"],
            f"{resultados[nome]['assertividade']:.1f}%"
        ])

    df_tabela = pd.DataFrame(table_data, columns=col_labels)
    print("\n" + "="*70)
    print("TABELA FINAL DE RESULTADOS")
    print("="*70)
    print(df_tabela.to_string(index=False))
    print("="*70)

    df_tabela.to_csv("resultados_experimentos.csv", index=False)
    print("\n[INFO] Tabela salva em 'resultados_experimentos.csv'")

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()
    print(f"[INFO] Gráfico salvo em '{nome_arquivo}'")