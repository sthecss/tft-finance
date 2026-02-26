# TFT na Bolsa de Valores
### Previsão de Ações com Temporal Fusion Transformer

Projeto acadêmico para previsão de retornos de ações brasileiras utilizando o modelo **Temporal Fusion Transformer (TFT)**, com foco em comparação experimental de arquiteturas e análise de acerto de direção (subiu/desceu).

---
<br><br>

## Sumário

1. [Visão Geral](#visão-geral)
2. [Estrutura de Arquivos](#estrutura-de-arquivos)
3. [Como Executar](#como-executar)
4. [Dados e Features](#dados-e-features)
5. [Pré-processamento e Normalização](#pré-processamento-e-normalização)
6. [O Modelo TFT](#o-modelo-tft)
7. [Parâmetros de Treinamento](#parâmetros-de-treinamento)
8. [Experimentos](#experimentos)
9. [Métricas de Avaliação](#métricas-de-avaliação)
10. [Saídas Geradas](#saídas-geradas)
11. [Dependências](#dependências)

---
<br><br>

## Visão Geral

O pipeline completo faz o seguinte em uma única execução:

1. Baixa dados históricos de múltiplos tickers via Yahoo Finance
2. Calcula indicadores técnicos e features derivadas
3. Adiciona a cotação do dólar (USDBRL) como variável macroeconômica
4. Normaliza as features que precisam de padronização
5. Treina 8 configurações diferentes do modelo TFT
6. Avalia o acerto de direção de cada experimento
7. Gera gráficos individuais e um painel comparativo completo

---
<br><br>

## Estrutura de Arquivos

```
projeto/
│
├── main_experimentos.py     # Pipeline principal — ponto de entrada
├── pre_processamento.py     # Download, features e normalização
├── modelo_ia.py             # Dataloaders, instanciação e treino do TFT
├── graficos.py              # Todas as funções de visualização
│
└── dataset_tft.csv          # Gerado automaticamente na primeira execução
```

---
<br><br>

## Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/sthecss/tft-finance.git
cd tft-finance
```

### 2. Instalar as dependências

<details> <summary>Opção 1: Se escolher rodar em IDE</summary>
   
```bash
pip install pytorch-forecasting lightning yfinance pandas numpy matplotlib
```  

</details>

<details><summary>Opção 2: Se escolher rodar em um Terminal</summary>
   
```bash
python -m venv venv
source venv/bin/activate
pip install pytorch-forecasting lightning yfinance pandas numpy matplotlib
python main_experimentos.py
```

OBS: Instalação teste feita em distro Linux.

</details>

### 3. Executar o pipeline

<details>
<summary>Via Terminal</summary>

No diretório raiz do projeto, rode:

```bash
python main_experimentos.py
```

Acompanhe o progresso diretamente no console. O download dos dados, o treino de cada experimento e a geração dos gráficos são impressos em tempo real.

</details>

<details>
<summary>Via IDE (VS Code / PyCharm)</summary>

**VS Code**
1. Abra a pasta do projeto: `File → Open Folder`
2. Selecione o interpretador Python correto: `Ctrl+Shift+P` → `Python: Select Interpreter`
3. Abra o arquivo `main_experimentos.py`
4. Clique no botão **Run Python File** no canto superior direito
   — ou pressione `Ctrl+F5` para rodar sem debug
   — ou `F5` para rodar com debug (permite inspecionar variáveis em cada passo)

<br>

**PyCharm**
1. Abra a pasta do projeto: `File → Open`
2. Configure o interpretador: `File → Settings → Project → Python Interpreter`
3. Abra o arquivo `main_experimentos.py`
4. Clique no botão **Run** na barra superior
   — ou clique com o botão direito no arquivo → `Run 'main_experimentos'`
   — ou pressione `Shift+F10`

> Em ambas as IDEs, o output aparece no painel **Terminal** ou **Run** integrado, igual ao terminal comum.

</details>

> **Primeira execução:** o programa baixa e processa todos os dados automaticamente. Nas execuções seguintes, o arquivo `dataset_tft.csv` já existe e o download é pulado.
> Para forçar novo download (ex: atualizar o período), basta deletar o `dataset_tft.csv`.

---
<br><br>

## Dados e Features

### Tickers utilizados

O projeto está configurado para baixar múltiplos ativos de diferentes setores simultaneamente. Todos são concatenados em um único CSV e o modelo aprende padrões entre eles.

| Setor | Tickers |
|---|---|
| Setor Financeiro | ITUB3, ITUB4, BBAS3, BBDC3, BBDC4, SANB11, BPAC11 |
| Energia Elétrica | AXIA3, CPFE3, ENGI3, CMIG4 |
| Saneamento | SBSP3, CSMG3, SAPR4 |

O ticker usado como referência nos gráficos de preço é definido pela variável `TICKER_PRINCIPAL`.

<br>

### Período

```
DATA_INICIO = "2022-01-01"
DATA_FIM    = "2025-12-31"
```

<br>

### Divisão treino / validação

O conjunto de validação corresponde aos **últimos 126 dias úteis (~6 meses)** de cada série. O restante é usado para treino.

<br>

### O que o modelo tenta prever

O modelo prevê o **Log_Return** — o retorno logarítmico diário do ativo:

```
Log_Return = log(Preço hoje / Preço ontem)
```

Um valor positivo significa que o preço subiu; negativo, que caiu. O uso do logaritmo torna a série mais estável matematicamente e comparável entre ativos com escalas de preço diferentes.

<details>
<summary>Features de entrada do modelo</summary>

Definidas em `main_experimentos.py`:

```python
FEATURES = ["Volume", "Log_Return", "SMA_14", "SMA_50", "Volatility_21", "RSI_14"]
```

Após adicionar o dólar, o vetor completo passa a incluir também:

```python
FEATURES = [
    "Volume", "Log_Return", "SMA_14", "SMA_50", "Volatility_21", "RSI_14",
    "USD_Close", "USD_Log_Return", "USD_Volatility_21"
]
```

**Features numéricas:**

| Feature | Fórmula / Origem | O que representa |
|---|---|---|
| **Volume** | `log1p(volume bruto)` | Quantidade de ações negociadas no dia. O log suaviza dias com volume extremo. Indica a "força" por trás de um movimento de preço. |
| **Log_Return** | `log(P_t / P_{t-1})` | Retorno diário do ativo. É a variável-alvo — o que o modelo tenta prever. |
| **SMA_14** | `Preço / média(14d)` | Preço relativo à média móvel de 14 dias. Valores > 1 indicam momento positivo de curto prazo; < 1 indicam fraqueza. |
| **SMA_50** | `Preço / média(50d)` | Mesma lógica da SMA_14, mas para tendência de médio prazo (~2 meses). |
| **Volatility_21** | `std(Log_Return, 21d)` | Desvio padrão dos retornos dos últimos 21 dias úteis (~1 mês). Mede a agitação recente do ativo. |
| **RSI_14** | Cálculo clássico RSI | Índice de Força Relativa (0–100). Acima de 70: sobrecomprado. Abaixo de 30: sobrevendido. |
| **USD_Close** | `USDBRL=X` (Yahoo Finance) | Cotação de fechamento do dólar frente ao real. Captura o contexto macroeconômico. |
| **USD_Log_Return** | `log(USD_t / USD_{t-1})` | Variação diária do dólar. Mesma escala do Log_Return das ações. |
| **USD_Volatility_21** | `std(USD_Log_Return, 21d)` | Volatilidade do dólar nos últimos 21 dias. Indica períodos de instabilidade cambial. |

<br>

**Variáveis categóricas (contexto temporal):**

| Variável | Valores possíveis | O que representa |
|---|---|---|
| **Day_of_Week** | 0 (segunda) a 4 (sexta) | Dia da semana — captura sazonalidade semanal |
| **Month** | 1 a 12 | Mês do ano — captura sazonalidade anual |
| **Ticker** | ex: "ITUB4.SA" | Identidade do ativo (variável estática) |
| **Setor** | ex: "Setor Financeiro" | Setor econômico do ativo (variável estática) |

</details>

---
<br><br>

## Pré-processamento e Normalização

O modelo TFT trata todas as features como números. Se uma feature tem escala 0–100 (RSI) e outra tem escala −0.05 a +0.05 (Log_Return), o modelo pode erroneamente tratar o RSI como ~1000× mais importante simplesmente pelo tamanho dos números. A normalização resolve isso.

**Método: Z-score por Ticker**

```python
df[col] = (df[col] - media_por_ticker) / (std_por_ticker + 1e-8)
```

O resultado é uma coluna com **média 0 e desvio padrão 1** para cada ativo individualmente. O `+ 1e-8` evita divisão por zero em casos de coluna constante.

<details>
<summary>Quais features são (e não são) normalizadas</summary>

**Normalizadas:**

| Feature | Motivo |
|---|---|
| **Volume** | Após o log1p ainda pode variar de 10 a 25+ entre ativos diferentes |
| **RSI_14** | Escala fixa de 0 a 100, incompatível com as demais features |
| **USD_Close** | Preço absoluto (~4.8 a 6.2 reais), escala incompatível |
| **Volatility_21** | Escala pequena mas inconsistente entre ativos |
| **USD_Volatility_21** | Mesma razão da Volatility_21 |

<br>
**Não normalizadas:**

| Feature | Por que já está ok |
|---|---|
| **Log_Return** | Retorno log já é pequeno e centrado em zero por natureza |
| **SMA_14** | É um ratio `Preço/Média`, sempre próximo de 1.0 |
| **SMA_50** | Mesma razão da SMA_14 |
| **USD_Log_Return** | Mesma razão do Log_Return |

> O `Adj_Close` não entra como feature do modelo — existe apenas no CSV para reconstruir o preço em R$ nos gráficos finais. Normalizá-lo faria os gráficos exibirem desvios padrão em vez de R$.

</details>

---
<br><br>

## O Modelo TFT

O TFT é uma arquitetura de deep learning desenvolvida pelo Google para previsão de séries temporais. Ele combina três mecanismos principais:

- **Gated Residual Networks (GRN)**: processam cada feature de forma independente antes de combiná-las
- **Variable Selection Networks**: aprendem automaticamente quais features são mais relevantes para cada previsão
- **Multi-head Attention**: identifica quais dias do passado são mais importantes para prever o futuro

<details>
<summary>Entradas, saídas e quantis</summary>

**O que entra no modelo:**

```
Passado (ENCODER_LENGTH = 30 dias)
  → Features numéricas variantes no tempo
  → Features categóricas (dia da semana, mês)
  → Ticker e Setor (estáticos)

Futuro conhecido (PREDICTION_LENGTH = 5 dias)
  → Dia da semana e Mês (sabemos de antemão)

Saída
  → 7 quantis de Log_Return para cada um dos 5 dias futuros
    (p10, p25, p40, p50, p60, p75, p90)
```

**Quantis de saída:**

O TFT não retorna uma previsão única — ele retorna uma **distribuição de probabilidade** na forma de quantis. O quantil 50% (mediana, índice `[3]` no vetor) é usado como previsão central nos gráficos e no cálculo de assertividade.

| Índice | Quantil | Interpretação |
|---|---|---|
| 0 | p10 | Cenário pessimista extremo |
| 1 | p25 | Cenário pessimista moderado |
| 2 | p40 | Ligeiramente abaixo da mediana |
| 3 | **p50** | **Previsão central (mediana)** |
| 4 | p60 | Ligeiramente acima da mediana |
| 5 | p75 | Cenário otimista moderado |
| 6 | p90 | Cenário otimista extremo |

</details>

---
<br><br>

## Parâmetros de Treinamento

```python
ENCODER_LENGTH    = 30     # Dias de histórico que o modelo olha para trás
PREDICTION_LENGTH = 5      # Dias de futuro que o modelo tenta prever
EPOCHS            = 50     # Máximo de épocas de treino
LR                = 0.001  # Taxa de aprendizado
```

<details>
<summary>Descrição completa dos parâmetros</summary>

| Parâmetro | Valor | Descrição |
|---|---|---|
| **ENCODER_LENGTH** | 30 | Janela de memória: o modelo analisa os 30 dias anteriores (≈4 semanas) antes de prever |
| **PREDICTION_LENGTH** | 5 | Horizonte de previsão: gera previsão para cada um dos 5 dias úteis seguintes (1 semana) |
| **EPOCHS** | 50 | Número máximo de passagens completas pelos dados de treino |
| **LR** | 0.001 | Velocidade de aprendizado — valor clássico e estável para redes neurais |
| **batch_size** | 64 | Amostras processadas por vez durante o treino |
| **gradient_clip_val** | 0.1 | Limita gradientes explosivos, estabilizando o treino |
| **EarlyStopping** | patience=10 | Interrompe o treino se a loss de validação não melhorar por 10 épocas seguidas |

</details>

---
<br><br>

## Experimentos

Foram definidos 8 experimentos variando os hiperparâmetros de arquitetura do TFT:

| Experimento | hidden_size | attn_heads | dropout | hidden_cont |
|---|---|---|---|---|
| EXP_01 | 16 | 1 | 0.1 | 8 |
| EXP_02 | 32 | 2 | 0.1 | 16 |
| EXP_03 | 64 | 4 | 0.1 | 32 |
| EXP_04 | 64 | 4 | 0.3 | 32 |
| EXP_05 | 128 | 4 | 0.1 | 64 |
| EXP_06 | 128 | 8 | 0.1 | 64 |
| EXP_07 | 128 | 8 | 0.3 | 64 |
| EXP_08 | 256 | 8 | 0.3 | 128 |

<details>
<summary>O que cada hiperparâmetro controla</summary>

| Parâmetro | Descrição |
|---|---|
| **hidden_size** | Número de neurônios nas camadas internas. Define a "capacidade de raciocínio" do modelo. Maior = mais complexo, porém mais lento e maior risco de overfitting. |
| **attention_head_size** | Quantos focos de atenção simultâneos o modelo tem. Cada head pode aprender a identificar um padrão diferente no histórico. |
| **dropout** | Porcentagem de neurônios desativados aleatoriamente durante o treino. Funciona como prevenção de overfitting — o modelo aprende a não depender de nenhum neurônio específico. |
| **hidden_continuous_size** | Tamanho da mini-rede interna que processa cada feature numérica antes de entrar no modelo principal. Convencionalmente definido como metade do `hidden_size`. |

</details>

---
<br><br>

## Métricas de Avaliação

A métrica principal é a **porcentagem de acerto de direção**: o modelo previu corretamente se o preço iria subir ou cair?

```python
acertos = np.sign(reais) == np.sign(previstos)
assertividade = acertos.mean() * 100
```

O `np.sign()` transforma qualquer número em `+1` (subiu), `−1` (caiu) ou `0`. A comparação verifica apenas se os sinais coincidem — não importa a magnitude do erro, apenas a direção.

**Referência:** uma previsão completamente aleatória acertaria ~50% das direções. Qualquer resultado consistentemente acima de 50% indica que o modelo capturou algum padrão real.

<details>
<summary>Reconstrução do preço em reais</summary>

Para os gráficos, o preço previsto é reconstruído a partir dos Log_Returns previstos:

```
Preço previsto = Preço base × exp(cumsum(Log_Returns previstos))
```

O "preço base" é o último fechamento real antes da janela de validação, garantindo que a curva prevista começa do ponto correto.

</details>

---
<br><br>

## Saídas Geradas

Após a execução completa, os seguintes arquivos são criados:

| Arquivo | Descrição |
|---|---|
| `dataset_tft.csv` | Dataset completo com todos os tickers, features e dólar |
| `resultados_experimentos.csv` | Tabela com a assertividade de cada experimento |
| `assertividade_experimentos.png` | Gráfico de barras comparativo de assertividade |
| `EXP_01_real_vs_previsto.png` | Gráfico de preço real vs previsto do EXP_01 |
| `EXP_02_real_vs_previsto.png` | Gráfico de preço real vs previsto do EXP_02 |
| `...` | (um arquivo por experimento) |
| `comparativo_todos_experimentos.png` | Painel com todos os experimentos em uma única imagem |

---
<br><br>

## Dependências

```bash
pip install pytorch-forecasting lightning yfinance pandas numpy matplotlib
```

<details>
<summary>Descrição de cada biblioteca</summary>

| Biblioteca | Uso |
|---|---|
| `pytorch-forecasting` | Implementação do TFT e utilitários de séries temporais |
| `lightning` (PyTorch Lightning) | Gerenciamento do treino, callbacks e checkpoints |
| `yfinance` | Download de dados históricos de ações e câmbio |
| `pandas` / `numpy` | Manipulação e cálculo das features |
| `matplotlib` | Geração dos gráficos |

</details>

---
<br><br>

*Projeto desenvolvido para fins acadêmicos.*






