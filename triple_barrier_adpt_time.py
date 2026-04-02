import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# =========================================
# CONFIG
# =========================================

DATA_FOLDER = "full_features"
RUN_FILE = "BTCUSDT_run_bars_features.csv"

PT_MULT = 1.5
SL_MULT = 1.0
MAX_HORIZON = 50

OUTPUT_FILE = "triple_barrier_dataset.csv"

# =========================================
# LOAD DATA
# =========================================

def load_data(folder):

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    dfs = {}

    for f in files:

        path = os.path.join(folder, f)
        df = pd.read_csv(path)

        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
        df = df.sort_values("datetime")

        dfs[f] = df

    return dfs


# =========================================
# MERGE (CRÍTICO)
# =========================================

def merge_features(run_df, feature_dfs):

    merged = run_df.copy()

    for name, df in feature_dfs.items():

        if name == RUN_FILE:
            continue

        df = df.sort_values("datetime")

        # remove colunas OHLC duplicadas
        drop_cols = ["open", "high", "low", "close"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # prefixo para evitar conflito
        df = df.add_prefix(f"{name}__")
        df.rename(columns={f"{name}__datetime": "datetime"}, inplace=True)

        merged = pd.merge_asof(
            merged,
            df,
            on="datetime",
            direction="backward"
        )

    return merged


# =========================================
# BUILD EVENTS (RUN BARS)
# =========================================

def build_events(df):

    df = df.copy()

    # ===============================
    # GARANTIR ROLLING HIGH
    # ===============================

    if "rolling_high" not in df.columns:
        df["rolling_high"] = df["high"].rolling(20).max()

    # ===============================
    # SINAL DE BREAKOUT
    # ===============================

    df["breakout"] = df["high"] > df["rolling_high"].shift(1)

    # ===============================
    # GERAR EVENTOS
    # ===============================

    events = []

    for i in range(len(df) - 1):

        # só entra se houver breakout
        if not df.loc[i, "breakout"]:
            continue

        # evitar NaN (início da série)
        if pd.isna(df.loc[i, "rolling_high"]):
            continue

        event = {
            "t0": df.loc[i, "datetime"],
            "index": int(i),
            "entry_price": df.loc[i, "rolling_high"]
        }

        events.append(event)

    events_df = pd.DataFrame(events)

    print(f"Eventos gerados: {len(events_df)}")

    return events_df


# =========================================
# TRIPLE BARRIER
# =========================================

def triple_barrier(df, events):

    labels = []
    exit_prices = []
    exit_indices = []
    exit_times = []

    # ===============================
    # GARANTIR VOLATILIDADE
    # ===============================

    if "volatility" not in df.columns:
        df["volatility"] = df["close"].pct_change().rolling(20).std()

    for _, ev in events.iterrows():

        idx = int(ev["index"])
        entry_price = ev["entry_price"]

        # proteção contra NaN / zero
        vol = df.loc[idx, "volatility"]
        if pd.isna(vol) or vol <= 0:
            vol = 1e-4

        # ===============================
        # DEFINIÇÃO DAS BARREIRAS
        # ===============================

        pt = entry_price * (1 + PT_MULT * vol)
        sl = entry_price * (1 - SL_MULT * vol)

        label = 0
        exit_index = None
        exit_price = entry_price

        # ===============================
        # LOOP FUTURO
        # ===============================

        for j in range(1, MAX_HORIZON + 1):

            i_exit = idx + j

            if i_exit >= len(df):
                break

            high = df.loc[i_exit, "high"]
            low = df.loc[i_exit, "low"]

            # ===============================
            # TAKE PROFIT (usa HIGH)
            # ===============================

            if high >= pt:
                label = 1
                exit_index = i_exit
                exit_price = df.loc[i_exit, "open"]
                break

            # ===============================
            # STOP LOSS (usa LOW)
            # ===============================

            if low <= sl:
                label = -1
                exit_index = i_exit
                exit_price = df.loc[i_exit, "open"]
                break

        # ===============================
        # FALLBACK (TIME BARRIER)
        # ===============================

        if exit_index is None:
            exit_index = min(idx + MAX_HORIZON, len(df) - 1)
            exit_price = df.loc[exit_index, "open"]
            label = 0

        labels.append(label)
        exit_prices.append(exit_price)
        exit_indices.append(exit_index)
        exit_times.append(df.loc[exit_index, "datetime"])

    events["label"] = labels
    events["exit_price"] = exit_prices
    events["exit_index"] = exit_indices
    events["t1"] = exit_times

    return events

def plot_last_trades(df, n_trades=100):

    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    # ===============================
    # GARANTIR TIPOS CORRETOS
    # ===============================

    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["t0"] = pd.to_datetime(df["t0"], errors="coerce")
    df["t1"] = pd.to_datetime(df["t1"], errors="coerce")

    # índices podem vir como float
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["exit_index"] = pd.to_numeric(df["exit_index"], errors="coerce")

    # ===============================
    # FILTRAR TRADES VÁLIDOS
    # ===============================

    df_trades = df.dropna(subset=["index", "exit_index", "entry_price", "exit_price"]).copy()

    df_trades["index"] = df_trades["index"].astype(int)
    df_trades["exit_index"] = df_trades["exit_index"].astype(int)

    df_trades = df_trades.tail(n_trades)

    if len(df_trades) == 0:
        print("Nenhum trade válido para plot.")
        return

    # ===============================
    # DEFINIR JANELA DE PLOT
    # ===============================

    start_idx = int(df_trades["index"].min())
    end_idx = int(df_trades["exit_index"].max())

    # proteção contra out-of-bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(df) - 1, end_idx)

    df_plot = df.iloc[start_idx:end_idx + 1].copy()

    # ===============================
    # CANDLESTICK
    # ===============================

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_plot["datetime"],
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="Run Bars"
    ))

    # ===============================
    # ROLLING HIGH
    # ===============================

    if "rolling_high" not in df_plot.columns:
        df_plot["rolling_high"] = df_plot["high"].rolling(20).max()

    fig.add_trace(go.Scatter(
        x=df_plot["datetime"],
        y=df_plot["rolling_high"].shift(1),  
        mode="lines",
        line=dict(
            dash="dash",
            width=2
        ),
        name="Rolling High (Resistência)"
    ))

    # ===============================
    # ENTRADAS
    # ===============================

    fig.add_trace(go.Scatter(
        x=df_trades["t0"],
        y=df_trades["entry_price"],
        mode="markers",
        marker=dict(
            symbol="triangle-up",
            size=10,
            color="green"
        ),
        name="Entry"
    ))

    # ===============================
    # SAÍDAS
    # ===============================

    color_map = {
        1: "green",
        -1: "red",
        0: "gray"
    }

    exit_colors = df_trades["label"].map(color_map).fillna("gray")

    fig.add_trace(go.Scatter(
        x=df_trades["t1"],
        y=df_trades["exit_price"],
        mode="markers",
        marker=dict(
            symbol="x",
            size=8,
            color=exit_colors
        ),
        name="Exit"
    ))

    # ===============================
    # LINHAS ENTRY → EXIT
    # ===============================

    for _, row in df_trades.iterrows():

        if pd.isna(row["t0"]) or pd.isna(row["t1"]):
            continue

        fig.add_trace(go.Scatter(
            x=[row["t0"], row["t1"]],
            y=[row["entry_price"], row["exit_price"]],
            mode="lines",
            line=dict(
                color="rgba(200,200,200,0.3)",
                width=1
            ),
            showlegend=False
        ))

    # ===============================
    # LAYOUT
    # ===============================

    fig.update_layout(
        title=f"Últimos {len(df_trades)} Trades",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700
    )

    fig.show()

def plot_existing_dataset(file):
    print("Plotting existing dataset...")
    df = pd.read_csv(file)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["t0"] = pd.to_datetime(df["t0"])
    df["t1"] = pd.to_datetime(df["t1"])

    plot_last_trades(df, n_trades=100)


# =========================================
# MAIN
# =========================================

def main():

    print("Loading data...")
    dfs = load_data(DATA_FOLDER)

    print("Preparing run bars...")
    run_df = dfs[RUN_FILE].copy()
    #run_df = run_df.iloc[0:10000]  # limitar para teste rápido

    # remover duplicados
    run_df = run_df.sort_values("datetime")
    run_df = run_df.drop_duplicates("datetime").reset_index(drop=True)

    print("Merging features...")
    merged = merge_features(run_df, dfs)

    print("Building events...")
    events = build_events(merged)

    print("Applying triple barrier...")
    events = triple_barrier(merged, events)

    print("Final dataset...")
    final = merged.copy()

    # merge correto usando index do evento
    final = final.merge(
        events[[
            "index",
            "t0",
            "entry_price",
            "label",
            "exit_price",
            "exit_index",
            "t1"
        ]],
        left_index=True,
        right_on="index",
        how="left"
    )

    # sanity check
    assert final["datetime"].is_unique, "Duplicated timestamps detected!"

    final.to_csv(OUTPUT_FILE, index=False)

    print("Tamanho: ", len(final))
    print("Datetime repetido: ", final["datetime"].is_unique)
    print("Tamanho: ", final["label"].value_counts())

    print("Saved:", OUTPUT_FILE)

    df = final.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["t0"] = pd.to_datetime(df["t0"])
    df["t1"] = pd.to_datetime(df["t1"])

    plot_last_trades(df, n_trades=100)


# =========================================

if __name__ == "__main__":
    #plot_existing_dataset(OUTPUT_FILE)
    main()