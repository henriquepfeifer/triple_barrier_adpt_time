import os
import pandas as pd
import numpy as np

# ===============================
# CONFIG
# ===============================

DATA_FOLDER = "features"
OUTPUT_FILE = "triple_barrier_dataset.csv"

RUN_FILE_KEY = "run_bars"

PT_MULT = 1.5
SL_MULT = 1.0
MAX_HORIZON = 50
MIN_PROB = 0.55

# ===============================
# HELPERS
# ===============================

def clean_features(df):
    valid_cols = []

    for col in df.columns:
        if col == "datetime":
            valid_cols.append(col)
            continue

        if df[col].iloc[60:].notna().sum() > 0:
            valid_cols.append(col)

    return df[valid_cols]


def load_all_feature_files(folder):
    feature_dfs = {}
    run_df = None

    for file in os.listdir(folder):

        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder, file)
        df = pd.read_csv(path)

        if "datetime" not in df.columns:
            continue

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            format="mixed",
            errors="coerce"
        )

        if RUN_FILE_KEY in file:
            run_df = df.copy()
        else:
            feature_dfs[file] = df.copy()

    return run_df, feature_dfs


def merge_features(run_df, feature_dfs):

    merged = run_df.sort_values("datetime").copy()

    for name, df_feat in feature_dfs.items():

        df_feat = clean_features(df_feat)
        df_feat = df_feat.sort_values("datetime")

        # 🔥 CRÍTICO: prefixo único baseado no nome do arquivo
        prefix = name.replace(".csv", "").replace("_features", "")

        rename_cols = {
            col: f"{prefix}__{col}"
            for col in df_feat.columns
            if col != "datetime"
        }

        df_feat = df_feat.rename(columns=rename_cols)

        merged = pd.merge_asof(
            merged,
            df_feat,
            on="datetime",
            direction="backward"
        )

    return merged


def build_events(df):

    events = []

    for i in range(len(df) - 1):

        t0 = df.loc[i, "datetime"]
        entry_price = df.loc[i+1, "open"]

        events.append({
            "t0": t0,
            "entry_price": entry_price,
            "index": i
        })

    return pd.DataFrame(events)

def estimate_prob_success(df, idx, entry_price):

    row = df.loc[idx]

    # ===== FEATURES CHAVE =====
    momentum = row.get("run_bars__momentum_10", 0)
    volatility = row.get("run_bars__volatility", 0)
    range_exp = row.get("run_bars__range_expansion", 1)
    trend = row.get("run_bars__trend_strength", 0)

    # ===== PRICE COMPONENT =====
    move = (row["close"] - entry_price) / entry_price

    # ===== SCORE =====
    score = (
        2.0 * move +           # direção
        0.5 * momentum +       # persistência
        0.3 * trend -          # regime
        0.7 * range_exp -      # exaustão (importante!)
        0.2 * volatility       # ruído
    )

    # squash → probabilidade
    prob = 1 / (1 + np.exp(-5 * score))

    return prob


def triple_barrier_adaptive(
    df,
    events,
    pt_mult=1.5,
    sl_mult=1.0,
    max_horizon=50,
    min_prob=0.55
):

    labels = []
    exit_prices = []

    for _, ev in events.iterrows():

        idx = int(ev["index"])
        entry_price = ev["entry_price"]

        vol = df.loc[idx, "volatility"] if "volatility" in df.columns else 0.001

        pt = entry_price * (1 + pt_mult * vol)
        sl = entry_price * (1 - sl_mult * vol)

        prob_path = []
        label = 0
        exit_price = entry_price

        for j in range(1, max_horizon):

            if idx + j >= len(df):
                break

            price = df.loc[idx + j, "close"]

            if price >= pt:
                label = 1
                exit_price = price
                break

            if price <= sl:
                label = -1
                exit_price = price
                break

            prob_success = estimate_prob_success(df, idx + j, entry_price)
            prob_path.append(prob_success)

            if len(prob_path) > 5:
                if len(prob_path) > 5:

                    recent_trend = prob_path[-1] - np.mean(prob_path[-5:])

                    if prob_path[-1] < min_prob or recent_trend < -0.05:
                        label = 0
                        exit_price = price
                        break

        labels.append(label)
        exit_prices.append(exit_price)

    events["label"] = labels
    events["exit_price"] = exit_prices

    return events


# ===============================
# MAIN
# ===============================

def main():

    print("Loading feature files...")
    run_df, feature_dfs = load_all_feature_files(DATA_FOLDER)

    if run_df is None:
        raise ValueError("Run bars file not found.")

    print("Merging features...")
    merged = merge_features(run_df, feature_dfs)
    print(len(merged.columns))

    print("Building events...")
    events = build_events(merged)

    print("Applying triple barrier...")
    events = triple_barrier_adaptive(
        merged,
        events,
        PT_MULT,
        SL_MULT,
        MAX_HORIZON,
        MIN_PROB
    )

    print("Merging dataset...")
    final_df = pd.merge(
        merged,
        events,
        left_on="datetime",
        right_on="t0",
        how="left"
    )

    print("Saving output...")
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("Done! Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
