import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from model import DisentangledRegressor, convert_to_bayesian

# 与 train_mlp.py 保持完全一致的物理列定义
NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp", "wp_over_wn", "wp_sum", "wn_sum", "is_inv",
    "stack_pu", "stack_pd", "log_slew", "log_cap", "req_p", "req_n", "rc_p", "rc_n",
    "rc_eff", "req_eff", "inv_v", "inv_temp", "pn_balance", "pol_bit",
]
TARGET_COL = "delay"


class TabularDataset(Dataset):
    def __init__(self, csv_path, x_mean, x_std, y_mean, y_std):
        if not os.path.exists(csv_path):
            # 允许文件不存在（比如只测 Test 不测 Val）
            self.x = np.array([])
            self.y = np.array([])
            return

        df = pd.read_csv(csv_path)

        # 预处理逻辑与 Training 严格一致
        if "pol_bit" not in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32) if "pol" in df.columns else 0.0
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0

        x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
        # 鲁棒归一化
        self.x = (x_raw - x_mean) / (x_std + 1e-6)

        y_raw = df[TARGET_COL].astype(np.float32).values
        self.y = (y_raw - y_mean) / (y_std + 1e-6)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


def load_scalers(data_dir):
    ss_path = os.path.join(data_dir, "scaler_stats.json")
    ys_path = os.path.join(data_dir, "y_scaler.json")
    if not os.path.exists(ss_path): return 0.0, 1.0, 0.0, 1.0
    stats = json.load(open(ss_path, "r"))
    yinfo = json.load(open(ys_path, "r"))
    x_mean = np.array([stats["mean"].get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats["std"].get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    y_mean, y_std = float(yinfo["mean"]), float(yinfo["std"])
    return x_mean, x_std, y_mean, y_std


def evaluate_dataset(model, loader, device, y_mean, y_std, mc_samples=50, desc="Eval"):
    if len(loader.dataset) == 0:
        return None, None

    # === 关键点：开启 MC Dropout / Bayesian Sampling ===
    # 我们需要模型处于 train 模式来触发 BayesianLinear 的随机采样
    # 但我们使用 torch.no_grad() 来关闭梯度计算
    model.train()

    all_preds_mu = []
    all_preds_std = []
    all_targets = []

    # 外部循环：Batch
    for xb, yb in tqdm(loader, desc=desc, leave=False):
        xb = xb.to(device)
        yb = yb.numpy()  # 保持归一化的 label 用于计算 Metric 也可以，这里还原一下更直观

        # Monte Carlo Sampling
        # shape: [MC, Batch]
        batch_preds = []

        with torch.no_grad():
            for _ in range(mc_samples):
                # Forward: mu, logvar, _, _
                # 注意：这里我们主要关心 mu 的预测分布
                # 如果要考虑 Aleatoric Uncertainty，还需要加上 exp(logvar)
                # 这里简化为 Epistemic Uncertainty (模型参数不确定性)
                mu, logv, _, _ = model(xb)

                # 反归一化
                mu_real = mu.cpu().numpy() * y_std + y_mean
                batch_preds.append(mu_real)

        # Stack -> [MC, Batch]
        batch_preds = np.array(batch_preds)

        # 计算均值和标准差 (Uncertainty)
        # mean over MC samples
        mu_mean = np.mean(batch_preds, axis=0)
        # std over MC samples (模型越不确定，每次预测波动越大)
        mu_std = np.std(batch_preds, axis=0)

        all_preds_mu.append(mu_mean)
        all_preds_std.append(mu_std)

        # Target 反归一化
        all_targets.append(yb * y_std + y_mean)

    # Concat all batches
    y_pred = np.concatenate(all_preds_mu)
    y_std_pred = np.concatenate(all_preds_std)
    y_true = np.concatenate(all_targets)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mean_unc = np.mean(y_std_pred)

    print(f"  [{desc}] MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | Mean Unc: {mean_unc:.4f}")

    results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "uncertainty": y_std_pred,
        "dataset": desc
    })

    metrics = {
        "Dataset": desc,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Uncertainty": mean_unc
    }

    return results, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", default="./output_physics")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (e.g., ckpt_s3.pt)")
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--mc_samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # 1. Load Scalers
    print(f"[Eval] Loading Scalers from {args.data_dir}...")
    x_mean, x_std, y_mean, y_std = load_scalers(args.data_dir)

    # 2. Prepare Datasets (No Vocab needed)
    datasets = {
        "Source": os.path.join(args.data_dir, "src_delay.csv"),
        "Target Train": os.path.join(args.data_dir, "tgt_train.csv"),
        "Target Val": os.path.join(args.data_dir, "tgt_val.csv"),  # 假设你有验证集文件
        "Target Test": os.path.join(args.data_dir, "tgt_test.csv")
    }

    # 3. Instantiate Model (Deterministic first)
    print(f"[Eval] Loading Checkpoint: {args.ckpt}")
    model = DisentangledRegressor(len(NUMERIC_COLS), args.hid)

    # 4. Convert to Bayesian Structure (Must match training structure)
    # Stage 3 checkpoint is saved AFTER conversion, so we must convert BEFORE loading
    model = convert_to_bayesian(model, prior_sigma=0.1)

    # 5. Load Weights
    try:
        ckpt = torch.load(args.ckpt, map_location=device)
    except:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    model.to(device)

    # 6. Run Evaluation
    print(f"==================================================")
    print(f" STARTING EVALUATION (MC Samples={args.mc_samples})")
    print(f"==================================================")

    all_results = []
    summary_metrics = []

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f">> Skipping {name} (File not found)")
            continue

        print(f"\n>> Evaluating on {name} ...")
        ds = TabularDataset(path, x_mean, x_std, y_mean, y_std)
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        res_df, metrics = evaluate_dataset(model, loader, device, y_mean, y_std, args.mc_samples, desc=name)
        if res_df is not None:
            all_results.append(res_df)
            summary_metrics.append(metrics)

    # 7. Save Summary
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        pred_path = os.path.join(args.save_dir, "eval_predictions_full.csv")
        final_df.to_csv(pred_path, index=False)
        print(f"\n[Done] Full predictions saved to: {pred_path}")

        summary_df = pd.DataFrame(summary_metrics)
        metric_path = os.path.join(args.save_dir, "eval_metrics_summary.csv")
        summary_df.to_csv(metric_path, index=False, float_format="%.6f")

        print("==================================================")
        print(" FINAL SUMMARY REPORT")
        print("==================================================")
        print(summary_df.to_string(index=False))
        print(f"[Done] Metrics summary saved to: {metric_path}")


if __name__ == "__main__":
    main()
