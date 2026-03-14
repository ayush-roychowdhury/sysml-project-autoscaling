"""
build_tensors.py — Convert collected CSVs into numpy tensors for training

Run AFTER data collection. Produces .npy files to upload to Google Drive → Colab.

Usage:
    python build_tensors.py --data-dir data/mixed_20260302 --save-dir tensors/ --window 10

Output:
    tensors/X_RH.npy          — (num_samples, N_tiers, T_window, 5)   resource history
    tensors/X_RH_doubled.npy  — (num_samples, N_tiers, T_window, 10)  doubled channels (Sinan+)
    tensors/X_LH.npy          — (num_samples, T_window, 3)            latency history (p50,p95,p99)
    tensors/X_RC.npy          — (num_samples, N_tiers)                resource config (CPU limits)
    tensors/Y_latency.npy     — (num_samples, 3)                      target: next-step p50,p95,p99
    tensors/Y_violation.npy   — (num_samples,)                        target: QoS violation (binary)
    tensors/fanout_labels.npy — (num_samples, N_tiers, T_window)      fan-out labels

Prerequisites:
    pip install numpy pandas
"""

import numpy as np
import csv
import os
import argparse
import json

# Must match data_collector.py
SERVICE_NAMES = [
    "nginx-thrift", "compose-post-service", "post-storage-service",
    "user-timeline-service", "home-timeline-service", "social-graph-service",
    "user-service", "text-service", "media-service", "url-shorten-service",
    "user-mention-service", "unique-id-service",
    "post-storage-memcached", "post-storage-mongodb",
    "user-timeline-redis", "user-timeline-mongodb",
    "home-timeline-redis", "social-graph-redis", "social-graph-mongodb",
    "user-memcached", "user-mongodb", "media-memcached", "media-mongodb",
    "url-shorten-memcached", "url-shorten-mongodb", "media-frontend",
]
N_TIERS = len(SERVICE_NAMES)
FEATURES = ['cpu_pct', 'mem_rss_mb', 'mem_cache_mb', 'net_rx_packets', 'net_tx_packets']
N_FEATURES = len(FEATURES)


def load_and_build(data_dir, save_dir, window=10, qos_target_ms=500.0, violation_horizon=5):
    """
    Build training tensors from collected CSV files.
    
    Args:
        data_dir:   directory containing container_metrics.csv, e2e_latency.csv, fanout_labels.csv
        save_dir:   directory to save .npy files
        window:     number of past timesteps per sample (T)
        qos_target_ms:  QoS target in ms (p99 latency threshold for violation)
        violation_horizon: number of future timesteps to check for violation
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load CSVs ----
    print("Loading CSVs...")

    # Load container_metrics.csv
    metrics_rows = []
    with open(os.path.join(data_dir, "container_metrics.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics_rows.append(row)

    # Load e2e_latency.csv
    latency_rows = []
    with open(os.path.join(data_dir, "e2e_latency.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            latency_rows.append(row)

    # Load fanout_labels.csv
    fanout_rows = []
    with open(os.path.join(data_dir, "fanout_labels.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fanout_rows.append(row)

    total_timesteps = len(latency_rows)
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Window size: {window}")
    print(f"  Usable samples: {total_timesteps - window - violation_horizon}")

    # ---- Build per-timestep resource tensor ----
    # Shape: (total_timesteps, N_tiers, N_features)
    print("Building resource tensor...")
    resource_tensor = np.zeros((total_timesteps, N_TIERS, N_FEATURES), dtype=np.float32)

    for row in metrics_rows:
        t = int(row['timestep'])
        idx = int(row['service_idx'])
        if t < total_timesteps and idx < N_TIERS:
            for f_i, feat in enumerate(FEATURES):
                resource_tensor[t, idx, f_i] = float(row[feat])

    # ---- Build fan-out label tensor ----
    # Shape: (total_timesteps, N_tiers)
    print("Building fan-out label tensor...")
    fanout_tensor = np.zeros((total_timesteps, N_TIERS), dtype=np.float32)

    for row in fanout_rows:
        t = int(row['timestep'])
        if t < total_timesteps:
            for i, svc in enumerate(SERVICE_NAMES):
                if svc in row:
                    fanout_tensor[t, i] = float(row[svc])

    # ---- Build latency tensor ----
    # Shape: (total_timesteps, 3) for p50, p95, p99
    print("Building latency tensor...")
    latency_tensor = np.zeros((total_timesteps, 3), dtype=np.float32)
    for row in latency_rows:
        t = int(row['timestep'])
        if t < total_timesteps:
            latency_tensor[t] = [float(row['p50_ms']), float(row['p95_ms']), float(row['p99_ms'])]

    # ---- Create windowed samples ----
    num_samples = total_timesteps - window - violation_horizon
    if num_samples <= 0:
        print("ERROR: Not enough timesteps for the given window + horizon. Collect more data.")
        return

    print(f"Creating {num_samples} windowed samples...")

    X_RH = np.zeros((num_samples, N_TIERS, window, N_FEATURES), dtype=np.float32)
    X_LH = np.zeros((num_samples, window, 3), dtype=np.float32)
    X_RC = np.zeros((num_samples, N_TIERS), dtype=np.float32)  # current CPU (simplified)
    fanout_labels = np.zeros((num_samples, N_TIERS, window), dtype=np.float32)
    Y_latency = np.zeros((num_samples, 3), dtype=np.float32)
    Y_violation = np.zeros((num_samples,), dtype=np.float32)

    for s in range(num_samples):
        t_start = s
        t_end = s + window
        t_next = t_end  # predict this timestep

        # Resource history: (N_tiers, window, features)
        X_RH[s] = resource_tensor[t_start:t_end].transpose(1, 0, 2)  # (T,N,F) -> (N,T,F)

        # Latency history: (window, 3)
        X_LH[s] = latency_tensor[t_start:t_end]

        # Resource config: use current CPU usage as proxy for allocation
        X_RC[s] = resource_tensor[t_end - 1, :, 0]  # last timestep's CPU usage

        # Fan-out labels: (N_tiers, window)
        fanout_labels[s] = fanout_tensor[t_start:t_end].T  # (T,N) -> (N,T)

        # Target: next timestep latency
        Y_latency[s] = latency_tensor[t_next]

        # Target: QoS violation in next k timesteps
        future_p99 = latency_tensor[t_next:t_next + violation_horizon, 2]  # p99 column
        Y_violation[s] = 1.0 if np.any(future_p99 > qos_target_ms) else 0.0

    # ---- Build doubled-channel tensor (Sinan+) ----
    print("Building doubled-channel tensor (Sinan+)...")
    X_RH_doubled = np.zeros((num_samples, N_TIERS, window, N_FEATURES * 2), dtype=np.float32)

    for s in range(num_samples):
        for i in range(N_TIERS):
            for t in range(window):
                if fanout_labels[s, i, t] == 0:
                    # Sequential: first F channels
                    X_RH_doubled[s, i, t, :N_FEATURES] = X_RH[s, i, t, :]
                else:
                    # Fan-out: second F channels
                    X_RH_doubled[s, i, t, N_FEATURES:] = X_RH[s, i, t, :]

    # ---- Save ----
    print(f"Saving to {save_dir}/...")
    np.save(os.path.join(save_dir, "X_RH.npy"), X_RH)
    np.save(os.path.join(save_dir, "X_RH_doubled.npy"), X_RH_doubled)
    np.save(os.path.join(save_dir, "X_LH.npy"), X_LH)
    np.save(os.path.join(save_dir, "X_RC.npy"), X_RC)
    np.save(os.path.join(save_dir, "Y_latency.npy"), Y_latency)
    np.save(os.path.join(save_dir, "Y_violation.npy"), Y_violation)
    np.save(os.path.join(save_dir, "fanout_labels.npy"), fanout_labels)

    # Save metadata
    meta = {
        "num_samples": int(num_samples),
        "n_tiers": N_TIERS,
        "window": window,
        "n_features": N_FEATURES,
        "qos_target_ms": qos_target_ms,
        "violation_horizon": violation_horizon,
        "service_names": SERVICE_NAMES,
        "features": FEATURES,
        "shapes": {
            "X_RH": list(X_RH.shape),
            "X_RH_doubled": list(X_RH_doubled.shape),
            "X_LH": list(X_LH.shape),
            "X_RC": list(X_RC.shape),
            "Y_latency": list(Y_latency.shape),
            "Y_violation": list(Y_violation.shape),
        },
        "violation_rate": float(Y_violation.mean()),
    }
    with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Shapes:")
    for k, v in meta["shapes"].items():
        print(f"  {k}: {v}")
    print(f"  Violation rate: {meta['violation_rate']:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with collected CSVs')
    parser.add_argument('--save-dir', type=str, default='tensors/', help='Directory to save .npy files')
    parser.add_argument('--window', type=int, default=10, help='Number of past timesteps per sample')
    parser.add_argument('--qos-target', type=float, default=500.0, help='QoS target p99 latency (ms)')
    args = parser.parse_args()

    load_and_build(args.data_dir, args.save_dir, window=args.window, qos_target_ms=args.qos_target)
