"""
data_collector.py — Run on your Mac alongside Docker Desktop + DeathStarBench

Collects per-container metrics (CPU, memory, network) at 1-second intervals
AND classifies each service invocation as sequential vs fan-out using Jaeger traces.

Usage:
    python3 data_collector.py --output data/run1 --duration 300

Prerequisites:
    pip install docker requests numpy
"""

import docker
import requests
import time
import json
import csv
import os
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIG
# ============================================================
JAEGER_URL = "http://localhost:16686"

# Social Network service names (in tier order matching Sinan's convention)
SERVICE_NAMES = [
    "nginx-thrift",
    "compose-post-service",
    "post-storage-service",
    "user-timeline-service",
    "home-timeline-service",
    "social-graph-service",
    "user-service",
    "text-service",
    "media-service",
    "url-shorten-service",
    "user-mention-service",
    "unique-id-service",
    "post-storage-memcached",
    "post-storage-mongodb",
    "user-timeline-redis",
    "user-timeline-mongodb",
    "home-timeline-redis",
    "social-graph-redis",
    "social-graph-mongodb",
    "user-memcached",
    "user-mongodb",
    "media-memcached",
    "media-mongodb",
    "url-shorten-memcached",
    "url-shorten-mongodb",
    "media-frontend",
]

N_TIERS = len(SERVICE_NAMES)
SERVICE_TO_IDX = {name: i for i, name in enumerate(SERVICE_NAMES)}


# ============================================================
# DOCKER METRICS COLLECTION (Mac-compatible)
# ============================================================
def get_container_map(client):
    """Map service names to container objects."""
    containers = client.containers.list()
    container_map = {}
    for c in containers:
        name = c.name.lower()
        for svc in SERVICE_NAMES:
            if svc.replace("-", "") in name.replace("-", "").replace("_", ""):
                container_map[svc] = c
                break
    return container_map


def collect_container_stats(container):
    """Collect CPU, memory, network from one container.
    Compatible with Docker Desktop on Mac (no system_cpu_usage)."""
    try:
        stats = container.stats(stream=False)
    except Exception as e:
        return None

    # ---- CPU ----
    cpu_pct = 0.0
    try:
        cpu_stats = stats.get('cpu_stats', {})
        precpu_stats = stats.get('precpu_stats', {})

        cpu_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - \
                    precpu_stats.get('cpu_usage', {}).get('total_usage', 0)

        if 'system_cpu_usage' in cpu_stats and 'system_cpu_usage' in precpu_stats:
            # Linux path
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            num_cpus = cpu_stats.get('online_cpus', 1)
            if system_delta > 0:
                cpu_pct = (cpu_delta / system_delta) * num_cpus * 100.0
        else:
            # Mac path: cpu_delta is in nanoseconds over ~1 second interval
            num_cpus = cpu_stats.get('online_cpus', 1)
            if cpu_delta > 0:
                cpu_pct = (cpu_delta / 1e9) * 100.0
    except Exception:
        cpu_pct = 0.0

    # ---- Memory ----
    mem_rss = 0.0
    mem_cache = 0.0
    try:
        mem_stats = stats.get('memory_stats', {})
        detailed = mem_stats.get('stats', {})
        if detailed:
            mem_rss = detailed.get('rss', detailed.get('active_anon', 0)) / (1024 * 1024)
            mem_cache = detailed.get('cache', detailed.get('inactive_file', 0)) / (1024 * 1024)
        else:
            mem_rss = mem_stats.get('usage', 0) / (1024 * 1024)
    except Exception:
        pass

    # ---- Network ----
    rx_packets = 0
    tx_packets = 0
    try:
        networks = stats.get('networks', {})
        rx_packets = sum(n.get('rx_packets', 0) for n in networks.values())
        tx_packets = sum(n.get('tx_packets', 0) for n in networks.values())
    except Exception:
        pass

    return {
        'cpu_pct': round(cpu_pct, 4),
        'mem_rss_mb': round(mem_rss, 4),
        'mem_cache_mb': round(mem_cache, 4),
        'net_rx_packets': rx_packets,
        'net_tx_packets': tx_packets,
    }


# ============================================================
# JAEGER TRACE COLLECTION + FAN-OUT CLASSIFICATION
# ============================================================
def get_jaeger_services():
    """Check what services Jaeger knows about."""
    try:
        resp = requests.get(f"{JAEGER_URL}/api/services", timeout=2)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            return data if data else []
    except Exception:
        pass
    return []


def get_recent_traces(service_name="nginx-thrift", lookback="2s", limit=100):
    """Fetch traces from Jaeger."""
    try:
        resp = requests.get(f"{JAEGER_URL}/api/traces", params={
            "service": service_name,
            "lookback": lookback,
            "limit": limit,
        }, timeout=3)
        if resp.status_code == 200:
            return resp.json().get('data', [])
    except Exception:
        pass
    return []


def classify_fanout(traces):
    """Classify each service as sequential or fan-out based on span overlap."""
    fan_out_services = set()
    e2e_latencies = []

    for trace in traces:
        spans = trace.get('spans', [])
        if not spans:
            continue

        root_span = min(spans, key=lambda s: s['startTime'])
        e2e_latencies.append(root_span['duration'] / 1000.0)  # us -> ms

        children_of = defaultdict(list)
        for span in spans:
            for ref in span.get('references', []):
                if ref['refType'] == 'CHILD_OF':
                    children_of[ref['spanID']].append(span)

        for parent_id, children in children_of.items():
            if len(children) <= 1:
                continue
            children_sorted = sorted(children, key=lambda s: s['startTime'])
            for i in range(len(children_sorted) - 1):
                end_i = children_sorted[i]['startTime'] + children_sorted[i]['duration']
                if children_sorted[i + 1]['startTime'] < end_i:
                    for c in children:
                        proc = c.get('process', {})
                        svc = proc.get('serviceName', '')
                        fan_out_services.add(svc)
                    break

    labels = {}
    for svc in SERVICE_NAMES:
        labels[svc] = 1 if svc in fan_out_services else 0

    return labels, e2e_latencies


def compute_latency_percentiles(latencies):
    """Compute p50, p95, p99 from a list of latencies."""
    if not latencies:
        return {'p50': 0, 'p95': 0, 'p99': 0, 'count': 0}
    arr = np.array(latencies)
    return {
        'p50': round(float(np.percentile(arr, 50)), 3),
        'p95': round(float(np.percentile(arr, 95)), 3),
        'p99': round(float(np.percentile(arr, 99)), 3),
        'count': len(latencies),
    }


# ============================================================
# MAIN COLLECTION LOOP
# ============================================================
def collect(output_dir, duration, interval=1.0):
    os.makedirs(output_dir, exist_ok=True)

    client = docker.from_env()
    container_map = get_container_map(client)

    print(f"Found {len(container_map)} containers matching service names:")
    for svc in SERVICE_NAMES:
        status = "✓" if svc in container_map else "✗ (not found)"
        print(f"  {svc}: {status}")
    print()

    # Check Jaeger
    jaeger_services = get_jaeger_services()
    if jaeger_services:
        print(f"Jaeger has {len(jaeger_services)} services: {jaeger_services[:5]}...")
    else:
        print("WARNING: Jaeger has no services registered.")
        print("  Latency data will be zeros — use Locust CSVs for latency instead.")
    print()

    # Test one container's stats
    test_svc = next(iter(container_map), None)
    if test_svc:
        test_stats = collect_container_stats(container_map[test_svc])
        print(f"Test stats for {test_svc}: {test_stats}")
        print()

    # CSV files
    metrics_file = os.path.join(output_dir, "container_metrics.csv")
    latency_file = os.path.join(output_dir, "e2e_latency.csv")
    fanout_file = os.path.join(output_dir, "fanout_labels.csv")

    metrics_fields = ['timestamp', 'timestep', 'service', 'service_idx',
                      'cpu_pct', 'mem_rss_mb', 'mem_cache_mb',
                      'net_rx_packets', 'net_tx_packets']
    latency_fields = ['timestamp', 'timestep', 'p50_ms', 'p95_ms', 'p99_ms', 'trace_count']
    fanout_fields = ['timestamp', 'timestep'] + SERVICE_NAMES

    f_metrics = open(metrics_file, 'w', newline='')
    f_latency = open(latency_file, 'w', newline='')
    f_fanout = open(fanout_file, 'w', newline='')

    w_metrics = csv.DictWriter(f_metrics, fieldnames=metrics_fields)
    w_latency = csv.DictWriter(f_latency, fieldnames=latency_fields)
    w_fanout = csv.DictWriter(f_fanout, fieldnames=fanout_fields)

    w_metrics.writeheader()
    w_latency.writeheader()
    w_fanout.writeheader()

    print(f"Collecting data for {duration}s at {interval}s intervals...")
    print(f"Output: {output_dir}/")
    print()

    # Determine which Jaeger service to query
    jaeger_query_service = None
    if jaeger_services:
        for candidate in ["nginx-thrift", "nginx", "frontend"]:
            if candidate in jaeger_services:
                jaeger_query_service = candidate
                break
        if not jaeger_query_service:
            jaeger_query_service = jaeger_services[0]
        print(f"Querying Jaeger traces for service: {jaeger_query_service}")
    print()

    start_time = time.time()
    timestep = 0

    while time.time() - start_time < duration:
        tick_start = time.time()
        ts = datetime.now().isoformat()

        # 1. Collect Docker stats for each service (in parallel)
        def fetch_stats(svc):
            if svc in container_map:
                stats = collect_container_stats(container_map[svc])
                if stats:
                    return {
                        'timestamp': ts,
                        'timestep': timestep,
                        'service': svc,
                        'service_idx': SERVICE_TO_IDX[svc],
                        **stats
                    }
            return {
                'timestamp': ts, 'timestep': timestep,
                'service': svc, 'service_idx': SERVICE_TO_IDX[svc],
                'cpu_pct': 0, 'mem_rss_mb': 0, 'mem_cache_mb': 0,
                'net_rx_packets': 0, 'net_tx_packets': 0,
            }

        with ThreadPoolExecutor(max_workers=13) as executor:
            futures = {executor.submit(fetch_stats, svc): svc for svc in SERVICE_NAMES}
            for future in as_completed(futures):
                row = future.result()
                w_metrics.writerow(row)

        # 2. Collect Jaeger traces + classify fan-out
        if jaeger_query_service:
            traces = get_recent_traces(service_name=jaeger_query_service, lookback="2s")
            fanout_labels, e2e_latencies = classify_fanout(traces)
        else:
            fanout_labels = {svc: 0 for svc in SERVICE_NAMES}
            e2e_latencies = []

        latency_pcts = compute_latency_percentiles(e2e_latencies)

        w_latency.writerow({
            'timestamp': ts, 'timestep': timestep,
            'p50_ms': latency_pcts['p50'],
            'p95_ms': latency_pcts['p95'],
            'p99_ms': latency_pcts['p99'],
            'trace_count': latency_pcts['count'],
        })

        fanout_row = {'timestamp': ts, 'timestep': timestep}
        fanout_row.update(fanout_labels)
        w_fanout.writerow(fanout_row)

        # Flush + status update every 10 timesteps
        if timestep % 10 == 0:
            f_metrics.flush()
            f_latency.flush()
            f_fanout.flush()
            elapsed = time.time() - start_time
            print(f"  t={timestep:4d} | elapsed={elapsed:.0f}s | traces={latency_pcts['count']} | "
                  f"p99={latency_pcts['p99']:.1f}ms | fanout_svcs={sum(fanout_labels.values())}")

        timestep += 1

        # Maintain interval
        elapsed_tick = time.time() - tick_start
        if elapsed_tick < interval:
            time.sleep(interval - elapsed_tick)

    f_metrics.close()
    f_latency.close()
    f_fanout.close()

    print(f"\nDone! Collected {timestep} timesteps.")
    print(f"Files saved to {output_dir}/:")
    print(f"  container_metrics.csv  — per-tier resource usage")
    print(f"  e2e_latency.csv        — end-to-end latency percentiles")
    print(f"  fanout_labels.csv      — per-tier fan-out classification")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect metrics + traces from DeathStarBench')
    parser.add_argument('--output', type=str, default='data/run1', help='Output directory')
    parser.add_argument('--duration', type=int, default=300, help='Collection duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Collection interval in seconds')
    args = parser.parse_args()

    collect(args.output, args.duration, args.interval)
