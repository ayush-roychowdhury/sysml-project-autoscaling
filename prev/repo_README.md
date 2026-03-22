# Transformer-Driven Microservice Autoscaling

**Group 10**: Ayush Roychowdhury, Sanika Nandpure, Wei-Ta Shao, Yung-Ta Chang, Ryan Chen

## Overview

This project investigates transformer-based approaches for microservice autoscaling. We argue that existing CNN-based methods (e.g., Sinan) fail to capture fan-out dependencies in microservice call graphs, motivating a transformer architecture with graph-masked attention.

## Preliminary Experiments

We run two baseline experiments to validate this claim:

- **Experiment A**: Compare Sinan's prediction accuracy on sequential-heavy vs. fan-out-heavy workloads from DeathStarBench Social Network.
- **Experiment B**: Test whether a simple 4D CNN modification (Sinan+, with doubled feature channels encoding call type) addresses the gap.

### Key Results

| | Sequential RMSE | Fan-out RMSE |
|---|---|---|
| Sinan (5ch) | 5.10 ms | 11.17 ms |
| Sinan+ (10ch) | 5.04 ms | 10.83 ms |

Sinan's accuracy degrades ~2× on fan-out workloads. The 4D modification provides only marginal improvement (~3%), confirming that the CNN architecture fundamentally cannot model non-local service dependencies.

## Repository Structure

```
├── data_collector.py          # Collects Docker container metrics + Jaeger traces
├── locustfile.py              # Locust workload generator (sequential, fanout, mixed profiles)
├── run_experiment.sh          # Runs Locust + data collector simultaneously
├── build_tensors.py           # Converts collected CSVs into numpy tensors for training
├── sinan_comparison.ipynb     # Colab notebook: trains Sinan vs Sinan+, plots comparison
├── preliminary_results.docx   # One-page summary of preliminary results
```

## Setup

### Prerequisites
```bash
pip install docker requests numpy locust
```

### Data Collection
Deploy DeathStarBench Social Network via Docker, then:
```bash
# Initialize social graph
cd DeathStarBench/socialNetwork
python3 scripts/init_social_graph.py --graph=socfb-Reed98

# Run experiments (sequential, fanout, mixed)
./run_experiment.sh mixed 300 50
./run_experiment.sh sequential 300 50
./run_experiment.sh fanout 300 50
```

### Build Tensors
```bash
python3 build_tensors.py --data-dir data/mixed_YYYYMMDD --save-dir tensors/mixed
python3 build_tensors.py --data-dir data/sequential_YYYYMMDD --save-dir tensors/sequential
python3 build_tensors.py --data-dir data/fanout_YYYYMMDD --save-dir tensors/fanout
```

### Training
Upload `tensors/` to Google Drive, open `sinan_comparison.ipynb` in Colab (GPU runtime), and run all cells.

## Next Steps

- Implement transformer architecture with graph-masked attention
- Construct dynamic service dependency graphs from Jaeger traces
- Evaluate under higher-load conditions on Chameleon cluster
- Compare against Sinan, Kubernetes HPA/VPA, LSTM, and GNN baselines
