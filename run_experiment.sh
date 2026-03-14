#!/bin/bash
# run_experiment.sh — Run Locust workload + data collection simultaneously
#
# Usage:
#   ./run_experiment.sh mixed 300 50
#   ./run_experiment.sh fanout 300 100
#   ./run_experiment.sh sequential 300 75

WORKLOAD=${1:-mixed}
DURATION=${2:-300}
USERS=${3:-50}
SPAWN_RATE=10

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/${WORKLOAD}_${TIMESTAMP}"

echo "============================================"
echo " Experiment: $WORKLOAD"
echo " Duration:   ${DURATION}s"
echo " Users:      $USERS"
echo " Output:     $OUTPUT_DIR"
echo "============================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Save experiment config
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "workload": "$WORKLOAD",
    "duration": $DURATION,
    "users": $USERS,
    "spawn_rate": $SPAWN_RATE,
    "timestamp": "$TIMESTAMP"
}
EOF

# 1. Start data collector in background
echo "[1/3] Starting data collector..."
python3 data_collector.py \
    --output "$OUTPUT_DIR" \
    --duration $DURATION \
    --interval 1.0 &
COLLECTOR_PID=$!

# 2. Wait a moment for collector to initialize
sleep 3

# 3. Start Locust workload with WORKLOAD env var
echo "[2/3] Starting Locust ($WORKLOAD workload, $USERS users)..."
WORKLOAD=$WORKLOAD locust -f locustfile.py \
    --host=http://localhost:8080 \
    --headless \
    -u $USERS \
    -r $SPAWN_RATE \
    --run-time ${DURATION}s \
    --csv="$OUTPUT_DIR/locust" \
    --csv-full-history

# 4. Wait for collector to finish
echo "[3/3] Waiting for data collector to finish..."
wait $COLLECTOR_PID 2>/dev/null

echo ""
echo "============================================"
echo " Experiment complete!"
echo " Output: $OUTPUT_DIR/"
echo ""
ls -la "$OUTPUT_DIR/"
echo "============================================"
