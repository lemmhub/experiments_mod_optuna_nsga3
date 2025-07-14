#!/bin/bash

# === CONFIGURATION ===
#MODEL_PATH="/home/lmorenom/BRICENO_MODULARIZED/MODULARIZED_OPTUNA/test2/xgboost/best_model.pkl"
MODEL_PATH="best_model.pkl"

PYTHON_SCRIPT="optimization.py"
PYTHON_ENV="~/.optuna_workflow/bin/activate"
RESULTS_DIR="."
MAX_CONCURRENT_SESSIONS=105
REQUIRED_RUNS=21

# === TIMESTAMP ===
DATE_TAG=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/${DATE_TAG}_orchestrator.log"
STATUS_FILE="$LOG_DIR/${DATE_TAG}_status.txt"

# === HYPERPARAMETER SPACE ===
PM_PROBS=(0.01 0.4 0.7)
SBX_PROBS=(0.5 0.7 0.9)
SBX_ETAS=(15 20)
ALGORITHMS=("NSGA2" "RNSGA2" "MOEAD" "SMSEMOA" "NSGA3")
SEEDS=( $(shuf -i 1-2000 -n 2000) )

# === INIT ===
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1" >> "$LOG_FILE"
}

count_tmux_sessions() {
    tmux ls 2>/dev/null | grep -c "^moo_"
}

get_completed_runs() {
    algo=$1
    pm_prob=$2
    sbx_prob=$3
    sbx_eta=$4
    pattern="*_M${pm_prob}_C${sbx_prob}-${sbx_eta}_${algo}_seed*.pkl"
    find "$RESULTS_DIR" -maxdepth 1 -name "$pattern" | wc -l
}

is_combo_complete() {
    algo=$1
    pm_prob=$2
    sbx_prob=$3
    sbx_eta=$4
    count=$(get_completed_runs "$algo" "$pm_prob" "$sbx_prob" "$sbx_eta")
    [ "$count" -ge "$REQUIRED_RUNS" ]
}

# === MAIN ===
ACTIVE=$(count_tmux_sessions)
if [ "$ACTIVE" -ge "$MAX_CONCURRENT_SESSIONS" ]; then
    log "🟡 $ACTIVE sessions already running. Waiting for free slots..."
    exit 0
fi

COUNT=0
SEED_INDEX=0

for pm_prob in "${PM_PROBS[@]}"; do
  for sbx_prob in "${SBX_PROBS[@]}"; do
    for sbx_eta in "${SBX_ETAS[@]}"; do
      for algo in "${ALGORITHMS[@]}"; do

        if is_combo_complete "$algo" "$pm_prob" "$sbx_prob" "$sbx_eta"; then
            log "⏭️  SKIP: $algo P=$pm_prob C=$sbx_prob eta=$sbx_eta already completed"
            continue
        fi

        for i in $(seq 1 $REQUIRED_RUNS); do
          if [ "$(count_tmux_sessions)" -ge "$MAX_CONCURRENT_SESSIONS" ]; then
              log "🚫 Max sessions ($MAX_CONCURRENT_SESSIONS) reached. Pausing after $COUNT launches."
              break 3
          fi

          seed=${SEEDS[$SEED_INDEX]}
          SEED_INDEX=$((SEED_INDEX+1))
          session_name="moo_${algo}_P${pm_prob}_C${sbx_prob}_E${sbx_eta}_S${seed}_${DATE_TAG}"
          CMD="source $PYTHON_ENV && python3 $PYTHON_SCRIPT $MODEL_PATH --algo $algo --pm_prob=$pm_prob --sbx_prob=$sbx_prob --sbx_eta=$sbx_eta --seed=$seed"

          tmux new-session -d -s "$session_name" "$CMD"
          log "✅ LAUNCHED [$DATE_TAG]: algo=$algo | P=$pm_prob | C=$sbx_prob | eta=$sbx_eta | seed=$seed | session=$session_name"

          COUNT=$((COUNT + 1))
        done
      done
    done
  done
done

# === WRITE STATUS FILE ===
> "$STATUS_FILE"

for algo in "${ALGORITHMS[@]}"; do
  for pm_prob in "${PM_PROBS[@]}"; do
    for sbx_prob in "${SBX_PROBS[@]}"; do
      for sbx_eta in "${SBX_ETAS[@]}"; do
        count=$(get_completed_runs "$algo" "$pm_prob" "$sbx_prob" "$sbx_eta")
        status="✅ Completed"
        if [ "$count" -lt "$REQUIRED_RUNS" ]; then
          if [ "$count" -eq 0 ]; then
            status="❌ Not Started"
          else
            status="🟡 In Progress"
          fi
        fi
        echo "[$(timestamp)] $algo | P=$pm_prob | C=$sbx_prob | eta=$sbx_eta => $count/$REQUIRED_RUNS runs -> $status" >> "$STATUS_FILE"
      done
    done
  done
done

log "🟢 DONE: Orchestrator launched $COUNT new sessions."
log "🗂️  Status file updated: $STATUS_FILE"
