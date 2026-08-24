#!/bin/bash
# Multi-seed training + aggregated Stability CSV.
#
# For every (env, algorithm, seed) combination it runs scripts/skrl/train.py to
# completion. Each run ends with train.py's own best-checkpoint evaluation —
# best_agent.pt is reloaded and rolled out, and the resulting Stability numbers
# (AUC, overshoot C, contraction rate lambda, contraction score, total reward)
# land in eval_results.json inside that run's log dir. Nothing here re-runs the
# evaluation; --skip_final_eval is therefore never passed, since that json is
# the thing being aggregated.
#
# Every run is stamped with the same CRL_RUN_TAG (see train_utils._run_metadata),
# so the final aggregation step picks up exactly this batch's runs rather than
# every historical run under logs/, and writes:
#
#   results/<tag>_runs.csv   one row per seed
#   results/<tag>.csv        one row per (env, algorithm): mean/std/CI across seeds
#
# Usage:
#   ./commands/run_seeds.sh                                         # fully interactive
#   ./commands/run_seeds.sh --algorithms ppo,sac --env car --seeds 5 --gpu 0 -y
#   ./commands/run_seeds.sh --algorithms c3m --env all --seeds "1 7 42" --gpu 0 -y
#   ./commands/run_seeds.sh --aggregate-only --tag seeds_20260720_120000
#
# Flags (all optional; anything omitted is prompted for):
#   --algorithms list    comma/space separated: ppo, sac, c3m, lqr, sdlqr,
#                        cvstem-lqr, c2rl-ppo, c2rl-sac
#   --env NAME           car|cartpole|segway|turtlebot|quadrotor, an Isaac Lab
#                        task id, or 'all' for every classic env
#   --seeds SPEC         a count (e.g. 5 → seeds 0..4) or an explicit list
#   --gpu SPEC           GPU index, a comma/space list of indices to round-robin
#                        over (e.g. "0,2,3"), or 'all' (default: all detected)
#   --num-timesteps N    override the cfg's training length
#   --parallel N         runs executed concurrently (default 1 = sequential)
#   --tag NAME           run tag / output basename (default: seeds_<timestamp>)
#   --aggregate-only     skip training, just rebuild the CSVs for --tag
#   --extra "ARGS"       extra args appended verbatim to every train.py call
#   -y, --yes            skip the confirmation prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# This script lives in commands/ but every path it uses (scripts/, logs/,
# search/configs/) is relative to the REPO ROOT, so cd there, not here.
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ── Pretty output ───────────────────────────────────────────────────────── #
# Colors only when stdout is a real terminal — this script also runs headless
# under nohup, where raw ANSI codes would just pollute the log file.
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    C_BOLD="$(tput bold)"; C_DIM="$(tput dim)"; C_RESET="$(tput sgr0)"
    C_CYAN="$(tput setaf 6)"; C_GREEN="$(tput setaf 2)"; C_YELLOW="$(tput setaf 3)"
    C_RED="$(tput setaf 1)"; C_BLUE="$(tput setaf 4)"; C_MAGENTA="$(tput setaf 5)"
else
    C_BOLD=""; C_DIM=""; C_RESET=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_MAGENTA=""
fi

_rule() { printf '%s\n' "${C_DIM}────────────────────────────────────────────────────────────${C_RESET}" >&2; }
_header() { echo "" >&2; _rule; printf '%s\n' "${C_BOLD}${C_CYAN}  $1${C_RESET}" >&2; _rule; }
_info()    { printf '%s\n' "  ${C_BLUE}ℹ${C_RESET}  $1" >&2; }
_success() { printf '%s\n' "  ${C_GREEN}✓${C_RESET}  $1" >&2; }
_warn()    { printf '%s\n' "  ${C_YELLOW}⚠${C_RESET}  $1" >&2; }
_error()   { printf '%s\n' "  ${C_RED}✗${C_RESET}  $1" >&2; }
_kv()      { printf '  %s%-18s%s %s\n' "${C_DIM}" "$1" "${C_RESET}" "${C_BOLD}$2${C_RESET}" >&2; }

ALGO_ARG=""
ENV_ARG=""
SEEDS_ARG=""
GPU_ARG=""
NUM_TIMESTEPS=""
PARALLEL=1
TAG=""
EXTRA=""
AGGREGATE_ONLY=0
YES=0
DETACHED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithms|--algorithm|--algo) ALGO_ARG="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --seeds) SEEDS_ARG="$2"; shift 2 ;;
        --gpu) GPU_ARG="$2"; shift 2 ;;
        --num-timesteps|--num_timesteps) NUM_TIMESTEPS="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --aggregate-only) AGGREGATE_ONLY=1; shift ;;
        -y|--yes) YES=1; shift ;;
        # Internal: set on the nohup'd re-exec so the second pass launches
        # instead of detaching again.
        --detached) DETACHED=1; YES=1; shift ;;
        -h|--help) grep -E '^# ?' "$0" | sed -E 's/^# ?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

prompt_choice() {
    local prompt_text="$1"; shift
    local opts=("$@") opt
    printf '\n%s\n' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}$prompt_text${C_RESET}" >&2
    PS3="  ${C_DIM}› ${C_RESET}"
    select opt in "${opts[@]}"; do
        if [[ -n "$opt" ]]; then echo "$opt"; return; fi
        _warn "Invalid choice, try again."
    done
}

# ── Aggregate-only shortcut ─────────────────────────────────────────────── #
# Rebuilding the CSV is cheap and independent of training, so it is reachable
# without re-running anything — useful after killing a batch early, or to
# refresh the table once stragglers finish.
if [[ "$AGGREGATE_ONLY" -eq 1 ]]; then
    if [[ -z "$TAG" ]]; then
        _error "--aggregate-only needs --tag <run tag> to know which batch to aggregate."
        exit 1
    fi
    _header "Aggregating $TAG"
    python scripts/aggregate_seeds.py --run-tag "$TAG" --out "results/$TAG"
    exit $?
fi

if [[ "$DETACHED" -ne 1 ]]; then
    _header "c2rl multi-seed runner"
fi

# ── Algorithms ──────────────────────────────────────────────────────────── #
ALL_ALGOS=(ppo sac c3m lqr sdlqr cvstem-lqr c2rl-ppo c2rl-sac)
if [[ -z "$ALGO_ARG" ]]; then
    ALGO_ARG=$(prompt_choice "Select algorithm to run (or 'all'):" "${ALL_ALGOS[@]}" all)
fi
if [[ "$ALGO_ARG" == "all" ]]; then
    ALGOS=("${ALL_ALGOS[@]}")
else
    # Accept comma- or space-separated lists interchangeably.
    IFS=', ' read -r -a ALGOS <<< "$ALGO_ARG"
fi

# ── Env ─────────────────────────────────────────────────────────────────── #
CLASSIC_ENVS=("classic-car-v0" "classic-car-v1" "classic-cartpole-v0" "classic-segway-v0" "classic-turtlebot-v0" "classic-quadrotor-v0")
ISAAC_ENVS=("Humanoid-PathTracking-v0" "Humanoid-VelTracking-v0" "Manipulator-PathTracking-v0" "Manipulator-VelTracking-v0" "Quadruped-PathTracking-v0" "Quadruped-VelTracking-v0")

if [[ -z "$ENV_ARG" ]]; then
    ENV_ARG=$(prompt_choice "Select env (or 'all' for every classic env):" \
        car cartpole segway turtlebot quadrotor "${ISAAC_ENVS[@]}" all)
fi

if [[ "$ENV_ARG" == "all" ]]; then
    ENVS=("${CLASSIC_ENVS[@]}")
else
    case "$ENV_ARG" in
        car|classic-car-v0)             ENVS=("classic-car-v0") ;;
        car_v1|classic-car-v1)   ENVS=("classic-car-v1") ;;
        cartpole|classic-cartpole-v0)   ENVS=("classic-cartpole-v0") ;;
        segway|classic-segway-v0)       ENVS=("classic-segway-v0") ;;
        turtlebot|classic-turtlebot-v0) ENVS=("classic-turtlebot-v0") ;;
        quadrotor|classic-quadrotor-v0) ENVS=("classic-quadrotor-v0") ;;
        *)
            _found=0
            for e in "${ISAAC_ENVS[@]}"; do [[ "$e" == "$ENV_ARG" ]] && _found=1; done
            if [[ "$_found" -ne 1 ]]; then
                _error "Unknown env '$ENV_ARG'. Classic: car/cartpole/segway/turtlebot/quadrotor (or 'all'). Isaac Lab: ${ISAAC_ENVS[*]}"
                exit 1
            fi
            ENVS=("$ENV_ARG")
            ;;
    esac
fi

# ── Seeds ───────────────────────────────────────────────────────────────── #
if [[ -z "$SEEDS_ARG" ]]; then
    printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Seeds — a count (5 → 0..4) or an explicit list${C_RESET} ${C_DIM}[5]${C_RESET}: " >&2
    read -r SEEDS_ARG
    SEEDS_ARG="${SEEDS_ARG:-5}"
fi
if [[ "$SEEDS_ARG" =~ ^[0-9]+$ ]]; then
    mapfile -t SEEDS < <(seq 0 $((SEEDS_ARG - 1)))
else
    IFS=', ' read -r -a SEEDS <<< "$SEEDS_ARG"
fi
if [[ "${#SEEDS[@]}" -lt 2 ]]; then
    _warn "Only ${#SEEDS[@]} seed(s) — the across-seed CI column will be 0."
fi

# ── GPU ─────────────────────────────────────────────────────────────────── #
# CUDA_VISIBLE_DEVICES=<index> for a nonexistent index silently hides the only
# real GPU from PyTorch, so every index is validated against what nvidia-smi
# actually reports rather than trusted. GPU_ARG may be a single index, a
# comma/space list to round-robin over, empty, or 'all' — all normalize to the
# GPU_POOL array the launch loop cycles through.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt 1 ]]; then
    _error "nvidia-smi reports 0 GPUs — aborting."
    exit 1
fi
if [[ -z "$GPU_ARG" ]]; then
    mapfile -t GPU_NAMES < <(nvidia-smi -L 2>/dev/null | sed -E 's/^GPU ([0-9]+): (.*) \(UUID.*/\1: \2/')
    GPU_CHOICE=$(prompt_choice "Select GPU to pin these runs to (or 'all' to round-robin):" "${GPU_NAMES[@]}" "all")
    [[ "$GPU_CHOICE" != "all" ]] && GPU_ARG="${GPU_CHOICE%%:*}"
fi
if [[ -z "$GPU_ARG" || "$GPU_ARG" == "all" ]]; then
    mapfile -t GPU_POOL < <(seq 0 $((NUM_GPUS - 1)))
else
    IFS=', ' read -r -a GPU_POOL <<< "$GPU_ARG"
    for g in "${GPU_POOL[@]}"; do
        if ! [[ "$g" =~ ^[0-9]+$ ]] || [[ "$g" -ge "$NUM_GPUS" ]]; then
            _error "--gpu index '$g' is out of range — nvidia-smi reports $NUM_GPUS GPU(s) (0-$((NUM_GPUS - 1)))."
            exit 1
        fi
    done
fi
NUM_POOL="${#GPU_POOL[@]}"

TAG="${TAG:-seeds_$(date '+%Y%m%d_%H%M%S')}"
TOTAL=$(( ${#ENVS[@]} * ${#ALGOS[@]} * ${#SEEDS[@]} ))

# ── Preview ─────────────────────────────────────────────────────────────── #
if [[ "$DETACHED" -ne 1 ]]; then
    _header "Run summary"
    _kv "Tag"            "$TAG"
    _kv "Algorithms"     "${ALGOS[*]}"
    _kv "Env(s)"         "${ENVS[*]}"
    _kv "Seeds"          "${SEEDS[*]}"
    _kv "GPU pool"       "${GPU_POOL[*]}$([[ "$NUM_POOL" -gt 1 ]] && echo " (round-robin)")"
    _kv "Concurrency"    "$PARALLEL"
    _kv "Total runs"     "$TOTAL"
    _kv "Output CSV"     "results/$TAG.csv"
    _rule
    echo "" >&2

    if [[ "$YES" -ne 1 ]]; then
        printf '%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Launch these $TOTAL run(s) now, detached via nohup?${C_RESET} ${C_DIM}[y/N]${C_RESET} " >&2
        read -r CONFIRM
        if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
            _warn "Aborted — nothing launched."
            exit 0
        fi
    fi
fi

# ── Detach ──────────────────────────────────────────────────────────────── #
# Everything above is interactive, so detaching happens only now, once every
# choice is resolved: re-exec this script under nohup with those choices as
# explicit flags plus --detached, then exit. The child ignores SIGHUP, so
# closing the terminal leaves a multi-hour seed batch running.
if [[ "$DETACHED" -ne 1 ]]; then
    mkdir -p "logs/seeds/$TAG"
    NOHUP_LOG="logs/seeds/$TAG/nohup.log"
    nohup "$SCRIPT_DIR/run_seeds.sh" \
        --algorithms "$(IFS=,; echo "${ALGOS[*]}")" \
        --env "$ENV_ARG" \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --parallel "$PARALLEL" --tag "$TAG" \
        ${NUM_TIMESTEPS:+--num-timesteps "$NUM_TIMESTEPS"} \
        --gpu "$(IFS=,; echo "${GPU_POOL[*]}")" \
        ${EXTRA:+--extra "$EXTRA"} \
        --detached > "$NOHUP_LOG" 2>&1 &
    disown
    _success "Launched (PID $!) — detached via nohup, safe to close this terminal."
    _info "Tail progress with: ${C_BOLD}tail -f $NOHUP_LOG${C_RESET}"
    _info "CSV lands at:       ${C_BOLD}results/$TAG.csv${C_RESET}"
    echo "" >&2
    exit 0
fi

# ── Launch ──────────────────────────────────────────────────────────────── #
LOG_DIR="logs/seeds/$TAG"
mkdir -p "$LOG_DIR"

# CRL_RUN_TAG is read by train_utils._run_metadata and written into every
# eval_results.json, which is how the aggregator below selects exactly this
# batch instead of every run ever recorded under logs/.
export CRL_RUN_TAG="$TAG"

FAILED=0
DONE=0
gpu_i=0

for ENV in "${ENVS[@]}"; do
    IS_CLASSIC=0
    for e in "${CLASSIC_ENVS[@]}"; do [[ "$e" == "$ENV" ]] && IS_CLASSIC=1; done

    for ALGO in "${ALGOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            GPU="${GPU_POOL[$((gpu_i % NUM_POOL))]}"
            gpu_i=$((gpu_i + 1))
            RUN_LOG="$LOG_DIR/${ENV}_${ALGO}_seed${SEED}.log"

            _info "[$((DONE + 1))/$TOTAL] $ENV / $ALGO / seed $SEED ${C_DIM}(GPU $GPU → $RUN_LOG)${C_RESET}"

            # `|| true` plus an explicit status check, because this script runs
            # under `set -euo pipefail`: a single diverged/crashed run must not
            # abort the whole batch. A failed run simply contributes no
            # eval_results.json, and the aggregator reports a smaller n_seeds.
            (
                CUDA_VISIBLE_DEVICES=$GPU python scripts/skrl/train.py \
                    $([[ "$IS_CLASSIC" -eq 1 ]] && echo "--classic") \
                    --task "$ENV" \
                    --algorithm "$ALGO" \
                    --seed "$SEED" \
                    ${NUM_TIMESTEPS:+--num_timesteps "$NUM_TIMESTEPS"} \
                    $EXTRA \
                    > "$RUN_LOG" 2>&1
                echo "$?" > "$RUN_LOG.status"
            ) &

            DONE=$((DONE + 1))
            # Throttle to --parallel concurrent runs. `wait -n` returns as soon
            # as any child finishes, which keeps the pipe full instead of
            # stalling on the slowest run of each batch of N.
            while [[ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]]; do
                wait -n 2>/dev/null || true
            done
        done
    done
done

wait

# Tally outcomes from the per-run status files.
for f in "$LOG_DIR"/*.status; do
    [[ -e "$f" ]] || continue
    if [[ "$(cat "$f")" != "0" ]]; then
        FAILED=$((FAILED + 1))
        _warn "Run failed: ${f%.status}"
    fi
done

if [[ "$FAILED" -gt 0 ]]; then
    _warn "$FAILED of $TOTAL run(s) exited nonzero — aggregating the rest."
else
    _success "All $TOTAL run(s) completed."
fi

# ── Aggregate ───────────────────────────────────────────────────────────── #
_header "Aggregating Stability results"
python scripts/aggregate_seeds.py --run-tag "$TAG" --out "results/$TAG" || \
    _error "Aggregation failed — rerun with: ./commands/run_seeds.sh --aggregate-only --tag $TAG"

_header "Done"
_info "Per-seed rows : ${C_BOLD}results/${TAG}_runs.csv${C_RESET}"
_info "Summary table : ${C_BOLD}results/${TAG}.csv${C_RESET}"
echo "" >&2
