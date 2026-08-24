#!/bin/bash
# Single entry point for every W&B sweep in this repo.
#
# Prompts for algorithm / env / agents / GPU, previews the exact metric and
# hyperparameter ranges that will be searched, then launches the sweep detached
# (nohup) so it survives closing the terminal.
#
# The searched space is not defined here — it lives in search/configs/, one
# yaml per algorithm, applying to every env. This script only discovers those
# configs, and search/build_sweep.py turns the chosen one into a W&B sweep yaml.
# To change what is searched, or to add an algorithm, edit search/configs/
# (see its README.md); nothing here needs touching.
#
# Usage:
#   ./commands/search.sh                                        # fully interactive
#   ./commands/search.sh --algorithm cvstem-lqr --env car --gpu 0 --num-agents 3 -y
#   ./commands/search.sh --algorithm c2rl-ppo-cvstem --env all --gpu 0 --runs-per-agent 40 -y
#
# Flags (all optional; anything omitted is prompted for):
#   --algorithm NAME     a stem in search/configs/ (e.g. ppo, c3m, cvstem-lqr)
#   --env NAME           car|cartpole|segway|turtlebot|quadrotor, an Isaac Lab
#                        task id, or 'all' for every classic env
#   --num-agents N       parallel wandb agents per env (default 3)
#   --gpu N              pin every agent to this GPU (default: round-robin)
#   --method M           bayes (default) | grid | random
#   --timeout T          per-run watchdog (default 24h)
#   --runs-per-agent N   stop each agent after N runs (default 0 = unbounded)
#   -y, --yes            skip the confirmation prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# The script moved to commands/, but the searched space and the sweep logs stay
# in search/ -- they are data, not commands. Name them off the repo root so the
# script's location and its inputs are independent.
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_DIR/search/configs"
SEARCH_DIR="$REPO_DIR/search"

# ── Pretty output ───────────────────────────────────────────────────────── #
# Colors only when stdout is a real terminal — this script also runs headless
# under nohup (see "Detach" below), where raw ANSI codes would just pollute
# the log file.
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

NUM_AGENTS=""
ALGORITHM=""
ENV_ARG=""
GPU_ARG=""
METHOD="bayes"
PER_RUN_TIMEOUT=24h
RUNS_PER_AGENT=0
YES=0
DETACHED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|--algo) ALGORITHM="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --num-agents) NUM_AGENTS="$2"; shift 2 ;;
        --gpu) GPU_ARG="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --timeout) PER_RUN_TIMEOUT="$2"; shift 2 ;;
        --runs-per-agent) RUNS_PER_AGENT="$2"; shift 2 ;;
        -y|--yes) YES=1; shift ;;
        # Internal: set on the nohup'd re-exec (see "Detach" below) so the
        # second pass launches instead of detaching again.
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

if [[ "$DETACHED" -ne 1 ]]; then
    _header "c2rl sweep launcher"
fi

# ── Algorithm ───────────────────────────────────────────────────────────── #
# Discovered by globbing configs/ — adding a yaml there is all it takes to make
# a new algorithm selectable here.
mapfile -t ALGOS < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' | sed 's/\.yaml$//' | sort)
if [[ "${#ALGOS[@]}" -eq 0 ]]; then
    _error "No algorithm configs found in $CONFIG_DIR."
    exit 1
fi
if [[ -z "$ALGORITHM" ]]; then
    ALGORITHM=$(prompt_choice "Select algorithm to sweep:" "${ALGOS[@]}")
fi
if [[ ! -f "$CONFIG_DIR/$ALGORITHM.yaml" ]]; then
    _error "No config '$ALGORITHM.yaml' in $CONFIG_DIR. Available: ${ALGOS[*]}"
    exit 1
fi
[[ "$DETACHED" -ne 1 ]] && _success "Algorithm: ${C_BOLD}$ALGORITHM${C_RESET}"

# ── Env ─────────────────────────────────────────────────────────────────── #
CLASSIC_ENVS=("classic-car-v0" "classic-cartpole-v0" "classic-segway-v0" "classic-turtlebot-v0" "classic-quadrotor-v0")
ISAAC_ENVS=("Humanoid-PathTracking-v0" "Humanoid-VelTracking-v0" "Manipulator-PathTracking-v0" "Manipulator-VelTracking-v0" "Quadruped-PathTracking-v0" "Quadruped-VelTracking-v0")

if [[ -z "$ENV_ARG" ]]; then
    ENV_ARG=$(prompt_choice "Select env to sweep (or 'all' for every classic env):" \
        car cartpole segway turtlebot quadrotor "${ISAAC_ENVS[@]}" all)
fi

if [[ "$ENV_ARG" == "all" ]]; then
    ENVS=("${CLASSIC_ENVS[@]}")
else
    case "$ENV_ARG" in
        car|classic-car-v0)             ENVS=("classic-car-v0") ;;
        cartpole|classic-cartpole-v0)   ENVS=("classic-cartpole-v0") ;;
        segway|classic-segway-v0)       ENVS=("classic-segway-v0") ;;
        turtlebot|classic-turtlebot-v0) ENVS=("classic-turtlebot-v0") ;;
        quadrotor|classic-quadrotor-v0) ENVS=("classic-quadrotor-v0") ;;
        *)
            # Isaac Lab task id — accept only ids that actually exist.
            _found=0
            for e in "${ISAAC_ENVS[@]}"; do [[ "$e" == "$ENV_ARG" ]] && _found=1; done
            if [[ "$_found" -ne 1 ]]; then
                echo "Unknown env '$ENV_ARG'. Classic: car/cartpole/segway/turtlebot/quadrotor (or 'all'). Isaac Lab: ${ISAAC_ENVS[*]}" >&2
                exit 1
            fi
            ENVS=("$ENV_ARG")
            ;;
    esac
fi
[[ "$DETACHED" -ne 1 ]] && _success "Env(s): ${C_BOLD}${ENVS[*]}${C_RESET}"

if [[ -z "$NUM_AGENTS" ]]; then
    printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Number of parallel wandb agents to spawn per env${C_RESET} ${C_DIM}[3]${C_RESET}: " >&2
    read -r NUM_AGENTS
    NUM_AGENTS="${NUM_AGENTS:-3}"
fi
[[ "$DETACHED" -ne 1 ]] && _success "Agents per env: ${C_BOLD}$NUM_AGENTS${C_RESET}"

# ── GPU ─────────────────────────────────────────────────────────────────── #
# CUDA_VISIBLE_DEVICES=<index> for a nonexistent index silently hides the only
# real GPU from PyTorch, so the index is validated against what nvidia-smi
# actually reports rather than trusted.
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt 1 ]]; then
    _error "nvidia-smi reports 0 GPUs — aborting."
    exit 1
fi
if [[ -z "$GPU_ARG" ]]; then
    mapfile -t GPU_NAMES < <(nvidia-smi -L 2>/dev/null | sed -E 's/^GPU ([0-9]+): (.*) \(UUID.*/\1: \2/')
    GPU_CHOICE=$(prompt_choice "Select GPU to pin this sweep to (or 'all' to round-robin across every GPU):" "${GPU_NAMES[@]}" "all")
    [[ "$GPU_CHOICE" != "all" ]] && GPU_ARG="${GPU_CHOICE%%:*}"
fi
if [[ -n "$GPU_ARG" ]]; then
    if ! [[ "$GPU_ARG" =~ ^[0-9]+$ ]] || [[ "$GPU_ARG" -ge "$NUM_GPUS" ]]; then
        _error "--gpu $GPU_ARG is out of range — nvidia-smi reports $NUM_GPUS GPU(s) (0-$((NUM_GPUS - 1)))."
        exit 1
    fi
fi
[[ "$DETACHED" -ne 1 ]] && _success "GPU: ${C_BOLD}${GPU_ARG:-all detected (round-robin)}${C_RESET}"

cd "$REPO_DIR"

# ── Preview ─────────────────────────────────────────────────────────────── #
if [[ "$DETACHED" -ne 1 ]]; then
    _header "Sweep summary"
    _kv "Algorithm"       "$ALGORITHM"
    _kv "Method"          "$METHOD"
    _kv "Env(s)"          "${ENVS[*]}"
    _kv "Agents per env"  "$NUM_AGENTS"
    _kv "GPU"             "${GPU_ARG:-all detected (round-robin)}"
    if [[ "$RUNS_PER_AGENT" -gt 0 ]]; then
        _kv "Runs per agent" "$RUNS_PER_AGENT (total per env: $((RUNS_PER_AGENT * NUM_AGENTS)))"
    else
        _kv "Runs per agent" "unbounded (bayes never exhausts — kill to stop)"
    fi
    _rule

    _header "Generated sweep (${ENVS[0]})"
    # Preview the real generated sweep, not a re-description of it — what is
    # shown here is exactly what wandb will be handed. Colored a bit if `pygmentize`
    # or `bat` happen to be around; a plain cat otherwise, so this never gains a
    # hard dependency just for a nicer preview.
    _SWEEP_PREVIEW="$(python search/build_sweep.py --algorithm "$ALGORITHM" --env "${ENVS[0]}" --method "$METHOD")"
    if command -v bat >/dev/null 2>&1; then
        echo "$_SWEEP_PREVIEW" | bat -l yaml --style=plain --color=always --paging=never >&2
    elif [[ -n "$C_CYAN" ]]; then
        echo "$_SWEEP_PREVIEW" | sed -E "s/^([A-Za-z_.]+):/${C_CYAN}\1${C_RESET}:/" >&2
    else
        echo "$_SWEEP_PREVIEW" >&2
    fi
    _rule
    echo "" >&2

    if [[ "$YES" -ne 1 ]]; then
        printf '%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}Launch this sweep now, detached via nohup?${C_RESET} ${C_DIM}[y/N]${C_RESET} " >&2
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
# explicit flags plus --detached, then exit. The child ignores SIGHUP and its
# agent subshells inherit that, so closing the terminal leaves the sweep running.
if [[ "$DETACHED" -ne 1 ]]; then
    mkdir -p "$SEARCH_DIR/logs"
    NOHUP_LOG="$SEARCH_DIR/logs/nohup_${ALGORITHM}_${ENV_ARG}_$(date '+%Y%m%d_%H%M%S').log"
    nohup "$SCRIPT_DIR/search.sh" \
        --algorithm "$ALGORITHM" --env "$ENV_ARG" --num-agents "$NUM_AGENTS" \
        --method "$METHOD" --timeout "$PER_RUN_TIMEOUT" \
        --runs-per-agent "$RUNS_PER_AGENT" \
        ${GPU_ARG:+--gpu "$GPU_ARG"} \
        --detached > "$NOHUP_LOG" 2>&1 &
    disown
    _success "Launched (PID $!) — detached via nohup, safe to close this terminal."
    _info "Tail progress with: ${C_BOLD}tail -f $NOHUP_LOG${C_RESET}"
    echo "" >&2
    exit 0
fi

# ── Launch ──────────────────────────────────────────────────────────────── #
RUN_TS="$(date '+%Y%m%d_%H%M%S')"

for i in "${!ENVS[@]}"; do
    ENV="${ENVS[$i]}"
    GPU="${GPU_ARG:-$((i % NUM_GPUS))}"

    LOG_DIR="$SEARCH_DIR/logs/${ALGORITHM}_${ENV}_${RUN_TS}"
    mkdir -p "$LOG_DIR"

    _header "$ENV  ${C_DIM}(${ALGORITHM}, GPU ${GPU})${C_RESET}"

    # Project and sweep name both come from build_sweep.py: One fixed project
    # (c2rl-Search) for everything, with the sweep named
    # "<env>-<algorithm>". Relaunching the same env+algorithm therefore reuses
    # the same name — wandb sweep names are not unique and each launch still
    # gets its own sweep id, so the runs stay grouped by what they're sweeping
    # rather than by when it was launched.
    #
    # Generated yamls are per-launch artifacts, so they go to the log dir rather
    # than into search/ (which is versioned, and holds the space itself).
    SWEEP_YAML="$LOG_DIR/sweep.yaml"
    # `|| { ...; continue; }` on each fallible step below, because this script
    # runs under `set -euo pipefail`: without it a failure here aborts the whole
    # script, so with --env all a single bad env would silently take every
    # remaining env down with it instead of being skipped.
    if ! python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV" \
            --method "$METHOD" --out "$SWEEP_YAML" > /dev/null; then
        _error "Failed to build a sweep yaml for $ENV — skipping."
        continue
    fi

    # `|| true` twice, for two different set -e traps:
    #   - the command substitution itself fails if `wandb sweep` exits nonzero;
    #   - `pipefail` fails the grep pipeline when grep matches nothing, which is
    #     exactly the "sweep creation failed" case the check below handles.
    # Without these the -z/Error branch is unreachable dead code.
    SWEEP_INIT_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1) || true
    SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}' || true)
    if [[ -z "$SWEEP_ID" || "$SWEEP_ID" == *"Error"* ]]; then
        _error "Failed to create sweep for $ENV."
        echo "$SWEEP_INIT_OUTPUT" | sed 's/^/      /' >&2
        continue
    fi
    _success "Sweep ID created: ${C_BOLD}$SWEEP_ID${C_RESET}"
    _info "Starting $NUM_AGENTS self-restarting agent(s) in parallel..."

    for j in $(seq 1 "$NUM_AGENTS"); do
        LOGFILE="$LOG_DIR/agent_${j}.log"
        _info "  Agent $j (auto-restart, ${PER_RUN_TIMEOUT} watchdog) → ${C_DIM}${LOGFILE}${C_RESET}"
        (
            run_count=0
            while true; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] (re)starting wandb agent for $SWEEP_ID" >> "$LOGFILE"
                CUDA_VISIBLE_DEVICES=$GPU timeout "$PER_RUN_TIMEOUT" wandb agent --count 1 "$SWEEP_ID" >> "$LOGFILE" 2>&1
                status=$?
                run_count=$((run_count + 1))
                # bayes never exhausts its space, so --runs-per-agent is the
                # only thing that ever stops an agent short of being killed.
                if [[ "$RUNS_PER_AGENT" -gt 0 && "$run_count" -ge "$RUNS_PER_AGENT" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] wandb agent exited (status=$status) — reached ${RUNS_PER_AGENT}/${RUNS_PER_AGENT} runs, stopping" >> "$LOGFILE"
                    break
                fi
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] wandb agent exited (status=$status) — restarting in 5s" >> "$LOGFILE"
                sleep 5
            done
        ) &
    done

    _success "All $NUM_AGENTS agent(s) for $ENV launched in the background."
done

_header "All sweeps launched"
_info "Monitor progress on your W&B dashboard."
_info "Each agent restarts itself — kill this script's process group to stop."
echo "" >&2
wait
