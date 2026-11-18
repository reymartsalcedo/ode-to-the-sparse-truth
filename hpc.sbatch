#!/bin/bash
# Every result of the paper on one cluster node, step by step (README.md, "On a cluster").
#
#   sbatch hpc.sbatch                       # every step, 1 to 9
#   STEPS="1 2 3" sbatch hpc.sbatch         # only these
#   PYTHON=/path/to/python3 sbatch hpc.sbatch   # another Python (default: python3 on the PATH)
#   PROGRESS=500 sbatch hpc.sbatch          # the exhaustive searches print a line every 500 supports (default 100)
#
# Everything is printed to sparse_truth_<job>.out as it happens, and each step also to results/step<n>.log.
# A step that was interrupted resumes when the job is submitted again: every scored support is reread. Stage 1 is
# loaded from results/stage1.pt, the paper's networks (HPC job 12514); delete the whole results folder to train it
# instead (it then depends on the node and the number of threads).
#
#   1  size known, k = 7: SBL, l0-IHT-HTP, the climb, the ODE refit, Stage 3                ~1 h
#   2  the exhaustive search, k = 7, sigma = 2e-7 (77,280 supports)                         ~4 h
#   3  the figures: fig_rivals_half.pdf and the solution panels                             ~1 min
#   4  size unknown, k = 19, ..., 2                                                         ~8 h
#   5  the exhaustive search from exact derivatives (the diagnostic check)                  ~4 h
#   6  the exhaustive search at sigma = 1e-5                                                ~4 h
#   7  the exhaustive searches at k = 6, 8, 9, 10, then k = 7 again for the posterior        ~1-2 days
#   8  the figures again (the panel of the size-unknown run)                                ~1 min
#   9  the SINDy baselines (baselines.py; needs pysindy, and numpyro for SBR)                ~5 min
#SBATCH --job-name=sparse_truth
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --output=sparse_truth_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# module load python                       # if the cluster provides Python as a module
PY=${PYTHON:-python3}                      # any Python with numpy, scipy, torch and matplotlib
"$PY" -c "import numpy, scipy, torch, matplotlib" || { echo "$PY lacks numpy, scipy, torch or matplotlib: set PYTHON="; exit 1; }
export PYTHONUNBUFFERED=1
STEPS=${STEPS:-"1 2 3 4 5 6 7 8 9"}
PROGRESS=${PROGRESS:-100}
W=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
mkdir -p results
"$PY" -c "import platform, numpy, scipy, torch; print(f'python {platform.python_version()}, numpy {numpy.__version__}, \
scipy {scipy.__version__}, torch {torch.__version__}, {torch.get_num_threads()} torch threads')"
echo "$(date '+%F %T') on $(hostname): steps $STEPS, $W scoring processes"

step() {                                   # $1 = the step, then its command; the job goes on after a failure
  local n=$1
  shift
  echo "== $(date '+%F %T') step $n: $*"
  "$@" 2>&1 | tee "results/step$n.log" || echo "   step $n FAILED (exit $?): see results/step$n.log"
}
for n in $STEPS; do
  case $n in
    1) step 1 "$PY" pipeline.py --sigma 2e-7 --k 7 --workers "$W"
       echo "   Stage 1 of the paper's run (HPC job 12514): PINN loss 3.504868e-06; this run: $(grep -m1 -o 'PINN loss [0-9.e+-]*' results/step1.log)" ;;
    2) step 2 "$PY" pipeline.py --sigma 2e-7 --k 7 --exhaustive --workers "$W" --progress "$PROGRESS" ;;
    3) step 3 "$PY" plot.py ;;
    4) step 4 "$PY" pipeline.py --sigma 2e-7 --workers "$W" ;;
    5) step 5 "$PY" pipeline.py --sigma 2e-7 --k 7 --exhaustive --exact-start --workers "$W" --progress "$PROGRESS" ;;
    6) step 6 "$PY" pipeline.py --sigma 1e-5 --k 7 --exhaustive --workers "$W" --progress "$PROGRESS" ;;
    7) step 7 bash -c "for k in 6 8 9 10 7; do $PY pipeline.py --sigma 2e-7 --k \$k --exhaustive --workers $W --progress $PROGRESS || exit 1; done" ;;
    8) step 8 "$PY" plot.py ;;
    9) step 9 "$PY" baselines.py ;;
    *) echo "unknown step $n" ;;
  esac
done
echo "$(date '+%F %T') done"
