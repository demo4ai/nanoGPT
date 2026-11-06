#!/usr/bin/env bash
# Launch one arm only. No implicit GPU selection, download, or experiment queue.
set -euo pipefail
cd "$(dirname "$0")"
arm=${1:-}
case "$arm" in
  baseline|attention|attention_lm_head) ;;
  *) echo "Usage: bash run_multimax.sh {baseline|attention|attention_lm_head} [--key=value ...]" >&2; exit 2 ;;
esac
shift
python_bin=${PYTHON_BIN:-python}
gpus=${NPROC_PER_NODE:-1}
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NPROC_PER_NODE must be a positive integer' >&2
  exit 2
fi
mkdir -p logs
log="logs/${arm}_$(date +%Y%m%d_%H%M%S)_$$.log"
if [[ "$gpus" == 1 ]]; then
  "$python_bin" -u train.py config/train_multimax_common.py "config/multimax_${arm}.py" "$@" 2>&1 | tee "$log"
else
  "$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node="$gpus" \
    train.py config/train_multimax_common.py "config/multimax_${arm}.py" "$@" 2>&1 | tee "$log"
fi
