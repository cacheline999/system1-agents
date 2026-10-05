#!/usr/bin/env bash
# Every run behind the README: baseline and head, both checkpoints, each device, RUNS times, then the controls.
# Usage: run.sh <repository> <baseline commit> <head commit> [out dir]
set -euo pipefail

repo=$(cd "$1" && pwd)
base=$2
head=$3
out=$(mkdir -p "${4:-records}" && cd "${4:-records}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
runs=${RUNS:-3}
devices=${DEVICES:-"cpu mps"}
checkpoints=${CHECKPOINTS:-"english multilingual"}

work=$(mktemp -d)
cleanup() {
  git -C "$repo" worktree remove --force "$work/baseline" 2>/dev/null || true
  git -C "$repo" worktree remove --force "$work/head" 2>/dev/null || true
  rm -rf "$work"
}
trap cleanup EXIT

for side in baseline head; do
  commit=$base
  [ "$side" = head ] && commit=$head
  git -C "$repo" worktree add --quiet --detach "$work/$side" "$commit"
  echo "$side: $commit"
  (cd "$work/$side" && uv sync --quiet --compile-bytecode --extra dev --extra laya)
  "$work/$side/.venv/bin/python" -c "import torch, transformers, laya" >/dev/null 2>&1  # first imports off the clock
done

# one <side> <checkpoint> <device> <name> [VAR=value ...] [-- laya_ab.py flags]
one() {
  local side=$1 checkpoint=$2 device=$3 name=$4
  shift 4
  local subfolder=""
  [ "$checkpoint" = multilingual ] && subfolder=multilingual
  local vars=() flags=()
  while [ $# -gt 0 ]; do
    if [ "$1" = -- ]; then shift; flags=("$@"); break; fi
    vars+=("$1"); shift
  done
  echo "  $name"
  (cd "$work/$side" && env LAYA_MODEL= LAYA_MAX_LEN= LAYA_HEAD_MAX_LEN= LAYA_MPS_AMP_MIN_ROWS= \
    LAYA_SUBFOLDER="$subfolder" LAYA_DEVICE="$device" ${vars[@]+"${vars[@]}"} \
    .venv/bin/python "$here/laya_ab.py" --label "$side" --out "$out/$name.json" ${flags[@]+"${flags[@]}"} >/dev/null 2>&1)
}

for i in $(seq "$runs"); do
  for checkpoint in $checkpoints; do
    for device in $devices; do
      one baseline "$checkpoint" "$device" "baseline-$checkpoint-$device-$i"
      one head "$checkpoint" "$device" "head-$checkpoint-$device-$i"
    done
  done
done
for checkpoint in $checkpoints; do
  case " $devices " in *" mps "*)
    one head "$checkpoint" mps "head-$checkpoint-mps-override5" LAYA_MPS_AMP_MIN_ROWS=5 ;;
  esac
  one baseline "$checkpoint" cpu "baseline-$checkpoint-cpu-noskip" -- --no-skip
done

"$work/head/.venv/bin/python" "$here/compare.py" "$out"
