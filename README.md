# laya 0.3.5 → 0.3.20: same answers from `--model laya`

Evidence for ThinkFlowLab/system1-agents#37: `--model laya` gives the same answers on main and on that pull
request, on CPU and on MPS, and the answers move only when Laya's MPS fp16 is turned back on. This branch holds
only the scripts and the outputs; they run against a clone of system1-agents.

| file | does |
|---|---|
| `run.sh` | checks out both commits, installs each one's lock, runs every case below, prints the tables |
| `laya_ab.py` | one fresh process: load through `LayaModel.from_env()`, answer the 30 ticket-router tickets, write a JSON |
| `compare.py` | the tables from those JSONs |
| `records/` | the 28 JSONs of the recorded run below |

## Steps

1. Use a Mac with Apple silicon for the MPS cases (on other machines, step 4 with `DEVICES=cpu`), with
   `git` and `uv` installed.
2. Get system1-agents with the pull request's commits, and this branch:

   ```sh
   git clone https://github.com/ThinkFlowLab/system1-agents
   git -C system1-agents fetch origin pull/37/head
   git clone --branch evidence/laya-upgrade https://github.com/cacheline999/system1-agents laya-upgrade
   ```

3. Fetch both checkpoints once, about 1.4 GB (`english` 804 MB, `multilingual` 614 MB):

   ```sh
   cd system1-agents
   uv run --extra laya python -c "import laya; laya.load('convaiinnovations/laya', device='cpu'); laya.load('convaiinnovations/laya', device='cpu', subfolder='multilingual')"
   cd ..
   ```

4. Run every case, baseline `222e656` (main at the pull request's base) against head `c197fe6`:

   ```sh
   HF_HUB_OFFLINE=1 laya-upgrade/run.sh system1-agents 222e656 c197fe6 my-records
   ```

   About 30 minutes on an M1 Pro. `RUNS` (default 3), `DEVICES` (`cpu mps`) and `CHECKPOINTS`
   (`english multilingual`) narrow it. The checkouts go to a temporary folder and are removed at the end.
   `python3 laya-upgrade/compare.py laya-upgrade/records` prints the recorded run's tables without running
   anything.

## What it runs

Each run is one process. It loads the model, then for each of the 30 tickets in
`s1a/agents/_data/ticket_router_eval.jsonl` asks one choice question, one noul question, and one request of
six questions (the choice and five nouls). Six questions is the case that changed: from 0.3.10 Laya answers a
request of five or more questions in fp16 on MPS, and #37 keeps it in fp32.

| case | runs |
|---|---|
| baseline and head, each checkpoint, each device | `RUNS` each, alternating baseline and head |
| head with `LAYA_MPS_AMP_MIN_ROWS=5` (Laya's default turned back on), MPS | 1 per checkpoint |
| baseline without its weight-init skip, CPU | 1 per checkpoint |

Every JSON records the commit, the laya, torch and transformers versions, the checkpoint revision, the
device and the Laya variables that were set.

## Reading the output

```
| checkpoint | device | load s, baseline | load s, head | ... | answers |
| convaiinnovations/laya | mps | 9.8 / 7.4 / 7.8 | 9.6 / 7.7 / 7.9 | ... | identical (3 + 3 runs) |
```

- `answers` is `identical` when every choice, probability, confidence and noul value of every run equals the
  first baseline run's. That is the claim; a `DIFFERENT` row disproves it.
- The `LAYA_MPS_AMP_MIN_ROWS=5` lines show what the fp32 default prevents: the largest probability change
  and each decision that changed, by ticket id. On an M1 Pro, `multilingual` changed two.
- The no-skip lines show the load the weight-init skip saves, now done by laya itself.
- Load and latency columns are for comparing the two sides of one row, run alternately on the same machine.
  The first load after the checkout can include reading the model files from disk.

## Recorded run

`records/` is the output of step 4 on 2026-10-05: Apple M1 Pro (16 GB), macOS 26.1, Python 3.14, torch 2.14.0,
transformers 5.17.0, checkpoint revision `7b928d8`, baseline `222e656` (laya 0.3.5), head `c197fe6` (laya 0.3.20).
The machine had other work running (1-min load average about 6 to 10). All four rows are `identical`; with
`LAYA_MPS_AMP_MIN_ROWS=5`, `multilingual` changed `t_6cb98186dde0` (pick, logistics to returns) and
`t_f5251390fb88` (cancel, 0.4959 to 0.5074); without the skip the baseline loaded in 36.8 s (`english`) and
42.5 s (`multilingual`) against 6 to 10 s with it. The first `english` CPU load of each side (26.2 and 24.8 s)
read the model files from disk.

Not covered: the `typed-decisions` checkpoint, a local checkpoint path, transformers 4.x, CUDA, and Apple chips
other than the M1 Pro.
