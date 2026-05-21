# Debugging Guide

This guide covers interactive debugging for both local development and containerized training/serving environments.

---

## Prerequisites

Install dev dependencies (includes `ipdb` and `debugpy`):

```bash
pip install -r requirements_dev.txt
```

---

## Local Debugging with ipdb

`ipdb` drops you into an interactive Python shell at any point in the code.

### Usage

Uncomment the breakpoint already placed in the NaN loss guard of `train_graphsage.py`:

```python
if torch.isnan(loss):
    import ipdb; ipdb.set_trace()   # ← uncomment this line
    raise ValueError("NaN loss detected")
```

Then run training normally:

```bash
python -m multimodal_anti_money_laundering.train_graphsage --epochs 5 --max_nodes 8000
```

### Key ipdb commands

| Command | Action |
|---------|--------|
| `n` | Next line (step over) |
| `s` | Step into function |
| `c` | Continue to next breakpoint |
| `u` / `d` | Move up/down the call stack |
| `p <expr>` | Print expression |
| `pp <expr>` | Pretty-print expression |
| `q` | Quit debugger |

### Useful inspection commands at the NaN breakpoint

```python
# Inside ipdb session:
p loss                        # confirm it is nan
p logits[train_mask_t].min()  # check for -inf logits
p data.y[train_mask_t].unique()  # confirm labels are 0/1 only
p criterion.pos_weight        # verify class weight is finite
pp dict(model.named_parameters())  # inspect weight norms
```

---

## Containerized Debugging with debugpy + VS Code

`debugpy` lets VS Code attach to a running process inside Docker over TCP port 5678.

### Step 1 — Start the container with debugpy enabled

```bash
docker run --rm -it \
  -e DEBUGPY_ENABLE=1 \
  -p 5678:5678 \
  -v $(pwd)/src:/app/src \
  aml-train \
  python -m multimodal_anti_money_laundering.train_graphsage --epochs 5 --max_nodes 8000
```

The process will print:

```
debugpy listening on port 5678 — waiting for VS Code to attach …
```

and pause until VS Code connects.

### Step 2 — Attach VS Code

Open the **Run and Debug** panel (`Ctrl+Shift+D` / `Cmd+Shift+D`) and select **"Attach to Docker container (port 5678)"** from the dropdown, then press **Start Debugging (F5)**.

The configuration is already in [`.vscode/launch.json`](../.vscode/launch.json):

```json
{
  "name": "Attach to Docker container (port 5678)",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "localhost", "port": 5678 },
  "pathMappings": [
    { "localRoot": "${workspaceFolder}/src", "remoteRoot": "/app/src" }
  ]
}
```

### Step 3 — Set breakpoints

Click in the VS Code gutter on any line in `train_graphsage.py` or `train_bilstm.py`. Execution will pause there and you can inspect variables in the **Variables** panel or the **Debug Console**.

### Using Docker Compose

Add the `DEBUGPY_ENABLE` env var and port mapping to `docker-compose.yaml` for the relevant service:

```yaml
services:
  train:
    environment:
      - DEBUGPY_ENABLE=1
    ports:
      - "5678:5678"
```

Then:

```bash
docker compose up train
# VS Code → Attach to Docker container (port 5678)
```

---

## Debug Scenarios

### Scenario 1 — NaN loss at epoch N

**Symptom:** Training stops with `ValueError: NaN loss detected` at epoch N.

**Reproduce:**

```bash
python -m multimodal_anti_money_laundering.train_graphsage --lr 0.5 --epochs 20
```

A very high learning rate causes gradient explosion → NaN loss.

**Debug steps:**

1. Uncomment `import ipdb; ipdb.set_trace()` in the NaN guard (line ~468 of `train_graphsage.py`).
2. Run training. When it hits the breakpoint, inspect:

```python
p loss                             # → tensor(nan)
p logits[train_mask_t].max()       # → tensor(inf)  — overflow confirmed
p [p.grad.norm() for p in model.parameters() if p.grad is not None]
```

3. Large gradient norms confirm explosion.

**Fix:** Lower `--lr` to ≤ 0.001, or verify gradient clipping is active:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This line is already in the training loop. If it was accidentally removed, restore it before `optimizer.step()`.

---

### Scenario 2 — Shape mismatch in feature matrix

**Symptom:** `AssertionError: Expected 2D features (N, F)` or a PyTorch shape error during the first forward pass.

**Reproduce:** Pass a 1D feature slice instead of a 2D matrix:

```python
# Broken caller code:
features = data.x[0]   # shape (165,) instead of (1, 165)
```

**Debug steps:**

1. Set a breakpoint at the assertion block (`train_graphsage.py`, `validate_inputs` function, ~line 180).
2. Inspect shapes:

```python
p features.shape      # → torch.Size([165])  — missing batch dimension
p features.ndim       # → 1
```

3. Walk up the call stack with `u` to find where the slice was taken.

**Fix:** Ensure the caller passes a 2D tensor:

```python
features = data.x[0].unsqueeze(0)   # → shape (1, 165)
# or for a batch:
features = data.x[node_ids]          # → shape (N, 165)
```

---

## Tips

- Set `LOG_LEVEL=DEBUG` when running locally to get per-line timing and assertion logs without a debugger:
  ```bash
  LOG_LEVEL=DEBUG python -m multimodal_anti_money_laundering.train_graphsage
  ```
- Log files rotate at 5 MB (3 backups) in `logs/`. Check `logs/graphsage_training.log` for the full trace after a crash.
- Use `PYTHONFAULTHANDLER=1` to get a C-level traceback on segfaults:
  ```bash
  PYTHONFAULTHANDLER=1 python -m multimodal_anti_money_laundering.train_graphsage
  ```
