# Debugging Guide — Multimodal AML System

This guide covers interactive debugging for local development and inside Docker containers, plus step-by-step solutions for three common failure scenarios in this project.

---

## 1. Local Interactive Debugging (pdb / ipdb)

### Setup

Install dev dependencies (includes `ipdb`):

```bash
pip install -r requirements_dev.txt
```

### Drop a breakpoint in code

```python
from multimodal_anti_money_laundering.utils.debug import set_trace

# Inside any training or serving script:
set_trace()  # only fires when AML_DEBUG=1
```

Run with the debug flag:

```bash
AML_DEBUG=1 python -m multimodal_anti_money_laundering.train_bilstm
```

When execution reaches `set_trace()` you get an ipdb shell:

```
> /app/src/.../train_bilstm.py(87)train()
-> loss = criterion(logits, labels)
ipdb> p logits.shape
torch.Size([128, 1])
ipdb> p labels.unique()
tensor([0., 1.])
ipdb> n         # next line
ipdb> c         # continue
ipdb> q         # quit
```

**Useful ipdb commands:**

| Command | Action |
|---------|--------|
| `n` | Next line (step over) |
| `s` | Step into function call |
| `c` | Continue to next breakpoint |
| `u` / `d` | Move up / down the call stack |
| `p <expr>` | Print expression |
| `pp <expr>` | Pretty-print |
| `l` | List source around current line |
| `bt` | Full traceback |
| `q` | Quit debugger |

---

## 2. Remote Debugging Inside Docker (VS Code)

Use this when you need to debug code running inside a Docker container.

### Step 1 — Add the debugger call to your script

```python
from multimodal_anti_money_laundering.utils.debug import attach_remote_debugger

attach_remote_debugger()   # blocks until VS Code attaches; only runs when AML_DEBUG=1
```

### Step 2 — Start the debug container

```bash
docker compose --profile debug up debug-train
```

The container starts, prints `debugpy listening on 0.0.0.0:5678 — waiting for client...`, and blocks.

### Step 3 — Attach from VS Code

Add this to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "AML Remote Debug",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}/src",
          "remoteRoot": "/app/src"
        }
      ]
    }
  ]
}
```

Press **F5** (or Run → Start Debugging → "AML Remote Debug"). Execution resumes inside the container and you can set breakpoints in VS Code normally.

### Step 4 — Stop the debug session

```bash
docker compose --profile debug down
```

---

## 3. Inline Assertion Helpers

The `debug` module provides helpers you can add around suspect code without a full breakpoint session.

### Check tensor shapes

```python
from multimodal_anti_money_laundering.utils.debug import assert_tensor_shape

graph_emb = graphsage_encoder(data)       # should be (batch, 128)
text_emb  = distilbert_encoder(tokens)   # should be (batch, 64)
ts_emb    = bilstm_encoder(sequences)    # should be (batch, 64)

assert_tensor_shape(graph_emb, (None, 128), "graph_emb")
assert_tensor_shape(text_emb,  (None, 64),  "text_emb")
assert_tensor_shape(ts_emb,    (None, 64),  "ts_emb")

fused = torch.cat([graph_emb, text_emb, ts_emb], dim=1)  # (batch, 256)
assert_tensor_shape(fused, (None, 256), "fused")
```

### Detect NaN / Inf

```python
from multimodal_anti_money_laundering.utils.debug import check_for_nans

if check_for_nans(loss, "train_loss"):
    # logged automatically; add a breakpoint here to investigate
    set_trace()
```

### Log memory usage (OOM prevention)

```python
from multimodal_anti_money_laundering.utils.debug import log_memory_usage

log_memory_usage("before_graph_load")
data = load_elliptic_graph()
log_memory_usage("after_graph_load")
```

---

## 4. Debug Scenarios

### Scenario 1 — Out-of-Memory (OOM) Inside Docker

**Symptom:**

```
Killed
# or
torch.cuda.OutOfMemoryError: CUDA out of memory.
# or Docker exits with code 137
```

**Root cause:** Docker containers have a default memory limit. Loading the full Elliptic graph (203K nodes × 166 features as float32 ≈ 128 MB) plus model parameters and gradients can exceed the container's memory ceiling, especially during GraphSAGE mini-batch sampling.

**Diagnosis steps:**

```bash
# 1. Check container memory limit
docker inspect aml_graphsage_train | grep -i memory

# 2. Watch live memory usage while the container runs
docker stats aml_graphsage_train

# 3. Add memory logging to the training script
from multimodal_anti_money_laundering.utils.debug import log_memory_usage
log_memory_usage("before_batch")
```

**Fix options:**

```yaml
# docker-compose.yaml — raise memory ceiling
services:
  graphsage-train:
    mem_limit: 8g        # add this line
    memswap_limit: 8g
```

```python
# train_graphsage.py — reduce batch size
loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=256)  # was 512
```

```python
# Use fp16 to halve activation memory
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    out = model(data.x, data.edge_index)
```

**Prevention:** Call `log_memory_usage()` at the start of each epoch and after loading data to catch memory growth early.

---

### Scenario 2 — Shape Mismatch in the Fusion Head

**Symptom:**

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x192 and 256x128)
```

**Root cause:** The fusion MLP expects input size 256 = 128 (GraphSAGE) + 64 (DistilBERT) + 64 (BiLSTM). If any encoder output dim changes (e.g. BiLSTM hidden_size doubled to 128 → output 256 instead of 128 → concat = 384) the fusion linear layer breaks.

**Diagnosis steps:**

```python
# Add these assertions before the fusion forward pass
from multimodal_anti_money_laundering.utils.debug import assert_tensor_shape

assert_tensor_shape(graph_emb, (None, 128), "graph_emb")   # GraphSAGE output
assert_tensor_shape(text_emb,  (None, 64),  "text_emb")    # DistilBERT CLS proj
assert_tensor_shape(ts_emb,    (None, 64),  "ts_emb")      # BiLSTM hidden state

# Then inspect at runtime
AML_DEBUG=1 python -m multimodal_anti_money_laundering.train_model
```

When `assert_tensor_shape` fires it prints the offending shape:

```
AssertionError: ts_emb: dim 1 mismatch — expected 64, got 128 (shape (32, 128))
```

**Fix options:**

```python
# Option A — fix the BiLSTM output projection to always emit 64-dim
self.proj = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional

# Option B — update config.py to match actual output sizes
@dataclass
class FusionConfig:
    graph_dim: int = 128
    text_dim: int = 64
    ts_dim: int = 64            # change this if BiLSTM hidden_size changes
    input_dim: int = 256        # must equal graph_dim + text_dim + ts_dim
```

**Prevention:** Use `assert_tensor_shape` at the top of every forward pass during development. Remove them (or guard with `if AML_DEBUG`) before production.

---

### Scenario 3 — MLflow Not Logging Inside Docker

**Symptom:**

Training finishes with no errors but the MLflow UI shows no runs, or you see:

```
WARNING mlflow.tracking: Run with UUID ... is already finished.
MlflowException: Run ... not found
```

**Root cause:** The container's `MLFLOW_TRACKING_URI` defaults to `file:///app/mlruns` (inside the container filesystem). If the `./mlruns` host directory is not volume-mounted, runs are written inside the ephemeral container layer and lost on exit. Alternatively the URI points to a remote MLflow server that is unreachable from inside the container network.

**Diagnosis steps:**

```bash
# 1. Confirm what URI the container sees
docker exec aml_train env | grep MLFLOW

# 2. Check if mlruns volume is mounted
docker inspect aml_train | grep -A5 Mounts

# 3. List runs from inside the container
docker exec aml_train python -c "import mlflow; print(mlflow.search_experiments())"
```

**Fix — local filesystem (development):**

```yaml
# docker-compose.yaml — mount mlruns so runs persist on the host
services:
  train:
    volumes:
      - ./mlruns:/app/mlruns   # ← ensure this line exists
    environment:
      - MLFLOW_TRACKING_URI=file:///app/mlruns
```

**Fix — remote MLflow server:**

```yaml
services:
  train:
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

```bash
# Verify network connectivity from inside the container
docker exec aml_train curl -s http://mlflow-server:5000/health
```

**Fix — experiment already exists under a different name:**

```python
# train_bilstm.py — always set experiment before starting run
mlflow.set_experiment("bilstm-training")
with mlflow.start_run():
    mlflow.log_params(...)
```

**Prevention:** Always include `./mlruns:/app/mlruns` in every service's `volumes` block in `docker-compose.yaml`. Add a smoke test that asserts `len(mlflow.search_runs()) > 0` after a training run completes.

---

## 5. Quick Reference

| Tool | When to use |
|------|-------------|
| `set_trace()` + `AML_DEBUG=1` | Local step-through debugging |
| `attach_remote_debugger()` + VS Code | Debugging inside a running Docker container |
| `assert_tensor_shape()` | Catch shape bugs before the forward pass |
| `check_for_nans()` | Diagnose exploding / vanishing gradients |
| `log_memory_usage()` | Track RSS and GPU memory to prevent OOM |
| `log_model_summary()` | Verify model architecture at startup |
| `docker stats <container>` | Live CPU / memory monitoring |
| `docker exec <container> env` | Inspect environment variables inside container |
