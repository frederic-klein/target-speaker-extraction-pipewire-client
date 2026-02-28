# Security Mitigation Plan: TSE-PipeWire

**Companion to:** `docs/security-audit.md`
**Purpose:** Actionable mitigation steps to run after implementation is complete
**Execution:** Run all steps before first production use

---

## Phase 1: Model Supply Chain Hardening

### 1.1 Sandboxed Model Export

Run the model download and ONNX export in an isolated environment to contain any malicious code in PyTorch checkpoints.

**Steps:**

```bash
# Option A: Podman/Docker container (preferred)
podman run --rm -it \
  --network=allow \
  -v ./models:/output:z \
  python:3.12-slim bash -c '
    pip install torch wesep modelscope pyyaml &&
    python /output/export_bsrnn_onnx.py --output-dir /output
  '

# Option B: Disposable VM / devbox shell with network
# Run export, copy ONNX files out, destroy environment
```

**Verification:**
- [ ] Export completed in isolated environment
- [ ] Only `speaker_encoder.onnx` and `tse_model.onnx` copied out
- [ ] Container/VM destroyed after export

### 1.2 Pin Model Checksums

After exporting ONNX models, compute and record SHA-256 checksums.

**Steps:**

```bash
# Compute checksums after trusted export
sha256sum models/speaker_encoder.onnx models/tse_model.onnx > models/checksums.sha256

# Content should look like:
# <hash>  models/speaker_encoder.onnx
# <hash>  models/tse_model.onnx
```

**Add checksum verification to application startup** — add a helper that verifies model files against pinned hashes before loading them into ONNX Runtime. Implementation sketch:

```python
# In config.py or a new verify.py
import hashlib

EXPECTED_CHECKSUMS = {
    "speaker_encoder.onnx": "<sha256-hash-here>",
    "tse_model.onnx": "<sha256-hash-here>",
}

def verify_model_integrity(model_path: Path) -> bool:
    expected = EXPECTED_CHECKSUMS.get(model_path.name)
    if expected is None:
        return True  # Unknown model, skip
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return sha256 == expected
```

**Verification:**
- [ ] `checksums.sha256` file exists and is committed to the repository
- [ ] Application refuses to load models with mismatched checksums
- [ ] Checksums match across independent exports (reproducibility check)

### 1.3 ONNX Model Structure Inspection

Verify exported ONNX models contain only expected operators and no suspicious nodes.

**Steps:**

```bash
pip install onnx
```

```python
import onnx

def inspect_model(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)

    print(f"Model: {path}")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  Graph nodes: {len(model.graph.node)}")

    op_types = set(n.op_type for n in model.graph.node)
    print(f"  Operators used: {sorted(op_types)}")

    # Flag unusual operators
    suspicious = {"If", "Loop", "Scan", "SequenceConstruct", "ConcatFromSequence"}
    flagged = op_types & suspicious
    if flagged:
        print(f"  WARNING: Unusual operators found: {flagged}")

    # Check model size is reasonable
    import os
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    # Print inputs/outputs
    for inp in model.graph.input:
        print(f"  Input: {inp.name} {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    for out in model.graph.output:
        print(f"  Output: {out.name} {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

inspect_model("models/speaker_encoder.onnx")
inspect_model("models/tse_model.onnx")
```

**Expected results:**
- `speaker_encoder.onnx`: Input `fbank` (1, N, 80) -> Output `embedding` (1, 256). Operators: Conv, Relu, BatchNormalization, Gemm, GlobalAveragePool, etc. Size: ~25-50 MB.
- `tse_model.onnx`: Input `audio` (1, T) + `embedding` (1, 256) -> Output `output` (1, T). Operators: Conv, LSTM/GRU, Sigmoid, Mul, etc. Size: ~10-100 MB.

**Verification:**
- [ ] Both models pass `onnx.checker.check_model()`
- [ ] No suspicious operators flagged
- [ ] Model sizes are within expected range
- [ ] Input/output shapes match expected signatures

---

## Phase 2: Dependency Pinning

### 2.1 Lock Runtime Dependencies

Pin exact versions of all runtime dependencies to prevent supply chain attacks via package updates.

**Steps:**

```bash
# Generate locked requirements from current working environment
pip freeze | grep -E '^(onnxruntime|numpy|sounddevice|soundfile|scipy|click|toml)==' \
  > requirements-locked.txt
```

Consider adding a `[tool.pip-audit]` section or running `pip-audit` periodically:

```bash
pip install pip-audit
pip-audit -r requirements-locked.txt
```

**Verification:**
- [ ] `requirements-locked.txt` exists with pinned versions (==)
- [ ] `pip-audit` reports no known vulnerabilities
- [ ] Application installs and runs correctly with locked versions

### 2.2 Lock Export-Time Dependencies (separate)

Create a separate requirements file for the one-time export environment:

```bash
# In the export container, after successful export:
pip freeze > requirements-export-locked.txt
```

**Verification:**
- [ ] `requirements-export-locked.txt` is committed for reproducibility
- [ ] Export-time packages are never installed in the runtime environment

---

## Phase 3: Runtime Network Isolation

### 3.1 Verify Zero Network Calls

Automated verification that the application makes no outbound connections.

**Steps:**

```bash
# Run the application under strace to log all network syscalls
strace -f -e trace=network -o /tmp/tse-network.log \
  tse-pipewire start --profile test --input-device default &
TSE_PID=$!
sleep 10
kill $TSE_PID

# Check for any connect() calls (excluding Unix domain sockets)
grep -v 'AF_UNIX\|AF_LOCAL' /tmp/tse-network.log | grep 'connect\|sendto\|sendmsg'
# Expected: empty output (no network connections)
```

**Alternative: Use unshare to deny network access entirely:**

```bash
# Run without network namespace — will fail if any network call is attempted
unshare --net tse-pipewire start --profile test --input-device default
```

**Verification:**
- [ ] `strace` shows zero `connect()` calls to IP addresses
- [ ] Application runs successfully under `unshare --net` (no network namespace)
- [ ] Only Unix domain socket connections (PipeWire IPC) are present

### 3.2 Firewall Rule (Defense-in-Depth)

For production deployments, add an optional systemd service hardening:

```ini
# In tse-pipewire.service [Service] section:
RestrictAddressFamilies=AF_UNIX
IPAddressDeny=any
```

This prevents the process from creating any IP sockets, even if a dependency is compromised.

**Verification:**
- [ ] systemd service starts successfully with network restrictions
- [ ] Audio processing works correctly with restrictions active
- [ ] Attempting `curl` or similar from within the service fails

---

## Phase 4: Sensitive Data Protection

### 4.1 File Permissions for Enrollment Data

Speaker embeddings and enrollment recordings are biometric data.

**Steps:**

```bash
# Set restrictive permissions on profile data
chmod 700 ~/.local/share/tse-pipewire/profiles/
chmod 600 ~/.local/share/tse-pipewire/profiles/*.npy
chmod 600 ~/.local/share/tse-pipewire/profiles/*.wav

# Set restrictive permissions on model files
chmod 700 ~/.local/share/tse-pipewire/models/
chmod 600 ~/.local/share/tse-pipewire/models/*.onnx
```

Consider adding permission enforcement to the application's config/enrollment code:

```python
import os
os.makedirs(profiles_dir, mode=0o700, exist_ok=True)
# After writing enrollment files:
os.chmod(enrollment_wav, 0o600)
os.chmod(embedding_npy, 0o600)
```

**Verification:**
- [ ] Profile directory is `drwx------` (700)
- [ ] All `.npy` and `.wav` files are `-rw-------` (600)
- [ ] Other users on the system cannot read enrollment data

### 4.2 Enrollment Data Retention Policy

Document and enforce how long enrollment data is kept:

- Enrollment WAV recordings should be deletable after embedding extraction
- Provide a `tse-pipewire delete-profile --name <name>` command
- Securely overwrite deleted files (optional: `shred`)

**Verification:**
- [ ] `delete-profile` command exists and removes both WAV and NPY files
- [ ] No orphaned enrollment data remains after profile deletion

---

## Phase 5: Automated Security Checks

### 5.1 CI/CD Security Script

Create `scripts/security_check.sh` to run as part of the development workflow:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Security Check: TSE-PipeWire ==="
PASS=0
FAIL=0

# Check 1: No network imports in runtime code
echo -n "[1/6] Checking for network imports in src/... "
if grep -rE 'import (requests|urllib|httpx|http\.client|socket|aiohttp)' src/; then
    echo "FAIL: Network library imported in runtime code"
    FAIL=$((FAIL + 1))
else
    echo "PASS"
    PASS=$((PASS + 1))
fi

# Check 2: No hardcoded URLs in runtime code
echo -n "[2/6] Checking for URLs in src/... "
if grep -rE 'https?://' src/; then
    echo "FAIL: URL found in runtime code"
    FAIL=$((FAIL + 1))
else
    echo "PASS"
    PASS=$((PASS + 1))
fi

# Check 3: Model checksums exist
echo -n "[3/6] Checking model checksums file... "
if [ -f models/checksums.sha256 ]; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL: models/checksums.sha256 not found"
    FAIL=$((FAIL + 1))
fi

# Check 4: Model checksums valid (if models present)
echo -n "[4/6] Verifying model integrity... "
if [ -f models/checksums.sha256 ] && [ -f models/speaker_encoder.onnx ]; then
    if sha256sum -c models/checksums.sha256 --quiet 2>/dev/null; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL: Checksum mismatch"
        FAIL=$((FAIL + 1))
    fi
else
    echo "SKIP (models not present)"
fi

# Check 5: No pickle/torch.load in runtime code
echo -n "[5/6] Checking for unsafe deserialization in src/... "
if grep -rE '(pickle\.load|torch\.load|joblib\.load)' src/; then
    echo "FAIL: Unsafe deserialization found in runtime code"
    FAIL=$((FAIL + 1))
else
    echo "PASS"
    PASS=$((PASS + 1))
fi

# Check 6: pip-audit (if available)
echo -n "[6/6] Running pip-audit... "
if command -v pip-audit &>/dev/null; then
    if pip-audit 2>/dev/null; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL: Vulnerabilities found"
        FAIL=$((FAIL + 1))
    fi
else
    echo "SKIP (pip-audit not installed)"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
```

**Verification:**
- [ ] Script runs without errors
- [ ] All checks pass on a clean build

---

## Mitigation Checklist Summary

Run these checks after implementation is complete and before first use:

| # | Action | Priority | Status |
|---|---|---|---|
| 1 | Export models in sandboxed container | High | [ ] |
| 2 | Compute and pin ONNX model SHA-256 checksums | High | [ ] |
| 3 | Inspect ONNX model structure for unexpected operators | High | [ ] |
| 4 | Verify zero network calls with strace | High | [ ] |
| 5 | Test runtime under `unshare --net` | High | [ ] |
| 6 | Pin all runtime dependency versions | Medium | [ ] |
| 7 | Run `pip-audit` on locked dependencies | Medium | [ ] |
| 8 | Set file permissions on profiles (700/600) | Medium | [ ] |
| 9 | Add systemd `RestrictAddressFamilies` / `IPAddressDeny` | Medium | [ ] |
| 10 | Create `scripts/security_check.sh` and run it | Medium | [ ] |
| 11 | Verify export-time deps are never in runtime env | Low | [ ] |
| 12 | Add `delete-profile` command for data retention | Low | [ ] |
