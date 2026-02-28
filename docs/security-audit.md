# Security Audit Report: TSE-PipeWire

**Project:** tse-pipewire (Target Speaker Extraction as PipeWire Client)
**Date:** 2026-02-28
**Scope:** Supply chain analysis of WeSep BSRNN and all upstream dependencies
**Classification:** Internal

---

## 1. Executive Summary

This project processes **live microphone audio** and **speaker voiceprints** (biometric data) on the user's local machine. The runtime application is well-isolated with zero network calls. However, the **model setup phase** downloads opaque binary model weights from Chinese cloud infrastructure (Alibaba ModelScope) through a chain of unverified dependencies. The primary risk is not runtime data exfiltration but **supply chain compromise during model preparation**.

**Overall risk rating: MEDIUM**
Runtime risk: LOW | Model supply chain risk: MEDIUM-HIGH | Dependency risk: LOW

---

## 2. Data Assets at Risk

| Asset | Sensitivity | Location |
|---|---|---|
| Live microphone audio | High (continuous speech) | In-memory, PipeWire buffers |
| Enrollment recording | High (stored WAV) | `~/.local/share/tse-pipewire/profiles/` |
| Speaker embedding | Medium (biometric vector, 256-dim) | `~/.local/share/tse-pipewire/profiles/*.npy` |
| ONNX model weights | Low (public models) | `~/.local/share/tse-pipewire/models/` |
| User config | Low | `~/.config/tse-pipewire/config.toml` |

---

## 3. Supply Chain Analysis

### 3.1 Model Origin Chain

```
ModelScope.cn (Alibaba Cloud, China)
  └── damo/speech_bsrnn_ecapa-tdnn_16k_vox1
        ├── model.pt (PyTorch weights, opaque binary)
        └── config.yaml
              │
              ▼  scripts/export_bsrnn_onnx.py
        ┌─────────────────┐
        │ PyTorch + WeSep  │  ← installed via pip during export
        │ (executes model) │
        └────────┬────────┘
                 │  torch.onnx.export()
                 ▼
        speaker_encoder.onnx
        tse_model.onnx
              │
              ▼  (runtime, no network)
        ONNX Runtime (inference only)
```

### 3.2 Model Download — No Integrity Verification

**Finding: MEDIUM-HIGH**

`scripts/export_bsrnn_onnx.py:22-25` uses `modelscope.hub.snapshot_download()` to fetch model weights. This:

- Downloads from `https://modelscope.cn/` (Alibaba Cloud, Chinese jurisdiction)
- Uses the `modelscope` PyPI package which resolves download URLs dynamically via API
- Performs **no checksum or signature verification** of downloaded files
- Caches to a local directory without integrity checking on subsequent loads

An attacker with access to ModelScope CDN or a MITM position could substitute model weights. Since the weights are loaded into PyTorch and executed (`model.eval()`, `model.separate()`), a malicious model could execute arbitrary code during the export phase.

### 3.3 Export-Time Dependencies

The following packages are installed **only during model export** (`scripts/download_models.sh:12`), not at runtime:

| Package | Source | Risk | Notes |
|---|---|---|---|
| `torch` | PyPI (Meta) | Low | Well-audited, massive user base |
| `wesep` | PyPI (wenet-e2e) | **Medium** | Small user base (~250 GitHub stars), Chinese maintainers |
| `modelscope` | PyPI (Alibaba) | **Medium** | Alibaba-maintained, executes download logic, dynamic URL resolution |
| `pyyaml` | PyPI | Low | Standard library |

The `wesep` package itself has no repository-level LICENSE file (only Apache-2.0 headers in individual source files), creating legal ambiguity.

### 3.4 Runtime Dependencies

| Package | Version | Risk | Audit Status |
|---|---|---|---|
| `onnxruntime` | >=1.17.0 | Low | Microsoft-maintained, widely deployed, no network calls in inference |
| `numpy` | >=1.24.0 | Low | Standard scientific Python |
| `sounddevice` | >=0.4.6 | Low | Thin wrapper around PortAudio, local audio only |
| `soundfile` | >=0.12.0 | Low | Wrapper around libsndfile, local file I/O only |
| `scipy` | >=1.11.0 | Low | Standard scientific Python |
| `click` | >=8.1.0 | Low | Pallets project, CLI only |
| `toml` | >=0.10.0 | Low | Config parsing only |

**No runtime dependency makes network calls.** The runtime attack surface is limited to local file I/O and PipeWire IPC.

---

## 4. Runtime Network Analysis

### 4.1 Application Code — Clean

Verified: zero imports of `requests`, `urllib`, `httpx`, `socket`, `http.client`, or any other network library in `src/tse_pipewire/`. All ONNX sessions load from local filesystem paths:

- `embedding.py:101`: `ort.InferenceSession(model_path, providers=providers)`
- `tse_engine.py:14`: `ort.InferenceSession(model_path, providers=providers)`

### 4.2 Subprocess Calls — Local Only

The application invokes only local PipeWire system commands:

- `pw-dump` — queries local PipeWire daemon state
- `pw-loopback` — creates local virtual audio device

### 4.3 ONNX Runtime Telemetry

ONNX Runtime has an optional telemetry feature that is **disabled by default** on Linux. The `CPUExecutionProvider` used by this project does not initiate network connections. Verified: no `CUDAExecutionProvider` or cloud-based execution providers are configured.

---

## 5. Threat Model

### 5.1 Threats During Model Setup (one-time)

| Threat | Likelihood | Impact | Risk |
|---|---|---|---|
| Malicious model weights on ModelScope | Low | High (arbitrary code execution during PyTorch load) | **Medium** |
| Compromised `modelscope` PyPI package | Low | High (arbitrary code at install time) | **Medium** |
| Compromised `wesep` PyPI package | Low | High (arbitrary code at install time) | **Medium** |
| MITM on ModelScope download (HTTPS downgrade) | Very Low | High | Low |
| Dynamic URL redirect to different model binary | Low | High | **Medium** |

### 5.2 Threats During Runtime

| Threat | Likelihood | Impact | Risk |
|---|---|---|---|
| Backdoored ONNX model exfiltrates audio | Very Low | High | Low |
| ONNX Runtime vulnerability | Very Low | Medium | Low |
| Local privilege escalation via PipeWire IPC | Very Low | Low | Very Low |
| Enrollment WAV/embedding stolen from disk | Low (requires local access) | Medium | Low |

### 5.3 Threat: Adversarial Model Weights

An ONNX model is a computational graph that processes tensors. Unlike PyTorch `.pt` files (which can execute arbitrary Python via `pickle`), ONNX models **cannot execute arbitrary code** at inference time. However:

- A malicious ONNX model could produce subtly wrong outputs (e.g., pass through all audio instead of filtering, degrading privacy)
- A malicious ONNX model could be crafted to cause excessive memory/CPU usage (DoS)
- The risk of code execution exists **during the PyTorch-to-ONNX export phase**, not during ONNX inference

---

## 6. Key Findings

### FINDING-1: No Model Integrity Verification (MEDIUM-HIGH)

Downloaded model weights are never verified against known-good checksums. The `modelscope` package handles downloads with dynamic URL resolution, meaning the actual download target could change server-side.

### FINDING-2: Export Script Executes Untrusted Model Code (MEDIUM)

`scripts/export_bsrnn_onnx.py` loads PyTorch checkpoint files via `load_pretrained_model()`, which internally uses `torch.load()`. PyTorch checkpoints use Python `pickle` deserialization, which **can execute arbitrary code**. A tampered `model.pt` file could run malicious code during the export phase.

### FINDING-3: ModelScope Downloads from Chinese Cloud Infrastructure (LOW-MEDIUM)

Model files are served from Alibaba Cloud (ModelScope.cn). While this is not inherently a security issue, it means:

- Downloads are subject to Chinese regulatory requirements
- Server-side model availability and content could change without notice
- Network traffic to Chinese infrastructure may be flagged by corporate security policies

### FINDING-4: WeSep Repository Has No Formal License (LOW)

The `wenet-e2e/wesep` repository lacks a repository-level LICENSE file. Individual source files carry Apache-2.0 headers, but the absence of a formal license creates legal ambiguity for redistribution and commercial use.

### FINDING-5: Runtime Is Well-Isolated (POSITIVE)

The runtime application makes zero network calls. All inference is local. Audio data never leaves the machine through the application. This is the correct architecture for a privacy-sensitive audio processing tool.

---

## 7. Recommendations

See `docs/security-mitigation-plan.md` for the full mitigation plan with verification steps.

### Priority Actions

1. **Pin model checksums** — Compute SHA-256 of exported ONNX files, verify on every load
2. **Isolate the export phase** — Run `scripts/download_models.sh` in a sandboxed environment (container/VM)
3. **Verify ONNX model structure** — Inspect exported models for unexpected operators or excessive complexity
4. **Network-isolate the runtime** — Enforce no-outbound-network as defense-in-depth
5. **Pin all dependency versions** — Lock exact versions of all PyPI packages

---

## Appendix A: Files Reviewed

- `src/tse_pipewire/embedding.py` — speaker embedding extraction (ONNX, local only)
- `src/tse_pipewire/tse_engine.py` — TSE inference (ONNX, local only)
- `src/tse_pipewire/pipewire_client.py` — PipeWire subprocess calls (local IPC only)
- `src/tse_pipewire/audio_pipeline.py` — audio orchestration (no network)
- `src/tse_pipewire/enrollment.py` — enrollment recording (sounddevice, local only)
- `src/tse_pipewire/config.py` — config management (TOML, local filesystem only)
- `src/tse_pipewire/cli.py` — CLI wiring (Click, no network)
- `scripts/export_bsrnn_onnx.py` — model download and ONNX export
- `scripts/download_models.sh` — shell wrapper for export script
- `pyproject.toml` — dependency declarations
