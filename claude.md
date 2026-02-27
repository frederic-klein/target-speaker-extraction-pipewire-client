# TSE-PipeWire: Target Speaker Extraction als PipeWire-Client

## Projektübersicht

Echtzeit-Stimmisolation für Linux-Arbeitsplätze im Großraumbüro. Die Software erstellt ein virtuelles Mikrofon, das nur die enrollte Stimme des Nutzers durchlässt und alle anderen Stimmen sowie Hintergrundgeräusche unterdrückt – funktional identisch mit Microsoft Teams Voice Isolation, aber plattformunabhängig als PipeWire-Client.

**Ziel-Stack:**
```
Physisches Mikrofon
  → Stufe 1: RNNoise (LADSPA) → Umgebungsgeräusche unterdrücken
  → Stufe 2: TSE-Modell (ONNX Runtime) → Nur enrollte Stimme extrahieren
  → Virtuelles Mikrofon ("TSE Filtered Mic")
```

**Zielplattform:** Linux mit PipeWire (Tuxedo OS / KDE Plasma, Ubuntu 24+)
**Sprache:** Python (Prototyp), C++ (Performance-Optimierung später)
**Lizenz:** MIT

---

## Architektur

```
┌─────────────────────────────────────────────────────────┐
│  tse-pipewire                                           │
│                                                         │
│  ┌──────────┐   ┌───────────┐   ┌───────────────────┐  │
│  │ PipeWire │──▶│ RNNoise   │──▶│ TSE ONNX Inferenz │  │
│  │ Capture  │   │ Denoiser  │   │ (Speaker-Conditioned)│ │
│  └──────────┘   └───────────┘   └─────────┬─────────┘  │
│                                           │             │
│  ┌──────────────────────┐                 │             │
│  │ Speaker Embedding    │─────────────────┘             │
│  │ (aus Enrollment WAV) │                               │
│  └──────────────────────┘                               │
│                                           │             │
│                               ┌───────────▼─────────┐  │
│                               │ PipeWire Virtual Mic │  │
│                               │ "TSE Filtered Mic"   │  │
│                               └─────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Projekt-Setup

### Verzeichnisstruktur

```
tse-pipewire/
├── CLAUDE.md              # Diese Datei – Projektplan & Kontext
├── README.md              # Nutzerdoku
├── pyproject.toml         # Python-Paketierung
├── requirements.txt       # Abhängigkeiten
│
├── src/
│   └── tse_pipewire/
│       ├── __init__.py
│       ├── cli.py             # Haupteinstiegspunkt (Click CLI)
│       ├── enrollment.py      # Stimmprofil erstellen
│       ├── embedding.py       # Speaker-Embedding berechnen
│       ├── tse_engine.py      # ONNX-Inferenz für TSE
│       ├── rnnoise_filter.py  # RNNoise-Vorfilterung
│       ├── pipewire_client.py # PipeWire Virtual-Mic Erstellung
│       ├── audio_pipeline.py  # Gesamte Audio-Pipeline orchestrieren
│       └── config.py          # Konfigurationsmanagement
│
├── models/                # ONNX-Modelle (nicht im Git, .gitignore)
│   ├── .gitkeep
│   └── README.md          # Anleitung zum Modell-Download
│
├── profiles/              # Gespeicherte Speaker-Embeddings
│   └── .gitkeep
│
├── scripts/
│   ├── download_models.sh # Modelle herunterladen & ONNX-Export
│   ├── install_deps.sh    # Systemabhängigkeiten installieren
│   └── setup_rnnoise.sh   # RNNoise LADSPA Plugin bauen/installieren
│
├── tests/
│   ├── test_enrollment.py
│   ├── test_embedding.py
│   ├── test_tse_engine.py
│   ├── test_pipeline.py
│   └── fixtures/          # Test-Audio-Dateien
│       └── .gitkeep
│
└── docs/
    ├── architecture.md
    └── troubleshooting.md
```

---

## Implementierungsplan (Tasks)

### Phase 1: Grundlagen & Modellvorbereitung

#### Task 1.1: Projektgerüst erstellen
- `pyproject.toml` mit Abhängigkeiten (onnxruntime, numpy, sounddevice, click)
- `requirements.txt` generieren
- Basis-CLI mit Click aufsetzen (`tse-pipewire enroll`, `tse-pipewire start`, `tse-pipewire stop`)
- Konfigurationsdatei-Handling (`~/.config/tse-pipewire/config.toml`)

#### Task 1.2: WeSep-Modell evaluieren und ONNX-Export vorbereiten
- `scripts/download_models.sh` erstellen:
  ```bash
  # WeSep klonen und vortrainiertes Modell herunterladen
  git clone https://github.com/wenet-e2e/wesep.git /tmp/wesep
  # WeSpeaker für Speaker-Encoder
  pip install wespeaker
  ```
- Recherche: Welches vortrainierte WeSep-Modell hat die beste Echtzeitfähigkeit?
  - Priorisiere kausale (streaming-fähige) Modelle
  - Bevorzuge Conv-TasNet oder S4D-basierte Architekturen (SpeakerBeam-SS-Stil)
- ONNX-Export-Skript schreiben mit WeSep-Boardmitteln:
  ```python
  # WeSep unterstützt ONNX und JIT Export nativ
  # Dokumentiert in wesep/cli/
  ```
- Ziel: `models/tse_model.onnx` und `models/speaker_encoder.onnx`

#### Task 1.3: Speaker-Embedding-Pipeline (`embedding.py`)
- WeSpeaker (ECAPA-TDNN) als Speaker-Encoder verwenden
  - WeSpeaker bietet vortrainierte Modelle und ONNX-Export
  - `pip install wespeaker` → Modell laden → Embedding extrahieren
- Funktion: `compute_embedding(audio: np.ndarray, sr: int) -> np.ndarray`
- Embedding als `.npy` Datei unter `profiles/<username>.npy` speichern
- Erwartete Embedding-Dimension: 192 oder 256 (modellabhängig)

#### Task 1.4: Enrollment-Prozess (`enrollment.py`)
- Audio aufnehmen via `sounddevice` (10-30 Sekunden)
- Nutzer-Prompt: "Bitte lesen Sie folgenden Text vor..." (wie bei Teams)
- Aufnahme als WAV speichern (`profiles/<username>_enrollment.wav`)
- Speaker-Embedding berechnen und speichern
- Qualitätsprüfung: SNR der Aufnahme checken, bei zu viel Rauschen warnen
- CLI-Befehl: `tse-pipewire enroll --name "frederic"`

### Phase 2: TSE-Inferenz-Engine

#### Task 2.1: ONNX Runtime Inferenz (`tse_engine.py`)
- ONNX Runtime Session initialisieren mit Provider-Auswahl:
  ```python
  # CPU: 'CPUExecutionProvider'
  # Optional GPU: 'CUDAExecutionProvider' (falls verfügbar)
  providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
  session = ort.InferenceSession("models/tse_model.onnx", providers=providers)
  ```
- Streaming-Inferenz implementieren:
  - Audio in Chunks verarbeiten (Chunk-Größe = Window-Länge des Modells)
  - Für WeSep/Conv-TasNet typisch: 16kHz, Fenster 2-20ms
  - Overlap-Add für artefaktfreie Ausgabe
  - State-Management für kausale Modelle (Hidden States zwischen Chunks weitergeben)
- Klasse `TSEEngine`:
  ```python
  class TSEEngine:
      def __init__(self, model_path: str, embedding: np.ndarray):
          ...
      def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
          """Verarbeitet einen Audio-Chunk und gibt extrahiertes Audio zurück."""
          ...
      def reset(self):
          """Setzt Hidden States zurück (bei Neustart)."""
          ...
  ```
- **Kritisch:** Latenz messen! Ziel: < 30ms algorithmische Latenz + < 10ms Inferenzzeit pro Chunk

#### Task 2.2: RNNoise-Vorfilterung (`rnnoise_filter.py`)
- Option A (bevorzugt): RNNoise als eigener PipeWire-Filter-Chain-Knoten (separate Konfiguration)
- Option B (Fallback): RNNoise als Python-Modul einbinden via `noisereduce` oder direkte `librnnoise` Bindung
- Für Option A: PipeWire Filter-Chain Konfiguration generieren:
  ```lua
  -- ~/.config/pipewire/pipewire.conf.d/tse-rnnoise.conf
  context.modules = [{
      name = libpipewire-module-filter-chain
      args = {
          node.description = "RNNoise Denoiser"
          filter.graph = {
              nodes = [{
                  type = ladspa
                  name = rnnoise
                  plugin = librnnoise_ladspa.so
                  label = noise_suppressor_mono
                  control = {
                      "VAD Threshold (%)" = 50.0
                      "VAD Grace Period (ms)" = 200
                  }
              }]
          }
          capture.props = {
              node.name = "capture.rnnoise"
              node.passive = true
              audio.rate = 48000
          }
          playback.props = {
              node.name = "rnnoise_source"
              media.class = Audio/Source
              audio.rate = 48000
          }
      }
  }]
  ```
- RNNoise arbeitet bei 48kHz, TSE-Modelle typisch bei 16kHz → Resampling nötig!

### Phase 3: PipeWire-Integration

#### Task 3.1: PipeWire Virtual-Mic Client (`pipewire_client.py`)
- **Ansatz 1 (empfohlen):** `pipewire` Python-Bindings über `pipewirebind` oder `pw-cat`/`pw-record`/`pw-play` als Subprozess
- **Ansatz 2:** Direkte libpipewire-Bindings via ctypes/cffi (komplexer aber leistungsfähiger)
- **Ansatz 3 (pragmatisch):** `sounddevice` mit JACK-Backend (PipeWire ist JACK-kompatibel)
  ```python
  import sounddevice as sd
  # PipeWire JACK-Kompatibilität nutzen
  # Input: physisches Mikrofon
  # Output: virtuelles PipeWire-Device
  ```
- Virtuelles Mikrofon erstellen, das in Teams/Zoom/etc. als Eingabegerät erscheint
- Implementierung:
  ```python
  class PipeWireClient:
      def __init__(self, input_device: str, sample_rate: int = 48000):
          ...
      def create_virtual_mic(self, name: str = "TSE Filtered Mic"):
          """Erstellt ein virtuelles PipeWire-Eingabegerät."""
          ...
      def start_processing(self, pipeline: AudioPipeline):
          """Startet die Audio-Verarbeitung in Echtzeit."""
          ...
      def stop(self):
          """Stoppt Verarbeitung und entfernt virtuelles Mikrofon."""
          ...
  ```

#### Task 3.2: Audio-Pipeline Orchestrierung (`audio_pipeline.py`)
- Gesamte Kette zusammenbauen:
  1. Audio von PipeWire Capture empfangen (48kHz, mono)
  2. Resample auf 16kHz für TSE-Modell
  3. RNNoise anwenden (wenn in-process statt als PipeWire-Knoten)
  4. TSE-Modell anwenden mit geladenem Speaker-Embedding
  5. Resample zurück auf 48kHz
  6. An virtuelles PipeWire-Mikrofon ausgeben
- Thread-Management: Audio-Callback darf nicht blockieren
  - Audio-Ringpuffer für asynchrone Verarbeitung
  - Inferenz in separatem Thread
- Latenz-Budget:
  ```
  Capture Buffer:    ~5ms (256 samples @ 48kHz)
  Resample 48→16:    ~1ms
  RNNoise:           ~5ms
  TSE Inferenz:      ~10-20ms
  Resample 16→48:    ~1ms
  Playback Buffer:   ~5ms
  ─────────────────────────
  Gesamt:            ~27-37ms (akzeptabel für VoIP)
  ```

### Phase 4: CLI & Nutzerinterface

#### Task 4.1: CLI fertigstellen (`cli.py`)
```bash
# Stimmprofil erstellen
tse-pipewire enroll --name "frederic" --duration 15

# Filter starten
tse-pipewire start --profile "frederic" --input-device "alsa_input.usb-..."

# Filter stoppen
tse-pipewire stop

# Verfügbare Audio-Geräte auflisten
tse-pipewire devices

# Profil testen (verarbeitet eine WAV-Datei)
tse-pipewire test --profile "frederic" --input test_mixture.wav --output test_extracted.wav

# Status anzeigen
tse-pipewire status
```

#### Task 4.2: Systemd-User-Service
```ini
# ~/.config/systemd/user/tse-pipewire.service
[Unit]
Description=TSE PipeWire Voice Isolation
After=pipewire.service
Requires=pipewire.service

[Service]
Type=simple
ExecStart=/usr/local/bin/tse-pipewire start --profile default
ExecStop=/usr/local/bin/tse-pipewire stop
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

### Phase 5: Qualitätssicherung & Optimierung

#### Task 5.1: Offline-Tests
- Test mit synthetischen Mischungen:
  - Eigene Stimme + andere Stimmen (deutsch!) + Bürogeräusche mischen
  - TSE anwenden und Ergebnis evaluieren
  - Metriken: SI-SDR (Scale-Invariant Signal-to-Distortion Ratio), PESQ, STOI
- Test mit echten Büro-Aufnahmen
- A/B-Vergleich: nur RNNoise vs. RNNoise+TSE

#### Task 5.2: Latenz-Profiling
- End-to-End-Latenz messen (Loopback-Test)
- ONNX Runtime Profiling: Inferenzzeit pro Chunk
- Falls > 30ms: Chunk-Größe anpassen, Modell-Optimierung (ONNX Graph Optimization, Quantisierung)

#### Task 5.3: Modell-Quantisierung (optional, bei Performanceproblemen)
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    "models/tse_model.onnx",
    "models/tse_model_int8.onnx",
    weight_type=QuantType.QInt8
)
```

---

## Abhängigkeiten

### Python-Pakete
```
onnxruntime>=1.17.0        # ONNX-Inferenz
numpy>=1.24.0              # Array-Verarbeitung
sounddevice>=0.4.6         # Audio I/O
soundfile>=0.12.0          # WAV lesen/schreiben
scipy>=1.11.0              # Resampling (signal.resample_poly)
click>=8.1.0               # CLI
toml>=0.10.0               # Konfiguration
wespeaker>=1.0.0           # Speaker-Embedding (Enrollment)
librosa>=0.10.0            # Audio-Utilities (optional)
```

### System-Pakete (apt/pacman)
```
pipewire                   # Audio-Server
pipewire-jack              # JACK-Kompatibilitätsschicht
libpipewire-0.3-dev        # PipeWire-Entwicklungsbibliotheken
python3-dev                # Python C-Extensions
libsndfile1-dev            # Soundfile-Backend
portaudio19-dev            # Sounddevice-Backend
ladspa-sdk                 # LADSPA-Plugin-System
```

### Externe Modelle (nicht im Repo)
```
# WeSep vortrainiertes TSE-Modell → ONNX exportieren
# WeSpeaker ECAPA-TDNN → für Speaker-Embedding
# RNNoise LADSPA Plugin → librnnoise_ladspa.so
```

---

## Wichtige technische Entscheidungen

### Warum WeSep statt SpeakerBeam direkt?
- WeSep hat nativen ONNX/JIT Export
- CLI und vortrainierte Modelle verfügbar
- Nutzt WeSpeaker für Speaker-Encoder (gleiche Toolchain)
- Aktiv maintained (wenet-e2e Ökosystem)
- SpeakerBeam-SS hat bessere Echtzeit-Performance, aber WeSep ist einfacher zu deployen

### Warum ONNX Runtime statt PyTorch direkt?
- Deutlich geringerer Memory-Footprint
- Schnellere Inferenz auf CPU (optimierte Kernel)
- Kein PyTorch als Runtime-Dependency nötig
- Graph-Optimierungen und Quantisierung out-of-the-box

### Sample Rate Handling
- RNNoise: **48kHz** (zwingend, andere Raten funktionieren nicht korrekt)
- TSE-Modelle: typisch **16kHz** (WeSep/SpeakerBeam)
- PipeWire Default: 48kHz
- → Resampling 48→16kHz vor TSE, 16→48kHz danach
- `scipy.signal.resample_poly` für effizientes ganzzahliges Resampling (Faktor 3)

### Kausale vs. Nicht-kausale Modelle
- **Zwingend kausal** für Echtzeit (kein Lookahead auf zukünftige Samples)
- WeSep unterstützt kausale Conv-TasNet-Varianten
- SpeakerBeam-SS ist explizit kausal mit ~20ms algorithmischer Latenz
- Bei ONNX-Export darauf achten, dass der Computational Graph kausal bleibt

---

## Bekannte Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Mitigation |
|--------|-------------------|------------|
| Vortrainierte Modelle generalisieren nicht auf deutsche Sprecher | Mittel | Testen mit echten Aufnahmen; ggf. Fine-Tuning auf deutschen Daten (VoxCeleb2 enthält diverse Sprachen) |
| ONNX-Inferenz zu langsam für Echtzeit auf CPU | Gering | INT8-Quantisierung; Chunk-Größe erhöhen; GPU-Fallback via CUDA Provider |
| PipeWire Virtual-Mic wird von Teams-Webapp nicht erkannt | Gering | PipeWire Virtual-Devices werden als reguläre ALSA-Quellen exponiert; Teams-Webapp sieht sie |
| Enrollment-Qualität im lauten Büro unzureichend | Mittel | Enrollment in ruhiger Umgebung empfehlen; RNNoise-Vorfilterung auf Enrollment-Audio anwenden; WeSep unterstützt Enrollment mit Noise-Augmentation |
| Audio-Artefakte bei Chunk-Grenzen | Mittel | Overlap-Add Synthese; ausreichend Overlap wählen |
| Hidden-State-Management bei kausalen Modellen in ONNX | Mittel | States als zusätzliche ONNX I/O modellieren; zwischen Inference-Calls durchreichen |

---

## Reihenfolge für Claude Code /todos

Wenn du dieses Projekt mit Claude Code umsetzt, arbeite die Tasks in dieser Reihenfolge ab:

```
1. [ ] Projektgerüst erstellen (pyproject.toml, Verzeichnisstruktur, CLI-Skeleton)
2. [ ] scripts/install_deps.sh – Systemabhängigkeiten installieren
3. [ ] scripts/setup_rnnoise.sh – RNNoise LADSPA Plugin bauen
4. [ ] scripts/download_models.sh – WeSep + WeSpeaker Modelle herunterladen
5. [ ] src/tse_pipewire/config.py – Konfigurationsmanagement
6. [ ] src/tse_pipewire/embedding.py – Speaker-Embedding mit WeSpeaker
7. [ ] src/tse_pipewire/enrollment.py – Enrollment-Prozess (Aufnahme + Embedding)
8. [ ] ONNX-Export: WeSep-Modell nach ONNX exportieren (ggf. als separates Skript)
9. [ ] src/tse_pipewire/tse_engine.py – ONNX-Inferenz mit Streaming/State-Management
10. [ ] tests/test_tse_engine.py – Offline-Test mit WAV-Dateien
11. [ ] src/tse_pipewire/rnnoise_filter.py – RNNoise-Integration
12. [ ] src/tse_pipewire/pipewire_client.py – Virtual-Mic erstellen
13. [ ] src/tse_pipewire/audio_pipeline.py – Pipeline zusammenbauen
14. [ ] src/tse_pipewire/cli.py – CLI fertigstellen (enroll, start, stop, devices, test)
15. [ ] End-to-End Test: Enrollment + Start + Teams-Webapp prüfen
16. [ ] Systemd-Service erstellen
17. [ ] Latenz-Profiling & Optimierung
18. [ ] README.md mit Installationsanleitung
```

---

## Referenzen

- **WeSep Toolkit:** https://github.com/wenet-e2e/wesep
- **SpeakerBeam-SS Paper:** arXiv:2407.01857
- **SpeakerBeam Repo:** https://github.com/BUTSpeechFIT/speakerbeam
- **WeSpeaker (Speaker Encoder):** https://github.com/wenet-e2e/wespeaker
- **noise-suppression-for-voice (RNNoise LADSPA):** https://github.com/werman/noise-suppression-for-voice
- **PipeWire Filter-Chain Docs:** https://docs.pipewire.org/page_module_filter_chain.html
- **ONNX Runtime:** https://onnxruntime.ai/
- **NoiseTorch-ng (Architektur-Referenz):** https://github.com/noisetorch/NoiseTorch
