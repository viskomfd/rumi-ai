# Rumi AI — Vision-Based Fall Detection Engine

Repository khusus pengembangan **model AI** (`yolo11_rumi.pt`) dan **skrip inferensi/kinematika deteksi jatuh** (`python_script/realtime_fall_detection.py`) untuk ekosistem **Rumi**.

> **Catatan Arsitektur Organisasi (`viskomfd`):**
> - `falldetection-fe` — Frontend Dashboard
> - `Rumi-be` — Main Backend (FastAPI, Database, Multi-Camera Supervisor)
> - **Repo ini (`model-only`)** — AI Model Weights & Fall Detection Vision Script

---

## Fitur Utama Engine AI
- **Fine-tuned Pose Model (`yolo11_rumi.pt`):** YOLO11-pose dengan kalibrasi kamera CCTV & IR (fallback otomatis ke `yolo11n-pose.pt`).
- **4-Parameter Kinematic Calibration (4D Grid Search):**
  - `ASPECT_RATIO_THRESHOLD = 0.80`
  - `TORSO_ANGLE_THRESHOLD = 52.5°`
  - `VERTICAL_VELOCITY_THRESHOLD = 14.0 px/step`
  - `FLOOR_COLLAPSE_THRESHOLD = 0.40` *(Floor Collapse Ratio `y1 / frame_h` untuk mengatasi longitudinal fall & inverted skeleton)*
- **4 Lapis Validasi Spatio-Temporal:**
  1. *Anatomical Upright Filter* (`is_anatomically_upright`)
  2. *Kinetic Transition Gate* (`has_fall_dynamics` — membedakan **peristiwa jatuh** dari **tidur/rebahan normal**)
  3. *Sliding Window Voting* (`POSTURE_WINDOW = 5`, `POSTURE_VOTE_RATIO = 0.4`)
  4. *Inactivity Confirmation & Latch* (`INACTIVITY_SECONDS = 1.2`, pergeseran `< 25 px`)
- **Tracker Continuity:** ByteTrack + Spatial ID Re-linking (`160 px`, `3.0 s`) + Room-level Context Memory.

---

## Quickstart & Local Testing

### 1. Setup Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Konfigurasi Environment (Opsional)
Salin `.env.example` menjadi `.env` jika ingin menyesuaikan parameter:
```bash
copy .env.example .env
```

### 3. Jalankan Engine Secara Standalone
Secara default skrip akan membaca `python_script/test_fall.mp4` dan menampilkan jendela HUD debug (`AR`, `FCR`, `Vel`):
```bash
python python_script/realtime_fall_detection.py
```

---

## Integrasi ke Main Backend (`Rumi-be`)
Tim backend di `Rumi-be` hanya perlu mengambil dua file utama dari repo ini:
1. `yolo11_rumi.pt`
2. `python_script/realtime_fall_detection.py`

Detail perubahan model dan logika dapat dibaca di [`FALL_DETECTION_MODEL_AND_LOGIC_CHANGELOG.md`](FALL_DETECTION_MODEL_AND_LOGIC_CHANGELOG.md).
