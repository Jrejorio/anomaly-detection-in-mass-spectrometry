# 🔬 Cassiopée — Anomaly Detection in Mass Spectrometry

> A convolutional autoencoder that automatically flags faulty mass spectra in exhaled-air analysis.

Cassiopée 2026 project, in partnership with **Hôpital Foch**.

## 🎯 Context

Hôpital Foch analyses the volatile organic compounds (VOCs) of exhaled air with a **PTR-TOF-MS** mass spectrometer, in order to diagnose diseases or monitor the response to treatments.

Today, an **expert must visually validate every acquisition**. This is not feasible in multi-centre clinical protocols.

**Goal:** automatically detect abnormal spectra, and check that AI does better than classical controls (intensity threshold, standard calibration).

Anomalies targeted:

| Anomaly | Description |
|---|---|
| **Mass shift** | The spectrum is shifted along m/z (bad calibration) |
| **Low intensity** | The overall signal is too weak |
| **Others** | No signal at all, miscellaneous artefacts… |

## 🧠 Method

The autoencoder is trained **on normal spectra only**. Given an abnormal spectrum, it reconstructs it poorly: the reconstruction error (MSE) is used as the anomaly score.

```
Raw CSVs → Preprocessing → Autoencoder → Reconstruction error → Normal / Anomaly
           (log1p, m/z windows, scaling)  (Conv1D)               (MSE threshold + intensity threshold)
```

Two detectors are evaluated:
- **AI only**: anomaly if `MSE > mse_threshold`
- **Hybrid**: anomaly if `MSE > mse_threshold` **or** total intensity `< int_threshold`

Real mass shifts are rare, so **artificial mass shifts** are simulated (shift of 0.2 to 0.8 m/z) to enrich the test set.

## 📊 Results

![Evaluation dashboard](docs/evaluation_hybride.png)

*The figure was generated before the code was translated, so its labels are in French; re-running `main.py` produces the English version.*

Hybrid model on the test set (390 spectra, thresholds chosen by hand):

| Metric | Value |
|---|---|
| Anomalies detected (recall) | **255 / 262 (97%)** |
| Normal spectra correctly classified | 104 / 128 (81%) |
| Overall accuracy | 92% |

Comparison of the continuous scores (ROC AUC):

| Score | AUC |
|---|---|
| AI only (reconstruction MSE) | **0.87** |
| Intensity only | 0.72 |
| Hybrid (max of both scores) | 0.73 |

**Take-aways**
- The autoencoder alone separates normal and abnormal spectra much better than the intensity check (AUC 0.87 vs 0.72), which is the main result of the project.
- Combining both scores with a simple `max` does *not* improve the ranking (AUC 0.73 vs 0.87 for the AI alone). The hybrid rule is, however, useful at the chosen operating point: it catches 97% of the anomalies, at the cost of some false alarms on normal spectra.
- These figures come from a small test set, with thresholds tuned on the same data, so they are optimistic and should be validated on fresh data.

## 📁 Project structure

```
hopital-foch/
├── main.py                 # Entry point: data → training → evaluation
├── src/
│   ├── preprocessing.py    # CSV loading, m/z alignment, filtering, scaling
│   ├── simulation.py       # Artificial mass-shift simulation
│   ├── models.py           # Autoencoder architecture (Conv1D)
│   └── evaluation.py       # Metrics and plots (ROC, PR, confusion matrix…)
├── config/windows.txt      # m/z windows kept for the analysis
├── docs/                   # Figures
└── requirements.txt
```

## 🚀 Usage

```bash
git clone https://github.com/Jrejorio/hopital-foch.git
cd hopital-foch
pip install -r requirements.txt
python main.py
```

- On the first run the model is **trained** (50 epochs) and its weights are saved to `models_saved/ae_v4_weights.h5`; later runs reload them instead of retraining.
- Spectra are **cached** as `data/processed/*.npy` on first read, which speeds up later runs.

> ⚠️ **The data and the trained weights are not included** (medical data from Hôpital Foch). To run the project, place the data in `data/` (`data/data/…`, `data/data2/…`).

## 🛠️ Technologies

Python · TensorFlow/Keras · NumPy · pandas · SciPy · scikit-learn · Matplotlib
