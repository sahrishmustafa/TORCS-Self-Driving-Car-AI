# TORCS AI Driver Framework

This repository contains a TORCS (The Open Racing Car Simulator) client-side Python framework for collecting gameplay data and training AI drivers using machine learning models such as Decision Trees.

---

## 📦 Features

### ✅ Manual Driving Data Collection
Record human driver actions while playing TORCS to create a training dataset.

### ✅ AI Driver Using ML Models
Train and deploy AI drivers based on user-collected data using various machine learning models.

### ✅ Model Training Support
Scripts provided to train with Decision Trees, Random Forests, and MLP Regressors.

---

## 🗂️ File Overview

| File | Description |
|------|-------------|
| `pyclient.py` | Launch the game for manual driving. Collects gameplay data. |
| `pyclient_ai.py` | Launch the AI driver using a trained model. |
| `train_decisiontree.py` | Trains a Decision Tree model on the dataset. |
| `train_mlpregressor.py` | Trains an MLP Regressor model on the dataset. |
| `train_randomforest.py` | Trains a Random Forest model on the dataset. |
| `aidriver.py` | Core AI driver logic used for inference. |
| `driver_manual.py` | Handles manual (keyboard) input from user. |
| `carControl.py` | Represents car control commands (acceleration, steering, etc.). |
| `carState.py` | Parses and stores the car’s sensor data. |
| `msgParser.py` | Handles message parsing to and from the TORCS server. |
| `logger.py` | Logs gameplay data into a CSV file. |
| `check.py`, `driver_tempy.py`, `driver.py` | Variations and debug/testing versions of the driver script. |

> `.pyc` files are compiled Python bytecode files (auto-generated; not needed in version control).

---

## 🚀 Getting Started

### 1. Dataset Collection

To collect gameplay data:

```bash
python pyclient.py --track [TRACK_NUM] --car [CAR_NUM]
Example:
python pyclient.py --track 0 --car 2

```

This will launch TORCS in manual mode, where your actions are recorded and saved for training.

## 2. Train an AI Driver
You can train a model using one of the provided scripts:

Decision Tree:
python train_decisiontree.py
## 3. Run the AI Driver
Once trained, launch the AI driver:
python pyclient_ai.py
This starts the TORCS client and uses the trained model to control the car.

## 🛠 Requirements
Python 3.x
TORCS installed and running in client-server mode
Python packages
numpy
pandas
sklearn
joblib
Install required packages:
pip install -r requirements.txt
(Create requirements.txt if needed with dependencies listed.)

## 📂 Logs and Models
Dataset Logs: Saved as .csv files containing features and controls from gameplay.

Model Files: Saved after training using joblib, used by aidriver.py.

## 📌 Notes
Make sure TORCS is running in client-server mode before launching the Python client.

Models are trained on the specific format of data collected using this framework. Ensure consistency.

