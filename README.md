# 🚗 Vehicle Telematics — Fuel Efficiency & Driver Behavior Analysis

Real-world OBD sensor data from a 16-vehicle fleet analyzed to identify optimal driving conditions, estimate gear usage, and calculate trip-level fuel consumption.

---

## 📊 Project Summary

| | |
|---|---|
| **Dataset** | LEVIN Vehicle Telematics (Yun Solutions) |
| **Vehicles** | 16 |
| **Trips** | 431 |
| **Records** | ~3.1M rows |
| **Period** | Nov 2017 – Jan 2018 |
| **Tools** | Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |

---

## 🔍 What Was Analyzed

### 1. Sensor Distributions
Cleaned and validated 6 OBD sensors: speed, RPM, engine load, fuel efficiency (kpl), coolant temperature, and throttle position. Identified and removed GPS speed outliers (up to 512 km/h), KPL spikes, and corrupted header rows embedded in the raw CSV.

### 2. Temporal Patterns
- Median speed drops significantly during rush hours (07:00–09:00 and 17:00–20:00)
- Trip distribution is relatively even across weekdays with a slight dip on Sundays

### 3. Speed-Efficiency Map (per vehicle)
Grouped all readings into 12 speed zones (0–10, 10–20, … 120+ km/h) and calculated average fuel efficiency per zone per vehicle.

**Key finding:** Fleet-wide optimal speed zone is **60–90 km/h** — vehicles operating in this range show up to 40% higher fuel efficiency compared to low-speed urban driving below 30 km/h.

### 4. Fleet Fuel Consumption Summary
Calculated total distance, total fuel consumed, and overall km/L per vehicle using time-series integration (speed × time delta).

| | Vehicle |
|---|---|
| Most efficient | Vehicle 9 → 12.4 kpl |
| Least efficient | Vehicle 6 → 3.4 kpl |
| Efficiency gap | ~9 kpl between best and worst |

### 5. Gear Estimation (per vehicle)
Since no gear sensor exists in OBD data, gear was estimated using:

```
gear_ratio = RPM / speed
```

Higher ratio = lower gear (high RPM, low speed). KMeans clustering was applied **per vehicle** so each vehicle is calibrated to its own RPM and speed range — no universal thresholds assumed.

---

## ⚠️ Data Quality Notes

| Vehicle | Issue |
|---|---|
| 0, 1, 2, 4, 11 | KPL sensor not recording — excluded from fuel analyses |
| Vehicle 7 | KPL sensor intermittent — excluded from fuel analyses |
| Vehicle 9 | Consistently higher KPL values — likely different engine type or displacement |

---

## 📁 Repository Structure

```
vehicle-telematics/
├── notebook/
│   └── 01_EDA.ipynb          ← Full analysis notebook
├── outputs/
│   ├── 01_sensor_distributions.png
│   ├── 02_temporal_patterns.png
│   ├── 03_speed_efficiency_map.png
│   ├── 03b_fleet_heatmap.png
│   ├── 04_fleet_fuel_summary.png
│   └── 05_gear_analysis.png
├── data/
│   └── processed/
│       ├── trip_summary.csv
│       ├── fleet_summary.csv
│       ├── speed_map.csv
│       └── gear_analysis.csv
├── .gitignore
└── README.md
```

---

## 🗄️ Dataset

This project uses the **LEVIN Vehicle Telematics** dataset published by Yun Solutions.

- **Source:** [Kaggle — LEVIN OBD Sensor Data](https://www.kaggle.com/datasets/yunlevin/levin-vehicle-telematics)
- **File:** `v2.csv` (692 MB — not included in this repo due to GitHub file size limits), updated version of allcars.csv data

**To run this notebook locally:**
1. Download `v2.csv` from the Kaggle link above
2. Place it in the `data/` folder as allcars.csv
3. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
4. Open `notebook/01_EDA.ipynb` and run all cells

---

## 🛠️ Tech Stack

- **Python 3.10** — Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Jupyter Notebook**
- **KMeans Clustering** — per-vehicle gear estimation
- **OBD Signal Processing** — speed/RPM/KPL/engine load

---

## 👤 Author

**Paşan Sancak** — Data Analyst | Ex-Bosch Automotive

- [LinkedIn](https://linkedin.com/in/pasansancak)
- [GitHub](https://github.com/pasansancak)
- [Upwork](https://www.upwork.com/freelancers/~019d81f8710bda98b5)
