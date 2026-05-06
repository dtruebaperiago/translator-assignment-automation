# 📊 iDISC Recommender System: Data Preprocessing & Feature Engineering

## 🎯 Overview

This document outlines the data preprocessing pipeline designed to feed a Dual-Tower Multi-Layer Perceptron (MLP) recommender system built in PyTorch. The goal of the model is to predict the optimal translation employee for a given translation task.

To achieve this, raw data from four sources (`data`, `schedules`, `Clients`, `Translators Costs+Pairs`) is ingested, cleaned, engineered into time-series features, and scored. The pipeline is strictly designed to prevent data leakage and solve the Cold Start problem.

---

## 🛠️ Pipeline Architecture

### 1. Data Cleaning & Noise Reduction

Before feature engineering, the raw data undergoes strict sanitation:

* **Missing Values & Outliers:** Logical incongruencies are fixed, missing values are imputed, and extreme outliers in `HOURS`, `HOURLY_RATE`, and `COST` are capped using IQR/Percentiles.
* **Feature Selection:** Redundant hierarchical categorical columns (e.g., Manufacturer Sub-Industries) are dropped to reduce dimensionality.
* **Encoding:** Categorical variables (`SOURCE_LANG`, `TARGET_LANG`, `TASK_TYPE`) are transformed using One-Hot Encoding.

### 2. Time-Series Feature Engineering (Input Features)

To prevent the model from "cheating" by seeing future performance, all engineered features are calculated as expanding rolling windows computed strictly prior to a task's start date:

* **Schedule Modeling:** Raw schedule hours are transformed into numeric features: `Daily_Shift_Length`, `Weekly_Availability_Hours`, and a `Works_Weekends` boolean.
* **Experience Counters:** Rolling counts of domain-specific and task-specific experience are maintained per translator.
* **Rolling Quality & Punctuality:** Historical performance is tracked via an Exponential Moving Average (EMA) for Quality, and expanding ratios for Punctuality and Forecast-vs-Actual Efficiency.
* **Latent Flags:** `IS_NEW_EMPLOYEE` (< 10 tasks) and `IS_SPECIALIST` (high rate + low volume) flags are generated to guide the neural network during cold-start scenarios.

### 3. Continuous Affinity Score Formulation (Target Label 'Y')

Instead of predicting who was historically chosen (which creates bias), the model predicts the Outcome Quality. The pipeline engineers a continuous `AFFINITY_LABEL` bounded between 0.0 and 1.0 using a weighted formula based on the post-task outcome:

* **Quality (40%):** Normalized quality evaluations with penalties for failing client minimums.
* **Time Efficiency (30%):** A decay function based on actual working hours versus forecasted hours.
* **Profit Margin (30%):** Normalized gross benefit derived from the Client Selling Price vs Translator Hourly Rate.

**🚨 Crucial Leakage Prevention:** After the Affinity Score and rolling features are calculated, all workflow timestamps (`START`, `END`, `WORKING`, `DELIVERED`, etc.) and exact costs are permanently dropped from the input features.

---

## 📂 Output Datasets Dictionary

The pipeline outputs the data in stages to facilitate different parts of the PyTorch training and evaluation loops.

### Core Unsplit Structures (Step 4)

These files represent the entire historical dataset, separated purely by architecture logic before any chronological splitting.

* 📄 **`tower_a_employee_features.csv`**: Contains only the inputs for the Translator neural network tower. Includes capacity metrics, rolling experience, historical quality/punctuality ratings, base costs, and cold-start flags.
* 📄 **`tower_b_task_features.csv`**: Contains only the inputs for the Task neural network tower. Includes task type, language pairs, forecasted hours, client data, and selling price.
* 📄 **`target_labels.csv`**: Contains the ground truth. Includes `TASK_ID`, `TRANSLATOR`, and the continuous `AFFINITY_LABEL` (0.0 to 1.0) calculated for that specific historical pairing.

### PyTorch Training & Validation Splits (Step 5)

The data is split chronologically (70% Train / 15% Val / 15% Test) to respect the flow of time.

* 📄 **`train_merged.csv`** (70% of data): Contains the merged historical interactions (Tower A + Tower B + Label) for the oldest 70% of the timeline. This paired data is used to calculate the Loss Function and train the weights via backpropagation.
* 📄 **`val_merged.csv`** (15% of data): Formatted identically to the training set. Used at the end of each training epoch to ensure the model is generalizing and not overfitting to the training data.

### PyTorch Offline Evaluation Split (Step 5)

Unlike the train/val sets, the Test set is intentionally kept separated. To properly evaluate a recommender system using ranking metrics (like Hit Rate@K), we cannot simply test pre-paired historical matches. We must take one new task and ask the model to rank all available translators.

* 📄 **`test_task.csv`**: Contains only the Tower B features for the newest 15% of tasks.
* 📄 **`test_translators.csv`**: Contains the state of the Tower A translator features at the exact moment the test tasks occurred.
* 📄 **`test_labels.csv`**: Contains the true Affinity Scores for the test period, used to verify if the model successfully pushed the best historical matches to the top of its predicted ranking.
