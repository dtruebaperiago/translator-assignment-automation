**THE FILES WE ARE TALKING ABOUT DO NOT APPEAR IN THE GITHUB SINCE THE FILE SIZE IS BIGGER THAN 25MB


##  Files in this Folder

The preprocessing pipeline (`notebooks/02_preprocessing/preprocessing.py`) took the raw/enriched data, cleaned it, encoded it, and split it into three separate CSV files:

1. **`train.csv` (70% of data - ~551k rows)**
   * **Utility**: This is the core dataset for teaching our Machine Learning model. The model will look at historical patterns in this file to learn how different combinations of translators, languages, and task types affect the final `QUALITY_EVALUATION`.
2. **`validation.csv` (15% of data - ~118k rows)**
   * **Utility**: This dataset is used to fine-tune our model. While training, we will repeatedly test the model against this data to tweak its settings (hyperparameter tuning) and prevent it from memorizing the training data (overfitting). It helps us choose the best algorithm (e.g., Random Forest vs. XGBoost).
3. **`test.csv` (15% of data - ~118k rows)**
   * **Utility**: This is our "blind test." The model never sees this data during training or tuning. We will only use this file at the very end to evaluate how accurately our final model predicts quality on completely unseen, real-world tasks. This gives us our final performance metrics before integrating the model into the Copilot dashboard.

---

## Dataset Structure & Information

All three files have the exact same structure (36 columns) and contain purely numeric data, meaning they are perfectly formatted to be fed directly into an ML model.

The information contained in these files can be broken down into the following categories:

### 1. The Target Variable (What we want to predict)
* `QUALITY_EVALUATION`: The numerical score representing the quality of the translation. **This is what our ML model (the Soft Constraint algorithm) will try to predict** for future tasks to rank the best translators.

### 2. Core Task Features
* `HOURS`: The estimated hours needed for the task.
* `HOURLY_RATE`: The rate for this specific task.
* `COST`: Total cost of the task.
* `TASK_TIME_HOURS`: The actual time it took to complete the task.
* `HOURS_ACCURACY`: The difference between estimated and actual hours.
* `PUNCTUALITY_SCORE`: How well deadlines were met.

### 3. Translator Variables
* `TRANSLATOR`: A unique numerical ID representing the translator.
* `TRANSLATOR_HOURLY_RATE` & `TRANSLATOR_COST`: The specific cost metrics associated with the assigned translator.
* `PROFIT`: The calculated profit margin for the task.
* `AVG_QUALITY_GENERAL`: The translator's historical average quality across all tasks.
* `AVG_QUALITY_LANG_PAIR`: The translator's historical average quality for this specific language pair.
* `EXP_SECTOR_COUNT` & `EXP_TASK_TYPE_COUNT`: Synthetic features representing the translator's cumulative experience in a specific sector or task type.

### 4. Categorical Encoded Data 
* **Languages**: `SOURCE_LANG` and `TARGET_LANG` are label-encoded. Following the project convention: English = 1, Spanish = 0, and all other languages = 2+.
* **Client / Industry Info**: `MANUFACTURER`, `MANUFACTURER_SECTOR`, `MANUFACTURER_INDUSTRY_GROUP`, `MANUFACTURER_INDUSTRY`, and `MANUFACTURER_SUBINDUSTRY` are label-encoded with unique IDs to represent the client and their specific domain.
* **Task Types (One-Hot Encoded)**: The original `TASK_TYPE` column was expanded into 14 binary (0 or 1) columns. A `1` means the task is of that type, and `0` means it is not.

---

## Future Utility in the Project

Having these clean files we accomplish a big part of the project pipeline. 

Moving forward into (Hard Constraints) and (Soft Constraints) part, these files provide the basis for:
1. **Filtering**: We will use the characteristics in these files to build rule-based filters 
2. **Predicting**: We will load `train.csv` into a model (to be defined) to predict `QUALITY_EVALUATION`. 
3. **Ranking**: The predicted quality, combined with cost and punctuality features, will be fed into our objective function to output the "Top 3" recommendations for the Project Managers.
