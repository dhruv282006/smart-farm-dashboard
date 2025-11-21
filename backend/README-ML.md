# ML Training and Report — backend/ml

This folder contains a demonstrative training pipeline for a regression task using a public dataset (UCI Wine Quality - red) by default.

What was added/changed

- `ml/train_model.py`: full training pipeline
  - Downloads dataset if provided a URL (default: UCI winequality-red.csv)
  - Validates dataset (minimum rows & features)
  - Basic preprocessing (imputation, scaling, encoding)
  - Trains two models: RandomForestRegressor and GradientBoostingRegressor
  - Evaluates on test set (RMSE & MAE) and optionally 5-fold CV
  - Saves trained pipeline models (.joblib) to `backend/ml/`
  - Saves reports (plots + `results_summary.csv` + JSON) to `backend/ml/reports/`

- `notebooks/analysis.ipynb`: Notebook that runs `train_model.py` and displays results
- `backend/requirements.txt`: added `matplotlib` (for plotting)

- `backend/requirements.txt`: added `matplotlib` and `seaborn` (for plotting)

How to run (Windows PowerShell)

1. Create a virtual environment and activate it (recommended):

```powershell
cd d:\smart-farm-dashboard\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python requirements:

```powershell
pip install -r requirements.txt
```

3. Run training (default uses UCI Wine Quality red dataset):

```powershell
python ml/train_model.py
```

4. Optional: provide your own dataset (CSV file or URL) and target name:

```powershell
python ml/train_model.py --data "https://path/to/your.csv" --target target_column --outdir ./ml
```

If you want to use a Kaggle dataset (for example the crop/soil datasets you provided), there are two options:

- Use the Kaggle CLI to auto-download directly into the backend folder. Install the Kaggle CLI and set your credentials as described here: https://github.com/Kaggle/kaggle-api
  Then run:

```powershell
# example: download and unzip the dataset to backend folder
kaggle datasets download -d <owner>/<dataset> -p . --unzip
``` 

- Or download the dataset manually from Kaggle and place the CSV file under `backend/data/` and then run the training script with:

```powershell
python ml/train_model.py --data ./data/your_dataset.csv --target target_column --outdir ./ml
```

Outputs

- Models: `backend/ml/<model_name>.joblib`
- Reports and plots: `backend/ml/reports/` (includes `results_summary.csv` and PNGs)

Notes

- The dataset used here (Wine Quality) is for demonstration purposes and meets the assignment requirement of >500 records and multiple features. If you prefer an agriculture-specific dataset, provide a dataset URL or file and re-run the script.
- If `pip install` fails for any package, ensure you have network access and a working Python installation.
