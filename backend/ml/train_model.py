"""
Train script for a simple ML prototype.

This script downloads (if needed) a public dataset, runs preprocessing,
trains two regression models (RandomForest and GradientBoosting), evaluates
them with RMSE and MAE, saves models and preprocessing pipeline, and
produces plots and a metrics CSV.

By default it will download the UCI Wine Quality (red) dataset which is a
public CSV with >1500 records and multiple numeric features — suitable
for demonstrating a regression pipeline. Use the --data flag to provide
your own CSV path or URL.
"""

import argparse
import json
import os
from datetime import datetime
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shutil
import subprocess
import sys
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


DEFAULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)


def download_file(url, dest_path):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def load_dataset(data_arg, work_dir):
    """Load dataset from a local path or URL. Returns pandas DataFrame."""
    if data_arg is None:
        # default to UCI winequality red (semicolon-separated)
        data_arg = DEFAULT_DATA_URL

    parsed = urlparse(data_arg)
    if parsed.scheme in ("http", "https"):
        # If it's a Kaggle dataset URL, try the kaggle CLI first
        dest = None
        if 'kaggle.com/datasets' in data_arg:
            try:
                success = try_kaggle_download(data_arg, work_dir)
                if success:
                    # find the first CSV in work_dir
                    for f in os.listdir(work_dir):
                        if f.lower().endswith('.csv'):
                            dest = os.path.join(work_dir, f)
                            break
            except Exception:
                dest = None

        if dest is None:
            fname = os.path.basename(parsed.path) or "dataset.csv"
            dest = os.path.join(work_dir, fname)
            if not os.path.exists(dest):
                print(f"Downloading dataset from {data_arg} -> {dest}")
                download_file(data_arg, dest)
        # UCI winequality uses semicolon separator
        try:
            df = pd.read_csv(dest, sep=';')
        except Exception:
            df = pd.read_csv(dest)
    else:
        if not os.path.exists(data_arg):
            raise FileNotFoundError(f"Dataset path not found: {data_arg}")
        try:
            df = pd.read_csv(data_arg)
        except Exception:
            # try semicolon
            df = pd.read_csv(data_arg, sep=';')
    return df


def try_kaggle_download(kaggle_url, work_dir):
    """Attempt to download a Kaggle dataset using the kaggle CLI.
    Returns True if download and unzip succeeded, False otherwise.
    """
    kaggle_cmd = shutil.which('kaggle')
    if not kaggle_cmd:
        print('kaggle CLI not found on PATH. To auto-download from Kaggle, install the kaggle CLI and set your credentials (https://github.com/Kaggle/kaggle-api).')
        return False

    # Extract owner/dataset from URL
    try:
        parts = kaggle_url.split('kaggle.com/datasets/')[-1].strip('/')
        owner_dataset = parts.split('?')[0]
        owner_dataset = owner_dataset.rstrip('/')
    except Exception:
        print('Could not parse Kaggle URL, please provide owner/dataset form.')
        return False

    cmd = ['kaggle', 'datasets', 'download', '-d', owner_dataset, '-p', work_dir, '--unzip']
    print('Running:', ' '.join(cmd))
    try:
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(completed.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print('kaggle download failed:', e.stderr)
        return False


def validate_dataset(df, min_rows=500, min_features=4):
    rows, cols = df.shape
    if rows < min_rows:
        raise ValueError(f"Dataset has {rows} rows; requires at least {min_rows} rows")
    if cols < min_features:
        raise ValueError(f"Dataset has {cols} columns; requires at least {min_features} features")


def build_preprocessor(df, target_column):
    # Separate feature types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    if categorical_cols:
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    else:
        categorical_pipeline = None

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_cols, categorical_cols


def train(args):
    work_dir = args.outdir or os.path.join(os.getcwd(), "ml")
    os.makedirs(work_dir, exist_ok=True)
    reports_dir = os.path.join(work_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.data, work_dir)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # For the winequality dataset, the quality column is the target
    target = args.target or ("quality" if "quality" in df.columns else df.columns[-1])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    # Validate dataset size/features
    validate_dataset(df, min_rows=args.min_rows, min_features=args.min_features)

    # Basic preprocessing: drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, target)

    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # Models to train
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=args.random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=args.random_state),
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        # cross-validated RMSE (neg MSE -> RMSE)
        try:
            cv_scores = cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=5)
            cv_rmse = float(np.sqrt(-cv_scores.mean()))
        except Exception:
            cv_rmse = None

        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        model_fname = f"{name.lower().replace(' ', '_')}_model_{ts}.joblib"
        model_path = os.path.join(work_dir, model_fname)
        joblib.dump(pipe, model_path)

        results.append({"model": name, "rmse": rmse, "mae": mae, "cv_rmse": cv_rmse, "model_path": model_path})

        # Save a predicted vs actual plot
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name} Predicted vs Actual')
        plt.tight_layout()
        ppath = os.path.join(reports_dir, f'{name}_pred_vs_actual_{ts}.png')
        plt.savefig(ppath)
        plt.close()

        # If model has feature_importances_, plot them
        try:
            fi = pipe.named_steps['model'].feature_importances_
            # Get transformed feature names (rough approximation)
            feat_names = numeric_cols[:]
            if categorical_cols:
                # OneHotEncoder creates many columns; just include categorical base names
                feat_names += categorical_cols
            if len(fi) == len(feat_names):
                plt.figure(figsize=(6, 4))
                idx = np.argsort(fi)[::-1][:20]
                plt.barh([feat_names[i] for i in idx], fi[idx])
                plt.gca().invert_yaxis()
                plt.title(f'{name} Feature Importances')
                plt.tight_layout()
                fpath = os.path.join(reports_dir, f'{name}_feature_importances_{ts}.png')
                plt.savefig(fpath)
                plt.close()
        except Exception:
            pass

    # Save results to JSON and CSV
    results_path = os.path.join(reports_dir, f'results_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(reports_dir, 'results_summary.csv'), index=False)

    # Additional domain-relevant visualizations
    try:
        # Target distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(y, kde=True)
        plt.title('Target Distribution')
        plt.tight_layout()
        dist_path = os.path.join(reports_dir, f'target_distribution_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.png')
        plt.savefig(dist_path)
        plt.close()

        # Correlation heatmap (numeric features)
        num_df = df.select_dtypes(include=["number"]).copy()
        corr = num_df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        heat_path = os.path.join(reports_dir, f'correlation_heatmap_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.png')
        plt.savefig(heat_path)
        plt.close()
    except Exception as e:
        print('Could not generate additional visualizations:', e)

    print('\nTraining complete. Results:')
    print(df_res.to_string(index=False))
    print('\nArtifacts saved to', work_dir)
    print('Reports saved to', reports_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path or URL to CSV dataset')
    parser.add_argument('--target', type=str, default=None, help='Target column name (default: quality or last column)')
    parser.add_argument('--outdir', type=str, default=None, help='Directory to save models and reports')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--min-rows', type=int, default=500)
    parser.add_argument('--min-features', type=int, default=4)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
