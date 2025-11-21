"""
Train two regression algorithms (DecisionTree and RandomForest) on the merged
datasets and print evaluation metrics to stdout. Also saves the printed output
as raw text and a PNG "terminal screenshot" under `backend/ml/reports/` so it can
be embedded in the notebook as the required screenshot for the "Algorithms Used" section.

This script chooses a safe default target column available in the provided
datasets (`soil_moisture`). If you want to train on a different target,
change `target_col` accordingly.
"""
import os
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn


def ensure_reports_dir():
    out = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(out, exist_ok=True)
    return out


def load_and_merge():
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    soil_path = os.path.join(base, 'soil_moisture.csv')
    plant_path = os.path.join(base, 'plant_vase1.CSV')
    pest_path = os.path.join(base, 'pesticides.csv')

    soil_df = pd.read_csv(soil_path) if os.path.exists(soil_path) else pd.DataFrame()
    plant_df = pd.read_csv(plant_path) if os.path.exists(plant_path) else pd.DataFrame()
    pest_df = pd.read_csv(pest_path) if os.path.exists(pest_path) else pd.DataFrame()

    # try to make year available for all as in the inspector script
    if 'datetime' in soil_df.columns:
        soil_df['datetime'] = pd.to_datetime(soil_df['datetime'], errors='coerce')
        soil_df['year'] = soil_df['datetime'].dt.year
    if set(['year','month','day','hour','minute','second']).issubset(plant_df.columns):
        plant_df['datetime'] = pd.to_datetime(plant_df[['year','month','day','hour','minute','second']], errors='coerce')
        # plant_df already has 'year'
    if 'Year' in pest_df.columns and 'year' not in pest_df.columns:
        pest_df = pest_df.rename(columns={'Year': 'year'})
    if 'year' in pest_df.columns:
        pest_df['year'] = pd.to_numeric(pest_df['year'], errors='coerce')

    # Merge on year if available, else concat
    common_year = all(x.shape[0] > 0 and 'year' in x.columns for x in [soil_df, plant_df, pest_df])
    if common_year:
        merged = pd.merge(soil_df, plant_df, on='year', how='outer', suffixes=('_soil', '_plant'))
        merged = pd.merge(merged, pest_df, on='year', how='left', suffixes=('', '_pest'))
    else:
        soil_df['_source'] = 'soil'
        plant_df['_source'] = 'plant'
        pest_df['_source'] = 'pesticides'
        merged = pd.concat([soil_df, plant_df, pest_df], sort=False, ignore_index=True)

    return merged


def reg_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred))
    }


def main():
    reports = ensure_reports_dir()
    final_df = load_and_merge()

    # --- CRITICAL: Choose target column ---
    # The assignment expects a crop yield column (Crop_Yield). In these datasets
    # that column does not exist. We'll use 'soil_moisture' as a meaningful numeric
    # target for demonstration. Change target_col if you have a different variable.
    target_col = 'soil_moisture'

    if target_col not in final_df.columns:
        raise SystemExit(f"Target column '{target_col}' not found in merged DataFrame. Columns: {final_df.columns.tolist()[:20]} ...")

    model_df = final_df.copy()
    X = model_df.drop(columns=[target_col])
    y = model_df[target_col].values

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    out_lines = []
    out_lines.append('--- Model Training ---')
    out_lines.append(f"Final feature counts: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    # OneHotEncoder API compatibility
    if 'sparse_output' in OneHotEncoder.__init__.__code__.co_varnames:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols) if num_cols else ('num', 'passthrough', []),
        ('cat', encoder, cat_cols) if cat_cols else ('cat', 'passthrough', [])
    ], remainder='drop')

    # Split (drop rows with missing target)
    mask = ~pd.isna(y)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_pipe = Pipeline([('pre', preprocessor), ('dt', DecisionTreeRegressor(random_state=42))])
    rf_pipe = Pipeline([('pre', preprocessor), ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

    out_lines.append('Training Decision Tree...')
    dt_pipe.fit(X_train, y_train)
    out_lines.append('Training Random Forest...')
    rf_pipe.fit(X_train, y_train)

    y_pred_dt = dt_pipe.predict(X_test)
    y_pred_rf = rf_pipe.predict(X_test)

    metrics_dt = reg_metrics(y_test, y_pred_dt)
    metrics_rf = reg_metrics(y_test, y_pred_rf)

    out_lines.append('\n--- MODEL METRICS (Screenshot This) ---')
    out_lines.append(f"Decision Tree metrics: {metrics_dt}")
    out_lines.append(f"Random Forest metrics: {metrics_rf}")
    out_lines.append('----------------------------------------')

    out_text = '\n'.join(out_lines)
    print(out_text)

    # save raw text
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    txt_path = os.path.join(reports, f'algorithms_terminal_{ts}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(out_text)

    # render PNG (terminal-like)
    try:
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('black')
        plt.axis('off')
        plt.text(0.01, 0.99, out_text, fontfamily='monospace', fontsize=10, color='white', va='top')
        png_path = os.path.join(reports, f'algorithms_terminal_{ts}.png')
        plt.savefig(png_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print('\nSaved terminal-like PNG to', png_path)
        print('Saved raw text to', txt_path)
    except Exception as e:
        print('Failed to render PNG:', e)

    # --- Charts for Results & Analysis ---
    try:
        # Chart 1: Model comparison bar chart
        metrics_df = pd.DataFrame([metrics_dt, metrics_rf], index=['DecisionTree', 'RandomForest'])
        bar_path = os.path.join(reports, f'model_comparison_{ts}.png')
        ax = metrics_df.plot(kind='bar', figsize=(8, 4), rot=0)
        ax.set_title('Model Comparison')
        plt.tight_layout()
        plt.savefig(bar_path)
        print('Saved model comparison chart to', bar_path)

        # Chart 2: Actual vs Predicted (Random Forest)
        scatter_path = os.path.join(reports, f'rf_actual_vs_predicted_{ts}.png')
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred_rf, alpha=0.6)
        minv = min(np.nanmin(y_test), np.nanmin(y_pred_rf))
        maxv = max(np.nanmax(y_test), np.nanmax(y_pred_rf))
        plt.plot([minv, maxv], [minv, maxv], 'r--')
        plt.xlabel(f'Actual {target_col}')
        plt.ylabel(f'Predicted {target_col}')
        plt.title(f'Random Forest: Actual vs Predicted ({target_col})')
        plt.tight_layout()
        plt.savefig(scatter_path)
        print('Saved actual vs predicted chart to', scatter_path)

        # Try to open plot windows (best-effort). In headless env this may be a no-op.
        try:
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print('Could not generate charts:', e)


if __name__ == '__main__':
    main()
