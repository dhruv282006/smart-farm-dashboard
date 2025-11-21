"""
Utility to load the three user CSVs (soil_moisture, plant_vase1, pesticides),
attempt a reasonable merge, and print inspection info. Also saves the
terminal-style output to a text file and a PNG under the run reports folder
so you have a reproducible "screenshot" to embed in the notebook.

This is intentionally conservative: it merges on 'year' when present which
is a robust common field across the three files. If you'd rather merge on
datetime/nearest timestamp, tell me and I can update the script.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def ensure_reports_dir():
    out = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(out, exist_ok=True)
    return out


def load_soil(path):
    df = pd.read_csv(path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['year'] = df['datetime'].dt.year
    return df


def load_plant(path):
    df = pd.read_csv(path)
    # Try to construct a datetime if components exist
    if set(['year','month','day','hour','minute','second']).issubset(df.columns):
        df['datetime'] = pd.to_datetime(df[['year','month','day','hour','minute','second']], errors='coerce')
        # keep the numeric year column
    else:
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df


def load_pesticides(path):
    df = pd.read_csv(path)
    # Normalize Year -> year
    if 'Year' in df.columns and 'year' not in df.columns:
        df = df.rename(columns={'Year': 'year'})
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data')
    base = os.path.normpath(base)
    soil_path = os.path.join(base, 'soil_moisture.csv')
    plant_path = os.path.join(base, 'plant_vase1.CSV')
    pest_path = os.path.join(base, 'pesticides.csv')

    reports = ensure_reports_dir()

    # Load files (best-effort)
    soil_df = load_soil(soil_path) if os.path.exists(soil_path) else pd.DataFrame()
    plant_df = load_plant(plant_path) if os.path.exists(plant_path) else pd.DataFrame()
    pest_df = load_pesticides(pest_path) if os.path.exists(pest_path) else pd.DataFrame()

    # Strategy: merge on 'year' if present in at least two dfs; otherwise concatenate
    common_year = all(x.shape[1] > 0 and 'year' in x.columns for x in [soil_df, plant_df, pest_df])

    if common_year:
        # Merge by 'year' (outer) so we keep all available records for inspection
        merged = pd.merge(soil_df, plant_df, on='year', how='outer', suffixes=('_soil','_plant'))
        merged = pd.merge(merged, pest_df, on='year', how='left', suffixes=('','_pest'))
    else:
        # fallback: concat with keys to keep provenance
        try:
            soil_df['_source'] = 'soil'
            plant_df['_source'] = 'plant'
            pest_df['_source'] = 'pesticides'
            merged = pd.concat([soil_df, plant_df, pest_df], sort=False, ignore_index=True)
        except Exception:
            merged = pd.DataFrame()

    final_df = merged

    # Prepare text output
    buf_lines = []
    buf_lines.append('--- Final Merged DataFrame Shape ---')
    buf_lines.append(str(final_df.shape))
    buf_lines.append('')
    buf_lines.append('--- Final Merged DataFrame Info ---')
    # capture info() output
    from io import StringIO
    s = StringIO()
    final_df.info(buf=s)
    buf_lines.append(s.getvalue())
    buf_lines.append('')
    buf_lines.append('--- Final Merged DataFrame Head ---')
    buf_lines.append(final_df.head().to_string())

    out_text = '\n'.join(buf_lines)

    # Print to stdout (so it appears in the terminal)
    print(out_text)

    # Save raw text
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    txt_path = os.path.join(reports, f'merge_terminal_{ts}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(out_text)

    # Also create a PNG that looks like a terminal screenshot using matplotlib
    try:
        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor('black')
        plt.axis('off')
        # Use monospaced font; wrap text
        plt.text(0.01, 0.99, out_text, fontfamily='monospace', fontsize=10, color='white', va='top')
        png_path = os.path.join(reports, f'merge_terminal_{ts}.png')
        plt.savefig(png_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print('\nSaved terminal-like PNG to', png_path)
        print('Saved raw text to', txt_path)
    except Exception as e:
        print('Failed to render PNG:', e)


if __name__ == '__main__':
    main()
