import os
import pandas as pd
from glob import glob

data_dir = os.path.join(os.path.dirname(__file__), 'data')
files = sorted(glob(os.path.join(data_dir, '*')))
for f in files:
    if f.lower().endswith('.csv'):
        try:
            # try common separators
            try:
                df = pd.read_csv(f)
            except Exception:
                df = pd.read_csv(f, sep=';')
            print('FILE:', os.path.basename(f))
            print('ROWS,COLS:', df.shape)
            print('COLUMNS:', list(df.columns))
            print('HEAD:')
            print(df.head(3).to_csv(index=False))
            print('---')
        except Exception as e:
            print('FILE:', os.path.basename(f), 'ERROR:', e)
