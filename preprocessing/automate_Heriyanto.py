
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def automate_preprocessing(
    file_path="data.csv",
    sep=';',
    handle_missing='median',
    drop_duplicates=True,
    handle_outliers=True,
    encoding_type='onehot',
    binning_info=None,
    scaling_method='standard',
    output_file='data_processed.csv'
):
    df = pd.read_csv(file_path, sep=sep)

    # 1. Handle Missing Values
    if handle_missing == 'drop':
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    if handle_missing == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif handle_missing == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Drop Duplicates
    if drop_duplicates:
        df = df.drop_duplicates()

    # 3. Handle Outliers
    if handle_outliers:
        num_cols = df.select_dtypes(include='number').columns
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    # 4. Encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if encoding_type == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif encoding_type == 'label':
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
    else:
        raise ValueError("encoding_type harus 'onehot' atau 'label'")

    # 5. Binning (optional)
    if binning_info:
        for col, bin_conf in binning_info.items():
            method = bin_conf.get('method', 'cut')
            bins = bin_conf['bins']
            labels = bin_conf.get('labels', None)

            if method == 'cut':
                df[f'{col}_bin'] = pd.cut(df[col], bins=bins, labels=labels)
            elif method == 'qcut':
                df[f'{col}_bin'] = pd.qcut(df[col], q=bins, labels=labels)
            else:
                raise ValueError("Metode binning harus 'cut' atau 'qcut'.")

    # 6. Scaling
    if scaling_method:
        scaler = MinMaxScaler() if scaling_method == 'minmax' else StandardScaler()
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # 7. Simpan hasil
    df.to_csv(output_file, index=False)
    print(f"✅ Data berhasil disimpan ke {output_file}")

    return df
