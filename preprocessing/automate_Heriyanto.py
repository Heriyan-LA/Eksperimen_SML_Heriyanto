import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def handle_missing_values(df, categorical_cols, numerical_cols):
    """
    Mengisi missing values untuk kolom numerik dan kategorikal
    """
    df = df.copy()

    # Imputasi numerik
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Imputasi kategorikal
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df

def encode_categorical_features(df, categorical_cols):
    """
    One-hot encoding untuk kolom kategorikal
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
    return encoded_df, encoder

def normalize_numeric_features(df, numerical_cols):
    """
    Normalisasi fitur numerik dengan MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=[f"{col}_scaled" for col in numerical_cols])
    return scaled_df, scaler

def preprocess_movie_data(df, categorical_cols=['genre'], numerical_cols=['duration', 'rating', 'budget']):
    """
    Pipeline preprocessing utama, mengembalikan dataframe siap latih
    """
    df = df.copy()

    # Step 1: Missing value handling
    df = handle_missing_values(df, categorical_cols, numerical_cols)

    # Step 2: Encoding kategorikal
    encoded_df, _ = encode_categorical_features(df, categorical_cols)

    # Step 3: Normalisasi numerik
    scaled_df, _ = normalize_numeric_features(df, numerical_cols)

    # Gabungkan hasil akhir
    df_final = pd.concat([
        df.drop(columns=categorical_cols + numerical_cols).reset_index(drop=True),
        encoded_df.reset_index(drop=True),
        scaled_df.reset_index(drop=True)
    ], axis=1)

    return df_final
