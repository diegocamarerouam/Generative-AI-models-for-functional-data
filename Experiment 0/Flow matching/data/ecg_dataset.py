# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ast
import wfdb
import torch

from scipy.signal import butter, filtfilt

def apply_bandpass_filter(signals, sampling_rate, lowcut=0.5, highcut=40.0, order=2):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered_signals = filtfilt(b, a, signals, axis=-1)
    
    return torch.tensor(filtered_signals.copy(), dtype=torch.float32)

def get_superclass(scp_dict, scp_df):
    diag = {k: v for k, v in scp_dict.items()
            if k in scp_df.index and scp_df.loc[k, 'diagnostic'] == 1.0}
    if not diag:
        return None
    return max(diag, key=diag.get)

def load_signals(df, sampling_rate, path, max_samples=None):
    col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    fnames = df[col].values
    if max_samples is not None:
        fnames = fnames[:max_samples]
    signals = []
    for fname in fnames:
        sig, _ = wfdb.rdsamp(path + fname)
        signals.append(sig)
    return np.array(signals)

def load_ecg_data(dataset_path, sampling_rate, lead=1, max_samples=None):
    df = pd.read_csv(dataset_path + 'ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp_df = pd.read_csv(dataset_path + 'scp_statements.csv', index_col=0)

    df['superclass'] = df.scp_codes.apply(get_superclass, args=(scp_df,))
    normal_df = df[df['superclass'] == 'NORM']

    X = load_signals(normal_df, sampling_rate, dataset_path, max_samples=max_samples)
    return torch.tensor(X[:, :, lead], dtype=torch.float32)

if __name__ == "__main__":
    import torch

    SAMPLING_RATE = 100
    DATASET_PATH = '/content/drive/MyDrive/Trabajo Fin de Máster/Code/Experiment 0/ECG dataset/'
    MAX_SAMPLES = None
    LEAD = 1
    CACHE_PATH = DATASET_PATH + f'cache_lead{LEAD}_sr{SAMPLING_RATE}_n{MAX_SAMPLES}.pt'
    
    data_train = load_ecg_data(
        dataset_path=DATASET_PATH,
        sampling_rate=SAMPLING_RATE,
        lead=LEAD,
        max_samples=MAX_SAMPLES,
    ).unsqueeze(1)
    
    print(f"Dataset shape: {data_train.shape}")
    
    torch.save({'data': data_train}, CACHE_PATH)
    print("Saved to cache.")