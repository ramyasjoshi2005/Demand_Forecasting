import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import pickle
import random
import shutil
from sklearn.preprocessing import StandardScaler
from config import (
    PROCESSED_DATA_PATH, TRAIN_START, TRAIN_END, VALID_START, VALID_END,
    TEST_START, TEST_END, LSTM_MODEL_PATH, SCALER_PATH, LSTM_SEQ_LEN,
    LSTM_MAX_EPOCHS, LSTM_PATIENCE, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, LSTM_SEED,
    LSTM_STORE_EMB_DIM, LSTM_ITEM_EMB_DIM,
)
from evaluation import evaluate_model


def set_seed(seed=LSTM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, store_ids, item_ids, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.store_ids = torch.tensor(store_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.store_ids[idx], self.item_ids[idx], self.targets[idx]


class LSTMDemandModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_stores=11,
        num_items=51,
        store_emb_dim=LSTM_STORE_EMB_DIM,
        item_emb_dim=LSTM_ITEM_EMB_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    ):
        super(LSTMDemandModel, self).__init__()
        self.store_emb = nn.Embedding(num_stores, store_emb_dim)
        self.item_emb = nn.Embedding(num_items, item_emb_dim)
        self.lstm = nn.LSTM(
            num_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + store_emb_dim + item_emb_dim, 1)

    def forward(self, x_seq, x_store, x_item):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        store_emb_out = self.store_emb(x_store)
        item_emb_out = self.item_emb(x_item)
        combined = torch.cat((lstm_out, store_emb_out, item_emb_out), dim=1)
        out = self.fc(combined)
        return out.squeeze(-1)


def create_sequences(df, features, seq_len):
    """
    One-step sequences grouped by store x item.

    Window data[i : i+seq_len] (length seq_len, last row = forecast date t)
    predicts sales[t]. Features at t are leakage-safe (lags/rolling use shift(1)).
    """
    grouped = df.groupby(['store', 'item'], sort=False)
    seq_parts, store_parts, item_parts, y_parts, date_parts, sales_parts = [], [], [], [], [], []

    for (store, item), group in grouped:
        data = np.ascontiguousarray(group[features].to_numpy(dtype=np.float32))
        n = len(data)
        n_seq = n - seq_len + 1
        if n_seq <= 0:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(data, seq_len, axis=0)
        # sliding_window_view -> (n_seq, n_features, seq_len)
        windows = np.transpose(windows, (0, 2, 1))
        seq_parts.append(np.ascontiguousarray(windows))
        store_parts.append(np.full(n_seq, store, dtype=np.int32))
        item_parts.append(np.full(n_seq, item, dtype=np.int32))
        y_parts.append(group['sales_scaled'].to_numpy(dtype=np.float32)[seq_len - 1:])
        date_parts.append(group['date'].to_numpy()[seq_len - 1:])
        sales_parts.append(group['sales'].to_numpy(dtype=np.float32)[seq_len - 1:])

    sequences = np.concatenate(seq_parts, axis=0)
    store_ids = np.concatenate(store_parts, axis=0)
    item_ids = np.concatenate(item_parts, axis=0)
    targets = np.concatenate(y_parts, axis=0)
    target_dates = np.concatenate(date_parts, axis=0).astype('datetime64[ns]')
    true_sales = np.concatenate(sales_parts, axis=0)
    return sequences, store_ids, item_ids, targets, target_dates, true_sales


def _mask_dates(target_dates, start, end):
    start_ts = np.datetime64(pd.Timestamp(start))
    end_ts = np.datetime64(pd.Timestamp(end))
    return (target_dates >= start_ts) & (target_dates <= end_ts)


def train_lstm():
    set_seed(LSTM_SEED)
    print("Preparing data for LSTM...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna().sort_values(['store', 'item', 'date']).reset_index(drop=True)

    features_to_scale = [
        'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_365',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7', 'rolling_std_28', 'day_of_week', 'month',
    ]

    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    train_df = df.loc[train_mask, features_to_scale + ['sales']].copy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_df[features_to_scale])
    df[features_to_scale] = df[features_to_scale].astype(np.float64)
    df.loc[:, features_to_scale] = feature_scaler.transform(df[features_to_scale])
    target_scaler.fit(train_df[['sales']])
    df['sales_scaled'] = target_scaler.transform(df[['sales']])

    os.makedirs('models', exist_ok=True)
    if os.path.exists('models/lstm_metrics.json') and not os.path.exists('models/lstm_metrics_previous.json'):
        shutil.copyfile('models/lstm_metrics.json', 'models/lstm_metrics_previous.json')

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'features': feature_scaler, 'target': target_scaler}, f)
    with open('models/lstm_features.json', 'w') as f:
        json.dump(features_to_scale, f)

    X_seq, store_ids, item_ids, y_scaled, target_dates, true_sales = create_sequences(
        df, features_to_scale, LSTM_SEQ_LEN
    )

    train_idx = _mask_dates(target_dates, TRAIN_START, TRAIN_END)
    valid_idx = _mask_dates(target_dates, VALID_START, VALID_END)
    test_idx = _mask_dates(target_dates, TEST_START, TEST_END)

    train_dataset = TimeSeriesDataset(X_seq[train_idx], store_ids[train_idx], item_ids[train_idx], y_scaled[train_idx])
    valid_dataset = TimeSeriesDataset(X_seq[valid_idx], store_ids[valid_idx], item_ids[valid_idx], y_scaled[valid_idx])

    g = torch.Generator()
    g.manual_seed(LSTM_SEED)
    # Sequences are internally chronological; shuffling independent training windows is standard.
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    model = LSTMDemandModel(
        num_features=len(features_to_scale),
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)

    max_epochs = LSTM_MAX_EPOCHS
    patience = LSTM_PATIENCE
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    stopped_early = False
    history = []

    print(f"Training LSTM on ALL 500 store-item series ({len(train_dataset)} samples)...")
    print(
        f"Config: max_epochs={max_epochs}, patience={patience}, batch_size={LSTM_BATCH_SIZE}, "
        f"lr={LSTM_LEARNING_RATE}, seq_len={LSTM_SEQ_LEN}, seed={LSTM_SEED}"
    )

    last_epoch = 0
    for epoch in range(max_epochs):
        last_epoch = epoch + 1
        model.train()
        train_loss = 0.0
        for batch_seq, batch_store, batch_item, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_seq, batch_store, batch_item)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_seq, batch_store, batch_item, batch_y in valid_loader:
                outputs = model(batch_seq, batch_store, batch_item)
                loss = criterion(outputs, batch_y)
                valid_loss += loss.item()

        t_loss = train_loss / max(len(train_loader), 1)
        v_loss = valid_loss / max(len(valid_loader), 1)
        print(f"Epoch {last_epoch}/{max_epochs} - Train Loss: {t_loss:.4f} - Valid Loss: {v_loss:.4f}")
        history.append({'epoch': last_epoch, 'train_loss': t_loss, 'valid_loss': v_loss})

        if v_loss < best_loss:
            best_loss = v_loss
            best_epoch = last_epoch
            torch.save(model.state_dict(), LSTM_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stopped_early = True
            print(f"Early stopping triggered at epoch {last_epoch} (best epoch {best_epoch}).")
            break

    pd.DataFrame(history).to_csv('models/lstm_training_history.csv', index=False)

    if best_epoch == 0:
        torch.save(model.state_dict(), LSTM_MODEL_PATH)
        best_epoch = last_epoch

    stop_reason = 'early_stopping' if stopped_early else 'max_epochs'

    print(f"Evaluating LSTM on FULL Test Set (Best Epoch: {best_epoch}, stop_reason: {stop_reason})...")
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True, map_location='cpu'))
    model.eval()

    test_dataset = TimeSeriesDataset(
        X_seq[test_idx], store_ids[test_idx], item_ids[test_idx], y_scaled[test_idx]
    )
    test_loader = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)
    test_true_sales = true_sales[test_idx]

    test_preds_scaled = []
    with torch.no_grad():
        for batch_seq, batch_store, batch_item, _ in test_loader:
            preds = model(batch_seq, batch_store, batch_item)
            test_preds_scaled.extend(preds.numpy())

    test_preds_scaled = np.array(test_preds_scaled).reshape(-1, 1)
    test_preds = target_scaler.inverse_transform(test_preds_scaled).flatten()
    # Clip negatives only. Do not round — LightGBM test metrics use raw predictions.
    test_preds = np.clip(test_preds, 0, None)

    n_series = pd.DataFrame({
        'store': store_ids[test_idx],
        'item': item_ids[test_idx],
    }).drop_duplicates().shape[0]
    print(f"Test evaluation: {n_series} store-item series, {len(test_true_sales)} rows, {TEST_START} to {TEST_END}")

    metrics = evaluate_model(test_true_sales, test_preds, "LSTM (All 500 Series)")

    metrics_out = {
        "model": "LSTM",
        "training_population": "all 500 series",
        "evaluation_population": "all 500 series",
        "sequence_length": LSTM_SEQ_LEN,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": LSTM_BATCH_SIZE,
        "learning_rate": LSTM_LEARNING_RATE,
        "hidden_size": LSTM_HIDDEN_SIZE,
        "num_layers": LSTM_NUM_LAYERS,
        "dropout": LSTM_DROPOUT,
        "seed": LSTM_SEED,
        "epochs_trained": last_epoch,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "stop_reason": stop_reason,
        "n_test_rows": int(len(test_true_sales)),
        "n_test_series": int(n_series),
        "mae": metrics['mae'],
        "rmse": metrics['rmse'],
        "mape": metrics['mape'],
        "smape": metrics['smape'],
        "test_start": TEST_START,
        "test_end": TEST_END,
    }

    with open('models/lstm_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    comp_file = 'models/model_comparison.csv'
    comp_df = pd.DataFrame([{
        'model': 'LSTM',
        'evaluation_population': 'all 500 series',
        'MAE': metrics['mae'],
        'RMSE': metrics['rmse'],
        'MAPE': metrics['mape'],
        'sMAPE': metrics['smape'],
        'test_start': TEST_START,
        'test_end': TEST_END,
    }])

    if os.path.exists(comp_file):
        existing = pd.read_csv(comp_file)
        existing = existing[existing['model'] != 'LSTM']
        pd.concat([existing, comp_df]).to_csv(comp_file, index=False)
    else:
        comp_df.to_csv(comp_file, index=False)


if __name__ == "__main__":
    train_lstm()
