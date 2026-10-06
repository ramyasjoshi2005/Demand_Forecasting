import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'train_featured.csv')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LIGHTGBM_MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_model.txt')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.pth')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

TRAIN_START = '2013-01-01'
TRAIN_END = '2016-12-31'
VALID_START = '2017-01-01'
VALID_END = '2017-09-30'
TEST_START = '2017-10-01'
TEST_END = '2017-12-31'

# Residual window used for the primary inventory error estimate.
# This overlaps the LightGBM/LSTM early-stopping validation period (documented limitation).
RESIDUAL_START = VALID_START
RESIDUAL_END = VALID_END
# Later slice of validation, used only as a sensitivity check (still not fully independent).
RESIDUAL_SENSITIVITY_START = '2017-07-01'
RESIDUAL_SENSITIVITY_END = VALID_END

LSTM_SEQ_LEN = 30
LSTM_MAX_EPOCHS = 40
LSTM_PATIENCE = 7
LSTM_BATCH_SIZE = 256
LSTM_LEARNING_RATE = 0.005
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2
LSTM_SEED = 42
LSTM_STORE_EMB_DIM = 4
LSTM_ITEM_EMB_DIM = 8
