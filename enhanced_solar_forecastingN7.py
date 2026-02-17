# ============================================================================
# ENHANCED SOLAR FORECASTING SYSTEM
# - Removed XGBoost
# - Added long-term prediction visualization for each model
# - Added CMIP6 scenario integration (SSP2-4.5 & SSP5-8.5)
# - Display with and without scenarios
# ============================================================================

import os
import sys
import json
import copy
import time
import random
import itertools
import warnings
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
GEN = torch.Generator()
GEN.manual_seed(SEED)

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ── Journal-quality plot styling ──────────────────────────────────────────
CB_PALETTE = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'vermilion': '#D55E00', 'sky_blue': '#56B4E9', 'pink': '#CC79A7',
    'yellow': '#F0E442', 'black': '#000000',
}

def set_journal_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.linewidth': 0.8, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    })

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
CONFIG = {
    'features': ['ALLSKY_KT', 'T2M', 'RH2M'],
    'target': 'GHI',
    'seq_len': 12,
    'label_len': 6,
    'pred_len': 1,
    'aggregation': 'weekly',
    'model_params': {
        'informer': {
            'embed_size': 512, 'n_heads': 8, 'e_layers': 3,
            'd_layers': 2, 'd_ff': 2048, 'dropout': 0.1,
            'activation': 'gelu'
        },
        'transformer': {
            'd_model': 256, 'n_heads': 8, 'num_encoder_layers': 4,
            'num_decoder_layers': 3, 'dim_feedforward': 1024,
            'dropout': 0.1
        },
        'lstm': {
            'hidden_size': 512, 'num_layers': 2, 'dropout': 0.3
        },
        'gru': {
            'hidden_size': 512, 'num_layers': 2, 'dropout': 0.3
        },
        'hybrid': {
            'lstm_hidden_size': 256, 'lstm_num_layers': 2,
            'lstm_dropout': 0.3,
            'embed_size': 512, 'n_heads': 8, 'e_layers': 3,
            'd_layers': 2, 'd_ff': 2048, 'dropout': 0.1,
            'activation': 'gelu'
        },
        'hybrid_transformer_lstm': {
            'lstm_hidden_size': 256, 'lstm_num_layers': 2,
            'lstm_dropout': 0.3,
            'd_model': 256, 'n_heads': 8,
            'num_encoder_layers': 3, 'num_decoder_layers': 2,
            'dim_feedforward': 1024, 'dropout': 0.1,
        },
    },
    'training_params': {
        'batch_size': 32, 'epochs': 150, 'learning_rate': 0.0001,
        'patience': 15, 'weight_decay': 1e-5
    },
    'monte_carlo': {
        'n_sims': 10000, 'use_mc_dropout': True, 'dropout_rate': 0.3,
        'n_mc_samples': 100,
        'confidence_levels': [0.50, 0.80, 0.90, 0.95, 0.99],
        'distribution': 'gaussian', 'residual_noise': False
    },
    'sobol': {'n_samples': 1024, 'calc_second_order': False},
    'backtesting': {
        'window_sizes': [120, 180], 'step_size': 12,
        'horizons': [12, 36, 48, 60]
    },
    'calibration': {'n_bins': 10, 'pit_bins': 20},
    'ablation': {
        'with_humidity': True, 'decade_sensitivity': False,
        'time_varying_stability': True
    },
    'multi_site': {
        'enabled': False, 'site_files': [], 'results_comparison': True
    },
    'cmip6': {
        'scenarios': ['ssp245', 'ssp585'],
        'bias_correction_method': 'quantile_mapping',
        'validation_years': [2025, 2030, 2035, 2040],
        'generate_synthetic': True,
        'synthetic_method': 'trend_plus_seasonality',
        'scenario_files': {
            'ssp245': 'cmip6_ssp245.csv',
            'ssp585': 'cmip6_ssp585.csv',
        },
    },
    'long_term': {
        'forecast_years': [2025, 2026, 2027, 2028, 2029, 2030,
                           2035, 2040],
        'n_mc_longterm': 200,
    },
    # ── OPTIMISED BASELINE CONFIGURATION ────────────────────────────
    'baseline': {
        'persistence': {
            'seasonal_periods': [4, 12, 26, 52],
            'use_drift': True,
        },
        'arima': {
            'max_p': 4, 'max_d': 2, 'max_q': 4,
            'information_criterion': 'aic',
            'max_order_sum': 8,
            'stepwise': True,
        },
        'sarima': {
            'max_P': 2, 'max_D': 1, 'max_Q': 2,
            'seasonal_periods': [12, 26, 52],
            'information_criterion': 'aic',
        },
    },
    'output_dir': 'solar_forecasting_results'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ============================================================================
# DATASET CLASSES
# ============================================================================
class InformerDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, label_len, pred_len):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.length = max(0, len(data_x) - seq_len - pred_len + 1)

    def __getitem__(self, i):
        s_end = i + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[i:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_label = self.data_y[r_begin:r_begin + self.label_len]
        dec_input = np.concatenate([
            seq_y_label,
            np.zeros((self.pred_len, self.data_y.shape[1]))
        ], axis=0)
        return (torch.FloatTensor(seq_x),
                torch.FloatTensor(dec_input),
                torch.FloatTensor(seq_y))

    def __len__(self):
        return self.length


class TransformerDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length = max(0, len(data_x) - seq_len - pred_len + 1)

    def __getitem__(self, i):
        src = self.data_x[i:i + self.seq_len]
        tgt = self.data_y[i:i + self.seq_len]
        tgt_y = self.data_y[i + self.seq_len:i + self.seq_len + self.pred_len]
        return (torch.FloatTensor(src),
                torch.FloatTensor(tgt),
                torch.FloatTensor(tgt_y))

    def __len__(self):
        return self.length


class RNNDataset(Dataset):
    def __init__(self, data, seq_len, feature_indices, target_index):
        self.data = data
        self.seq_len = seq_len
        self.feature_indices = feature_indices
        self.target_index = target_index
        self.length = max(0, len(data) - seq_len)

    def __getitem__(self, i):
        seq_x = self.data[i:i + self.seq_len, self.feature_indices]
        seq_y = self.data[i + self.seq_len, self.target_index]
        return (torch.FloatTensor(seq_x),
                torch.FloatTensor([seq_y]))

    def __len__(self):
        return self.length


# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_keys = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_keys ** -0.5
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scores * self.scale, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", attn, values).contiguous()
        V = V.view(B, L, -1)
        return self.out_projection(V)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()
        self.self_attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.cross_attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = nn.Embedding(500, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).long()
        pos = torch.clamp(pos, 0, self.position_embedding.num_embeddings - 1)
        x = self.value_embedding(x) + self.position_embedding(pos)
        return self.dropout(x)


# ============================================================================
# MODELS
# ============================================================================
class InformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, **params):
        super().__init__()
        self.enc_embedding = DataEmbedding(
            enc_in, params['embed_size'], params['dropout'])
        self.dec_embedding = DataEmbedding(
            dec_in, params['embed_size'], params['dropout'])
        self.encoder = nn.ModuleList([
            EncoderLayer(params['embed_size'], params['n_heads'],
                         params['d_ff'], params['dropout'],
                         params['activation'])
            for _ in range(params['e_layers'])
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(params['embed_size'], params['n_heads'],
                         params['d_ff'], params['dropout'],
                         params['activation'])
            for _ in range(params['d_layers'])
        ])
        self.projection = nn.Linear(params['embed_size'], c_out, bias=True)
        self.mc_p = CONFIG['monte_carlo']['dropout_rate']

    def forward(self, x_enc, x_dec, use_mc_dropout=False):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out)
            if use_mc_dropout:
                enc_out = F.dropout(enc_out, p=self.mc_p, training=True)
        dec_out = self.dec_embedding(x_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)
            if use_mc_dropout:
                dec_out = F.dropout(dec_out, p=self.mc_p, training=True)
        return self.projection(dec_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, **params):
        super().__init__()
        self.d_model = params['d_model']
        self.enc_input_projection = nn.Linear(enc_in, self.d_model)
        self.dec_input_projection = nn.Linear(dec_in, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, params['dropout'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=params['n_heads'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'], activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=params['num_encoder_layers'])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=params['n_heads'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'], activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=params['num_decoder_layers'])
        self.output_projection = nn.Linear(self.d_model, c_out)
        self.mc_p = CONFIG['monte_carlo']['dropout_rate']

    def forward(self, x_enc, x_dec, use_mc_dropout=False):
        tgt_seq_len = x_dec.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq_len).to(x_enc.device)
        x_enc_embed = self.pos_encoder(
            self.enc_input_projection(x_enc) * np.sqrt(self.d_model))
        memory = self.transformer_encoder(x_enc_embed)
        if use_mc_dropout:
            memory = F.dropout(memory, p=self.mc_p, training=True)
        x_dec_embed = self.pos_encoder(
            self.dec_input_projection(x_dec) * np.sqrt(self.d_model))
        output = self.transformer_decoder(
            x_dec_embed, memory, tgt_mask=tgt_mask)
        if use_mc_dropout:
            output = F.dropout(output, p=self.mc_p, training=True)
        final_prediction = self.output_projection(output)
        return final_prediction[:, -1, :].unsqueeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.output_dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.mc_p = CONFIG['monte_carlo']['dropout_rate']

    def forward(self, x, use_mc_dropout=False):
        if use_mc_dropout:
            x = F.dropout(x, p=self.mc_p, training=True)
        else:
            x = self.input_dropout(x)
        if use_mc_dropout:
            self.lstm.train()
        lstm_out, _ = self.lstm(x)
        if use_mc_dropout:
            lstm_out = F.dropout(lstm_out, p=self.mc_p, training=True)
            self.lstm.eval()
        last_output = lstm_out[:, -1, :]
        if use_mc_dropout:
            last_output = F.dropout(last_output, p=self.mc_p, training=True)
        else:
            last_output = self.output_dropout(last_output)
        return self.linear(last_output)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.output_dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.mc_p = CONFIG['monte_carlo']['dropout_rate']

    def forward(self, x, use_mc_dropout=False):
        if use_mc_dropout:
            x = F.dropout(x, p=self.mc_p, training=True)
        else:
            x = self.input_dropout(x)
        gru_out, _ = self.gru(x)
        if use_mc_dropout:
            gru_out = F.dropout(gru_out, p=self.mc_p, training=True)
        last_output = gru_out[:, -1, :]
        if use_mc_dropout:
            last_output = F.dropout(last_output, p=self.mc_p, training=True)
        else:
            last_output = self.output_dropout(last_output)
        return self.linear(last_output)


class HybridLSTMInformerModel(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, **params):
        super().__init__()
        lstm_hidden = params.get('lstm_hidden_size', 128)
        lstm_layers = params.get('lstm_num_layers', 2)
        lstm_dropout = params.get('lstm_dropout', 0.3)
        embed_size = params.get('embed_size', 512)
        n_heads = params.get('n_heads', 4)
        e_layers = params.get('e_layers', 3)
        d_layers = params.get('d_layers', 2)
        d_ff = params.get('d_ff', 256)
        dropout = params.get('dropout', 0.1)
        activation = params.get('activation', 'gelu')
        self.input_projection = nn.Linear(enc_in, lstm_hidden)
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=lstm_dropout if lstm_layers > 1 else 0,
                            bidirectional=False)
        self.lstm_to_informer = nn.Linear(lstm_hidden, embed_size)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.enc_embedding = DataEmbedding(embed_size, embed_size, dropout)
        self.dec_embedding = DataEmbedding(dec_in, embed_size, dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(embed_size, n_heads, d_ff, dropout, activation)
            for _ in range(e_layers)])
        self.decoder = nn.ModuleList([
            DecoderLayer(embed_size, n_heads, d_ff, dropout, activation)
            for _ in range(d_layers)])
        self.projection = nn.Linear(embed_size, c_out, bias=True)
        self.mc_p = CONFIG['monte_carlo']['dropout_rate']
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden

    def forward(self, x_enc, x_dec, use_mc_dropout=False):
        lstm_input = self.input_projection(x_enc)
        if use_mc_dropout:
            lstm_input = F.dropout(lstm_input, p=self.mc_p, training=True)
        lstm_out, _ = self.lstm(lstm_input)
        if use_mc_dropout:
            lstm_out = F.dropout(lstm_out, p=self.mc_p, training=True)
        lstm_latent = self.lstm_to_informer(lstm_out)
        if use_mc_dropout:
            lstm_latent = F.dropout(lstm_latent, p=self.mc_p, training=True)
        enc_out = self.enc_embedding(lstm_latent)
        for layer in self.encoder:
            enc_out = layer(enc_out)
            if use_mc_dropout:
                enc_out = F.dropout(enc_out, p=self.mc_p, training=True)
        dec_out = self.dec_embedding(x_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)
            if use_mc_dropout:
                dec_out = F.dropout(dec_out, p=self.mc_p, training=True)
        return self.projection(dec_out)


class HybridTransformerLSTMModel(nn.Module):
    """Hybrid Transformer-LSTM with BiLSTM encoder, Transformer decoder,
    and gated fusion."""
    def __init__(self, enc_in, dec_in, c_out, **params):
        super().__init__()
        lstm_hidden = params.get('lstm_hidden_size', 256)
        lstm_layers = params.get('lstm_num_layers', 2)
        lstm_dropout = params.get('lstm_dropout', 0.3)
        d_model = params.get('d_model', 256)
        n_heads = params.get('n_heads', 8)
        n_enc_layers = params.get('num_encoder_layers', 3)
        n_dec_layers = params.get('num_decoder_layers', 2)
        dim_ff = params.get('dim_feedforward', 1024)
        dropout = params.get('dropout', 0.1)
        self.d_model = d_model
        self.lstm_input_proj = nn.Linear(enc_in, lstm_hidden)
        self.bilstm = nn.LSTM(
            input_size=lstm_hidden, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True)
        self.lstm_to_transformer = nn.Linear(2 * lstm_hidden, d_model)
        self.lstm_layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dec_input_proj = nn.Linear(dec_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_dec_layers)
        self.output_projection = nn.Linear(d_model, c_out)
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.gate_sigmoid = nn.Sigmoid()
        self.mc_p = CONFIG['monte_carlo']['dropout_rate']

    def forward(self, x_enc, x_dec, use_mc_dropout=False):
        lstm_in = self.lstm_input_proj(x_enc)
        if use_mc_dropout:
            lstm_in = F.dropout(lstm_in, p=self.mc_p, training=True)
        lstm_out, _ = self.bilstm(lstm_in)
        if use_mc_dropout:
            lstm_out = F.dropout(lstm_out, p=self.mc_p, training=True)
        lstm_features = self.lstm_layer_norm(
            self.lstm_to_transformer(lstm_out))
        if use_mc_dropout:
            lstm_features = F.dropout(
                lstm_features, p=self.mc_p, training=True)
        enc_input = self.pos_encoder(
            lstm_features * np.sqrt(self.d_model))
        memory = self.transformer_encoder(enc_input)
        if use_mc_dropout:
            memory = F.dropout(memory, p=self.mc_p, training=True)
        tgt_seq_len = x_dec.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq_len).to(x_enc.device)
        dec_embed = self.pos_encoder(
            self.dec_input_proj(x_dec) * np.sqrt(self.d_model))
        dec_out = self.transformer_decoder(
            dec_embed, memory, tgt_mask=tgt_mask)
        if use_mc_dropout:
            dec_out = F.dropout(dec_out, p=self.mc_p, training=True)
        lstm_summary = lstm_features[:, -1:, :].expand_as(dec_out)
        gate_input = torch.cat([dec_out, lstm_summary], dim=-1)
        gate = self.gate_sigmoid(self.gate_linear(gate_input))
        fused = gate * dec_out + (1 - gate) * lstm_summary
        return self.output_projection(fused)


# ============================================================================
# OPTIMISED BASELINE MODELS (No XGBoost)
# ============================================================================
class BaselineModels:
    def __init__(self, train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 target: str = 'GHI'):
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.train_series = train_data[target].copy()
        self.test_series = test_data[target].copy()
        self.n_test = len(test_data)
        self.cfg = CONFIG['baseline']
        self._detected_period = self._detect_seasonal_period()
        logger.info(f"Baseline auto-detected seasonal period: "
                    f"{self._detected_period}")

    def _detect_seasonal_period(self) -> int:
        try:
            max_lag = min(len(self.train_series) // 2, 200)
            if max_lag < 4:
                return 52
            acf_vals = acf(self.train_series.values, nlags=max_lag, fft=True)
            acf_no_zero = acf_vals[1:]
            peaks = []
            for i in range(1, len(acf_no_zero) - 1):
                if (acf_no_zero[i] > acf_no_zero[i - 1] and
                        acf_no_zero[i] > acf_no_zero[i + 1] and
                        acf_no_zero[i] > 0.05):
                    peaks.append((i + 1, acf_no_zero[i]))
            if peaks:
                best_lag = max(peaks, key=lambda x: x[1])[0]
                return max(best_lag, 2)
        except Exception as e:
            logger.warning(f"ACF seasonal detection failed: {e}")
        return 52

    @staticmethod
    def _metrics(y_true: np.ndarray,
                 y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        nonzero = np.abs(y_true) > 1e-8
        if nonzero.sum() > 0:
            mape = float(
                np.mean(np.abs((y_true[nonzero] - y_pred[nonzero])
                               / y_true[nonzero])) * 100)
        else:
            mape = float('nan')
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

    def persistence_forecast(self) -> Optional[Dict]:
        logger.info("▸ Persistence: optimising seasonal period + drift …")
        t0 = time.time()
        try:
            full_series = pd.concat([self.train_series, self.test_series])
            train_len = len(self.train_series)
            use_drift = self.cfg['persistence'].get('use_drift', True)
            if use_drift and len(self.train_series) > 1:
                diffs = np.diff(self.train_series.values)
                drift_per_step = float(np.mean(diffs))
            else:
                drift_per_step = 0.0
            candidates = sorted(set(
                self.cfg['persistence']['seasonal_periods']
                + [self._detected_period]))
            candidates = [m for m in candidates if m < train_len]
            if not candidates:
                candidates = [1]
            best_m, best_score = 1, float('inf')
            eval_len = min(self.n_test, train_len // 3, 52)
            for m in candidates:
                if m >= train_len:
                    continue
                holdout_start = train_len - eval_len
                if holdout_start - m < 0:
                    continue
                y_true_ho = self.train_series.values[holdout_start:]
                y_pred_ho = (
                    self.train_series.values[holdout_start - m:
                                             train_len - m]
                    + drift_per_step * m)
                if len(y_true_ho) != len(y_pred_ho):
                    continue
                score = float(mean_absolute_error(y_true_ho, y_pred_ho))
                if score < best_score:
                    best_score = score
                    best_m = m
            logger.info(f"  Best seasonal period m = {best_m}, "
                        f"drift/step = {drift_per_step:.6f}")
            predictions = np.empty(self.n_test)
            for i in range(self.n_test):
                ref_idx = train_len + i - best_m
                if ref_idx >= 0:
                    predictions[i] = (full_series.iloc[ref_idx]
                                      + drift_per_step * best_m)
                else:
                    predictions[i] = (self.train_series.iloc[-1]
                                      + drift_per_step)
            elapsed = time.time() - t0
            return {
                'predictions': predictions,
                'name': 'Persistence (seasonal+drift)',
                'best_params': {'m': best_m,
                                'drift_per_step': drift_per_step},
                'fit_time': elapsed,
            }
        except Exception as e:
            logger.error(f"Persistence forecast failed: {e}")
            return None

    def arima_forecast(self) -> Optional[Dict]:
        logger.info("▸ ARIMA: AIC grid search …")
        t0 = time.time()
        cfg = self.cfg['arima']
        try:
            adf_pvalue = adfuller(self.train_series.values,
                                  maxlag=min(20, len(self.train_series)
                                             // 3))[1]
            d_est = 0 if adf_pvalue < 0.05 else 1
            d_range = sorted(set(
                [max(0, d_est - 1), d_est,
                 min(d_est + 1, cfg['max_d'])]))
            p_range = range(0, cfg['max_p'] + 1)
            q_range = range(0, cfg['max_q'] + 1)
            best_aic = float('inf')
            best_order = (1, d_est, 1)
            ic_key = cfg['information_criterion']
            n_tried = 0
            combos = list(itertools.product(p_range, d_range, q_range))
            random.shuffle(combos)
            for p, d, q in tqdm(combos, desc="  ARIMA grid", leave=False):
                if p + d + q > cfg['max_order_sum']:
                    continue
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(self.train_series, order=(p, d, q))
                    fit = model.fit()
                    ic = getattr(fit, ic_key, fit.aic)
                    n_tried += 1
                    if ic < best_aic:
                        best_aic = ic
                        best_order = (p, d, q)
                except Exception:
                    continue
            logger.info(f"  Best ARIMA order = {best_order}  "
                        f"({ic_key.upper()} = {best_aic:.2f}, "
                        f"tried {n_tried} combos)")
            model = ARIMA(self.train_series, order=best_order)
            fit = model.fit()
            predictions = fit.forecast(steps=self.n_test).values
            elapsed = time.time() - t0
            return {
                'predictions': predictions,
                'name': f'ARIMA{best_order}',
                'best_params': {'order': best_order,
                                ic_key: best_aic,
                                'n_tried': n_tried},
                'fit_time': elapsed,
            }
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            return None

    def sarima_forecast(self) -> Optional[Dict]:
        logger.info("▸ SARIMA: AIC grid search …")
        t0 = time.time()
        cfg_s = self.cfg['sarima']
        cfg_a = self.cfg['arima']
        try:
            adf_pvalue = adfuller(self.train_series.values,
                                  maxlag=min(20, len(self.train_series)
                                             // 3))[1]
            d_est = 0 if adf_pvalue < 0.05 else 1
            seasonal_candidates = sorted(set(
                cfg_s['seasonal_periods'] + [self._detected_period]))
            seasonal_candidates = [
                m for m in seasonal_candidates
                if m >= 2 and 2 * m < len(self.train_series)]
            if not seasonal_candidates:
                seasonal_candidates = [
                    min(12, len(self.train_series) // 3)]
            ic_key = cfg_s['information_criterion']
            best_ic = float('inf')
            best_order = (1, d_est, 1)
            best_seasonal = (1, 0, 1, seasonal_candidates[0])
            n_tried = 0
            p_range = range(0, min(cfg_a['max_p'], 3) + 1)
            q_range = range(0, min(cfg_a['max_q'], 3) + 1)
            d_range = sorted(set([max(0, d_est - 1), d_est,
                                  min(d_est + 1, cfg_a['max_d'])]))
            P_range = range(0, cfg_s['max_P'] + 1)
            D_range = range(0, cfg_s['max_D'] + 1)
            Q_range = range(0, cfg_s['max_Q'] + 1)
            all_combos = list(itertools.product(
                p_range, d_range, q_range,
                P_range, D_range, Q_range,
                seasonal_candidates))
            random.shuffle(all_combos)
            max_combos = min(len(all_combos), 120)
            all_combos = all_combos[:max_combos]
            for p, d, q, P, D, Q, m in tqdm(all_combos,
                                              desc="  SARIMA grid",
                                              leave=False):
                if p + d + q > cfg_a['max_order_sum']:
                    continue
                if p == 0 and q == 0 and P == 0 and Q == 0:
                    continue
                try:
                    model = SARIMAX(
                        self.train_series,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, m),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
                    fit = model.fit(disp=False, maxiter=200)
                    ic = getattr(fit, ic_key, fit.aic)
                    n_tried += 1
                    if ic < best_ic:
                        best_ic = ic
                        best_order = (p, d, q)
                        best_seasonal = (P, D, Q, m)
                except Exception:
                    continue
            logger.info(
                f"  Best SARIMA({best_order})×({best_seasonal[:3]})"
                f"[{best_seasonal[3]}]  "
                f"({ic_key.upper()} = {best_ic:.2f}, tried {n_tried})")
            model = SARIMAX(
                self.train_series,
                order=best_order,
                seasonal_order=best_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=500)
            predictions = fit.forecast(steps=self.n_test).values
            elapsed = time.time() - t0
            return {
                'predictions': predictions,
                'name': (f'SARIMA({best_order})×'
                         f'({best_seasonal[:3]})[{best_seasonal[3]}]'),
                'best_params': {'order': best_order,
                                'seasonal_order': best_seasonal,
                                ic_key: best_ic,
                                'n_tried': n_tried},
                'fit_time': elapsed,
            }
        except Exception as e:
            logger.error(f"SARIMA forecast failed: {e}")
            return None

    def evaluate_all(self) -> Dict[str, Dict]:
        logger.info(f"\n{'═' * 55}")
        logger.info(" Evaluating OPTIMISED baseline models")
        logger.info(f"{'═' * 55}")
        actuals = self.test_series.values
        results = {}
        runners = [
            ('persistence', self.persistence_forecast),
            ('arima', self.arima_forecast),
            ('sarima', self.sarima_forecast),
        ]
        for key, method in runners:
            out = method()
            if out is None:
                logger.warning(f"  {key} — FAILED")
                results[key] = None
                continue
            preds = np.asarray(out['predictions']).flatten()
            min_len = min(len(actuals), len(preds))
            a = actuals[:min_len]
            p = preds[:min_len]
            metrics = self._metrics(a, p)
            entry = {
                'predictions': p,
                'name': out['name'],
                'best_params': out.get('best_params'),
                'fit_time': out.get('fit_time', 0),
                **metrics,
            }
            results[key] = entry
            logger.info(
                f"  {out['name']:40s}  MAE={metrics['MAE']:.4f}  "
                f"RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.4f}  "
                f"MAPE={metrics['MAPE']:.1f}%  "
                f"({out.get('fit_time', 0):.1f}s)")
        return results


# ============================================================================
# CMIP6 SCENARIO LOADER
# ============================================================================
class CMIP6ScenarioLoader:
    """Load and process CMIP6 scenario files (SSP2-4.5 and SSP5-8.5).

    Expected CSV format (monthly values):
        datetime, T2M, GHI, RH2M
        1/16/2020 12:00, 8.250671, 3438.898, 63.08517
        ...
    """

    def __init__(self, scenario_files: Dict[str, str],
                 historical_data: pd.DataFrame,
                 aggregation: str = 'weekly'):
        self.scenario_files = scenario_files
        self.historical_data = historical_data
        self.aggregation = aggregation
        self.scenario_data = {}
        self.bias_corrected = {}

    def load_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Load CMIP6 scenario CSV files."""
        for scenario, filepath in self.scenario_files.items():
            if not os.path.exists(filepath):
                logger.warning(
                    f"CMIP6 file not found: {filepath} — "
                    f"will generate synthetic for {scenario}")
                continue
            try:
                df = pd.read_csv(filepath)
                # Parse datetime
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.set_index('datetime').sort_index()
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                # Resample to match aggregation
                if self.aggregation == 'weekly':
                    df = df.resample('W').mean().dropna()
                elif self.aggregation == 'daily':
                    df = df.resample('D').mean().dropna()
                elif self.aggregation == 'monthly':
                    df = df.resample('M').mean().dropna()
                self.scenario_data[scenario] = df
                logger.info(
                    f"Loaded CMIP6 {scenario}: {len(df)} records, "
                    f"{df.index.min()} to {df.index.max()}")
            except Exception as e:
                logger.error(f"Failed to load {scenario}: {e}")
        return self.scenario_data

    def bias_correct_quantile_mapping(self) -> Dict[str, pd.DataFrame]:
        """Apply quantile mapping bias correction using historical overlap."""
        for scenario, sc_data in self.scenario_data.items():
            try:
                # Find overlap period
                overlap_start = max(self.historical_data.index.min(),
                                    sc_data.index.min())
                overlap_end = min(self.historical_data.index.max(),
                                  sc_data.index.max())
                hist_overlap = self.historical_data[
                    overlap_start:overlap_end]
                sc_overlap = sc_data[overlap_start:overlap_end]

                if len(hist_overlap) < 10 or len(sc_overlap) < 10:
                    logger.warning(
                        f"Insufficient overlap for {scenario} bias "
                        f"correction ({len(hist_overlap)} pts). "
                        f"Using raw data.")
                    self.bias_corrected[scenario] = sc_data.copy()
                    continue

                corrected = sc_data.copy()
                for col in sc_data.columns:
                    if col not in hist_overlap.columns:
                        continue
                    hist_vals = hist_overlap[col].dropna().values
                    sc_vals = sc_overlap[col].dropna().values
                    if len(hist_vals) < 5 or len(sc_vals) < 5:
                        continue
                    # Quantile mapping
                    n_quantiles = min(100, len(hist_vals), len(sc_vals))
                    quantiles = np.linspace(0, 100, n_quantiles)
                    hist_q = np.percentile(hist_vals, quantiles)
                    sc_q = np.percentile(sc_vals, quantiles)
                    # Apply correction to full scenario series
                    raw = corrected[col].values
                    corrected_vals = np.interp(
                        raw,
                        sc_q,
                        hist_q,
                        left=hist_q[0],
                        right=hist_q[-1])
                    corrected[col] = corrected_vals

                self.bias_corrected[scenario] = corrected
                logger.info(
                    f"Bias-corrected {scenario} using "
                    f"{len(hist_overlap)} overlap points")
            except Exception as e:
                logger.error(f"Bias correction failed for {scenario}: {e}")
                self.bias_corrected[scenario] = sc_data.copy()

        return self.bias_corrected

    def get_scenario_ghi_annual(
            self, scenario: str) -> Optional[pd.Series]:
        """Get annual mean GHI from a scenario."""
        data = self.bias_corrected.get(
            scenario, self.scenario_data.get(scenario))
        if data is None or 'GHI' not in data.columns:
            return None
        return data['GHI'].resample('Y').mean()

    def compute_closest_scenario(
            self, predictions: np.ndarray,
            pred_dates: pd.DatetimeIndex) -> str:
        """Find which scenario is closest to the model predictions."""
        best_scenario = None
        best_distance = float('inf')

        pred_series = pd.Series(predictions, index=pred_dates)
        pred_annual = pred_series.resample('Y').mean()

        for scenario in self.scenario_data:
            sc_ghi = self.get_scenario_ghi_annual(scenario)
            if sc_ghi is None:
                continue
            # Find overlapping years
            common_years = pred_annual.index.intersection(sc_ghi.index)
            if len(common_years) == 0:
                # Try matching by year value
                pred_years = pred_annual.index.year
                sc_years = sc_ghi.index.year
                common = set(pred_years) & set(sc_years)
                if not common:
                    continue
                p_vals = [pred_annual[pred_annual.index.year == y].values[0]
                          for y in sorted(common)
                          if len(pred_annual[pred_annual.index.year == y]) > 0]
                s_vals = [sc_ghi[sc_ghi.index.year == y].values[0]
                          for y in sorted(common)
                          if len(sc_ghi[sc_ghi.index.year == y]) > 0]
                min_len = min(len(p_vals), len(s_vals))
                if min_len == 0:
                    continue
                distance = np.mean(
                    np.abs(np.array(p_vals[:min_len])
                           - np.array(s_vals[:min_len])))
            else:
                p_vals = pred_annual.loc[common_years].values
                s_vals = sc_ghi.loc[common_years].values
                distance = np.mean(np.abs(p_vals - s_vals))

            if distance < best_distance:
                best_distance = distance
                best_scenario = scenario

        logger.info(f"Closest scenario: {best_scenario} "
                    f"(distance={best_distance:.4f})")
        return best_scenario


# ============================================================================
# DATA PROCESSOR
# ============================================================================
class SolarDataProcessor:
    def __init__(self, file_path, site_name=None):
        self.file_path = file_path
        self.site_name = (site_name
                          or os.path.basename(file_path).split('.')[0])
        self.scaler = StandardScaler()
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.aggregation = CONFIG['aggregation']

    def load_and_preprocess(self):
        logger.info(f"Loading data for site: {self.site_name}")
        try:
            if self.file_path.endswith('.xlsx'):
                df = pd.read_excel(self.file_path, sheet_name='Feuil1')
            elif self.file_path.endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                raise ValueError(
                    "Unsupported file format. Use .xlsx or .csv")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return None

        if 'YEAR' in df.columns:
            df['datetime'] = pd.to_datetime(df['YEAR'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()

        cols = [col for col in CONFIG['features'] + [CONFIG['target']]
                if col in df.columns]
        self.raw_data = df[cols].copy()

        logger.info(
            f"Data quality assessment for {self.site_name}:\n"
            f" • Total records: {len(self.raw_data):,}\n"
            f" • Missing values: "
            f"{self.raw_data.isnull().sum().sum()}\n"
            f" • Date range: {self.raw_data.index.min()} to "
            f"{self.raw_data.index.max()}")

        if self.aggregation == 'daily':
            self.processed_data = (
                self.raw_data.resample('D').mean().dropna())
        elif self.aggregation == 'weekly':
            self.processed_data = (
                self.raw_data.resample('W').mean().dropna())
        elif self.aggregation == 'hourly':
            self.processed_data = (
                self.raw_data.resample('H').mean().dropna())
        else:
            self.processed_data = self.raw_data.copy()

        if len(self.processed_data) > 100:
            train_end = '2019-12-31'
            val_end = '2021-12-31'
            self.train_data = self.processed_data[:train_end]
            self.val_data = self.processed_data[train_end:val_end]
            self.test_data = self.processed_data[val_end:]
        else:
            split_idx = int(len(self.processed_data) * 0.7)
            val_idx = int(len(self.processed_data) * 0.85)
            self.train_data = self.processed_data.iloc[:split_idx]
            self.val_data = self.processed_data.iloc[split_idx:val_idx]
            self.test_data = self.processed_data.iloc[val_idx:]

        logger.info(
            f"Data split for {self.site_name}:\n"
            f" • Training:   {len(self.train_data)} periods\n"
            f" • Validation: {len(self.val_data)} periods\n"
            f" • Test:       {len(self.test_data)} periods")
        return self.processed_data

    def create_sequences(self, data, model_type='informer',
                         is_train=True):
        if is_train:
            self.scaler.fit(data)

        features = CONFIG['features'].copy()
        if (not CONFIG['ablation']['with_humidity']
                and 'RH2M' in features):
            features.remove('RH2M')

        feature_cols = [data.columns.get_loc(col) for col in features]
        target_col = data.columns.get_loc(CONFIG['target'])
        data_scaled = self.scaler.transform(data)

        if model_type in ('informer', 'hybrid',
                          'hybrid_transformer_lstm'):
            return InformerDataset(
                data_scaled[:, feature_cols],
                data_scaled[:, [target_col]],
                CONFIG['seq_len'], CONFIG['label_len'],
                CONFIG['pred_len'])
        elif model_type == 'transformer':
            return TransformerDataset(
                data_scaled[:, feature_cols],
                data_scaled[:, [target_col]],
                CONFIG['seq_len'], CONFIG['pred_len'])
        else:
            return RNNDataset(data_scaled, CONFIG['seq_len'],
                              feature_cols, target_col)

    def generate_synthetic_future_data(self, years,
                                       method='trend_plus_seasonality'):
        if not years:
            return pd.DataFrame()
        logger.info(f"Generating synthetic data for years: {years}")

        if self.aggregation == 'daily':
            freq, periods_per_year, season_period = 'D', 366, 365
        elif self.aggregation == 'weekly':
            freq, periods_per_year, season_period = 'W', 53, 52
        elif self.aggregation == 'hourly':
            freq, periods_per_year, season_period = 'H', 8760, 24
        else:
            freq, periods_per_year, season_period = 'D', 366, 365

        synthetic_years = []
        for year in years:
            date_range = pd.date_range(
                start=f'{year}-01-01', periods=periods_per_year,
                freq=freq)
            col_series = {}
            for col in self.processed_data.columns:
                noise_scale = (
                    0.1 * np.std(self.processed_data[col])
                    if np.std(self.processed_data[col]) > 0 else 0.0)
                try:
                    decomposition = sm.tsa.seasonal_decompose(
                        self.processed_data[col], model='additive',
                        period=season_period)
                    trend = decomposition.trend.dropna()
                    if len(trend) > 1:
                        trend_model = np.polyfit(
                            np.arange(len(trend)), trend.values, 1)
                        trend_values = np.polyval(
                            trend_model,
                            np.arange(len(trend),
                                      len(trend) + periods_per_year))
                    else:
                        base = (trend.iloc[0] if len(trend) > 0
                                else float(
                                    self.processed_data[col].iloc[-1]))
                        trend_values = np.full(periods_per_year, base)
                    seasonality = decomposition.seasonal.dropna()
                    if len(seasonality) > 0:
                        season_vals = np.tile(
                            seasonality.values[-season_period:],
                            (periods_per_year // season_period + 1)
                        )[:periods_per_year]
                    else:
                        season_vals = np.zeros(periods_per_year)
                    noise = np.random.normal(
                        0, noise_scale, periods_per_year)
                    synth = trend_values + season_vals + noise
                except Exception:
                    last = (
                        self.processed_data[col]
                        .iloc[-periods_per_year:]
                        if len(self.processed_data) >= periods_per_year
                        else self.processed_data[col])
                    if len(last) < periods_per_year:
                        reps = (periods_per_year // len(last)) + 1
                        last = np.tile(
                            np.array(last), reps)[:periods_per_year]
                    noise = np.random.normal(
                        0,
                        0.05 * (np.std(self.processed_data[col])
                                if np.std(
                                    self.processed_data[col]) > 0
                                else 1.0),
                        periods_per_year)
                    synth = np.array(last) + noise
                if col in ['GHI', 'ALLSKY_KT']:
                    synth = np.maximum(synth, 0)
                col_series[col] = synth
            df_year = pd.DataFrame(col_series, index=date_range)
            synthetic_years.append(df_year)

        return pd.concat(synthetic_years).sort_index()


# ============================================================================
# UNCERTAINTY METRICS
# ============================================================================
class UncertaintyMetrics:
    @staticmethod
    def calculate_picp(y_true, lower_bound, upper_bound):
        return np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

    @staticmethod
    def calculate_pinaw(lower_bound, upper_bound, y_range):
        width = np.mean(upper_bound - lower_bound)
        return width / y_range if y_range > 0 else 0

    @staticmethod
    def calculate_crps(y_true, predictions_ensemble):
        if predictions_ensemble.ndim == 1:
            predictions_ensemble = predictions_ensemble.reshape(-1, 1)
        crps_values = []
        for i, y in enumerate(y_true):
            ensemble = predictions_ensemble[i, :]
            ensemble_sorted = np.sort(ensemble)
            n = len(ensemble_sorted)
            if np.std(ensemble) < 1e-6:
                crps_values.append(abs(y - np.mean(ensemble)))
                continue
            crps = 0
            for j, pred in enumerate(ensemble_sorted):
                crps += abs(pred - y) * (2 * (j + 1) - 1)
            crps = crps / (n ** 2)
            for j in range(n):
                for k in range(j + 1, n):
                    crps -= abs(ensemble_sorted[j]
                                - ensemble_sorted[k]) / (n ** 2)
            crps_values.append(max(0, crps))
        return np.mean(crps_values) if crps_values else 0

    @staticmethod
    def calculate_interval_score(y_true, lower_bound, upper_bound,
                                 alpha=0.05):
        width = upper_bound - lower_bound
        lower_penalty = (2 / alpha
                         * np.maximum(0, lower_bound - y_true))
        upper_penalty = (2 / alpha
                         * np.maximum(0, y_true - upper_bound))
        return np.mean(width + lower_penalty + upper_penalty)

    @staticmethod
    def calculate_pit(y_true, predictions_ensemble):
        pit_values = []
        for i, y in enumerate(y_true):
            ensemble = predictions_ensemble[i, :]
            pit = np.mean(ensemble <= y + 1e-10)
            pit_values.append(np.clip(pit, 0.01, 0.99))
        return np.array(pit_values)

    @staticmethod
    def calculate_coverage_vs_nominal(y_true, predictions_ensemble,
                                      confidence_levels):
        results = {}
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            lower = np.percentile(predictions_ensemble,
                                  alpha * 100, axis=1)
            upper = np.percentile(predictions_ensemble,
                                  (1 - alpha) * 100, axis=1)
            empirical = UncertaintyMetrics.calculate_picp(
                y_true, lower, upper)
            results[conf_level] = {
                'nominal': conf_level,
                'empirical': empirical,
                'difference': abs(empirical - conf_level)
            }
        return results

    @staticmethod
    def time_varying_stability(y_true, predictions_ensemble,
                               window_size=30):
        if len(y_true) < window_size:
            return {}
        n_windows = len(y_true) // window_size
        sm_ = {'picp_by_window': [], 'pinaw_by_window': [],
               'window_indices': []}
        for i in range(n_windows):
            s = i * window_size
            e = (i + 1) * window_size
            wy = y_true[s:e]
            wp = predictions_ensemble[s:e, :]
            lo = np.percentile(wp, 5, axis=1)
            hi = np.percentile(wp, 95, axis=1)
            yr = np.max(wy) - np.min(wy)
            sm_['picp_by_window'].append(
                UncertaintyMetrics.calculate_picp(wy, lo, hi))
            sm_['pinaw_by_window'].append(
                UncertaintyMetrics.calculate_pinaw(lo, hi, yr))
            sm_['window_indices'].append(i)
        sm_['picp_std'] = np.std(sm_['picp_by_window'])
        sm_['pinaw_std'] = np.std(sm_['pinaw_by_window'])
        return sm_

    @staticmethod
    def reliability_diagram(y_true, predictions_ensemble, n_bins=10):
        predicted_probs = []
        observed_freqs = []
        bin_edges = np.linspace(0, 1, n_bins + 1)
        pq = np.mean(
            predictions_ensemble <= y_true[:, np.newaxis], axis=1)
        for i in range(n_bins):
            bl = bin_edges[i]
            bu = bin_edges[i + 1]
            bc = (bl + bu) / 2
            ib = (pq >= bl) & (pq < bu)
            if np.sum(ib) > 0:
                predicted_probs.append(bc)
                observed_freqs.append(np.mean(pq[ib]))
        return {'predicted_probs': np.array(predicted_probs),
                'observed_freqs': np.array(observed_freqs)}

    @staticmethod
    def pit_histogram(y_true, predictions_ensemble, n_bins=20):
        pit = UncertaintyMetrics.calculate_pit(
            y_true, predictions_ensemble)
        hist, edges = np.histogram(
            pit, bins=n_bins, range=(0, 1), density=True)
        return {'pit_values': pit, 'histogram': hist,
                'bin_edges': edges}

    @staticmethod
    def statistical_tests(y_true, predictions_ensemble,
                          predictions_mean):
        pit = UncertaintyMetrics.calculate_pit(
            y_true, predictions_ensemble)
        ks_stat, ks_p = stats.kstest(pit, stats.uniform.cdf)
        resid = y_true - predictions_mean
        if len(resid) > 5000:
            resid = np.random.choice(resid, 5000, replace=False)
        sw_stat, sw_p = stats.shapiro(resid)
        return {
            'ks_test': {'stat': ks_stat, 'pvalue': ks_p,
                        'is_uniform': ks_p > 0.05},
            'shapiro_test': {'stat': sw_stat, 'pvalue': sw_p,
                             'is_normal': sw_p > 0.05},
        }


# ============================================================================
# VISUALIZATION CLASS
# ============================================================================
class SolarForecastVisualizer:
    def __init__(self):
        sns.set_palette("husl")
        os.makedirs(CONFIG['output_dir'], exist_ok=True)

    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        output_path = os.path.join(CONFIG['output_dir'], filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Saved plot to {output_path}")
        plt.close()

    def _save(self, filename):
        self.save_plot(filename)

    def plot_training_history(self, train_losses, val_losses,
                              model_type, save=True):
        set_journal_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        ep = range(1, len(train_losses) + 1)
        ax.plot(ep, train_losses, color=CB_PALETTE['blue'], lw=1.8,
                label='Training loss')
        ax.plot(ep, val_losses, color=CB_PALETTE['vermilion'], lw=1.8,
                linestyle='--', label='Validation loss')
        gap = abs(float(train_losses[-1]) - float(val_losses[-1]))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'Training History – {model_type.upper()}')
        ax.legend(framealpha=0.9)
        ax.annotate(
            f'Final gap = {gap:.3f}',
            xy=(len(train_losses), float(val_losses[-1])),
            xytext=(max(1, len(train_losses) * 0.6),
                    max(val_losses) * 0.85),
            arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
            fontsize=8, color='grey')
        if save:
            self._save(f'fig_training_history_{model_type}.pdf')

    def plot_predictions(self, actuals, predictions, dates,
                         model_type, save=True):
        set_journal_style()
        min_len = min(len(actuals), len(predictions), len(dates))
        actuals = np.asarray(actuals)[:min_len]
        predictions = np.asarray(predictions)[:min_len]
        dates = dates[:min_len]
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 6),
            gridspec_kw={'height_ratios': [3, 1]})
        ax, ax_r = axes
        ax.plot(dates, actuals, color=CB_PALETTE['blue'], lw=1.2,
                alpha=0.85, label='Observed GHI')
        ax.plot(dates, predictions, color=CB_PALETTE['vermilion'],
                lw=1.2, alpha=0.85, linestyle='--',
                label='Predicted GHI')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.legend(framealpha=0.9)
        ax.set_title(f'Observed vs Predicted – {model_type.upper()}')
        resid = actuals - predictions
        ax_r.axhline(0, color='grey', lw=0.8, linestyle='--')
        ax_r.fill_between(dates, resid, 0, where=(resid >= 0),
                          color=CB_PALETTE['sky_blue'], alpha=0.55,
                          label='+')
        ax_r.fill_between(dates, resid, 0, where=(resid < 0),
                          color=CB_PALETTE['orange'], alpha=0.55,
                          label='−')
        ax_r.set_xlabel('Date')
        ax_r.set_ylabel('Residual')
        ax_r.legend(fontsize=8, framealpha=0.9)
        if save:
            self._save(f'fig_predictions_{model_type}.pdf')

    def plot_uncertainty_intervals(self, actuals, mc_results, dates,
                                   model_type, save=True):
        samples = mc_results.get('samples', None)
        if samples is not None:
            ml = min(samples.shape[0], len(dates), len(actuals))
            mc_results = mc_results.copy()
            mc_results['samples'] = samples[:ml, :]
            actuals = np.asarray(actuals)[:ml]
            dates = dates[:ml]
        self.plot_fan_chart(actuals, mc_results, dates, model_type,
                            save=save)

    def plot_fan_chart(self, actuals, mc_results, dates, model_type,
                       save=True):
        set_journal_style()
        samples = mc_results.get('samples', None)
        if samples is None:
            return
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        n = min(len(dates), len(actuals), samples.shape[0])
        dates = dates[:n]
        actuals = np.array(actuals)[:n]
        samples = samples[:n, :]
        p10 = np.percentile(samples, 10, axis=1)
        p25 = np.percentile(samples, 25, axis=1)
        p50 = np.percentile(samples, 50, axis=1)
        p75 = np.percentile(samples, 75, axis=1)
        p90 = np.percentile(samples, 90, axis=1)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(dates, p10, p90,
                        color=CB_PALETTE['sky_blue'], alpha=0.22,
                        label='P10–P90')
        ax.fill_between(dates, p25, p75,
                        color=CB_PALETTE['sky_blue'], alpha=0.45,
                        label='P25–P75')
        ax.plot(dates, p50, color=CB_PALETTE['blue'], lw=2.2,
                label='P50', zorder=4)
        ax.plot(dates, actuals, color=CB_PALETTE['black'], lw=1.0,
                alpha=0.75, label='Observed', zorder=5)
        ax.set_xlabel('Date')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.set_title(f'Fan Chart – {model_type.upper()}')
        ax.legend(framealpha=0.9, loc='upper left')
        if save:
            self._save(f'fig_fan_chart_{model_type}.pdf')

    # ── Long-term prediction visualization (WITHOUT scenarios) ──────
    def plot_longterm_prediction(self, hist_dates, hist_ghi,
                                 future_dates, future_samples,
                                 model_type, save=True):
        """Plot long-term prediction for a single model without scenarios."""
        set_journal_style()
        p05 = np.percentile(future_samples, 5, axis=0)
        p10 = np.percentile(future_samples, 10, axis=0)
        p25 = np.percentile(future_samples, 25, axis=0)
        p50 = np.percentile(future_samples, 50, axis=0)
        p75 = np.percentile(future_samples, 75, axis=0)
        p90 = np.percentile(future_samples, 90, axis=0)
        p95 = np.percentile(future_samples, 95, axis=0)

        fig, ax = plt.subplots(figsize=(14, 5.5))

        # Historical
        ax.plot(hist_dates, hist_ghi, color=CB_PALETTE['black'],
                lw=1.2, alpha=0.85, label='Historical', zorder=5)

        # Future uncertainty bands
        ax.fill_between(future_dates, p05, p95,
                        color=CB_PALETTE['sky_blue'], alpha=0.10,
                        label='P5–P95')
        ax.fill_between(future_dates, p10, p90,
                        color=CB_PALETTE['sky_blue'], alpha=0.20,
                        label='P10–P90')
        ax.fill_between(future_dates, p25, p75,
                        color=CB_PALETTE['sky_blue'], alpha=0.38,
                        label='P25–P75')
        ax.plot(future_dates, p50, color=CB_PALETTE['blue'], lw=2.5,
                label='P50 (Median Forecast)', zorder=4)

        # Vertical line at transition
        ax.axvline(hist_dates[-1], color='grey', lw=1.0,
                   linestyle=':', alpha=0.7)
        ax.text(hist_dates[-1], ax.get_ylim()[1] * 0.95,
                ' Forecast →', fontsize=9, color='grey',
                va='top')

        ax.set_xlabel('Year')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.set_title(
            f'Long-Term Forecast – {model_type.upper()} '
            f'(Without Scenarios)')
        ax.legend(framealpha=0.9, loc='upper left', ncol=2, fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        if save:
            self._save(f'fig_longterm_no_scenario_{model_type}.pdf')

    # ── Long-term prediction visualization (WITH scenarios) ─────────
    def plot_longterm_with_scenarios(
            self, hist_dates, hist_ghi,
            future_dates, future_samples,
            scenario_data: Dict[str, pd.DataFrame],
            closest_scenario: str,
            model_type, save=True):
        """Plot long-term prediction overlaid with CMIP6 scenarios."""
        set_journal_style()
        p10 = np.percentile(future_samples, 10, axis=0)
        p25 = np.percentile(future_samples, 25, axis=0)
        p50 = np.percentile(future_samples, 50, axis=0)
        p75 = np.percentile(future_samples, 75, axis=0)
        p90 = np.percentile(future_samples, 90, axis=0)

        fig, ax = plt.subplots(figsize=(16, 6.5))

        # Historical
        ax.plot(hist_dates, hist_ghi, color=CB_PALETTE['black'],
                lw=1.4, alpha=0.85, label='Historical GHI', zorder=5)

        # Model prediction band
        ax.fill_between(future_dates, p10, p90,
                        color=CB_PALETTE['sky_blue'], alpha=0.15,
                        label='Model P10–P90')
        ax.fill_between(future_dates, p25, p75,
                        color=CB_PALETTE['sky_blue'], alpha=0.35,
                        label='Model P25–P75')
        ax.plot(future_dates, p50, color=CB_PALETTE['blue'], lw=2.5,
                label='Model P50', zorder=4)

        # Scenario curves
        scenario_colors = {
            'ssp245': CB_PALETTE['green'],
            'ssp585': CB_PALETTE['vermilion'],
        }
        scenario_labels = {
            'ssp245': 'CMIP6 SSP2-4.5',
            'ssp585': 'CMIP6 SSP5-8.5',
        }
        for scenario, sc_df in scenario_data.items():
            if 'GHI' not in sc_df.columns:
                continue
            color = scenario_colors.get(scenario, CB_PALETTE['orange'])
            label = scenario_labels.get(scenario, scenario.upper())
            lw = 2.8 if scenario == closest_scenario else 1.5
            ls = '-' if scenario == closest_scenario else '--'
            alpha = 1.0 if scenario == closest_scenario else 0.7

            # Resample to match prediction frequency
            sc_ghi = sc_df['GHI']
            # Filter to future period
            future_start = future_dates[0]
            future_end = future_dates[-1]
            sc_future = sc_ghi[
                (sc_ghi.index >= future_start) &
                (sc_ghi.index <= future_end)]

            if len(sc_future) > 0:
                ax.plot(sc_future.index, sc_future.values,
                        color=color, lw=lw, linestyle=ls,
                        alpha=alpha, label=label, zorder=3)

            # Mark closest
            if scenario == closest_scenario:
                # Add star marker at midpoint
                mid_idx = len(sc_future) // 2
                if mid_idx < len(sc_future):
                    ax.scatter(
                        [sc_future.index[mid_idx]],
                        [sc_future.values[mid_idx]],
                        marker='*', s=200, color=color,
                        edgecolors='black', linewidths=0.8,
                        zorder=6,
                        label=f'Closest: {label}')

        # Vertical line
        ax.axvline(hist_dates[-1], color='grey', lw=1.0,
                   linestyle=':', alpha=0.7)
        ax.text(hist_dates[-1], ax.get_ylim()[1] * 0.95,
                ' Forecast →', fontsize=9, color='grey', va='top')

        ax.set_xlabel('Year')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.set_title(
            f'Long-Term Forecast with CMIP6 Scenarios – '
            f'{model_type.upper()}\n'
            f'(Closest scenario: '
            f'{scenario_labels.get(closest_scenario, closest_scenario)})')
        ax.legend(framealpha=0.9, loc='upper left', ncol=2, fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        if save:
            self._save(
                f'fig_longterm_with_scenarios_{model_type}.pdf')

    # ── All models comparison (long-term, without scenarios) ────────
    def plot_all_models_longterm(
            self, hist_dates, hist_ghi,
            model_forecasts: Dict[str, Dict],
            save=True):
        """Compare long-term predictions from all models on one plot."""
        set_journal_style()
        fig, ax = plt.subplots(figsize=(16, 6))

        # Historical
        ax.plot(hist_dates, hist_ghi, color=CB_PALETTE['black'],
                lw=1.5, alpha=0.85, label='Historical', zorder=5)

        model_colors = {
            'informer': CB_PALETTE['blue'],
            'transformer': CB_PALETTE['sky_blue'],
            'lstm': CB_PALETTE['orange'],
            'gru': CB_PALETTE['green'],
            'hybrid': CB_PALETTE['pink'],
            'hybrid_transformer_lstm': CB_PALETTE['vermilion'],
        }

        for model_name, forecast_info in model_forecasts.items():
            future_dates = forecast_info['dates']
            samples = forecast_info['samples']
            p50 = np.percentile(samples, 50, axis=0)
            p25 = np.percentile(samples, 25, axis=0)
            p75 = np.percentile(samples, 75, axis=0)
            color = model_colors.get(model_name, CB_PALETTE['blue'])

            ax.fill_between(future_dates, p25, p75,
                            color=color, alpha=0.12)
            ax.plot(future_dates, p50, color=color, lw=1.8,
                    label=f'{model_name.upper()} P50', zorder=4)

        ax.axvline(hist_dates[-1], color='grey', lw=1.0,
                   linestyle=':', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.set_title('Long-Term Forecast Comparison – All Models')
        ax.legend(framealpha=0.9, loc='upper left', ncol=2, fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        if save:
            self._save('fig_longterm_all_models_comparison.pdf')

    # ── All models comparison (long-term, WITH scenarios) ───────────
    def plot_all_models_longterm_with_scenarios(
            self, hist_dates, hist_ghi,
            model_forecasts: Dict[str, Dict],
            scenario_data: Dict[str, pd.DataFrame],
            best_model_name: str,
            closest_scenario: str,
            save=True):
        """Compare all models + overlay CMIP6 scenarios. Highlight best."""
        set_journal_style()
        fig, ax = plt.subplots(figsize=(18, 7))

        ax.plot(hist_dates, hist_ghi, color=CB_PALETTE['black'],
                lw=1.5, alpha=0.85, label='Historical', zorder=5)

        model_colors = {
            'informer': '#0072B2',
            'transformer': '#56B4E9',
            'lstm': '#E69F00',
            'gru': '#009E73',
            'hybrid': '#CC79A7',
            'hybrid_transformer_lstm': '#D55E00',
        }

        for model_name, forecast_info in model_forecasts.items():
            future_dates = forecast_info['dates']
            samples = forecast_info['samples']
            p50 = np.percentile(samples, 50, axis=0)
            color = model_colors.get(model_name, '#888888')
            lw = 2.8 if model_name == best_model_name else 1.2
            alpha = 1.0 if model_name == best_model_name else 0.5
            ls = '-' if model_name == best_model_name else '--'

            ax.plot(future_dates, p50, color=color, lw=lw,
                    alpha=alpha, linestyle=ls,
                    label=f'{model_name.upper()}', zorder=4)

            if model_name == best_model_name:
                p25 = np.percentile(samples, 25, axis=0)
                p75 = np.percentile(samples, 75, axis=0)
                ax.fill_between(future_dates, p25, p75,
                                color=color, alpha=0.15,
                                label=f'{model_name.upper()} P25-P75')

        # Scenario overlays
        scenario_styles = {
            'ssp245': {'color': '#228B22', 'label': 'SSP2-4.5'},
            'ssp585': {'color': '#DC143C', 'label': 'SSP5-8.5'},
        }
        for scenario, sc_df in scenario_data.items():
            if 'GHI' not in sc_df.columns:
                continue
            style = scenario_styles.get(
                scenario, {'color': '#888', 'label': scenario})
            sc_ghi = sc_df['GHI']
            lw = 2.5 if scenario == closest_scenario else 1.5
            ls = '-' if scenario == closest_scenario else ':'
            ax.plot(sc_ghi.index, sc_ghi.values,
                    color=style['color'], lw=lw, linestyle=ls,
                    alpha=0.8,
                    label=f"CMIP6 {style['label']}", zorder=3)

        ax.axvline(hist_dates[-1], color='grey', lw=1.0,
                   linestyle=':', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.set_title(
            f'Long-Term Forecast – All Models + CMIP6 Scenarios\n'
            f'Best model: {best_model_name.upper()}, '
            f'Closest scenario: '
            f'{scenario_styles.get(closest_scenario, {}).get("label", closest_scenario)}')
        ax.legend(framealpha=0.9, loc='upper left', ncol=3, fontsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        if save:
            self._save(
                'fig_longterm_all_models_with_scenarios.pdf')

    # ── Best model + scenarios detail plot ──────────────────────────
    def plot_best_model_with_scenarios(
            self, hist_dates, hist_ghi,
            future_dates, future_samples,
            scenario_data: Dict[str, pd.DataFrame],
            closest_scenario: str,
            model_type: str, save=True):
        """Detailed plot: best model prediction curve + both scenarios."""
        set_journal_style()
        p05 = np.percentile(future_samples, 5, axis=0)
        p10 = np.percentile(future_samples, 10, axis=0)
        p25 = np.percentile(future_samples, 25, axis=0)
        p50 = np.percentile(future_samples, 50, axis=0)
        p75 = np.percentile(future_samples, 75, axis=0)
        p90 = np.percentile(future_samples, 90, axis=0)
        p95 = np.percentile(future_samples, 95, axis=0)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})
        ax_main, ax_diff = axes

        # Main panel
        ax_main.plot(hist_dates, hist_ghi, color=CB_PALETTE['black'],
                     lw=1.3, alpha=0.85, label='Historical', zorder=5)

        ax_main.fill_between(future_dates, p05, p95,
                             color=CB_PALETTE['sky_blue'], alpha=0.08,
                             label='P5–P95')
        ax_main.fill_between(future_dates, p10, p90,
                             color=CB_PALETTE['sky_blue'], alpha=0.18,
                             label='P10–P90')
        ax_main.fill_between(future_dates, p25, p75,
                             color=CB_PALETTE['sky_blue'], alpha=0.35,
                             label='P25–P75')
        ax_main.plot(future_dates, p50, color=CB_PALETTE['blue'],
                     lw=2.5, label=f'{model_type.upper()} P50',
                     zorder=4)

        scenario_colors = {
            'ssp245': '#228B22', 'ssp585': '#DC143C'}
        scenario_labels = {
            'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}

        for scenario, sc_df in scenario_data.items():
            if 'GHI' not in sc_df.columns:
                continue
            sc_ghi = sc_df['GHI']
            future_mask = ((sc_ghi.index >= future_dates[0]) &
                           (sc_ghi.index <= future_dates[-1]))
            sc_future = sc_ghi[future_mask]
            if len(sc_future) == 0:
                continue
            color = scenario_colors.get(scenario, '#888')
            label = scenario_labels.get(scenario, scenario)
            lw = 2.5 if scenario == closest_scenario else 1.5
            ls = '-' if scenario == closest_scenario else '--'
            ax_main.plot(sc_future.index, sc_future.values,
                         color=color, lw=lw, linestyle=ls,
                         label=f'CMIP6 {label}', zorder=3)

        ax_main.axvline(hist_dates[-1], color='grey', lw=1.0,
                        linestyle=':', alpha=0.7)
        ax_main.set_ylabel('GHI (kWh m⁻²)')
        ax_main.set_title(
            f'Best Model ({model_type.upper()}) + CMIP6 Scenarios\n'
            f'Closest scenario: '
            f'{scenario_labels.get(closest_scenario, closest_scenario)}')
        ax_main.legend(framealpha=0.9, loc='upper left', ncol=2,
                       fontsize=8)

        # Difference panel: model P50 vs scenarios
        for scenario, sc_df in scenario_data.items():
            if 'GHI' not in sc_df.columns:
                continue
            sc_ghi = sc_df['GHI']
            # Resample scenario to match future_dates
            sc_resampled = sc_ghi.reindex(future_dates, method='nearest')
            if len(sc_resampled.dropna()) == 0:
                continue
            diff = p50 - sc_resampled.values
            color = scenario_colors.get(scenario, '#888')
            label = scenario_labels.get(scenario, scenario)
            ax_diff.plot(future_dates, diff, color=color, lw=1.5,
                         label=f'Model − {label}')
            ax_diff.fill_between(future_dates, 0, diff,
                                 color=color, alpha=0.15)

        ax_diff.axhline(0, color='grey', lw=1.0, linestyle='--')
        ax_diff.set_xlabel('Year')
        ax_diff.set_ylabel('Difference (kWh m⁻²)')
        ax_diff.set_title('Model P50 − Scenario GHI')
        ax_diff.legend(framealpha=0.9, fontsize=8)
        ax_diff.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_diff.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            self._save(
                f'fig_best_model_scenarios_detail_{model_type}.pdf')

    def plot_longterm_fan_chart(self, hist_dates, hist_ghi,
                                future_dates, future_samples,
                                model_type, save=True):
        set_journal_style()
        p01 = np.percentile(future_samples, 1, axis=0)
        p10 = np.percentile(future_samples, 10, axis=0)
        p25 = np.percentile(future_samples, 25, axis=0)
        p50 = np.percentile(future_samples, 50, axis=0)
        p75 = np.percentile(future_samples, 75, axis=0)
        p90 = np.percentile(future_samples, 90, axis=0)
        p99 = np.percentile(future_samples, 99, axis=0)
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(hist_dates, hist_ghi, color=CB_PALETTE['black'],
                lw=1.2, alpha=0.85, label='Historical', zorder=5)
        ax.fill_between(future_dates, p01, p99,
                        color=CB_PALETTE['sky_blue'], alpha=0.10,
                        label='P1–P99')
        ax.fill_between(future_dates, p10, p90,
                        color=CB_PALETTE['sky_blue'], alpha=0.20,
                        label='P10–P90')
        ax.fill_between(future_dates, p25, p75,
                        color=CB_PALETTE['sky_blue'], alpha=0.38,
                        label='P25–P75')
        ax.plot(future_dates, p50, color=CB_PALETTE['blue'], lw=2.5,
                label='P50', zorder=4)
        ax.set_xlabel('Year')
        ax.set_ylabel('GHI (kWh m⁻²)')
        ax.set_title(f'Long-Term Forecast – {model_type.upper()}')
        ax.legend(framealpha=0.9, loc='upper left', ncol=2, fontsize=9)
        if save:
            self._save(f'fig_longterm_fan_{model_type}.pdf')

    def plot_model_comparison(self, model_results_dict, save=True):
        set_journal_style()
        models = [m for m in model_results_dict
                  if isinstance(model_results_dict[m], dict)
                  and 'deterministic' in model_results_dict[m]]
        if not models:
            return
        det = {m: model_results_dict[m]['deterministic']
               for m in models}
        mae = [float(det[m].get('MAE', 0)) for m in models]
        rmse = [float(det[m].get('RMSE', 0)) for m in models]
        r2 = [float(det[m].get('R2', 0)) for m in models]
        mae_e = [float(det[m].get('MAE_CI', mae[i] * 0.05))
                 for i, m in enumerate(models)]
        rmse_e = [float(det[m].get('RMSE_CI', rmse[i] * 0.05))
                  for i, m in enumerate(models)]
        palette = [
            CB_PALETTE['blue'] if 'inform' in m else
            CB_PALETTE['orange'] if 'lstm' in m else
            CB_PALETTE['green'] if 'gru' in m else
            CB_PALETTE['pink']
            if 'hybrid_transformer_lstm' in m else
            CB_PALETTE['sky_blue']
            for m in models]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        x = np.arange(len(models))
        w = 0.55
        for ax, vals, ci, ylabel, title in zip(
                axes, [mae, rmse, r2],
                [mae_e, rmse_e, [0] * len(models)],
                ['MAE', 'RMSE', 'R²'],
                ['MAE ↓', 'RMSE ↓', 'R² ↑']):
            bars = ax.bar(x, vals, width=w, color=palette,
                          edgecolor='white', linewidth=0.5)
            ax.errorbar(x, vals, yerr=ci, fmt='none',
                        ecolor='black', elinewidth=1.2, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha='right',
                               fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        fig.suptitle('Model Performance Comparison', fontsize=12,
                     fontweight='bold')
        if save:
            self._save('fig_model_comparison.pdf')

    def plot_baseline_comparison(self, baseline_results: Dict,
                                 save=True):
        set_journal_style()
        names, maes, rmses, r2s = [], [], [], []
        for key, res in baseline_results.items():
            if res is None:
                continue
            names.append(res.get('name', key))
            maes.append(res['MAE'])
            rmses.append(res['RMSE'])
            r2s.append(res['R2'])
        if not names:
            return
        palette = [CB_PALETTE['blue'], CB_PALETTE['orange'],
                   CB_PALETTE['green']][:len(names)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = np.arange(len(names))
        w = 0.55
        for ax, vals, ylabel, title in zip(
                axes, [maes, rmses, r2s],
                ['MAE', 'RMSE', 'R²'],
                ['MAE ↓', 'RMSE ↓', 'R² ↑']):
            bars = ax.bar(x, vals, width=w, color=palette,
                          edgecolor='white', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=25, ha='right',
                               fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        fig.suptitle('Optimised Baseline Comparison', fontsize=12,
                     fontweight='bold')
        if save:
            self._save('fig_baseline_comparison.pdf')

    def plot_coverage_vs_nominal(self, coverage_results, model_type,
                                 save=True):
        set_journal_style()
        if not coverage_results:
            return
        nominals = sorted(
            [k for k in coverage_results
             if isinstance(k, (int, float))])
        if not nominals:
            return
        empiricals = [coverage_results[k]['empirical']
                      for k in nominals]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2,
                label='Perfect calibration')
        ax.scatter(nominals, empiricals,
                   color=CB_PALETTE['vermilion'], s=65, zorder=5,
                   label='Model')
        for n, e in zip(nominals, empiricals):
            ax.plot([n, n], [n, e], color='grey', lw=0.7,
                    linestyle=':')
        ax.set_xlabel('Nominal confidence level')
        ax.set_ylabel('Empirical coverage')
        ax.set_title(
            f'Coverage vs Nominal – {model_type.upper()}')
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(0.4, 1.05)
        ax.legend(framealpha=0.9)
        ax.set_aspect('equal')
        if save:
            self._save(f'fig_coverage_{model_type}.pdf')

    def plot_time_varying_stability(self, stability_metrics,
                                    model_type, save=True):
        if (not stability_metrics
                or 'picp_by_window' not in stability_metrics):
            return
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        wi = stability_metrics['window_indices']
        axes[0].plot(wi, stability_metrics['picp_by_window'], 'o-',
                     color='blue')
        axes[0].axhline(y=0.90, color='r', linestyle='--',
                        label='Target 90%')
        axes[0].set_ylabel('PICP')
        axes[0].set_title(
            f'Time-Varying PICP – {model_type.upper()}')
        axes[0].legend()
        axes[1].plot(wi, stability_metrics['pinaw_by_window'], 'o-',
                     color='green')
        axes[1].set_xlabel('Window Index')
        axes[1].set_ylabel('PINAW')
        axes[1].set_title(
            f'Time-Varying PINAW – {model_type.upper()}')
        plt.suptitle(
            f'Time-Varying Stability – {model_type.upper()}',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            self.save_plot(
                f'time_varying_stability_{model_type}.png')

    def plot_reliability_diagram(self, reliability_data, model_type,
                                 save=True):
        set_journal_style()
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2,
                label='Perfect calibration')
        ax.plot(reliability_data['predicted_probs'],
                reliability_data['observed_freqs'],
                'o-', color=CB_PALETTE['pink'], lw=1.8, ms=6,
                label='Model')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.set_title(
            f'Reliability Diagram – {model_type.upper()}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(framealpha=0.9)
        ax.set_aspect('equal')
        if save:
            self._save(f'fig_reliability_{model_type}.pdf')

    def plot_pit_histogram(self, pit_data, model_type, save=True):
        set_journal_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        widths = np.diff(pit_data['bin_edges'])
        ax.bar(pit_data['bin_edges'][:-1], pit_data['histogram'],
               width=widths, align='edge',
               color=CB_PALETTE['pink'], edgecolor='white',
               alpha=0.80, label='Model PIT')
        ax.axhline(1.0, color=CB_PALETTE['black'], lw=1.2,
                   linestyle='--', label='Uniform (ideal)')
        ax.set_xlabel('PIT value')
        ax.set_ylabel('Density')
        ax.set_title(f'PIT Histogram – {model_type.upper()}')
        ax.legend(framealpha=0.9)
        ax.set_xlim(0, 1)
        if save:
            self._save(f'fig_pit_{model_type}.pdf')

    def plot_pipeline_schematic(self, save=True):
        set_journal_style()
        fig, ax = plt.subplots(figsize=(14, 3.8))
        ax.axis('off')
        boxes = [
            ('Raw Climate\nData\n(Kt, T, RH)', 0.06),
            ('EQM\nBias\nCorrection', 0.20),
            ('FastICA\nDecomp.\n(IC1–IC3)', 0.34),
            ('LSTM\nEncoder\n(local)', 0.48),
            ('Transformer\nForecaster\n(Attention)', 0.62),
            ('Monte Carlo\nSimulation\n(S=10 000)', 0.76),
            ('P10/P50/P90\nRisk\nEnvelope', 0.90),
        ]
        box_colors = [
            CB_PALETTE['sky_blue'], CB_PALETTE['green'],
            CB_PALETTE['orange'], CB_PALETTE['blue'],
            CB_PALETTE['blue'], CB_PALETTE['pink'],
            CB_PALETTE['vermilion']]
        bh, bw = 0.60, 0.11
        for (label, xc), bc in zip(boxes, box_colors):
            rect = mpatches.FancyBboxPatch(
                (xc - bw / 2, 0.15), bw, bh,
                boxstyle='round,pad=0.02', linewidth=1.2,
                edgecolor='white', facecolor=bc, alpha=0.85,
                transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(xc, 0.15 + bh / 2, label,
                    transform=ax.transAxes, ha='center',
                    va='center', fontsize=8.5, fontweight='bold',
                    color='white', multialignment='center')
        for i in range(len(boxes) - 1):
            x0 = boxes[i][1] + bw / 2 + 0.005
            x1 = boxes[i + 1][1] - bw / 2 - 0.005
            ax.annotate(
                '', xy=(x1, 0.15 + bh / 2),
                xytext=(x0, 0.15 + bh / 2),
                xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.6))
        ax.text(0.50, 0.93,
                'Hybrid LSTM–Transformer–Monte Carlo Pipeline',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold')
        if save:
            self._save('fig_pipeline_schematic.pdf')

    def plot_ghi_distribution(self, ghi_values, save=True):
        set_journal_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ghi_values, bins=30, density=True,
                color=CB_PALETTE['sky_blue'], edgecolor='white',
                alpha=0.75, label='Empirical')
        kde = gaussian_kde(ghi_values, bw_method='scott')
        xs = np.linspace(float(ghi_values.min()),
                         float(ghi_values.max()), 300)
        ax.plot(xs, kde(xs), color=CB_PALETTE['blue'], lw=2,
                label='KDE')
        ax.axvline(float(ghi_values.mean()),
                   color=CB_PALETTE['vermilion'], lw=1.5,
                   linestyle='--',
                   label=f'Mean = {float(ghi_values.mean()):.2f}')
        ax.set_xlabel('GHI (kWh m⁻²)')
        ax.set_ylabel('Density')
        ax.set_title('Empirical Distribution of GHI')
        ax.legend(framealpha=0.9)
        if save:
            self._save('fig_ghi_distribution.pdf')

    def plot_temporal_variability(self, processed_data, save=True):
        set_journal_style()
        ghi = processed_data['GHI']
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        groups_m = [ghi[ghi.index.month == m].values
                    for m in range(1, 13)]
        bp1 = axes[0].boxplot(
            groups_m, patch_artist=True,
            medianprops={'color': CB_PALETTE['vermilion'], 'lw': 2.0},
            whiskerprops={'color': 'grey'},
            capprops={'color': 'grey'},
            flierprops={'marker': 'o', 'markersize': 3,
                        'markerfacecolor': CB_PALETTE['orange'],
                        'alpha': 0.4})
        for p in bp1['boxes']:
            p.set_facecolor(CB_PALETTE['sky_blue'])
            p.set_alpha(0.60)
        axes[0].set_xticklabels(month_names, fontsize=9)
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('GHI (kWh m⁻²)')
        axes[0].set_title('(a) Monthly')
        years = sorted(ghi.index.year.unique())
        groups_y = [ghi[ghi.index.year == y].values for y in years]
        bp2 = axes[1].boxplot(
            groups_y, patch_artist=True,
            medianprops={'color': CB_PALETTE['vermilion'], 'lw': 2.0},
            whiskerprops={'color': 'grey'},
            capprops={'color': 'grey'},
            flierprops={'marker': 'o', 'markersize': 3,
                        'markerfacecolor': CB_PALETTE['orange'],
                        'alpha': 0.4})
        for p in bp2['boxes']:
            p.set_facecolor(CB_PALETTE['green'])
            p.set_alpha(0.50)
        axes[1].set_xticks(range(1, len(years) + 1))
        axes[1].set_xticklabels([str(y) for y in years],
                                rotation=45, ha='right', fontsize=8)
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('GHI (kWh m⁻²)')
        axes[1].set_title('(b) Inter-Annual')
        fig.suptitle('Temporal Variability of GHI', fontsize=12,
                     fontweight='bold')
        if save:
            self._save('fig_temporal_variability.pdf')

    def plot_ablation_study(self, ablation_results, save=True):
        set_journal_style()
        if not ablation_results:
            return
        configs = list(ablation_results.keys())
        rmse = [ablation_results[c].get('RMSE', 0) for c in configs]
        crps = [ablation_results[c].get('CRPS', 0) for c in configs]
        c_list = [CB_PALETTE['blue'], CB_PALETTE['sky_blue'],
                  CB_PALETTE['green'], CB_PALETTE['orange'],
                  CB_PALETTE['vermilion']][:len(configs)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, vals, xlabel, title in zip(
                axes, [rmse, crps], ['RMSE', 'CRPS'],
                ['Ablation: RMSE', 'Ablation: CRPS']):
            bars = ax.barh(configs, vals, color=c_list,
                           edgecolor='white', height=0.55)
            ax.set_xlabel(xlabel)
            ax.invert_yaxis()
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
            ax.set_title(title)
        fig.suptitle('Ablation Study', fontsize=12, fontweight='bold')
        if save:
            self._save('fig_ablation_study.pdf')

    def plot_sobol_sensitivity(self, sobol_results, save=True):
        set_journal_style()
        if not sobol_results:
            return
        features = list(sobol_results.get('S1', {}).keys())
        s1 = [sobol_results['S1'].get(f, 0) for f in features]
        st = [sobol_results['ST'].get(f, 0) for f in features]
        ci1 = [sobol_results.get('S1_conf', {}).get(f, 0.02)
               for f in features]
        cit = [sobol_results.get('ST_conf', {}).get(f, 0.02)
               for f in features]
        x = np.arange(len(features))
        w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w / 2, s1, w, color=CB_PALETTE['blue'],
               alpha=0.85, label='S1',
               yerr=ci1,
               error_kw={'ecolor': 'black', 'capsize': 4,
                         'elinewidth': 1})
        ax.bar(x + w / 2, st, w, color=CB_PALETTE['orange'],
               alpha=0.85, label='ST',
               yerr=cit,
               error_kw={'ecolor': 'black', 'capsize': 4,
                         'elinewidth': 1})
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=10)
        ax.set_ylabel('Sobol index')
        ax.set_title('Sobol Sensitivity Analysis')
        ax.legend(framealpha=0.9)
        ax.set_ylim(0, max(max(st), 0.6) * 1.2)
        if save:
            self._save('fig_sobol_sensitivity.pdf')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def diebold_mariano(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    md = np.mean(d)
    gamma = [np.cov(d[:-lag], d[lag:])[0, 1]
             for lag in range(1, max(2, h))]
    var_d = (np.var(d) + 2 * np.sum(gamma)) / len(d)
    dm = md / np.sqrt(max(var_d, 1e-12))
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


def kupiec_test(picp, n, nominal):
    k = max(1, min(int(round(picp * n)), n - 1))
    LR = -2 * (
        k * np.log(max(nominal, 1e-10))
        + (n - k) * np.log(max(1 - nominal, 1e-10))
        - k * np.log(max(k / n, 1e-10))
        - (n - k) * np.log(max(1 - k / n, 1e-10)))
    return LR, 1 - stats.chi2.cdf(LR, df=1)


def plot_capacity_risk(predictions, q05, q95, save_fn=None):
    set_journal_style()
    if save_fn is None:
        save_fn = os.path.join(CONFIG['output_dir'],
                               'fig_capacity_risk.pdf')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions, color=CB_PALETTE['blue'], lw=1.8,
            label='P50')
    ax.fill_between(range(len(predictions)), q05, q95,
                    color=CB_PALETTE['sky_blue'], alpha=0.35,
                    label='P5–P95')
    ax.set_ylabel('GHI (kWh m⁻²)')
    ax.set_xlabel('Time step')
    ax.set_title('Capacity Planning Risk View')
    ax.legend(framealpha=0.9)
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DatetimeIndex):
        return [t.isoformat() for t in obj]
    elif isinstance(obj, pd.Series):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return convert_to_serializable(obj.to_dict(orient='records'))
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v)
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return (str(obj)
                if not isinstance(
                    obj, (str, int, float, bool, type(None)))
                else obj)


# ============================================================================
# MAIN SOLAR FORECASTER
# ============================================================================
ENCODER_DECODER_MODELS = frozenset([
    'informer', 'transformer', 'hybrid', 'hybrid_transformer_lstm'
])


class SolarForecaster:
    def __init__(self, model_type='informer', data_processor=None):
        if data_processor is None:
            raise ValueError("A data_processor must be provided")
        self.model_type = model_type
        self.data_processor = data_processor
        self.device = device
        self.visualizer = SolarForecastVisualizer()
        self.results = {}
        self.uncertainty_metrics = {}
        self.calibration_results = {}
        self.ablation_results = {}
        self.model = self._build_model()
        self.model.to(self.device)
        logger.info(
            f"Initialized {model_type.upper()} model with "
            f"{sum(p.numel() for p in self.model.parameters()):,} "
            f"parameters")

    def _build_model(self):
        features = CONFIG['features'].copy()
        if (not CONFIG['ablation']['with_humidity']
                and 'RH2M' in features):
            features.remove('RH2M')
        n_features = len(features)
        if self.model_type == 'informer':
            p = CONFIG['model_params']['informer']
            return InformerModel(n_features, 1, 1, **p)
        elif self.model_type == 'transformer':
            p = CONFIG['model_params']['transformer']
            return TransformerModel(n_features, 1, 1, **p)
        elif self.model_type == 'lstm':
            p = CONFIG['model_params']['lstm']
            return LSTMModel(n_features, p['hidden_size'],
                             p['num_layers'], 1, p['dropout'])
        elif self.model_type == 'gru':
            p = CONFIG['model_params']['gru']
            return GRUModel(n_features, p['hidden_size'],
                            p['num_layers'], 1, p['dropout'])
        elif self.model_type == 'hybrid':
            p = CONFIG['model_params']['hybrid']
            return HybridLSTMInformerModel(n_features, 1, 1, **p)
        elif self.model_type == 'hybrid_transformer_lstm':
            p = CONFIG['model_params']['hybrid_transformer_lstm']
            return HybridTransformerLSTMModel(n_features, 1, 1, **p)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_model(self):
        logger.info(f"Training {self.model_type} model...")
        train_ds = self.data_processor.create_sequences(
            self.data_processor.train_data, self.model_type,
            is_train=True)
        val_ds = self.data_processor.create_sequences(
            self.data_processor.val_data, self.model_type,
            is_train=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG['training_params']['batch_size'],
            shuffle=True, worker_init_fn=seed_worker,
            generator=GEN)
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG['training_params']['batch_size'],
            shuffle=False, worker_init_fn=seed_worker,
            generator=GEN)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=CONFIG['training_params']['learning_rate'],
            weight_decay=CONFIG['training_params']['weight_decay'])
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        best_state = None
        for epoch in range(CONFIG['training_params']['epochs']):
            self.model.train()
            etl = []
            loop = tqdm(
                train_loader,
                desc=f"Training {self.model_type} E{epoch + 1}",
                leave=False)
            for batch in loop:
                optimizer.zero_grad()
                if self.model_type in ENCODER_DECODER_MODELS:
                    xe, xd, y = [b.to(self.device) for b in batch]
                    out = self.model(xe, xd)
                else:
                    x, y = (batch[0].to(self.device),
                            batch[1].to(self.device))
                    out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                etl.append(loss.item())
                loop.set_postfix(loss=np.mean(etl))

            self.model.eval()
            evl = []
            with torch.no_grad():
                for batch in val_loader:
                    if self.model_type in ENCODER_DECODER_MODELS:
                        xe, xd, y = [b.to(self.device) for b in batch]
                        out = self.model(xe, xd)
                    else:
                        x, y = (batch[0].to(self.device),
                                batch[1].to(self.device))
                        out = self.model(x)
                    evl.append(criterion(out, y).item())

            atl, avl = np.mean(etl), np.mean(evl)
            train_losses.append(atl)
            val_losses.append(avl)
            logger.info(
                f"Epoch {epoch + 1}/{CONFIG['training_params']['epochs']}"
                f" — Train: {atl:.4f}, Val: {avl:.4f}")
            if avl < best_val_loss:
                best_val_loss = avl
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(
                    {'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'val_loss': avl, 'epoch': epoch},
                    os.path.join(
                        CONFIG['output_dir'],
                        f'best_model_{self.model_type}.pth'))
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['training_params']['patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.visualizer.plot_training_history(
            train_losses, val_losses, self.model_type)
        return self.model

    def evaluate_model(self):
        logger.info(f"Evaluating {self.model_type.upper()} model...")
        test_ds = self.data_processor.create_sequences(
            self.data_processor.test_data,
            model_type=self.model_type, is_train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=CONFIG['training_params']['batch_size'],
            shuffle=False, worker_init_fn=seed_worker,
            generator=GEN)
        preds_list, acts_list = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if self.model_type in ENCODER_DECODER_MODELS:
                    xe, xd, y = [b.to(self.device) for b in batch]
                    out = self.model(xe, xd)
                    p_ = out[:, -CONFIG['pred_len']:, :].cpu().numpy()
                    a_ = y[:, -CONFIG['pred_len']:, :].cpu().numpy()
                else:
                    x, y = (batch[0].to(self.device),
                            batch[1].to(self.device))
                    out = self.model(x)
                    p_ = out.cpu().numpy()
                    a_ = y.cpu().numpy()
                preds_list.extend(p_.flatten())
                acts_list.extend(a_.flatten())

        predictions = np.array(preds_list)
        actuals = np.array(acts_list)

        ti = self.data_processor.train_data.columns.get_loc(CONFIG['target'])
        nc = len(self.data_processor.train_data.columns)
        dummy = np.zeros((len(predictions), nc))
        dummy[:, ti] = predictions
        po = self.data_processor.scaler.inverse_transform(dummy)[:, ti]
        dummy[:, ti] = actuals
        ao = self.data_processor.scaler.inverse_transform(dummy)[:, ti]

        mae = mean_absolute_error(ao, po)
        rmse = np.sqrt(mean_squared_error(ao, po))
        r2 = r2_score(ao, po)
        logger.info(
            f"Deterministic: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        mc = self.monte_carlo_uncertainty(test_loader)
        ci = mc['confidence_intervals']
        ml = min(len(ao), mc['samples'].shape[0])
        ao, po = ao[:ml], po[:ml]
        mcs = mc['samples'][:ml, :]

        picp90 = UncertaintyMetrics.calculate_picp(
            ao, ci[0.90][0][:ml], ci[0.90][1][:ml])
        pinaw90 = UncertaintyMetrics.calculate_pinaw(
            ci[0.90][0][:ml], ci[0.90][1][:ml],
            np.max(ao) - np.min(ao))
        crps = UncertaintyMetrics.calculate_crps(ao, mcs)
        logger.info(
            f"Probabilistic: CRPS={crps:.4f}, PICP90={picp90:.4f}, "
            f"PINAW90={pinaw90:.4f}")

        self.results = {
            'deterministic': {
                'MAE': mae, 'RMSE': rmse, 'R2': r2,
                'predictions': po, 'actuals': ao},
            'probabilistic': {
                'CRPS': crps, 'PICP_90': picp90,
                'PINAW_90': pinaw90,
                'mc_results': {
                    'mean': mc['mean'][:ml],
                    'std': mc['std'][:ml],
                    'samples': mcs,
                    'confidence_intervals': {
                        k: (v[0][:ml], v[1][:ml])
                        for k, v in mc['confidence_intervals'].items()}}}}

        td = self.data_processor.test_data
        na = len(td) - CONFIG['seq_len']
        dates = td.index[
            CONFIG['seq_len']:CONFIG['seq_len'] + min(na, ml)]
        pl = min(len(dates), ml)
        self.visualizer.plot_predictions(
            ao[:pl], po[:pl], dates[:pl], self.model_type)
        self.visualizer.plot_uncertainty_intervals(
            ao[:pl],
            self.results['probabilistic']['mc_results'],
            dates[:pl], self.model_type)
        return self.results

    def monte_carlo_uncertainty(self, test_loader):
        logger.info(f"MC uncertainty for {self.model_type}...")
        ns = CONFIG['monte_carlo']['n_mc_samples']
        all_preds = []
        old_det = None
        try:
            old_det = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        for _ in tqdm(range(ns), desc="MC Sampling"):
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    if self.model_type in ENCODER_DECODER_MODELS:
                        xe, xd, _ = batch
                        xe = xe.to(self.device)
                        xd = xd.to(self.device)
                        out = self.model(xe, xd, use_mc_dropout=True)
                        p_ = out[:, -CONFIG['pred_len']:, :].cpu().numpy()
                    else:
                        x, _ = batch
                        x = x.to(self.device)
                        out = self.model(x, use_mc_dropout=True)
                        p_ = out.cpu().numpy()
                    preds.extend(p_.flatten())
            all_preds.append(preds)
        if old_det is not None:
            try:
                torch.use_deterministic_algorithms(old_det)
            except Exception:
                pass

        mc_s = np.array(all_preds).T
        mean_p = np.mean(mc_s, axis=1)
        std_p = np.std(mc_s, axis=1)
        logger.info(f"  MC mean std: {np.mean(std_p):.6f}")

        ci = {}
        for cl in CONFIG['monte_carlo']['confidence_levels']:
            a = (1 - cl) / 2
            ci[cl] = (np.percentile(mc_s, a * 100, axis=1),
                      np.percentile(mc_s, (1 - a) * 100, axis=1))

        ti = self.data_processor.train_data.columns.get_loc(CONFIG['target'])
        nc = len(self.data_processor.train_data.columns)
        dummy = np.zeros((len(mean_p), nc))
        dummy[:, ti] = mean_p
        mean_o = self.data_processor.scaler.inverse_transform(dummy)[:, ti]

        mc_o = np.zeros_like(mc_s)
        for i in range(mc_s.shape[1]):
            dummy[:, ti] = mc_s[:, i]
            mc_o[:, i] = self.data_processor.scaler.inverse_transform(
                dummy)[:, ti]

        ci_o = {}
        for cl, (lo, hi) in ci.items():
            dummy[:, ti] = lo
            lo_o = self.data_processor.scaler.inverse_transform(
                dummy)[:, ti]
            dummy[:, ti] = hi
            hi_o = self.data_processor.scaler.inverse_transform(
                dummy)[:, ti]
            ci_o[cl] = (lo_o, hi_o)

        return {'mean': mean_o, 'std': std_p,
                'samples': mc_o,
                'confidence_intervals': ci_o}

    def generate_longterm_forecast(self, future_data: pd.DataFrame):
        """Generate long-term forecasts using synthetic/scenario future data.

        Returns dict with 'dates' and 'samples' (n_mc x n_steps).
        Uses the same prediction frequency as the model (no change).
        """
        logger.info(
            f"Generating long-term forecast for {self.model_type}...")

        if len(future_data) < CONFIG['seq_len'] + CONFIG['pred_len']:
            logger.warning("Insufficient future data for long-term forecast")
            return None

        # Create sequences from future data
        future_ds = self.data_processor.create_sequences(
            future_data, model_type=self.model_type, is_train=False)
        if len(future_ds) == 0:
            logger.warning("No sequences created from future data")
            return None

        future_loader = DataLoader(
            future_ds,
            batch_size=CONFIG['training_params']['batch_size'],
            shuffle=False, worker_init_fn=seed_worker,
            generator=GEN)

        n_mc = CONFIG['long_term']['n_mc_longterm']
        all_mc_preds = []

        old_det = None
        try:
            old_det = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

        for mc_idx in tqdm(range(n_mc), desc=f"LT MC {self.model_type}"):
            preds = []
            with torch.no_grad():
                for batch in future_loader:
                    if self.model_type in ENCODER_DECODER_MODELS:
                        xe, xd, _ = batch
                        xe = xe.to(self.device)
                        xd = xd.to(self.device)
                        out = self.model(xe, xd, use_mc_dropout=True)
                        p_ = out[:, -CONFIG['pred_len']:, :].cpu().numpy()
                    else:
                        x, _ = batch
                        x = x.to(self.device)
                        out = self.model(x, use_mc_dropout=True)
                        p_ = out.cpu().numpy()
                    preds.extend(p_.flatten())
            all_mc_preds.append(preds)

        if old_det is not None:
            try:
                torch.use_deterministic_algorithms(old_det)
            except Exception:
                pass

        mc_array = np.array(all_mc_preds)  # (n_mc, n_steps)

        # Inverse transform
        ti = self.data_processor.train_data.columns.get_loc(CONFIG['target'])
        nc = len(self.data_processor.train_data.columns)
        n_steps = mc_array.shape[1]
        mc_original = np.zeros_like(mc_array)

        for i in range(n_mc):
            dummy = np.zeros((n_steps, nc))
            dummy[:, ti] = mc_array[i, :]
            mc_original[i, :] = \
                self.data_processor.scaler.inverse_transform(dummy)[:, ti]

        # Dates for the predictions
        offset = CONFIG['seq_len']
        future_dates = future_data.index[offset:offset + n_steps]

        return {
            'dates': future_dates,
            'samples': mc_original,  # (n_mc, n_steps)
        }

    def assess_stationarity(self):
        tv = self.data_processor.train_data[CONFIG['target']].values
        adf_p = adfuller(tv)[1]
        kpss_p = kpss(tv, regression='c')[1]
        logger.info(f"  ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}")
        return {
            'adf': {'pvalue': adf_p, 'stationary': adf_p < 0.05},
            'kpss': {'pvalue': kpss_p, 'stationary': kpss_p > 0.05}}

    def validate_climate_scenarios(self):
        logger.info(
            f"Climate validation for {self.model_type.upper()}...")
        sr = {}
        for scenario in CONFIG['cmip6']['scenarios']:
            logger.info(f"  Scenario: {scenario}")
            if CONFIG['cmip6']['generate_synthetic']:
                sd = self.data_processor.generate_synthetic_future_data(
                    CONFIG['cmip6']['validation_years'],
                    CONFIG['cmip6']['synthetic_method'])
                if sd.empty:
                    continue
            else:
                sd = pd.DataFrame()
            sr[scenario] = {}
            for year in CONFIG['cmip6']['validation_years']:
                yd = (sd[sd.index.year == year]
                      if not sd.empty
                      else self.data_processor.processed_data[
                          self.data_processor.processed_data
                          .index.year == year])
                if len(yd) == 0:
                    continue
                yds = self.data_processor.create_sequences(
                    yd, model_type=self.model_type, is_train=False)
                if len(yds) == 0:
                    continue
                yl = DataLoader(
                    yds,
                    batch_size=CONFIG['training_params']['batch_size'],
                    shuffle=False, worker_init_fn=seed_worker,
                    generator=GEN)
                preds, acts = [], []
                self.model.eval()
                with torch.no_grad():
                    for batch in yl:
                        if self.model_type in ENCODER_DECODER_MODELS:
                            xe, xd, y = [b.to(self.device) for b in batch]
                            out = self.model(xe, xd)
                            p_ = out[:, -CONFIG['pred_len']:, :].cpu().numpy()
                            a_ = y[:, -CONFIG['pred_len']:, :].cpu().numpy()
                        else:
                            x, y = (batch[0].to(self.device),
                                    batch[1].to(self.device))
                            out = self.model(x)
                            p_ = out.cpu().numpy()
                            a_ = y.cpu().numpy()
                        preds.extend(p_.flatten())
                        acts.extend(a_.flatten())
                if not preds:
                    continue
                preds = np.array(preds)
                acts = np.array(acts)
                ti = self.data_processor.train_data.columns.get_loc(
                    CONFIG['target'])
                nc = len(self.data_processor.train_data.columns)
                dummy = np.zeros((len(preds), nc))
                dummy[:, ti] = preds
                po = self.data_processor.scaler.inverse_transform(
                    dummy)[:, ti]
                dummy[:, ti] = acts
                ao = self.data_processor.scaler.inverse_transform(
                    dummy)[:, ti]
                mae = mean_absolute_error(ao, po)
                rmse = np.sqrt(mean_squared_error(ao, po))
                logger.info(
                    f"    Year {year} — MAE={mae:.4f}, RMSE={rmse:.4f}")
                sr[scenario][year] = {'MAE': mae, 'RMSE': rmse}
        return sr

    def sobol_sensitivity_analysis(self):
        if not SALIB_AVAILABLE:
            logger.warning("SALib not available — skipping Sobol.")
            return None
        features = CONFIG['features'].copy()
        if (not CONFIG['ablation']['with_humidity']
                and 'RH2M' in features):
            features.remove('RH2M')
        problem = {'num_vars': len(features), 'names': features,
                   'bounds': [[0, 1] for _ in features]}
        try:
            pv = saltelli.sample(
                problem, CONFIG['sobol']['n_samples'],
                calc_second_order=CONFIG['sobol']['calc_second_order'])
            self.model.eval()
            predictions = []
            with torch.no_grad():
                for params in tqdm(pv, desc="Sobol eval"):
                    si = (torch.FloatTensor(params)
                          .unsqueeze(0).unsqueeze(0)
                          .repeat(1, CONFIG['seq_len'], 1)
                          .to(self.device))
                    if self.model_type in ('lstm', 'gru'):
                        out = self.model(si)
                    else:
                        di = torch.zeros(
                            1,
                            CONFIG['label_len'] + CONFIG['pred_len'],
                            1).to(self.device)
                        out = self.model(si, di)
                    predictions.append(out.cpu().numpy().flatten()[0])
            predictions = np.array(predictions)
            si = sobol.analyze(
                problem, predictions,
                calc_second_order=CONFIG['sobol']['calc_second_order'])
            return {
                'S1': dict(zip(features, si['S1'])),
                'ST': dict(zip(features, si['ST'])),
                'S1_conf': dict(zip(features, si['S1_conf'])),
                'ST_conf': dict(zip(features, si['ST_conf']))}
        except Exception as e:
            logger.error(f"Sobol failed: {e}")
            return None

    def backtesting(self):
        logger.info(f"Backtesting for {self.model_type.upper()}...")
        results = {}
        data = self.data_processor.processed_data
        for ws in CONFIG['backtesting']['window_sizes']:
            wr = []
            for hz in CONFIG['backtesting']['horizons']:
                if len(data) < ws + hz:
                    continue
                preds, acts = [], []
                for i in range(0, len(data) - ws - hz,
                               CONFIG['backtesting']['step_size']):
                    te = i + ws
                    tw = data.iloc[te:te + hz]
                    if len(tw) < hz:
                        continue
                    try:
                        tds = self.data_processor.create_sequences(
                            tw, model_type=self.model_type,
                            is_train=False)
                        if len(tds) == 0:
                            continue
                        tl = DataLoader(
                            tds,
                            batch_size=CONFIG['training_params'][
                                'batch_size'],
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=GEN)
                        self.model.eval()
                        with torch.no_grad():
                            for batch in tl:
                                if self.model_type in ENCODER_DECODER_MODELS:
                                    xe, xd, y = [b.to(self.device)
                                                 for b in batch]
                                    out = self.model(xe, xd)
                                    p_ = (out[:, -CONFIG['pred_len']:, :]
                                          .cpu().numpy().flatten())
                                    a_ = (y[:, -CONFIG['pred_len']:, :]
                                          .cpu().numpy().flatten())
                                else:
                                    x, y = (batch[0].to(self.device),
                                            batch[1].to(self.device))
                                    out = self.model(x)
                                    p_ = out.cpu().numpy().flatten()
                                    a_ = y.cpu().numpy().flatten()
                                preds.extend(p_)
                                acts.extend(a_)
                    except Exception:
                        continue
                if preds:
                    preds = np.array(preds)
                    acts = np.array(acts)
                    wr.append({
                        'horizon': hz,
                        'RMSE': np.sqrt(mean_squared_error(acts, preds)),
                        'MAE': mean_absolute_error(acts, preds),
                        'n_samples': len(preds)})
            results[f'window_{ws}'] = wr
        return results

    def calibration_diagnostics(self):
        if 'probabilistic' not in self.results:
            logger.warning("No probabilistic results.")
            return None
        mc = self.results['probabilistic']['mc_results']
        ao = self.results['deterministic']['actuals']
        po = self.results['deterministic']['predictions']
        s = mc['samples']
        ml = min(len(ao), s.shape[0])
        ao, po, s = ao[:ml], po[:ml], s[:ml, :]
        cr = {}
        cr['coverage'] = UncertaintyMetrics.calculate_coverage_vs_nominal(
            ao, s, CONFIG['monte_carlo']['confidence_levels'])
        cr['pit'] = UncertaintyMetrics.pit_histogram(
            ao, s, CONFIG['calibration']['pit_bins'])
        cr['reliability'] = UncertaintyMetrics.reliability_diagram(
            ao, s, CONFIG['calibration']['n_bins'])
        cr['statistical_tests'] = UncertaintyMetrics.statistical_tests(
            ao, s, po)
        if CONFIG['ablation']['time_varying_stability']:
            cr['stability'] = UncertaintyMetrics.time_varying_stability(
                ao, s)
        self.visualizer.plot_coverage_vs_nominal(
            cr['coverage'], self.model_type)
        self.visualizer.plot_reliability_diagram(
            cr['reliability'], self.model_type)
        self.visualizer.plot_pit_histogram(
            cr['pit'], self.model_type)
        return cr

    def ablation_study(self):
        logger.info(f"Ablation for {self.model_type.upper()}...")
        results = {}
        if self.results:
            results['baseline'] = {
                'RMSE': self.results['deterministic']['RMSE'],
                'MAE': self.results['deterministic']['MAE'],
                'CRPS': self.results['probabilistic']['CRPS']}
        else:
            return results
        orig = CONFIG['ablation']['with_humidity']
        if 'RH2M' in CONFIG['features'] and orig:
            results['without_humidity'] = {
                'RMSE': None, 'MAE': None, 'CRPS': None,
                'note': 'Requires separate training run'}
        CONFIG['ablation']['with_humidity'] = orig
        return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    logger.info("Starting Enhanced Solar Forecasting System")

    file_path = "solar_data.xlsx"
    data_processor = SolarDataProcessor(file_path)
    processed = data_processor.load_and_preprocess()
    if processed is None:
        logger.error("Failed to load data. Exiting.")
        return

    viz_main = SolarForecastVisualizer()
    viz_main.plot_pipeline_schematic()
    viz_main.plot_ghi_distribution(
        data_processor.processed_data['GHI'].values)
    viz_main.plot_temporal_variability(
        data_processor.processed_data)

    # ── Load CMIP6 scenarios ──────────────────────────────────────────
    cmip6_loader = CMIP6ScenarioLoader(
        scenario_files=CONFIG['cmip6']['scenario_files'],
        historical_data=data_processor.processed_data,
        aggregation=CONFIG['aggregation'])
    cmip6_raw = cmip6_loader.load_scenarios()

    # If scenario files weren't found, generate synthetic scenario data
    if not cmip6_raw:
        logger.info("No CMIP6 files found. Generating synthetic scenarios.")
        for scenario in CONFIG['cmip6']['scenarios']:
            future_years = list(range(2025, 2041))
            synth = data_processor.generate_synthetic_future_data(
                future_years, CONFIG['cmip6']['synthetic_method'])
            if not synth.empty:
                # Apply scenario-specific trends
                if scenario == 'ssp585':
                    # Stronger warming → slight GHI change
                    trend = np.linspace(0, 0.02, len(synth))
                    if 'GHI' in synth.columns:
                        synth['GHI'] = synth['GHI'] * (1 + trend)
                    if 'T2M' in synth.columns:
                        synth['T2M'] = synth['T2M'] + np.linspace(0, 3, len(synth))
                elif scenario == 'ssp245':
                    trend = np.linspace(0, 0.01, len(synth))
                    if 'GHI' in synth.columns:
                        synth['GHI'] = synth['GHI'] * (1 + trend)
                    if 'T2M' in synth.columns:
                        synth['T2M'] = synth['T2M'] + np.linspace(0, 1.5, len(synth))
                cmip6_loader.scenario_data[scenario] = synth

    # Bias correct
    if cmip6_loader.scenario_data:
        cmip6_loader.bias_correct_quantile_mapping()
    scenario_data_corrected = cmip6_loader.bias_corrected \
        if cmip6_loader.bias_corrected else cmip6_loader.scenario_data

    # ── Generate synthetic future data for long-term forecasts ────────
    future_years = CONFIG['long_term']['forecast_years']
    future_data_synthetic = data_processor.generate_synthetic_future_data(
        future_years, CONFIG['cmip6']['synthetic_method'])
    logger.info(
        f"Generated {len(future_data_synthetic)} synthetic future records")

    # ── Deep-learning models ──────────────────────────────────────────
    model_names = [
        'hybrid', 'hybrid_transformer_lstm', 'informer',
        'transformer', 'lstm', 'gru'
    ]
    models = {}
    for name in model_names:
        models[name] = SolarForecaster(name, data_processor)

    all_results = {}
    longterm_forecasts = {}  # Store long-term forecasts for all models

    for model_name, forecaster in models.items():
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing {model_name.upper()}")
        logger.info(f"{'=' * 50}")

        forecaster.train_model()
        mr = forecaster.evaluate_model()
        mr['stationarity'] = forecaster.assess_stationarity()
        mr['climate_scenarios'] = forecaster.validate_climate_scenarios()

        if SALIB_AVAILABLE:
            sobol_res = forecaster.sobol_sensitivity_analysis()
            if sobol_res:
                mr['sobol'] = sobol_res
                viz_main.plot_sobol_sensitivity(sobol_res)

        mr['backtesting'] = forecaster.backtesting()
        cal = forecaster.calibration_diagnostics()
        if cal:
            mr['calibration'] = cal
        mr['ablation'] = forecaster.ablation_study()
        all_results[model_name] = mr

        # ── Long-term forecast for this model ─────────────────────────
        if not future_data_synthetic.empty:
            lt_forecast = forecaster.generate_longterm_forecast(
                future_data_synthetic)
            if lt_forecast is not None:
                longterm_forecasts[model_name] = lt_forecast
                hist_dates = data_processor.processed_data.index
                hist_ghi = data_processor.processed_data['GHI'].values

                # Plot WITHOUT scenarios
                viz_main.plot_longterm_prediction(
                    hist_dates, hist_ghi,
                    lt_forecast['dates'],
                    lt_forecast['samples'],
                    model_name)

                # Plot WITH scenarios
                if scenario_data_corrected:
                    # Find closest scenario for this model
                    p50 = np.percentile(lt_forecast['samples'], 50, axis=0)
                    closest = cmip6_loader.compute_closest_scenario(
                        p50, lt_forecast['dates'])

                    viz_main.plot_longterm_with_scenarios(
                        hist_dates, hist_ghi,
                        lt_forecast['dates'],
                        lt_forecast['samples'],
                        scenario_data_corrected,
                        closest if closest else 'ssp245',
                        model_name)

    # ── Find best model ───────────────────────────────────────────────
    best_model_name = None
    best_rmse = float('inf')
    for name, res in all_results.items():
        if (isinstance(res, dict) and 'deterministic' in res
                and name in model_names):
            rmse = res['deterministic'].get('RMSE', float('inf'))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
    logger.info(f"\n★ Best model: {best_model_name} (RMSE={best_rmse:.4f})")

    # ── All-models long-term comparison plots ─────────────────────────
    if longterm_forecasts:
        hist_dates = data_processor.processed_data.index
        hist_ghi = data_processor.processed_data['GHI'].values

        # All models comparison WITHOUT scenarios
        viz_main.plot_all_models_longterm(
            hist_dates, hist_ghi, longterm_forecasts)

        # All models comparison WITH scenarios
        if scenario_data_corrected and best_model_name:
            # Compute closest scenario using best model
            best_lt = longterm_forecasts.get(best_model_name)
            if best_lt is not None:
                p50_best = np.percentile(best_lt['samples'], 50, axis=0)
                closest_overall = cmip6_loader.compute_closest_scenario(
                    p50_best, best_lt['dates'])

                viz_main.plot_all_models_longterm_with_scenarios(
                    hist_dates, hist_ghi,
                    longterm_forecasts,
                    scenario_data_corrected,
                    best_model_name,
                    closest_overall if closest_overall else 'ssp245')

                # Detailed best model + scenarios plot
                viz_main.plot_best_model_with_scenarios(
                    hist_dates, hist_ghi,
                    best_lt['dates'], best_lt['samples'],
                    scenario_data_corrected,
                    closest_overall if closest_overall else 'ssp245',
                    best_model_name)

    # ── Optimised baselines ───────────────────────────────────────────
    baseline_engine = BaselineModels(
        data_processor.train_data, data_processor.test_data)
    baseline_results = baseline_engine.evaluate_all()

    # Store into all_results
    for key, res in baseline_results.items():
        if res is not None:
            all_results[f'baseline_{key}'] = {
                'deterministic': {
                    'MAE': res['MAE'],
                    'RMSE': res['RMSE'],
                    'R2': res['R2'],
                    'MAPE': res.get('MAPE'),
                    'predictions': res['predictions'],
                    'best_params': res.get('best_params'),
                    'fit_time': res.get('fit_time'),
                    'name': res.get('name'),
                }
            }
    all_results['baseline_summary'] = baseline_results

    # Visualise baseline comparison
    viz_main.plot_baseline_comparison(baseline_results)

    # Overall model comparison (DL + baselines)
    viz_main.plot_model_comparison(all_results)

    # ── Save results ──────────────────────────────────────────────────
    # Remove non-serializable arrays from longterm for JSON
    serializable_results = {}
    for k, v in all_results.items():
        serializable_results[k] = v

    with open(os.path.join(CONFIG['output_dir'],
                           'solar_forecasting_results.json'), 'w') as f:
        json.dump(convert_to_serializable(serializable_results), f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────
    logger.info(f"\n{'═' * 70}")
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'═' * 70}")
    logger.info(f"{'Model':<30s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
    logger.info(f"{'─' * 70}")
    for name in model_names:
        if name in all_results and 'deterministic' in all_results[name]:
            det = all_results[name]['deterministic']
            logger.info(
                f"{name.upper():<30s} {det['MAE']:>8.4f} "
                f"{det['RMSE']:>8.4f} {det['R2']:>8.4f}")
    for key, res in baseline_results.items():
        if res is not None:
            logger.info(
                f"{res['name']:<30s} {res['MAE']:>8.4f} "
                f"{res['RMSE']:>8.4f} {res['R2']:>8.4f}")
    logger.info(f"{'═' * 70}")
    if best_model_name:
        logger.info(f"★ Best DL model: {best_model_name.upper()}")
    if scenario_data_corrected:
        logger.info(f"★ CMIP6 scenarios loaded: "
                    f"{list(scenario_data_corrected.keys())}")

    logger.info(
        "\n✓ Solar forecasting analysis completed successfully!")
    logger.info(f"  Output directory: {CONFIG['output_dir']}")
    logger.info(f"  Plots generated:")
    logger.info(f"    • Per-model: training history, predictions, "
                f"fan charts, uncertainty")
    logger.info(f"    • Long-term (no scenarios): one per model + "
                f"all-models comparison")
    logger.info(f"    • Long-term (with CMIP6): one per model + "
                f"all-models + best model detail")
    logger.info(f"    • Baselines, model comparison, sensitivity, "
                f"calibration")


if __name__ == '__main__':
    main()