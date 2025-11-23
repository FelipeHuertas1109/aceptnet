"""
Entrenamiento Dual-Encoder para AFDs (Optimizado para GPU)
===========================================================

Este script entrena el modelo usando el dataset pre-generado.
Requiere GPU (T4 en Colab) para entrenamiento rápido.

Requisitos:
- dataset6000.csv (para features de AFDs)
- dataset_generated.csv (generado por gen_dataset_exhaustive.py)

Instrucciones para Google Colab:
1. Sube dataset6000.csv y dataset_generated.csv a /content/sample_data/
2. Ejecuta este script completo
3. Los resultados se guardarán en /content/

Salidas:
- best_model.pt: Modelo entrenado
- thresholds.json: Umbrales calibrados
- training_history.png: Gráficas de entrenamiento
"""

import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
MAX_STATES = 16
NUM_SYMBOLS = len(ALPHABET)
PAD_IDX = len(ALPHABET)

# Configuración de entrenamiento
TRAIN_CONFIG = {
    'label_smoothing': 0.0,      # Sin smoothing: Y1 es lógica exacta (0 o 1)
    'lambda1': 1.0,              # Peso de Y1 (pertenencia)
    'lambda2': 0.1,              # Peso de Y2 (compartida) - REDUCIDO para evitar que Y2 contamine Y1
                                  # (Mejora A: reduce influencia de Y2 en el encoder de cadenas)
    'batch_size': 128,
    'num_epochs': 40,            # Máximo de épocas (puede parar antes)
    'early_stop_patience': 7,    # Espera 7 épocas sin mejora
    'early_stop_min_delta': 1e-4 # Mejora mínima requerida (0.0001)
}


# ============================================================================
# PARSER DE AFDs
# ============================================================================

class AFDParser:
    """Parser para extraer información estructurada de los AFDs del dataset"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.afd_cache = {}
        
    def parse_states(self, estados_str: str) -> List[int]:
        """Parsea 'S0 S1 S2 S3' -> [0, 1, 2, 3]"""
        if pd.isna(estados_str):
            return []
        return [int(s.replace('S', '')) for s in estados_str.split()]
    
    def parse_accept_states(self, accept_str: str) -> Set[int]:
        """Parsea estados de aceptación"""
        if pd.isna(accept_str):
            return set()
        return set(int(s.replace('S', '')) for s in accept_str.split())
    
    def parse_transitions(self, trans_str: str, alphabet_str: str) -> Dict[Tuple[int, str], int]:
        """
        Parsea transiciones: 'S0 --A--> S1 | S0 --B--> S2' 
        -> {(0, 'A'): 1, (0, 'B'): 2}
        """
        transitions = {}
        if pd.isna(trans_str):
            return transitions
            
        # Regex para capturar: S0 --A--> S1
        pattern = r'S(\d+)\s*--(\w+)-->\s*S(\d+)'
        matches = re.findall(pattern, trans_str)
        
        for from_state, symbol, to_state in matches:
            transitions[(int(from_state), symbol)] = int(to_state)
            
        return transitions
    
    def get_afd_features(self, dfa_id: int) -> np.ndarray:
        """
        Extrae representación vectorial del AFD:
        Total: 16*12*16 + 16 + 16 = 3104 features
        """
        if dfa_id in self.afd_cache:
            return self.afd_cache[dfa_id]
            
        row = self.df.iloc[dfa_id]
        
        # Parsear información del AFD
        estados = self.parse_states(row['Estados'])
        accept_states = self.parse_accept_states(row['Estados de aceptación'])
        transitions = self.parse_transitions(row['Transiciones'], row['Alfabeto'])
        
        # Crear vectores
        accept_vec = np.zeros(MAX_STATES, dtype=np.float32)
        valid_states = np.zeros(MAX_STATES, dtype=np.float32)
        one_hot_T = np.zeros((MAX_STATES, NUM_SYMBOLS, MAX_STATES), dtype=np.float32)
        
        # Rellenar vectores
        for state in estados:
            if state < MAX_STATES:
                valid_states[state] = 1.0
                
        for state in accept_states:
            if state < MAX_STATES:
                accept_vec[state] = 1.0
        
        # Rellenar matriz de transiciones
        for (from_state, symbol), to_state in transitions.items():
            if symbol in CHAR_TO_IDX and from_state < MAX_STATES and to_state < MAX_STATES:
                symbol_idx = CHAR_TO_IDX[symbol]
                one_hot_T[from_state, symbol_idx, to_state] = 1.0
        
        # Concatenar todo en un vector plano
        features = np.concatenate([
            one_hot_T.flatten(),
            accept_vec,
            valid_states
        ])
        
        self.afd_cache[dfa_id] = features
        return features


# ============================================================================
# DATASET DE PYTORCH
# ============================================================================

class AFDStringDataset(Dataset):
    """Dataset de PyTorch para pares (dfa_id, string, y1, y2)"""
    
    def __init__(self, df: pd.DataFrame, parser: AFDParser):
        self.df = df.reset_index(drop=True)
        self.parser = parser
        
    def __len__(self):
        return len(self.df)
    
    def string_to_indices(self, string: str) -> List[int]:
        """Convierte cadena a índices: 'ABC' -> [0, 1, 2]"""
        if string == "<EPS>" or string == "":
            return []
        return [CHAR_TO_IDX[char] for char in string if char in CHAR_TO_IDX]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        afd_features = self.parser.get_afd_features(row['dfa_id'])
        string_tokens = self.string_to_indices(row['string'])
        
        y1 = float(row['label'])
        y2 = float(row['y2'])
        
        return {
            'afd_features': torch.tensor(afd_features, dtype=torch.float32),
            'string_tokens': torch.tensor(string_tokens, dtype=torch.long),
            'string_length': len(string_tokens),
            'y1': torch.tensor(y1, dtype=torch.float32),
            'y2': torch.tensor(y2, dtype=torch.float32)
        }


# ============================================================================
# UTILIDADES DE ENTRENAMIENTO
# ============================================================================

def smooth_targets(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Aplica label smoothing para mitigar sobreajuste extremo."""
    if epsilon <= 0:
        return targets
    return targets * (1 - epsilon) + 0.5 * epsilon


class EarlyStopping:
    """Detiene entrenamiento cuando la mejora en validación se estanca."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def collate_fn(batch):
    """Collate function para manejar secuencias de longitud variable"""
    afd_features = torch.stack([item['afd_features'] for item in batch])
    y1 = torch.stack([item['y1'] for item in batch])
    y2 = torch.stack([item['y2'] for item in batch])
    
    string_lengths = [item['string_length'] for item in batch]
    max_len = max(string_lengths) if string_lengths and max(string_lengths) > 0 else 1
    
    padded_strings = []
    for item in batch:
        tokens = item['string_tokens']
        padded = F.pad(tokens, (0, max_len - len(tokens)), value=PAD_IDX)
        padded_strings.append(padded)
    
    string_tokens = torch.stack(padded_strings)
    string_lengths = torch.tensor(string_lengths, dtype=torch.long)
    
    return {
        'afd_features': afd_features,
        'string_tokens': string_tokens,
        'string_lengths': string_lengths,
        'y1': y1,
        'y2': y2
    }


# ============================================================================
# MODELO DUAL-ENCODER
# ============================================================================

class DualEncoderModel(nn.Module):
    """Modelo dual-encoder con dos cabezas de salida"""
    
    def __init__(self, 
                 vocab_size: int = NUM_SYMBOLS + 1,
                 embed_dim: int = 32,
                 rnn_hidden_dim: int = 64,
                 afd_input_dim: int = 3104,
                 afd_hidden_dim: int = 128,
                 combined_hidden_dim: int = 128):
        super().__init__()
        
        # String Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_dim, 
                         num_layers=2, 
                         bidirectional=True, 
                         batch_first=True,
                         dropout=0.2)
        self.rnn_output_dim = rnn_hidden_dim * 2
        
        # AFD Encoder
        self.afd_encoder = nn.Sequential(
            nn.Linear(afd_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, afd_hidden_dim),
            nn.ReLU()
        )
        
        # Head 1: Pertenencia
        self.head1 = nn.Sequential(
            nn.Linear(self.rnn_output_dim + afd_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Cadena compartida
        self.head2 = nn.Sequential(
            nn.Linear(self.rnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, string_tokens, string_lengths, afd_features):
        batch_size = string_tokens.size(0)
        device = string_tokens.device
        
        # Manejar cadenas vacías
        empty_mask = string_lengths == 0
        non_empty_mask = ~empty_mask
        
        h_str = torch.zeros(batch_size, self.rnn_output_dim, device=device)
        
        if non_empty_mask.any():
            non_empty_tokens = string_tokens[non_empty_mask]
            non_empty_lengths = string_lengths[non_empty_mask]
            
            embedded = self.embedding(non_empty_tokens)
            
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                non_empty_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            _, hidden = self.rnn(packed)
            h_str_non_empty = torch.cat([hidden[-2], hidden[-1]], dim=1)
            h_str[non_empty_mask] = h_str_non_empty
        
        # Encode AFD
        h_afd = self.afd_encoder(afd_features)
        
        # Concatenar y predecir
        h_combined = torch.cat([h_str, h_afd], dim=1)
        
        y1_hat = self.head1(h_combined).squeeze(1)
        y2_hat = self.head2(h_str).squeeze(1)
        
        return y1_hat, y2_hat


# ============================================================================
# ENTRENADOR
# ============================================================================

class Trainer:
    """Entrenador del modelo dual-task optimizado para GPU"""
    
    def __init__(self, model, train_loader, val_loader, 
                 lambda1=1.0, lambda2=1.0, lr=0.001, device='cuda',
                 label_smoothing: float = 0.0,
                 early_stop_patience: int = 5,
                 early_stop_min_delta: float = 1e-3,
                 train_df: pd.DataFrame = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        self.label_smoothing = label_smoothing
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Mejora B: Calcular pos_weight para Y1 (balancear positivos vs negativos)
        # Esto penaliza más los falsos negativos (cuando el modelo rechaza una cadena aceptada)
        # Útil para casos como "AA", "AAAA", "IBBB" que son aceptadas pero el modelo las rechaza
        if train_df is not None:
            pos_count = (train_df['label'] == 1).sum()
            neg_count = (train_df['label'] == 0).sum()
            if pos_count > 0 and neg_count > 0:
                pos_weight = neg_count / pos_count
                print(f"   📊 Balance Y1: {pos_count:,} positivos, {neg_count:,} negativos")
                print(f"   ⚖️  Pos weight para Y1: {pos_weight:.3f} (penaliza más falsos negativos)")
                self.pos_weight_y1 = torch.tensor(pos_weight, device=device, dtype=torch.float32)
            else:
                self.pos_weight_y1 = None
        else:
            self.pos_weight_y1 = None
        
        self.criterion_y2 = nn.BCELoss()  # Y2 sin peso (ya está balanceada por diseño)
        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta
        )
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc_y1': [], 'val_acc_y1': [],
            'train_acc_y2': [], 'val_acc_y2': []
        }

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return smooth_targets(targets, self.label_smoothing)
    
    def _weighted_bce_y1(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        BCE ponderado para Y1: penaliza más cuando fallas un positivo.
        preds y targets están entre 0 y 1 (ya pasaron por Sigmoid).
        """
        eps = 1e-7
        preds = preds.clamp(eps, 1.0 - eps)  # Evitar log(0)
        
        if self.pos_weight_y1 is None:
            # BCE normal si no hay peso
            return -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds)).mean()
        
        # Peso extra para positivos (falsos negativos duelen más)
        pos_w = self.pos_weight_y1
        loss_pos = -pos_w * targets * torch.log(preds)
        loss_neg = -(1 - targets) * torch.log(1 - preds)
        return (loss_pos + loss_neg).mean()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_y1_pred, all_y1_true = [], []
        all_y2_pred, all_y2_true = [], []
        
        pbar = tqdm(self.train_loader, desc="Entrenando", leave=False)
        for batch in pbar:
            string_tokens = batch['string_tokens'].to(self.device)
            string_lengths = batch['string_lengths'].to(self.device)
            afd_features = batch['afd_features'].to(self.device)
            y1_true = batch['y1'].to(self.device)
            y2_true = batch['y2'].to(self.device)
            
            y1_hat, y2_hat = self.model(string_tokens, string_lengths, afd_features)
            
            y1_targets = self._prepare_targets(y1_true)
            y2_targets = self._prepare_targets(y2_true)
            
            loss1 = self._weighted_bce_y1(y1_hat, y1_targets)  # BCE ponderado manual para Y1
            loss2 = self.criterion_y2(y2_hat, y2_targets)  # BCELoss normal para Y2
            loss = self.lambda1 * loss1 + self.lambda2 * loss2
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            all_y1_pred.extend((y1_hat > 0.5).cpu().numpy())
            all_y1_true.extend(y1_true.cpu().numpy())
            all_y2_pred.extend((y2_hat > 0.5).cpu().numpy())
            all_y2_true.extend(y2_true.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        acc_y1 = accuracy_score(all_y1_true, all_y1_pred)
        acc_y2 = accuracy_score(all_y2_true, all_y2_pred)
        
        return avg_loss, acc_y1, acc_y2
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_y1_pred, all_y1_true = [], []
        all_y2_pred, all_y2_true = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validando", leave=False):
                string_tokens = batch['string_tokens'].to(self.device)
                string_lengths = batch['string_lengths'].to(self.device)
                afd_features = batch['afd_features'].to(self.device)
                y1_true = batch['y1'].to(self.device)
                y2_true = batch['y2'].to(self.device)
                
                y1_hat, y2_hat = self.model(string_tokens, string_lengths, afd_features)
                
                y1_targets = self._prepare_targets(y1_true)
                y2_targets = self._prepare_targets(y2_true)
                
                loss1 = self._weighted_bce_y1(y1_hat, y1_targets)  # BCE ponderado manual para Y1
                loss2 = self.criterion_y2(y2_hat, y2_targets)  # BCELoss normal para Y2
                loss = self.lambda1 * loss1 + self.lambda2 * loss2
                
                total_loss += loss.item()
                
                all_y1_pred.extend((y1_hat > 0.5).cpu().numpy())
                all_y1_true.extend(y1_true.cpu().numpy())
                all_y2_pred.extend((y2_hat > 0.5).cpu().numpy())
                all_y2_true.extend(y2_true.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        acc_y1 = accuracy_score(all_y1_true, all_y1_pred)
        acc_y2 = accuracy_score(all_y2_true, all_y2_pred)
        
        return avg_loss, acc_y1, acc_y2
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        best_epoch = 0
        
        print(f"\n🚀 Entrenando modelo durante hasta {num_epochs} épocas en {self.device.upper()}...")
        print(f"   ⏹️  Early stopping: paciencia={self.early_stopping.patience}, min_delta={self.early_stopping.min_delta}")
        print("="*70)
        
        for epoch in range(num_epochs):
            train_loss, train_acc_y1, train_acc_y2 = self.train_epoch()
            val_loss, val_acc_y1, val_acc_y2 = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc_y1'].append(train_acc_y1)
            self.history['val_acc_y1'].append(val_acc_y1)
            self.history['train_acc_y2'].append(train_acc_y2)
            self.history['val_acc_y2'].append(val_acc_y2)
            
            self.scheduler.step(val_loss)
            
            # Guardar mejor modelo
            improved = False
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), 'best_model.pt')
                improved = True
            
            # Mostrar progreso
            status = "💾 MEJOR" if improved else f"⏳ {self.early_stopping.counter}/{self.early_stopping.patience}"
            print(f"Época {epoch+1:02d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Y1: {train_acc_y1:.4f} Y2: {train_acc_y2:.4f} | "
                  f"Val Loss: {val_loss:.4f} Y1: {val_acc_y1:.4f} Y2: {val_acc_y2:.4f} | "
                  f"{status}")

            # Early stopping
            if self.early_stopping.step(val_loss):
                print("\n" + "="*70)
                print(f"⏹️  EARLY STOPPING activado en época {epoch+1}")
                print(f"   La pérdida de validación no mejoró durante {self.early_stopping.patience} épocas")
                print(f"   Mejor modelo guardado en época {best_epoch} (val_loss={best_val_loss:.4f})")
                print("="*70)
                break
        else:
            # Si completó todas las épocas sin early stopping
            print("\n" + "="*70)
            print(f"✅ Entrenamiento completado: {num_epochs} épocas")
            print(f"   Mejor modelo en época {best_epoch} (val_loss={best_val_loss:.4f})")
            print("="*70)


# ============================================================================
# CALIBRACIÓN Y EVALUACIÓN
# ============================================================================

def find_best_threshold(model, loader, device, task: str):
    """Busca el umbral que maximiza F1 para la tarea especificada."""
    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for batch in loader:
            string_tokens = batch['string_tokens'].to(device)
            string_lengths = batch['string_lengths'].to(device)
            afd_features = batch['afd_features'].to(device)

            y1_hat, y2_hat = model(string_tokens, string_lengths, afd_features)

            if task == 'y1':
                scores.extend(y1_hat.cpu().numpy())
                labels.extend(batch['y1'].cpu().numpy())
            else:
                scores.extend(y2_hat.cpu().numpy())
                labels.extend(batch['y2'].cpu().numpy())

    if not scores:
        return 0.5, 0.0

    scores = np.array(scores)
    labels = np.array(labels).astype(int)
    candidate_thresholds = np.linspace(0.2, 0.8, 61)

    best_threshold, best_f1 = 0.5, 0.0
    for threshold in candidate_thresholds:
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calibrate_thresholds(model, val_loader, device):
    """Calibra y guarda los mejores umbrales para Y1 y Y2 usando validación."""
    print("\n🔧 Calibrando umbrales con validación...")
    thresholds = {}

    y1_th, y1_f1 = find_best_threshold(model, val_loader, device, task='y1')
    print(f"   Y1 → umbral óptimo: {y1_th:.3f} (F1={y1_f1:.3f})")
    thresholds['y1'] = round(y1_th, 4)

    y2_th, y2_f1 = find_best_threshold(model, val_loader, device, task='y2')
    print(f"   Y2 → umbral óptimo: {y2_th:.3f} (F1={y2_f1:.3f})")
    thresholds['y2'] = round(y2_th, 4)

    with open('thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    print("   ✓ Umbrales guardados en 'thresholds.json'")

    return thresholds


class Evaluator:
    """Evaluador completo con métricas detalladas"""
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self, thresholds=None):
        self.model.eval()
        
        all_y1_pred, all_y1_true, all_y1_scores = [], [], []
        all_y2_pred, all_y2_true, all_y2_scores = [], [], []

        y1_threshold = thresholds.get('y1', 0.5) if thresholds else 0.5
        y2_threshold = thresholds.get('y2', 0.5) if thresholds else 0.5
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluando"):
                string_tokens = batch['string_tokens'].to(self.device)
                string_lengths = batch['string_lengths'].to(self.device)
                afd_features = batch['afd_features'].to(self.device)
                y1_true = batch['y1'].cpu().numpy()
                y2_true = batch['y2'].cpu().numpy()
                
                y1_hat, y2_hat = self.model(string_tokens, string_lengths, afd_features)
                
                y1_scores = y1_hat.cpu().numpy()
                y2_scores = y2_hat.cpu().numpy()
                y1_pred = (y1_scores >= y1_threshold).astype(int)
                y2_pred = (y2_scores >= y2_threshold).astype(int)
                
                all_y1_pred.extend(y1_pred)
                all_y1_true.extend(y1_true)
                all_y1_scores.extend(y1_scores)
                
                all_y2_pred.extend(y2_pred)
                all_y2_true.extend(y2_true)
                all_y2_scores.extend(y2_scores)
        
        print("\n" + "="*70)
        print("EVALUACIÓN EN TEST SET")
        print("="*70)
        
        print("\n📊 TAREA 1: Pertenencia a AFD (Y1)")
        print("-" * 70)
        acc_y1 = accuracy_score(all_y1_true, all_y1_pred)
        f1_y1 = f1_score(all_y1_true, all_y1_pred, average='binary')
        print(f"  Accuracy: {acc_y1:.4f}")
        print(f"  F1 Score: {f1_y1:.4f}")
        
        if acc_y1 >= 0.95 and f1_y1 >= 0.95:
            print("  ✅ Rendimiento: MUY BUENO")
        elif acc_y1 >= 0.90 and f1_y1 >= 0.90:
            print("  ✔️  Rendimiento: BUENO")
        elif acc_y1 >= 0.85:
            print("  ⚠️  Rendimiento: REGULAR")
        else:
            print("  ❌ Rendimiento: MALO")
        
        print("\n📊 TAREA 2: Cadena compartida entre AFDs (Y2)")
        print("-" * 70)
        acc_y2 = accuracy_score(all_y2_true, all_y2_pred)
        f1_y2 = f1_score(all_y2_true, all_y2_pred, average='binary', zero_division=0)
        
        precision, recall, _ = precision_recall_curve(all_y2_true, all_y2_scores)
        pr_auc_y2 = auc(recall, precision)
        
        print(f"  Accuracy: {acc_y2:.4f}")
        print(f"  F1 Score: {f1_y2:.4f}")
        print(f"  PR-AUC:   {pr_auc_y2:.4f}")
        
        if f1_y2 >= 0.9 and pr_auc_y2 >= 0.9:
            print("  ✅ Rendimiento: BUENO")
        elif f1_y2 >= 0.8 and pr_auc_y2 >= 0.8:
            print("  ⚠️  Rendimiento: REGULAR")
        else:
            print("  ❌ Rendimiento: MALO")
        
        print("\n" + "="*70)
        
        return {
            'y1_accuracy': acc_y1,
            'y1_f1': f1_y1,
            'y2_accuracy': acc_y2,
            'y2_f1': f1_y2,
            'y2_pr_auc': pr_auc_y2
        }


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def plot_training_history(history):
    """Visualiza el historial de entrenamiento"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc_y1'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc_y1'], label='Val', linewidth=2)
    axes[1].set_title('Y1 Accuracy (Pertenencia)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['train_acc_y2'], label='Train', linewidth=2)
    axes[2].plot(history['val_acc_y2'], label='Val', linewidth=2)
    axes[2].set_title('Y2 Accuracy (Compartida)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("📈 Gráficas guardadas en 'training_history.png'")
    plt.show()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Pipeline completo de entrenamiento"""
    
    print("="*70)
    print("🤖 ENTRENAMIENTO DUAL-ENCODER PARA AFDs")
    print("="*70)
    print("\nEste script entrena el modelo usando el dataset pre-generado.")
    print("Optimizado para ejecutarse en GPU (T4 en Google Colab).\n")
    
    # Verificar CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  GPU no disponible, usando CPU (será más lento)")
    print()
    
    # 1. Cargar AFDs (para features)
    print("1️⃣  Cargando AFDs...")
    csv_afd_path = '/content/sample_data/dataset6000.csv'  # Ajusta para tu entorno
    
    try:
        parser = AFDParser(csv_afd_path)
        print(f"   ✓ {len(parser.df)} AFDs cargados\n")
    except FileNotFoundError:
        print(f"   ⚠️  No se encontró {csv_afd_path}, intentando ruta local...")
        csv_afd_path = 'dataset6000.csv'
        try:
            parser = AFDParser(csv_afd_path)
            print(f"   ✓ {len(parser.df)} AFDs cargados\n")
        except FileNotFoundError:
            print(f"   ❌ ERROR: No se encontró dataset6000.csv")
            print(f"   Por favor, coloca dataset6000.csv en la carpeta actual.")
            return
    
    # 2. Cargar dataset generado
    print("2️⃣  Cargando dataset generado...")
    dataset_path = '/content/sample_data/dataset_generated.csv'
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"   ✓ {len(df):,} ejemplos cargados\n")
    except FileNotFoundError:
        print(f"   ⚠️  No se encontró {dataset_path}, intentando ruta local...")
        dataset_path = 'dataset_generated.csv'
        try:
            df = pd.read_csv(dataset_path)
            print(f"   ✓ {len(df):,} ejemplos cargados\n")
        except FileNotFoundError:
            print(f"   ❌ ERROR: No se encontró dataset_generated.csv")
            print(f"   Primero ejecuta gen_dataset_exhaustive.py para generar el dataset.")
            return
    
    # 3. Split por DFA
    print("3️⃣  Dividiendo dataset...")
    unique_dfas = df['dfa_id'].unique()
    train_dfas, temp_dfas = train_test_split(unique_dfas, test_size=0.3, random_state=42)
    val_dfas, test_dfas = train_test_split(temp_dfas, test_size=0.5, random_state=42)
    
    train_df = df[df['dfa_id'].isin(train_dfas)]
    val_df = df[df['dfa_id'].isin(val_dfas)]
    test_df = df[df['dfa_id'].isin(test_dfas)]
    
    print(f"   Train: {len(train_df):,} ejemplos ({len(train_dfas)} AFDs)")
    print(f"   Val:   {len(val_df):,} ejemplos ({len(val_dfas)} AFDs)")
    print(f"   Test:  {len(test_df):,} ejemplos ({len(test_dfas)} AFDs)\n")
    
    # 4. Dataloaders
    print("4️⃣  Creando dataloaders...")
    batch_size = TRAIN_CONFIG['batch_size']
    
    train_dataset = AFDStringDataset(train_df, parser)
    val_dataset = AFDStringDataset(val_df, parser)
    test_dataset = AFDStringDataset(test_df, parser)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)
    print(f"   ✓ Batch size: {batch_size}\n")
    
    # 5. Crear modelo
    print("5️⃣  Creando modelo...")
    model = DualEncoderModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parámetros: {num_params:,}\n")
    
    # 6. Entrenar
    print("6️⃣  Entrenando modelo...")
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        lambda1=TRAIN_CONFIG['lambda1'],
        lambda2=TRAIN_CONFIG['lambda2'],
        device=device,
        lr=0.001,
        label_smoothing=TRAIN_CONFIG['label_smoothing'],
        early_stop_patience=TRAIN_CONFIG['early_stop_patience'],
        early_stop_min_delta=TRAIN_CONFIG['early_stop_min_delta'],
        train_df=train_df  # Pasar train_df para calcular pos_weight
    )
    trainer.train(num_epochs=TRAIN_CONFIG['num_epochs'])
    print("\n✅ Entrenamiento completado!\n")
    
    # 7. Calibrar umbrales y evaluar
    print("7️⃣  Calibrando umbrales y evaluando en test set...")
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    thresholds = calibrate_thresholds(model, val_loader, device=device)
    evaluator = Evaluator(model, test_loader, device=device)
    metrics = evaluator.evaluate(thresholds=thresholds)
    print()
    
    # 8. Visualizar
    print("8️⃣  Generando visualizaciones...")
    plot_training_history(trainer.history)
    print()
    
    # Resumen final
    print("="*70)
    print("✅ PIPELINE DE ENTRENAMIENTO COMPLETO")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   - best_model.pt (modelo entrenado)")
    print("   - thresholds.json (umbrales calibrados)")
    print("   - training_history.png (gráficas)")
    print("\n💡 Para usar en producción, copia estos archivos junto con dataset6000.csv")
    print("="*70)


if __name__ == "__main__":
    main()

