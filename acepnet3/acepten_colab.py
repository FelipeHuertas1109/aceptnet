"""
Modelo Dual-Encoder para Google Colab con GPU
Versión optimizada para entrenamiento en CUDA

Instrucciones:
1. Sube dataset6000.csv a /content/sample_data/
2. Ejecuta este script completo
3. Los resultados se guardarán en /content/
"""

import re
import json
import pandas as pd
import numpy as np
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

# Constantes globales
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
MAX_STATES = 16
NUM_SYMBOLS = len(ALPHABET)
PAD_IDX = len(ALPHABET)

# Configuración base para controlar regularizaciones desde un solo lugar
TRAIN_CONFIG = {
    'pos_samples_per_dfa': 100,  # Máximo de positivos (exhaustivos)
    'neg_samples_per_dfa': 150,  # Máximo de negativos (exhaustivos)
    'max_string_length': 5,      # 🆕 Longitud máxima para generación exhaustiva
    'label_smoothing': 0.0,      # Sin smoothing: Y1 es lógica exacta (0 o 1)
    'lambda1': 1.0,              # Peso de Y1 (pertenencia)
    'lambda2': 0.3,              # Peso de Y2 (compartida) - menor prioridad
    'batch_size': 128,
    'num_epochs': 40,            # Máximo de épocas (puede parar antes)
    'early_stop_patience': 7,    # Espera 7 épocas sin mejora
    'early_stop_min_delta': 1e-4 # Mejora mínima requerida (0.0001)
}


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
    
    def simulate_afd(self, dfa_id: int, string: str) -> bool:
        """Simula el AFD para verificar si acepta una cadena"""
        row = self.df.iloc[dfa_id]
        
        if string == "<EPS>" or string == "":
            string = ""
            
        estados = self.parse_states(row['Estados'])
        accept_states = self.parse_accept_states(row['Estados de aceptación'])
        transitions = self.parse_transitions(row['Transiciones'], row['Alfabeto'])
        
        current_state = 0
        
        for char in string:
            if (current_state, char) not in transitions:
                return False
            current_state = transitions[(current_state, char)]
        
        return current_state in accept_states


class StringDatasetGenerator:
    """Generador de dataset de pares (dfa_id, string, label)"""
    
    def __init__(self, parser: AFDParser):
        self.parser = parser
        self.generated_cache = defaultdict(set)  # Cache para evitar duplicados
        
    def generate_exhaustive_strings(self, dfa_id: int, max_len: int = 5) -> Tuple[List[str], List[str]]:
        """
        Genera TODAS las cadenas hasta longitud max_len
        y las clasifica usando el simulador del AFD real.
        Esto produce el dataset perfecto sin ruido.
        """
        row = self.parser.df.iloc[dfa_id]
        alfabeto = row['Alfabeto'].split() if not pd.isna(row['Alfabeto']) else []
        
        positives = []
        negatives = []
        
        # Incluir cadena vacía
        if self.parser.simulate_afd(dfa_id, ""):
            positives.append("<EPS>")
        else:
            negatives.append("<EPS>")
        
        # Explorar exhaustivamente todas las combinaciones
        for length in range(1, max_len + 1):
            def backtrack(prefix, depth):
                if depth == length:
                    accepted = self.parser.simulate_afd(dfa_id, prefix)
                    if accepted:
                        positives.append(prefix)
                    else:
                        negatives.append(prefix)
                    return
                for sym in alfabeto:
                    backtrack(prefix + sym, depth + 1)
            
            backtrack("", 0)
        
        return positives, negatives
    
    def generate_full_dataset(self, pos_samples_per_dfa: int = 50,
                             neg_samples_per_dfa: int = 50,
                             max_string_length: int = 5) -> pd.DataFrame:
        """
        Genera dataset completo con pares (dfa_id, string, label)
        usando generación EXHAUSTIVA de todas las cadenas hasta max_string_length.
        
        Args:
            pos_samples_per_dfa: Máximo de muestras positivas por AFD
            neg_samples_per_dfa: Máximo de muestras negativas por AFD
            max_string_length: Longitud máxima de cadenas a generar exhaustivamente
        """
        data = []
        num_dfas = len(self.parser.df)
        
        print(f"Generando dataset desde {num_dfas} AFDs...")
        print(f"   🔬 Generación EXHAUSTIVA hasta longitud {max_string_length}")
        print(f"   ✨ Dataset SIN RUIDO: todas las cadenas simuladas con el AFD real")
        
        for dfa_id in tqdm(range(num_dfas), desc="Procesando AFDs"):
            # 🆕 Generar dataset exhaustivo hasta longitud especificada
            pos_strings, neg_strings = self.generate_exhaustive_strings(dfa_id, max_len=max_string_length)
            
            # Mezclar para variedad
            np.random.shuffle(pos_strings)
            np.random.shuffle(neg_strings)
            
            # Limitar tamaño por eficiencia
            pos_strings = pos_strings[:pos_samples_per_dfa]
            neg_strings = neg_strings[:neg_samples_per_dfa]
            
            # Agregar positivos al dataset
            for string in pos_strings:
                data.append({
                    'dfa_id': dfa_id,
                    'string': string,
                    'label': 1
                })
            
            # Agregar negativos al dataset
            for string in neg_strings:
                data.append({
                    'dfa_id': dfa_id,
                    'string': string,
                    'label': 0
                })
        
        df = pd.DataFrame(data)
        print(f"✓ Dataset generado: {len(df):,} ejemplos")
        
        # Estadísticas
        pos_count = (df['label'] == 1).sum()
        neg_count = (df['label'] == 0).sum()
        print(f"   📊 Positivos: {pos_count:,} | Negativos: {neg_count:,}")
        print(f"   📈 Promedio por AFD: {len(df) / num_dfas:.1f} ejemplos")
        print(f"   ✅ Dataset 100% preciso: cada etiqueta verificada por simulación")
        
        return df
    
    def compute_shared_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula y2: si una cadena es aceptada por múltiples AFDs"""
        string_counts = defaultdict(set)
        
        for _, row in df.iterrows():
            if row['label'] == 1:
                string_counts[row['string']].add(row['dfa_id'])
        
        df['y2'] = df['string'].apply(
            lambda s: 1 if len(string_counts.get(s, set())) >= 2 else 0
        )
        
        num_shared = df['y2'].sum()
        print(f"✓ Cadenas compartidas: {num_shared}/{len(df)} ({num_shared/len(df)*100:.1f}%)")
        
        return df


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


class Trainer:
    """Entrenador del modelo dual-task optimizado para GPU"""
    
    def __init__(self, model, train_loader, val_loader, 
                 lambda1=1.0, lambda2=1.0, lr=0.001, device='cuda',
                 label_smoothing: float = 0.0,
                 early_stop_patience: int = 5,
                 early_stop_min_delta: float = 1e-3):
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
        self.criterion = nn.BCELoss()
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
            
            loss1 = self.criterion(y1_hat, y1_targets)
            loss2 = self.criterion(y2_hat, y2_targets)
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
                
                loss1 = self.criterion(y1_hat, y1_targets)
                loss2 = self.criterion(y2_hat, y2_targets)
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


def main():
    """Pipeline completo optimizado para Colab"""
    
    print("="*70)
    print("🤖 MODELO DUAL-ENCODER PARA AFDs - GOOGLE COLAB")
    print("="*70)
    
    # Verificar CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  GPU no disponible, usando CPU")
    print()
    
    # 1. Cargar AFDs
    print("1️⃣  Cargando AFDs...")
    csv_path = '/content/sample_data/dataset6000.csv'
    parser = AFDParser(csv_path)
    print(f"   ✓ {len(parser.df)} AFDs cargados\n")
    
    # 2. Generar dataset
    print("2️⃣  Generando dataset...")
    generator = StringDatasetGenerator(parser)
    df = generator.generate_full_dataset(
        pos_samples_per_dfa=TRAIN_CONFIG['pos_samples_per_dfa'],
        neg_samples_per_dfa=TRAIN_CONFIG['neg_samples_per_dfa'],
        max_string_length=TRAIN_CONFIG['max_string_length']
    )
    print()
    
    # 3. Calcular y2
    print("3️⃣  Calculando Y2...")
    df = generator.compute_shared_label(df)
    df.to_csv('dataset_generated.csv', index=False)
    print(f"   ✓ Guardado en 'dataset_generated.csv'\n")
    
    # 4. Split
    print("4️⃣  Dividiendo dataset...")
    unique_dfas = df['dfa_id'].unique()
    train_dfas, temp_dfas = train_test_split(unique_dfas, test_size=0.3, random_state=42)
    val_dfas, test_dfas = train_test_split(temp_dfas, test_size=0.5, random_state=42)
    
    train_df = df[df['dfa_id'].isin(train_dfas)]
    val_df = df[df['dfa_id'].isin(val_dfas)]
    test_df = df[df['dfa_id'].isin(test_dfas)]
    
    print(f"   Train: {len(train_df):,} ejemplos ({len(train_dfas)} AFDs)")
    print(f"   Val:   {len(val_df):,} ejemplos ({len(val_dfas)} AFDs)")
    print(f"   Test:  {len(test_df):,} ejemplos ({len(test_dfas)} AFDs)\n")
    
    # 5. Dataloaders
    print("5️⃣  Creando dataloaders...")
    batch_size = TRAIN_CONFIG['batch_size']  # Mayor batch size para GPU
    
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
    
    # 6. Crear modelo
    print("6️⃣  Creando modelo...")
    model = DualEncoderModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parámetros: {num_params:,}\n")
    
    # 7. Entrenar
    print("7️⃣  Entrenando...")
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
        early_stop_min_delta=TRAIN_CONFIG['early_stop_min_delta']
    )
    trainer.train(num_epochs=TRAIN_CONFIG['num_epochs'])
    print("\n✅ Entrenamiento completado!\n")
    
    # 8. Calibrar umbrales y evaluar
    print("8️⃣  Calibrando umbrales y evaluando en test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    thresholds = calibrate_thresholds(model, val_loader, device=device)
    evaluator = Evaluator(model, test_loader, device=device)
    metrics = evaluator.evaluate(thresholds=thresholds)
    print()
    
    # 9. Visualizar
    print("9️⃣  Generando visualizaciones...")
    plot_training_history(trainer.history)
    print()
    
    print("="*70)
    print("✅ PIPELINE COMPLETO!")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   - best_model.pt")
    print("   - dataset_generated.csv")
    print("   - training_history.png")


if __name__ == "__main__":
    main()

