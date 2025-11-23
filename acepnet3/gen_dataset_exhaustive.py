"""
Generador Exhaustivo de Dataset para AFDs (Solo CPU)
=====================================================

Este script se ejecuta UNA SOLA VEZ para generar el dataset completo.
No requiere GPU y puede ejecutarse en cualquier máquina.

Instrucciones para Google Colab:
1. Sube dataset6000.csv a /content/sample_data/
2. Sube este script a /content/
3. Ejecuta: !python gen_dataset_exhaustive.py

Instrucciones para PC local:
1. Coloca dataset6000.csv en la misma carpeta que este script
2. Ejecuta: python gen_dataset_exhaustive.py

Salida:
- dataset_generated.csv: Dataset completo listo para entrenar
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
MAX_STATES = 16
NUM_SYMBOLS = len(ALPHABET)

# Configuración de generación
GEN_CONFIG = {
    'pos_samples_per_dfa': 100,  # Máximo de positivos por AFD
    'neg_samples_per_dfa': 150,  # Máximo de negativos por AFD
    'max_string_length': 5,      # Longitud máxima de cadenas
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
        Extrae representación vectorial del AFD.
        Nota: Esta función no se usa en generación, pero se mantiene 
        por compatibilidad con el script de entrenamiento.
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


# ============================================================================
# GENERADOR DE DATASET
# ============================================================================

class StringDatasetGenerator:
    """Generador exhaustivo de dataset de pares (dfa_id, string, label)"""
    
    def __init__(self, parser: AFDParser):
        self.parser = parser
        self.generated_cache = defaultdict(set)
        
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
            # Generar dataset exhaustivo hasta longitud especificada
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
        print("Calculando etiqueta compartida (Y2)...")
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


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Pipeline de generación de dataset"""
    
    print("="*70)
    print("🧱 GENERADOR EXHAUSTIVO DE DATASET PARA AFDs")
    print("="*70)
    print("\nEste script genera el dataset completo en CPU.")
    print("No requiere GPU y se ejecuta una sola vez.\n")
    
    # 1. Cargar AFDs
    print("1️⃣  Cargando AFDs...")
    csv_path = '/content/sample_data/dataset6000.csv'  # Ruta para Colab
    
    try:
        parser = AFDParser(csv_path)
        print(f"   ✓ {len(parser.df)} AFDs cargados desde Colab\n")
    except FileNotFoundError:
        print(f"   ⚠️  No se encontró en Colab, intentando ruta local...")
        csv_path = 'dataset6000.csv'
        try:
            parser = AFDParser(csv_path)
            print(f"   ✓ {len(parser.df)} AFDs cargados desde ruta local\n")
        except FileNotFoundError:
            print(f"   ❌ ERROR: No se encontró dataset6000.csv")
            print(f"   Por favor, coloca dataset6000.csv en /content/sample_data/ (Colab)")
            print(f"   o en la carpeta actual (local).")
            return
    
    # 2. Generar dataset exhaustivo
    print("2️⃣  Generando dataset exhaustivo...")
    generator = StringDatasetGenerator(parser)
    df = generator.generate_full_dataset(
        pos_samples_per_dfa=GEN_CONFIG['pos_samples_per_dfa'],
        neg_samples_per_dfa=GEN_CONFIG['neg_samples_per_dfa'],
        max_string_length=GEN_CONFIG['max_string_length']
    )
    print()
    
    # 3. Calcular etiqueta compartida (Y2)
    print("3️⃣  Calculando Y2 (etiqueta compartida)...")
    df = generator.compute_shared_label(df)
    print()
    
    # 4. Guardar dataset
    print("4️⃣  Guardando dataset...")
    output_path = 'dataset_generated.csv'
    df.to_csv(output_path, index=False)
    print(f"   ✓ Guardado en '{output_path}'\n")
    
    # Resumen final
    print("="*70)
    print("✅ GENERACIÓN COMPLETA")
    print("="*70)
    print(f"\n📁 Archivo generado: {output_path}")
    print(f"📊 Total de ejemplos: {len(df):,}")
    print(f"📈 Columnas: dfa_id, string, label, y2")
    print(f"\n💡 Próximo paso: usa train_dual_encoder.py para entrenar en GPU")
    print("="*70)


if __name__ == "__main__":
    main()

