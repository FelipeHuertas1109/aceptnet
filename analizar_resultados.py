"""
Script de Análisis de Resultados del Modelo
Analiza el modelo entrenado y genera reportes detallados
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from acepten import AFDParser, DualEncoderModel, AFDStringDataset, collate_fn
from torch.utils.data import DataLoader

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def cargar_modelo(model_path='result/best_model.pt'):
    """Carga el modelo entrenado"""
    print("🔄 Cargando modelo...")
    model = DualEncoderModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"✅ Modelo cargado: {sum(p.numel() for p in model.parameters()):,} parámetros")
    return model


def analizar_dataset(csv_path='result/dataset_generated.csv'):
    """Analiza estadísticas del dataset generado"""
    print("\n" + "="*70)
    print("📊 ANÁLISIS DEL DATASET")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    print(f"\nTotal de ejemplos: {len(df):,}")
    print(f"\nDistribución de labels:")
    print(f"  Y1 (pertenencia):")
    print(f"    - Positivos (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"    - Negativos (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
    
    print(f"\n  Y2 (compartida):")
    print(f"    - Compartidas (1): {(df['y2']==1).sum():,} ({(df['y2']==1).mean()*100:.1f}%)")
    print(f"    - Únicas (0): {(df['y2']==0).sum():,} ({(df['y2']==0).mean()*100:.1f}%)")
    
    # Análisis de longitud de cadenas
    df['string_len'] = df['string'].apply(lambda x: 0 if x == '<EPS>' else len(str(x)))
    
    print(f"\nLongitud de cadenas:")
    print(f"  - Promedio: {df['string_len'].mean():.2f}")
    print(f"  - Mediana: {df['string_len'].median():.0f}")
    print(f"  - Mín: {df['string_len'].min():.0f}")
    print(f"  - Máx: {df['string_len'].max():.0f}")
    print(f"  - Cadenas vacías: {(df['string_len']==0).sum():,}")
    
    # Análisis por AFD
    print(f"\nAFDs únicos: {df['dfa_id'].nunique():,}")
    print(f"Ejemplos por AFD: {len(df) / df['dfa_id'].nunique():.1f} promedio")
    
    # Top cadenas más comunes
    print(f"\n🔝 Top 10 cadenas más frecuentes:")
    top_strings = df['string'].value_counts().head(10)
    for i, (string, count) in enumerate(top_strings.items(), 1):
        display_str = string if len(str(string)) <= 20 else str(string)[:17] + "..."
        print(f"  {i:2d}. '{display_str}': {count:,} veces")
    
    return df


def predecir_ejemplos(model, parser, df, num_ejemplos=20):
    """Hace predicciones en ejemplos aleatorios del dataset"""
    print("\n" + "="*70)
    print("🎯 PREDICCIONES EN EJEMPLOS ALEATORIOS")
    print("="*70)
    
    # Seleccionar ejemplos aleatorios
    samples = df.sample(n=num_ejemplos)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    results = []
    
    for _, row in samples.iterrows():
        # Preparar input
        afd_features = torch.tensor(parser.get_afd_features(row['dfa_id']), 
                                    dtype=torch.float32).unsqueeze(0).to(device)
        
        string = row['string']
        if string == '<EPS>' or string == '':
            tokens = []
        else:
            from acepten import CHAR_TO_IDX
            tokens = [CHAR_TO_IDX.get(c, 12) for c in str(string) if c in CHAR_TO_IDX]
        
        if len(tokens) == 0:
            string_tokens = torch.zeros((1, 1), dtype=torch.long).to(device)
            string_lengths = torch.tensor([0], dtype=torch.long).to(device)
        else:
            string_tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            string_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
        
        # Predecir
        with torch.no_grad():
            y1_pred, y2_pred = model(string_tokens, string_lengths, afd_features)
            y1_pred = y1_pred.item()
            y2_pred = y2_pred.item()
        
        # Guardar resultado
        result = {
            'dfa_id': row['dfa_id'],
            'string': string,
            'y1_true': row['label'],
            'y1_pred': y1_pred,
            'y1_correct': (y1_pred > 0.5) == row['label'],
            'y2_true': row['y2'],
            'y2_pred': y2_pred,
            'y2_correct': (y2_pred > 0.5) == row['y2']
        }
        results.append(result)
    
    # Mostrar resultados
    results_df = pd.DataFrame(results)
    
    print(f"\n✅ Accuracy Y1: {results_df['y1_correct'].mean()*100:.1f}%")
    print(f"✅ Accuracy Y2: {results_df['y2_correct'].mean()*100:.1f}%")
    
    print("\n📋 Ejemplos de predicciones:")
    print("-" * 70)
    
    for i, row in results_df.head(10).iterrows():
        display_str = row['string'] if len(str(row['string'])) <= 30 else str(row['string'])[:27] + "..."
        
        y1_icon = "✅" if row['y1_correct'] else "❌"
        y2_icon = "✅" if row['y2_correct'] else "❌"
        
        print(f"\nEjemplo {i+1}:")
        print(f"  AFD: {row['dfa_id']} | String: '{display_str}'")
        print(f"  {y1_icon} Y1: Real={row['y1_true']}, Pred={row['y1_pred']:.3f} ({'✓' if row['y1_correct'] else '✗'})")
        print(f"  {y2_icon} Y2: Real={row['y2_true']}, Pred={row['y2_pred']:.3f} ({'✓' if row['y2_correct'] else '✗'})")
    
    return results_df


def analizar_errores(model, parser, df, num_errores=10):
    """Analiza los errores más comunes del modelo"""
    print("\n" + "="*70)
    print("🔍 ANÁLISIS DE ERRORES")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Muestrear dataset
    sample_size = min(1000, len(df))
    samples = df.sample(n=sample_size, random_state=42)
    
    errors_y1 = []
    
    print(f"\nAnalizando {sample_size} ejemplos...")
    
    for _, row in samples.iterrows():
        # Preparar input
        afd_features = torch.tensor(parser.get_afd_features(row['dfa_id']), 
                                    dtype=torch.float32).unsqueeze(0).to(device)
        
        string = row['string']
        if string == '<EPS>' or string == '':
            tokens = []
        else:
            from acepten import CHAR_TO_IDX
            tokens = [CHAR_TO_IDX.get(c, 12) for c in str(string) if c in CHAR_TO_IDX]
        
        if len(tokens) == 0:
            string_tokens = torch.zeros((1, 1), dtype=torch.long).to(device)
            string_lengths = torch.tensor([0], dtype=torch.long).to(device)
        else:
            string_tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            string_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
        
        # Predecir
        with torch.no_grad():
            y1_pred, _ = model(string_tokens, string_lengths, afd_features)
            y1_pred = y1_pred.item()
        
        # Detectar errores en Y1
        y1_class = 1 if y1_pred > 0.5 else 0
        if y1_class != row['label']:
            errors_y1.append({
                'dfa_id': row['dfa_id'],
                'string': string,
                'string_len': len(tokens),
                'true': row['label'],
                'pred': y1_pred,
                'confidence': abs(y1_pred - 0.5)
            })
    
    print(f"\n❌ Errores encontrados en Y1: {len(errors_y1)} / {sample_size} ({len(errors_y1)/sample_size*100:.1f}%)")
    
    if errors_y1:
        errors_df = pd.DataFrame(errors_y1)
        
        print(f"\n🔍 Análisis de errores:")
        print(f"  - Confianza promedio en errores: {errors_df['confidence'].mean():.3f}")
        print(f"  - Longitud promedio de strings con error: {errors_df['string_len'].mean():.1f}")
        
        # Mostrar ejemplos de errores
        print(f"\n📋 Top {min(num_errores, len(errors_df))} errores más confiados (peores):")
        print("-" * 70)
        
        worst_errors = errors_df.nlargest(num_errores, 'confidence')
        
        for i, (_, row) in enumerate(worst_errors.iterrows(), 1):
            display_str = row['string'] if len(str(row['string'])) <= 30 else str(row['string'])[:27] + "..."
            print(f"\n{i}. AFD {row['dfa_id']} | String: '{display_str}'")
            print(f"   Real: {row['true']} | Pred: {row['pred']:.3f} | Confianza: {row['confidence']:.3f}")
    
    return errors_y1


def visualizar_training_history():
    """Muestra la imagen de entrenamiento guardada"""
    print("\n" + "="*70)
    print("📈 HISTORIAL DE ENTRENAMIENTO")
    print("="*70)
    
    try:
        img = plt.imread('result/training_history.png')
        plt.figure(figsize=(18, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('historial_detallado.png', dpi=150, bbox_inches='tight')
        print("\n✅ Visualización guardada en 'historial_detallado.png'")
        plt.show()
    except Exception as e:
        print(f"\n⚠️  No se pudo cargar training_history.png: {e}")


def generar_reporte_completo():
    """Genera un reporte completo del análisis"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "ANÁLISIS COMPLETO DE RESULTADOS" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    # 1. Cargar modelo
    model = cargar_modelo()
    
    # 2. Analizar dataset
    df = analizar_dataset()
    
    # 3. Inicializar parser
    print("\n🔄 Cargando parser de AFDs...")
    parser = AFDParser('dataset6000.csv')
    print("✅ Parser cargado")
    
    # 4. Predicciones en ejemplos
    results = predecir_ejemplos(model, parser, df, num_ejemplos=20)
    
    # 5. Análisis de errores
    errors = analizar_errores(model, parser, df, num_errores=10)
    
    # 6. Visualizar historial
    visualizar_training_history()
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   - historial_detallado.png")
    print("\n💡 Para hacer predicciones personalizadas, ejecuta:")
    print("   python inferencia_interactiva.py")


if __name__ == "__main__":
    generar_reporte_completo()

