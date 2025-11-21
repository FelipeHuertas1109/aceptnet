"""
Script para Comparar y Visualizar Resultados del Modelo
Genera visualizaciones detalladas de rendimiento
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

sns.set_style("whitegrid")
sns.set_palette("husl")


def visualizar_distribucion_dataset(csv_path='result/dataset_generated.csv'):
    """Visualiza la distribución del dataset"""
    print("📊 Generando visualizaciones del dataset...")
    
    df = pd.read_csv(csv_path)
    df['string_len'] = df['string'].apply(lambda x: 0 if x == '<EPS>' else len(str(x)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Análisis del Dataset Generado', fontsize=16, fontweight='bold')
    
    # 1. Distribución Y1
    axes[0, 0].pie(df['label'].value_counts(), 
                   labels=['No Pertenece (0)', 'Pertenece (1)'],
                   autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    axes[0, 0].set_title('Distribución Y1 (Pertenencia)')
    
    # 2. Distribución Y2
    axes[0, 1].pie(df['y2'].value_counts(), 
                   labels=['No Compartida (0)', 'Compartida (1)'],
                   autopct='%1.1f%%', startangle=90, colors=['#ffcc99', '#99ff99'])
    axes[0, 1].set_title('Distribución Y2 (Compartida)')
    
    # 3. Histograma longitud de cadenas
    axes[0, 2].hist(df['string_len'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Longitud de Cadena')
    axes[0, 2].set_ylabel('Frecuencia')
    axes[0, 2].set_title('Distribución de Longitud de Cadenas')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Y1 por longitud
    length_groups = df.groupby('string_len')['label'].agg(['mean', 'count'])
    length_groups = length_groups[length_groups['count'] > 10]  # Filtrar grupos pequeños
    axes[1, 0].plot(length_groups.index, length_groups['mean'], marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Longitud de Cadena')
    axes[1, 0].set_ylabel('Proporción Y1=1')
    axes[1, 0].set_title('Y1 (Pertenencia) vs Longitud')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Ejemplos por AFD
    afd_counts = df['dfa_id'].value_counts().sort_values()
    axes[1, 1].hist(afd_counts, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Ejemplos por AFD')
    axes[1, 1].set_ylabel('Número de AFDs')
    axes[1, 1].set_title('Distribución de Ejemplos por AFD')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Correlación Y1 y Y2
    contingency = pd.crosstab(df['label'], df['y2'], normalize='all') * 100
    sns.heatmap(contingency, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 2],
                xticklabels=['No Comp.', 'Comp.'], yticklabels=['No Pert.', 'Pert.'])
    axes[1, 2].set_title('Correlación Y1 vs Y2 (%)')
    axes[1, 2].set_xlabel('Y2 (Compartida)')
    axes[1, 2].set_ylabel('Y1 (Pertenencia)')
    
    plt.tight_layout()
    plt.savefig('analisis_dataset.png', dpi=150, bbox_inches='tight')
    print("✅ Guardado en 'analisis_dataset.png'")
    plt.show()


def comparar_metricas_finales():
    """Muestra un resumen visual de las métricas finales"""
    print("\n📊 Generando resumen de métricas...")
    
    # Métricas del modelo (del entrenamiento)
    y1_accuracy = 0.8938
    y1_f1 = 0.8682
    y2_accuracy = 0.9887
    y2_f1 = 0.9924
    y2_pr_auc = 0.9997
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Métricas Finales del Modelo', fontsize=16, fontweight='bold')
    
    # 1. Comparación Y1
    metrics_y1 = ['Accuracy', 'F1 Score']
    values_y1 = [y1_accuracy, y1_f1]
    colors_y1 = ['#ff6b6b' if v < 0.90 else '#51cf66' for v in values_y1]
    
    bars1 = axes[0].bar(metrics_y1, values_y1, color=colors_y1, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0.90, color='orange', linestyle='--', label='Umbral Bueno (0.90)')
    axes[0].axhline(y=0.95, color='green', linestyle='--', label='Umbral Muy Bueno (0.95)')
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Score')
    axes[0].set_title('Tarea 1: Pertenencia a AFD (Y1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars1, values_y1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Comparación Y2
    metrics_y2 = ['Accuracy', 'F1 Score', 'PR-AUC']
    values_y2 = [y2_accuracy, y2_f1, y2_pr_auc]
    colors_y2 = ['#51cf66' if v >= 0.90 else '#ff6b6b' for v in values_y2]
    
    bars2 = axes[1].bar(metrics_y2, values_y2, color=colors_y2, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0.90, color='green', linestyle='--', label='Umbral Bueno (0.90)')
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Tarea 2: Cadena Compartida (Y2)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, values_y2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Radar chart comparativo
    categories = ['Y1\nAccuracy', 'Y1\nF1', 'Y2\nAccuracy', 'Y2\nF1', 'Y2\nPR-AUC']
    values = [y1_accuracy, y1_f1, y2_accuracy, y2_f1, y2_pr_auc]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax3 = plt.subplot(133, projection='polar')
    ax3.plot(angles, values, 'o-', linewidth=2, color='#4c72b0')
    ax3.fill(angles, values, alpha=0.25, color='#4c72b0')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=9)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.5, 0.7, 0.9, 1.0])
    ax3.set_yticklabels(['0.5', '0.7', '0.9', '1.0'], size=8)
    ax3.set_title('Vista Radar de Métricas', pad=20)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('metricas_finales.png', dpi=150, bbox_inches='tight')
    print("✅ Guardado en 'metricas_finales.png'")
    plt.show()


def mostrar_training_history(img_path='result/training_history.png'):
    """Muestra el historial de entrenamiento"""
    print("\n📈 Mostrando historial de entrenamiento...")
    
    try:
        img = Image.open(img_path)
        
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Historial de Entrenamiento (30 Épocas)', fontsize=14, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig('training_history_display.png', dpi=150, bbox_inches='tight')
        print("✅ Guardado en 'training_history_display.png'")
        plt.show()
    except Exception as e:
        print(f"⚠️  No se pudo cargar la imagen: {e}")


def generar_reporte_visual_completo():
    """Genera todas las visualizaciones"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "ANÁLISIS VISUAL COMPLETO" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # 1. Análisis del dataset
    visualizar_distribucion_dataset()
    
    # 2. Métricas finales
    comparar_metricas_finales()
    
    # 3. Historial de entrenamiento
    mostrar_training_history()
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS VISUAL COMPLETO")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   - analisis_dataset.png")
    print("   - metricas_finales.png")
    print("   - training_history_display.png")
    print()
    print("🎯 Resumen de Resultados:")
    print("   Tarea 1 (Pertenencia): Accuracy=0.8938, F1=0.8682 → REGULAR ⚠️")
    print("   Tarea 2 (Compartida):  F1=0.9924, PR-AUC=0.9997  → BUENO ✅")
    print()
    print("💡 El modelo es excelente en detectar cadenas compartidas,")
    print("   pero tiene margen de mejora en la tarea de pertenencia.")


if __name__ == "__main__":
    generar_reporte_visual_completo()

