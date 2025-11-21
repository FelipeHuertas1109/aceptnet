"""
Script de Resumen Rápido de Resultados
Ejecuta esto primero para ver un overview completo
"""

import os


def print_header(text):
    """Imprime un header bonito"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def verificar_archivos():
    """Verifica que los archivos de resultado existan"""
    print_header("📁 VERIFICACIÓN DE ARCHIVOS")
    
    archivos_esperados = {
        'result/best_model.pt': 'Modelo entrenado',
        'result/dataset_generated.csv': 'Dataset generado',
        'result/training_history.png': 'Gráficas de entrenamiento',
        'dataset6000.csv': 'Dataset original de AFDs'
    }
    
    todos_ok = True
    for archivo, desc in archivos_esperados.items():
        if os.path.exists(archivo):
            size_mb = os.path.getsize(archivo) / (1024 * 1024)
            print(f"✅ {archivo:35s} - {desc:25s} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {archivo:35s} - FALTA")
            todos_ok = False
    
    return todos_ok


def mostrar_metricas():
    """Muestra las métricas finales del entrenamiento"""
    print_header("📊 MÉTRICAS FINALES (TEST SET)")
    
    print("\n🎯 TAREA 1: Pertenencia a AFD (Y1)")
    print("-" * 70)
    print("  ┌─────────────┬──────────┬────────────────────┐")
    print("  │   Métrica   │  Valor   │   Rendimiento      │")
    print("  ├─────────────┼──────────┼────────────────────┤")
    print("  │  Accuracy   │  0.8938  │  ⚠️  REGULAR       │")
    print("  │  F1 Score   │  0.8682  │  ⚠️  REGULAR       │")
    print("  └─────────────┴──────────┴────────────────────┘")
    
    print("\n💫 TAREA 2: Cadena Compartida (Y2)")
    print("-" * 70)
    print("  ┌─────────────┬──────────┬────────────────────┐")
    print("  │   Métrica   │  Valor   │   Rendimiento      │")
    print("  ├─────────────┼──────────┼────────────────────┤")
    print("  │  Accuracy   │  0.9887  │  ✅ MUY BUENO      │")
    print("  │  F1 Score   │  0.9924  │  ✅ MUY BUENO      │")
    print("  │  PR-AUC     │  0.9997  │  ✅ EXCELENTE      │")
    print("  └─────────────┴──────────┴────────────────────┘")


def mostrar_resumen_entrenamiento():
    """Muestra resumen del proceso de entrenamiento"""
    print_header("🚀 RESUMEN DEL ENTRENAMIENTO")
    
    print("\n📌 Configuración:")
    print("  • Épocas: 30")
    print("  • Batch size: 128")
    print("  • Device: CUDA (Tesla T4)")
    print("  • Parámetros: 1,918,114")
    print("  • Optimizador: Adam (lr=0.001)")
    
    print("\n📊 Dataset:")
    print("  • Total ejemplos: 253,751")
    print("  • Train: 178,547 (4200 AFDs)")
    print("  • Val: 37,698 (900 AFDs)")
    print("  • Test: 37,506 (900 AFDs)")
    print("  • Cadenas compartidas: 73.9%")
    
    print("\n📈 Progreso durante entrenamiento:")
    print("  Época  1: Train Y1=0.8557, Val Y1=0.8870")
    print("  Época 10: Train Y1=0.9047, Val Y1=0.8895")
    print("  Época 20: Train Y1=0.9287, Val Y1=0.8887")
    print("  Época 30: Train Y1=0.9379, Val Y1=0.8859")
    
    print("\n  ✅ Modelo estable - Sin overfitting visible")
    print("  ⚠️  Val accuracy se estancó ~época 7 (early stopping hubiera ayudado)")


def mostrar_interpretacion():
    """Muestra interpretación de los resultados"""
    print_header("🔍 INTERPRETACIÓN DE RESULTADOS")
    
    print("\n✅ FORTALEZAS:")
    print("  • Excelente en detectar cadenas compartidas (Y2: 99.24% F1)")
    print("  • PR-AUC casi perfecto (99.97%)")
    print("  • Modelo generaliza bien a AFDs nuevos")
    print("  • Sin overfitting (train/val loss coherentes)")
    
    print("\n⚠️  ÁREAS DE MEJORA:")
    print("  • Y1 accuracy está en 89.38% (objetivo: ≥90%)")
    print("  • F1 de Y1 podría mejorar (86.82% → 90%)")
    print("  • Validación se estancó temprano")
    
    print("\n💡 POSIBLES CAUSAS:")
    print("  1. Dataset desbalanceado en algunos AFDs")
    print("  2. Complejidad variable de autómatas")
    print("  3. Cadenas muy cortas difíciles de clasificar")
    print("  4. Modelo necesita más capacidad o épocas")


def mostrar_siguientes_pasos():
    """Muestra qué hacer ahora"""
    print_header("🎯 PRÓXIMOS PASOS")
    
    print("\n1️⃣  ANALIZAR EN DETALLE:")
    print("    python analizar_resultados.py")
    print("    → Genera reporte completo con ejemplos y errores")
    
    print("\n2️⃣  VER VISUALIZACIONES:")
    print("    python comparar_resultados.py")
    print("    → Gráficas detalladas del dataset y métricas")
    
    print("\n3️⃣  PROBAR EL MODELO:")
    print("    python inferencia_interactiva.py")
    print("    → Demo rápido con ejemplos predefinidos")
    print()
    print("    python inferencia_interactiva.py --interactivo")
    print("    → Modo interactivo completo con menú")
    
    print("\n4️⃣  MEJORAR EL MODELO (opcional):")
    print("    • Entrenar más épocas (50-100)")
    print("    • Aumentar data (50 samples/AFD)")
    print("    • Probar arquitecturas más grandes")
    print("    • Implementar early stopping")


def mostrar_ejemplos_uso():
    """Muestra ejemplos de código para usar el modelo"""
    print_header("💻 EJEMPLOS DE USO")
    
    print("\n📝 Hacer una predicción:")
    print('''
from inferencia_interactiva import Predictor

# Cargar modelo
predictor = Predictor()

# Predecir
result = predictor.predecir(dfa_id=0, string="ABC")

print(f"Pertenece: {result['y1_pred']}")
print(f"Probabilidad: {result['y1_prob']:.2%}")
print(f"Compartida: {result['y2_pred']}")
''')
    
    print("\n📝 Probar múltiples cadenas:")
    print('''
predictor = Predictor()

# Ver info del AFD
predictor.mostrar_info_afd(dfa_id=0)

# Probar varias cadenas
cadenas = ["C", "CG", "CC", "ABC", "<EPS>"]
predictor.test_multiples_cadenas(dfa_id=0, cadenas=cadenas)
''')


def main():
    """Función principal"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "RESUMEN DE RESULTADOS DEL MODELO" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    # 1. Verificar archivos
    if not verificar_archivos():
        print("\n⚠️  Algunos archivos faltan. Asegúrate de haber copiado todo desde Colab.")
        return
    
    # 2. Mostrar métricas
    mostrar_metricas()
    
    # 3. Resumen del entrenamiento
    mostrar_resumen_entrenamiento()
    
    # 4. Interpretación
    mostrar_interpretacion()
    
    # 5. Siguientes pasos
    mostrar_siguientes_pasos()
    
    # 6. Ejemplos
    mostrar_ejemplos_uso()
    
    # Footer
    print("\n" + "="*70)
    print("✅ RESUMEN COMPLETO")
    print("="*70)
    print("\n📖 Para más detalles, lee: COMO_USAR_RESULTADOS.md")
    print("\n🎉 ¡Felicitaciones por entrenar tu modelo!")
    print()


if __name__ == "__main__":
    main()

