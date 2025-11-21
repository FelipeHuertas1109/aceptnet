"""
Test Rápido - Verifica que todo esté listo para Colab
Ejecuta esto localmente antes de subir a Colab
"""

import os
import sys

def check_files():
    """Verifica que todos los archivos necesarios existan"""
    print("="*70)
    print("🔍 VERIFICACIÓN DE ARCHIVOS")
    print("="*70)
    
    required_files = [
        ('dataset6000.csv', 'Dataset principal'),
        ('acepten_colab.py', 'Script para Colab'),
        ('COLAB_INSTRUCTIONS.md', 'Instrucciones'),
        ('RESUMEN.md', 'Resumen del proyecto')
    ]
    
    all_ok = True
    for filename, description in required_files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"✅ {filename:25s} - {description:30s} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename:25s} - FALTA!")
            all_ok = False
    
    print()
    return all_ok


def check_imports():
    """Verifica que las librerías estén instaladas"""
    print("="*70)
    print("📦 VERIFICACIÓN DE DEPENDENCIAS")
    print("="*70)
    
    packages = [
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:20s} - Instalado")
        except ImportError:
            print(f"❌ {name:20s} - NO instalado")
            all_ok = False
    
    print()
    return all_ok


def test_dataset():
    """Prueba básica del dataset"""
    print("="*70)
    print("🧪 TEST DEL DATASET")
    print("="*70)
    
    try:
        import pandas as pd
        
        if not os.path.exists('dataset6000.csv'):
            print("❌ dataset6000.csv no encontrado")
            return False
        
        df = pd.read_csv('dataset6000.csv')
        
        print(f"✅ Dataset cargado: {len(df)} filas")
        print(f"✅ Columnas: {list(df.columns)}")
        
        # Verificar columnas necesarias
        required_cols = ['Alfabeto', 'Estados', 'Estados de aceptación', 'Transiciones', 'Clase']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"⚠️  Columnas faltantes: {missing}")
            return False
        
        print(f"✅ Todas las columnas necesarias presentes")
        print(f"✅ Primer AFD:")
        print(f"   - Regex: {df.iloc[0]['Regex'][:50]}...")
        print(f"   - Alfabeto: {df.iloc[0]['Alfabeto']}")
        print(f"   - Estados: {df.iloc[0]['Estados']}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error al leer dataset: {e}")
        return False


def test_parser():
    """Prueba el parser de AFDs"""
    print("="*70)
    print("🧪 TEST DEL PARSER")
    print("="*70)
    
    try:
        from acepten import AFDParser
        
        parser = AFDParser('dataset6000.csv')
        
        # Test 1: Extracción de features
        features = parser.get_afd_features(0)
        print(f"✅ Features extraídos: shape {features.shape}")
        
        if features.shape != (3104,):
            print(f"⚠️  Shape incorrecto: esperado (3104,), obtenido {features.shape}")
            return False
        
        # Test 2: Simulación
        result = parser.simulate_afd(0, "C")
        print(f"✅ Simulación AFD 0 con 'C': {result}")
        
        # Test 3: Cache
        features2 = parser.get_afd_features(0)
        print(f"✅ Cache funcionando: {len(parser.afd_cache)} AFDs cacheados")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error en parser: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Prueba el modelo"""
    print("="*70)
    print("🧪 TEST DEL MODELO")
    print("="*70)
    
    try:
        import torch
        from acepten import DualEncoderModel
        
        model = DualEncoderModel()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Modelo creado: {num_params:,} parámetros")
        
        # Test forward pass
        batch_size = 4
        max_len = 10
        
        string_tokens = torch.randint(0, 12, (batch_size, max_len))
        string_lengths = torch.randint(1, max_len + 1, (batch_size,))
        afd_features = torch.randn(batch_size, 3104)
        
        model.eval()
        with torch.no_grad():
            y1_hat, y2_hat = model(string_tokens, string_lengths, afd_features)
        
        print(f"✅ Forward pass exitoso")
        print(f"   - y1_hat shape: {y1_hat.shape}")
        print(f"   - y2_hat shape: {y2_hat.shape}")
        print(f"   - y1_hat range: [{y1_hat.min():.3f}, {y1_hat.max():.3f}]")
        print(f"   - y2_hat range: [{y2_hat.min():.3f}, {y2_hat.max():.3f}]")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error en modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_colab_instructions():
    """Imprime instrucciones finales"""
    print("="*70)
    print("📋 PASOS PARA COLAB")
    print("="*70)
    print()
    print("1️⃣  Ve a: https://colab.research.google.com/")
    print("2️⃣  Crea un nuevo notebook")
    print("3️⃣  Activa GPU: Runtime → Change runtime type → GPU")
    print()
    print("4️⃣  Sube estos archivos:")
    print("    - dataset6000.csv  (a /content/sample_data/)")
    print("    - acepten_colab.py (a /content/)")
    print()
    print("5️⃣  Ejecuta: !python acepten_colab.py")
    print()
    print("📁 Consulta COLAB_INSTRUCTIONS.md para más detalles")
    print()


def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "TEST PRE-COLAB VERIFICATION" + " "*26 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    results = {
        'Archivos': check_files(),
        'Dependencias': check_imports(),
        'Dataset': test_dataset(),
        'Parser': test_parser(),
        'Modelo': test_model()
    }
    
    print("="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    print()
    
    if all(results.values()):
        print("🎉 ¡TODO LISTO PARA COLAB!")
        print()
        print_colab_instructions()
        print("="*70)
        return 0
    else:
        print("⚠️  Hay problemas que resolver antes de ir a Colab")
        print()
        failed = [name for name, passed in results.items() if not passed]
        print(f"Tests fallidos: {', '.join(failed)}")
        print()
        print("Revisa los mensajes de error arriba")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

