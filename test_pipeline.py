"""
Script de prueba rápida del pipeline
Verifica que todos los componentes funcionen correctamente
"""

from acepten import AFDParser, StringDatasetGenerator, AFDStringDataset, DualEncoderModel
import torch

def test_parser():
    """Prueba el parser de AFDs"""
    print("🧪 Test 1: Parser de AFDs")
    parser = AFDParser('dataset6000.csv')
    
    # Probar parseo del primer AFD
    afd_features = parser.get_afd_features(0)
    print(f"   ✓ Features extraídos: shape {afd_features.shape}")
    assert afd_features.shape == (3104,), "Error: shape incorrecto"
    
    # Probar simulación
    result = parser.simulate_afd(0, "C")
    print(f"   ✓ Simulación AFD 0 con 'C': {result}")
    print()

def test_generator():
    """Prueba el generador de dataset"""
    print("🧪 Test 2: Generador de Dataset")
    parser = AFDParser('dataset6000.csv')
    generator = StringDatasetGenerator(parser)
    
    # Generar muestras para los primeros 10 AFDs
    df = generator.generate_full_dataset(pos_samples_per_dfa=5, neg_samples_per_dfa=5)
    print(f"   ✓ Dataset generado: {len(df)} ejemplos")
    print(f"   ✓ Columnas: {list(df.columns)}")
    
    # Calcular y2
    df = generator.compute_shared_label(df)
    print(f"   ✓ Y2 calculado: {df['y2'].sum()} cadenas compartidas")
    print()
    
    return df, parser

def test_dataset():
    """Prueba el Dataset de PyTorch"""
    print("🧪 Test 3: PyTorch Dataset")
    df, parser = test_generator()
    
    dataset = AFDStringDataset(df, parser)
    print(f"   ✓ Dataset creado: {len(dataset)} ejemplos")
    
    # Probar __getitem__
    sample = dataset[0]
    print(f"   ✓ Sample 0:")
    print(f"      - afd_features: {sample['afd_features'].shape}")
    print(f"      - string_tokens: {sample['string_tokens'].shape}")
    print(f"      - string_length: {sample['string_length']}")
    print(f"      - y1: {sample['y1']}")
    print(f"      - y2: {sample['y2']}")
    print()

def test_model():
    """Prueba el modelo dual-encoder"""
    print("🧪 Test 4: Modelo Dual-Encoder")
    
    model = DualEncoderModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Modelo creado: {num_params:,} parámetros")
    
    # Crear batch dummy
    batch_size = 4
    max_len = 10
    
    string_tokens = torch.randint(0, 12, (batch_size, max_len))
    string_lengths = torch.randint(1, max_len + 1, (batch_size,))
    afd_features = torch.randn(batch_size, 3104)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        y1_hat, y2_hat = model(string_tokens, string_lengths, afd_features)
    
    print(f"   ✓ Forward pass exitoso:")
    print(f"      - y1_hat: {y1_hat.shape}")
    print(f"      - y2_hat: {y2_hat.shape}")
    print(f"      - y1_hat range: [{y1_hat.min():.3f}, {y1_hat.max():.3f}]")
    print(f"      - y2_hat range: [{y2_hat.min():.3f}, {y2_hat.max():.3f}]")
    print()

def main():
    print("="*70)
    print("PRUEBAS DE PIPELINE")
    print("="*70)
    print()
    
    try:
        test_parser()
        test_generator()
        test_dataset()
        test_model()
        
        print("="*70)
        print("✅ TODAS LAS PRUEBAS PASARON")
        print("="*70)
        print("\n🚀 Puedes ejecutar el entrenamiento completo con:")
        print("   python acepten.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

