"""
Script Interactivo para Hacer Predicciones con el Modelo
Permite testear el modelo con cadenas y AFDs personalizados
"""

import torch
import pandas as pd
from acepten import AFDParser, DualEncoderModel, CHAR_TO_IDX


class Predictor:
    """Clase para hacer predicciones con el modelo entrenado"""
    
    def __init__(self, model_path='result/best_model.pt', dataset_path='dataset6000.csv'):
        print("🔄 Cargando modelo y parser...")
        
        # Cargar modelo
        self.model = DualEncoderModel()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Cargar parser
        self.parser = AFDParser(dataset_path)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        print(f"✅ Listo! Device: {self.device}")
    
    def predecir(self, dfa_id: int, string: str):
        """
        Hace una predicción para una cadena y un AFD
        
        Args:
            dfa_id: ID del autómata (0-5999)
            string: Cadena a evaluar (puede ser vacía o '<EPS>')
        
        Returns:
            dict con predicciones y probabilidades
        """
        # Validar dfa_id
        if dfa_id < 0 or dfa_id >= len(self.parser.df):
            raise ValueError(f"dfa_id debe estar entre 0 y {len(self.parser.df)-1}")
        
        # Obtener información del AFD
        row = self.parser.df.iloc[dfa_id]
        afd_info = {
            'Regex': row['Regex'],
            'Alfabeto': row['Alfabeto'],
            'Estados': row['Estados'],
            'Aceptacion': row['Estados de aceptación'],
        }

        # 🔹 REGLA LÓGICA: Verificar alfabeto del AFD
        alfabeto = set(row['Alfabeto'].split())
        
        # Si la cadena tiene símbolos fuera del alfabeto, Y1 = 0 automáticamente
        alphabet_mismatch = False
        if string not in ("", "<EPS>"):
            for ch in string:
                if ch not in alfabeto:
                    alphabet_mismatch = True
                    break
        
        if alphabet_mismatch:
            # No pertenece por regla lógica (símbolos fuera del alfabeto)
            y1_prob = 0.0
            y1_pred = False
            y2_prob = 0.0
            y2_pred = False
        else:
            # Obtener features del AFD
            afd_features = torch.tensor(
                self.parser.get_afd_features(dfa_id),
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Tokenizar cadena
            if string == '<EPS>' or string == '':
                tokens = []
            else:
                tokens = [CHAR_TO_IDX.get(c, 12) for c in string if c in CHAR_TO_IDX]
            
            # Preparar tensors
            if len(tokens) == 0:
                string_tokens = torch.zeros((1, 1), dtype=torch.long).to(self.device)
                string_lengths = torch.tensor([0], dtype=torch.long).to(self.device)
            else:
                string_tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
                string_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(self.device)
            
            # Predecir con el modelo
            with torch.no_grad():
                y1_prob, y2_prob = self.model(string_tokens, string_lengths, afd_features)
                y1_prob = y1_prob.item()
                y2_prob = y2_prob.item()
            
            y1_pred = y1_prob > 0.5
            y2_pred = y2_prob > 0.5
        
        # Simular AFD para comparar (ground truth si es posible)
        try:
            ground_truth = self.parser.simulate_afd(dfa_id, string)
        except:
            ground_truth = None
        
        return {
            'dfa_id': dfa_id,
            'string': string,
            'y1_prob': y1_prob,
            'y1_pred': y1_pred,
            'y1_ground_truth': ground_truth,
            'y1_alphabet_mismatch': alphabet_mismatch,
            'y2_prob': y2_prob,
            'y2_pred': y2_pred,
            'afd_info': afd_info,
        }
    
    def mostrar_info_afd(self, dfa_id: int):
        """Muestra información sobre un AFD"""
        row = self.parser.df.iloc[dfa_id]
        
        print(f"\n📋 Información del AFD {dfa_id}:")
        print("-" * 70)
        print(f"  Regex: {row['Regex']}")
        print(f"  Alfabeto: {row['Alfabeto']}")
        print(f"  Estados: {row['Estados']}")
        print(f"  Estados de aceptación: {row['Estados de aceptación']}")
        print(f"  Transiciones: {row['Transiciones'][:100]}...")
    
    def test_multiples_cadenas(self, dfa_id: int, cadenas: list):
        """Testa múltiples cadenas contra un AFD"""
        print(f"\n🎯 Testing {len(cadenas)} cadenas contra AFD {dfa_id}")
        print("="*70)
        
        self.mostrar_info_afd(dfa_id)
        
        print(f"\n📊 Resultados:")
        print("-" * 70)
        
        for i, string in enumerate(cadenas, 1):
            result = self.predecir(dfa_id, string)
            
            display_str = string if len(string) <= 20 else string[:17] + "..."
            
            # Icono de corrección
            if result['y1_ground_truth'] is not None:
                correct = result['y1_pred'] == result['y1_ground_truth']
                icon = "✅" if correct else "❌"
            else:
                icon = "❓"
            
            print(f"\n{i}. String: '{display_str}'")
            print(f"   {icon} Y1 (Pertenencia): {result['y1_prob']:.3f} → {'SÍ' if result['y1_pred'] else 'NO'}")
            
            if result['y1_ground_truth'] is not None:
                print(f"      Ground Truth: {'SÍ' if result['y1_ground_truth'] else 'NO'}")
            
            print(f"   💫 Y2 (Compartida): {result['y2_prob']:.3f} → {'SÍ' if result['y2_pred'] else 'NO'}")


def modo_interactivo():
    """Modo interactivo para hacer predicciones"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "INFERENCIA INTERACTIVA" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    predictor = Predictor()
    
    while True:
        print("\n" + "="*70)
        print("🎮 MENÚ PRINCIPAL")
        print("="*70)
        print("1. Predecir para una cadena y AFD específico")
        print("2. Probar múltiples cadenas contra un AFD")
        print("3. Ver información de un AFD")
        print("4. Ejemplos predefinidos")
        print("5. Salir")
        print()
        
        opcion = input("Selecciona una opción (1-5): ").strip()
        
        if opcion == '1':
            # Predicción simple
            try:
                dfa_id = int(input("\nIngresa el ID del AFD (0-5999): ").strip())
                string = input("Ingresa la cadena (o <EPS> para cadena vacía): ").strip()
                
                result = predictor.predecir(dfa_id, string)
                
                print("\n" + "="*70)
                print("📊 RESULTADO")
                print("="*70)
                afd_info = result.get('afd_info', {})
                print(f"AFD seleccionado: {result['dfa_id']}")
                if afd_info:
                    print(f"  Regex: {afd_info.get('Regex', 'N/A')}")
                    print(f"  Alfabeto: {afd_info.get('Alfabeto', 'N/A')}")
                    print(f"  Estados: {afd_info.get('Estados', 'N/A')}")
                    print(f"  Aceptación: {afd_info.get('Aceptacion', 'N/A')}")
                print(f"\nCadena: '{result['string']}'")
                print()
                print(f"🎯 Y1 (Pertenencia a este AFD):")
                print(f"   Probabilidad: {result['y1_prob']:.4f}")
                print(f"   Predicción: {'SÍ pertenece' if result['y1_pred'] else 'NO pertenece'}")
                
                if result['y1_ground_truth'] is not None:
                    correct = result['y1_pred'] == result['y1_ground_truth']
                    icon = "✅" if correct else "❌"
                    print(f"   {icon} Ground Truth: {'SÍ' if result['y1_ground_truth'] else 'NO'}")
                    print(f"   ▶ Simulación del AFD: {'ACEPTA' if result['y1_ground_truth'] else 'RECHAZA'} la cadena")
                
                print()
                print(f"💫 Y2 (Compartida con otros AFDs):")
                print(f"   Probabilidad: {result['y2_prob']:.4f}")
                print(f"   Predicción: {'SÍ es compartida' if result['y2_pred'] else 'NO es compartida'}")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif opcion == '2':
            # Múltiples cadenas
            try:
                dfa_id = int(input("\nIngresa el ID del AFD (0-5999): ").strip())
                print("Ingresa las cadenas (separadas por coma):")
                cadenas_str = input("> ").strip()
                cadenas = [c.strip() for c in cadenas_str.split(',')]
                
                predictor.test_multiples_cadenas(dfa_id, cadenas)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif opcion == '3':
            # Ver info AFD
            try:
                dfa_id = int(input("\nIngresa el ID del AFD (0-5999): ").strip())
                predictor.mostrar_info_afd(dfa_id)
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif opcion == '4':
            # Ejemplos predefinidos
            print("\n🎲 Ejecutando ejemplos predefinidos...")
            
            ejemplos = [
                (0, "C"),
                (0, "CG"),
                (0, "CC"),
                (1, "G"),
                (1, "GG"),
                (1, "<EPS>"),
            ]
            
            for dfa_id, string in ejemplos:
                result = predictor.predecir(dfa_id, string)
                print(f"\nAFD {dfa_id} | String '{string}': Y1={result['y1_pred']} ({result['y1_prob']:.3f}), Y2={result['y2_pred']} ({result['y2_prob']:.3f})")
        
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n⚠️  Opción inválida. Intenta de nuevo.")


def demo_rapido():
    """Demo rápido con ejemplos predefinidos"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*25 + "DEMO RÁPIDO" + " "*32 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    predictor = Predictor()
    
    # Ejemplo 1: AFD 0
    print("\n📌 EJEMPLO 1: AFD 0")
    predictor.test_multiples_cadenas(0, ["C", "CG", "CC", "CCG", "A", "B"])
    
    # Ejemplo 2: AFD 1
    print("\n\n📌 EJEMPLO 2: AFD 1")
    predictor.test_multiples_cadenas(1, ["<EPS>", "G", "GG", "GGG", "D", "K"])
    
    print("\n" + "="*70)
    print("✅ Demo completado!")
    print("\n💡 Para modo interactivo completo, ejecuta:")
    print("   python inferencia_interactiva.py --interactivo")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactivo':
        modo_interactivo()
    else:
        demo_rapido()

