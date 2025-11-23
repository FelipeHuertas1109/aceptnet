"""
Script Interactivo Mejorado para Hacer Predicciones con el Modelo
Versión con interfaz mejorada y soporte para múltiples modelos
"""

import torch
import pandas as pd
import json
import os
from acepten import AFDParser, DualEncoderModel, CHAR_TO_IDX


class Predictor:
    """Clase para hacer predicciones con el modelo entrenado"""
    
    def __init__(self, model_path='result/best_model.pt', dataset_path='dataset6000.csv', 
                 thresholds_path=None):
        print("🔄 Cargando modelo y parser...")
        
        # Cargar modelo
        self.model = DualEncoderModel()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        self.model.eval()
        
        # Cargar parser
        self.parser = AFDParser(dataset_path)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Cargar umbrales calibrados si existen
        self.thresholds = {'y1': 0.5, 'y2': 0.5}  # Valores por defecto
        if thresholds_path and os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r') as f:
                    self.thresholds = json.load(f)
                print(f"✅ Umbrales calibrados: Y1={self.thresholds['y1']}, Y2={self.thresholds['y2']}")
            except:
                print("⚠️  Usando umbrales por defecto (0.5)")
        
        print(f"✅ Listo! Device: {self.device}")
    
    def predecir(self, dfa_id: int, string: str):
        """Hace una predicción para una cadena y un AFD"""
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

        # Verificar alfabeto del AFD
        alfabeto = set(row['Alfabeto'].split())
        alphabet_mismatch = False
        if string not in ("", "<EPS>"):
            for ch in string:
                if ch not in alfabeto:
                    alphabet_mismatch = True
                    break
        
        if alphabet_mismatch:
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
            
            y1_pred = y1_prob >= self.thresholds['y1']
            y2_pred = y2_prob >= self.thresholds['y2']
        
        # Simular AFD para comparar (ground truth)
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
            'afd_info': afd_info,
            'y2_prob': y2_prob,
            'y2_pred': y2_pred,
        }
    
    def mostrar_info_afd(self, dfa_id: int):
        """Muestra información sobre un AFD"""
        row = self.parser.df.iloc[dfa_id]
        
        print(f"  📌 Regex: {row['Regex']}")
        print(f"  🔤 Alfabeto: {row['Alfabeto']}")
        print(f"  🔢 Estados: {row['Estados']}")
        print(f"  ✅ Estados de aceptación: {row['Estados de aceptación']}")


def modo_interactivo():
    """Modo interactivo mejorado para hacer predicciones"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "INFERENCIA INTERACTIVA v2.0" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Detectar automáticamente modelos disponibles
    model_options = [
        ('result-negative-positive/best_model (1).pt', 'result-negative-positive/thresholds.json', '🆕 Modelo Mejorado (con augmentación)'),
        ('result/best_model.pt', None, '📦 Modelo Original'),
    ]
    
    print("🔍 Modelos disponibles:")
    available_models = []
    for i, (model_path, thresh_path, desc) in enumerate(model_options, 1):
        if os.path.exists(model_path):
            available_models.append((model_path, thresh_path, desc))
            print(f"   {i}. {desc}")
    
    if not available_models:
        print("❌ No se encontró ningún modelo. Asegúrate de tener best_model.pt")
        return
    
    print()
    if len(available_models) > 1:
        choice = input(f"Selecciona modelo (1-{len(available_models)}) [1]: ").strip()
        model_idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(available_models) else 0
    else:
        model_idx = 0
    
    model_path, thresh_path, desc = available_models[model_idx]
    print(f"✅ Usando: {desc}\n")
    
    predictor = Predictor(model_path=model_path, thresholds_path=thresh_path)
    
    while True:
        print("\n" + "="*70)
        print("🎮 MENÚ PRINCIPAL")
        print("="*70)
        print("1. 🎯 Probar cadena con un AFD (por ID)")
        print("2. 🔍 Buscar AFD por palabra clave en regex")
        print("3. 📋 Ver información de un AFD")
        print("4. 🎲 Ejemplos predefinidos")
        print("5. 🚪 Salir")
        print()
        
        opcion = input("Selecciona una opción (1-5): ").strip()
        
        if opcion == '1':
            # Predicción por ID
            try:
                dfa_id = int(input("\n🔢 Ingresa el ID del AFD (0-5999): ").strip())
                
                # Mostrar información del AFD seleccionado
                print("\n" + "="*70)
                print(f"📋 AFD SELECCIONADO: #{dfa_id}")
                print("="*70)
                predictor.mostrar_info_afd(dfa_id)
                print()
                
                string = input("✍️  Ingresa la cadena a evaluar (o <EPS> para vacía): ").strip()
                
                result = predictor.predecir(dfa_id, string)
                
                print("\n" + "="*70)
                print("📊 RESULTADO DE LA PREDICCIÓN")
                print("="*70)
                print(f"Cadena evaluada: '{result['string']}'")
                print()
                
                # Advertencia de alfabeto
                if result['y1_alphabet_mismatch']:
                    print("⚠️  ADVERTENCIA: Símbolos fuera del alfabeto del AFD")
                    print(f"   Alfabeto permitido: {result['afd_info']['Alfabeto']}")
                    print()
                
                print(f"🤖 PREDICCIÓN DEL MODELO:")
                print(f"   Probabilidad: {result['y1_prob']:.4f}")
                print(f"   Veredicto: {'✅ ACEPTA' if result['y1_pred'] else '❌ RECHAZA'}")
                print()
                
                print(f"🎯 SIMULADOR REAL (Ground Truth):")
                if result['y1_ground_truth'] is not None:
                    print(f"   Veredicto: {'✅ ACEPTA' if result['y1_ground_truth'] else '❌ RECHAZA'}")
                    print()
                    
                    # Comparación final
                    if result['y1_pred'] == result['y1_ground_truth']:
                        print("🎉 ¡CORRECTO! El modelo predijo correctamente")
                    else:
                        print("⚠️  ERROR: El modelo se equivocó")
                        if result['y1_pred'] and not result['y1_ground_truth']:
                            print("   → Falso Positivo: predijo ACEPTA pero debería RECHAZAR")
                        else:
                            print("   → Falso Negativo: predijo RECHAZA pero debería ACEPTA")
                else:
                    print("   ❓ No se pudo simular")
                
                print()
                print(f"💫 Y2 (Compartida con otros AFDs): {result['y2_prob']:.4f} → {'SÍ' if result['y2_pred'] else 'NO'}")
                
            except ValueError:
                print("\n❌ Error: Debes ingresar un número válido")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif opcion == '2':
            # Buscar por regex
            try:
                keyword = input("\n🔍 Ingresa palabra clave para buscar en regex: ").strip()
                
                # Buscar AFDs que contengan la palabra clave
                matches = []
                for idx, row in predictor.parser.df.iterrows():
                    if keyword.lower() in str(row['Regex']).lower():
                        matches.append((idx, row['Regex']))
                        if len(matches) >= 10:  # Limitar a 10 resultados
                            break
                
                if not matches:
                    print(f"\n❌ No se encontraron AFDs con '{keyword}' en la regex")
                    continue
                
                print(f"\n📋 AFDs encontrados ({len(matches)}):")
                print("="*70)
                for i, (idx, regex) in enumerate(matches, 1):
                    print(f"{i}. ID={idx:4d} | Regex: {regex[:50]}{'...' if len(regex) > 50 else ''}")
                
                print()
                choice = int(input(f"Selecciona un AFD (1-{len(matches)}): ").strip())
                
                if 1 <= choice <= len(matches):
                    dfa_id = matches[choice - 1][0]
                    
                    print("\n" + "="*70)
                    print(f"📋 AFD SELECCIONADO: #{dfa_id}")
                    print("="*70)
                    predictor.mostrar_info_afd(dfa_id)
                    print()
                    
                    string = input("✍️  Ingresa la cadena a evaluar: ").strip()
                    result = predictor.predecir(dfa_id, string)
                    
                    # Mostrar resultado (mismo formato que opción 1)
                    print("\n" + "="*70)
                    print("📊 RESULTADO")
                    print("="*70)
                    print(f"🤖 Modelo: {'✅ ACEPTA' if result['y1_pred'] else '❌ RECHAZA'} (prob={result['y1_prob']:.4f})")
                    print(f"🎯 Real:   {'✅ ACEPTA' if result['y1_ground_truth'] else '❌ RECHAZA'}")
                    
                    if result['y1_pred'] == result['y1_ground_truth']:
                        print("🎉 ¡Predicción CORRECTA!")
                    else:
                        print("⚠️  Predicción INCORRECTA")
                
            except ValueError:
                print("\n❌ Error: Selección inválida")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif opcion == '3':
            # Ver info AFD
            try:
                dfa_id = int(input("\n🔢 Ingresa el ID del AFD (0-5999): ").strip())
                print("\n" + "="*70)
                print(f"📋 INFORMACIÓN DEL AFD #{dfa_id}")
                print("="*70)
                predictor.mostrar_info_afd(dfa_id)
            except ValueError:
                print("\n❌ Error: Debes ingresar un número válido")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif opcion == '4':
            # Ejemplos predefinidos
            print("\n🎲 Ejecutando ejemplos predefinidos...")
            
            ejemplos = [
                (0, "C"),
                (0, "A"),
                (1, "G"),
                (1, "AC"),
            ]
            
            for dfa_id, string in ejemplos:
                result = predictor.predecir(dfa_id, string)
                modelo = "✅" if result['y1_pred'] else "❌"
                real = "✅" if result['y1_ground_truth'] else "❌"
                correcto = "✓" if result['y1_pred'] == result['y1_ground_truth'] else "✗"
                print(f"\nAFD {dfa_id} | '{string}': Modelo={modelo} Real={real} [{correcto}]")
        
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n⚠️  Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    modo_interactivo()

