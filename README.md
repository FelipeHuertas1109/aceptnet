# 🤖 Modelo Dual-Encoder para Clasificación de Cadenas en AFDs

Modelo multi-tarea basado en deep learning que aprende a:
1. **Determinar si una cadena pertenece a un autómata específico**
2. **Predecir si una cadena puede ser aceptada por múltiples autómatas**

## 🏗️ Arquitectura

El modelo utiliza una arquitectura **dual-encoder** con dos cabezas de salida:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Cadena (string)        AFD (Autómata)                     │
│       ↓                       ↓                             │
│  Embedding              Matriz de                           │
│       ↓                 Transiciones                        │
│  BiGRU (2 capas)             ↓                              │
│       ↓                   MLP                               │
│    h_str                    ↓                               │
│       │                  h_afd                              │
│       │                     │                               │
│       ├─────────┬───────────┘                               │
│       │         │                                           │
│       │    concat(h_str, h_afd)                             │
│       │         │                                           │
│       │      ┌──┴──┐                                        │
│       │      │ MLP │                                        │
│       │      └──┬──┘                                        │
│       │         │                                           │
│       │    y1: ¿Pertenece a este AFD?                       │
│       │                                                     │
│    ┌──┴──┐                                                  │
│    │ MLP │                                                  │
│    └──┬──┘                                                  │
│       │                                                     │
│  y2: ¿Cadena compartida con otros AFDs?                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Clave

1. **String Encoder**: 
   - Embedding de símbolos (A-L)
   - BiGRU bidireccional de 2 capas
   - Captura patrones secuenciales en las cadenas

2. **AFD Encoder**:
   - MLP de 3 capas
   - Entrada: representación vectorial del AFD (3104 dims)
     - Matriz de transiciones one-hot: 16 × 12 × 16
     - Vector de estados de aceptación: 16
     - Máscara de estados válidos: 16

3. **Multi-Task Heads**:
   - **Head 1** (pertenencia): combina información de cadena + AFD
   - **Head 2** (compartida): usa solo información de la cadena

## 📊 Datos

### Entrada: `dataset6000.csv`
Contiene 6000 autómatas con:
- Regex
- Alfabeto
- Estados y estados de aceptación
- Transiciones
- Clase (cadenas aceptadas/rechazadas)

### Generación Automática
El script genera automáticamente pares `(dfa_id, string, label)`:
- **Positivos**: extraídos de la columna `Clase` (cadenas aceptadas)
- **Negativos**: generados aleatoriamente y verificados por simulación

### Labels
- **y1**: ¿La cadena pertenece a este AFD? (0 o 1)
- **y2**: ¿La cadena es aceptada por ≥2 AFDs diferentes? (0 o 1)

## 🚀 Uso

### Instalación

```bash
pip install -r requirements.txt
```

### Entrenamiento

```bash
python acepten.py
```

El script ejecuta el pipeline completo:
1. ✅ Carga y parsea los 6000 AFDs
2. ✅ Genera dataset de pares (dfa_id, string, label)
3. ✅ Calcula etiqueta y2 (cadenas compartidas)
4. ✅ Divide en train/val/test por dfa_id (70/15/15)
5. ✅ Entrena el modelo dual-encoder
6. ✅ Evalúa en test set
7. ✅ Genera visualizaciones

### Salidas

- `dataset_generated.csv`: Dataset completo generado
- `best_model.pt`: Mejor modelo entrenado
- `training_history.png`: Gráficas de entrenamiento

## 📈 Métricas de Evaluación

### Tarea 1: Pertenencia a AFD (y1)

| Métrica | Muy Bueno | Bueno | Regular | Malo |
|---------|-----------|-------|---------|------|
| **Accuracy** | ≥ 0.95 | 0.90-0.95 | 0.85-0.90 | < 0.85 |
| **F1 Score** | ≥ 0.95 | 0.90-0.95 | 0.85-0.90 | < 0.85 |

### Tarea 2: Cadena Compartida (y2)

| Métrica | Bueno | Regular | Malo |
|---------|-------|---------|------|
| **F1 Score** | ≥ 0.90 | 0.80-0.90 | < 0.80 |
| **PR-AUC** | ≥ 0.90 | 0.80-0.90 | < 0.80 |

## 🔧 Configuración

Puedes ajustar hiperparámetros en la función `main()`:

```python
# Generación de datos
generator.generate_full_dataset(
    pos_samples_per_dfa=30,  # Muestras positivas por AFD
    neg_samples_per_dfa=30   # Muestras negativas por AFD
)

# Entrenamiento
trainer = Trainer(
    model, 
    train_loader, 
    val_loader,
    lambda1=1.0,    # Peso de loss y1
    lambda2=1.0,    # Peso de loss y2
    lr=0.001,       # Learning rate
    device=device
)
trainer.train(num_epochs=30)
```

## 🎯 Generalización a Autómatas Nuevos

El split por `dfa_id` asegura que el modelo aprende patrones generales de autómatas, no memoriza autómatas específicos. Los AFDs en test nunca se vieron durante el entrenamiento.

## 🧪 Extensiones Posibles

1. **GNN para AFDs**: Reemplazar MLP con Graph Neural Network
2. **Attention Mechanism**: Agregar atención entre string y AFD
3. **Data Augmentation**: Generar más cadenas usando regex
4. **Transfer Learning**: Pre-entrenar en lenguajes formales
5. **Multi-length Analysis**: Evaluar por longitud de cadena

## 📝 Notas Técnicas

- **Alfabeto global**: A, B, C, D, E, F, G, H, I, J, K, L (12 símbolos)
- **Max estados**: 16 (S0-S15)
- **Padding**: Secuencias variables con pad_idx=12
- **Device**: Auto-detecta CUDA/CPU
- **Optimización**: Adam + ReduceLROnPlateau
- **Regularización**: Dropout (0.2-0.3) + Weight Decay

## 📄 Licencia

Este proyecto es parte de un experimento de investigación en aprendizaje automático aplicado a teoría de autómatas.

