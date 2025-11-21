# 🎉 Resumen del Proyecto - Modelo Dual-Encoder para AFDs

## 📁 Archivos Creados

### 🖥️ Para Uso Local (CPU)
- **`acepten.py`** - Script completo para entrenar localmente (769 líneas)
- **`test_pipeline.py`** - Script de pruebas rápidas
- **`requirements.txt`** - Dependencias del proyecto

### ☁️ Para Google Colab (GPU) ⭐ RECOMENDADO
- **`acepten_colab.py`** - Script optimizado para Colab con GPU (650 líneas)
  - ✅ Barras de progreso con tqdm
  - ✅ Batch size optimizado para GPU (128)
  - ✅ Detección automática de CUDA
  - ✅ Rutas configuradas para `/content/sample_data/`

### 📚 Documentación
- **`README.md`** - Documentación completa del proyecto
- **`COLAB_INSTRUCTIONS.md`** - Guía paso a paso para Colab
- **`quick_colab_setup.txt`** - Setup super rápido (copiar/pegar)

### 📊 Dataset
- **`dataset6000.csv`** - 6000 AFDs originales (ya tienes)

---

## 🚀 QUICK START - Google Colab (Recomendado)

### 1️⃣ Abrir Google Colab
- Ve a: https://colab.research.google.com/
- Crea un nuevo notebook
- **Activar GPU**: `Runtime` → `Change runtime type` → **GPU**

### 2️⃣ Celda 1: Setup
```python
# Instalar dependencias
!pip install -q torch pandas numpy scikit-learn matplotlib tqdm

# Subir archivos
from google.colab import files
import shutil

print("📤 Sube dataset6000.csv:")
uploaded = files.upload()
!mkdir -p /content/sample_data
shutil.move('dataset6000.csv', '/content/sample_data/dataset6000.csv')

print("\n📤 Sube acepten_colab.py:")
uploaded = files.upload()

# Verificar GPU
import torch
print(f"\n{'✅' if torch.cuda.is_available() else '⚠️'} GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'}")
```

### 3️⃣ Celda 2: Entrenar
```python
!python acepten_colab.py
```

### 4️⃣ Celda 3: Descargar Resultados
```python
from google.colab import files
files.download('best_model.pt')
files.download('dataset_generated.csv')
files.download('training_history.png')
```

⏱️ **Tiempo total**: ~20-25 minutos en GPU T4

---

## 🏗️ Arquitectura del Modelo

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Cadena (e.g., "ABC")    AFD (dfa_id=0)        │
│        ↓                      ↓                 │
│   Tokenización          Matriz 16×12×16         │
│    [0,1,2]              + Accept Vec            │
│        ↓                      ↓                 │
│   Embedding              MLP Encoder            │
│        ↓                      ↓                 │
│  BiGRU (2 capas)          h_afd (128)           │
│        ↓                                        │
│   h_str (128)                                   │
│        │                      │                 │
│        ├──────────┬───────────┘                 │
│        │          │                             │
│        │     concat(h_str, h_afd)               │
│        │          │                             │
│        │      ┌───┴───┐                         │
│        │      │  MLP  │                         │
│        │      └───┬───┘                         │
│        │          │                             │
│        │     y1: Pertenencia                    │
│        │     (¿acepta este AFD?)                │
│        │                                        │
│     ┌──┴──┐                                     │
│     │ MLP │                                     │
│     └──┬──┘                                     │
│        │                                        │
│   y2: Compartida                                │
│   (¿múltiples AFDs?)                            │
│                                                 │
└─────────────────────────────────────────────────┘

Total parámetros: ~1.9M
```

---

## 📊 Datos y Entrenamiento

### Dataset Generado
- **Entrada**: 6000 AFDs
- **Salida**: ~250K ejemplos
  - 30 positivos + 30 negativos por AFD
  - ~74% son cadenas compartidas (y2=1)

### Split
- **Train**: 70% (4200 AFDs)
- **Val**: 15% (900 AFDs)
- **Test**: 15% (900 AFDs)

### Hiperparámetros
- **Optimizer**: Adam (lr=0.001)
- **Loss**: λ₁·BCE(y1) + λ₂·BCE(y2)
- **Batch size**: 128 (GPU) / 64 (CPU)
- **Épocas**: 30
- **Regularización**: Dropout (0.2-0.3) + Weight Decay

---

## 🎯 Métricas de Éxito

### Tarea 1: Pertenencia a AFD (y1)
| Nivel | Accuracy | F1 Score |
|-------|----------|----------|
| 🥇 Muy Bueno | ≥ 0.95 | ≥ 0.95 |
| 🥈 Bueno | 0.90-0.95 | 0.90-0.95 |
| 🥉 Regular | 0.85-0.90 | 0.85-0.90 |
| ❌ Malo | < 0.85 | < 0.85 |

### Tarea 2: Cadena Compartida (y2)
| Nivel | F1 Score | PR-AUC |
|-------|----------|--------|
| 🥇 Bueno | ≥ 0.90 | ≥ 0.90 |
| 🥈 Regular | 0.80-0.90 | 0.80-0.90 |
| ❌ Malo | < 0.80 | < 0.80 |

---

## 📦 Salidas del Modelo

Después del entrenamiento, obtendrás:

1. **`best_model.pt`** - Mejor modelo guardado (pesos)
2. **`dataset_generated.csv`** - Dataset completo con y1 e y2
3. **`training_history.png`** - Gráficas de:
   - Loss (train/val)
   - Accuracy Y1 (train/val)
   - Accuracy Y2 (train/val)

---

## 🔧 Personalización

### Cambiar número de épocas
En `acepten_colab.py`, línea 743:
```python
trainer.train(num_epochs=50)  # default: 30
```

### Más datos por AFD
Línea 713:
```python
df = generator.generate_full_dataset(
    pos_samples_per_dfa=50,  # default: 30
    neg_samples_per_dfa=50   # default: 30
)
```

### Ajustar batch size
Línea 734:
```python
batch_size = 256  # default: 128 para GPU
```

---

## 💡 Características Principales

✅ **Dual-Encoder Architecture**
- String encoder: BiGRU sobre tokens
- AFD encoder: MLP sobre matriz de transiciones

✅ **Multi-Task Learning**
- Head 1: Pertenencia (usa string + AFD)
- Head 2: Ambigüedad (usa solo string)

✅ **Generalización**
- Split por dfa_id → evalúa en AFDs nunca vistos

✅ **Manejo de Casos Especiales**
- Cadenas vacías (épsilon)
- Secuencias de longitud variable
- Alfabetos distintos por AFD

✅ **Optimizado para GPU**
- Batch processing eficiente
- Pin memory para transferencias rápidas
- Mixed precision ready

---

## 📞 Próximos Pasos

### Después del Entrenamiento

1. **Analizar resultados**
   - Revisar `training_history.png`
   - Verificar métricas en test set

2. **Experimentos**
   - Cambiar arquitectura (más capas, GNN)
   - Data augmentation
   - Diferentes splits

3. **Deployment**
   - Cargar modelo: `torch.load('best_model.pt')`
   - Inferencia en nuevos AFDs

### Ejemplo de Inferencia

```python
# Cargar modelo
model = DualEncoderModel()
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Preparar input
parser = AFDParser('dataset6000.csv')
afd_features = parser.get_afd_features(dfa_id=0)
string_tokens = [0, 1, 2]  # "ABC"

# Predecir
with torch.no_grad():
    y1_hat, y2_hat = model(...)
    print(f"Pertenece: {y1_hat > 0.5}")
    print(f"Compartida: {y2_hat > 0.5}")
```

---

## 🎓 Referencias

Este proyecto implementa las ideas de tu plan original:
- Parseo de AFDs desde dataset estructurado
- Representación vectorial con one-hot encoding
- Arquitectura dual-encoder con multi-task learning
- Evaluación rigurosa con métricas claras

**Características implementadas exactamente como especificaste**:
- ✅ Alfabeto global A-L (12 símbolos)
- ✅ Max 16 estados (S0-S15)
- ✅ Matriz de transiciones 16×12×16
- ✅ Two-head architecture
- ✅ y2 basado en conteo de AFDs por string
- ✅ Split por dfa_id (no por strings)

---

## 🙌 ¡Todo Listo!

Tienes todo lo necesario para entrenar el modelo. Sube los archivos a Colab y ¡a entrenar! 🚀

**Archivos que necesitas subir a Colab**:
1. `dataset6000.csv`
2. `acepten_colab.py`

¡Éxito con el entrenamiento! 🎉

