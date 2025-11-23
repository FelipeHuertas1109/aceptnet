# Pipeline de Entrenamiento de AFD Dual-Encoder

Este proyecto está dividido en dos scripts independientes para optimizar el flujo de trabajo:

## 📂 Archivos

1. **`gen_dataset_exhaustive.py`** - Generación de dataset (CPU)
2. **`train_dual_encoder.py`** - Entrenamiento del modelo (GPU)
3. **`dataset6000.csv`** - Dataset original de AFDs
4. **`acepten_colab.py`** - Script original (referencia)

---

## 🔄 Flujo de Trabajo

### Paso 1: Generar Dataset (Una sola vez)

Este script genera todas las cadenas exhaustivamente y crea el dataset completo.

**Características:**
- ✅ Solo CPU (no necesita GPU)
- ✅ Se ejecuta una sola vez (o cuando cambias la lógica)
- ✅ Genera ~1.5M ejemplos en pocos minutos
- ✅ 100% preciso: cada etiqueta verificada por simulación

**Ejecución en Google Colab:**

```python
# 1. Sube dataset6000.csv a /content/sample_data/
# 2. Sube gen_dataset_exhaustive.py a /content/
# 3. Ejecuta
!python gen_dataset_exhaustive.py
```

**Ejecución Local:**

```bash
# En tu PC local
cd acepnet3
python gen_dataset_exhaustive.py
```

**Salida:**
- `dataset_generated.csv` → Dataset completo con columnas: `dfa_id`, `string`, `label`, `y2`

**Configuración (en el archivo):**
```python
GEN_CONFIG = {
    'pos_samples_per_dfa': 100,  # Máximo de positivos
    'neg_samples_per_dfa': 150,  # Máximo de negativos
    'max_string_length': 5,      # Longitud máxima
}
```

---

### Paso 2: Entrenar Modelo (Repetir cuando quieras)

Este script entrena el modelo usando el dataset ya generado.

**Características:**
- ✅ Optimizado para GPU (T4 en Colab)
- ✅ Se puede ejecutar múltiples veces
- ✅ Rápido: ~5-10 min en T4
- ✅ Early stopping automático

**Ejecución en Google Colab:**

```python
# 1. Sube estos archivos a /content/sample_data/
#    - dataset6000.csv
#    - dataset_generated.csv

# 2. Sube train_dual_encoder.py a /content/

# 3. Ejecuta
!python train_dual_encoder.py
```

**Ejecución Local (si tienes GPU):**

```bash
cd acepnet3
python train_dual_encoder.py
```

**Salidas:**
- `best_model.pt` → Modelo entrenado
- `thresholds.json` → Umbrales calibrados
- `training_history.png` → Gráficas de entrenamiento

**Configuración (en el archivo):**
```python
TRAIN_CONFIG = {
    'label_smoothing': 0.0,      # Sin smoothing
    'lambda1': 1.0,              # Peso de Y1
    'lambda2': 0.3,              # Peso de Y2
    'batch_size': 128,
    'num_epochs': 40,            # Máximo
    'early_stop_patience': 7,    # Épocas sin mejora
    'early_stop_min_delta': 1e-4 # Mejora mínima
}
```

---

## 🎯 Integración con Django Backend

Una vez entrenado, copia estos archivos a tu backend:

```
C:\Users\Felipe Huertas\Documents\Codigos\lenguajes-back\models\acepnet\
├── dataset6000.csv        ← Para reconstruir AFDs
├── best_model.pt          ← Modelo entrenado
└── thresholds.json        ← Umbrales calibrados
```

Tu servicio Django (`acepnet_service.py`) solo necesita:
1. Cargar `dataset6000.csv` para obtener features de AFDs
2. Cargar `best_model.pt` para inferencia
3. Cargar `thresholds.json` para clasificación

---

## 📊 Ventajas de esta Separación

| Aspecto | Antes | Ahora |
|---------|-------|-------|
| **Generación** | Cada entrenamiento | Una sola vez |
| **GPU para generación** | Desperdiciada | No se usa (CPU) |
| **Experimentación** | Lenta | Rápida |
| **Reproducibilidad** | Difícil | Fácil (mismo CSV) |

---

## 🔧 Modificar Configuración

### Para cambiar el tamaño del dataset:

Edita `gen_dataset_exhaustive.py`:
```python
GEN_CONFIG = {
    'pos_samples_per_dfa': 150,  # Más positivos
    'neg_samples_per_dfa': 200,  # Más negativos
    'max_string_length': 6,      # Cadenas más largas
}
```

Luego regenera el dataset:
```bash
python gen_dataset_exhaustive.py
```

### Para cambiar hiperparámetros de entrenamiento:

Edita `train_dual_encoder.py`:
```python
TRAIN_CONFIG = {
    'lambda1': 1.5,              # Mayor peso a Y1
    'lambda2': 0.2,              # Menor peso a Y2
    'batch_size': 256,           # Batch más grande (si tienes memoria)
    'num_epochs': 50,            # Más épocas
    'early_stop_patience': 10,   # Más paciencia
}
```

---

## 🚀 Ejemplo Completo (Colab)

### Opción A: Generar dataset EN Colab (recomendado si no lo tienes)

```python
# ===== CELDA 1: Subir archivos =====
from google.colab import files

print("📤 Sube dataset6000.csv")
uploaded = files.upload()

print("📤 Sube gen_dataset_exhaustive.py")
uploaded = files.upload()

print("📤 Sube train_dual_encoder.py")
uploaded = files.upload()

# Mover a /content/sample_data/
!mkdir -p /content/sample_data
!mv dataset6000.csv /content/sample_data/

# ===== CELDA 2: Generar dataset (CPU) =====
!python gen_dataset_exhaustive.py

# Mover dataset generado a sample_data
!mv dataset_generated.csv /content/sample_data/

# ===== CELDA 3: Entrenar (GPU) =====
!python train_dual_encoder.py

# ===== CELDA 4: Descargar resultados =====
files.download('best_model.pt')
files.download('thresholds.json')
files.download('training_history.png')
```

### Opción B: Solo entrenar (si ya tienes dataset_generated.csv)

```python
# ===== CELDA 1: Subir archivos =====
from google.colab import files

print("📤 Sube dataset6000.csv")
uploaded = files.upload()

print("📤 Sube dataset_generated.csv")
uploaded = files.upload()

print("📤 Sube train_dual_encoder.py")
uploaded = files.upload()

# Mover a /content/sample_data/
!mkdir -p /content/sample_data
!mv dataset6000.csv /content/sample_data/
!mv dataset_generated.csv /content/sample_data/

# ===== CELDA 2: Entrenar =====
!python train_dual_encoder.py

# ===== CELDA 3: Descargar resultados =====
files.download('best_model.pt')
files.download('thresholds.json')
files.download('training_history.png')
```

---

## ❓ FAQ

**P: ¿Tengo que regenerar el dataset cada vez?**  
R: No, solo cuando cambies la configuración de generación o el dataset de AFDs.

**P: ¿Puedo entrenar sin GPU?**  
R: Sí, pero será mucho más lento (~1-2 horas vs 5-10 minutos en T4).

**P: ¿El dataset generado es diferente cada vez?**  
R: Las cadenas son exhaustivas, pero el orden es aleatorio. Los resultados son equivalentes.

**P: ¿Cómo sé si el modelo está bien entrenado?**  
R: Busca en la salida:
- Y1 Accuracy > 0.95
- Y1 F1 Score > 0.95
- Y2 PR-AUC > 0.80

**P: ¿Puedo usar este pipeline en producción?**  
R: Sí, solo necesitas los 3 archivos de salida:
  - `dataset6000.csv`
  - `best_model.pt`
  - `thresholds.json`

---

## 📝 Notas

- El script de generación usa `tqdm` para mostrar progreso
- El script de entrenamiento usa early stopping para evitar overfitting
- Ambos scripts tienen manejo de errores para rutas de archivos
- Los scripts son compatibles con Windows, Linux y macOS

