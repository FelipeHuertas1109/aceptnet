# 🎯 Cómo Usar los Resultados del Modelo

Has entrenado con éxito el modelo en Colab. Ahora tienes 3 archivos en `result/`:

```
result/
├── best_model.pt             (7.3 MB) - Modelo entrenado
├── dataset_generated.csv     (4.0 MB) - Dataset completo
└── training_history.png      (128 KB) - Gráficas de entrenamiento
```

## 📊 Tus Resultados

### ✅ Métricas Finales en Test Set

**Tarea 1: Pertenencia a AFD (Y1)**
- Accuracy: **0.8938** (89.38%)
- F1 Score: **0.8682** (86.82%)
- Rendimiento: **REGULAR** ⚠️ (cerca de BUENO)

**Tarea 2: Cadena Compartida (Y2)**
- Accuracy: **0.9887** (98.87%)
- F1 Score: **0.9924** (99.24%)
- PR-AUC: **0.9997** (99.97%)
- Rendimiento: **BUENO** ✅ (casi perfecto!)

### 📈 Interpretación

✅ **Fortalezas:**
- Excelente en detectar si una cadena es compartida por múltiples AFDs (Y2)
- Buen accuracy general (~89%)
- Modelo estable (no overfitting visible)

⚠️ **Áreas de Mejora:**
- Y1 está en zona "REGULAR" (objetivo: ≥0.90)
- Puede mejorar con más épocas o data augmentation

---

## 🚀 Scripts de Análisis Creados

He creado 3 scripts para analizar y usar tu modelo:

### 1️⃣ **Análisis Completo** (`analizar_resultados.py`)

Genera un reporte completo con:
- Estadísticas del dataset
- Predicciones en ejemplos aleatorios
- Análisis de errores
- Visualización del historial

```bash
python analizar_resultados.py
```

**Salida:**
- `historial_detallado.png`
- Reporte en consola con ejemplos y errores

---

### 2️⃣ **Inferencia Interactiva** (`inferencia_interactiva.py`)

Haz predicciones con tu modelo entrenado:

**Modo Demo (rápido):**
```bash
python inferencia_interactiva.py
```

**Modo Interactivo (menú completo):**
```bash
python inferencia_interactiva.py --interactivo
```

**Funciones:**
- ✅ Predecir si una cadena pertenece a un AFD
- ✅ Predecir si una cadena es compartida
- ✅ Comparar con ground truth (simulación real)
- ✅ Probar múltiples cadenas a la vez
- ✅ Ver información de AFDs

**Ejemplo de uso:**
```python
from inferencia_interactiva import Predictor

predictor = Predictor()

# Predecir
result = predictor.predecir(dfa_id=0, string="ABC")
print(f"Pertenece: {result['y1_pred']}")
print(f"Compartida: {result['y2_pred']}")
```

---

### 3️⃣ **Visualizaciones** (`comparar_resultados.py`)

Genera gráficas detalladas:

```bash
python comparar_resultados.py
```

**Genera:**
- `analisis_dataset.png` - Distribución de datos
- `metricas_finales.png` - Comparación de métricas
- `training_history_display.png` - Historial mejorado

---

## 🎮 Ejemplos de Uso Rápido

### Ejemplo 1: Ver Resultados Rápidamente

```bash
# 1. Análisis completo del modelo
python analizar_resultados.py

# 2. Visualizaciones bonitas
python comparar_resultados.py

# 3. Probar predicciones
python inferencia_interactiva.py
```

### Ejemplo 2: Hacer Predicciones desde Código

```python
from inferencia_interactiva import Predictor

# Cargar modelo
predictor = Predictor(
    model_path='result/best_model.pt',
    dataset_path='dataset6000.csv'
)

# Predecir
result = predictor.predecir(dfa_id=0, string="CG")

print(f"Cadena: {result['string']}")
print(f"AFD: {result['dfa_id']}")
print(f"Y1 (Pertenece): {result['y1_pred']} (prob: {result['y1_prob']:.3f})")
print(f"Y2 (Compartida): {result['y2_pred']} (prob: {result['y2_prob']:.3f})")
print(f"Ground Truth: {result['y1_ground_truth']}")
```

### Ejemplo 3: Probar Múltiples Cadenas

```python
from inferencia_interactiva import Predictor

predictor = Predictor()

# Ver info del AFD
predictor.mostrar_info_afd(dfa_id=0)

# Probar varias cadenas
cadenas = ["C", "CG", "CC", "CCG", "ABC", "<EPS>"]
predictor.test_multiples_cadenas(dfa_id=0, cadenas=cadenas)
```

---

## 📊 Análisis del Dataset Generado

Tu dataset tiene:
- **253,751 ejemplos** totales
- **6,000 AFDs** únicos
- **~42 ejemplos por AFD** (30 positivos + 30 negativos aprox.)
- **73.9% cadenas compartidas** (Y2=1)

### Cargar Dataset

```python
import pandas as pd

df = pd.read_csv('result/dataset_generated.csv')

print(df.head())
print(df.columns)  # ['dfa_id', 'string', 'label', 'y2']
```

---

## 🔍 Entender las Predicciones

### Y1: Pertenencia a AFD

**Pregunta:** ¿La cadena "ABC" es aceptada por el AFD #42?

```python
result = predictor.predecir(dfa_id=42, string="ABC")
if result['y1_pred']:
    print(f"✅ SÍ, con confianza {result['y1_prob']:.1%}")
else:
    print(f"❌ NO, con confianza {1-result['y1_prob']:.1%}")
```

### Y2: Cadena Compartida

**Pregunta:** ¿La cadena "ABC" es aceptada por múltiples AFDs?

```python
if result['y2_pred']:
    print("💫 Esta cadena es COMPARTIDA por varios AFDs")
else:
    print("🎯 Esta cadena es ÚNICA a este AFD")
```

---

## 🎯 Casos de Uso

### 1. Validar Autómatas

```python
# ¿Este AFD acepta estas cadenas?
predictor.test_multiples_cadenas(
    dfa_id=0, 
    cadenas=["valid1", "valid2", "invalid1"]
)
```

### 2. Encontrar Ambigüedades

```python
# ¿Qué cadenas son aceptadas por múltiples AFDs?
df = pd.read_csv('result/dataset_generated.csv')
ambiguas = df[(df['label'] == 1) & (df['y2'] == 1)]
print(f"Cadenas ambiguas: {len(ambiguas)}")
```

### 3. Debuggear AFD

```python
# Ver por qué el modelo falla en ciertos casos
predictor.mostrar_info_afd(dfa_id=123)
result = predictor.predecir(123, "problemática_string")
```

---

## 📈 Próximos Pasos

### Mejorar el Modelo

1. **Más épocas**: Entrenar 50-100 épocas
   ```python
   # En acepten_colab.py línea 743
   trainer.train(num_epochs=50)
   ```

2. **Data augmentation**: Generar más ejemplos
   ```python
   # Línea 713
   df = generator.generate_full_dataset(
       pos_samples_per_dfa=50,
       neg_samples_per_dfa=50
   )
   ```

3. **Aumentar capacidad**: Más capas/neuronas
   ```python
   model = DualEncoderModel(
       rnn_hidden_dim=128,  # era 64
       afd_hidden_dim=256   # era 128
   )
   ```

### Experimentos

- ✅ Probar diferentes arquitecturas (GNN para AFDs)
- ✅ Transfer learning desde otros modelos
- ✅ Ensembles de múltiples modelos
- ✅ Attention mechanism entre string y AFD

---

## 🆘 Troubleshooting

### Error: "No module named 'acepten'"

```bash
# Asegúrate de estar en el directorio correcto
cd C:\Users\Felipe\Documents\codes\aceptnet
python analizar_resultados.py
```

### Error: "File not found: result/best_model.pt"

```bash
# Verifica que los archivos estén en result/
dir result
# O en Linux/Mac:
ls result/
```

### Modelo muy lento

```python
# Usa CPU si CUDA no está disponible
predictor = Predictor()
# El script detecta automáticamente
```

---

## 📚 Archivos de Referencia

- `acepten.py` - Código del modelo original
- `acepten_colab.py` - Versión para Colab
- `README.md` - Documentación completa del proyecto
- `RESUMEN.md` - Arquitectura y detalles técnicos

---

## ✅ Checklist de Análisis

- [ ] Ejecutar `analizar_resultados.py`
- [ ] Ejecutar `comparar_resultados.py`
- [ ] Probar inferencia con `inferencia_interactiva.py`
- [ ] Revisar visualizaciones generadas
- [ ] Entender por qué Y1 está en "REGULAR"
- [ ] Identificar patrones de error
- [ ] Decidir si re-entrenar con mejoras

---

## 🎉 ¡Felicitaciones!

Has entrenado exitosamente un modelo dual-task para clasificación de cadenas en AFDs con:
- ✅ ~1.9M parámetros
- ✅ 253K ejemplos de entrenamiento
- ✅ Métricas competitivas en Y2 (99.24% F1)
- ✅ Base sólida para mejoras en Y1

**¡Ahora a analizar y experimentar!** 🚀

