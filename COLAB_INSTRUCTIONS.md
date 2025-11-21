# 🚀 Instrucciones para Entrenar en Google Colab

## Paso 1: Preparar Google Colab

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Crea un nuevo notebook
3. **Activar GPU**: 
   - Menú: `Runtime` → `Change runtime type`
   - En "Hardware accelerator" selecciona: **GPU**
   - Guarda

## Paso 2: Subir el Dataset

En Colab, ejecuta esta celda para subir `dataset6000.csv`:

```python
from google.colab import files
import shutil

# Subir archivo
uploaded = files.upload()

# Mover a sample_data
!mkdir -p /content/sample_data
shutil.move('dataset6000.csv', '/content/sample_data/dataset6000.csv')

print("✅ Dataset subido correctamente!")
```

## Paso 3: Instalar Dependencias

```python
!pip install -q torch torchvision torchaudio
!pip install -q pandas numpy scikit-learn matplotlib tqdm

print("✅ Dependencias instaladas!")
```

## Paso 4: Subir el Script Principal

Opción A - **Subir archivo** (Recomendado):
```python
from google.colab import files
uploaded = files.upload()  # Selecciona acepten_colab.py
```

Opción B - **Crear desde código**:
```python
%%writefile acepten_colab.py
# Copia y pega todo el contenido de acepten_colab.py aquí
```

## Paso 5: Ejecutar Entrenamiento

```python
!python acepten_colab.py
```

## Paso 6: Descargar Resultados

```python
from google.colab import files

# Descargar modelo entrenado
files.download('best_model.pt')

# Descargar dataset generado
files.download('dataset_generated.csv')

# Descargar gráficas
files.download('training_history.png')
```

---

## 📋 Notebook Completo (Todo en Uno)

Alternativamente, copia y pega esto en UNA SOLA celda:

```python
# ============================================================================
# CONFIGURACIÓN Y SETUP
# ============================================================================

# 1. Instalar dependencias
!pip install -q torch torchvision torchaudio
!pip install -q pandas numpy scikit-learn matplotlib tqdm

# 2. Subir dataset6000.csv
from google.colab import files
import shutil

print("📤 Sube dataset6000.csv:")
uploaded = files.upload()

!mkdir -p /content/sample_data
shutil.move('dataset6000.csv', '/content/sample_data/dataset6000.csv')

# 3. Descargar script desde GitHub (si lo subes ahí)
# O usa: uploaded = files.upload() y selecciona acepten_colab.py

print("\n📤 Sube acepten_colab.py:")
uploaded = files.upload()

# 4. Ejecutar entrenamiento
print("\n🚀 Iniciando entrenamiento...")
!python acepten_colab.py

# 5. Descargar resultados
print("\n📥 Descargando resultados...")
files.download('best_model.pt')
files.download('dataset_generated.csv')
files.download('training_history.png')

print("\n✅ ¡TODO LISTO!")
```

---

## ⚙️ Ajustes Opcionales

### Cambiar número de épocas

Edita en `acepten_colab.py`, línea ~743:

```python
trainer.train(num_epochs=50)  # Por defecto: 30
```

### Cambiar tamaño del dataset

Línea ~713:

```python
df = generator.generate_full_dataset(
    pos_samples_per_dfa=50,  # Por defecto: 30
    neg_samples_per_dfa=50   # Por defecto: 30
)
```

### Cambiar batch size

Línea ~734:

```python
batch_size = 256  # Por defecto: 128 (óptimo para GPU)
```

---

## 📊 Tiempos Esperados (GPU T4 de Colab)

- **Generación de dataset**: ~3-5 minutos
- **Entrenamiento (30 épocas)**: ~15-20 minutos
- **Evaluación**: ~1 minuto
- **Total**: ~20-25 minutos

---

## 🎯 Resultados Esperados

### Tarea 1: Pertenencia a AFD
- **Meta**: Accuracy ≥ 0.90, F1 ≥ 0.90
- **Muy bueno**: Accuracy ≥ 0.95, F1 ≥ 0.95

### Tarea 2: Cadena Compartida
- **Meta**: F1 ≥ 0.80, PR-AUC ≥ 0.80
- **Bueno**: F1 ≥ 0.90, PR-AUC ≥ 0.90

---

## ⚠️ Troubleshooting

### Error: "No se encuentra dataset6000.csv"
```python
# Verifica la ruta
!ls /content/sample_data/
```

### Error: "CUDA out of memory"
Reduce el batch size:
```python
batch_size = 64  # en lugar de 128
```

### GPU no se detecta
- Verifica: `Runtime` → `Change runtime type` → `GPU`
- Reinicia el runtime: `Runtime` → `Restart runtime`

---

## 💾 Guardar en Google Drive

Para no perder resultados si se desconecta:

```python
from google.colab import drive
drive.mount('/content/drive')

# Al final, copia resultados a Drive
!cp best_model.pt /content/drive/MyDrive/
!cp dataset_generated.csv /content/drive/MyDrive/
!cp training_history.png /content/drive/MyDrive/
```

---

## 🔗 Enlaces Útiles

- [Google Colab](https://colab.research.google.com/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Límites de Colab](https://research.google.com/colaboratory/faq.html)

