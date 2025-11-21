# ✅ Checklist para Google Colab

## 📋 Antes de Subir a Colab

- [x] ✅ Archivos verificados
- [x] ✅ Dependencias instaladas localmente
- [x] ✅ Dataset validado (6000 AFDs)
- [x] ✅ Parser funcionando
- [x] ✅ Modelo funcional (1.9M parámetros)

---

## 🚀 Pasos en Google Colab

### 1. Configuración Inicial
```
□ Abrir https://colab.research.google.com/
□ Crear nuevo notebook
□ Runtime → Change runtime type → GPU ⚡
□ Verificar que dice "GPU" en la esquina superior derecha
```

### 2. Instalar Dependencias (Celda 1)
```python
!pip install -q torch pandas numpy scikit-learn matplotlib tqdm
```
```
□ Ejecutar celda
□ Esperar instalación (~1 minuto)
```

### 3. Subir Dataset (Celda 2)
```python
from google.colab import files
import shutil

print("📤 Sube dataset6000.csv:")
uploaded = files.upload()

!mkdir -p /content/sample_data
shutil.move('dataset6000.csv', '/content/sample_data/dataset6000.csv')

print("✅ Dataset subido!")
```
```
□ Ejecutar celda
□ Hacer clic en "Choose Files"
□ Seleccionar dataset6000.csv (~3 MB)
□ Esperar carga (~30 segundos)
```

### 4. Subir Script (Celda 3)
```python
from google.colab import files

print("📤 Sube acepten_colab.py:")
uploaded = files.upload()

print("✅ Script subido!")
```
```
□ Ejecutar celda
□ Seleccionar acepten_colab.py
□ Esperar carga (~5 segundos)
```

### 5. Verificar GPU (Celda 4)
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No disponible'}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
```
```
□ Ejecutar celda
□ Verificar que muestra nombre de GPU (ej: "Tesla T4")
```

### 6. Entrenar Modelo (Celda 5) ⏱️
```python
!python acepten_colab.py
```
```
□ Ejecutar celda
□ ☕ Esperar ~20-25 minutos
□ Ver progreso con barras de tqdm
```

**Qué verás durante el entrenamiento:**
```
🤖 MODELO DUAL-ENCODER PARA AFDs - GOOGLE COLAB
✅ GPU Detectada: Tesla T4
1️⃣  Cargando AFDs...
2️⃣  Generando dataset...
   [████████████████████] 6000/6000 AFDs
3️⃣  Calculando Y2...
4️⃣  Dividiendo dataset...
5️⃣  Creando dataloaders...
6️⃣  Creando modelo...
7️⃣  Entrenando...
   Época 01/30 | Train Loss: X.XXXX Y1: X.XXXX Y2: X.XXXX | Val Loss: ...
   Época 02/30 | ...
   ...
8️⃣  Evaluando en test set...
   📊 TAREA 1: Pertenencia a AFD (Y1)
      Accuracy: X.XXXX
      F1 Score: X.XXXX
      ✅ Rendimiento: MUY BUENO / BUENO / REGULAR
   
   📊 TAREA 2: Cadena compartida entre AFDs (Y2)
      Accuracy: X.XXXX
      F1 Score: X.XXXX
      PR-AUC:   X.XXXX
      ✅ Rendimiento: BUENO / REGULAR
9️⃣  Generando visualizaciones...
✅ PIPELINE COMPLETO!
```

### 7. Descargar Resultados (Celda 6)
```python
from google.colab import files

files.download('best_model.pt')
files.download('dataset_generated.csv')
files.download('training_history.png')

print("✅ Archivos descargados!")
```
```
□ Ejecutar celda
□ Verificar 3 descargas en tu navegador
```

---

## 📁 Archivos que Obtendrás

```
✅ best_model.pt            (~7.5 MB)  - Modelo entrenado
✅ dataset_generated.csv    (~12 MB)   - Dataset completo con y1, y2
✅ training_history.png     (~100 KB)  - Gráficas de entrenamiento
```

---

## 🎯 Métricas Esperadas

### Tarea 1: Pertenencia a AFD
- **Objetivo**: Accuracy ≥ 0.90, F1 ≥ 0.90
- **Muy bueno**: Accuracy ≥ 0.95, F1 ≥ 0.95

### Tarea 2: Cadena Compartida
- **Objetivo**: F1 ≥ 0.80, PR-AUC ≥ 0.80
- **Bueno**: F1 ≥ 0.90, PR-AUC ≥ 0.90

---

## ⚠️ Troubleshooting

### ❌ "Runtime disconnected"
```
□ Volver a conectar: Runtime → Reconnect
□ Re-ejecutar todas las celdas
```

### ❌ "CUDA out of memory"
```
□ Runtime → Restart runtime
□ Editar acepten_colab.py: batch_size = 64
□ Re-ejecutar
```

### ❌ "No module named 'torch'"
```
□ Re-ejecutar celda de instalación de dependencias
□ Verificar que no haya errores en la instalación
```

### ❌ "File not found: dataset6000.csv"
```
□ Verificar ruta: !ls /content/sample_data/
□ Re-subir archivo si es necesario
```

---

## 💾 Guardar en Google Drive (Opcional)

Añade al final (Celda 7):
```python
from google.colab import drive
drive.mount('/content/drive')

!cp best_model.pt /content/drive/MyDrive/
!cp dataset_generated.csv /content/drive/MyDrive/
!cp training_history.png /content/drive/MyDrive/

print("✅ Guardado en Google Drive!")
```

---

## 📊 Tiempos Estimados (GPU T4)

| Fase | Tiempo |
|------|--------|
| Setup + instalación | 2 min |
| Subir archivos | 1 min |
| Generación dataset | 3 min |
| Entrenamiento (30 épocas) | 15-20 min |
| Evaluación | 1 min |
| **TOTAL** | **~25 min** |

---

## 🎉 ¡Listo!

Una vez que veas:
```
✅ PIPELINE COMPLETO!
```

¡Ya tienes tu modelo entrenado! 🚀

Revisa las métricas y las gráficas para evaluar el rendimiento.

---

**🔗 Más información**: 
- `COLAB_INSTRUCTIONS.md` - Instrucciones detalladas
- `RESUMEN.md` - Arquitectura y detalles técnicos
- `README.md` - Documentación completa

