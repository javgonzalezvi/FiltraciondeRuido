# Instrucciones para ejecutar el proyecto

## ✅ ¡El proyecto ya está funcionando!

### 🚀 **Ejecución Correcta:**
```bash
.\.venv\Scripts\python.exe CargadeAudio.py
```

### ⚠️ **Error Común:**
❌ NO usar: `python CargadeAudio.py` (usa Python del sistema sin dependencias)
✅ USAR: `.\.venv\Scripts\python.exe CargadeAudio.py` (usa entorno virtual)

## 🎵 **Archivos de Audio Disponibles:**
- `audio_demo.wav` - Audio sintético creado por demo_bandas.py
- `entrevista.wav` - Audio de entrevista existente
- `song.wav` - Canción existente
- `sfx1.wav` - Efectos de sonido existentes

## 🎛️ **Nuevas Funcionalidades a Probar:**

### 2️⃣ Separación por Bandas de Frecuencia
- Elige 2-8 bandas espectrales
- Ve la separación visual de cada banda
- Exporta cada banda como archivo independiente

### 3️⃣ Componentes Musicales
- 7 bandas musicales típicas (Bajos, Medios, Agudos, etc.)
- Ideal para analizar música

### 4️⃣ Visualización 3D
- Espectrograma 3D interactivo
- Evolución temporal de frecuencias

### 5️⃣ Análisis Comparativo
- Todos los métodos en una sola ejecución
- Comparación visual completa

## 🔧 **Solución para Demucs (Opcional):**
Si quieres usar la separación por IA (opción 1), instala Demucs globalmente:
```bash
pip install demucs
```

## 🎯 **Recomendación:**
1. Ejecuta: `.\.venv\Scripts\python.exe demo_bandas.py` para crear audio de prueba
2. Ejecuta: `.\.venv\Scripts\python.exe CargadeAudio.py`
3. Usa `audio_demo.wav` como archivo de entrada
4. Prueba las opciones 2, 3, 4, o 5 (son las nuevas funcionalidades)

## 📊 **Conceptos de FFT Demostrados:**
- ✅ Separación espectral por bandas
- ✅ Filtrado en dominio de frecuencia
- ✅ Visualización 3D de espectrogramas
- ✅ Análisis tiempo-frecuencia
- ✅ Reconstrucción temporal con IFFT
