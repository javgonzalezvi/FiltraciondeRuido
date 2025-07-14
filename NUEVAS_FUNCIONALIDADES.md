# 🎵 Guía de Nuevas Funcionalidades - Separación Espectral

## 📋 Resumen de Funcionalidades Agregadas

### 🆕 Funciones Principales Nuevas

1. **`separar_bandas_frecuencia()`** - Separación personalizable en bandas
2. **`separar_bandas_musicales()`** - Separación en componentes musicales típicos
3. **`crear_visualizacion_3d_espectral()`** - Visualización 3D del espectrograma
4. **`visualizar_separacion_bandas()`** - Gráficos de bandas separadas
5. **`visualizar_espectros_bandas()`** - Espectros FFT de cada banda
6. **`visualizar_componentes_musicales()`** - Gráficos de componentes musicales
7. **`guardar_bandas_separadas()`** - Exportar bandas como archivos WAV
8. **`guardar_componentes_musicales()`** - Exportar componentes musicales

### 🎛️ Sistema de Menú Interactivo

El programa ahora incluye un menú que permite elegir entre:

```
1️⃣  Análisis completo tradicional (FFT + Separación Demucs)
2️⃣  Separación por bandas de frecuencia personalizadas
3️⃣  Separación en componentes musicales típicos
4️⃣  Visualización 3D espectral
5️⃣  Análisis comparativo completo
```

## 🧮 Conceptos de Transformada de Fourier Implementados

### 1. **FFT para Separación por Bandas**
```python
# Calcular FFT de la señal completa
y_fft = fft(y)
freqs = fftfreq(N, 1/sr)

# Crear máscara para cada banda de frecuencia
mask = np.zeros_like(y_fft)
freq_indices = (np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)
mask[freq_indices] = 1

# Filtrar en dominio de frecuencia
y_fft_banda = y_fft * mask

# Transformada inversa para obtener señal temporal
y_banda = np.real(np.fft.ifft(y_fft_banda))
```

**Conceptos Demostrados:**
- **Descomposición espectral** de señales complejas
- **Filtrado selectivo** en el dominio de frecuencia
- **Reconstrucción temporal** mediante IFFT
- **Análisis de contenido frecuencial**

### 2. **STFT para Análisis Tiempo-Frecuencia**
```python
# Short-Time Fourier Transform para espectrograma 3D
S = librosa.stft(y, n_fft=ventana_tiempo, hop_length=ventana_tiempo//4)
S_mag = np.abs(S)
S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
```

**Conceptos Demostrados:**
- **Análisis localizado** en ventanas temporales
- **Resolución tiempo-frecuencia**
- **Representación 3D** de evolución espectral

## 🎵 Separación por Bandas de Frecuencia

### Bandas Logarítmicas Personalizables
El sistema crea bandas con distribución logarítmica:

```python
freq_limits = np.logspace(np.log10(20), np.log10(freq_max), num_bandas + 1)
```

**Ejemplo con 4 bandas para audio de 44.1 kHz:**
- Banda 1: 20 - 315 Hz (Graves profundos)
- Banda 2: 315 - 1,414 Hz (Medios-graves)
- Banda 3: 1,414 - 6,325 Hz (Medios-agudos)
- Banda 4: 6,325 - 22,050 Hz (Agudos)

### Visualización Múltiple
- **Dominio temporal**: Cada banda como función del tiempo
- **Dominio frecuencial**: Espectro FFT de cada banda
- **Codificación por colores**: Fácil identificación visual

## 🎼 Separación en Componentes Musicales

### 7 Bandas Musicales Estándar

| Componente | Rango (Hz) | Instrumentos/Sonidos |
|------------|------------|----------------------|
| **Sub-Bajos** | 20-60 | Frecuencias muy profundas |
| **Bajos** | 60-250 | Bajo, bombo, fundamentos |
| **Medios-Bajos** | 250-500 | Voz masculina, graves |
| **Medios** | 500-2000 | Voz principal, guitarras |
| **Medios-Altos** | 2000-4000 | Voz aguda, presencia |
| **Agudos** | 4000-8000 | Detalles, brillo |
| **Super-Agudos** | 8000+ | Armónicos altos |

### Aplicaciones Prácticas
- **Ecualización inteligente**
- **Análisis de mezcla musical**
- **Identificación de instrumentos**
- **Procesamiento selectivo por banda**

## 🌈 Visualización 3D Espectral

### Espectrograma Interactivo
```python
# Crear superficie 3D: Tiempo x Frecuencia x Magnitud
T, F = np.meshgrid(times, freqs[:len(freqs)//2])
Z = S_db[:len(freqs)//2, :]
ax.plot_surface(T, F, Z, cmap='plasma', alpha=0.8)
```

**Interpretación Visual:**
- **Eje X**: Tiempo (evolución temporal)
- **Eje Y**: Frecuencia (contenido espectral)
- **Eje Z**: Magnitud (intensidad en dB)
- **Colores**: Mapa de calor de intensidades

## 🔬 Análisis Comparativo

### Comparación Múltiple
El análisis comparativo muestra:

1. **Señal original** completa
2. **Bandas espectrales** separadas por FFT
3. **Componentes musicales** típicos
4. **Visualización 3D** del contenido espectral

### Beneficios Educativos
- **Comprensión visual** de la descomposición espectral
- **Comparación directa** de métodos
- **Análisis completo** en una sola ejecución

## 📁 Archivos de Salida

### Bandas Espectrales
```
audio_banda_1_20-315Hz.wav
audio_banda_2_315-1414Hz.wav
audio_banda_3_1414-6325Hz.wav
audio_banda_4_6325-22050Hz.wav
```

### Componentes Musicales
```
audio_Sub_Bajos.wav
audio_Bajos.wav
audio_Medios_Bajos.wav
audio_Medios.wav
audio_Medios_Altos.wav
audio_Agudos.wav
audio_Super_Agudos.wav
```

## 🚀 Instrucciones de Uso

### 1. Preparar Audio de Prueba
```bash
python demo_bandas.py
```

### 2. Ejecutar Programa Principal
```bash
python CargadeAudio.py
```

### 3. Seleccionar Funcionalidad
- Introduce el nombre del archivo (ej: `audio_demo.wav`)
- Elige opción del menú (2-5 para nuevas funcionalidades)
- Sigue las instrucciones interactivas

### 4. Experimentar
- Prueba diferentes números de bandas (2-8)
- Compara resultados entre métodos
- Analiza archivos musicales reales
- Exporta componentes para análisis posterior

## 🎯 Objetivos Pedagógicos Logrados

### Conceptos Demostrados Visualmente
1. **Descomposición espectral** mediante FFT
2. **Filtrado selectivo** en frecuencia
3. **Análisis tiempo-frecuencia** con STFT
4. **Reconstrucción de señales** con IFFT
5. **Separación de componentes** espectrales

### Aplicaciones Prácticas
- **Procesamiento de audio musical**
- **Análisis de voz y habla**
- **Reducción de ruido selectiva**
- **Ecualización automática**
- **Investigación acústica**

Esta implementación proporciona una **herramienta educativa completa** para entender y visualizar los conceptos fundamentales de la Transformada de Fourier aplicada al procesamiento de señales de audio.
