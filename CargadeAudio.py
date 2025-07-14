# CargadeAudio.py
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import librosa
import librosa.display
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import write
import subprocess
import torchaudio

def cargar_audio():
    """Pide al usuario la ruta del archivo de audio y lo carga."""
    file_path = input("🔍 Ingresa el nombre del archivo de audio (.wav): ").strip()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ El archivo {file_path} no existe.")
    
    y, sr = librosa.load(file_path, sr=None)
    print(f"✔ Audio cargado. Tasa de muestreo: {sr} Hz | Muestras: {len(y)}")
    
    return y, sr, file_path

def visualizar_tiempo(signal, title):
    """Grafica la señal en el dominio del tiempo."""
    plt.figure(figsize=(12, 3))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.show()

def aplicar_filtro_pasabanda(y, sr, lowcut=200, highcut=3500, order=5):
    """Filtra la señal en un rango de frecuencias típicas de la voz humana."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    
    return y_filtered

def visualizar_fft(signal, sr, title):
    """Grafica el espectro de magnitudes."""
    N = len(signal)
    T = 1.0 / sr
    y_fft = fft(signal)
    freqs = fftfreq(N, T)[:N // 2]
    magnitude = np.abs(y_fft[:N // 2])
    plt.figure(figsize=(12, 3))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualizar_spectrograma(signal, sr, title):
    """Genera y muestra un espectrograma."""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def separar_fuentes_demucs(file_path):
    try:
        # Crear carpeta de salida
        output_dir = "output_demucs"
        os.makedirs(output_dir, exist_ok=True)

        print("🎧 Ejecutando Demucs para separar voz e instrumentos...")

        # Ejecutar comando de separación
        subprocess.run(
            ["demucs", "--two-stems", "vocals", "--out", output_dir, file_path],
            check=True
        )

        # Obtener nombres de archivo
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        vocals_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")
        instr_path = os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")

        if not os.path.exists(vocals_path) or not os.path.exists(instr_path):
            raise FileNotFoundError("❌ No se encontraron los archivos separados por Demucs.")

        # Cargar los audios separados
        y_vocals, sr = torchaudio.load(vocals_path)
        y_instr, _ = torchaudio.load(instr_path)

        # Convertir a numpy y normalizar
        y_vocals = y_vocals[0].numpy()
        y_instr = y_instr[0].numpy()

        print("✔ Separación completada con Demucs.")
        return y_vocals, y_instr, vocals_path, instr_path

    except Exception as e:
        print("⚠️ Error:", e)
        raise RuntimeError("❌ Error al ejecutar Demucs.") from e

def suprimir_ruido_por_spectrograma(y, sr, prop_ruido=0.1, factor_atenuacion=1.5):
    """
    Suprime ruido de fondo usando espectrograma y máscara espectral suave.
    """
    # Convertir a mono si es estéreo
    if len(y.shape) == 2:
        y = y.mean(axis=0)

    # STFT compleja
    S = librosa.stft(y)
    
    # Magnitud y fase
    S_mag, S_phase = np.abs(S), np.angle(S)

    # Estimar el perfil promedio del ruido con los frames más bajos
    n_frames_ruido = max(1, int(S_mag.shape[1] * prop_ruido))
    perfil_ruido = np.mean(np.sort(S_mag, axis=1)[:, :n_frames_ruido], axis=1)
    umbral = perfil_ruido[:, np.newaxis] * factor_atenuacion

    # Máscara suave
    mask = S_mag > umbral
    S_mag_denoised = S_mag * mask

    # Reconstrucción
    S_denoised = S_mag_denoised * np.exp(1j * S_phase)
    y_denoised = librosa.istft(S_denoised)

    # Normalizar para guardar el .wav sin distorsión
    y_denoised /= np.max(np.abs(y_denoised) + 1e-8)

    return y_denoised

def guardar_audio(nombre_archivo, signal, sr):
    """Guarda la señal de audio en formato .wav"""
    write(nombre_archivo, sr, np.int16(signal * 32767))
    print(f"💾 Audio guardado: {nombre_archivo}")

def comparar_senales(original, voz, instr):
    """Grafica todas las señales juntas para comparación visual."""
    plt.figure(figsize=(12, 4))
    plt.plot(original, label='Original', alpha=0.5)
    plt.plot(voz, label='Voz', alpha=0.7)
    plt.plot(instr, label='Instrumentos', alpha=0.7)
    plt.title("Comparación Temporal: Original vs Voz vs Instrumentos")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.tight_layout()
    plt.show()

def separar_bandas_frecuencia(y, sr, num_bandas=4, visualizar=True):
    """
    Separa una señal de audio en múltiples bandas de frecuencia usando FFT.
    
    Args:
        y: Señal de audio
        sr: Tasa de muestreo
        num_bandas: Número de bandas a crear (default: 4)
        visualizar: Si mostrar gráficos (default: True)
    
    Returns:
        bandas: Lista de señales separadas por banda
        freq_ranges: Rangos de frecuencia de cada banda
    """
    print(f"🎵 Separando señal en {num_bandas} bandas espectrales...")
    
    # Calcular FFT
    N = len(y)
    y_fft = fft(y)
    freqs = fftfreq(N, 1/sr)
    
    # Definir rangos de frecuencia para cada banda
    freq_max = sr // 2  # Frecuencia de Nyquist
    freq_ranges = []
    bandas = []
    
    # Crear bandas logarítmicas para mejor distribución
    freq_limits = np.logspace(np.log10(20), np.log10(freq_max), num_bandas + 1)
    
    for i in range(num_bandas):
        # Definir límites de la banda
        freq_low = freq_limits[i]
        freq_high = freq_limits[i + 1]
        freq_ranges.append((freq_low, freq_high))
        
        # Crear máscara para filtrar la banda
        mask = np.zeros_like(y_fft)
        
        # Aplicar filtro en dominio de frecuencia (FFT)
        freq_indices = (np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)
        mask[freq_indices] = 1
        
        # Filtrar la señal en el dominio de frecuencia
        y_fft_banda = y_fft * mask
        
        # Transformada inversa para obtener señal filtrada
        y_banda = np.real(np.fft.ifft(y_fft_banda))
        bandas.append(y_banda)
        
        print(f"   Banda {i+1}: {freq_low:.1f} - {freq_high:.1f} Hz")
    
    if visualizar:
        visualizar_separacion_bandas(y, bandas, freq_ranges, sr)
        visualizar_espectros_bandas(bandas, freq_ranges, sr)
    
    return bandas, freq_ranges

def separar_bandas_musicales(y, sr, visualizar=True):
    """
    Separa una señal de audio en bandas musicales típicas.
    
    Args:
        y: Señal de audio
        sr: Tasa de muestreo
        visualizar: Si mostrar gráficos
    
    Returns:
        componentes: Diccionario con las señales separadas
    """
    print("🎼 Separando en componentes musicales...")
    
    # Definir bandas musicales típicas
    bandas_musicales = {
        'Sub-Bajos': (20, 60),      # Frecuencias muy bajas
        'Bajos': (60, 250),         # Instrumentos de bajo, bombo
        'Medios-Bajos': (250, 500), # Voz masculina, instrumentos graves
        'Medios': (500, 2000),      # Voz principal, guitarras
        'Medios-Altos': (2000, 4000), # Voz aguda, platillos
        'Agudos': (4000, 8000),     # Detalles, brillo
        'Super-Agudos': (8000, sr//2) # Armónicos altos
    }
    
    # Calcular FFT
    N = len(y)
    y_fft = fft(y)
    freqs = fftfreq(N, 1/sr)
    
    componentes = {}
    
    for nombre, (freq_low, freq_high) in bandas_musicales.items():
        # Crear máscara para la banda
        mask = np.zeros_like(y_fft)
        freq_indices = (np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)
        mask[freq_indices] = 1
        
        # Filtrar y reconstruir
        y_fft_banda = y_fft * mask
        y_banda = np.real(np.fft.ifft(y_fft_banda))
        componentes[nombre] = y_banda
        
        print(f"   {nombre}: {freq_low} - {freq_high} Hz")
    
    if visualizar:
        visualizar_componentes_musicales(y, componentes, sr)
    
    return componentes

def visualizar_separacion_bandas(original, bandas, freq_ranges, sr):
    """Visualiza la señal original y las bandas separadas."""
    num_bandas = len(bandas)
    fig, axes = plt.subplots(num_bandas + 1, 1, figsize=(15, 2 * (num_bandas + 1)))
    
    # Señal original
    time_axis = np.arange(len(original)) / sr
    axes[0].plot(time_axis, original, color='black', alpha=0.7)
    axes[0].set_title("🎵 Señal Original", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(True, alpha=0.3)
    
    # Bandas separadas
    colors = plt.cm.viridis(np.linspace(0, 1, num_bandas))
    for i, (banda, freq_range, color) in enumerate(zip(bandas, freq_ranges, colors)):
        axes[i + 1].plot(time_axis, banda, color=color, alpha=0.8)
        axes[i + 1].set_title(f"🎛️ Banda {i+1}: {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz", 
                             fontsize=11, fontweight='bold')
        axes[i + 1].set_ylabel("Amplitud")
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Tiempo (segundos)")
    plt.tight_layout()
    plt.show()

def visualizar_espectros_bandas(bandas, freq_ranges, sr):
    """Visualiza los espectros de frecuencia de cada banda."""
    num_bandas = len(bandas)
    fig, axes = plt.subplots(2, (num_bandas + 1) // 2, figsize=(15, 8))
    axes = axes.flatten() if num_bandas > 1 else [axes]
    
    colors = plt.cm.plasma(np.linspace(0, 1, num_bandas))
    
    for i, (banda, freq_range, color) in enumerate(zip(bandas, freq_ranges, colors)):
        # Calcular FFT de la banda
        N = len(banda)
        banda_fft = fft(banda)
        freqs = fftfreq(N, 1/sr)[:N//2]
        magnitude = np.abs(banda_fft[:N//2])
        
        # Graficar espectro
        axes[i].plot(freqs, magnitude, color=color, linewidth=1.5)
        axes[i].set_title(f"Espectro Banda {i+1}\n{freq_range[0]:.1f} - {freq_range[1]:.1f} Hz", 
                         fontsize=10, fontweight='bold')
        axes[i].set_xlabel("Frecuencia (Hz)")
        axes[i].set_ylabel("Magnitud")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, freq_range[1] * 1.2)
    
    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def visualizar_componentes_musicales(original, componentes, sr):
    """Visualiza los componentes musicales separados."""
    num_comp = len(componentes)
    fig, axes = plt.subplots(num_comp + 1, 1, figsize=(15, 2 * (num_comp + 1)))
    
    # Señal original
    time_axis = np.arange(len(original)) / sr
    axes[0].plot(time_axis, original, color='black', alpha=0.7)
    axes[0].set_title("🎵 Señal Musical Original", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(True, alpha=0.3)
    
    # Componentes musicales
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FFB347']
    for i, (nombre, componente) in enumerate(componentes.items()):
        color = colors[i % len(colors)]
        axes[i + 1].plot(time_axis, componente, color=color, alpha=0.8)
        axes[i + 1].set_title(f"🎛️ {nombre}", fontsize=11, fontweight='bold')
        axes[i + 1].set_ylabel("Amplitud")
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Tiempo (segundos)")
    plt.tight_layout()
    plt.show()

def crear_visualizacion_3d_espectral(y, sr, ventana_tiempo=2048):
    """
    Crea una visualización 3D del espectrograma mostrando evolución temporal de frecuencias.
    """
    print("🌈 Creando visualización 3D espectral...")
    
    # Calcular STFT para análisis tiempo-frecuencia
    S = librosa.stft(y, n_fft=ventana_tiempo, hop_length=ventana_tiempo//4)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    
    # Configurar ejes para gráfico 3D
    freqs = librosa.fft_frequencies(sr=sr, n_fft=ventana_tiempo)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, 
                                   hop_length=ventana_tiempo//4)
    
    # Crear malla 3D
    T, F = np.meshgrid(times, freqs[:len(freqs)//2])
    Z = S_db[:len(freqs)//2, :]
    
    # Gráfico 3D
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_surface(T, F, Z, cmap='plasma', alpha=0.8)
    
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_zlabel('Magnitud (dB)')
    ax.set_title('🌈 Visualización 3D: Evolución Espectral en el Tiempo', 
                 fontsize=14, fontweight='bold')
    
    # Añadir barra de color
    fig.colorbar(surface, shrink=0.5, aspect=5, label='Magnitud (dB)')
    
    plt.tight_layout()
    plt.show()

def guardar_bandas_separadas(bandas, freq_ranges, sr, nombre_base):
    """Guarda cada banda como archivo de audio independiente."""
    print("💾 Guardando bandas separadas...")
    
    for i, (banda, freq_range) in enumerate(zip(bandas, freq_ranges)):
        # Normalizar la banda
        banda_norm = banda / (np.max(np.abs(banda)) + 1e-8)
        
        # Nombre del archivo
        nombre_archivo = f"{nombre_base}_banda_{i+1}_{freq_range[0]:.0f}-{freq_range[1]:.0f}Hz.wav"
        
        # Guardar
        write(nombre_archivo, sr, np.int16(banda_norm * 32767))
        print(f"   ✅ {nombre_archivo}")

def guardar_componentes_musicales(componentes, sr, nombre_base):
    """Guarda cada componente musical como archivo independiente."""
    print("💾 Guardando componentes musicales...")
    
    for nombre, componente in componentes.items():
        # Normalizar
        comp_norm = componente / (np.max(np.abs(componente)) + 1e-8)
        
        # Nombre del archivo
        nombre_archivo = f"{nombre_base}_{nombre.replace('-', '_').replace(' ', '_')}.wav"
        
        # Guardar
        write(nombre_archivo, sr, np.int16(comp_norm * 32767))
        print(f"   ✅ {nombre_archivo}")

def mostrar_menu():
    """Muestra el menú principal de opciones."""
    print("\n" + "="*60)
    print("🎵 SISTEMA DE ANÁLISIS Y SEPARACIÓN ESPECTRAL DE AUDIO 🎵")
    print("="*60)
    print("1️⃣  Análisis completo tradicional (FFT + Separación Demucs)")
    print("2️⃣  Separación por bandas de frecuencia personalizadas")
    print("3️⃣  Separación en componentes musicales típicos")
    print("4️⃣  Visualización 3D espectral")
    print("5️⃣  Análisis comparativo completo")
    print("0️⃣  Salir")
    print("="*60)
    return input("🔍 Selecciona una opción: ").strip()

def ejecutar_analisis_tradicional(y, sr, file_path):
    """Ejecuta el análisis tradicional completo."""
    print("\n🎯 Ejecutando análisis tradicional...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Visualizar señal original
    visualizar_tiempo(y, "Señal de Audio Original (Dominio Temporal)")
    
    # Filtrar ruido
    y_filtrada = aplicar_filtro_pasabanda(y, sr)
    visualizar_tiempo(y_filtrada, "Señal Filtrada (200-3500 Hz)")
    
    # Espectros
    visualizar_fft(y, sr, "Espectro - Original")
    visualizar_fft(y_filtrada, sr, "Espectro - Filtrada")
    
    # Espectrogramas
    visualizar_spectrograma(y, sr, "Espectrograma - Original")
    visualizar_spectrograma(y_filtrada, sr, "Espectrograma - Filtrada")
    
    # Intentar separar con Demucs (opcional)
    try:
        y_voz, y_instr, voz_path, instr_path = separar_fuentes_demucs(file_path)
        
        # Espectrogramas separados
        visualizar_spectrograma(y_voz, sr, "Espectrograma - Solo Voz")
        visualizar_spectrograma(y_instr, sr, "Espectrograma - Solo Instrumentos")
        
        # Comparación final
        comparar_senales(y, y_voz, y_instr)
        
        # Guardar resultados
        guardar_audio(f"{base_name}_filtrado.wav", y_filtrada, sr)
        guardar_audio(f"{base_name}_voz.wav", y_voz, sr)
        guardar_audio(f"{base_name}_instrumentos.wav", y_instr, sr)
        
    except Exception as e:
        print(f"⚠️ Demucs no disponible: {e}")
        print("🔄 Continuando con análisis espectral básico...")
        
        # Análisis alternativo sin Demucs
        print("\n📊 Realizando separación básica por bandas...")
        bandas_basicas, freq_ranges = separar_bandas_frecuencia(y, sr, 3, visualizar=False)
        
        # Simular separación voz/instrumentos con bandas
        voz_simulada = bandas_basicas[1]  # Banda media (voz típica)
        instr_simulado = bandas_basicas[0] + bandas_basicas[2]  # Bandas baja + alta
        
        # Comparación alternativa
        comparar_senales(y, voz_simulada, instr_simulado)
        
        # Guardar resultados básicos
        guardar_audio(f"{base_name}_filtrado.wav", y_filtrada, sr)
        guardar_audio(f"{base_name}_voz_estimada.wav", voz_simulada, sr)
        guardar_audio(f"{base_name}_instrumentos_estimados.wav", instr_simulado, sr)

def ejecutar_separacion_bandas(y, sr, file_path):
    """Ejecuta la separación por bandas de frecuencia."""
    print("\n🎛️ Separación por bandas de frecuencia...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Preguntar número de bandas
    try:
        num_bandas = int(input("🔢 ¿Cuántas bandas quieres crear? (2-8, recomendado: 4): ") or "4")
        num_bandas = max(2, min(8, num_bandas))  # Limitar entre 2 y 8
    except ValueError:
        num_bandas = 4
        print("⚠️ Valor inválido, usando 4 bandas por defecto.")
    
    # Separar en bandas
    bandas, freq_ranges = separar_bandas_frecuencia(y, sr, num_bandas, visualizar=True)
    
    # Preguntar si guardar
    guardar = input("💾 ¿Guardar bandas como archivos separados? (s/N): ").strip().lower()
    if guardar in ['s', 'si', 'yes', 'y']:
        guardar_bandas_separadas(bandas, freq_ranges, sr, base_name)
    
    return bandas, freq_ranges

def ejecutar_componentes_musicales(y, sr, file_path):
    """Ejecuta la separación en componentes musicales."""
    print("\n🎼 Separación en componentes musicales...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Separar componentes
    componentes = separar_bandas_musicales(y, sr, visualizar=True)
    
    # Preguntar si guardar
    guardar = input("💾 ¿Guardar componentes como archivos separados? (s/N): ").strip().lower()
    if guardar in ['s', 'si', 'yes', 'y']:
        guardar_componentes_musicales(componentes, sr, base_name)
    
    return componentes

def ejecutar_analisis_comparativo(y, sr, file_path):
    """Ejecuta un análisis comparativo completo."""
    print("\n🔬 Análisis comparativo completo...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print("\n📊 1. Separación en 4 bandas espectrales...")
    bandas, freq_ranges = separar_bandas_frecuencia(y, sr, 4, visualizar=False)
    
    print("\n📊 2. Separación en componentes musicales...")
    componentes = separar_bandas_musicales(y, sr, visualizar=False)
    
    print("\n📊 3. Análisis FFT tradicional...")
    visualizar_fft(y, sr, "Espectro Original - Análisis Completo")
    
    print("\n📊 4. Visualización 3D...")
    crear_visualizacion_3d_espectral(y, sr)
    
    print("\n📊 5. Comparación visual final...")
    # Crear una visualización comparativa especial
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    time_axis = np.arange(len(y)) / sr
    
    # Original
    axes[0].plot(time_axis, y, color='black', alpha=0.7, label='Original')
    axes[0].set_title("🎵 Señal Original", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Amplitud")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Bandas espectrales combinadas
    colors_bandas = plt.cm.viridis(np.linspace(0, 1, len(bandas)))
    for i, (banda, color) in enumerate(zip(bandas, colors_bandas)):
        axes[1].plot(time_axis, banda, color=color, alpha=0.6, 
                    label=f'Banda {i+1} ({freq_ranges[i][0]:.0f}-{freq_ranges[i][1]:.0f}Hz)')
    axes[1].set_title("🎛️ Bandas Espectrales Separadas", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Amplitud")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Algunos componentes musicales principales
    comp_principales = ['Bajos', 'Medios', 'Agudos']
    colors_comp = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for comp, color in zip(comp_principales, colors_comp):
        if comp in componentes:
            axes[2].plot(time_axis, componentes[comp], color=color, alpha=0.7, label=comp)
    axes[2].set_title("🎼 Componentes Musicales Principales", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Tiempo (segundos)")
    axes[2].set_ylabel("Amplitud")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Guardar todo si el usuario quiere
    guardar_todo = input("💾 ¿Guardar todos los componentes y bandas? (s/N): ").strip().lower()
    if guardar_todo in ['s', 'si', 'yes', 'y']:
        guardar_bandas_separadas(bandas, freq_ranges, sr, base_name)
        guardar_componentes_musicales(componentes, sr, base_name)

# -----------------------------------
# FUNCIÓN PRINCIPAL MEJORADA
# -----------------------------------
def main():
    """Función principal con menú interactivo."""
    try:
        # Cargar archivo una sola vez
        y, sr, file_path = cargar_audio()
        
        while True:
            opcion = mostrar_menu()
            
            if opcion == "0":
                print("👋 ¡Hasta luego!")
                break
            elif opcion == "1":
                ejecutar_analisis_tradicional(y, sr, file_path)
            elif opcion == "2":
                ejecutar_separacion_bandas(y, sr, file_path)
            elif opcion == "3":
                ejecutar_componentes_musicales(y, sr, file_path)
            elif opcion == "4":
                print("\n🌈 Creando visualización 3D...")
                crear_visualizacion_3d_espectral(y, sr)
            elif opcion == "5":
                ejecutar_analisis_comparativo(y, sr, file_path)
            else:
                print("⚠️ Opción inválida. Intenta de nuevo.")
            
            # Preguntar si continuar
            continuar = input("\n🔄 ¿Quieres realizar otro análisis? (S/n): ").strip().lower()
            if continuar in ['n', 'no']:
                break
        
        print("\n✅ ¡Sesión completada!")
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        
    except KeyboardInterrupt:
        print("\n\n👋 ¡Análisis interrumpido por el usuario!")

# -----------------------------------
# FUNCIÓN PRINCIPAL ORIGINAL (PRESERVADA)
# -----------------------------------
def main_original():
    try:
        # Paso 1: Cargar archivo
        y, sr, file_path = cargar_audio()
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Paso 2: Visualizar señal original
        visualizar_tiempo(y, "Señal de Audio Original (Dominio Temporal)")

        # Paso 3: Filtrar ruido (frecuencias fuera del rango vocal)
        y_filtrada = aplicar_filtro_pasabanda(y, sr)
        visualizar_tiempo(y_filtrada, "Señal Filtrada (200-3500 Hz)")

        # Paso 4: Espectros
        visualizar_fft(y, sr, "Espectro - Original")
        visualizar_fft(y_filtrada, sr, "Espectro - Filtrada")

        # Paso 5: Espectrogramas
        visualizar_spectrograma(y, sr, "Espectrograma - Original")
        visualizar_spectrograma(y_filtrada, sr, "Espectrograma - Filtrada")

        # Paso 6: Separar con Spleeter
        y_voz, y_instr, voz_path, instr_path = separar_fuentes_demucs(file_path)

        #y_voz_limpia = suprimir_ruido_por_spectrograma(y_voz, sr)
        #visualizar_spectrograma(y_voz_limpia, sr, "Espectrograma - Voz Limpia (Denoised)")

        # Paso 7: Espectrogramas separados
        visualizar_spectrograma(y_voz, sr, "Espectrograma - Solo Voz")
        visualizar_spectrograma(y_instr, sr, "Espectrograma - Solo Instrumentos")

        # Paso 8: Comparación final
        comparar_senales(y, y_voz, y_instr)

        # Paso 9: Guardar resultados
        guardar_audio(f"{base_name}_filtrado.wav", y_filtrada, sr)
        guardar_audio(f"{base_name}_voz.wav", y_voz, sr)
        guardar_audio(f"{base_name}_instrumentos.wav", y_instr, sr)
        #guardar_audio(f"{base_name}_voz_limpia.wav", y_voz_limpia, sr)

        print("\n✅ ¡Proceso completado con éxito!")

    except Exception as e:
        print(f"⚠️ Error: {e}")

# -----------------------------------
# EJECUCIÓN
# -----------------------------------
if __name__ == "__main__":
    main()
