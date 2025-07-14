# CargadeAudio.py
import os
import numpy as np
import matplotlib.pyplot as plt
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

# -----------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------
def main():
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
