# Demo de las nuevas funcionalidades
# Ejecutar con: python demo_bandas.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io.wavfile import write

def crear_audio_demo():
    """Crea una señal de audio sintética para demostrar la separación por bandas."""
    
    # Parámetros
    sr = 44100  # Tasa de muestreo
    duracion = 3  # segundos
    t = np.linspace(0, duracion, int(sr * duracion))
    
    # Crear componentes en diferentes bandas de frecuencia
    # Simular "bajos" - frecuencias bajas (50-200 Hz)
    bajos = np.sin(2 * np.pi * 80 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    
    # Simular "medios" - voz/guitarra (200-2000 Hz)
    medios = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)
    
    # Simular "agudos" - platillos/detalles (2000-8000 Hz)
    agudos = 0.3 * np.sin(2 * np.pi * 4000 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t)
    
    # Agregar modulación de amplitud para hacer más interesante
    mod_bajos = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)
    mod_medios = 1 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    mod_agudos = 1 + 0.4 * np.sin(2 * np.pi * 3 * t)
    
    # Aplicar modulación
    bajos *= mod_bajos
    medios *= mod_medios
    agudos *= mod_agudos
    
    # Combinar todas las componentes
    señal_completa = bajos + medios + agudos
    
    # Normalizar
    señal_completa = señal_completa / np.max(np.abs(señal_completa))
    
    # Guardar como archivo WAV
    write("audio_demo.wav", sr, np.int16(señal_completa * 32767))
    
    print("✅ Audio demo creado: 'audio_demo.wav'")
    print("   - Componente de bajos: 80-120 Hz")
    print("   - Componente de medios: 440-880 Hz") 
    print("   - Componente de agudos: 4000-6000 Hz")
    print("   - Duración: 3 segundos")
    
    return señal_completa, sr

if __name__ == "__main__":
    print("🎵 Creando audio de demostración para la separación por bandas...")
    crear_audio_demo()
    print("\n🚀 Ahora ejecuta el programa principal con: python CargadeAudio.py")
    print("   y usa 'audio_demo.wav' como archivo de entrada.")
