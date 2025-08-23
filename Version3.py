import sounddevice as sd
import numpy as np

############################################
# Hiperparámetros ajustables
############################################

INPUT_DEVICE = 20  # Índice del VB-Cable (entrada)
OUTPUT_DEVICE = 17  # Índice de la salida real (tu Panasonic, auriculares, etc.)

BLOCK_SIZE = 1024  # Tamaño de cada bloque de audio
SAMPLERATE = 48000  # Frecuencia de muestreo

TARGET_LEVEL = 0.5  # Volumen objetivo [0.0 - 1.0], 0.5 ≈ mitad de amplitud percibida
MAX_GAIN = 5.0  # Ganancia máxima permitida para evitar boost extremo
LIMITER_THRESHOLD = 0.9  # Nivel máximo absoluto antes de clipping [0.0 - 1.0]
SMOOTHING = 0.1  # Constante para suavizar cambios de ganancia (0-1)


############################################
# Callback de procesamiento en tiempo real
############################################
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Calcular RMS del bloque
    rms = np.sqrt(np.mean(indata ** 2))

    # Ganancia para mantener nivel objetivo
    gain = TARGET_LEVEL / (rms + 1e-6)

    # Suavizado de ganancia para evitar cambios bruscos
    if not hasattr(audio_callback, "prev_gain"):
        audio_callback.prev_gain = 1.0
    gain = audio_callback.prev_gain * (1 - SMOOTHING) + gain * SMOOTHING
    audio_callback.prev_gain = min(gain, MAX_GAIN)

    # Aplicar ganancia y limitar
    processed = np.clip(indata * audio_callback.prev_gain, -LIMITER_THRESHOLD, LIMITER_THRESHOLD)

    outdata[:] = processed


############################################
# Stream de audio en tiempo real
############################################
try:
    with sd.Stream(device=(INPUT_DEVICE, OUTPUT_DEVICE),
                   samplerate=SAMPLERATE,
                   blocksize=BLOCK_SIZE,
                   dtype='float32',
                   channels=2,
                   callback=audio_callback):
        print("🎧 Limitador/Compresor en tiempo real corriendo... Ctrl+C para salir.")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("🔴 Proceso detenido por el usuario.")

except Exception as e:
    print(f"❌ Error: {e}")
