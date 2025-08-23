import sounddevice as sd
import numpy as np
import queue

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
SAMPLE_RATE = 48000      # igual que tu sistema (48kHz es estándar en Windows)
BLOCKSIZE = 1024         # tamaño de bloque (baja = menos latencia)
CHANNELS = 2             # estéreo
LOOKAHEAD = 5            # ms de lookahead (buffer pequeño)
THRESHOLD = 0.8          # nivel máximo permitido (0.0 - 1.0)
ATTACK = 0.05            # qué tan rápido baja la ganancia
RELEASE = 0.001          # qué tan rápido recupera volumen

# -------------------------------
# Buffers para lookahead
# -------------------------------
lookahead_samples = int(SAMPLE_RATE * LOOKAHEAD / 1000)
buffer = queue.Queue(maxsize=lookahead_samples)

gain = 1.0

def limiter(block):
    global gain
    # Calcular pico del bloque
    peak = np.max(np.abs(block))

    # Si se pasa del umbral, reducimos ganancia (attack rápido)
    if peak > THRESHOLD:
        target_gain = THRESHOLD / (peak + 1e-9)
        gain = min(gain, target_gain)  # solo bajar, nunca subir de golpe
    else:
        # Recuperación gradual (release)
        gain = min(1.0, gain + RELEASE)

    # Aplicar ganancia al bloque
    return block * gain

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("⚠️", status)

    # Guardar bloque en buffer (lookahead)
    buffer.put(indata.copy())

    if buffer.qsize() >= lookahead_samples:
        # Sacar bloque atrasado (lookahead delay)
        delayed = buffer.get()
        outdata[:] = limiter(delayed)
    else:
        # Si aún no hay suficiente lookahead, silencio
        outdata[:] = np.zeros_like(indata)

# -------------------------------
# Main: abrir stream
# -------------------------------
# Paso 1: Identificar dispositivos disponibles
# print("\nDispositivos de audio disponibles:")
# print(sd.query_devices())

# IMPORTANTE: Ajusta estos índices según tu setup
INPUT_DEVICE = 20   # el nombre exacto de VB-CABLE
OUTPUT_DEVICE = 5    # pon aquí el nombre de tu salida real

with sd.Stream(device=(INPUT_DEVICE, OUTPUT_DEVICE),
               samplerate=SAMPLE_RATE,
               blocksize=BLOCKSIZE,
               dtype='float32',
               channels=CHANNELS,
               callback=audio_callback):
    print("\n🎧 Limitador en tiempo real corriendo... Ctrl+C para salir.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n👋 Saliendo...")
