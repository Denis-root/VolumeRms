import sounddevice as sd
import numpy as np

# ===========================
# CONFIGURACIÓN DE DISPOSITIVOS
# ===========================
INPUT_DEVICE = 20    # VB-Cable WASAPI
OUTPUT_DEVICE = 17   # Panasonic HDMI WASAPI (según tu lista)
SAMPLERATE = 48000   # típicamente 48 kHz para HDMI y VB-Cable
CHANNELS = 2
BLOCKSIZE = 1024     # tamaño de bloque en frames

# ===========================
# FUNCIONES DE AUDIO
# ===========================
def limiter(audio, threshold=0.6):
    """
    Limita la señal de audio para que no supere el umbral.
    """
    return np.clip(audio, -threshold, threshold)

# ===========================
# STREAMS SEPARADOS
# ===========================
try:
    with sd.InputStream(device=INPUT_DEVICE,
                        channels=CHANNELS,
                        samplerate=SAMPLERATE,
                        blocksize=BLOCKSIZE,
                        dtype='float32') as ins, \
         sd.OutputStream(device=OUTPUT_DEVICE,
                         channels=CHANNELS,
                         samplerate=SAMPLERATE,
                         blocksize=BLOCKSIZE,
                         dtype='float32') as outs:

        print("🎧 Limitador en tiempo real corriendo... Ctrl+C para salir")
        while True:
            # Leemos del input
            data, overflowed = ins.read(BLOCKSIZE)
            # Aplicamos limitador
            processed = limiter(data)
            # Escribimos al output
            outs.write(processed)

except KeyboardInterrupt:
    print("\n🛑 Proceso detenido por el usuario")
except Exception as e:
    print("❌ Error:", e)
