import sounddevice as sd
import numpy as np
from collections import deque

############################################
# Configuración
############################################
INPUT_NAME_HINT  = "CABLE Output"   # VB-CABLE a la salida del sistema
OUTPUT_NAME_HINT = "Panasonic"      # Cambia por parte del nombre de tus audífonos/parlantes

SAMPLERATE  = 48000
BLOCK_SIZE  = 1024                   # ~21.3 ms a 48 kHz
CHANNELS    = 2

# Objetivo de nivel (RMS lineal ~0.25…0.35 es razonable)
TARGET_RMS  = 0.25

# AGC
MAX_GAIN        = 5.0                # tope de ganancia
RMS_FLOOR       = 1e-4               # no intentar amplificar por debajo de esto (ruido/silencio)
ATTACK_TIME_S   = 0.03               # qué tan rápido baja la ganancia si hay pico (ganancia ↓)
RELEASE_TIME_S  = 0.5                # qué tan lento sube la ganancia si está bajo (ganancia ↑)

# Limitador
LIMITER_THRESHOLD = 0.9              # techo
LOOKAHEAD_MS      = 5                # mira picos ~5 ms al futuro
SOFTCLIP_START    = 0.95             # región de soft-clip

# Latencia (mejor probar "low" o "high" si cruje)
LATENCY = "low"

############################################
# Utilidades
############################################
def db_to_alpha(time_s, samplerate, blocksize):
    # coeficiente exponencial por bloque
    tau = max(time_s, 1e-6)
    return np.exp(-blocksize / (tau * samplerate))

def find_device_index(name_hint, kind="input"):
    hits = []
    for i, dev in enumerate(sd.query_devices()):
        if name_hint.lower() in dev['name'].lower():
            if kind == "input"  and dev['max_input_channels']  >= CHANNELS:
                hits.append(i)
            if kind == "output" and dev['max_output_channels'] >= CHANNELS:
                hits.append(i)
    if not hits:
        raise RuntimeError(f"No encontré dispositivo '{name_hint}' ({kind}). Revisa sd.query_devices().")
    return hits[0]

def soft_clip(x, start=0.95):
    # Región lineal hasta 'start', luego curva suave (tanh) hacia 1.0
    a = np.abs(x)
    mask = a > start
    y = np.copy(x)
    if np.any(mask):
        sign = np.sign(x[mask])
        over = (a[mask] - start) / (1.0 - start)
        y[mask] = sign * (start + (1.0 - start) * np.tanh(over))
    return y

############################################
# Estado global del procesador
############################################
state = {
    "gain": 1.0,
    "attack_alpha": db_to_alpha(ATTACK_TIME_S, SAMPLERATE, BLOCK_SIZE),
    "release_alpha": db_to_alpha(RELEASE_TIME_S, SAMPLERATE, BLOCK_SIZE),
}

# Buffer de look-ahead por canal
lookahead_len = max(1, int(SAMPLERATE * LOOKAHEAD_MS / 1000 / BLOCK_SIZE))
look_buffers = [deque(maxlen=lookahead_len) for _ in range(CHANNELS)]

############################################
# Callback
############################################
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    x = indata.copy()  # (N, C)
    # RMS del bloque completo (estéreo)
    rms = float(np.sqrt(np.mean(np.square(x), axis=0)).mean())
    rms_eff = max(rms, RMS_FLOOR)

    # Ganancia objetivo para alcanzar TARGET_RMS
    desired_gain = TARGET_RMS / rms_eff

    # Smoothing con ataque/recuperación
    g_prev = state["gain"]
    if desired_gain < g_prev:
        # atacar rápido (bajar ganancia)
        alpha = state["attack_alpha"]
    else:
        # soltar lento (subir ganancia)
        alpha = state["release_alpha"]
    g = alpha * g_prev + (1 - alpha) * desired_gain
    g = min(g, MAX_GAIN)
    state["gain"] = g

    # Aplicar AGC
    y = x * g

    # Push al buffer de lookahead por canal
    # Guardamos el RMS pico por canal para decidir reducción previa
    peak_now = np.max(np.abs(y), axis=0)  # por canal
    for c in range(CHANNELS):
        look_buffers[c].append(peak_now[c])

    # Estimar pico futuro (máximo en la ventana de lookahead)
    if lookahead_len > 1 and all(len(b) == lookahead_len for b in look_buffers):
        future_peak = max(np.max(b) for b in look_buffers)
    else:
        future_peak = np.max(peak_now)

    # Si el pico futuro excede el umbral, recorta con factor de reducción
    if future_peak > LIMITER_THRESHOLD:
        reduce = LIMITER_THRESHOLD / (future_peak + 1e-9)
        y *= reduce

    # Soft-clip final de seguridad
    y = soft_clip(y, start=SOFTCLIP_START)

    # Asegurar rango [-1, 1] (float32)
    y = np.clip(y, -1.0, 1.0)
    outdata[:] = y.astype(np.float32)

############################################
# Arranque
############################################
def main():
    # Descubrir dispositivos por nombre
    in_idx  = find_device_index(INPUT_NAME_HINT, kind="input")
    out_idx = find_device_index(OUTPUT_NAME_HINT, kind="output")

    print(f"Usando IN: {sd.query_devices(in_idx)['name']}  |  OUT: {sd.query_devices(out_idx)['name']}")
    print("🎧 AGC+Limitador en tiempo real... Ctrl+C para salir.")

    with sd.Stream(
        device=(in_idx, out_idx),
        samplerate=SAMPLERATE,
        blocksize=BLOCK_SIZE,
        dtype='float32',
        channels=CHANNELS,
        latency=LATENCY,
        callback=audio_callback
    ):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🔴 Detenido por el usuario.")
    except Exception as e:
        print(f"❌ Error: {e}")
