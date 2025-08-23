import sounddevice as sd
import numpy as np
import threading
import time
import msvcrt

############################################
# Configuración de dispositivos por nombre
############################################
INPUT_NAME_HINT  = "CABLE Output"   # VB-CABLE (entrada desde el sistema)
OUTPUT_NAME_HINT = "Panasonic"      # Cambiá por el nombre (o parte) de tu salida física

SAMPLERATE  = 48000
BLOCK_SIZE  = 1024
CHANNELS    = 2

# Control en tiempo real
state = {
    "TARGET_RMS": 0.25,   # meta de nivel (AGC)
    "MAX_GAIN": 3.0,      # tope de amplificación del AGC
    "MASTER_VOL": 1.00,   # fader manual post-procesamiento (0..1.5)
    "MUTED": False,
    "gain_agc": 1.0,      # estado interno
}

# AGC
RMS_FLOOR       = 1e-4
ATTACK_TIME_S   = 0.03
RELEASE_TIME_S  = 0.6

# Limitador
LIMITER_THRESHOLD = 0.9
SOFTCLIP_START    = 0.95

def db_to_alpha(time_s, samplerate, blocksize):
    tau = max(time_s, 1e-6)
    return np.exp(-blocksize / (tau * samplerate))

attack_alpha  = db_to_alpha(ATTACK_TIME_S, SAMPLERATE, BLOCK_SIZE)
release_alpha = db_to_alpha(RELEASE_TIME_S, SAMPLERATE, BLOCK_SIZE)

def soft_clip(x, start=0.95):
    a = np.abs(x)
    mask = a > start
    if np.any(mask):
        y = np.copy(x)
        sign = np.sign(x[mask])
        over = (a[mask] - start) / (1.0 - start)
        y[mask] = sign * (start + (1.0 - start) * np.tanh(over))
        return y
    return x

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

def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(status)

    x = indata.astype(np.float32, copy=False)

    # --- AGC: medir RMS del bloque (estéreo) ---
    rms = float(np.sqrt(np.mean(np.square(x), axis=0)).mean())
    rms_eff = max(rms, RMS_FLOOR)
    desired_gain = state["TARGET_RMS"] / rms_eff

    g_prev = state["gain_agc"]
    alpha = attack_alpha if desired_gain < g_prev else release_alpha
    g = alpha * g_prev + (1 - alpha) * desired_gain
    g = min(g, state["MAX_GAIN"])
    state["gain_agc"] = g

    y = x * g

    # --- Limitador + softclip de seguridad ---
    peak = np.max(np.abs(y))
    if peak > LIMITER_THRESHOLD:
        y *= (LIMITER_THRESHOLD / (peak + 1e-12))
    y = soft_clip(y, start=SOFTCLIP_START)

    # --- Fader manual (MASTER_VOL) ---
    if state["MUTED"]:
        y *= 0.0
    else:
        # escala exponencial suave para sensación más natural del fader
        # (lineal también sirve: y *= state["MASTER_VOL"])
        mv = float(state["MASTER_VOL"])
        mv = max(0.0, min(mv, 1.5))
        y *= (10 ** (np.interp(mv, [0.0, 1.0, 1.5], [-60, 0, +3]) / 20.0))

    # Output
    outdata[:] = np.clip(y, -1.0, 1.0)

def key_listener():
    print("\nControles en vivo:  [ ] MASTER_VOL  |  - = TARGET_RMS  |  m mute  |  q salir")
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('q', 'Q'):
                break
            elif ch == '[':  # bajar master
                state["MASTER_VOL"] = max(0.0, state["MASTER_VOL"] - 0.05)
            elif ch == ']':  # subir master
                state["MASTER_VOL"] = min(1.5, state["MASTER_VOL"] + 0.05)
            elif ch == '-':  # bajar target
                state["TARGET_RMS"] = max(0.05, state["TARGET_RMS"] - 0.02)
            elif ch == '=':  # subir target (tecla =/+, según teclado)
                state["TARGET_RMS"] = min(0.50, state["TARGET_RMS"] + 0.02)
            elif ch in ('m', 'M'):
                state["MUTED"] = not state["MUTED"]

            print(f"MASTER_VOL={state['MASTER_VOL']:.2f}  TARGET_RMS={state['TARGET_RMS']:.2f}  "
                  f"MAX_GAIN={state['MAX_GAIN']:.2f}  MUTED={state['MUTED']}")

        time.sleep(0.02)

def main():
    in_idx  = find_device_index(INPUT_NAME_HINT, kind="input")
    out_idx = find_device_index(OUTPUT_NAME_HINT, kind="output")

    print(f"IN:  {sd.query_devices(in_idx)['name']}")
    print(f"OUT: {sd.query_devices(out_idx)['name']}")

    t = threading.Thread(target=key_listener, daemon=True)
    t.start()

    print("🎧 Procesando... q para salir")
    with sd.Stream(
        device=(in_idx, out_idx),
        samplerate=SAMPLERATE,
        blocksize=BLOCK_SIZE,
        dtype='float32',
        channels=CHANNELS,
        latency="low",
        callback=audio_callback
    ):
        try:
            while t.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    print("🔴 Fin.")

if __name__ == "__main__":
    main()
