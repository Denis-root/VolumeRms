import sys, threading, time
import numpy as np
import sounddevice as sd

from PyQt5 import QtWidgets, QtCore

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    HAS_PG = False

# --------- DSP utils ----------
def db_to_alpha(time_s, samplerate, blocksize):
    tau = max(time_s, 1e-6)
    return np.exp(-blocksize / (tau * samplerate))

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

# --------- Audio Processor (sounddevice) ----------
class AudioProcessor:
    def __init__(self):
        self.samplerate = 48000
        self.blocksize  = 1024
        self.channels   = 2
        self.latency    = "low"

        # Runtime controls (shared with GUI)
        self.state_lock = threading.Lock()
        self.state = {
            "TARGET_RMS": 0.25,
            "MAX_GAIN":   3.0,
            "MASTER_VOL": 1.0,
            "MUTED": False,
            "gain_agc": 1.0,
            "running": False,
            "in_idx": None,
            "out_idx": None,
            # meters
            "rms_in": 0.0,
            "rms_out": 0.0,
            "peak_out": 0.0,
        }

        # AGC & limiter params
        self.RMS_FLOOR = 1e-4
        self.ATTACK_TIME_S  = 0.03
        self.RELEASE_TIME_S = 0.6
        self.LIMITER_THRESHOLD = 0.90
        self.SOFTCLIP_START   = 0.95

        self.attack_alpha  = db_to_alpha(self.ATTACK_TIME_S,  self.samplerate, self.blocksize)
        self.release_alpha = db_to_alpha(self.RELEASE_TIME_S, self.samplerate, self.blocksize)

        self.stream = None

    def list_devices(self):
        return sd.query_devices()

    def set_devices(self, in_idx, out_idx):
        with self.state_lock:
            self.state["in_idx"] = in_idx
            self.state["out_idx"] = out_idx

    def set_target_rms(self, v):
        with self.state_lock:
            self.state["TARGET_RMS"] = float(v)

    def set_master_vol(self, v):
        with self.state_lock:
            self.state["MASTER_VOL"] = float(v)

    def set_max_gain(self, v):
        with self.state_lock:
            self.state["MAX_GAIN"] = float(v)

    def toggle_mute(self, on):
        with self.state_lock:
            self.state["MUTED"] = bool(on)

    def is_running(self):
        with self.state_lock:
            return self.state["running"]

    def start(self):
        if self.is_running():
            return
        with self.state_lock:
            in_idx  = self.state["in_idx"]
            out_idx = self.state["out_idx"]
        if in_idx is None or out_idx is None:
            raise RuntimeError("Selecciona dispositivos de entrada y salida.")

        def cb(indata, outdata, frames, time_info, status):
            if status:
                print(status)

            x = indata.astype(np.float32, copy=False)

            with self.state_lock:
                TARGET_RMS = self.state["TARGET_RMS"]
                MAX_GAIN   = self.state["MAX_GAIN"]
                MASTER_VOL = self.state["MASTER_VOL"]
                MUTED      = self.state["MUTED"]
                g_prev     = self.state["gain_agc"]

            # --- meters in ---
            rms_in = float(np.sqrt(np.mean(np.square(x), axis=0)).mean())
            rms_eff = max(rms_in, self.RMS_FLOOR)

            # --- AGC ---
            desired_gain = TARGET_RMS / rms_eff
            alpha = self.attack_alpha if desired_gain < g_prev else self.release_alpha
            g = alpha * g_prev + (1 - alpha) * desired_gain
            g = min(g, MAX_GAIN)

            y = x * g

            # --- limiter + softclip ---
            peak = np.max(np.abs(y))
            if peak > self.LIMITER_THRESHOLD:
                y *= (self.LIMITER_THRESHOLD / (peak + 1e-12))
            y = soft_clip(y, start=self.SOFTCLIP_START)

            # --- master fader ---
            if MUTED:
                y *= 0.0
            else:
                # lineal (simple y predecible). Si querés ley log, lo cambiamos.
                mv = max(0.0, min(MASTER_VOL, 1.5))
                y *= mv

            y = np.clip(y, -1.0, 1.0)
            outdata[:] = y

            # meters out + persist state
            rms_out = float(np.sqrt(np.mean(np.square(y), axis=0)).mean())
            peak_out = float(np.max(np.abs(y)))

            with self.state_lock:
                self.state["gain_agc"] = g
                self.state["rms_in"] = rms_in
                self.state["rms_out"] = rms_out
                self.state["peak_out"] = peak_out

        self.stream = sd.Stream(
            device=(in_idx, out_idx),
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32',
            channels=self.channels,
            latency=self.latency,
            callback=cb
        )
        self.stream.start()
        with self.state_lock:
            self.state["running"] = True

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.state_lock:
            self.state["running"] = False

    def get_meters(self):
        with self.state_lock:
            return (
                self.state["rms_in"],
                self.state["rms_out"],
                self.state["peak_out"],
                self.state["gain_agc"],
                self.state["TARGET_RMS"],
                self.state["MASTER_VOL"],
                self.state["MAX_GAIN"],
                self.state["MUTED"],
            )

# --------- GUI (PyQt5) ----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGC + Limiter (VB-Cable) — PyQt5")
        self.proc = AudioProcessor()

        # Layouts
        main = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        main.addLayout(grid)

        # Device selectors
        self.in_combo  = QtWidgets.QComboBox()
        self.out_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refrescar dispositivos")

        grid.addWidget(QtWidgets.QLabel("Entrada (VB-CABLE):"), 0, 0)
        grid.addWidget(self.in_combo, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Salida (parlantes/auriculares):"), 1, 0)
        grid.addWidget(self.out_combo, 1, 1, 1, 2)
        grid.addWidget(self.refresh_btn, 2, 0, 1, 3)

        # Controls
        self.target_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.target_slider.setRange(5, 50)   # 0.05 .. 0.50
        self.target_slider.setValue(25)
        self.target_label = QtWidgets.QLabel("Target RMS: 0.25")

        self.master_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.master_slider.setRange(0, 150)  # 0.00 .. 1.50
        self.master_slider.setValue(100)
        self.master_label = QtWidgets.QLabel("Master Vol: 1.00")

        self.maxgain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maxgain_slider.setRange(100, 600)  # 1.00 .. 6.00
        self.maxgain_slider.setValue(300)
        self.maxgain_label = QtWidgets.QLabel("Max Gain: 3.00")

        self.mute_check = QtWidgets.QCheckBox("Mute")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        row = 3
        grid.addWidget(self.target_label, row, 0)
        grid.addWidget(self.target_slider, row, 1, 1, 2); row += 1
        grid.addWidget(self.master_label, row, 0)
        grid.addWidget(self.master_slider, row, 1, 1, 2); row += 1
        grid.addWidget(self.maxgain_label, row, 0)
        grid.addWidget(self.maxgain_slider, row, 1, 1, 2); row += 1
        grid.addWidget(self.mute_check, row, 0)
        grid.addWidget(self.start_btn, row, 1)
        grid.addWidget(self.stop_btn,  row, 2); row += 1

        # Meters
        meter_box = QtWidgets.QGroupBox("Monitoreo")
        meter_layout = QtWidgets.QGridLayout(meter_box)
        self.rms_in_lbl  = QtWidgets.QLabel("RMS In: 0.000")
        self.rms_out_lbl = QtWidgets.QLabel("RMS Out: 0.000")
        self.gain_lbl    = QtWidgets.QLabel("Gain AGC: 1.00")
        self.peak_lbl    = QtWidgets.QLabel("Peak Out: 0.00")
        meter_layout.addWidget(self.rms_in_lbl,  0, 0)
        meter_layout.addWidget(self.rms_out_lbl, 0, 1)
        meter_layout.addWidget(self.gain_lbl,    1, 0)
        meter_layout.addWidget(self.peak_lbl,    1, 1)
        main.addWidget(meter_box)

        # Graph (optional)
        if HAS_PG:
            self.plot = pg.PlotWidget()
            self.plot.setYRange(0, 1.0)
            self.plot.showGrid(x=True, y=True, alpha=0.3)
            self.curve = self.plot.plot(pen=None, symbol='o', symbolSize=6)
            self.level_history = [0.0]*60
            main.addWidget(self.plot)
        else:
            self.plot = None

        # Wiring
        self.refresh_btn.clicked.connect(self.fill_devices)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.mute_check.toggled.connect(self.proc.toggle_mute)

        self.target_slider.valueChanged.connect(self.on_target_changed)
        self.master_slider.valueChanged.connect(self.on_master_changed)
        self.maxgain_slider.valueChanged.connect(self.on_maxgain_changed)

        # Timers
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(100)  # 10 Hz
        self.update_timer.timeout.connect(self.refresh_meters)
        self.update_timer.start()

        self.fill_devices()

    def fill_devices(self):
        self.in_combo.clear()
        self.out_combo.clear()
        devs = self.proc.list_devices()
        for idx, d in enumerate(devs):
            name = d["name"]
            ins, outs = d["max_input_channels"], d["max_output_channels"]
            if ins >= self.proc.channels:
                self.in_combo.addItem(f"[{idx}] {name}", userData=idx)
            if outs >= self.proc.channels:
                self.out_combo.addItem(f"[{idx}] {name}", userData=idx)

        # heurística: preseleccionar VB-CABLE en IN
        for i in range(self.in_combo.count()):
            if "cable" in self.in_combo.itemText(i).lower():
                self.in_combo.setCurrentIndex(i)
                break

    def on_target_changed(self, v):
        t = round(v/100.0, 2)  # 0.05..0.50
        t = max(0.05, min(t, 0.50))
        self.target_label.setText(f"Target RMS: {t:.2f}")
        self.proc.set_target_rms(t)

    def on_master_changed(self, v):
        mv = round(v/100.0, 2)  # 0.00..1.50
        mv = max(0.0, min(mv, 1.50))
        self.master_label.setText(f"Master Vol: {mv:.2f}")
        self.proc.set_master_vol(mv)

    def on_maxgain_changed(self, v):
        mg = round(v/100.0, 2)  # 1.00..6.00
        mg = max(1.0, min(mg, 6.0))
        self.maxgain_label.setText(f"Max Gain: {mg:.2f}")
        self.proc.set_max_gain(mg)

    def on_start(self):
        in_idx  = self.in_combo.currentData()
        out_idx = self.out_combo.currentData()
        if in_idx is None or out_idx is None:
            QtWidgets.QMessageBox.warning(self, "Dispositivos", "Selecciona entrada y salida.")
            return
        self.proc.set_devices(in_idx, out_idx)
        try:
            self.proc.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al iniciar", str(e))

    def on_stop(self):
        self.proc.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def refresh_meters(self):
        rms_in, rms_out, peak_out, gain, *_ = self.proc.get_meters()
        self.rms_in_lbl.setText(f"RMS In: {rms_in:.3f}")
        self.rms_out_lbl.setText(f"RMS Out: {rms_out:.3f}")
        self.gain_lbl.setText(f"Gain AGC: {gain:.2f}")
        self.peak_lbl.setText(f"Peak Out: {peak_out:.2f}")

        if HAS_PG and self.plot is not None:
            # simple stripchart del nivel de salida
            self.level_history.append(rms_out)
            if len(self.level_history) > 60:
                self.level_history = self.level_history[-60:]
            x = np.arange(len(self.level_history))
            self.curve.setData(x, self.level_history)

    def closeEvent(self, e):
        self.proc.stop()
        super().closeEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(740, 520 if HAS_PG else 360)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
