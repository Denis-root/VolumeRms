import sys, threading, time
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    HAS_PG = False

# ------------ utilidades dBFS ------------
def lin_to_dbfs(x):
    x = max(x, 1e-12)
    return 20.0 * np.log10(x)

def dbfs_to_lin(db):
    return 10.0 ** (db / 20.0)

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

# ------------- Procesador ---------------
class AudioProcessor:
    def __init__(self):
        self.fs        = 48000
        self.blocksize = 1024
        self.ch        = 2
        self.latency   = "low"

        # Control ÚNICO (dBFS)
        self.crest_margin_db = 12.0   # margen típico voz/música
        self.level_ceiling_db = -6.0  # slider principal (techo del limitador)
        self.max_gain_db      = 18.0  # límite de subida del AGC (+18 dB máx)
        self.master_vol       = 1.0   # fijo (si querés lo exponemos luego)

        # Estados/medidores
        self.state_lock = threading.Lock()
        self.running = False
        self.in_idx  = None
        self.out_idx = None
        self.gain_agc_lin = 1.0
        self.rms_in_db  = -120.0
        self.rms_out_db = -120.0
        self.peak_out_db = -120.0

        # AGC
        self.rms_floor = 1e-5
        self.attack_a  = db_to_alpha(0.03, self.fs, self.blocksize)
        self.release_a = db_to_alpha(0.60, self.fs, self.blocksize)

        self.stream = None

    # --- API GUI ---
    def list_devices(self):
        return sd.query_devices()

    def set_devices(self, in_idx, out_idx):
        with self.state_lock:
            self.in_idx  = in_idx
            self.out_idx = out_idx

    def set_ceiling_db(self, db):
        with self.state_lock:
            self.level_ceiling_db = float(db)

    # --- DSP callback ---
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)

        x = indata.astype(np.float32, copy=False)

        # RMS in (estéreo → promedio canales)
        rms_in = float(np.sqrt(np.mean(np.square(x), axis=0)).mean())
        rms_in = max(rms_in, self.rms_floor)
        rms_in_db = lin_to_dbfs(rms_in)

        with self.state_lock:
            ceiling_db   = self.level_ceiling_db
            crest_margin = self.crest_margin_db
            max_gain_db  = self.max_gain_db
            g_prev       = self.gain_agc_lin
            mv           = self.master_vol

        # Objetivo RMS en dBFS
        target_rms_db = ceiling_db - crest_margin
        target_rms_lin = dbfs_to_lin(target_rms_db)

        # Ganancia deseada para alcanzar target
        desired_gain_lin = target_rms_lin / rms_in

        # Límite de ganancia máxima (en dB)
        desired_gain_db = lin_to_dbfs(desired_gain_lin)
        desired_gain_db = min(desired_gain_db, max_gain_db)
        desired_gain_lin = dbfs_to_lin(desired_gain_db)

        # Ataque/Release (suavizado de la ganancia)
        if desired_gain_lin < g_prev:
            a = self.attack_a
        else:
            a = self.release_a
        g = a * g_prev + (1 - a) * desired_gain_lin

        # Aplicar AGC
        y = x * g

        # Limitador: techo en dBFS
        peak = float(np.max(np.abs(y)))
        peak_db = lin_to_dbfs(peak) if peak > 1e-12 else -120.0
        if peak_db > ceiling_db:
            # reducción necesaria en dB
            reduce_db = ceiling_db - peak_db
            y *= dbfs_to_lin(reduce_db)

        # soft-clip de seguridad (por si redondeos)
        y = soft_clip(y, start=0.98)

        # Master (fijo por ahora)
        y *= mv

        y = np.clip(y, -1.0, 1.0)
        outdata[:] = y

        # Medidores de salida
        rms_out = float(np.sqrt(np.mean(np.square(y), axis=0)).mean())
        rms_out = max(rms_out, 1e-12)
        rms_out_db  = lin_to_dbfs(rms_out)
        peak_out_db = lin_to_dbfs(float(np.max(np.abs(y)))) if np.max(np.abs(y)) > 1e-12 else -120.0

        with self.state_lock:
            self.gain_agc_lin = g
            self.rms_in_db    = rms_in_db
            self.rms_out_db   = rms_out_db
            self.peak_out_db  = peak_out_db

    def start(self):
        with self.state_lock:
            if self.running:
                return
            if self.in_idx is None or self.out_idx is None:
                raise RuntimeError("Selecciona entrada y salida en el panel.")
            in_idx, out_idx = self.in_idx, self.out_idx

        self.stream = sd.Stream(
            device=(in_idx, out_idx),
            samplerate=self.fs,
            blocksize=self.blocksize,
            dtype='float32',
            channels=self.ch,
            latency=self.latency,
            callback=self._callback
        )
        self.stream.start()
        with self.state_lock:
            self.running = True

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.state_lock:
            self.running = False

    def meters(self):
        with self.state_lock:
            return (self.rms_in_db, self.rms_out_db, self.peak_out_db, self.level_ceiling_db)

# ------------- GUI ----------------
class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Limitador/AGC a dBFS — PyQt5")
        self.proc = AudioProcessor()

        main = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout(); main.addLayout(grid)

        self.in_combo  = QtWidgets.QComboBox()
        self.out_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refrescar dispositivos")

        grid.addWidget(QtWidgets.QLabel("Entrada (VB-CABLE o Loopback):"), 0, 0)
        grid.addWidget(self.in_combo, 0, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Salida (parlantes/auriculares):"), 1, 0)
        grid.addWidget(self.out_combo, 1, 1, 1, 2)
        grid.addWidget(self.refresh_btn, 2, 0, 1, 3)

        # Slider ÚNICO: techo en dBFS
        self.ceiling_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ceiling_slider.setRange(-36, -1)   # de -36 dBFS a -1 dBFS
        self.ceiling_slider.setValue(-6)
        self.ceiling_label  = QtWidgets.QLabel("Nivel máx (techo): -6 dBFS  |  Target RMS: -18 dBFS (margen 12 dB)")

        row = 3
        grid.addWidget(self.ceiling_label, row, 0, 1, 3); row += 1
        grid.addWidget(self.ceiling_slider, row, 0, 1, 3); row += 1

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        grid.addWidget(self.start_btn, row, 1)
        grid.addWidget(self.stop_btn,  row, 2); row += 1

        # Meters
        meter_box = QtWidgets.QGroupBox("Monitoreo (dBFS)")
        meter_layout = QtWidgets.QGridLayout(meter_box)
        self.rms_in_lbl  = QtWidgets.QLabel("RMS In: -∞ dBFS")
        self.rms_out_lbl = QtWidgets.QLabel("RMS Out: -∞ dBFS")
        self.peak_out_lbl= QtWidgets.QLabel("Peak Out: -∞ dBFS")
        meter_layout.addWidget(self.rms_in_lbl,  0, 0)
        meter_layout.addWidget(self.rms_out_lbl, 0, 1)
        meter_layout.addWidget(self.peak_out_lbl,1, 0)
        main.addWidget(meter_box)

        # Graph
        if HAS_PG:
            self.plot = pg.PlotWidget()
            self.plot.showGrid(x=True, y=True, alpha=0.3)
            self.plot.setLabel('left', 'dBFS')
            self.plot.setLabel('bottom', 'Tiempo (s, ~6s ventana)')
            self.plot.setYRange(-60, 0)  # dBFS
            self.curve_rms  = self.plot.plot(pen=pg.mkPen(width=2))
            self.curve_peak = self.plot.plot(pen=pg.mkPen(style=QtCore.Qt.DashLine, width=1))
            self.hist_rms  = [-60.0]*60
            self.hist_peak = [-60.0]*60
            main.addWidget(self.plot)
        else:
            self.plot = None

        # Wiring
        self.refresh_btn.clicked.connect(self.fill_devices)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.ceiling_slider.valueChanged.connect(self.on_ceiling_changed)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh_meters)
        self.timer.start()

        self.fill_devices()
        self.on_ceiling_changed(self.ceiling_slider.value())

    def fill_devices(self):
        self.in_combo.clear(); self.out_combo.clear()
        devs = self.proc.list_devices()
        for idx, d in enumerate(devs):
            name = d["name"]; ins, outs = d["max_input_channels"], d["max_output_channels"]
            if ins  >= self.proc.ch: self.in_combo.addItem(f"[{idx}] {name}", userData=idx)
            if outs >= self.proc.ch: self.out_combo.addItem(f"[{idx}] {name}", userData=idx)
        # Heurística: preseleccionar VB-CABLE en IN
        for i in range(self.in_combo.count()):
            if "cable" in self.in_combo.itemText(i).lower():
                self.in_combo.setCurrentIndex(i); break

    def on_ceiling_changed(self, val_db):
        ceiling = int(val_db)
        target  = ceiling - int(self.proc.crest_margin_db)
        self.ceiling_label.setText(f"Nivel máx (techo): {ceiling} dBFS  |  Target RMS: {target} dBFS (margen {int(self.proc.crest_margin_db)} dB)")
        self.proc.set_ceiling_db(ceiling)

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
        rms_in, rms_out, peak_out, ceiling = self.proc.meters()
        self.rms_in_lbl.setText(f"RMS In: {rms_in:5.1f} dBFS")
        self.rms_out_lbl.setText(f"RMS Out: {rms_out:5.1f} dBFS")
        self.peak_out_lbl.setText(f"Peak Out: {peak_out:5.1f} dBFS")

        if HAS_PG and self.plot is not None:
            self.hist_rms.append(rms_out);  self.hist_peak.append(peak_out)
            if len(self.hist_rms)  > 60: self.hist_rms  = self.hist_rms[-60:]
            if len(self.hist_peak) > 60: self.hist_peak = self.hist_peak[-60:]
            x = np.linspace(-6, 0, len(self.hist_rms))
            self.curve_rms.setData(x,  self.hist_rms)
            self.curve_peak.setData(x, self.hist_peak)

    def closeEvent(self, e):
        self.proc.stop()
        super().closeEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    w.resize(800, 520 if HAS_PG else 320)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
