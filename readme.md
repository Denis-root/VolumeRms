# Limitador/AGC en tiempo real (PyQt5 + sounddevice)

Procesa audio del sistema **en vivo**: captura → AGC → limitador → soft-clip → salida. Incluye GUI con medidores y gráfica (si está PyQtGraph).

---

## Requisitos
- Windows 10/11
- **VB-CABLE Virtual Audio Device** instalado
- Python 3.9–3.12

---

## Instalación
1. Instalá **VB-CABLE** (driver) y reiniciá si lo pide.
2. Cloná el repo:
   - $ git clone https://github.com/Denis-root/VolumeRms.git
   - $ cd VolumeRms
3. (Opcional) Creá y activá un entorno virtual:
   - $ python -m venv .venv
   - $ .venv\Scripts\activate
4. Instalá dependencias (archivo **instalaciones**):
   - $ pip install -r instalaciones

---

## Configuración de audio en Windows
1. **Salida del sistema → VB-CABLE**
   - Configuración de Sonido → seleccioná **CABLE Input (VB-Audio Virtual Cable)** como **Salida** predeterminada.
   - Resultado: TODO el audio de Windows entra a la app por la entrada virtual.
2. **Parlantes/Audífonos reales**
   - No los pongás como salida del sistema.
   - Los vas a elegir **dentro de la GUI** como **Salida** de monitoreo.

---

## Ejecutar
- $ python Version8.py

---

## Uso (GUI)
1. **Entrada**: elegí el dispositivo que contenga **“CABLE”** (la línea virtual donde Windows está enviando el audio).
2. **Salida**: elegí **tus parlantes/auriculares físicos** (Realtek, USB, HDMI, etc.).
3. **Refrescar dispositivos**: usá el botón si no aparecen.
4. **Nivel máx (techo)**: dejá **−6 dBFS** para empezar.
   - El **RMS objetivo** se calcula como `techo − margen de cresta` (margen por defecto **2 dB**).
5. **Start** para comenzar a procesar, **Stop** para detener.

---

## Flujo de señal
Windows (Salida) → VB-CABLE → [Entrada de la app] → AGC + Limitador + Soft-Clip → [Salida de la app] → Parlantes/Audífonos

---

## Tips rápidos de ajuste
- **Pumping/respira**: bajá `max_gain_db` (p.ej. 12–18 dB) o aumentá el *release* en el código (0.30 s → 0.5–1.0 s).
- **Levanta ruido**: hacé más estricto el *noise gate* (p.ej. −60 dBFS).
- **Transientes clippean**: bajá el **techo** (−8/−10 dBFS) o subí el **margen de cresta** (6–8 dB).

---

## Problemas comunes
- **No suena**: verificá que Windows tenga **VB-CABLE** como **Salida**; en la GUI: **Entrada = CABLE**, **Salida = parlantes reales**.
- **Eco/feedback**: no uses **VB-CABLE** como **Salida** en la GUI.
- **No aparece “CABLE”**: falta reinicio/driver; usá **Refrescar dispositivos**.

---



