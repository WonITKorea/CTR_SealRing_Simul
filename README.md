# CTR_SealRing_Simul

Korean README: [README_ko.md](README_ko.md)


### How to Install

1. **Install New Dependencies**:
Save the text above as `requirements.txt` and run:

```bash
pip install -r requirements.txt
```
2. **Run the Main.py**:

```bash
python3 main.py
```

### cDAQ-SL1100 USB Input Mode

The program supports a live USB input mode for an NI `cDAQ-SL1100` bundle.

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install the NI-DAQmx driver on the PC that is connected to the hardware.
3. In the app, switch `Input Source` to `NI cDAQ USB (1ch -> 6ch)`.
4. Set the load-cell channel, rated load, sensitivity, bridge resistance, excitation voltage, and sample rate.
5. Start monitoring. The single load-cell value is mirrored into all 6 axes.

### UNIPULSE FC400 RS-485 Mode

The program also supports a real hardware mode for `UNIPULSE FC400-DAC-FA` through `USB to RS-485`.

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Power the FC400 with `DC 24V` and connect the RS-485 port to the PC through a USB-RS485 converter.
3. Set the FC400 communication mode to `Modbus-RTU`.
4. Match the FC400 baud rate, parity, stop bit, and slave ID with the values in the app.
5. In the app, switch `Input Source` to `UNIPULSE FC400 RS-485`.
6. Select whether the app should read `Gross` or `Net` weight and set the FC400 engineering unit (`N` or `kgf`).
7. Start monitoring. The FC400 value is mirrored into all 6 axes.

Recommended starting point for the current jig:

1. Load cell: `KD80S-1KN`
2. Rated range: `1000 N`
3. FC400 engineering unit: `N`
4. Display/report unit in the app: `N` (the UI now auto-switches to `N` the first time you enter FC400 mode)
5. Use `Gross` unless your FC400 setup already uses tare/net as part of the production sequence.

### MR-MC240N Position Monitor

The app can optionally read feedback position from a `Mitsubishi MR-MC240N` position board.

1. Install the Mitsubishi MR-MC200 series utility/device driver and API library on a Windows PC.
2. Make `mc2xxstd.dll` or `mc2xxstd_x64.dll` available in the same folder as the program or in `PATH`.
3. Enable `MR-MC240N feedback position monitor` in the app.
4. Set `Board ID`, `Axis No`, and `Command Units / mm` to match the actual servo system.
5. If needed, enable the optional `sscSystemStart()` checkbox when the board is not already in the running state.
6. `Command Units / mm` must come from your servo/axis parameter set because it depends on the mechanical pitch, gear ratio, and electronic gearing of the machine.

### Why these packages?

* **`scikit-fem`**: Replaces FElupe as the Finite Element Analysis engine. It's lighter and stateless, making it perfect for real-time GUI updates.
* **`scipy`**: Essential for solving sparse linear systems (`splinalg.spsolve`), which `scikit-fem` relies on.
* **`numpy`**: Handles all matrix and vector operations.
* **`PyQt5` \& `matplotlib`**: Same as before, for the GUI and plotting.
