# CTR_SealRing_Simul



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
3. In the app, switch `Input Source` to `cDAQ USB (1ch -> 6ch)`.
4. Set the load-cell channel, rated load, sensitivity, bridge resistance, excitation voltage, and sample rate.
5. Start monitoring. The single load-cell value is mirrored into all 6 axes.

### Why these packages?

* **`scikit-fem`**: Replaces FElupe as the Finite Element Analysis engine. It's lighter and stateless, making it perfect for real-time GUI updates.
* **`scipy`**: Essential for solving sparse linear systems (`splinalg.spsolve`), which `scikit-fem` relies on.
* **`numpy`**: Handles all matrix and vector operations.
* **`PyQt5` \& `matplotlib`**: Same as before, for the GUI and plotting.
