# CTR_SealRing_Simul



### How to Install

1. **Uninstall FElupe** (to avoid conflicts):

```bash
pip uninstall felupe
```

2. **Install New Dependencies**:
Save the text above as `requirements.txt` and run:

```bash
pip install -r requirements.txt
```


### Why these packages?

* **`scikit-fem`**: Replaces FElupe as the Finite Element Analysis engine. It's lighter and stateless, making it perfect for real-time GUI updates.
* **`scipy`**: Essential for solving sparse linear systems (`splinalg.spsolve`), which `scikit-fem` relies on.
* **`numpy`**: Handles all matrix and vector operations.
* **`PyQt5` \& `matplotlib`**: Same as before, for the GUI and plotting.
