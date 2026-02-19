import sys
import numpy as np
import felupe as fe
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================================================================
# 1. Physics Engine: O-Ring Simulation Class (FElupe)
# ==============================================================================
class ORingSimulation:
    def __init__(self):
        # --- Geometry & Mesh (Annulus mapped from Grid) ---
        # 10mm Inner Radius, 12mm Outer Radius (2mm thickness)
        self.Ri, self.Ro = 10.0, 12.0
        n_radial, n_circum = 6, 60  # Mesh density
        
        # Create grid in (r, theta) space and map to (x, y)
        r = np.linspace(self.Ri, self.Ro, n_radial)
        t = np.linspace(0, 2*np.pi, n_circum + 1)[:-1] # 0 to 360 (periodic)
        R, T = np.meshgrid(r, t)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        
        # Create FElupe Mesh (Quad elements)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        # Simple connectivity generation for structured grid
        cells = []
        for i in range(n_circum):
            for j in range(n_radial - 1):
                p0 = i * n_radial + j
                p1 = p0 + 1
                p2 = ((i + 1) % n_circum) * n_radial + j + 1
                p3 = ((i + 1) % n_circum) * n_radial + j
                cells.append([p0, p1, p2, p3])
        
        self.mesh = fe.Mesh(points, np.array(cells), "quad")
        
        # --- Constitutive Model (Rubber-like Material) ---
        # Neo-Hookean (Incompressible approximation)
        self.region = fe.RegionQuad(self.mesh)
        self.field = fe.FieldContainer([fe.Field(self.region, dim=2)])
        self.solid = fe.SolidBody(self.umat, self.field)
        
        # --- Boundary Condition Setup (6 Points) ---
        # Find node indices closest to 6 servo angles (0, 60, 120, ...)
        self.servo_angles_deg = np.array([0, 60, 120, 180, 240, 300])
        self.servo_nodes = []
        
        # We hook the INNER radius (pulling outward)
        inner_nodes = np.where(np.abs(np.linalg.norm(self.mesh.points, axis=1) - self.Ri) < 0.1)[0]
        
        for angle in self.servo_angles_deg:
            rad = np.radians(angle)
            # Find closest node on inner ring to this angle
            target_pos = np.array([self.Ri * np.cos(rad), self.Ri * np.sin(rad)])
            dists = np.linalg.norm(self.mesh.points[inner_nodes] - target_pos, axis=1)
            closest = inner_nodes[np.argmin(dists)]
            self.servo_nodes.append(closest)

    def umat(self, F, mu=1.0, bulk=50.0):
        """Neo-Hookean Hyperelastic Material Definition"""
        C = fe.math.dot(fe.math.transpose(F), F)
        J = fe.math.det(F)
        I1 = fe.math.trace(C)
        # Strain Energy Function W
        W = (mu / 2) * (I1 - 2 - 2 * fe.math.ln(J)) + (bulk / 2) * (J - 1)**2
        # First Piola-Kirchhoff Stress P = dW/dF
        P = fe.math.d_diff(W, F)
        return P

    def solve(self, displacements_mm):
        """
        Run one step of static equilibrium.
        displacements_mm: list of 6 radial displacements (floats)
        """
        boundaries = {}
        
        # Apply Dirichlet BCs for each servo
        for i, u_r in enumerate(displacements_mm):
            node_idx = self.servo_nodes[i]
            angle_rad = np.radians(self.servo_angles_deg[i])
            
            # Convert radial displacement to (dx, dy)
            # Total position = Original + Displacement
            # BC in FElupe is usually total displacement value
            ux = u_r * np.cos(angle_rad)
            uy = u_r * np.sin(angle_rad)
            
            # Apply to both X and Y DOFs of that node
            boundaries[f"servo_{i}_x"] = fe.Boundary(self.field[0], fx=ux, skip=(0,1), mask=node_idx)
            boundaries[f"servo_{i}_y"] = fe.Boundary(self.field[0], fy=uy, skip=(1,0), mask=node_idx)
            
        # Add basic rigid body constraint if needed (e.g. fix center if it were a disk)
        # For a ring pulled 6 ways, it's self-equilibrated, but numerical drift can occur.
        # Here we rely on the 6 points to hold it in place.

        # Solve (Newton-Raphson)
        try:
            step = fe.Step(items=[self.solid], boundaries=boundaries)
            job = fe.Job(steps=[step])
            job.evaluate(verbose=False) # Run the solver
        except Exception as e:
            print(f"Solver diverged: {e}")
            return None, None

        # --- Post-Processing: Reaction Forces ---
        # F_reaction = F_internal (at constrained nodes)
        # FElupe stores internal force vector in step.fun (residual) 
        # But easier is to re-evaluate internal forces with current deformation
        forces = self.solid.results.force # shape (N_nodes, 2)
        
        reaction_forces = []
        for i, node_idx in enumerate(self.servo_nodes):
            fx = forces[node_idx][0]
            fy = forces[node_idx][1]
            # Project force onto radial direction (Pulling force)
            angle_rad = np.radians(self.servo_angles_deg[i])
            f_radial = fx * np.cos(angle_rad) + fy * np.sin(angle_rad)
            reaction_forces.append(f_radial)
            
        # Get deformed points for plotting
        deformed_points = self.mesh.points + self.field[0].values
        
        return deformed_points, reaction_forces

# ==============================================================================
# 2. GUI: Servo Control & Visualization (PyQt5)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("O-Ring Multi-Axis Tension Simulation")
        self.resize(1000, 700)
        
        # Initialize Physics
        self.sim = ORingSimulation()
        self.servo_vals = [0.0] * 6  # Current servo displacements (mm)

        # Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Left Panel: Controls ---
        control_panel = QGroupBox("Servo Controls (0° - 180°)")
        control_layout = QVBoxLayout()
        control_panel.setFixedWidth(300)
        
        self.sliders = []
        self.force_labels = []
        
        grid = QGridLayout()
        for i in range(6):
            # Label
            lbl = QLabel(f"Servo #{i+1} (Pos: {self.sim.servo_angles_deg[i]}°)")
            grid.addWidget(lbl, i, 0)
            
            # Slider (0-180 degrees -> maps to 0-5mm)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 180)
            slider.setValue(0)
            slider.valueChanged.connect(self.on_slider_change)
            self.sliders.append(slider)
            grid.addWidget(slider, i, 1)
            
            # Force Display
            f_lbl = QLabel("0.00 N")
            f_lbl.setStyleSheet("color: red; font-weight: bold;")
            self.force_labels.append(f_lbl)
            grid.addWidget(f_lbl, i, 2)
            
        control_layout.addLayout(grid)
        
        # Global controls
        btn_reset = QLabel("Drag sliders to pull O-ring.\nCalculations are real-time.")
        control_layout.addWidget(btn_reset)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # --- Right Panel: Visualization ---
        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas)
        
        # Initial Plot
        self.plot_mesh(self.sim.mesh.points, [0]*6)

    def on_slider_change(self):
        # 1. Read Sliders -> Convert Angle to Displacement
        # Mapping: Servo 0-180 deg -> 0-5 mm Extension
        displacements = []
        for s in self.sliders:
            angle = s.value()
            # Simple Linear Mapping: 180 deg = 5mm stretch
            d = (angle / 180.0) * 5.0 
            displacements.append(d)
        
        # 2. Solve Physics
        deformed_pts, forces = self.sim.solve(displacements)
        
        if deformed_pts is not None:
            # 3. Update GUI
            self.plot_mesh(deformed_pts, forces)
            
            for i, f in enumerate(forces):
                self.force_labels[i].setText(f"{f:.2f} N")

    def plot_mesh(self, points, forces):
        self.ax.clear()
        
        # Plot O-Ring Elements (as simple polygons or scatter for speed)
        # Using a simple fill for visualization
        # Separate Inner/Outer loop for clean drawing? 
        # For simplicity in 'quad' mesh, we just plot all cells as polygons
        
        # Create a QuadMesh collection for matplotlib (fastest)
        from matplotlib.collections import PolyCollection
        
        # Extract cell coordinates
        cells = self.sim.mesh.cells
        coords = points[cells] # shape (n_cells, 4, 2)
        
        pc = PolyCollection(coords, edgecolors='k', facecolors='skyblue', alpha=0.7, linewidths=0.5)
        self.ax.add_collection(pc)
        
        # Draw Arrows for Forces at Servo Nodes
        for i, node_idx in enumerate(self.sim.servo_nodes):
            p = points[node_idx]
            f = forces[i] if forces else 0
            # Scale arrow by force
            scale = 0.5
            angle_rad = np.radians(self.sim.servo_angles_deg[i])
            dx = (f * scale) * np.cos(angle_rad)
            dy = (f * scale) * np.sin(angle_rad)
            
            # Anchor point
            self.ax.plot(p[0], p[1], 'ro', markersize=5) # Servo point
            if abs(f) > 0.1:
                self.ax.arrow(p[0], p[1], dx, dy, head_width=0.5, head_length=0.5, fc='r', ec='r')
        
        # Set limits
        limit = 18
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.grid(True, linestyle=':')
        self.ax.set_title("O-Ring Deformation & Reaction Forces")
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
