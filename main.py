import sys
import numpy as np
import warnings
from scipy.sparse import linalg as splinalg

# 경고 메시지 무시
warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QDoubleValidator, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

# Scikit-FEM imports
from skfem import *
from skfem.models.elasticity import lame_parameters, linear_elasticity
from skfem.helpers import dot, grad

# ==============================================================================
# 1. Physics Engine: Scikit-FEM (Explicit Matrix Solving)
# ==============================================================================
class SkfemSolver:
    def __init__(self):
        # --- Mesh Generation ---
        self.n_radial, self.n_circum = 6, 60
        r_min, r_max = 10.0, 12.0
        
        # Create topology
        r = np.linspace(r_min, r_max, self.n_radial)
        t = np.linspace(0, 2*np.pi, self.n_circum + 1)[:-1]
        R, T = np.meshgrid(r, t)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        
        # Flatten for skfem (2, N_nodes)
        p = np.vstack([X.ravel(), Y.ravel()])
        
        # Create elements
        t_conn = []
        for i in range(self.n_circum):
            for j in range(self.n_radial - 1):
                n0 = i * self.n_radial + j
                n1 = n0 + 1
                n2 = ((i + 1) % self.n_circum) * self.n_radial + j + 1
                n3 = ((i + 1) % self.n_circum) * self.n_radial + j
                t_conn.append([n0, n1, n2, n3])
        
        t_conn = np.array(t_conn).T
        self.mesh = MeshQuad(p, t_conn)
        self.n_nodes = self.mesh.p.shape[1]
        
        # --- Material ---
        self.lam, self.mu = lame_parameters(E=10.0, nu=0.4)
        
        # --- Basis & Assembly ---
        self.element = ElementVector(ElementQuad1())
        self.basis = Basis(self.mesh, self.element, intorder=2)
        
        # Stiffness Matrix (K) - Assembled ONCE
        self.K = asm(linear_elasticity(self.lam, self.mu), self.basis)
        
        # --- Identify Servo Nodes ---
        self.servo_angles = np.array([0, 60, 120, 180, 240, 300])
        self.servo_node_indices = []
        
        pts = self.mesh.p
        dist_from_center = np.linalg.norm(pts, axis=0)
        inner_mask = np.abs(dist_from_center - r_min) < 0.1
        inner_indices = np.where(inner_mask)[0]
        
        for angle_deg in self.servo_angles:
            angle_rad = np.radians(angle_deg)
            target = np.array([r_min * np.cos(angle_rad), r_min * np.sin(angle_rad)])
            dists = np.linalg.norm(pts[:, inner_indices].T - target, axis=1)
            closest_local_idx = np.argmin(dists)
            self.servo_node_indices.append(inner_indices[closest_local_idx])

    def solve(self, displacements_mm):
        dof_map = self.basis.nodal_dofs 
        
        cons_dofs = []
        cons_vals = []
        
        for i, dist in enumerate(displacements_mm):
            angle_rad = np.radians(self.servo_angles[i])
            ux = dist * np.cos(angle_rad)
            uy = dist * np.sin(angle_rad)
            
            node_idx = self.servo_node_indices[i]
            
            # X-DOF constraint
            cons_dofs.append(dof_map[0, node_idx])
            cons_vals.append(ux)
            
            # Y-DOF constraint
            cons_dofs.append(dof_map[1, node_idx])
            cons_vals.append(uy)
            
        cons_dofs = np.array(cons_dofs)
        cons_vals = np.array(cons_vals)
        
        # Solve using condense manually
        all_dofs = np.arange(self.basis.N)
        free_dofs = np.setdiff1d(all_dofs, cons_dofs)
        
        u = np.zeros(self.basis.N)
        u[cons_dofs] = cons_vals 
        
        K_ff = self.K[free_dofs, :][:, free_dofs]
        K_fc = self.K[free_dofs, :][:, cons_dofs]
        
        rhs = -K_fc @ cons_vals
        
        u_free = splinalg.spsolve(K_ff, rhs)
        u[free_dofs] = u_free
        
        # Calculate Reaction Forces (R = K * u)
        f_reaction_global = self.K @ u
        
        reaction_forces = []
        for i in range(6):
            node_idx = self.servo_node_indices[i]
            dx = dof_map[0, node_idx]
            dy = dof_map[1, node_idx]
            
            rx = f_reaction_global[dx]
            ry = f_reaction_global[dy]
            
            # Project to radial
            angle_rad = np.radians(self.servo_angles[i])
            fr = rx * np.cos(angle_rad) + ry * np.sin(angle_rad)
            reaction_forces.append(fr)
            
        # Deform Mesh
        u_reshaped = u[dof_map] 
        deformed_p = self.mesh.p + u_reshaped
        
        return deformed_p.T, reaction_forces, self.mesh.t.T

# ==============================================================================
# 2. Worker Thread
# ==============================================================================
class SimulationWorker(QObject):
    result_ready = pyqtSignal(object, object, object)

    def __init__(self):
        super().__init__()
        self.solver = SkfemSolver()

    def run_sim(self, inputs):
        pts, forces, cells = self.solver.solve(inputs)
        self.result_ready.emit(pts, forces, cells)

# ==============================================================================
# 3. Main GUI
# ==============================================================================
class MainWindow(QMainWindow):
    request_solve = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("O-Ring Simulation (Inputs + Reset)")
        self.resize(1100, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Controls
        control_panel = QGroupBox("Servo Inputs (0° - 180°)")
        control_panel.setFixedWidth(350)
        layout = QVBoxLayout()
        
        self.inputs = []
        self.labels = []
        grid = QGridLayout()
        
        angles = [0, 60, 120, 180, 240, 300]
        
        # Add Header
        grid.addWidget(QLabel("<b>Servo</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Input Angle (°)</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Reaction (N)</b>"), 0, 2)

        for i in range(6):
            # 1. Servo Label
            grid.addWidget(QLabel(f"Servo {i+1} ({angles[i]}°)"), i+1, 0)
            
            # 2. Input Box (QLineEdit)
            inp = QLineEdit("0.0")
            inp.setValidator(QDoubleValidator(0.0, 180.0, 2)) # 0-180 range, 2 decimals
            inp.setAlignment(Qt.AlignRight)
            inp.editingFinished.connect(self.trigger_solve) # Update on Enter/FocusOut
            self.inputs.append(inp)
            grid.addWidget(inp, i+1, 1)
            
            # 3. Force Label
            l = QLabel("0.00 N")
            l.setStyleSheet("color: red; font-weight: bold")
            l.setAlignment(Qt.AlignRight)
            self.labels.append(l)
            grid.addWidget(l, i+1, 2)
            
        layout.addLayout(grid)
        
        # RESET BUTTON
        btn_reset = QPushButton("RESET ALL")
        btn_reset.setFixedHeight(40)
        btn_reset.setStyleSheet("background-color: #ffcccc; font-weight: bold; border-radius: 5px;")
        btn_reset.clicked.connect(self.reset_all)
        layout.addWidget(btn_reset)
        
        layout.addStretch()
        
        info_label = QLabel("Enter angle (0-180) and press Enter.\n0° = 0mm, 180° = 3mm stretch")
        info_label.setStyleSheet("color: gray")
        layout.addWidget(info_label)
        
        control_panel.setLayout(layout)
        
        # Plot
        self.canvas = FigureCanvas(Figure(figsize=(5,5)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.poly = None
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas)
        
        # Threading
        self.thread = QThread()
        self.worker = SimulationWorker()
        self.worker.moveToThread(self.thread)
        self.request_solve.connect(self.worker.run_sim)
        self.worker.result_ready.connect(self.update_view)
        
        self.thread.start()
        
        # Init
        self.trigger_solve()

    def reset_all(self):
        # Set all inputs to 0.0
        for inp in self.inputs:
            inp.setText("0.0")
        # Trigger solve to update physics
        self.trigger_solve()

    def trigger_solve(self):
        vals = []
        for inp in self.inputs:
            try:
                text = inp.text()
                angle = float(text) if text else 0.0
            except ValueError:
                angle = 0.0
            
            # Mapping: 0-180 degree input -> 0-3.0 mm displacement
            # Clamp to safe range
            angle = max(0.0, min(180.0, angle))
            disp = (angle / 180.0) * 3.0 
            vals.append(disp)
            
        self.request_solve.emit(vals)

    def update_view(self, pts, forces, cells):
        if self.poly is None:
            self.poly = PolyCollection(pts[cells], facecolors='skyblue', edgecolors='k', alpha=0.7, linewidths=0.5)
            self.ax.add_collection(self.poly)
        else:
            self.poly.set_verts(pts[cells])
            
        for i, f in enumerate(forces):
            self.labels[i].setText(f"{f:.2f} N")
            
        self.canvas.draw()

    def closeEvent(self, e):
        self.thread.quit()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
