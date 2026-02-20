import sys
import numpy as np
import warnings
from scipy.sparse import linalg as splinalg

warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGroupBox, QGridLayout, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection

from skfem import *
from skfem.models.elasticity import lame_parameters, linear_elasticity

# ==============================================================================
# 1. Physics Engine
# ==============================================================================
class SkfemSolver:
    def __init__(self):
        self.n_radial, self.n_circum = 6, 60
        r_min, r_max = 10.0, 12.0
        
        r = np.linspace(r_min, r_max, self.n_radial)
        t = np.linspace(0, 2*np.pi, self.n_circum + 1)[:-1]
        R, T = np.meshgrid(r, t)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        
        p = np.vstack([X.ravel(), Y.ravel()])
        
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
        
        self.lam, self.mu = lame_parameters(E=10.0, nu=0.4)
        
        self.element = ElementVector(ElementQuad1())
        self.basis = Basis(self.mesh, self.element, intorder=2)
        self.K = asm(linear_elasticity(self.lam, self.mu), self.basis)
        
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
            self.servo_node_indices.append(inner_indices[np.argmin(dists)])

    def solve(self, displacements_mm):
        dof_map = self.basis.nodal_dofs 
        
        cons_dofs = []
        cons_vals = []
        
        for i, dist in enumerate(displacements_mm):
            angle_rad = np.radians(self.servo_angles[i])
            # Negative dist = push inward, positive = pull outward
            ux = dist * np.cos(angle_rad)
            uy = dist * np.sin(angle_rad)
            
            node_idx = self.servo_node_indices[i]
            cons_dofs.append(dof_map[0, node_idx])
            cons_vals.append(ux)
            cons_dofs.append(dof_map[1, node_idx])
            cons_vals.append(uy)
            
        cons_dofs = np.array(cons_dofs)
        cons_vals = np.array(cons_vals)
        
        all_dofs = np.arange(self.basis.N)
        free_dofs = np.setdiff1d(all_dofs, cons_dofs)
        
        u = np.zeros(self.basis.N)
        u[cons_dofs] = cons_vals 
        
        K_ff = self.K[free_dofs, :][:, free_dofs]
        K_fc = self.K[free_dofs, :][:, cons_dofs]
        rhs = -K_fc @ cons_vals
        
        u[free_dofs] = splinalg.spsolve(K_ff, rhs)
        
        f_reaction_global = self.K @ u
        
        reaction_forces = []
        for i in range(6):
            node_idx = self.servo_node_indices[i]
            dx = dof_map[0, node_idx]
            dy = dof_map[1, node_idx]
            rx = f_reaction_global[dx]
            ry = f_reaction_global[dy]
            angle_rad = np.radians(self.servo_angles[i])
            fr = rx * np.cos(angle_rad) + ry * np.sin(angle_rad)
            reaction_forces.append(fr)
            
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
        self.setWindowTitle("O-Ring Simulation")
        self.resize(1100, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Controls
        control_panel = QGroupBox("Servo Inputs (-180° to +180°)")
        control_panel.setFixedWidth(380)
        layout = QVBoxLayout()
        
        self.inputs = []
        self.labels = []
        grid = QGridLayout()
        
        angles = [0, 60, 120, 180, 240, 300]
        
        # Header
        grid.addWidget(QLabel("<b>Servo</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Input (°)</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Reaction (N)</b>"), 0, 2)

        for i in range(6):
            grid.addWidget(QLabel(f"Servo {i+1} ({angles[i]}°)"), i+1, 0)
            
            inp = QLineEdit("0.0")
            # [KEY CHANGE] Allow negative values: range -180 to +180
            inp.setValidator(QDoubleValidator(-180.0, 180.0, 2))
            inp.setAlignment(Qt.AlignRight)
            inp.setPlaceholderText("-180 ~ +180")
            inp.editingFinished.connect(self.trigger_solve)
            self.inputs.append(inp)
            grid.addWidget(inp, i+1, 1)
            
            l = QLabel("0.00 N")
            l.setStyleSheet("color: red; font-weight: bold")
            l.setAlignment(Qt.AlignRight)
            self.labels.append(l)
            grid.addWidget(l, i+1, 2)
            
        layout.addLayout(grid)
        
        # RESET BUTTON
        btn_reset = QPushButton("RESET ALL")
        btn_reset.setFixedHeight(40)
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #ffcccc;
                font-weight: bold;
                border-radius: 5px;
                border: 1px solid #ff9999;
            }
            QPushButton:hover {
                background-color: #ff9999;
            }
            QPushButton:pressed {
                background-color: #ff6666;
            }
        """)
        btn_reset.clicked.connect(self.reset_all)
        layout.addWidget(btn_reset)
        
        layout.addStretch()
        
        # Info Box
        info_label = QLabel(
            "📌 Input Guide:\n"
            "  Positive (+) → Pull outward\n"
            "  Negative (−) → Push inward\n\n"
            "  +180° = +3mm (max pull)\n"
            "  −180° = −3mm (max push)\n\n"
            "Press Enter to apply."
        )
        info_label.setStyleSheet(
            "color: #555; background-color: #f9f9f9;"
            "border: 1px solid #ddd; border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(info_label)
        
        control_panel.setLayout(layout)
        
        # Plot
        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_title("O-Ring Deformation")
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
        
        self.trigger_solve()

    def reset_all(self):
        for inp in self.inputs:
            inp.setText("0.0")
        self.trigger_solve()

    def trigger_solve(self):
        vals = []
        for inp in self.inputs:
            try:
                angle = float(inp.text()) if inp.text() else 0.0
            except ValueError:
                angle = 0.0
            
            # [KEY CHANGE] Clamp range to -180 ~ +180
            angle = max(-180.0, min(180.0, angle))
            
            # Map -180~+180 degrees -> -3.0~+3.0 mm displacement
            # Negative = inward push, Positive = outward pull
            disp = (angle / 180.0) * 3.0
            vals.append(disp)
            
        self.request_solve.emit(vals)

    def update_view(self, pts, forces, cells):
        if self.poly is None:
            self.poly = PolyCollection(
                pts[cells], facecolors='skyblue', edgecolors='k', alpha=0.7, linewidths=0.5
            )
            self.ax.add_collection(self.poly)
        else:
            self.poly.set_verts(pts[cells])
            
        for i, f in enumerate(forces):
            # Color code: red = tension (pull), blue = compression (push)
            color = "red" if f >= 0 else "blue"
            self.labels[i].setStyleSheet(f"color: {color}; font-weight: bold")
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
