import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFileDialog, QMessageBox, QScrollArea)
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import platform

# 한글 폰트 강제 로드 (OS 자동 인식)
font_path = None
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'
elif platform.system() == 'Darwin':
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
else:
    # Linux (Ubuntu)
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

if os.path.exists(font_path):
    font_entry = fm.FontEntry(fname=font_path, name='NanumGothic_Force')
    fm.fontManager.ttflist.insert(0, font_entry)
    plt.rcParams.update({'font.family': 'NanumGothic_Force'})
    font_prop = fm.FontProperties(fname=font_path)
else:
    print(f"폰트를 찾을 수 없습니다: {font_path}")
    font_prop = fm.FontProperties()

plt.rcParams['axes.unicode_minus'] = False

class SpiderChartCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, polar=True)
        super(SpiderChartCanvas, self).__init__(self.fig)
        self.angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
        self.angles_closed = self.angles + [self.angles[0]]  

    def plot_data(self, data, interpolate_type="Linear (직선)", unit="kgf"):
        self.ax.clear()
        self.ax.set_theta_offset(np.pi / 2) 
        self.ax.set_theta_direction(-1)     
        self.ax.set_xticks(self.angles)
        self.ax.set_xticklabels(['Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'], fontproperties=font_prop)
        
        max_val = max(data) if data else 10
        if max_val < 1: max_val = 10 
        self.ax.set_ylim(0, max_val * 1.2)
        self.ax.set_ylabel(f"Load ({unit})", labelpad=20, fontproperties=font_prop)

        plot_data = data + [data[0]]

        if interpolate_type == "Smooth (Spline 곡선)":
            try:
                extended_angles = np.concatenate([
                    np.array(self.angles) - 2*np.pi, 
                    self.angles, 
                    np.array(self.angles) + 2*np.pi
                ])
                extended_data = data * 3
                f = interp1d(extended_angles, extended_data, kind='cubic')
                t_smooth = np.linspace(0, 2 * np.pi, 100)
                smooth_data = f(t_smooth)
                smooth_data = np.clip(smooth_data, 0, None) 
                
                self.ax.plot(t_smooth, smooth_data, color='blue', linewidth=2)
                self.ax.fill(t_smooth, smooth_data, color='blue', alpha=0.1)
            except Exception as e:
                self.ax.plot(self.angles_closed, plot_data, color='blue', linewidth=2)
                self.ax.fill(self.angles_closed, plot_data, color='blue', alpha=0.1)
        else:
            self.ax.plot(self.angles_closed, plot_data, color='blue', linewidth=2)
            self.ax.fill(self.angles_closed, plot_data, color='blue', alpha=0.1)

        self.ax.scatter(self.angles, data, color='red', s=40, zorder=5)
        self.draw()

class ClampSimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6-Axis Clamp Test Machine Simulator")
        self.setGeometry(100, 100, 1300, 900)
        self.unit = "kgf"
        
        self.is_simulating = False
        self.sim_state = "IDLE" 
        self.current_stroke = 0
        self.target_strokes = 3
        self.target_load = 10.0
        self.hold_time = 5.0
        self.hold_ticks = 0
        self.sim_current_load = 0.0
        self.timer_interval = 100 
        
        self.sensor_zeros = [0.0] * 6
        self.raw_data = [0.0] * 6
        self.stroke_data_history = []
        
        # 파일 저장 시 고정될 테스트 시작 타임스탬프
        self.test_start_ts = None
        self.test_start_display_time = None
        
        self.initUI()
        
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        left_panel = QVBoxLayout(scroll_widget)
        
        # 성적서 입력 탭
        group_report = QGroupBox("Report Information")
        layout_report = QGridLayout()
        layout_report.addWidget(QLabel("관리번호 (Report No):"), 0, 0)
        self.in_report_no = QLineEdit(f"Q-26-{datetime.now().strftime('%m%d')}-001")
        layout_report.addWidget(self.in_report_no, 0, 1)
        layout_report.addWidget(QLabel("고객사 (Customer):"), 1, 0)
        self.in_customer = QLineEdit("TESLA")
        layout_report.addWidget(self.in_customer, 1, 1)
        layout_report.addWidget(QLabel("차종 (Model):"), 2, 0)
        self.in_model = QLineEdit("PMY")
        layout_report.addWidget(self.in_model, 2, 1)
        layout_report.addWidget(QLabel("품명 (Part Name):"), 3, 0)
        self.in_part_name = QLineEdit("CABJ O-Ring")
        layout_report.addWidget(self.in_part_name, 3, 1)
        layout_report.addWidget(QLabel("품번 (Part No):"), 4, 0)
        self.in_part_no = QLineEdit("GCR0127")
        layout_report.addWidget(self.in_part_no, 4, 1)
        group_report.setLayout(layout_report)
        left_panel.addWidget(group_report)

        # 지그 사이즈 설정
        group_jig = QGroupBox("Jig Size & Camera Focus")
        layout_jig = QVBoxLayout()
        self.jig_combo = QComboBox()
        self.jig_combo.addItems(["10.6 mm", "22.6 mm", "32.6 mm"])
        self.jig_combo.currentTextChanged.connect(self.update_camera_focus)
        self.lbl_camera = QLabel("Camera Focus Action: Adjusted to 10.6 mm")
        layout_jig.addWidget(QLabel("Select Jig Size:"))
        layout_jig.addWidget(self.jig_combo)
        layout_jig.addWidget(self.lbl_camera)
        group_jig.setLayout(layout_jig)
        left_panel.addWidget(group_jig)
        
        # 테스트 파라미터
        group_params = QGroupBox("Test Parameters")
        layout_params = QGridLayout()
        layout_params.addWidget(QLabel("Min Length (mm):"), 0, 0)
        self.in_min_len = QLineEdit("0.0")
        layout_params.addWidget(self.in_min_len, 0, 1)
        layout_params.addWidget(QLabel("Max Length (mm):"), 1, 0)
        self.in_max_len = QLineEdit("50.0")
        layout_params.addWidget(self.in_max_len, 1, 1)
        layout_params.addWidget(QLabel("Speed (mm/min):"), 2, 0)
        self.in_speed = QLineEdit("10")
        layout_params.addWidget(self.in_speed, 2, 1)
        layout_params.addWidget(QLabel("Hold Time (sec):"), 3, 0)
        self.in_hold = QLineEdit("5.0")
        layout_params.addWidget(self.in_hold, 3, 1)
        layout_params.addWidget(QLabel("Target Load:"), 4, 0)
        self.in_load = QLineEdit("10.0")
        layout_params.addWidget(self.in_load, 4, 1)
        layout_params.addWidget(QLabel("Operation Strokes:"), 5, 0)
        self.in_strokes = QLineEdit("3")
        layout_params.addWidget(self.in_strokes, 5, 1)
        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")
        layout_params.addWidget(self.lbl_status, 6, 0, 1, 2)
        group_params.setLayout(layout_params)
        left_panel.addWidget(group_params)
        
        # 세팅
        group_settings = QGroupBox("Settings")
        layout_settings = QVBoxLayout()
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["kgf", "N"])
        self.unit_combo.currentTextChanged.connect(self.change_unit)
        layout_settings.addWidget(QLabel("Unit Selection:"))
        layout_settings.addWidget(self.unit_combo)
        
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Linear (직선)", "Smooth (Spline 곡선)"])
        self.interp_combo.currentTextChanged.connect(self.update_chart)
        layout_settings.addWidget(QLabel("Graph Interpolation:"))
        layout_settings.addWidget(self.interp_combo)
        
        self.btn_zero = QPushButton("Zero Sensors & Reset Data (영점 맞춤 및 초기화)")
        self.btn_zero.clicked.connect(self.zero_sensors)
        layout_settings.addWidget(self.btn_zero)
        group_settings.setLayout(layout_settings)
        left_panel.addWidget(group_settings)
        
        self.btn_start = QPushButton("Start Test Simulation")
        self.btn_start.clicked.connect(self.toggle_simulation)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        left_panel.addWidget(self.btn_start)
        left_panel.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area, 1)
        
        # 우측 패널
        right_panel = QVBoxLayout()
        self.chart = SpiderChartCanvas(self, width=6, height=5)
        right_panel.addWidget(self.chart, 3)
        
        self.table = QTableWidget(6, 4)
        self.table.setHorizontalHeaderLabels(["Axis", "Raw Data", "Zero Offset", "Calibrated Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(6):
            self.table.setItem(i, 0, QTableWidgetItem(f"Axis {i+1}"))
            for j in range(1, 4):
                self.table.setItem(i, j, QTableWidgetItem("0.0"))
        right_panel.addWidget(self.table, 1)
        
        btn_layout = QHBoxLayout()
        self.btn_csv = QPushButton("Export CSV (데이터 저장)")
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_pdf = QPushButton("Print Report (A4 성적서 출력)")
        self.btn_pdf.clicked.connect(self.export_pdf)
        btn_layout.addWidget(self.btn_csv)
        btn_layout.addWidget(self.btn_pdf)
        right_panel.addLayout(btn_layout)
        
        main_layout.addLayout(right_panel, 3)
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        
        self.update_chart()

    def update_camera_focus(self, size):
        self.lbl_camera.setText(f"Camera Focus Action: Adjusted to {size}")

    def change_unit(self, unit):
        self.unit = unit
        self.update_table()
        self.update_chart()

    def zero_sensors(self):
        # 모든 물리적 데이터와 영점 기준을 0으로 강제 초기화
        self.raw_data = [0.0] * 6
        self.sensor_zeros = [0.0] * 6
        self.sim_current_load = 0.0
        self.stroke_data_history = []
        self.test_start_ts = None
        self.test_start_display_time = None
        
        # 표(Table) 내부 텍스트를 강제로 "0.00"으로 덮어쓰기
        for i in range(6):
            # Raw Data
            self.table.setItem(i, 1, QTableWidgetItem("0.00"))
            # Zero Offset
            self.table.setItem(i, 2, QTableWidgetItem("0.00"))
            # Calibrated Value 
            self.table.setItem(i, 3, QTableWidgetItem("0.00"))

        self.lbl_status.setText("Status: Ready (Data Reset)")
        
        self.update_chart() # 차트도 0점으로 다시 그림
        
        QMessageBox.information(self, "Zeroed", "센서 영점 맞춤 및 데이터 초기화가 완료되었습니다.")

    def update_table(self):
        for i in range(6):
            self.table.setItem(i, 1, QTableWidgetItem(f"{self.raw_data[i]:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{self.sensor_zeros[i]:.2f}"))
            calibrated = self.raw_data[i] - self.sensor_zeros[i]
            display_val = calibrated * 9.80665 if self.unit == "N" else calibrated
            self.table.setItem(i, 3, QTableWidgetItem(f"{display_val:.2f}"))

    def get_calibrated_data(self):
        data = []
        for i in range(6):
            calibrated = self.raw_data[i] - self.sensor_zeros[i]
            display_val = calibrated * 9.80665 if self.unit == "N" else calibrated
            data.append(max(0, display_val))
        return data

    def update_chart(self):
        data = self.get_calibrated_data()
        interp = self.interp_combo.currentText()
        self.chart.plot_data(data, interpolate_type=interp, unit=self.unit)

    def stop_simulation(self, completed=False):
        self.is_simulating = False
        self.sim_state = "IDLE"
        self.btn_start.setText("Start Test Simulation")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.timer.stop()
        if completed:
            if len(self.stroke_data_history) > 0:
                self.raw_data = [val + self.sensor_zeros[i] for i, val in enumerate(self.stroke_data_history[-1])]
            self.lbl_status.setText(f"Status: Test Completed ({len(self.stroke_data_history)} Strokes)")
        else:
            self.lbl_status.setText(f"Status: Stopped ({self.current_stroke} / {self.target_strokes} Strokes)")
        self.update_table()
        self.update_chart()

    def toggle_simulation(self):
        if not self.is_simulating:
            try:
                self.target_strokes = int(self.in_strokes.text())
                self.target_load = float(self.in_load.text())
                self.hold_time = float(self.in_hold.text())
            except ValueError:
                QMessageBox.warning(self, "Input Error", "숫자를 정확히 입력해주세요.")
                return

            self.is_simulating = True
            self.current_stroke = 0
            self.stroke_data_history = []
            
            # 테스트 시작 시점에 파일 이름용 타임스탬프 저장
            now = datetime.now()
            self.test_start_ts = now.strftime("%Y%m%d_%H%M%S")
            self.test_start_display_time = now.strftime('%Y-%m-%d %H:%M:%S')
            
            self.sim_state = "PULLING" 
            self.sim_current_load = 0.0
            
            self.btn_start.setText("Stop Simulation")
            self.btn_start.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
            self.lbl_status.setText(f"Status: PULLING ({self.current_stroke} / {self.target_strokes} Strokes)")
            self.timer.start(self.timer_interval)
        else:
            self.stop_simulation(completed=False)

    def simulation_step(self):
        load_step = self.target_load / (1500 / self.timer_interval) 
        if self.sim_state == "PULLING":
            self.sim_current_load += load_step
            if self.sim_current_load >= self.target_load:
                self.sim_current_load = self.target_load
                self.sim_state = "HOLDING"
                self.hold_ticks = int((self.hold_time * 1000) / self.timer_interval)
        elif self.sim_state == "HOLDING":
            if self.hold_ticks == int((self.hold_time * 1000) / self.timer_interval):
                calibrated = [(self.raw_data[i] - self.sensor_zeros[i]) for i in range(6)]
                self.stroke_data_history.append(calibrated)
            self.hold_ticks -= 1
            if self.hold_ticks <= 0:
                self.sim_state = "RELEASING"
        elif self.sim_state == "RELEASING":
            self.sim_current_load -= load_step
            if self.sim_current_load <= 0:
                self.sim_current_load = 0.0
                self.current_stroke += 1
                if self.current_stroke >= self.target_strokes:
                    self.stop_simulation(completed=True)
                    QMessageBox.information(self, "Test Complete", f"{self.target_strokes} Strokes 테스트가 완료되었습니다.")
                    return
                else:
                    self.sim_state = "PULLING"

        self.lbl_status.setText(f"Status: {self.sim_state} ({self.current_stroke} / {self.target_strokes} Strokes)")
        noise = np.random.normal(0, max(0.05, self.target_load * 0.01), 6)
        dist_factors = [1.0, 0.96, 1.04, 0.98, 1.02, 1.0] 
        for i in range(6):
            if self.sim_current_load <= 0:
                val = 0
            else:
                val = self.sim_current_load * dist_factors[i] + noise[i]
            self.raw_data[i] = max(0, val) + self.sensor_zeros[i]
        self.update_table()
        self.update_chart()

    def export_csv(self):
        if len(self.stroke_data_history) == 0:
            QMessageBox.warning(self, "No Data", "저장할 테스트 데이터가 없습니다.")
            return
            
        ts = self.test_start_ts if self.test_start_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ClampData_{ts}.csv"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV Files (*.csv)")
        
        if file_name:
            stroke_records = []
            for idx, stroke_data in enumerate(self.stroke_data_history):
                for axis_idx, value in enumerate(stroke_data):
                    display_val = value * 9.80665 if self.unit == "N" else value
                    stroke_records.append({
                        'Stroke': idx + 1,
                        'Axis': f'Axis {axis_idx + 1}',
                        f'Load ({self.unit})': round(display_val, 2)
                    })
            pd.DataFrame(stroke_records).to_csv(file_name, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Saved", f"CSV 파일이 저장되었습니다.\n{file_name}")

    def export_pdf(self):
        if len(self.stroke_data_history) == 0:
            QMessageBox.warning(self, "No Data", "저장할 테스트 데이터가 없습니다.")
            return
            
        ts = self.test_start_ts if self.test_start_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ClampReport_{ts}.pdf"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Report", default_name, "PDF Files (*.pdf)")
        if not file_name:
            return
            
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor('white')
        
        # 메인 타이틀
        fig.text(0.35, 0.94, 'Test Report', ha='center', fontsize=20, fontproperties=font_prop, weight='bold')
        
        # ==================== 수정된 결재란 테이블 ====================
        ax_sign = fig.add_axes([0.62, 0.89, 0.30, 0.07])
        ax_sign.axis('off')
        
        sign_data = [
            ['', 'EDIT', 'CHECK', 'APPROVE'],
            ['SIGN', '', '', '']
        ]
        
        table_sign = ax_sign.table(cellText=sign_data, cellLoc='center', loc='center',
                                   colWidths=[0.12, 0.29, 0.29, 0.29])
        table_sign.auto_set_font_size(False)
        table_sign.set_fontsize(8)
        
        # scale() 대신 각 행의 높이를 개별적으로 디테일하게 제어
        for key, cell in table_sign.get_celld().items():
            row, col = key
            cell.set_edgecolor('black')
            cell.set_linewidth(0.8)
            cell.set_text_props(fontproperties=font_prop)
            cell.set_facecolor('white')
            
            # 행(row)에 따른 높이 설정: 헤더는 낮게(0.3), 사인칸은 높게(1.5)
            if row == 0:
                cell.set_height(0.3)
            else:
                cell.set_height(1.5)
            
            # 우측 3칸 헤더 배경색
            if row == 0 and col > 0:
                cell.set_facecolor('#F0F0F0')
            
            # SIGN 열 병합 트릭 및 세로 텍스트 적용
            if col == 0:
                cell.set_facecolor('#F0F0F0')
                if row == 0:    
                    cell.visible_edges = 'LRT' # 위쪽 테두리만
                if row == 1:
                    cell.visible_edges = 'LRB' # 아래쪽 테두리만
                    text_obj = cell.get_text()
                    text_obj.set_rotation('vertical') # 세로 쓰기 적용
                    text_obj.set_position((1, 1.0)) # 위치를 아래쪽(빈칸) 중간으로 내림
                    
        # =============================================================

        # 헤더 테이블
        ax_header = fig.add_axes([0.08, 0.82, 0.84, 0.08])
        ax_header.axis('off')
        date_str = self.test_start_display_time if self.test_start_display_time else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        target_spec = f"기준하중: {self.in_load.text()}{self.unit}"
        
        header_data = [
            ['관리번호 (Report No)', self.in_report_no.text(), '시험항목 (Test Item)', '힘 (Load)'],
            ['고객사 (Customer)', self.in_customer.text(), '판정기준 (Test Spec)', '합격'],
            ['차종 (Model)', self.in_model.text(), '시험목적 (Test purpose)', target_spec],
            ['품명 (Part Name)', self.in_part_name.text(), '의뢰자 (Client)', ''],
            ['품번 (Part No)', self.in_part_no.text(), 'Test Start', date_str]
        ]
        
        table_header = ax_header.table(cellText=header_data, cellLoc='center', loc='center',
                                       colWidths=[0.2, 0.3, 0.2, 0.3])
        table_header.auto_set_font_size(False)
        table_header.set_fontsize(8)
        table_header.scale(1, 1.6)
        for key, cell in table_header.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(0.8)
            cell.set_text_props(fontproperties=font_prop)
            if key[1] % 2 == 0: cell.set_facecolor('#F0F0F0')

        # 데이터 테이블
        ax_table = fig.add_axes([0.08, 0.44, 0.84, 0.35])
        ax_table.axis('off')
        
        all_data = np.array(self.stroke_data_history)
        if self.unit == "N":
            all_data = all_data * 9.80665
            
        table_data = [['No', 'Stroke\n[mm]', 'Axis 1', 'Axis 2', 'Axis 3', 
                       'Axis 4', 'Axis 5', 'Axis 6', f'Average']]
        
        for idx, stroke_data in enumerate(all_data):
            stroke_mm = self.in_max_len.text()
            avg = np.mean(stroke_data)
            row = [str(idx+1), stroke_mm] + [f'{v:.2f}' for v in stroke_data] + [f'{avg:.2f}']
            table_data.append(row)
        
        for i in range(10 - len(all_data)):
            table_data.append(['-', '-', '-', '-', '-', '-', '-', '-', '-'])
            
        min_vals = np.min(all_data, axis=0)
        max_vals = np.max(all_data, axis=0)
        range_vals = max_vals - min_vals
        avg_vals = np.mean(all_data, axis=0)
        
        table_data.append(['Min', self.in_max_len.text()] + [f'{v:.2f}' for v in min_vals] + [f'{np.min(all_data):.2f}'])
        table_data.append(['Max', self.in_max_len.text()] + [f'{v:.2f}' for v in max_vals] + [f'{np.max(all_data):.2f}'])
        table_data.append(['R', '0.00'] + [f'{v:.2f}' for v in range_vals] + [f'{np.max(range_vals):.2f}'])
        table_data.append(['Ave', self.in_max_len.text()] + [f'{v:.2f}' for v in avg_vals] + [f'{np.mean(all_data):.2f}'])
        
        data_table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                                    colWidths=[0.06, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        data_table.auto_set_font_size(False)
        data_table.set_fontsize(7.5)
        data_table.scale(1, 1.45)
        
        for key, cell in data_table.get_celld().items():
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
            cell.set_text_props(fontproperties=font_prop)
            row, col = key
            if row == 0:
                cell.set_facecolor('#E0E0E0')
                cell.set_text_props(weight='bold', fontproperties=font_prop)
            elif row > len(all_data) and row <= 10:
                cell.set_text_props(color='#999999')
            elif row > 10:
                cell.set_facecolor('#FFF5E6')
                if col == 0:
                    cell.set_text_props(weight='bold', fontproperties=font_prop)
                    
        # Remark 섹션
        ax_remark = fig.add_axes([0.08, 0.40, 0.84, 0.03])
        ax_remark.axis('off')
        ax_remark.text(0.01, 0.5, ' ', fontsize=8, fontproperties=font_prop, weight='bold', va='center')
        rect = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none', transform=ax_remark.transAxes)
        ax_remark.add_patch(rect)
        
        # 방사형 그래프
        ax_graph = fig.add_axes([0.25, 0.06, 0.5, 0.32], polar=True)
        data = self.get_calibrated_data()
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        plot_data = data + [data[0]]
        angles_plot = np.append(angles, angles[0])
        
        ax_graph.set_theta_offset(np.pi / 2)
        ax_graph.set_theta_direction(-1)
        ax_graph.set_xticks(angles)
        ax_graph.set_xticklabels(['Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5', 'Axis 6'], fontproperties=font_prop, fontsize=9)
        
        interpolate_type = self.interp_combo.currentText()
        if interpolate_type == "Smooth (Spline 곡선)":
            try:
                extended_angles = np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi])
                extended_data = data * 3
                f = interp1d(extended_angles, extended_data, kind='cubic')
                t_smooth = np.linspace(0, 2 * np.pi, 100)
                smooth_data = np.clip(f(t_smooth), 0, None)
                ax_graph.plot(t_smooth, smooth_data, 'b-', linewidth=2)
                ax_graph.fill(t_smooth, smooth_data, 'b', alpha=0.15)
            except Exception:
                ax_graph.plot(angles_plot, plot_data, 'b-', linewidth=2)
                ax_graph.fill(angles_plot, plot_data, 'b', alpha=0.15)
        else:
            ax_graph.plot(angles_plot, plot_data, 'b-', linewidth=2)
            ax_graph.fill(angles_plot, plot_data, 'b', alpha=0.15)

        ax_graph.scatter(angles, data, color='red', s=40, zorder=5)
        ax_graph.set_ylabel(f'Load [{self.unit}]', labelpad=25, fontproperties=font_prop, fontsize=9)
        ax_graph.set_title('Final Load Distribution (6-Axis)', pad=15, fontproperties=font_prop, fontsize=11, weight='bold')
        ax_graph.grid(True, linestyle='--', alpha=0.7)
        
        fig.savefig(file_name, format='pdf', dpi=300)
        plt.close(fig)
        
        QMessageBox.information(self, "Saved", f"A4 성적서가 성공적으로 저장되었습니다.\n{file_name}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClampSimulatorApp()
    ex.show()
    sys.exit(app.exec_())
