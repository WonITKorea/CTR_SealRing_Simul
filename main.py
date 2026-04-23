import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
                             QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import platform

from hardware import (
    FC400_GROSS_MODE,
    FC400_NET_MODE,
    FC400ModbusClient,
    MR_MC240N_WINDOWS_ONLY_MESSAGE,
    MrMc240nPositionMonitor,
    SERIAL_AVAILABLE,
    SERIAL_IMPORT_ERROR,
    list_serial_port_names,
)

try:
    import nidaqmx
    from nidaqmx.system import System
    from nidaqmx.constants import (
        AcquisitionType,
        BridgeConfiguration,
        BridgeElectricalUnits,
        BridgePhysicalUnits,
        ExcitationSource,
        ForceUnits,
        READ_ALL_AVAILABLE,
    )
    NIDAQMX_AVAILABLE = True
    NIDAQMX_IMPORT_ERROR = ""
except Exception as exc:
    nidaqmx = None
    System = None
    AcquisitionType = None
    BridgeConfiguration = None
    BridgeElectricalUnits = None
    BridgePhysicalUnits = None
    ExcitationSource = None
    ForceUnits = None
    READ_ALL_AVAILABLE = None
    NIDAQMX_AVAILABLE = False
    NIDAQMX_IMPORT_ERROR = str(exc)

try:
    import cv2
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)


def _sanitize_qt_plugin_env_after_cv2_import():
    if not CV2_AVAILABLE:
        return

    cv2_file = getattr(cv2, "__file__", "")
    if not cv2_file:
        return

    cv2_package_dir = os.path.dirname(cv2_file)
    cv2_qt_dir = os.path.normpath(os.path.join(cv2_package_dir, "qt"))

    for env_key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        env_value = os.environ.get(env_key)
        if not env_value:
            continue

        normalized_value = os.path.normpath(env_value)
        if normalized_value.startswith(cv2_qt_dir):
            os.environ.pop(env_key, None)


_sanitize_qt_plugin_env_after_cv2_import()

SIMULATION_SOURCE = "Simulation"
CDAQ_SOURCE = "NI cDAQ USB (1ch -> 6ch)"
FC400_SOURCE = "UNIPULSE FC400 RS-485"
DEFAULT_NI_9237_SAMPLE_RATE = 12_800_000.0 / 256.0 / 31.0

# 한글 폰트 강제 로드 (OS 자동 인식)
font_path = None
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'
elif platform.system() == 'Darwin':
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
else:
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
        self.configure_window_geometry()
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
        self.input_source = SIMULATION_SOURCE
        self.data_unit = "kgf"
        
        self.sensor_zeros = [0.0] * 6
        self.raw_data = [0.0] * 6
        self.latest_live_snapshot = [0.0] * 6
        self.cdaq_task = None
        self.fc400_client = None
        self.position_monitor = None
        self.latest_live_position_mm = None
        self.latest_live_position_counts = None
        self.position_zero_offset_mm = 0.0
        self.camera_capture = None
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.camera_timer_step)
        self.camera_timer_interval_ms = 50
        self.latest_ring_measurement = None
        self.camera_baseline = None
        self.camera_read_failures = 0
        
        # 데이터 저장소
        self.stroke_data_history = []  # 각 스트로크 최종 결과 저장
        self.stroke_position_history = []  # 각 스트로크의 대표 위치(mm) 저장
        self.time_series_data = []     # 실시간 시계열 로깅 데이터 저장
        self.time_elapsed = 0.0        # 시계열용 누적 시간
        
        self.test_start_ts = None
        self.test_start_display_time = None
        
        self.initUI()

    def configure_window_geometry(self):
        base_width = 1820
        base_height = 1180
        fallback_min_width = 1200
        fallback_min_height = 760

        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(base_width, base_height)
            self.setMinimumSize(fallback_min_width, fallback_min_height)
            return

        available = screen.availableGeometry()
        start_width = min(base_width, available.width())
        start_height = min(base_height, available.height())
        min_width = min(fallback_min_width, available.width())
        min_height = min(fallback_min_height, available.height())

        self.setGeometry(available.x(), available.y(), start_width, start_height)
        self.setMinimumSize(min_width, min_height)
        
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        left_container = QWidget()
        left_container.setMinimumWidth(920)
        left_layout = QGridLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setHorizontalSpacing(12)
        left_layout.setVerticalSpacing(10)
        left_layout.setColumnStretch(0, 1)
        left_layout.setColumnStretch(1, 1)
        
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
        left_layout.addWidget(group_report, 0, 0)

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
        left_layout.addWidget(group_jig, 0, 1)
        
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
        left_layout.addWidget(group_params, 1, 0)
        
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
        left_layout.addWidget(group_settings, 1, 1)

        group_daq = QGroupBox("Load Input Source / NI cDAQ")
        layout_daq = QGridLayout()
        layout_daq.setColumnStretch(1, 1)

        layout_daq.addWidget(QLabel("Input Source:"), 0, 0, 1, 2)
        source_button_layout = QHBoxLayout()
        source_button_layout.setSpacing(8)
        self.source_buttons = {}
        source_button_specs = [
            ("Simulation", SIMULATION_SOURCE),
            ("NI cDAQ", CDAQ_SOURCE),
            ("FC400 RS-485", FC400_SOURCE),
        ]
        for button_label, source_name in source_button_specs:
            button = QPushButton(button_label)
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setMinimumHeight(34)
            button.setStyleSheet(
                "QPushButton {padding: 8px 10px; background-color: #F5F7FA; border: 1px solid #B8C1CC;}"
                "QPushButton:checked {background-color: #1976D2; color: white; border: 1px solid #0D47A1;}"
            )
            button.clicked.connect(
                lambda _checked, selected_source=source_name: self.on_input_source_button_clicked(selected_source)
            )
            self.source_buttons[source_name] = button
            source_button_layout.addWidget(button)
        layout_daq.addLayout(source_button_layout, 1, 0, 1, 2)

        layout_daq.addWidget(QLabel("Physical Channel:"), 2, 0)
        self.in_daq_channel = QLineEdit("cDAQ1Mod1/ai0")
        layout_daq.addWidget(self.in_daq_channel, 2, 1)

        layout_daq.addWidget(QLabel("Rated Load [kgf]:"), 3, 0)
        self.in_daq_capacity = QLineEdit("100.0")
        layout_daq.addWidget(self.in_daq_capacity, 3, 1)

        layout_daq.addWidget(QLabel("Sensitivity [mV/V]:"), 4, 0)
        self.in_daq_sensitivity = QLineEdit("2.0")
        layout_daq.addWidget(self.in_daq_sensitivity, 4, 1)

        layout_daq.addWidget(QLabel("Bridge Resistance [Ohm]:"), 5, 0)
        self.in_daq_bridge_res = QLineEdit("350")
        layout_daq.addWidget(self.in_daq_bridge_res, 5, 1)

        layout_daq.addWidget(QLabel("Excitation [V]:"), 6, 0)
        self.in_daq_excitation = QLineEdit("5.0")
        layout_daq.addWidget(self.in_daq_excitation, 6, 1)

        layout_daq.addWidget(QLabel("Sample Rate [S/s]:"), 7, 0)
        self.in_daq_sample_rate = QLineEdit(f"{DEFAULT_NI_9237_SAMPLE_RATE:.3f}")
        layout_daq.addWidget(self.in_daq_sample_rate, 7, 1)

        self.btn_refresh_daq = QPushButton("Refresh NI Devices")
        self.btn_refresh_daq.clicked.connect(self.refresh_cdaq_devices)
        layout_daq.addWidget(self.btn_refresh_daq, 8, 0, 1, 2)

        init_daq_status = "cDAQ: select USB mode to scan devices"
        if not NIDAQMX_AVAILABLE:
            init_daq_status = f"cDAQ: nidaqmx import failed - {NIDAQMX_IMPORT_ERROR}"
        self.lbl_daq_status = QLabel(init_daq_status)
        self.lbl_daq_status.setWordWrap(True)
        layout_daq.addWidget(self.lbl_daq_status, 9, 0, 1, 2)

        group_daq.setLayout(layout_daq)
        left_layout.addWidget(group_daq, 2, 0)

        group_fc400 = QGroupBox("UNIPULSE FC400 RS-485")
        layout_fc400 = QGridLayout()

        layout_fc400.addWidget(QLabel("Serial Port:"), 0, 0)
        self.in_fc400_port = QLineEdit("")
        layout_fc400.addWidget(self.in_fc400_port, 0, 1)

        self.btn_refresh_serial = QPushButton("Refresh Serial Ports")
        self.btn_refresh_serial.clicked.connect(self.refresh_serial_ports)
        layout_fc400.addWidget(self.btn_refresh_serial, 1, 0, 1, 2)

        layout_fc400.addWidget(QLabel("Baud Rate:"), 2, 0)
        self.fc400_baud_combo = QComboBox()
        self.fc400_baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.fc400_baud_combo.setCurrentText("115200")
        layout_fc400.addWidget(self.fc400_baud_combo, 2, 1)

        layout_fc400.addWidget(QLabel("Parity:"), 3, 0)
        self.fc400_parity_combo = QComboBox()
        self.fc400_parity_combo.addItems(["None", "Even", "Odd"])
        layout_fc400.addWidget(self.fc400_parity_combo, 3, 1)

        layout_fc400.addWidget(QLabel("Stop Bits:"), 4, 0)
        self.fc400_stopbits_combo = QComboBox()
        self.fc400_stopbits_combo.addItems(["1", "2"])
        layout_fc400.addWidget(self.fc400_stopbits_combo, 4, 1)

        layout_fc400.addWidget(QLabel("Slave ID:"), 5, 0)
        self.in_fc400_slave_id = QLineEdit("1")
        layout_fc400.addWidget(self.in_fc400_slave_id, 5, 1)

        layout_fc400.addWidget(QLabel("Read Value:"), 6, 0)
        self.fc400_weight_mode_combo = QComboBox()
        self.fc400_weight_mode_combo.addItems([FC400_GROSS_MODE, FC400_NET_MODE])
        layout_fc400.addWidget(self.fc400_weight_mode_combo, 6, 1)

        layout_fc400.addWidget(QLabel("FC400 Unit:"), 7, 0)
        self.fc400_device_unit_combo = QComboBox()
        self.fc400_device_unit_combo.addItems(["N", "kgf"])
        self.fc400_device_unit_combo.setCurrentText("N")
        self.fc400_device_unit_combo.currentTextChanged.connect(self.on_source_configuration_changed)
        layout_fc400.addWidget(self.fc400_device_unit_combo, 7, 1)

        init_fc400_status = "FC400: set serial port and match FC400 RS-485 settings"
        if not SERIAL_AVAILABLE:
            init_fc400_status = f"FC400: pyserial import failed - {SERIAL_IMPORT_ERROR}"
        self.lbl_fc400_status = QLabel(init_fc400_status)
        self.lbl_fc400_status.setWordWrap(True)
        layout_fc400.addWidget(self.lbl_fc400_status, 8, 0, 1, 2)

        group_fc400.setLayout(layout_fc400)
        left_layout.addWidget(group_fc400, 2, 1)

        group_position = QGroupBox("Mitsubishi MR-MC240N Position Monitor")
        layout_position = QGridLayout()

        self.chk_position_monitor = QCheckBox("Enable MR-MC240N feedback position monitor")
        self.chk_position_monitor.toggled.connect(self.on_position_monitor_toggled)
        if os.name != "nt":
            self.chk_position_monitor.setEnabled(False)
        layout_position.addWidget(self.chk_position_monitor, 0, 0, 1, 2)

        layout_position.addWidget(QLabel("API DLL Path (optional):"), 1, 0)
        self.in_mr_dll_path = QLineEdit("")
        layout_position.addWidget(self.in_mr_dll_path, 1, 1)

        layout_position.addWidget(QLabel("Board ID:"), 2, 0)
        self.in_mr_board_id = QLineEdit("0")
        layout_position.addWidget(self.in_mr_board_id, 2, 1)

        layout_position.addWidget(QLabel("Axis No:"), 3, 0)
        self.in_mr_axis_no = QLineEdit("1")
        layout_position.addWidget(self.in_mr_axis_no, 3, 1)

        layout_position.addWidget(QLabel("Command Units / mm:"), 4, 0)
        self.in_mr_counts_per_mm = QLineEdit("1.0")
        layout_position.addWidget(self.in_mr_counts_per_mm, 4, 1)

        self.chk_mr_auto_start = QCheckBox("Try sscSystemStart() if the board is not running")
        layout_position.addWidget(self.chk_mr_auto_start, 5, 0, 1, 2)

        mr_status = "MR-MC240N: optional monitor, Windows + Mitsubishi API DLL required"
        if os.name != "nt":
            mr_status = f"MR-MC240N: {MR_MC240N_WINDOWS_ONLY_MESSAGE}"
        self.lbl_mr_status = QLabel(mr_status)
        self.lbl_mr_status.setWordWrap(True)
        layout_position.addWidget(self.lbl_mr_status, 6, 0, 1, 2)

        group_position.setLayout(layout_position)
        left_layout.addWidget(group_position, 3, 1)
        
        self.btn_start = QPushButton("Start Test Simulation")
        self.btn_start.clicked.connect(self.toggle_simulation)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        left_layout.addWidget(self.btn_start, 4, 0, 1, 2)
        left_layout.setRowStretch(5, 1)
        
        main_layout.addWidget(left_container, 0)
        
        right_panel = QVBoxLayout()
        top_visual_layout = QHBoxLayout()
        top_visual_layout.setSpacing(12)

        self.chart = SpiderChartCanvas(self, width=6, height=5)
        top_visual_layout.addWidget(self.chart, 3)

        group_camera = QGroupBox("UVC Camera / Ring Deformation")
        group_camera.setMinimumWidth(470)
        layout_camera = QGridLayout()
        layout_camera.setColumnStretch(1, 1)
        layout_camera.setColumnStretch(3, 1)

        layout_camera.addWidget(QLabel("Camera Index:"), 0, 0)
        self.in_camera_index = QLineEdit("0")
        layout_camera.addWidget(self.in_camera_index, 0, 1)

        layout_camera.addWidget(QLabel("Resolution:"), 0, 2)
        self.camera_resolution_combo = QComboBox()
        self.camera_resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.camera_resolution_combo.setCurrentText("1280x720")
        layout_camera.addWidget(self.camera_resolution_combo, 0, 3)

        layout_camera.addWidget(QLabel("Known Ring OD [mm]:"), 1, 0)
        self.in_camera_reference_diameter = QLineEdit(self.jig_combo.currentText().split()[0])
        self.in_camera_reference_diameter.setModified(False)
        self.in_camera_reference_diameter.setToolTip("실제 링 외경(mm)을 입력하면 pixel 값을 mm로 환산합니다.")
        layout_camera.addWidget(self.in_camera_reference_diameter, 1, 1)

        self.btn_camera_toggle = QPushButton("Open Camera")
        self.btn_camera_toggle.clicked.connect(self.toggle_camera)
        layout_camera.addWidget(self.btn_camera_toggle, 1, 2)

        self.btn_camera_baseline = QPushButton("Capture Baseline")
        self.btn_camera_baseline.clicked.connect(self.capture_ring_baseline)
        self.btn_camera_baseline.setEnabled(False)
        layout_camera.addWidget(self.btn_camera_baseline, 1, 3)

        self.btn_camera_clear_baseline = QPushButton("Clear Baseline")
        self.btn_camera_clear_baseline.clicked.connect(self.clear_ring_baseline)
        self.btn_camera_clear_baseline.setEnabled(False)
        layout_camera.addWidget(self.btn_camera_clear_baseline, 2, 2, 1, 2)

        init_camera_status = "Camera: connect a UVC camera and click Open Camera"
        if not CV2_AVAILABLE:
            init_camera_status = f"Camera: OpenCV import failed - {CV2_IMPORT_ERROR}"
            self.btn_camera_toggle.setEnabled(False)
        self.lbl_camera_status = QLabel(init_camera_status)
        self.lbl_camera_status.setWordWrap(True)
        layout_camera.addWidget(self.lbl_camera_status, 2, 0, 1, 2)

        self.lbl_camera_preview = QLabel("Camera preview is not running.")
        self.lbl_camera_preview.setAlignment(Qt.AlignCenter)
        self.lbl_camera_preview.setMinimumSize(440, 300)
        self.lbl_camera_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_camera_preview.setStyleSheet(
            "background-color: #111111; color: #DDDDDD; border: 1px solid #444444; padding: 8px;"
        )
        layout_camera.addWidget(self.lbl_camera_preview, 3, 0, 1, 4)

        self.lbl_ring_metrics = QLabel(
            "Ring measurement:\n"
            "- Open the camera to start preview.\n"
            "- Capture Baseline after the ring is centered."
        )
        self.lbl_ring_metrics.setWordWrap(True)
        self.lbl_ring_metrics.setStyleSheet(
            "background-color: #F8F8F8; border: 1px solid #D0D0D0; padding: 8px; font-family: monospace;"
        )
        layout_camera.addWidget(self.lbl_ring_metrics, 4, 0, 1, 4)

        group_camera.setLayout(layout_camera)
        top_visual_layout.addWidget(group_camera, 2)
        right_panel.addLayout(top_visual_layout, 3)
        
        self.table = QTableWidget(6, 4)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(6):
            self.table.setItem(i, 0, QTableWidgetItem(f"Axis {i+1}"))
            for j in range(1, 4):
                self.table.setItem(i, j, QTableWidgetItem("0.00"))
        right_panel.addWidget(self.table, 1)
        
        btn_layout = QHBoxLayout()
        self.btn_csv = QPushButton("Export CSV (데이터 저장)")
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_pdf = QPushButton("Print Report (A4 성적서 출력)")
        self.btn_pdf.clicked.connect(self.export_pdf)
        btn_layout.addWidget(self.btn_csv)
        btn_layout.addWidget(self.btn_pdf)
        right_panel.addLayout(btn_layout)
        
        main_layout.addLayout(right_panel, 1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_step)
        
        self.refresh_serial_ports()
        self.on_position_monitor_toggled(False)
        self.update_table_headers()
        self.update_chart()
        self.on_input_source_changed(self.input_source)
        self.update_camera_button_state()
        self.update_ring_metrics_label()

    def update_camera_focus(self, size):
        self.lbl_camera.setText(f"Camera Focus Action: Adjusted to {size}")
        if hasattr(self, "in_camera_reference_diameter") and not self.in_camera_reference_diameter.isModified():
            self.in_camera_reference_diameter.setText(size.split()[0])

    def set_camera_preview_message(self, message):
        self.lbl_camera_preview.clear()
        self.lbl_camera_preview.setText(message)

    def update_camera_button_state(self):
        camera_open = self.camera_capture is not None
        if CV2_AVAILABLE:
            self.btn_camera_toggle.setEnabled(True)
            self.btn_camera_toggle.setText("Close Camera" if camera_open else "Open Camera")
        else:
            self.btn_camera_toggle.setEnabled(False)
            self.btn_camera_toggle.setText("Open Camera")

        self.btn_camera_baseline.setEnabled(camera_open and self.latest_ring_measurement is not None)
        self.btn_camera_clear_baseline.setEnabled(self.camera_baseline is not None)

    def get_camera_config(self):
        if not CV2_AVAILABLE:
            raise RuntimeError(f"OpenCV를 불러오지 못했습니다: {CV2_IMPORT_ERROR}")

        camera_index = int(self.in_camera_index.text())
        if camera_index < 0:
            raise ValueError("Camera Index는 0 이상의 정수여야 합니다.")

        resolution_text = self.camera_resolution_combo.currentText()
        frame_width, frame_height = [int(token) for token in resolution_text.split("x")]

        reference_text = self.in_camera_reference_diameter.text().strip()
        reference_diameter_mm = None
        if reference_text:
            reference_diameter_mm = float(reference_text)
            if reference_diameter_mm <= 0:
                raise ValueError("Known Ring OD [mm]는 0보다 커야 합니다.")

        return {
            "camera_index": camera_index,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "reference_diameter_mm": reference_diameter_mm,
        }

    def toggle_camera(self):
        if self.camera_capture is None:
            try:
                self.open_camera()
            except Exception as exc:
                self.close_camera(reset_status=False)
                QMessageBox.warning(self, "Camera Error", f"UVC 카메라를 열지 못했습니다.\n{exc}")
        else:
            self.close_camera()

    def open_camera(self):
        if self.camera_capture is not None:
            return

        config = self.get_camera_config()
        api_preference = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(config["camera_index"], api_preference)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Camera index {config['camera_index']}를 열 수 없습니다.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera_capture = capture
        self.camera_read_failures = 0
        self.latest_ring_measurement = None
        self.camera_timer.start(self.camera_timer_interval_ms)
        self.lbl_camera_status.setText(
            f"Camera: opened UVC index {config['camera_index']} @ {config['frame_width']}x{config['frame_height']}"
        )
        self.set_camera_preview_message("Camera is starting...")
        self.update_ring_metrics_label()
        self.update_camera_button_state()

    def close_camera(self, reset_status=True):
        self.camera_timer.stop()
        if self.camera_capture is not None:
            try:
                self.camera_capture.release()
            except Exception:
                pass

        self.camera_capture = None
        self.camera_read_failures = 0
        self.latest_ring_measurement = None
        if reset_status:
            if CV2_AVAILABLE:
                self.lbl_camera_status.setText("Camera: closed")
            else:
                self.lbl_camera_status.setText(f"Camera: OpenCV import failed - {CV2_IMPORT_ERROR}")
        self.set_camera_preview_message("Camera preview is not running.")
        self.update_ring_metrics_label()
        self.update_camera_button_state()

    def clear_ring_baseline(self):
        self.camera_baseline = None
        self.update_ring_metrics_label()
        self.update_camera_button_state()
        if self.camera_capture is not None:
            self.lbl_camera_status.setText("Camera: running, baseline cleared")
        else:
            self.lbl_camera_status.setText("Camera: baseline cleared")

    def capture_ring_baseline(self):
        if self.latest_ring_measurement is None:
            QMessageBox.warning(self, "Baseline Error", "기준 형상을 캡처하려면 먼저 링이 검출되어야 합니다.")
            return

        measurement = self.latest_ring_measurement
        self.camera_baseline = {
            "major_px": measurement["major_px"],
            "minor_px": measurement["minor_px"],
            "mean_px": measurement["mean_px"],
            "ovality_px": measurement["ovality_px"],
            "reference_diameter_mm": measurement.get("reference_diameter_mm"),
            "mm_per_px": measurement.get("mm_per_px"),
        }
        self.lbl_camera_status.setText("Camera: baseline captured from current ring shape")
        self.update_ring_metrics_label()
        self.update_camera_button_state()

    def camera_timer_step(self):
        if self.camera_capture is None:
            return

        ok, frame = self.camera_capture.read()
        if not ok or frame is None:
            self.camera_read_failures += 1
            if self.camera_read_failures >= 5:
                self.lbl_camera_status.setText("Camera: frame read failed. Check the UVC device connection.")
            return

        self.camera_read_failures = 0
        try:
            measurement = self.measure_ring_from_frame(frame)
        except Exception as exc:
            self.lbl_camera_status.setText(f"Camera: measurement failed - {exc}")
            measurement = None
        self.latest_ring_measurement = measurement

        display_frame = frame.copy()
        display_frame = self.draw_ring_measurement_overlay(display_frame, measurement)
        self.render_camera_frame(display_frame)
        self.update_ring_metrics_label()
        self.update_camera_button_state()

        if measurement is None:
            self.lbl_camera_status.setText("Camera: running, but the ring was not detected")
        elif self.camera_baseline is not None:
            self.lbl_camera_status.setText("Camera: running, ring detected, baseline active")
        else:
            self.lbl_camera_status.setText("Camera: running, ring detected")

    def measure_ring_from_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 40, 120)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours_info = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return None

        frame_height, frame_width = gray.shape
        frame_area = frame_height * frame_width
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)

        best_candidate = None
        best_score = None
        for contour in contours:
            if len(contour) < 20:
                continue

            area = cv2.contourArea(contour)
            if area < frame_area * 0.01:
                continue

            try:
                ellipse = cv2.fitEllipse(contour)
            except cv2.error:
                continue

            (center_x, center_y), (axis_a, axis_b), angle = ellipse
            major_px, minor_px = sorted([float(axis_a), float(axis_b)], reverse=True)
            if major_px <= 10 or minor_px <= 10:
                continue
            if minor_px / max(major_px, 1e-6) < 0.35:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if x <= 1 or y <= 1 or (x + w) >= (frame_width - 1) or (y + h) >= (frame_height - 1):
                continue

            ellipse_area = np.pi * (major_px * 0.5) * (minor_px * 0.5)
            fill_ratio = area / max(ellipse_area, 1.0)
            if fill_ratio < 0.35 or fill_ratio > 1.35:
                continue

            center_distance = float(np.linalg.norm(np.array([center_x, center_y]) - frame_center))
            score = area - (center_distance * 1.5)
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = {
                    "ellipse": ellipse,
                    "center": (float(center_x), float(center_y)),
                    "angle": float(angle),
                    "major_px": major_px,
                    "minor_px": minor_px,
                    "area": float(area),
                    "fill_ratio": float(fill_ratio),
                }

        if best_candidate is None:
            return None

        mean_px = 0.5 * (best_candidate["major_px"] + best_candidate["minor_px"])
        ovality_px = best_candidate["major_px"] - best_candidate["minor_px"]
        measurement = {
            **best_candidate,
            "mean_px": float(mean_px),
            "ovality_px": float(ovality_px),
            "reference_diameter_mm": self.get_camera_config()["reference_diameter_mm"],
        }

        mm_per_px = None
        if self.camera_baseline is not None and self.camera_baseline.get("mm_per_px"):
            mm_per_px = self.camera_baseline["mm_per_px"]
        elif measurement["reference_diameter_mm"] is not None and mean_px > 0:
            mm_per_px = measurement["reference_diameter_mm"] / mean_px
        measurement["mm_per_px"] = mm_per_px

        if mm_per_px is not None:
            measurement["major_mm"] = measurement["major_px"] * mm_per_px
            measurement["minor_mm"] = measurement["minor_px"] * mm_per_px
            measurement["mean_mm"] = measurement["mean_px"] * mm_per_px
            measurement["ovality_mm"] = measurement["ovality_px"] * mm_per_px

        if self.camera_baseline is not None:
            measurement["delta_major_px"] = measurement["major_px"] - self.camera_baseline["major_px"]
            measurement["delta_minor_px"] = measurement["minor_px"] - self.camera_baseline["minor_px"]
            measurement["delta_mean_px"] = measurement["mean_px"] - self.camera_baseline["mean_px"]
            measurement["delta_ovality_px"] = measurement["ovality_px"] - self.camera_baseline["ovality_px"]
            if mm_per_px is not None:
                measurement["delta_major_mm"] = measurement["delta_major_px"] * mm_per_px
                measurement["delta_minor_mm"] = measurement["delta_minor_px"] * mm_per_px
                measurement["delta_mean_mm"] = measurement["delta_mean_px"] * mm_per_px
                measurement["delta_ovality_mm"] = measurement["delta_ovality_px"] * mm_per_px
                measurement["deformation_mm"] = abs(measurement["delta_minor_mm"])

        return measurement

    def draw_ring_measurement_overlay(self, frame, measurement):
        if measurement is None:
            cv2.putText(
                frame,
                "Ring not detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return frame

        cv2.ellipse(frame, measurement["ellipse"], (0, 255, 0), 2)
        center_x, center_y = measurement["center"]
        cv2.circle(frame, (int(center_x), int(center_y)), 4, (255, 0, 0), -1)

        overlay_lines = [
            f"Major: {measurement['major_px']:.1f}px",
            f"Minor: {measurement['minor_px']:.1f}px",
            f"Ovality: {measurement['ovality_px']:.1f}px",
        ]
        if measurement.get("major_mm") is not None:
            overlay_lines = [
                f"Major: {measurement['major_mm']:.3f}mm",
                f"Minor: {measurement['minor_mm']:.3f}mm",
                f"Ovality: {measurement['ovality_mm']:.3f}mm",
            ]
        if measurement.get("deformation_mm") is not None:
            overlay_lines.append(f"Deformation: {measurement['deformation_mm']:.3f}mm")

        origin_y = 30
        for line in overlay_lines:
            cv2.putText(
                frame,
                line,
                (20, origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (36, 255, 12),
                2,
                cv2.LINE_AA,
            )
            origin_y += 26

        return frame

    def render_camera_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.lbl_camera_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_camera_preview.setPixmap(scaled_pixmap)

    def update_ring_metrics_label(self):
        if self.latest_ring_measurement is None:
            baseline_text = "captured" if self.camera_baseline is not None else "not captured"
            self.lbl_ring_metrics.setText(
                "Ring measurement:\n"
                "- Detection: waiting for a visible ring contour\n"
                f"- Baseline: {baseline_text}\n"
                "- Tip: center the ring and use a plain background for stable detection"
            )
            return

        measurement = self.latest_ring_measurement
        lines = [
            "Ring measurement:",
            f"- Major Axis: {measurement['major_px']:.2f} px",
            f"- Minor Axis: {measurement['minor_px']:.2f} px",
            f"- Mean Diameter: {measurement['mean_px']:.2f} px",
            f"- Ovality: {measurement['ovality_px']:.2f} px",
        ]
        if measurement.get("major_mm") is not None:
            lines.extend(
                [
                    f"- Major Axis: {measurement['major_mm']:.3f} mm",
                    f"- Minor Axis: {measurement['minor_mm']:.3f} mm",
                    f"- Mean Diameter: {measurement['mean_mm']:.3f} mm",
                    f"- Ovality: {measurement['ovality_mm']:.3f} mm",
                ]
            )

        if self.camera_baseline is None:
            lines.append("- Baseline: not captured")
        else:
            lines.append("- Baseline: captured")
            if "delta_minor_px" in measurement and "delta_ovality_px" in measurement:
                lines.append(f"- Delta Minor Axis: {measurement['delta_minor_px']:+.2f} px")
                lines.append(f"- Delta Ovality: {measurement['delta_ovality_px']:+.2f} px")
            else:
                lines.append("- Delta: waiting for the next frame after baseline capture")

            if measurement.get("deformation_mm") is not None:
                lines.append(f"- Delta Minor Axis: {measurement['delta_minor_mm']:+.3f} mm")
                lines.append(f"- Deformation Amount: {measurement['deformation_mm']:.3f} mm")

        self.lbl_ring_metrics.setText("\n".join(lines))

    def append_camera_metrics_to_log_row(self, log_row):
        measurement = self.latest_ring_measurement
        if measurement is None:
            return

        log_row["Ring Major [px]"] = round(measurement["major_px"], 2)
        log_row["Ring Minor [px]"] = round(measurement["minor_px"], 2)
        log_row["Ring Mean Diameter [px]"] = round(measurement["mean_px"], 2)
        log_row["Ring Ovality [px]"] = round(measurement["ovality_px"], 2)

        if measurement.get("major_mm") is not None:
            log_row["Ring Major [mm]"] = round(measurement["major_mm"], 3)
            log_row["Ring Minor [mm]"] = round(measurement["minor_mm"], 3)
            log_row["Ring Mean Diameter [mm]"] = round(measurement["mean_mm"], 3)
            log_row["Ring Ovality [mm]"] = round(measurement["ovality_mm"], 3)

        if self.camera_baseline is not None:
            if "delta_minor_px" in measurement and "delta_ovality_px" in measurement:
                log_row["Ring Delta Minor [px]"] = round(measurement["delta_minor_px"], 2)
                log_row["Ring Delta Ovality [px]"] = round(measurement["delta_ovality_px"], 2)
            if measurement.get("deformation_mm") is not None:
                log_row["Ring Delta Minor [mm]"] = round(measurement["delta_minor_mm"], 3)
                log_row["Ring Deformation [mm]"] = round(measurement["deformation_mm"], 3)

    def is_cdaq_mode(self):
        return self.input_source == CDAQ_SOURCE

    def is_fc400_mode(self):
        return self.input_source == FC400_SOURCE

    def is_live_monitor_mode(self):
        return self.input_source in {CDAQ_SOURCE, FC400_SOURCE}

    def is_position_monitor_enabled(self):
        return self.chk_position_monitor.isChecked()

    def get_source_data_unit(self):
        if self.is_fc400_mode():
            return self.fc400_device_unit_combo.currentText()
        return "kgf"

    def convert_value_units(self, value, from_unit, to_unit):
        if from_unit == to_unit:
            return value
        if from_unit == "kgf" and to_unit == "N":
            return value * 9.80665
        if from_unit == "N" and to_unit == "kgf":
            return value / 9.80665
        raise ValueError(f"지원하지 않는 단위 변환입니다: {from_unit} -> {to_unit}")

    def convert_array_units(self, values, from_unit, to_unit):
        if from_unit == to_unit:
            return values
        factor = self.convert_value_units(1.0, from_unit, to_unit)
        return values * factor

    def update_table_headers(self):
        self.table.setHorizontalHeaderLabels(
            [
                "Axis",
                f"Raw Data [{self.data_unit}]",
                f"Zero Offset [{self.data_unit}]",
                f"Calibrated Value [{self.unit}]",
            ]
        )

    def refresh_serial_ports(self):
        if not SERIAL_AVAILABLE:
            self.lbl_fc400_status.setText(f"FC400: pyserial import failed - {SERIAL_IMPORT_ERROR}")
            return

        ports = list_serial_port_names()
        preferred_ports = list_serial_port_names(include_low_confidence=False)
        current_port = self.in_fc400_port.text().strip()
        if preferred_ports and not current_port:
            self.in_fc400_port.setText(preferred_ports[0])

        if ports:
            self.lbl_fc400_status.setText("FC400: detected serial ports - " + ", ".join(ports))
        else:
            self.lbl_fc400_status.setText("FC400: no serial ports detected")

    def update_input_source_buttons(self):
        for source_name, button in self.source_buttons.items():
            previous_state = button.blockSignals(True)
            button.setChecked(source_name == self.input_source)
            button.blockSignals(previous_state)

    def on_input_source_button_clicked(self, source):
        if source == self.input_source:
            self.update_input_source_buttons()
            return
        self.on_input_source_changed(source)

    def on_position_monitor_toggled(self, enabled):
        widgets = [
            self.in_mr_dll_path,
            self.in_mr_board_id,
            self.in_mr_axis_no,
            self.in_mr_counts_per_mm,
            self.chk_mr_auto_start,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

        if enabled:
            if os.name != "nt":
                self.lbl_mr_status.setText(f"MR-MC240N: {MR_MC240N_WINDOWS_ONLY_MESSAGE}")
            else:
                self.lbl_mr_status.setText(
                    "MR-MC240N: monitoring enabled, board/API DLL will be opened when live monitoring starts"
                )
        else:
            self.close_position_monitor()
            if os.name != "nt":
                self.lbl_mr_status.setText(f"MR-MC240N: {MR_MC240N_WINDOWS_ONLY_MESSAGE}")
            else:
                self.lbl_mr_status.setText("MR-MC240N: disabled")

    def on_source_configuration_changed(self, *_args):
        previous_source = getattr(self, "input_source", SIMULATION_SOURCE)
        if self.is_simulating and self.is_live_monitor_mode():
            self.stop_simulation(completed=False, source_override=previous_source)

        self.data_unit = self.get_source_data_unit()
        self.raw_data = [0.0] * 6
        self.sensor_zeros = [0.0] * 6
        self.update_table_headers()
        self.update_table()
        self.update_chart()

    def update_start_button_idle_state(self):
        if self.is_cdaq_mode():
            self.btn_start.setText("Start cDAQ Monitoring")
        elif self.is_fc400_mode():
            self.btn_start.setText("Start FC400 Monitoring")
        else:
            self.btn_start.setText("Start Test Simulation")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")

    def on_input_source_changed(self, source):
        previous_source = getattr(self, "input_source", SIMULATION_SOURCE)
        if self.is_simulating:
            self.stop_simulation(completed=False, source_override=previous_source)

        self.input_source = source
        self.update_input_source_buttons()
        use_cdaq = self.is_cdaq_mode()
        use_fc400 = self.is_fc400_mode()
        if (
            use_fc400
            and previous_source != FC400_SOURCE
            and self.unit_combo.currentText() == "kgf"
            and self.fc400_device_unit_combo.currentText() == "N"
        ):
            self.unit_combo.setCurrentText("N")

        cdaq_widgets = [
            self.in_daq_channel,
            self.in_daq_capacity,
            self.in_daq_sensitivity,
            self.in_daq_bridge_res,
            self.in_daq_excitation,
            self.in_daq_sample_rate,
            self.btn_refresh_daq,
        ]
        for widget in cdaq_widgets:
            widget.setEnabled(use_cdaq)

        fc400_widgets = [
            self.in_fc400_port,
            self.btn_refresh_serial,
            self.fc400_baud_combo,
            self.fc400_parity_combo,
            self.fc400_stopbits_combo,
            self.in_fc400_slave_id,
            self.fc400_weight_mode_combo,
            self.fc400_device_unit_combo,
        ]
        for widget in fc400_widgets:
            widget.setEnabled(use_fc400)

        if use_cdaq:
            self.lbl_status.setText("Status: Ready (cDAQ USB)")
            if not NIDAQMX_AVAILABLE:
                self.lbl_daq_status.setText(f"cDAQ: nidaqmx import failed - {NIDAQMX_IMPORT_ERROR}")
            else:
                self.refresh_cdaq_devices()
            self.close_fc400_client()
        elif use_fc400:
            self.close_cdaq_task()
            self.lbl_status.setText("Status: Ready (FC400 RS-485)")
            if not SERIAL_AVAILABLE:
                self.lbl_fc400_status.setText(f"FC400: pyserial import failed - {SERIAL_IMPORT_ERROR}")
            else:
                self.refresh_serial_ports()
        else:
            self.close_cdaq_task()
            self.close_fc400_client()
            self.lbl_status.setText("Status: Ready")
            self.lbl_daq_status.setText("cDAQ: NI test mode disabled")
            self.lbl_fc400_status.setText("FC400: RS-485 mode disabled")

        self.data_unit = self.get_source_data_unit()
        self.raw_data = [0.0] * 6
        self.sensor_zeros = [0.0] * 6
        self.latest_live_snapshot = [0.0] * 6
        self.latest_live_position_mm = None
        self.latest_live_position_counts = None
        self.position_zero_offset_mm = 0.0
        self.update_table_headers()
        self.update_table()
        self.update_chart()

        self.update_start_button_idle_state()

    def refresh_cdaq_devices(self):
        if not NIDAQMX_AVAILABLE:
            self.lbl_daq_status.setText(f"cDAQ: nidaqmx import failed - {NIDAQMX_IMPORT_ERROR}")
            return

        try:
            device_summaries = []
            first_ai_channel = None
            for device in System.local().devices:
                try:
                    ai_channels = device.ai_physical_chans.channel_names
                except Exception:
                    ai_channels = []

                if not ai_channels:
                    continue

                product_type = ""
                try:
                    product_type = device.product_type
                except Exception:
                    product_type = "NI Device"

                device_summaries.append(f"{device.name} ({product_type})")
                if first_ai_channel is None:
                    first_ai_channel = ai_channels[0]

            current_channel = self.in_daq_channel.text().strip()
            if first_ai_channel and current_channel in {"", "cDAQ1Mod1/ai0"}:
                self.in_daq_channel.setText(first_ai_channel)

            if device_summaries:
                self.lbl_daq_status.setText("cDAQ: " + ", ".join(device_summaries))
            else:
                self.lbl_daq_status.setText("cDAQ: no NI analog-input device found")
        except Exception as exc:
            self.lbl_daq_status.setText(f"cDAQ: device scan failed - {exc}")

    def get_cdaq_config(self):
        physical_channel = self.in_daq_channel.text().strip()
        if not physical_channel:
            raise ValueError("Physical Channel을 입력해주세요. 예: cDAQ1Mod1/ai0")

        rated_load_kgf = float(self.in_daq_capacity.text())
        sensitivity_mv_v = float(self.in_daq_sensitivity.text())
        bridge_resistance = float(self.in_daq_bridge_res.text())
        excitation_voltage = float(self.in_daq_excitation.text())
        sample_rate_hz = float(self.in_daq_sample_rate.text())

        if rated_load_kgf <= 0:
            raise ValueError("Rated Load는 0보다 커야 합니다.")
        if sensitivity_mv_v <= 0:
            raise ValueError("Sensitivity는 0보다 커야 합니다.")
        if bridge_resistance <= 0:
            raise ValueError("Bridge Resistance는 0보다 커야 합니다.")
        if excitation_voltage <= 0:
            raise ValueError("Excitation은 0보다 커야 합니다.")
        if sample_rate_hz <= 0:
            raise ValueError("Sample Rate는 0보다 커야 합니다.")

        return {
            "physical_channel": physical_channel,
            "rated_load_kgf": rated_load_kgf,
            "sensitivity_mv_v": sensitivity_mv_v,
            "bridge_resistance": bridge_resistance,
            "excitation_voltage": excitation_voltage,
            "sample_rate_hz": sample_rate_hz,
        }

    def open_cdaq_task(self):
        if self.cdaq_task is not None:
            return

        if not NIDAQMX_AVAILABLE:
            raise RuntimeError(f"nidaqmx를 불러오지 못했습니다: {NIDAQMX_IMPORT_ERROR}")

        config = self.get_cdaq_config()
        task = nidaqmx.Task()
        try:
            task.ai_channels.add_ai_force_bridge_two_point_lin_chan(
                config["physical_channel"],
                min_val=-config["rated_load_kgf"],
                max_val=config["rated_load_kgf"],
                units=ForceUnits.KILOGRAM_FORCE,
                bridge_config=BridgeConfiguration.FULL_BRIDGE,
                voltage_excit_source=ExcitationSource.INTERNAL,
                voltage_excit_val=config["excitation_voltage"],
                nominal_bridge_resistance=config["bridge_resistance"],
                first_electrical_val=0.0,
                second_electrical_val=config["sensitivity_mv_v"],
                electrical_units=BridgeElectricalUnits.MILLIVOLTS_PER_VOLT,
                first_physical_val=0.0,
                second_physical_val=config["rated_load_kgf"],
                physical_units=BridgePhysicalUnits.KILOGRAM_FORCE,
            )
            # NI 9237 같은 delta-sigma C Series 모듈은 On Demand 읽기 대신
            # 하드웨어 타이밍된 샘플 클록을 명시해야 합니다.
            task.timing.cfg_samp_clk_timing(
                config["sample_rate_hz"],
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(int(config["sample_rate_hz"] * 2), 1000),
            )
            task.in_stream.read_all_avail_samp = True
            task.start()
        except Exception:
            task.close()
            raise

        self.cdaq_task = task
        actual_sample_rate = task.timing.samp_clk_rate
        self.in_daq_sample_rate.setText(f"{actual_sample_rate:.3f}")
        self.lbl_daq_status.setText(
            f"cDAQ: connected to {config['physical_channel']} @ {actual_sample_rate:.3f} S/s"
        )

    def close_cdaq_task(self):
        if self.cdaq_task is None:
            return

        try:
            self.cdaq_task.stop()
        except Exception:
            pass

        try:
            self.cdaq_task.close()
        except Exception:
            pass

        self.cdaq_task = None

    def read_cdaq_value(self):
        opened_here = False
        if self.cdaq_task is None:
            self.open_cdaq_task()
            opened_here = True

        try:
            available_samples = self.cdaq_task.in_stream.avail_samp_per_chan
            if available_samples < 1:
                value = self.cdaq_task.read(number_of_samples_per_channel=1, timeout=1.0)
            else:
                value = self.cdaq_task.read(
                    number_of_samples_per_channel=READ_ALL_AVAILABLE,
                    timeout=1.0,
                )
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    raise RuntimeError("cDAQ 버퍼에 읽을 샘플이 없습니다.")
                return float(value[-1])
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    raise RuntimeError("cDAQ 버퍼에 읽을 샘플이 없습니다.")
                last_value = value[-1]
                if isinstance(last_value, (list, tuple, np.ndarray)):
                    if len(last_value) == 0:
                        raise RuntimeError("cDAQ 버퍼에 읽을 샘플이 없습니다.")
                    last_value = last_value[-1]
                return float(last_value)
            return float(value)
        finally:
            if opened_here and not self.is_simulating:
                self.close_cdaq_task()

    def get_fc400_config(self):
        serial_port = self.in_fc400_port.text().strip()
        if not serial_port:
            raise ValueError("FC400 Serial Port를 입력해주세요. 예: COM5 또는 /dev/ttyUSB0")

        slave_id = int(self.in_fc400_slave_id.text())
        if not 1 <= slave_id <= 247:
            raise ValueError("FC400 Slave ID는 1~247 범위로 입력해주세요.")

        return {
            "port": serial_port,
            "baudrate": int(self.fc400_baud_combo.currentText()),
            "parity": self.fc400_parity_combo.currentText(),
            "stopbits": self.fc400_stopbits_combo.currentText(),
            "slave_id": slave_id,
            "weight_mode": self.fc400_weight_mode_combo.currentText(),
            "device_unit": self.fc400_device_unit_combo.currentText(),
        }

    def open_fc400_client(self):
        if self.fc400_client is not None:
            return

        config = self.get_fc400_config()
        client = FC400ModbusClient(
            port=config["port"],
            baudrate=config["baudrate"],
            parity=config["parity"],
            stopbits=config["stopbits"],
            slave_id=config["slave_id"],
            weight_mode=config["weight_mode"],
        )
        try:
            client.open()
        except Exception:
            client.close()
            raise

        self.fc400_client = client
        self.data_unit = config["device_unit"]
        self.update_table_headers()
        self.lbl_fc400_status.setText(
            f"FC400: connected to {config['port']} @ {config['baudrate']} bps, slave {config['slave_id']}"
        )

    def close_fc400_client(self):
        if self.fc400_client is None:
            return
        try:
            self.fc400_client.close()
        finally:
            self.fc400_client = None

    def read_fc400_measurement(self):
        opened_here = False
        if self.fc400_client is None:
            self.open_fc400_client()
            opened_here = True

        try:
            return self.fc400_client.read_measurement()
        finally:
            if opened_here and not self.is_simulating:
                self.close_fc400_client()

    def get_position_monitor_config(self):
        board_id = int(self.in_mr_board_id.text())
        axis_number = int(self.in_mr_axis_no.text())
        counts_per_mm = float(self.in_mr_counts_per_mm.text())

        if not 0 <= board_id <= 3:
            raise ValueError("MR-MC240N Board ID는 0~3 범위로 입력해주세요.")
        if axis_number <= 0:
            raise ValueError("MR-MC240N Axis No는 1 이상이어야 합니다.")
        if counts_per_mm <= 0:
            raise ValueError("Command Units / mm는 0보다 커야 합니다.")

        return {
            "dll_path": self.in_mr_dll_path.text().strip(),
            "board_id": board_id,
            "axis_number": axis_number,
            "counts_per_mm": counts_per_mm,
            "auto_start_system": self.chk_mr_auto_start.isChecked(),
        }

    def open_position_monitor(self):
        if not self.is_position_monitor_enabled():
            return
        if self.position_monitor is not None:
            return

        config = self.get_position_monitor_config()
        monitor = MrMc240nPositionMonitor(
            board_id=config["board_id"],
            axis_number=config["axis_number"],
            dll_path=config["dll_path"],
            auto_start_system=config["auto_start_system"],
        )
        try:
            monitor.open()
        except Exception:
            monitor.close()
            raise

        self.position_monitor = monitor
        self.lbl_mr_status.setText(
            f"MR-MC240N: board {config['board_id']} axis {config['axis_number']} monitor opened"
        )

    def close_position_monitor(self):
        if self.position_monitor is None:
            return
        try:
            self.position_monitor.close()
        except Exception:
            pass
        finally:
            self.position_monitor = None

    def read_position_feedback(self):
        if not self.is_position_monitor_enabled():
            return None, None

        opened_here = False
        if self.position_monitor is None:
            self.open_position_monitor()
            opened_here = True

        try:
            config = self.get_position_monitor_config()
            raw_counts = self.position_monitor.read_feedback_position_counts()
            absolute_position_mm = raw_counts / config["counts_per_mm"]
            relative_position_mm = absolute_position_mm - self.position_zero_offset_mm
            return relative_position_mm, raw_counts
        finally:
            if opened_here and not self.is_simulating:
                self.close_position_monitor()

    def get_default_stroke_mm(self):
        try:
            return float(self.in_max_len.text())
        except ValueError:
            return 0.0

    def start_live_monitoring(self):
        try:
            if self.is_cdaq_mode():
                self.open_cdaq_task()
                self.data_unit = "kgf"
            elif self.is_fc400_mode():
                self.open_fc400_client()
                self.data_unit = self.get_fc400_config()["device_unit"]
            if self.is_position_monitor_enabled():
                self.open_position_monitor()
        except Exception as exc:
            self.close_cdaq_task()
            self.close_fc400_client()
            self.close_position_monitor()
            QMessageBox.warning(self, "Hardware Connection Error", f"실장비 연결에 실패했습니다.\n{exc}")
            return

        self.is_simulating = True
        self.current_stroke = 0
        self.stroke_data_history = []
        self.stroke_position_history = []
        self.time_series_data = []
        self.time_elapsed = 0.0
        self.latest_live_snapshot = [0.0] * 6
        self.latest_live_position_mm = None
        self.latest_live_position_counts = None
        self.update_table_headers()

        now = datetime.now()
        self.test_start_ts = now.strftime("%Y%m%d_%H%M%S")
        self.test_start_display_time = now.strftime('%Y-%m-%d %H:%M:%S')

        self.sim_state = "LIVE"
        if self.is_cdaq_mode():
            self.btn_start.setText("Stop cDAQ Monitoring")
        else:
            self.btn_start.setText("Stop FC400 Monitoring")
        self.btn_start.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.lbl_status.setText("Status: LIVE (실장비 모니터링)")
        self.timer.start(self.timer_interval)
        self.live_step()

    def live_step(self):
        try:
            if self.is_cdaq_mode():
                load_value = self.read_cdaq_value()
                live_stable = None
            else:
                measurement = self.read_fc400_measurement()
                load_value = measurement["value"]
                live_stable = measurement["stable"]
        except Exception as exc:
            active_source = CDAQ_SOURCE if self.is_cdaq_mode() else FC400_SOURCE
            self.stop_simulation(completed=False, source_override=active_source)
            title = "cDAQ Read Error" if self.is_cdaq_mode() else "FC400 Read Error"
            QMessageBox.critical(self, title, f"하중 값을 읽지 못했습니다.\n{exc}")
            return

        try:
            position_mm, position_counts = self.read_position_feedback()
        except Exception as exc:
            active_source = CDAQ_SOURCE if self.is_cdaq_mode() else FC400_SOURCE
            self.stop_simulation(completed=False, source_override=active_source)
            QMessageBox.critical(self, "MR-MC240N Read Error", f"위치 값을 읽지 못했습니다.\n{exc}")
            return

        self.raw_data = [load_value] * 6
        self.update_table()
        self.update_chart()

        self.time_elapsed += self.timer_interval / 1000.0
        current_calibrated_base = [max(0, self.raw_data[i] - self.sensor_zeros[i]) for i in range(6)]
        self.latest_live_snapshot = current_calibrated_base.copy()
        self.latest_live_position_mm = position_mm
        self.latest_live_position_counts = position_counts

        current_calibrated_display = [
            self.convert_value_units(value, self.data_unit, self.unit)
            for value in current_calibrated_base
        ]

        log_row = {
            'Time [sec]': round(self.time_elapsed, 1),
            'Stroke': 1,
            'State': 'LIVE'
        }
        if position_mm is not None:
            log_row['Position [mm]'] = round(position_mm, 3)
        if position_counts is not None:
            log_row['Position Raw [cmd]'] = int(position_counts)
        if live_stable is not None:
            log_row['Stable'] = int(bool(live_stable))

        for i in range(6):
            log_row[f'Axis {i+1} Raw [{self.data_unit}]'] = round(self.raw_data[i], 3)
        for i in range(6):
            log_row[f'Axis {i+1} Calibrated [{self.unit}]'] = round(current_calibrated_display[i], 3)
        self.append_camera_metrics_to_log_row(log_row)
        self.time_series_data.append(log_row)

        source_name = "cDAQ" if self.is_cdaq_mode() else "FC400"
        if position_mm is not None:
            self.lbl_status.setText(
                f"Status: LIVE ({source_name} {current_calibrated_display[0]:.2f} {self.unit}, Stroke {position_mm:.3f} mm)"
            )
        else:
            self.lbl_status.setText(
                f"Status: LIVE ({source_name} {current_calibrated_display[0]:.2f} {self.unit}, 6채널 동일)"
            )

    def timer_step(self):
        if self.is_live_monitor_mode():
            self.live_step()
        else:
            self.simulation_step()

    def ensure_export_snapshot(self):
        if self.stroke_data_history:
            return
        if self.is_live_monitor_mode() and len(self.time_series_data) > 0:
            self.stroke_data_history = [self.latest_live_snapshot.copy()]
            snapshot_mm = (
                self.latest_live_position_mm
                if self.latest_live_position_mm is not None
                else self.get_default_stroke_mm()
            )
            self.stroke_position_history = [snapshot_mm]

    def change_unit(self, unit):
        self.unit = unit
        self.update_table_headers()
        self.update_table()
        self.update_chart()

    def zero_sensors(self):
        # 모든 물리적 데이터, 영점 기준, 그리고 이전 테스트 기록과 시계열 데이터 완전히 리셋
        self.stroke_data_history = []
        self.stroke_position_history = []
        self.time_series_data = []
        self.time_elapsed = 0.0
        self.latest_live_snapshot = [0.0] * 6
        self.latest_live_position_mm = None
        self.latest_live_position_counts = None
        
        self.test_start_ts = None
        self.test_start_display_time = None

        if self.is_live_monitor_mode():
            try:
                if self.is_cdaq_mode():
                    current_value = self.read_cdaq_value()
                else:
                    current_value = self.read_fc400_measurement()["value"]
            except Exception as exc:
                title = "cDAQ Zero Error" if self.is_cdaq_mode() else "FC400 Zero Error"
                QMessageBox.warning(self, title, f"로드셀 영점 설정에 실패했습니다.\n{exc}")
                return

            self.raw_data = [current_value] * 6
            self.sensor_zeros = self.raw_data.copy()
            self.sim_current_load = 0.0
            status_text = "Status: Ready (Load Cell Tared)"
            message_text = "로드셀 영점(Tare) 및 이전 데이터 초기화가 완료되었습니다."

            if self.is_position_monitor_enabled():
                try:
                    current_position_mm, current_position_counts = self.read_position_feedback()
                    self.position_zero_offset_mm += current_position_mm if current_position_mm is not None else 0.0
                    self.latest_live_position_mm = 0.0
                    self.latest_live_position_counts = current_position_counts
                except Exception as exc:
                    self.lbl_mr_status.setText(f"MR-MC240N: 위치 영점은 유지됨 - {exc}")
        else:
            self.raw_data = [0.0] * 6
            self.sensor_zeros = [0.0] * 6
            self.sim_current_load = 0.0
            self.position_zero_offset_mm = 0.0
            status_text = "Status: Ready (Data Reset)"
            message_text = "센서 영점 맞춤 및 이전 데이터 초기화가 완료되었습니다."

        self.lbl_status.setText(status_text)
        self.update_table()
        self.update_chart()
        QMessageBox.information(self, "Zeroed", message_text)

    def update_table(self):
        for i in range(6):
            raw_display = self.raw_data[i]
            zero_display = self.sensor_zeros[i]
            calibrated_base = self.raw_data[i] - self.sensor_zeros[i]
            calibrated_display = self.convert_value_units(calibrated_base, self.data_unit, self.unit)
            self.table.setItem(i, 1, QTableWidgetItem(f"{raw_display:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{zero_display:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{calibrated_display:.2f}"))

    def get_calibrated_data(self):
        data = []
        for i in range(6):
            calibrated_base = self.raw_data[i] - self.sensor_zeros[i]
            display_val = self.convert_value_units(calibrated_base, self.data_unit, self.unit)
            data.append(max(0, display_val))
        return data

    def update_chart(self):
        data = self.get_calibrated_data()
        interp = self.interp_combo.currentText()
        self.chart.plot_data(data, interpolate_type=interp, unit=self.unit)

    def stop_simulation(self, completed=False, source_override=None):
        active_source = source_override if source_override is not None else self.input_source
        using_live_hardware = active_source in {CDAQ_SOURCE, FC400_SOURCE}
        self.is_simulating = False
        self.sim_state = "IDLE"
        self.timer.stop()
        self.update_start_button_idle_state()

        if using_live_hardware:
            self.close_cdaq_task()
            self.close_fc400_client()
            self.close_position_monitor()
            self.ensure_export_snapshot()
            if active_source == CDAQ_SOURCE:
                self.lbl_status.setText("Status: cDAQ Monitoring Stopped")
            else:
                self.lbl_status.setText("Status: FC400 Monitoring Stopped")
            self.update_table()
            self.update_chart()
            return

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
            if self.is_live_monitor_mode():
                self.start_live_monitoring()
                return

            try:
                self.target_strokes = int(self.in_strokes.text())
                self.target_load = float(self.in_load.text())
                self.hold_time = float(self.in_hold.text())
            except ValueError:
                QMessageBox.warning(self, "Input Error", "숫자를 정확히 입력해주세요.")
                return

            self.is_simulating = True
            self.current_stroke = 0
            self.data_unit = "kgf"
            
            # 새 테스트 시작 시 이력 리셋
            self.stroke_data_history = []
            self.stroke_position_history = []
            self.time_series_data = []
            self.time_elapsed = 0.0
            self.latest_live_position_mm = None
            self.latest_live_position_counts = None
            self.position_zero_offset_mm = 0.0
            self.update_table_headers()
            
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
        
        # 1. 상태에 따른 하중 논리 갱신
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
                self.stroke_position_history.append(self.get_default_stroke_mm())
            self.hold_ticks -= 1
            if self.hold_ticks <= 0:
                self.sim_state = "RELEASING"
        elif self.sim_state == "RELEASING":
            self.sim_current_load -= load_step
            if self.sim_current_load <= 0:
                self.sim_current_load = 0.0

        # 2. 물리적 Raw Data 시뮬레이션 적용 및 UI 반영
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

        # 3. 실시간 시계열 데이터 로깅 (Raw 데이터 및 Calibrated 데이터 모두 저장)
        self.time_elapsed += self.timer_interval / 1000.0
        current_calibrated = [max(0, self.raw_data[i] - self.sensor_zeros[i]) for i in range(6)]
        if self.unit == "N":
            current_calibrated = [v * 9.80665 for v in current_calibrated]
            
        log_row = {
            'Time [sec]': round(self.time_elapsed, 1),
            'Stroke': self.current_stroke + 1 if self.current_stroke < self.target_strokes else self.target_strokes,
            'State': self.sim_state
        }
        
        # 1~6축 Raw Data 추가
        for i in range(6):
            log_row[f'Axis {i+1} Raw'] = round(self.raw_data[i], 2)
            
        # 1~6축 Calibrated Data 추가
        for i in range(6):
            log_row[f'Axis {i+1} Calibrated [{self.unit}]'] = round(current_calibrated[i], 2)
        self.append_camera_metrics_to_log_row(log_row)
            
        self.time_series_data.append(log_row)

        # 4. 스트로크 완료 여부 체크 및 상태 전환
        if self.sim_state == "RELEASING" and self.sim_current_load <= 0:
            self.current_stroke += 1
            if self.current_stroke >= self.target_strokes:
                self.stop_simulation(completed=True)
                QMessageBox.information(self, "Test Complete", f"{self.target_strokes} Strokes 테스트가 완료되었습니다.")
                return
            else:
                self.sim_state = "PULLING"

        if self.is_simulating:
            self.lbl_status.setText(f"Status: {self.sim_state} ({self.current_stroke} / {self.target_strokes} Strokes)")

    def get_stroke_position_value(self, index):
        if index < len(self.stroke_position_history):
            return self.stroke_position_history[index]
        return self.get_default_stroke_mm()

    def export_csv(self):
        self.ensure_export_snapshot()
        if len(self.stroke_data_history) == 0:
            QMessageBox.warning(self, "No Data", "저장할 테스트 데이터가 없습니다.")
            return
            
        ts = self.test_start_ts if self.test_start_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ClampData_{ts}.csv"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV Files (*.csv)")
        
        if file_name:
            all_data = np.array(self.stroke_data_history)
            all_data = self.convert_array_units(all_data, self.data_unit, self.unit)

            # ============ 1. 상단 요약본 (스트로크 통계) ============
            records = []
            for idx, stroke_data in enumerate(all_data):
                row_dict = {'No': f'Stroke {idx + 1}', 'Stroke [mm]': round(self.get_stroke_position_value(idx), 3)}
                for axis_idx in range(6):
                    row_dict[f'Axis {axis_idx + 1} [{self.unit}]'] = round(stroke_data[axis_idx], 2)
                row_dict[f'Average [{self.unit}]'] = round(np.mean(stroke_data), 2)
                records.append(row_dict)

            df_main = pd.DataFrame(records)

            min_vals = np.min(all_data, axis=0)
            max_vals = np.max(all_data, axis=0)
            range_vals = max_vals - min_vals
            avg_vals = np.mean(all_data, axis=0)

            stat_rows = [
                {'No': 'Min', 'Stroke [mm]': round(np.min(self.stroke_position_history), 3) if self.stroke_position_history else self.get_default_stroke_mm()},
                {'No': 'Max', 'Stroke [mm]': round(np.max(self.stroke_position_history), 3) if self.stroke_position_history else self.get_default_stroke_mm()},
                {'No': 'R (Range)', 'Stroke [mm]': '0.00'},
                {'No': 'Ave (Total)', 'Stroke [mm]': round(np.mean(self.stroke_position_history), 3) if self.stroke_position_history else self.get_default_stroke_mm()}
            ]

            for i in range(6):
                col_name = f'Axis {i + 1} [{self.unit}]'
                stat_rows[0][col_name] = round(min_vals[i], 2)
                stat_rows[1][col_name] = round(max_vals[i], 2)
                stat_rows[2][col_name] = round(range_vals[i], 2)
                stat_rows[3][col_name] = round(avg_vals[i], 2)

            avg_col_name = f'Average [{self.unit}]'
            stat_rows[0][avg_col_name] = round(np.min(all_data), 2)
            stat_rows[1][avg_col_name] = round(np.max(all_data), 2)
            stat_rows[2][avg_col_name] = round(np.max(range_vals), 2)
            stat_rows[3][avg_col_name] = round(np.mean(all_data), 2)

            df_stats = pd.DataFrame(stat_rows)
            df_final = pd.concat([df_main, df_stats], ignore_index=True)
            
            # 요약본 파일에 우선 저장
            df_final.to_csv(file_name, index=False, encoding='utf-8-sig')
            
            # ============ 2. 하단 시계열 (Time Series Raw Data) ============
            with open(file_name, 'a', encoding='utf-8-sig') as f:
                f.write('\n\n--- Time Series Raw & Calibrated Data ---\n')
            
            df_ts = pd.DataFrame(self.time_series_data)
            # 모드를 'a'(Append)로 설정하여 기존 CSV 파일 아래에 이어서 작성
            df_ts.to_csv(file_name, mode='a', index=False, encoding='utf-8-sig')
            
            QMessageBox.information(self, "Saved", f"CSV 파일이 성공적으로 저장되었습니다.\n{file_name}")

    def export_pdf(self):
        self.ensure_export_snapshot()
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
        
        # ==================== 완벽하게 고정된 결재란 ====================
        ax_sign = fig.add_axes([0.65, 0.89, 0.27, 0.07])
        ax_sign.axis('off')
        
        rect = Rectangle((0, 0), 0.15, 1, transform=ax_sign.transAxes, 
                         facecolor='#F0F0F0', edgecolor='black', linewidth=0.8)
        ax_sign.add_patch(rect)
        
        ax_sign.text(0.075, 0.5, 'SIGN', fontproperties=font_prop, fontsize=8, 
                     ha='center', va='center', rotation='vertical')

        sign_data = [
            ['EDIT', 'CHECK', 'APPROVE'],
            ['', '', '']
        ]
        
        table_sign = ax_sign.table(cellText=sign_data, cellLoc='center',
                                   bbox=[0.15, 0, 0.85, 1])
        table_sign.auto_set_font_size(False)
        table_sign.set_fontsize(8)
        
        for key, cell in table_sign.get_celld().items():
            row, col = key
            cell.set_edgecolor('black')
            cell.set_linewidth(0.8)
            cell.set_text_props(fontproperties=font_prop)
            if row == 0:
                cell.set_height(0.3)
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_height(0.7)
                cell.set_facecolor('white')
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
        all_data = self.convert_array_units(all_data, self.data_unit, self.unit)
            
        table_data = [['No', 'Stroke\n[mm]', 'Axis 1', 'Axis 2', 'Axis 3', 
                       'Axis 4', 'Axis 5', 'Axis 6', f'Average']]
        
        for idx, stroke_data in enumerate(all_data):
            stroke_mm = f"{self.get_stroke_position_value(idx):.3f}"
            avg = np.mean(stroke_data)
            row = [str(idx+1), stroke_mm] + [f'{v:.2f}' for v in stroke_data] + [f'{avg:.2f}']
            table_data.append(row)
        
        for i in range(10 - len(all_data)):
            table_data.append(['-', '-', '-', '-', '-', '-', '-', '-', '-'])
            
        min_vals = np.min(all_data, axis=0)
        max_vals = np.max(all_data, axis=0)
        range_vals = max_vals - min_vals
        avg_vals = np.mean(all_data, axis=0)
        min_stroke = round(np.min(self.stroke_position_history), 3) if self.stroke_position_history else self.get_default_stroke_mm()
        max_stroke = round(np.max(self.stroke_position_history), 3) if self.stroke_position_history else self.get_default_stroke_mm()
        avg_stroke = round(np.mean(self.stroke_position_history), 3) if self.stroke_position_history else self.get_default_stroke_mm()
        
        table_data.append(['Min', f'{min_stroke:.3f}'] + [f'{v:.2f}' for v in min_vals] + [f'{np.min(all_data):.2f}'])
        table_data.append(['Max', f'{max_stroke:.3f}'] + [f'{v:.2f}' for v in max_vals] + [f'{np.max(all_data):.2f}'])
        table_data.append(['R', '0.00'] + [f'{v:.2f}' for v in range_vals] + [f'{np.max(range_vals):.2f}'])
        table_data.append(['Ave', f'{avg_stroke:.3f}'] + [f'{v:.2f}' for v in avg_vals] + [f'{np.mean(all_data):.2f}'])
        
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

    def closeEvent(self, event):
        self.close_camera(reset_status=False)
        self.close_cdaq_task()
        self.close_fc400_client()
        self.close_position_monitor()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()
            event.accept()
            return

        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showMaximized()
            event.accept()
            return

        super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClampSimulatorApp()
    ex.showFullScreen()
    sys.exit(app.exec_())
