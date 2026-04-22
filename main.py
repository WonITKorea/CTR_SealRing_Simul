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
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import platform

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

SIMULATION_SOURCE = "Simulation"
CDAQ_SOURCE = "cDAQ USB (1ch -> 6ch)"
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
        self.input_source = SIMULATION_SOURCE
        
        self.sensor_zeros = [0.0] * 6
        self.raw_data = [0.0] * 6
        self.latest_live_snapshot = [0.0] * 6
        self.cdaq_task = None
        
        # 데이터 저장소
        self.stroke_data_history = []  # 각 스트로크 최종 결과 저장
        self.time_series_data = []     # 실시간 시계열 로깅 데이터 저장
        self.time_elapsed = 0.0        # 시계열용 누적 시간
        
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

        group_daq = QGroupBox("cDAQ USB Input")
        layout_daq = QGridLayout()

        layout_daq.addWidget(QLabel("Input Source:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems([SIMULATION_SOURCE, CDAQ_SOURCE])
        self.source_combo.currentTextChanged.connect(self.on_input_source_changed)
        layout_daq.addWidget(self.source_combo, 0, 1)

        layout_daq.addWidget(QLabel("Physical Channel:"), 1, 0)
        self.in_daq_channel = QLineEdit("cDAQ1Mod1/ai0")
        layout_daq.addWidget(self.in_daq_channel, 1, 1)

        layout_daq.addWidget(QLabel("Rated Load [kgf]:"), 2, 0)
        self.in_daq_capacity = QLineEdit("100.0")
        layout_daq.addWidget(self.in_daq_capacity, 2, 1)

        layout_daq.addWidget(QLabel("Sensitivity [mV/V]:"), 3, 0)
        self.in_daq_sensitivity = QLineEdit("2.0")
        layout_daq.addWidget(self.in_daq_sensitivity, 3, 1)

        layout_daq.addWidget(QLabel("Bridge Resistance [Ohm]:"), 4, 0)
        self.in_daq_bridge_res = QLineEdit("350")
        layout_daq.addWidget(self.in_daq_bridge_res, 4, 1)

        layout_daq.addWidget(QLabel("Excitation [V]:"), 5, 0)
        self.in_daq_excitation = QLineEdit("5.0")
        layout_daq.addWidget(self.in_daq_excitation, 5, 1)

        layout_daq.addWidget(QLabel("Sample Rate [S/s]:"), 6, 0)
        self.in_daq_sample_rate = QLineEdit(f"{DEFAULT_NI_9237_SAMPLE_RATE:.3f}")
        layout_daq.addWidget(self.in_daq_sample_rate, 6, 1)

        self.btn_refresh_daq = QPushButton("Refresh NI Devices")
        self.btn_refresh_daq.clicked.connect(self.refresh_cdaq_devices)
        layout_daq.addWidget(self.btn_refresh_daq, 7, 0, 1, 2)

        init_daq_status = "cDAQ: select USB mode to scan devices"
        if not NIDAQMX_AVAILABLE:
            init_daq_status = f"cDAQ: nidaqmx import failed - {NIDAQMX_IMPORT_ERROR}"
        self.lbl_daq_status = QLabel(init_daq_status)
        self.lbl_daq_status.setWordWrap(True)
        layout_daq.addWidget(self.lbl_daq_status, 8, 0, 1, 2)

        group_daq.setLayout(layout_daq)
        left_panel.addWidget(group_daq)
        
        self.btn_start = QPushButton("Start Test Simulation")
        self.btn_start.clicked.connect(self.toggle_simulation)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        left_panel.addWidget(self.btn_start)
        left_panel.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area, 1)
        
        right_panel = QVBoxLayout()
        self.chart = SpiderChartCanvas(self, width=6, height=5)
        right_panel.addWidget(self.chart, 3)
        
        self.table = QTableWidget(6, 4)
        self.table.setHorizontalHeaderLabels(["Axis", "Raw Data", "Zero Offset", "Calibrated Value"])
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
        
        main_layout.addLayout(right_panel, 3)
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_step)
        
        self.update_chart()
        self.on_input_source_changed(self.source_combo.currentText())

    def update_camera_focus(self, size):
        self.lbl_camera.setText(f"Camera Focus Action: Adjusted to {size}")

    def is_cdaq_mode(self):
        return self.input_source == CDAQ_SOURCE

    def update_start_button_idle_state(self):
        if self.is_cdaq_mode():
            self.btn_start.setText("Start cDAQ Monitoring")
        else:
            self.btn_start.setText("Start Test Simulation")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")

    def on_input_source_changed(self, source):
        previous_source = getattr(self, "input_source", SIMULATION_SOURCE)
        if self.is_simulating:
            self.stop_simulation(completed=False, source_override=previous_source)

        self.input_source = source
        use_cdaq = self.is_cdaq_mode()
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

        if use_cdaq:
            self.lbl_status.setText("Status: Ready (cDAQ USB)")
            if not NIDAQMX_AVAILABLE:
                self.lbl_daq_status.setText(f"cDAQ: nidaqmx import failed - {NIDAQMX_IMPORT_ERROR}")
            else:
                self.refresh_cdaq_devices()
        else:
            self.close_cdaq_task()
            self.lbl_status.setText("Status: Ready")
            self.lbl_daq_status.setText("cDAQ: USB input disabled (simulation mode)")

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

    def start_cdaq_monitoring(self):
        try:
            self.open_cdaq_task()
        except Exception as exc:
            self.close_cdaq_task()
            QMessageBox.warning(self, "cDAQ Connection Error", f"cDAQ 연결에 실패했습니다.\n{exc}")
            return

        self.is_simulating = True
        self.current_stroke = 0
        self.stroke_data_history = []
        self.time_series_data = []
        self.time_elapsed = 0.0
        self.latest_live_snapshot = [0.0] * 6

        now = datetime.now()
        self.test_start_ts = now.strftime("%Y%m%d_%H%M%S")
        self.test_start_display_time = now.strftime('%Y-%m-%d %H:%M:%S')

        self.sim_state = "LIVE"
        self.btn_start.setText("Stop cDAQ Monitoring")
        self.btn_start.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.lbl_status.setText("Status: LIVE (6채널 동일 입력)")
        self.timer.start(self.timer_interval)
        self.cdaq_step()

    def cdaq_step(self):
        try:
            load_value = self.read_cdaq_value()
        except Exception as exc:
            self.stop_simulation(completed=False, source_override=CDAQ_SOURCE)
            QMessageBox.critical(self, "cDAQ Read Error", f"로드셀 값을 읽지 못했습니다.\n{exc}")
            return

        self.raw_data = [load_value] * 6
        self.update_table()
        self.update_chart()

        self.time_elapsed += self.timer_interval / 1000.0
        current_calibrated_kgf = [max(0, self.raw_data[i] - self.sensor_zeros[i]) for i in range(6)]
        self.latest_live_snapshot = current_calibrated_kgf.copy()

        current_calibrated_display = current_calibrated_kgf.copy()
        if self.unit == "N":
            current_calibrated_display = [v * 9.80665 for v in current_calibrated_display]

        log_row = {
            'Time [sec]': round(self.time_elapsed, 1),
            'Stroke': 1,
            'State': 'LIVE'
        }
        for i in range(6):
            log_row[f'Axis {i+1} Raw'] = round(self.raw_data[i], 2)
        for i in range(6):
            log_row[f'Axis {i+1} Calibrated [{self.unit}]'] = round(current_calibrated_display[i], 2)
        self.time_series_data.append(log_row)

        self.lbl_status.setText(f"Status: LIVE ({current_calibrated_display[0]:.2f} {self.unit}, 6채널 동일)")

    def timer_step(self):
        if self.is_cdaq_mode():
            self.cdaq_step()
        else:
            self.simulation_step()

    def ensure_export_snapshot(self):
        if self.stroke_data_history:
            return
        if self.is_cdaq_mode() and len(self.time_series_data) > 0:
            self.stroke_data_history = [self.latest_live_snapshot.copy()]

    def change_unit(self, unit):
        self.unit = unit
        self.update_table()
        self.update_chart()

    def zero_sensors(self):
        # 모든 물리적 데이터, 영점 기준, 그리고 이전 테스트 기록과 시계열 데이터 완전히 리셋
        self.stroke_data_history = []
        self.time_series_data = []
        self.time_elapsed = 0.0
        self.latest_live_snapshot = [0.0] * 6
        
        self.test_start_ts = None
        self.test_start_display_time = None

        if self.is_cdaq_mode():
            try:
                current_value = self.read_cdaq_value()
            except Exception as exc:
                QMessageBox.warning(self, "cDAQ Zero Error", f"로드셀 영점 설정에 실패했습니다.\n{exc}")
                return

            self.raw_data = [current_value] * 6
            self.sensor_zeros = self.raw_data.copy()
            self.sim_current_load = 0.0
            status_text = "Status: Ready (Load Cell Tared)"
            message_text = "로드셀 영점(Tare) 및 이전 데이터 초기화가 완료되었습니다."
        else:
            self.raw_data = [0.0] * 6
            self.sensor_zeros = [0.0] * 6
            self.sim_current_load = 0.0
            status_text = "Status: Ready (Data Reset)"
            message_text = "센서 영점 맞춤 및 이전 데이터 초기화가 완료되었습니다."
        
        for i in range(6):
            self.table.setItem(i, 1, QTableWidgetItem("0.00"))
            self.table.setItem(i, 2, QTableWidgetItem("0.00"))
            self.table.setItem(i, 3, QTableWidgetItem("0.00"))

        self.lbl_status.setText(status_text)
        self.update_table()
        self.update_chart()
        QMessageBox.information(self, "Zeroed", message_text)

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

    def stop_simulation(self, completed=False, source_override=None):
        active_source = source_override if source_override is not None else self.input_source
        using_cdaq = active_source == CDAQ_SOURCE
        self.is_simulating = False
        self.sim_state = "IDLE"
        self.timer.stop()
        self.update_start_button_idle_state()

        if using_cdaq:
            self.close_cdaq_task()
            self.ensure_export_snapshot()
            self.lbl_status.setText("Status: cDAQ Monitoring Stopped")
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
            if self.is_cdaq_mode():
                self.start_cdaq_monitoring()
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
            
            # 새 테스트 시작 시 이력 리셋
            self.stroke_data_history = []
            self.time_series_data = []
            self.time_elapsed = 0.0
            
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
            if self.unit == "N":
                all_data = all_data * 9.80665

            # ============ 1. 상단 요약본 (스트로크 통계) ============
            records = []
            for idx, stroke_data in enumerate(all_data):
                row_dict = {'No': f'Stroke {idx + 1}', 'Stroke [mm]': self.in_max_len.text()}
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
                {'No': 'Min', 'Stroke [mm]': self.in_max_len.text()},
                {'No': 'Max', 'Stroke [mm]': self.in_max_len.text()},
                {'No': 'R (Range)', 'Stroke [mm]': '0.00'},
                {'No': 'Ave (Total)', 'Stroke [mm]': self.in_max_len.text()}
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

    def closeEvent(self, event):
        self.close_cdaq_task()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClampSimulatorApp()
    ex.show()
    sys.exit(app.exec_())
