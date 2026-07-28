import ctypes
import os
import platform
import re
import struct

try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
    SERIAL_IMPORT_ERROR = ""
except Exception as exc:
    serial = None
    list_ports = None
    SERIAL_AVAILABLE = False
    SERIAL_IMPORT_ERROR = str(exc)

FC400_GROSS_MODE = "Gross"
FC400_NET_MODE = "Net"

MR_MC240N_WINDOWS_ONLY_MESSAGE = "MR-MC240N position board monitoring is supported only on Windows."


def list_serial_port_names(include_low_confidence=True):
    if not SERIAL_AVAILABLE:
        return []

    ports = list(list_ports.comports())
    ports.sort(key=_serial_port_sort_key)
    if not include_low_confidence:
        ports = [port for port in ports if not _is_low_confidence_builtin_port(port)]
    return [port.device for port in ports]


def _serial_port_sort_key(port_info):
    device = (port_info.device or "").lower()
    description = (port_info.description or "").strip().lower()
    hwid = (port_info.hwid or "").strip().lower()
    manufacturer = (getattr(port_info, "manufacturer", None) or "").strip().lower()
    interface = (getattr(port_info, "interface", None) or "").strip().lower()

    text_blob = " ".join(
        value
        for value in [device, description, hwid, manufacturer, interface]
        if value and value != "n/a"
    )

    usb_like_markers = (
        "usb",
        "acm",
        "rs485",
        "ftdi",
        "cp210",
        "silicon labs",
        "wch",
        "ch340",
        "prolific",
        "serial converter",
        "uart",
    )
    is_probably_usb = any(marker in text_blob for marker in usb_like_markers)
    is_low_confidence_builtin = _is_low_confidence_builtin_port(port_info)

    return (
        0 if is_probably_usb else 1,
        1 if is_low_confidence_builtin else 0,
        _natural_sort_key(port_info.device or ""),
    )


def _is_low_confidence_builtin_port(port_info):
    device = (port_info.device or "").strip().lower()
    description = (port_info.description or "").strip().lower()
    hwid = (port_info.hwid or "").strip().lower()

    return device.startswith("/dev/ttys") and description in {"", "n/a"} and hwid in {"", "n/a"}


def _natural_sort_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def get_windows_pe_architecture(file_path):
    """Return x86/x64 for a PE file, or an empty string when it cannot be read."""
    try:
        with open(file_path, "rb") as handle:
            if handle.read(2) != b"MZ":
                return ""
            handle.seek(0x3C)
            pe_offset = struct.unpack("<I", handle.read(4))[0]
            handle.seek(pe_offset)
            if handle.read(4) != b"PE\x00\x00":
                return ""
            machine = struct.unpack("<H", handle.read(2))[0]
    except (OSError, struct.error):
        return ""

    return {
        0x014C: "x86",
        0x8664: "x64",
    }.get(machine, "")


def _to_serial_parity(parity_name):
    mapping = {
        "None": serial.PARITY_NONE,
        "Even": serial.PARITY_EVEN,
        "Odd": serial.PARITY_ODD,
    }
    return mapping[parity_name]


def _to_serial_stopbits(stopbits_name):
    mapping = {
        "1": serial.STOPBITS_ONE,
        "2": serial.STOPBITS_TWO,
    }
    return mapping[stopbits_name]


class FC400ModbusClient:
    """Minimal Modbus-RTU client for UNIPULSE FC400 over 2-wire RS-485."""

    STATUS3_REL_ADDR = 2
    GROSS_WEIGHT_NO_STATUS_REL_ADDR = 10
    NET_WEIGHT_NO_STATUS_REL_ADDR = 12
    READ_START_REL_ADDR = 0
    READ_COUNT = 14

    def __init__(
        self,
        port,
        baudrate,
        parity,
        stopbits,
        slave_id,
        weight_mode,
        timeout=0.3,
    ):
        if not SERIAL_AVAILABLE:
            raise RuntimeError(f"pyserial import failed: {SERIAL_IMPORT_ERROR}")

        self.port = port
        self.baudrate = int(baudrate)
        self.parity = parity
        self.stopbits = stopbits
        self.slave_id = int(slave_id)
        self.weight_mode = weight_mode
        self.timeout = float(timeout)
        self.serial_handle = None

    @staticmethod
    def _crc16_modbus(payload):
        crc = 0xFFFF
        for value in payload:
            crc ^= value
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    @classmethod
    def _append_crc(cls, payload):
        crc = cls._crc16_modbus(payload)
        return payload + struct.pack("<H", crc)

    @classmethod
    def _validate_crc(cls, payload):
        if len(payload) < 3:
            raise RuntimeError("FC400 response is too short.")
        frame = payload[:-2]
        expected_crc = struct.unpack("<H", payload[-2:])[0]
        actual_crc = cls._crc16_modbus(frame)
        if expected_crc != actual_crc:
            raise RuntimeError(
                f"FC400 CRC mismatch. expected=0x{expected_crc:04X}, actual=0x{actual_crc:04X}"
            )

    def open(self):
        if self.serial_handle is not None and self.serial_handle.is_open:
            return

        try:
            self.serial_handle = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=_to_serial_parity(self.parity),
                stopbits=_to_serial_stopbits(self.stopbits),
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            self.serial_handle.reset_input_buffer()
            self.serial_handle.reset_output_buffer()
        except serial.SerialException as exc:
            raise RuntimeError(f"FC400 serial port open failed: {self.port} ({exc})") from exc

    def close(self):
        if self.serial_handle is None:
            return
        try:
            if self.serial_handle.is_open:
                self.serial_handle.close()
        finally:
            self.serial_handle = None

    def _read_exactly(self, size):
        chunks = bytearray()
        while len(chunks) < size:
            piece = self.serial_handle.read(size - len(chunks))
            if not piece:
                break
            chunks.extend(piece)
        return bytes(chunks)

    def read_input_registers(self, start_rel_addr, register_count):
        if self.serial_handle is None or not self.serial_handle.is_open:
            self.open()

        try:
            request = struct.pack(
                ">BBHH",
                self.slave_id,
                0x04,
                int(start_rel_addr),
                int(register_count),
            )
            request = self._append_crc(request)

            self.serial_handle.reset_input_buffer()
            self.serial_handle.write(request)
            self.serial_handle.flush()

            expected_bytes = 5 + register_count * 2
            response = self._read_exactly(expected_bytes)
            if len(response) != expected_bytes:
                raise RuntimeError(
                    f"FC400 response timeout. expected {expected_bytes} bytes, got {len(response)} bytes."
                )
        except serial.SerialException as exc:
            raise RuntimeError(f"FC400 serial I/O failed on {self.port}: {exc}") from exc

        self._validate_crc(response)

        slave_id = response[0]
        function_code = response[1]
        if slave_id != self.slave_id:
            raise RuntimeError(
                f"FC400 slave ID mismatch. expected {self.slave_id}, got {slave_id}."
            )

        if function_code == 0x84:
            exception_code = response[2]
            raise RuntimeError(f"FC400 Modbus exception response: 0x{exception_code:02X}")
        if function_code != 0x04:
            raise RuntimeError(f"Unsupported FC400 function code response: 0x{function_code:02X}")

        byte_count = response[2]
        expected_data_length = register_count * 2
        if byte_count != expected_data_length:
            raise RuntimeError(
                f"FC400 byte count mismatch. expected {expected_data_length}, got {byte_count}."
            )

        payload = response[3:-2]
        registers = []
        for index in range(0, len(payload), 2):
            registers.append(struct.unpack(">H", payload[index:index + 2])[0])
        return registers

    @staticmethod
    def _to_signed_32bit(value):
        if value & 0x80000000:
            return value - 0x100000000
        return value

    def read_measurement(self):
        registers = self.read_input_registers(self.READ_START_REL_ADDR, self.READ_COUNT)
        status3 = registers[self.STATUS3_REL_ADDR]
        decimal_places = status3 & 0x0003

        if self.weight_mode == FC400_NET_MODE:
            hi_word = registers[self.NET_WEIGHT_NO_STATUS_REL_ADDR]
            lo_word = registers[self.NET_WEIGHT_NO_STATUS_REL_ADDR + 1]
        else:
            hi_word = registers[self.GROSS_WEIGHT_NO_STATUS_REL_ADDR]
            lo_word = registers[self.GROSS_WEIGHT_NO_STATUS_REL_ADDR + 1]

        raw_value = (hi_word << 16) | lo_word
        signed_value = self._to_signed_32bit(raw_value)
        scaled_value = signed_value / (10 ** decimal_places)

        return {
            "value": float(scaled_value),
            "decimal_places": decimal_places,
            "status1": registers[0],
            "status2": registers[1],
            "status3": status3,
            "stable": bool((status3 >> 5) & 0x0001),
            "tare_on": bool((status3 >> 7) & 0x0001),
        }


class MrMc240nPositionController:
    """ctypes wrapper for MR-MC200 monitoring and standard-mode axis control."""

    DEFAULT_LIBRARY_CANDIDATES = (
        "mc2xxstd_x64.dll",
        "mc2xxstd.dll",
    )

    SSC_BIT_OFF = 0
    SSC_BIT_ON = 1
    SSC_DIR_PLUS = 0
    SSC_DIR_MINUS = 1

    # mc2xxstd.h axis command/status bit numbers.
    SSC_CMDBIT_AX_SON = 1
    SSC_STSBIT_AX_RDY = 1
    SSC_STSBIT_AX_INP = 2
    SSC_STSBIT_AX_SALM = 6
    SSC_STSBIT_AX_OP = 9
    SSC_STSBIT_AX_ZP = 12
    SSC_STSBIT_AX_OALM = 14
    SSC_STSBIT_AX_OPF = 15

    def __init__(self, board_id, axis_number, dll_path="", auto_start_system=False):
        self.board_id = int(board_id)
        self.channel = 1
        self.axis_number = int(axis_number)
        self.dll_path = dll_path.strip()
        self.auto_start_system = bool(auto_start_system)
        self.library = None
        self._is_open = False
        self._system_start_attempted = False
        self._servo_commanded_on = False
        self._jog_active = False

    @staticmethod
    def is_supported_platform():
        return os.name == "nt"

    def _bind_api(self, name, argtypes):
        try:
            function = getattr(self.library, name)
        except AttributeError:
            return
        function.argtypes = argtypes
        function.restype = ctypes.c_int

    def _load_library(self):
        if not self.is_supported_platform():
            raise RuntimeError(MR_MC240N_WINDOWS_ONLY_MESSAGE)

        if self.library is not None:
            return

        library_candidates = []
        if self.dll_path:
            library_candidates.append(self.dll_path)
        if platform.architecture()[0] == "64bit":
            library_candidates.extend(["mc2xxstd_x64.dll", "mc2xxstd.dll"])
        else:
            library_candidates.extend(["mc2xxstd.dll", "mc2xxstd_x64.dll"])

        load_error = None
        architecture_errors = []
        python_arch = "x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"
        for candidate in library_candidates:
            resolved_candidate = os.path.abspath(candidate) if os.path.isfile(candidate) else candidate
            if os.path.isfile(resolved_candidate):
                dll_arch = get_windows_pe_architecture(resolved_candidate)
                if dll_arch and dll_arch != python_arch:
                    architecture_errors.append(f"{candidate} is {dll_arch}, Python is {python_arch}")
                    continue
            try:
                self.library = ctypes.WinDLL(resolved_candidate)
                break
            except Exception as exc:
                load_error = exc

        if self.library is None:
            architecture_hint = ""
            if architecture_errors:
                architecture_hint = " Architecture mismatch: " + "; ".join(architecture_errors) + "."
            raise RuntimeError(
                "MR-MC240N API library could not be loaded. "
                "Use mc2xxstd_x64.dll with 64-bit Python or mc2xxstd.dll with 32-bit Python."
                + architecture_hint
            ) from load_error

        self._bind_api("sscOpen", [ctypes.c_int])
        self._bind_api("sscClose", [ctypes.c_int])
        self._bind_api("sscGetLastError", [])
        self._bind_api("sscSystemStart", [ctypes.c_int, ctypes.c_int, ctypes.c_int])
        self._bind_api(
            "sscGetCurrentFbPositionFast",
            [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_long)],
        )
        self._bind_api(
            "sscSetCommandBitSignalEx",
            [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        )
        self._bind_api(
            "sscGetStatusBitSignalEx",
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ],
        )
        self._bind_api(
            "sscJogStart",
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_long,
                ctypes.c_short,
                ctypes.c_short,
                ctypes.c_char,
            ],
        )
        self._bind_api(
            "sscJogStop",
            [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        )
        self._bind_api(
            "sscIncStart",
            [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_long,
                ctypes.c_long,
                ctypes.c_short,
                ctypes.c_short,
            ],
        )
        self._bind_api(
            "sscHomeReturnStart",
            [ctypes.c_int, ctypes.c_int, ctypes.c_int],
        )
        self._bind_api(
            "sscDriveStop",
            [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        )
        self._bind_api(
            "sscDriveRapidStop",
            [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        )

    def _raise_api_error(self, action, status_code):
        detailed_error = None
        if self.library is not None:
            try:
                detailed_error = self.library.sscGetLastError()
            except Exception:
                detailed_error = None

        if detailed_error is None:
            raise RuntimeError(f"{action} failed. API status={status_code}.")
        raise RuntimeError(f"{action} failed. API status={status_code}, detail={detailed_error}.")

    def _get_api(self, name):
        if self.library is None:
            raise RuntimeError("MR-MC240N API library is not loaded.")
        try:
            return getattr(self.library, name)
        except AttributeError as exc:
            raise RuntimeError(
                f"{name} is not available in this MR-MC200 API DLL. "
                "Install the API library supplied with the current Position Board Utility2."
            ) from exc

    def _call_api(self, name, *args):
        self.open()
        status = self._get_api(name)(*args)
        if status != 0:
            self._raise_api_error(name, status)

    def open(self):
        self._load_library()
        if self._is_open:
            return

        status = self.library.sscOpen(self.board_id)
        if status != 0:
            self._raise_api_error("sscOpen", status)

        self._is_open = True

    def close(self):
        if not self._is_open or self.library is None:
            return
        try:
            status = self.library.sscClose(self.board_id)
            if status != 0:
                self._raise_api_error("sscClose", status)
        finally:
            self._is_open = False
            self._system_start_attempted = False
            self._servo_commanded_on = False
            self._jog_active = False

    def ensure_running_if_requested(self):
        if not self.auto_start_system or self._system_start_attempted:
            return

        status = self.library.sscSystemStart(self.board_id, self.channel, 0)
        self._system_start_attempted = True
        if status != 0:
            self._raise_api_error("sscSystemStart", status)

    def read_feedback_position_counts(self):
        self.open()

        position = ctypes.c_long()
        status = self.library.sscGetCurrentFbPositionFast(
            self.board_id,
            self.channel,
            self.axis_number,
            ctypes.byref(position),
        )
        if status == 0:
            return int(position.value)

        if self.auto_start_system and not self._system_start_attempted:
            self.ensure_running_if_requested()
            status = self.library.sscGetCurrentFbPositionFast(
                self.board_id,
                self.channel,
                self.axis_number,
                ctypes.byref(position),
            )
            if status == 0:
                return int(position.value)

        self._raise_api_error("sscGetCurrentFbPositionFast", status)

    @staticmethod
    def _validate_motion_values(speed, acceleration_ms, deceleration_ms):
        speed = int(speed)
        acceleration_ms = int(acceleration_ms)
        deceleration_ms = int(deceleration_ms)
        if not 1 <= speed <= 2_147_483_647:
            raise ValueError("Speed must be between 1 and 2147483647 board speed units.")
        if not 0 <= acceleration_ms <= 20_000:
            raise ValueError("Acceleration time must be between 0 and 20000 ms.")
        if not 0 <= deceleration_ms <= 20_000:
            raise ValueError("Deceleration time must be between 0 and 20000 ms.")
        return speed, acceleration_ms, deceleration_ms

    def set_servo_on(self, enabled):
        if enabled:
            self.read_feedback_position_counts()
        bit_value = self.SSC_BIT_ON if enabled else self.SSC_BIT_OFF
        self._call_api(
            "sscSetCommandBitSignalEx",
            self.board_id,
            self.channel,
            self.axis_number,
            self.SSC_CMDBIT_AX_SON,
            bit_value,
        )
        self._servo_commanded_on = bool(enabled)
        if not enabled:
            self._jog_active = False

    def get_axis_status_bit(self, bit_number):
        self.open()
        bit_status = ctypes.c_int()
        status = self._get_api("sscGetStatusBitSignalEx")(
            self.board_id,
            self.channel,
            self.axis_number,
            int(bit_number),
            ctypes.byref(bit_status),
        )
        if status != 0:
            self._raise_api_error("sscGetStatusBitSignalEx", status)
        return bool(bit_status.value)

    def read_axis_status(self):
        status_bits = {
            "servo_ready": self.SSC_STSBIT_AX_RDY,
            "in_position": self.SSC_STSBIT_AX_INP,
            "servo_alarm": self.SSC_STSBIT_AX_SALM,
            "operating": self.SSC_STSBIT_AX_OP,
            "home_complete": self.SSC_STSBIT_AX_ZP,
            "operation_alarm": self.SSC_STSBIT_AX_OALM,
            "operation_complete": self.SSC_STSBIT_AX_OPF,
        }
        return {
            name: self.get_axis_status_bit(bit_number)
            for name, bit_number in status_bits.items()
        }

    def start_jog(self, direction, speed, acceleration_ms, deceleration_ms):
        self.read_feedback_position_counts()
        speed, acceleration_ms, deceleration_ms = self._validate_motion_values(
            speed, acceleration_ms, deceleration_ms
        )
        if direction not in (self.SSC_DIR_PLUS, self.SSC_DIR_MINUS):
            raise ValueError("Jog direction must be SSC_DIR_PLUS or SSC_DIR_MINUS.")

        self._call_api(
            "sscJogStart",
            self.board_id,
            self.channel,
            self.axis_number,
            speed,
            acceleration_ms,
            deceleration_ms,
            bytes([direction]),
        )
        self._jog_active = True

    def stop_jog(self, timeout_ms=3000):
        timeout_ms = int(timeout_ms)
        if not 0 <= timeout_ms <= 65_535:
            raise ValueError("Stop timeout must be between 0 and 65535 ms.")
        self._call_api(
            "sscJogStop",
            self.board_id,
            self.channel,
            self.axis_number,
            timeout_ms,
        )
        self._jog_active = False

    def move_relative(self, distance_counts, speed, acceleration_ms, deceleration_ms):
        self.read_feedback_position_counts()
        distance_counts = int(distance_counts)
        if not -2_147_483_647 <= distance_counts <= 2_147_483_647:
            raise ValueError("Relative distance exceeds the signed 32-bit command range.")
        if distance_counts == 0:
            raise ValueError("Relative distance must not be zero.")
        speed, acceleration_ms, deceleration_ms = self._validate_motion_values(
            speed, acceleration_ms, deceleration_ms
        )

        self._call_api(
            "sscIncStart",
            self.board_id,
            self.channel,
            self.axis_number,
            distance_counts,
            speed,
            acceleration_ms,
            deceleration_ms,
        )

    def start_home_return(self):
        self.read_feedback_position_counts()
        self._call_api(
            "sscHomeReturnStart",
            self.board_id,
            self.channel,
            self.axis_number,
        )

    def stop(self, rapid=False, timeout_ms=3000):
        timeout_ms = int(timeout_ms)
        if not 0 <= timeout_ms <= 65_535:
            raise ValueError("Stop timeout must be between 0 and 65535 ms.")
        api_name = "sscDriveRapidStop" if rapid else "sscDriveStop"
        self._call_api(
            api_name,
            self.board_id,
            self.channel,
            self.axis_number,
            timeout_ms,
        )
        self._jog_active = False


# Backward-compatible name for integrations that imported the original monitor.
MrMc240nPositionMonitor = MrMc240nPositionController
