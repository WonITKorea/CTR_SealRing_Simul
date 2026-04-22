import ctypes
import os
import platform
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


def list_serial_port_names():
    if not SERIAL_AVAILABLE:
        return []
    return sorted(port.device for port in list_ports.comports())


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


class MrMc240nPositionMonitor:
    """Thin ctypes wrapper around Mitsubishi MR-MC200 series API library."""

    DEFAULT_LIBRARY_CANDIDATES = (
        "mc2xxstd_x64.dll",
        "mc2xxstd.dll",
    )

    def __init__(self, board_id, axis_number, dll_path="", auto_start_system=False):
        self.board_id = int(board_id)
        self.channel = 1
        self.axis_number = int(axis_number)
        self.dll_path = dll_path.strip()
        self.auto_start_system = bool(auto_start_system)
        self.library = None
        self._is_open = False
        self._system_start_attempted = False

    @staticmethod
    def is_supported_platform():
        return os.name == "nt"

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
        for candidate in library_candidates:
            try:
                self.library = ctypes.WinDLL(candidate)
                break
            except Exception as exc:
                load_error = exc

        if self.library is None:
            raise RuntimeError(
                "MR-MC240N API library could not be loaded. "
                "Place mc2xxstd.dll or mc2xxstd_x64.dll beside the program or in PATH."
            ) from load_error

        self.library.sscOpen.argtypes = [ctypes.c_int]
        self.library.sscOpen.restype = ctypes.c_int

        self.library.sscClose.argtypes = [ctypes.c_int]
        self.library.sscClose.restype = ctypes.c_int

        self.library.sscGetLastError.argtypes = []
        self.library.sscGetLastError.restype = ctypes.c_int

        self.library.sscSystemStart.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.library.sscSystemStart.restype = ctypes.c_int

        self.library.sscGetCurrentFbPositionFast.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_long),
        ]
        self.library.sscGetCurrentFbPositionFast.restype = ctypes.c_int

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
