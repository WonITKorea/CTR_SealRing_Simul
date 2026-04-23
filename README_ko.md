# CTR_SealRing_Simul

영문 문서: [README.md](README.md)

## 개요

이 프로그램은 6축 클램프 시험기를 위한 GUI 프로그램입니다.

- 시뮬레이션 모드 지원
- NI `cDAQ-SL1100` 1채널 로드셀 입력을 6채널로 동일 반영하는 테스트 모드 지원
- `UNIPULSE FC400-DAC-FA` + `USB to RS-485` 실장비 입력 지원
- `Mitsubishi MR-MC240N` 위치 피드백 모니터링 지원
- UVC 카메라 + OpenCV 기반 링 변형량 측정 지원

## 설치 방법

1. 의존성 설치:

```bash
pip install -r requirements.txt
```

2. 프로그램 실행:

```bash
python3 main.py
```

## 입력 소스 개요

앱의 `Input Source` 버튼으로 아래 3가지 모드를 선택할 수 있습니다.

- `Simulation`
- `NI cDAQ`
- `FC400 RS-485`

선택한 모드에 따라 왼쪽 설정 영역의 활성/비활성 항목이 자동으로 바뀝니다.

## NI cDAQ USB 모드

NI `cDAQ-SL1100` 번들의 1채널 로드셀 값을 읽어서 6개 축에 동일하게 넣는 테스트 모드입니다.

1. 하드웨어 PC에 NI-DAQmx 드라이버를 설치합니다.
2. 앱에서 `Input Source`를 `NI cDAQ`로 선택합니다.
3. 아래 항목을 실제 장비에 맞게 설정합니다.

- `Physical Channel`
- `Rated Load [kgf]`
- `Sensitivity [mV/V]`
- `Bridge Resistance [Ohm]`
- `Excitation [V]`
- `Sample Rate [S/s]`

4. `Start cDAQ Monitoring` 버튼으로 모니터링을 시작합니다.

## UNIPULSE FC400 RS-485 모드

`UNIPULSE FC400-DAC-FA`의 값을 `USB to RS-485`를 통해 PC에서 읽는 실장비 모드입니다.

1. FC400에 `DC 24V` 전원을 연결합니다.
2. FC400 RS-485 포트를 USB-RS485 컨버터로 PC에 연결합니다.
3. FC400 통신 방식을 `Modbus-RTU`로 맞춥니다.
4. 앱에서 `Input Source`를 `FC400 RS-485`로 선택합니다.
5. 아래 항목을 FC400과 동일하게 설정합니다.

- `Serial Port`
- `Baud Rate`
- `Parity`
- `Stop Bits`
- `Slave ID`
- `Read Value` (`Gross` 또는 `Net`)
- `FC400 Unit` (`N` 또는 `kgf`)

6. `Start FC400 Monitoring` 버튼으로 모니터링을 시작합니다.

로드셀 1채널 값은 프로그램 내부에서 6개 축에 동일하게 반영됩니다.

## 현재 지그 기준 추천 시작값

현재 공유해주신 하드웨어 기준 권장 시작점입니다.

- 로드셀: `KD80S-1KN`
- 정격 범위: `1000 N`
- FC400 공학 단위: `N`
- 앱 표시/리포트 단위: `N`
- 일반적으로 `Gross`부터 시작

FC400 모드에 처음 들어갈 때 장치 단위가 `N`이면 앱 표시 단위도 자동으로 `N`으로 맞춰집니다.

## Mitsubishi MR-MC240N 위치 모니터

이 기능은 선택 사항이며, 하중값과 함께 위치 피드백을 `Stroke [mm]`로 기록할 수 있습니다.

1. Windows PC에 Mitsubishi MR-MC200 계열 유틸리티/드라이버/API 라이브러리를 설치합니다.
2. `mc2xxstd.dll` 또는 `mc2xxstd_x64.dll`이 프로그램 폴더 또는 `PATH`에서 보이도록 둡니다.
3. 앱에서 `Enable MR-MC240N feedback position monitor`를 켭니다.
4. 아래 값을 실제 서보 설정에 맞게 입력합니다.

- `Board ID`
- `Axis No`
- `Command Units / mm`

5. 필요하면 `Try sscSystemStart()` 옵션을 켭니다.

`Command Units / mm` 값은 기구 피치, 감속비, 전자기어비 등 실제 축 설정에 따라 달라집니다.

## UVC 카메라 / 링 변형량 측정

앱 오른쪽의 `UVC Camera / Ring Deformation` 패널에서 일반 UVC 카메라를 열고 링 형상을 측정할 수 있습니다.

1. 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

2. UVC 카메라를 PC에 연결합니다.
3. 아래 항목을 설정합니다.

- `Camera Index`
- `Resolution`
- `Known Ring OD [mm]`

4. `Open Camera`를 눌러 미리보기를 시작합니다.
5. 링을 화면 중앙에 오도록 배치합니다.
6. 무부하 기준 형상에서 `Capture Baseline`을 눌러 기준을 저장합니다.
7. 이후 시험 중 현재 링 형상에서 아래 값을 확인할 수 있습니다.

- `Major Axis`
- `Minor Axis`
- `Mean Diameter`
- `Ovality`
- baseline 대비 변형량

현재 구현은 검출된 링 외곽 컨투어를 타원으로 피팅하고, 기준 형상 대비 `minor axis` 변화량을 주된 변형량으로 계산합니다.

OpenCV 설치 후 Qt `xcb` 플러그인 충돌 에러가 나면 `opencv-python`을 제거하고 `opencv-python-headless`만 남겨두는 것을 권장합니다.

```bash
pip uninstall -y opencv-python
pip install -r requirements.txt
```

## 데이터 저장

앱에서 다음 기능을 사용할 수 있습니다.

- `Export CSV`: 스트로크 요약 + 시계열 데이터 저장
- `Print Report`: A4 PDF 성적서 저장

실장비 모드에서는 마지막 라이브 스냅샷도 저장 대상에 포함됩니다.

## 주요 패키지

- `scikit-fem`: 유한요소 계산 엔진
- `scipy`: 수치해석 및 선형계 계산
- `numpy`: 배열 및 수치 연산
- `PyQt5`: GUI
- `matplotlib`: 차트 및 PDF 출력
- `nidaqmx`: NI 장비 통신
- `pyserial`: FC400 RS-485 통신
- `opencv-python-headless`: PyQt GUI와 충돌을 줄인 UVC 카메라 입력 및 링 형상 분석
