# SmartLabelingApp

**YOLOv11 기반의 Segmentation 라벨링 & 학습 보조 툴**
Windows 환경에서 **Segmentation 마스크 라벨링**을 직관적으로 생성·관리할 수 있는 WinForms 애플리케이션입니다.

---

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/61acca3a-380c-473a-badd-ed4579a1bf91" />

---

![데모 시연](./assets/VIDEO.gif)

## 📦 프로젝트 개요

* **플랫폼:** Windows Forms (.NET Framework 4.8.1)
* **목적:** 이미지 데이터를 직관적으로 라벨링하고, AI 학습 및 추론을 위한 어노테이션을 효율적으로 관리
* **대상 모델:** YOLOv11 (Segmentation 전용, PyTorch `.pt` 및 TensorRT `.engine`)

---

## ✅ 주요 기능

### 🎨 라벨링 도구

* **Box, Polygon, Circle, Triangle, Ngon, Brush, Eraser, Mask** 지원
* **Add Vertex**로 폴리곤 수정 가능
* **회전 핸들**을 이용한 객체 회전 지원
* 다중 객체 선택 및 그룹 편집 (크기/이동/삭제) 지원

### 🤖 AI Tool

* YOLOv11 세그멘테이션 모델을 활용한 **자동 라벨링**
* 추론 결과를 Overlay로 표시 후 수동 라벨과 함께 보정 가능
* GrabCut 기반 보조 세그멘테이션도 포함됨

### 🖼 이미지 뷰어 (Cyotek ImageBox 기반)

* **ZoomToFit**, **Pan Mode**, **Pointer Mode** 지원
* 프레임 영역 내 이미지 로드 및 다중 객체 어노테이션 표시
* 선택 객체에 대해 **바운딩 박스/라벨 태그 표시**

### 🏷 라벨 관리 시스템

* **LabelCreateWindow** 다이얼로그를 통한 라벨 생성
* 색상 + 이름 기반 라벨 칩 관리 (Chip UI)
* 선택 시 테두리/그림자 강조
* **Default Label (DeepSkyBlue)** 자동 생성

### 📂 파일 브라우저 (TreeView)

* 좌측 패널에서 폴더 및 이미지 탐색
* 라벨링 여부에 따라 **상태 뱃지(Dot/Ring/Square/Check)** 표시 (LabelStatusService)
* 저장된 라벨 개수와 클래스별 요약 툴팁 제공

### 📋 컨텍스트 메뉴

* 이미지 캔버스 우클릭 시 제공:

  * `📌 Pointer` (기본 모드)
  * `✋ Pan` (이미지 이동)
  * `📐 Fit` (화면 크기 맞춤)
  * `🗑 Clear Annotations` (라벨 전체 삭제)
  * `🔄 Invert Mask` (마스크 반전)

### ⌨️ 단축키 지원

#### 도구 단축키

* **1** : Pointer Tool
* **2** : Box Tool
* **3** : Polygon Tool
* **4** : Circle Tool
* **5** : Triangle Tool
* **6** : Ngon Tool
* **7** : Brush Tool
* **8** : Eraser Tool
* **9** : Mask Tool
* **0** : AI Tool

#### 편집 단축키

* **Ctrl + Z** : Undo (되돌리기, HistoryService)
* **Ctrl + C** : Copy 선택된 라벨 (ClipboardService)
* **Ctrl + V** : Paste 복사한 라벨
* **Ctrl + A** : 모든 라벨 선택
* **Delete** : 선택한 라벨 삭제
* **화살표키** : 선택 라벨 이동 (Ctrl=빠르게, 기본=5px 단위)
* **Shift + ↑/↓** : 선택 라벨 크기 확대/축소
* **Esc** : 선택/편집 취소

### 📊 결과 관리

* 라벨 데이터는 **이미지 좌표계** 기준으로 저장
* **AnnotationData/** 및 **Result/** 폴더 구조로 관리
* Export 다이얼로그를 통해 데이터 저장/분할 가능
* LabelStatusService를 통해 저장 후에도 **라벨 상태 영구 추적**

### ⚙️ 학습 및 환경 관리 (ProcessRunner 기반)

* **가상환경 생성/점검** (venv) 자동화
* **PyTorch + Ultralytics 설치 및 GPU(CUDA) 자동 감지**
* **YOLO 학습 진행률 추적** (Epoch 기반 퍼센트 표시)
* **Dataset ZIP 해제 및 data.yaml 자동 보정** 지원

---

## 🧰 사용 방법

1. 좌측 TreeView에서 이미지를 선택
2. 우측 도킹 툴바에서 도구(Box, Polygon, Brush 등) 선택
3. 캔버스에서 라벨 생성 및 필요 시 **Add Vertex/회전 핸들**로 수정
4. AI Tool을 통해 자동 세그멘테이션 후 수동 보정
5. 라벨 칩으로 활성 라벨 지정
6. `Export` 메뉴를 통해 결과 저장

---

## 🔧 개발 환경 및 라이브러리

| 구성 요소     | 내용                                                                             |
| --------- | ------------------------------------------------------------------------------ |
| 언어        | C# (.NET Framework 4.8.1)                                                      |
| UI 프레임워크  | WinForms + Guna.UI2                                                            |
| 이미지 뷰어    | `Cyotek.Windows.Forms.ImageBox`                                                |
| AI 추론     | Ultralytics YOLOv11 (Segmentation 전용, `.pt`/`.engine`)                         |
| OpenCV 연동 | `OpenCvSharp4`                                                                 |
| 데이터/상태 관리 | `HistoryService`, `SelectionService`, `ClipboardService`, `LabelStatusService` |
| 학습 파이프라인  | `ProcessRunner`, `YoloTrainer`, `EnvSetup`                                     |
| 기타 컨트롤    | ProgressBarOverlay, Docking UI, Custom Popover                                 |

---

## 📁 프로젝트 구조 (일부)

```
SmartLabelingApp/
 ├── SmartLabelingApp.csproj
 ├── MainForm.cs
 ├── Labeling/
 │    ├── Canvas/
 │    │    └── ImageCanvas.cs
 │    ├── Tools/
 │    │    ├── BoxTool.cs
 │    │    ├── PolygonTool.cs
 │    │    ├── BrushTool.cs
 │    │    └── ...
 │    └── Services/
 │         ├── LabelStatusService.cs
 │         ├── HistoryService.cs
 │         ├── SelectionService.cs
 │         ├── ClipboardService.cs
 │         └── ProcessRunner.cs
 ├── Dialogs/
 │    ├── LabelCreateWindow.cs
 │    ├── BrushSizeWindow.cs
 │    ├── PretrainedWeightsDialog.cs
 │    └── ...
 ├── AnnotationData/
 └── Result/
```

---

## 📦 릴리즈 노트

| 날짜         | 버전     | 주요 변경 내용                                                       |
| ---------- | ------ | --------------------------------------------------------                 |
| 2025-08-18 | v0.9.0 | YOLOv11 Segmentation GUI (Tkinter, Python) 초기 구현                      |
| 2025-08-19 | v0.9.1 | YOLOv11 Segmentation 전용 WinForms 앱 구조 도입                           |
| 2025-08-21 | v1.0.0 | WinForms + Guna.UI2 UI 구축, 기본 툴 (Pointer/Box/Polygon) 지원           |
| 2025-08-26 | v1.1.0 | Circle, Triangle, Ngon, Brush, Eraser, Mask, AI Tool 추가                |
| 2025-08-27 | v1.1.1 | 라벨 칩 UI + Default Label 시스템, Add Vertex 기능 추가                    |
| 2025-08-29 | v1.2.0 | Pretrained Weights 다운로드/진행률 UI 및 가상환경 관리 기능                 |
| 2025-09-01 | v1.2.1 | 단축키 확장 (Ctrl+C, Ctrl+V, Ctrl+A, 회전/이동/확대 단축키) 및 문서 최적화   |
| 2025-09-13 | v1.3.0 | 라벨링 결과 자동 저장/불러오기, Dataset Export 개선                         |
| 2025-09-26 | v1.3.1 | onnx, engine 추론 기능 추가 완료                                           |
