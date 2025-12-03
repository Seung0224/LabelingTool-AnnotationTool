using Guna.UI2.WinForms;
using Guna.UI2.WinForms.Enums;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Media.Imaging;

namespace SmartLabelingApp
{
    /// <summary>
    /// GPU 환경 사용 시 Microsoft.ML.OnnxRuntime.Gpu 패키지 필요
    /// CPU 환경 사용 시 Microsoft.ML.OnnxRuntime 패키지 필요
    /// CPU 환경 사용 및 속도 개선 시 Microsoft.ML.OnnxRuntime.DirectML 패키지 필요
    /// CPU /GPU 런타임은 실행환경에 맞게 별도 설치 필요
    /// 중복으로 설치 시 충돌 발생
    /// </summary>

    public partial class MainForm : Form
    {
        #region 0) DeepLearning Fields
        // Yolo Segmentation
        private InferenceSession _onnxSession = null;
        private YoloSegEngine _engineSession = null;

        // PatchCore Anomaly Segmentation
        private InferenceSession _patchcoreSession = null;
        private Artifacts _patchcoreArtifacts;
        #endregion
        #region 1) Constants & Static Data (상수/정적 데이터)


        private const byte COLORMAP_TARGET_START_VALUE_8 = 146;
        private const byte COLORMAP_TARGET_END_VALUE_8 = 255;

        private const ushort COLORMAP_TARGET_START_VALUE_16 = 36828;
        private const ushort COLORMAP_TARGET_END_VALUE_16 = 65535;

        private const string DEFAULT_MODEL_PATH = @"D:\SLA_Model\AnnotationData\best.onnx";
        private string _currentModelName = "UNKNOWN";
        private System.Threading.CancellationTokenSource _autoInferCts;
        public static string _currentRunTypeName = "CPU";

        private const int MODEL_HEADER_H = 39;
        private const int MODEL_HEADER_Y = -43;
        private const int MODEL_HEADER_GAP = 4;

        private const int RUNTYPE_HEADER_H = 39;
        private const int RUNTYPE_HEADER_Y = -43;

        private const int HOTKEY_PANEL_X = -4;
        private const int HOTKEY_PANEL_Y = -46;
        private const int HOTKEY_PANEL_W = 1606;
        private const int HOTKEY_PANEL_H = 40;
        private const int HOTKEY_PANEL_RADIUS = 8;
        private static readonly Color HOTKEY_PANEL_FILL = Color.LightSkyBlue;
        private static readonly Color HOTKEY_PANEL_BORDER = Color.LightGray;

        private const int TOPBAR_H = 32;
        private const int PAD_V = 2;
        private const int PAD_H = 8;
        private const int GAP = 4;

        private const int RIGHT_DOCK_W = 90;
        private const int RIGHT_DOCK_T = 1;

        private const int RIGHT_ICON_PX = 22;
        private const int RIGHT_SLOT_H = 35;
        private const int RIGHT_ICON_GAP = 8;
        private const int RIGHT_ICON_PAD = 2;

        private const int RIGHT_BAR1_H = 446;
        private const int RIGHT_BAR2_H = 255;
        private const int RIGHT_BAR_GAP = 4;

        private const int RIGHT_BAR3_H = 80;
        private const bool RIGHT_BAR3_SNAP_TO_VIEWER = true;
        private const int RIGHT_BAR3_TAIL = 5;
        private const int RIGHT_BAR_MIN_H = 40;

        private const int ACTION3_TOP = 5;
        private const int ACTION3_GAP = 8;

        private const int FRAME_X = 207;
        private const int FRAME_X_OFFSET = 85;
        private const int FRAME_Y = 46;
        private const int FRAME_Y_OFFSET = 200;
        private const int FRAME_W = 800;
        private const int FRAME_H = 547;
        private const int FRAME_BORDER = 2;

        private const int VIEWER_MIN_W = 1024;
        private const int VIEWER_HORIZONTAL_MARGIN = 320;

        private const int LOG_X = 6;
        private const int LOG_W = 4;
        private const int LOG_GAP = 39;           // imageDisplay와 logDisplay 사이 간격
        private const int OUTER_MARGIN = 9;      // 작업영역 하단 여백(바닥에 딱 붙지 않게)

        private const int LABEL_CHIP_MIN_W = 74;

        private string _savePathOverride;

        private Guna2Panel _logPanel;
        private ListBox _logListBox;

        private int VIEWER_MAX_W =>
            Math.Max(VIEWER_MIN_W, Screen.FromControl(this).WorkingArea.Width - VIEWER_HORIZONTAL_MARGIN);

        private string _lastYoloExportRoot;
        private readonly Dictionary<string, Color> _classColorMap = new Dictionary<string, Color>(StringComparer.OrdinalIgnoreCase);

        private string _lastExportZipPath;

        private struct LabelInfo
        {
            public string Name { get; set; }
            public Color Color { get; set; }

            public LabelInfo(string name, Color color) : this()
            {
                Name = name ?? string.Empty;
                Color = color;
            }
        }

        private static readonly HashSet<string> _imgExts =
            new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff" };

        private Bitmap _sourceImage;

        #endregion
        #region 2) UI Components (컨트롤/뷰 구성요소)
        private Guna.UI2.WinForms.Guna2Panel _hotkeyPanel;
        private System.Windows.Forms.Label _hotkeyLabel;


        private readonly Guna2BorderlessForm _borderless;
        private readonly Guna2Elipse _elipse;
        private readonly Guna2ShadowForm _shadow;


        private readonly Guna2GradientPanel _topBar;
        private Guna2ControlBox _btnMin;
        private Guna2ControlBox _btnMax;
        private Guna2ControlBox _btnClose;
        private Guna2DragControl _dragControl;
        private Guna2DragControl _dragTitle;


        private Guna2Panel _canvasHost;
        private Panel _canvasLayer;
        private readonly ImageCanvas _canvas;

        private ToolStripMenuItem _miAddVertex;
        private System.Drawing.PointF _lastCtxImgPointImg;

        private Guna2Panel _modelHeaderPanel;
        private Label _modelHeaderLabel;

        private Guna2Panel _runtypeHeaderPanel;
        private Label _runtypeHeaderLabel;



        private Guna2Panel _rightRail;
        private Guna2Panel _rightToolDock;
        private Guna2Panel _rightToolDock2;
        private Guna2Panel _rightToolDock3;
        private FlowLayoutPanel _rightTools;
        private FlowLayoutPanel _rightTools2;
        private FlowLayoutPanel _rightTools3;


        private readonly Guna2ImageButton _btnPointer;
        private readonly Guna2ImageButton _btnTriangle;
        private readonly Guna2ImageButton _btnBox;
        private readonly Guna2ImageButton _btnNgon;
        private readonly Guna2ImageButton _btnCircle;
        private readonly Guna2ImageButton _btnBrush;
        private readonly Guna2ImageButton _btnEraser;
        private readonly Guna2ImageButton _btnMask;
        private readonly Guna2ImageButton _btnAI;
        private readonly Guna2ImageButton _btnPolygon;
        private readonly Guna2ImageButton _btnPrev;
        private readonly Guna2ImageButton _btnNext;
        private readonly Guna2ImageButton _btnToggle;
        private Guna2Panel _slotToggle;
        private bool _toggleOn = false;
        private Panel _navRow;

        private readonly Guna2Button _btnAdd;
        private readonly Guna2Button _btnOpen;
        private readonly Guna2Button _btnSave;
        private readonly Guna2Button _btnExport;
        private readonly Guna2Button _btnTrain;
        private readonly Guna2Button _btnInfer;


        private Guna2Panel _leftRail;
        private Guna2Panel _leftDock;
        private TreeView _fileTree;
        #endregion
        #region 3) State & Model (상태/모델)
        private string _currentImagePath;
        private BrushSizeWindow _brushWin;
        private Control _brushAnchorBtn;
        private int _brushDiameterPx = 18;
        private string _currentFolder;


        private LabelCreateWindow _labelWin;
        private Control _labelAnchorBtn;
        private int _labelSeq = 1;


        private enum AiSubMode { Off, Free, Roi }
        private AiSubMode _aiSubMode = AiSubMode.Off;

        private RectangleF? _lastRoiNorm = null;
        private Image _aiRainbowBg = null;

        private ToolTip _tt = new ToolTip
        {
            AutoPopDelay = 8000,
            InitialDelay = 400,
            ReshowDelay = 100,
            ShowAlways = true,
            IsBalloon = true
        };
        #endregion
        #region 4) Initialization & Layout (Constructor)
        public MainForm()
        {

            Text = "SmartLabelingApp";
            StartPosition = FormStartPosition.Manual;
            this.WindowState = FormWindowState.Maximized;
            this.Load += (_, __) =>
            {

                this.WindowState = FormWindowState.Maximized;
            };

            MinimumSize = new Size(900, 600);
            BackColor = Color.White;
            FormBorderStyle = FormBorderStyle.None;

            this.KeyPreview = true;
            this.KeyDown += (s, e) =>
            {
                bool isShortcut =
                    (e.Control && (e.KeyCode == Keys.A || e.KeyCode == Keys.C || e.KeyCode == Keys.V || e.KeyCode == Keys.Z)) ||
                    e.KeyCode == Keys.Delete || e.KeyCode == Keys.Escape;

                if (isShortcut && _canvas != null && !_canvas.Focused)
                    _canvas.Focus();
            };


            _elipse = new Guna2Elipse { BorderRadius = 2, TargetControl = this };
            _borderless = new Guna2BorderlessForm
            {
                ContainerControl = this,
                BorderRadius = 2,
                TransparentWhileDrag = true,
                ResizeForm = true
            };
            _shadow = new Guna2ShadowForm { ShadowColor = Color.Black };


            _canvasHost = new Guna2Panel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(8, 0, 8, 8),
                BorderColor = Color.FromArgb(220, 224, 230),
                BorderThickness = 1,
                BorderRadius = 12,
                FillColor = Color.White
            };
            _canvasHost.ShadowDecoration.Parent = _canvasHost;
            Controls.Add(_canvasHost);


            _topBar = new Guna2GradientPanel
            {
                Dock = DockStyle.Top,
                Height = TOPBAR_H,
                FillColor = Color.FromArgb(120, 161, 255),
                FillColor2 = Color.FromArgb(146, 228, 255),
                Padding = new Padding(PAD_H, PAD_V, PAD_H, PAD_V)
            };
            _topBar.DoubleClick += (s, e) => ToggleMaximizeRestore();
            Controls.Add(_topBar);

            _dragControl = new Guna2DragControl
            {
                TargetControl = _topBar,
                DockIndicatorTransparencyValue = 0.6f,
                UseTransparentDrag = true
            };

            int toolEdge = TOPBAR_H - PAD_V * 2;

            var rightPanel = new Panel
            {
                Dock = DockStyle.Right,
                Width = toolEdge * 3 + GAP * 4,
                BackColor = Color.Transparent
            };
            _topBar.Controls.Add(rightPanel);

            int y = PAD_V;
            int cbEdge = toolEdge;

            _btnMin = new Guna2ControlBox
            {
                ControlBoxType = ControlBoxType.MinimizeBox,
                FillColor = Color.Transparent,
                IconColor = Color.Black,
                BorderRadius = 2,
                UseTransparentBackground = true,
                Size = new Size(cbEdge, cbEdge),
                Location = new Point(rightPanel.Width - (cbEdge * 3 + GAP * 3), y)
            };
            _btnMax = new Guna2ControlBox
            {
                ControlBoxType = ControlBoxType.MaximizeBox,
                FillColor = Color.Transparent,
                IconColor = Color.Black,
                BorderRadius = 2,
                UseTransparentBackground = true,
                Size = new Size(cbEdge, cbEdge),
                Location = new Point(rightPanel.Width - (cbEdge * 2 + GAP * 2), y)
            };
            _btnClose = new Guna2ControlBox
            {
                FillColor = Color.Transparent,
                IconColor = Color.Black,
                HoverState = { FillColor = Color.FromArgb(255, 80, 80), IconColor = Color.White },
                BorderRadius = 2,
                UseTransparentBackground = true,
                Size = new Size(cbEdge, cbEdge),
                Location = new Point(rightPanel.Width - (cbEdge + GAP), y)
            };
            rightPanel.Controls.Add(_btnMin);
            rightPanel.Controls.Add(_btnMax);
            rightPanel.Controls.Add(_btnClose);

            var lblTitle = new Label
            {
                Text = "SmartLabelingApp",
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                AutoEllipsis = true,
                BackColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 10, FontStyle.Regular)
            };
            lblTitle.DoubleClick += (s, e) => ToggleMaximizeRestore();
            _topBar.Controls.Add(lblTitle);

            _dragTitle = new Guna2DragControl
            {
                TargetControl = lblTitle,
                DockIndicatorTransparencyValue = 0.6f,
                UseTransparentDrag = true
            };


            _rightRail = new Guna2Panel { Dock = DockStyle.Right, Width = RIGHT_DOCK_W };
            _canvasHost.Controls.Add(_rightRail);


            _rightToolDock = new Guna2Panel
            {
                Dock = DockStyle.Top,
                Height = RIGHT_BAR1_H,
                Padding = new Padding(6, 8, 6, 8),
                FillColor = Color.Transparent,
                BackColor = Color.Transparent,
                BorderThickness = 2,
                BorderColor = Color.Silver,
                BorderRadius = 2
            };
            _rightTools = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true,
                Padding = new Padding(0),
                Margin = new Padding(0),
                BackColor = Color.Transparent
            };
            _rightToolDock.Controls.Add(_rightTools);
            _rightRail.Controls.Add(_rightToolDock);


            _rightToolDock2 = new Guna2Panel
            {
                Dock = DockStyle.None,
                Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right,
                Height = RIGHT_BAR2_H,
                Padding = new Padding(6, 8, 6, 8),
                FillColor = Color.Transparent,
                BackColor = Color.Transparent,
                BorderThickness = 2,
                BorderColor = Color.Silver,
                BorderRadius = 2
            };
            _rightTools2 = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true,
                Padding = new Padding(0),
                Margin = new Padding(0),
                BackColor = Color.Transparent
            };
            _rightToolDock2.Controls.Add(_rightTools2);
            _rightRail.Controls.Add(_rightToolDock2);

            _rightToolDock3 = new Guna2Panel
            {
                Dock = DockStyle.None,
                Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right,
                Height = RIGHT_BAR3_H,
                Padding = new Padding(6, 8, 6, 8),
                FillColor = Color.Transparent,
                BackColor = Color.Transparent,
                BorderThickness = 2,
                BorderColor = Color.Silver,
                BorderRadius = 2
            };
            _rightTools3 = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true,
                Padding = new Padding(0),
                Margin = new Padding(0),
                BackColor = Color.Transparent
            };
            _rightTools3.Padding = new Padding(0, ACTION3_TOP, 0, ACTION3_TOP);
            _rightToolDock3.Controls.Add(_rightTools3);
            _rightRail.Controls.Add(_rightToolDock3);

            int innerW3 = RIGHT_DOCK_W - _rightToolDock3.Padding.Horizontal;

            _rightToolDock3.Resize += (s, e) =>
            {
                int w3 = Math.Max(LABEL_CHIP_MIN_W, _rightToolDock3.ClientSize.Width - _rightToolDock3.Padding.Horizontal);
                if (_btnExport != null) { _btnExport.Width = w3; _btnExport.Margin = new Padding(0, 0, 0, ACTION3_GAP); }
                if (_btnTrain != null) { _btnTrain.Width = w3; _btnTrain.Margin = new Padding(0, 0, 0, ACTION3_GAP); }
                if (_navRow != null) { _navRow.Width = w3; LayoutNavRow(); }
                if (_slotToggle != null) { _slotToggle.Width = w3; }
                if (_btnInfer != null) { _btnInfer.Width = w3; _btnInfer.Margin = new Padding(0, 0, 0, ACTION3_GAP); }
            };


            int innerW2 = RIGHT_DOCK_W - _rightToolDock2.Padding.Horizontal;
            _btnAdd = new Guna2Button
            {
                Text = "ADD",
                BorderRadius = 12,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                AutoSize = false,
                Size = new Size(innerW2, toolEdge),
                Margin = new Padding(0, 0, 0, 8),
                TabStop = false
            };
            _btnAdd.Click += OnAddClick;
            _btnAdd.MouseDown += (s, e) => { if (!_canvas.Focused) _canvas.Focus(); };
            _rightTools2.Controls.Add(_btnAdd);

            _btnAdd.Width = Math.Max(LABEL_CHIP_MIN_W, _rightToolDock2.ClientSize.Width - _rightToolDock2.Padding.Horizontal);
            AdjustLabelChipWidths();

            _rightToolDock2.Resize += (s, e) =>
            {
                int targetW = Math.Max(LABEL_CHIP_MIN_W, _rightToolDock2.ClientSize.Width - _rightToolDock2.Padding.Horizontal);
                _btnAdd.Width = targetW;
                AdjustLabelChipWidths();
            };

            _btnOpen = new Guna2Button
            {
                Text = "1:OPEN",
                BorderRadius = 12,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                AutoSize = false,
                Size = new Size(innerW3, toolEdge),
                Margin = new Padding(0, 0, 0, 8),
                TabStop = false
            };
            _btnOpen.Click += OnOpenClick;
            _rightTools3.Controls.Add(_btnOpen);

            _btnSave = new Guna2Button
            {
                Text = "2:SAVE",
                BorderRadius = 12,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                AutoSize = false,
                Size = new Size(innerW3, toolEdge),
                Margin = new Padding(0, 0, 0, 8),
                TabStop = false
            };
            _btnSave.Click += OnSaveClick;
            _rightTools3.Controls.Add(_btnSave);

            _btnExport = new Guna2Button
            {
                Text = "3:EXPRT",
                BorderRadius = 12,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                AutoSize = false,
                Size = new Size(innerW3, toolEdge),
                Margin = new Padding(0, 0, 0, 8),
                TabStop = false
            };
            _btnExport.Click += OnExportClick;
            _rightTools3.Controls.Add(_btnExport);

            _btnTrain = new Guna2Button
            {
                Text = "4:TRAIN",
                BorderRadius = 12,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                AutoSize = false,
                Size = new Size(innerW3, toolEdge),
                Margin = new Padding(0, 0, 0, 8),
                TabStop = false
            };
            _btnTrain.Click += OnTrainClick;
            _rightTools3.Controls.Add(_btnTrain);

            _btnInfer = new Guna2Button
            {
                Text = "5:INFER",
                BorderRadius = 12,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.Transparent,
                ForeColor = Color.Black,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                AutoSize = false,
                Size = new Size(innerW3, toolEdge),
                Margin = new Padding(0, 0, 0, 8),
                TabStop = false
            };
            _btnInfer.Click += OnInferClick;
            _rightTools3.Controls.Add(_btnInfer);

            _btnPrev = CreateToolIcon(Properties.Resources.Prev, "Prev Image", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnNext = CreateToolIcon(Properties.Resources.Next, "Next Image", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnToggle = CreateToolIcon(Properties.Resources.Toggleoff2, "AutoRun Toggle", RIGHT_SLOT_H, 38);

            var prevSlot = WrapToolSlot(_btnPrev, innerW3 / 2, RIGHT_SLOT_H);
            var nextSlot = WrapToolSlot(_btnNext, innerW3 / 2, RIGHT_SLOT_H);
            _slotToggle = WrapToolSlot(_btnToggle, innerW3, RIGHT_SLOT_H);

            _btnPrev.Click += OnPrevClick;
            _btnNext.Click += OnNextClick;
            _btnToggle.Click += OnToggleClick;

            _navRow = new FlowLayoutPanel
            {
                Height = RIGHT_SLOT_H,
                Width = innerW3,
                Margin = new Padding(0, 0, 0, 8),
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false,
                BackColor = Color.Transparent
            };

            _navRow.Controls.Add(prevSlot);
            _navRow.Controls.Add(nextSlot);
            prevSlot.BorderColor = nextSlot.BorderColor = Color.LightGray;

            _rightTools3.Controls.Add(_navRow);
            _rightTools3.Controls.Add(_slotToggle);


            _tt.SetToolTip(_btnOpen, "이미지 파일 또는 폴더를 열어 작업을 시작합니다.");
            _tt.SetToolTip(_btnAdd, "새 라벨(클래스)을 추가합니다.");
            _tt.SetToolTip(_btnExport, "라벨링 데이터를 YOLO Seg 데이터셋으로 내보냅니다.");
            _tt.SetToolTip(_btnTrain, "Export한 데이터셋으로 YOLO Seg 모델을 학습하여 .pt 모델과 .onnx 모델을 생성합니다.");
            _tt.SetToolTip(_btnInfer, "선택한 ONNX 모델로 추론하고 오버레이로 확인합니다.");


            _btnPointer = CreateToolIcon(Properties.Resources.Arrow, "Pointer", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnCircle = CreateToolIcon(Properties.Resources.Circle, "Circle", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnTriangle = CreateToolIcon(Properties.Resources.Triangle, "Triangle", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnBox = CreateToolIcon(Properties.Resources.Rectangle, "Box", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnNgon = CreateToolIcon(Properties.Resources.Ngon, "N-gon", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnPolygon = CreateToolIcon(Properties.Resources.Polyline, "Polygon", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnBrush = CreateToolIcon(Properties.Resources.Brush, "Brush", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnEraser = CreateToolIcon(Properties.Resources.Eraser, "Eraser", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnMask = CreateToolIcon(Properties.Resources.Masktoggle, "Mask", RIGHT_SLOT_H, RIGHT_ICON_PX);
            _btnAI = CreateToolIcon(Properties.Resources.AI, "AI", RIGHT_SLOT_H, RIGHT_ICON_PX);

            _btnPointer.Click += delegate { SetTool(ToolMode.Pointer, _btnPointer); };
            _btnCircle.Click += delegate { SetTool(ToolMode.Circle, _btnCircle); };
            _btnTriangle.Click += delegate
            {
                SetTool(ToolMode.Polygon, _btnTriangle);
                _canvas.SetPolygonPreset(PolygonPreset.Triangle, 0);
                if (!_canvas.Focused) _canvas.Focus();
            };
            _btnBox.Click += delegate
            {
                SetTool(ToolMode.Polygon, _btnBox);
                _canvas.SetPolygonPreset(PolygonPreset.RectBox, 0);
                if (!_canvas.Focused) _canvas.Focus();
            };
            _btnNgon.Click += (s, e) =>
            {
                int currentSides = 5;
                var polyTool = _canvas.GetTool(ToolMode.Ngon) as PolygonTool;
                if (polyTool != null && polyTool.RegularSides >= 3)
                    currentSides = polyTool.RegularSides;

                using (var dlg = new NgonSidesDialog(currentSides))
                {
                    dlg.StartPosition = FormStartPosition.CenterParent;
                    var result = dlg.ShowDialog(this);
                    if (result != DialogResult.OK) return;
                    int sides = dlg.Sides;
                    if (sides < 3) sides = 3;
                    if (polyTool != null) polyTool.RegularSides = sides;
                }
                SetTool(ToolMode.Ngon, _btnNgon);
                if (!_canvas.Focused) _canvas.Focus();
            };
            _btnPolygon.Click += delegate
            {
                SetTool(ToolMode.Polygon, _btnPolygon);
                _canvas.SetPolygonPreset(PolygonPreset.Free, 0);
                if (!_canvas.Focused) _canvas.Focus();
            };
            _btnBrush.Click += delegate
            {
                SetTool(ToolMode.Brush, _btnBrush);
                ShowBrushWindowNear(_btnBrush.Parent != null ? _btnBrush.Parent : (Control)_btnBrush);
            };
            _btnEraser.Click += delegate
            {
                SetTool(ToolMode.Eraser, _btnEraser);
                ShowBrushWindowNear(_btnEraser.Parent != null ? _btnEraser.Parent : (Control)_btnEraser);
            };
            _btnMask.Click += delegate
            {
                SetTool(ToolMode.Mask, _btnMask);
                if (_brushWin != null && _brushWin.Visible) _brushWin.Hide();
            };


            _btnAI.MouseUp += (s, e) =>
            {
                if (e.Button == MouseButtons.Left)
                {
                    EnterAiFreeformMode();
                }
                else if (e.Button == MouseButtons.Right)
                {
                    EnterAiRoiMode();
                }
            };

            _tt.SetToolTip(_btnPointer, "포인터: 선택/이동/편집 모드로 전환합니다.");
            _tt.SetToolTip(_btnCircle, "원/타원: 드래그하여 원형(타원) 마스크를 그립니다.");
            _tt.SetToolTip(_btnTriangle, "삼각형: 드래그하여 삼각형 마스크를 그립니다.");
            _tt.SetToolTip(_btnBox, "사각형: 드래그하여 직사각형 마스크를 그립니다.");
            _tt.SetToolTip(_btnNgon, "N-각형: 변의 개수를 설정해 규칙 다각형을 그립니다.");
            _tt.SetToolTip(_btnPolygon, "폴리곤: 점을 찍어 자유형 다각형 마스크를 그립니다.");
            _tt.SetToolTip(_btnBrush, "브러시: 브러시로 마스크를 칠합니다.");
            _tt.SetToolTip(_btnEraser, "지우개: 마스크를 지웁니다.");
            _tt.SetToolTip(_btnMask, "Reverse 토글: 현재 그린 영역을 반전 시킵니다.");
            _tt.SetToolTip(_btnAI,
             "AI 도구\n" +
             "• 좌클릭: 프리폼 박스 그려 자동 분할 (Enter=확정, Esc=취소)\n" +
             "• 우클릭: 자동 AI Labeling Mode — 사각형 생성/이동/리사이즈\n" +
             "• Ctrl+D: 현재 ROI로 즉시 분할 후 폴리곤 확정\n" +
             "• Ctrl+S: 저장(툴 유지, 편집 해제)\n" +
             "• Ctrl+E: 현재 폴더 ‘직속’ 이미지 일괄 자동 라벨링(기존 라벨 있으면 스킵)\n" +
             "※ 라벨은 ‘활성 라벨’로 적용, ROI는 다음 이미지에도 같은 비율로 복원");


            int innerWSlot = RIGHT_DOCK_W - _rightToolDock.Padding.Horizontal;
            var slotPointer = WrapToolSlot(_btnPointer, innerWSlot, RIGHT_SLOT_H);
            var slotCircle = WrapToolSlot(_btnCircle, innerWSlot, RIGHT_SLOT_H);
            var slotTriangle = WrapToolSlot(_btnTriangle, innerWSlot, RIGHT_SLOT_H);
            var slotBox = WrapToolSlot(_btnBox, innerWSlot, RIGHT_SLOT_H);
            var slotNgon = WrapToolSlot(_btnNgon, innerWSlot, RIGHT_SLOT_H);
            var slotPolygon = WrapToolSlot(_btnPolygon, innerWSlot, RIGHT_SLOT_H);
            var slotBrush = WrapToolSlot(_btnBrush, innerWSlot, RIGHT_SLOT_H);
            var slotEraser = WrapToolSlot(_btnEraser, innerWSlot, RIGHT_SLOT_H);
            var slotMask = WrapToolSlot(_btnMask, innerWSlot, RIGHT_SLOT_H);
            var slotAI = WrapToolSlot(_btnAI, innerWSlot, RIGHT_SLOT_H);

            _rightTools.Controls.Add(slotPointer);
            _rightTools.Controls.Add(slotCircle);
            _rightTools.Controls.Add(slotTriangle);
            _rightTools.Controls.Add(slotBox);
            _rightTools.Controls.Add(slotNgon);
            _rightTools.Controls.Add(slotPolygon);
            _rightTools.Controls.Add(slotBrush);
            _rightTools.Controls.Add(slotEraser);
            _rightTools.Controls.Add(slotMask);
            _rightTools.Controls.Add(slotAI);


            Action<Control> bindFocus = c =>
            {
                c.TabStop = false;
                c.MouseDown += (s, e) => { if (!_canvas.Focused) _canvas.Focus(); };
                c.Click += (s, e) => { if (!_canvas.Focused) _canvas.Focus(); };
            };
            bindFocus(_btnOpen);
            bindFocus(_btnPointer);
            bindFocus(_btnTriangle);
            bindFocus(_btnBox);
            bindFocus(_btnNgon);
            bindFocus(_btnPolygon);
            bindFocus(_btnCircle);
            bindFocus(_btnBrush);
            bindFocus(_btnEraser);
            bindFocus(_btnMask);
            bindFocus(_btnAI);
            bindFocus(_btnAdd);
            bindFocus(_btnExport);
            bindFocus(_btnTrain);
            bindFocus(_btnInfer);


            _leftRail = new Guna2Panel { Dock = DockStyle.Left, Width = 200, BackColor = Color.Transparent };
            _canvasHost.Controls.Add(_leftRail);

            _leftDock = new Guna2Panel
            {

                Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom,
                Padding = new Padding(6, 8, 6, 8),
                FillColor = Color.Transparent,
                BackColor = Color.Transparent,
                BorderThickness = 2,
                BorderColor = Color.Silver,
                BorderRadius = 2
            };
            var leftContent = new Panel { Dock = DockStyle.Fill, BackColor = Color.White, Padding = new Padding(0), Margin = new Padding(0) };
            _fileTree = new TreeView
            {
                Dock = DockStyle.Fill,
                BorderStyle = BorderStyle.None,
                BackColor = Color.White,
                FullRowSelect = true,
                HideSelection = false,
                ShowLines = true,
                ShowPlusMinus = true,
                HotTracking = true,
                ItemHeight = 20,
                Font = new Font("Segoe UI", 9f)
            };

            _fileTree.AfterSelect += (s, e) =>
            {
                if (e.Node == null) return;
                var path = e.Node.Tag as string;
                if (string.IsNullOrEmpty(path)) return;

                if (System.IO.File.Exists(path) && IsImageFile(path))
                {

                    if (_aiSubMode == AiSubMode.Roi && _canvas != null && _canvas.Image != null)
                    {
                        var ai = _canvas.GetTool(ToolMode.AI) as AITool;
                        if (ai != null) _lastRoiNorm = ai.GetRoiNormalized(_canvas.Transform.ImageSize.ToSize());
                    }
                    LoadImageAtPath(path);


                    if (_aiSubMode == AiSubMode.Roi)
                    {
                        var ai2 = _canvas.GetTool(ToolMode.AI) as AITool;
                        ai2?.EnsureRoiForCurrentImage(_canvas, _lastRoiNorm);
                    }

                    _canvas.Focus();
                }
            };

            _fileTree.StateImageList = LabelStatusService.BuildStateImageList(LabelStatusService.BadgeStyle.Check, 16, 2);
            _fileTree.ShowNodeToolTips = true;

            _leftDock.Controls.Add(leftContent);
            CreateModelHeaderPanel("UNKNOWN");
            CreateRunTypeHeaderPanel("CPU");

            UpdateModelDependentControls();

            this.Shown += async (s, e) => await TryAutoLoadDefaultModelAsync();

            leftContent.Controls.Add(_fileTree);
            _leftRail.Controls.Add(_leftDock);
            _leftRail.BringToFront();


            _canvasLayer = new Panel { Dock = DockStyle.Fill, BackColor = Color.White };
            _canvasLayer.Paint += (s, e) => DrawPlaceholder(e.Graphics);
            _canvasHost.Controls.Add(_canvasLayer);

            _canvas = new ImageCanvas
            {
                TabStop = true,
                BorderStyle = BorderStyle.None,
                Parent = _canvasLayer
            };
            _canvas.SetBrushDiameter(_brushDiameterPx);
            _canvas.ToolEditBegan += () => HideBrushWindow();

            _canvas.MouseDown += (s, e) =>
            {
                if (!_canvas.Focused) _canvas.Focus();
                if ((_canvas.Mode == ToolMode.Brush || _canvas.Mode == ToolMode.Eraser)
                    && e.Button == MouseButtons.Left
                    && _canvas.Image != null)
                {
                    var ip = _canvas.Transform.ScreenToImage(e.Location);
                    var sz = _canvas.Transform.ImageSize;
                    bool insideImage = (ip.X >= 0 && ip.Y >= 0 && ip.X < sz.Width && ip.Y < sz.Height);
                    if (insideImage) HideBrushWindow();
                }
            };

            _canvas.ModeChanged += (mode) =>
            {
                int iconPixel = RIGHT_ICON_PX;
                HighlightTool(_btnPointer, mode == ToolMode.Pointer, iconPixel);
                HighlightTool(_btnBox, mode == ToolMode.Box, iconPixel);
                HighlightTool(_btnPolygon, mode == ToolMode.Polygon, iconPixel);
                HighlightTool(_btnNgon, mode == ToolMode.Ngon, iconPixel);
                HighlightTool(_btnCircle, mode == ToolMode.Circle, iconPixel);
                HighlightTool(_btnTriangle, mode == ToolMode.Triangle, iconPixel);
                HighlightTool(_btnBrush, mode == ToolMode.Brush, iconPixel);
                HighlightTool(_btnEraser, mode == ToolMode.Eraser, iconPixel);
                HighlightTool(_btnMask, mode == ToolMode.Mask, iconPixel);
                HighlightTool(_btnAI, mode == ToolMode.AI, iconPixel);

                if (mode == ToolMode.AI && _aiSubMode == AiSubMode.Roi)
                    ApplyAiButtonRainbowTint(true);
                else
                    ApplyAiButtonRainbowTint(false);

                if (mode == ToolMode.Brush || mode == ToolMode.Eraser)
                {
                    var anchor = (mode == ToolMode.Brush)
                        ? (_btnBrush.Parent != null ? _btnBrush.Parent : (Control)_btnBrush)
                        : (_btnEraser.Parent != null ? _btnEraser.Parent : (Control)_btnEraser);
                    ShowBrushWindowNear(anchor);
                }
                else
                {
                    HideBrushWindow();
                }

                if (!_canvas.Focused) _canvas.Focus();
                this.ActiveControl = _canvas;

                if (mode == ToolMode.Pointer && _canvas.PanMode)
                    DisablePanMode();
            };


            var ctxImage = new ContextMenuStrip { ShowImageMargin = false, Font = new Font("Segoe UI Emoji", 9f) };
            var miPointer = new ToolStripMenuItem("🖱 | Image Pointer");
            var miPan = new ToolStripMenuItem("✋ | Image Pan");
            var miFit = new ToolStripMenuItem("📐 | Image Fit");
            var miColorMap = new ToolStripMenuItem("🎨 | Color Map");
            var miClear = new ToolStripMenuItem("🧹 | Clear Annotations");

            _miAddVertex = new ToolStripMenuItem("➕ | Add Vertex");
            _miAddVertex.Click += (s, e) =>
            {
                var poly = _miAddVertex.Tag as PolygonShape;
                if (poly == null || _canvas == null || _canvas.Image == null) return;

                int idx = poly.InsertVertexAtClosestEdge(_lastCtxImgPointImg);
                if (idx >= 0)
                {
                    if (_canvas.Selection != null)
                        _canvas.Selection.Set(poly);
                    _canvas.Invalidate();
                }
            };
            miPointer.Click += (s, e) =>
            {
                DisablePanMode();
                _canvas.Mode = ToolMode.Pointer;
                if (!_canvas.Focused) _canvas.Focus();
            };
            miPan.Click += (s, e) =>
            {
                if (_canvas == null || _canvas.Image == null) return;
                _canvas.Mode = ToolMode.Pointer;
                if (_canvas.Selection != null) _canvas.Selection.Clear();
                EnablePanMode();
                if (!_canvas.Focused) _canvas.Focus();
            };
            miFit.Click += (s, e) =>
            {
                if (_canvas != null && _canvas.Image != null)
                {
                    _canvas.ZoomToFit();
                    _canvas.Focus();
                }
            };
            miColorMap.Click += (s, e) =>
            {
                OnColorMapClick();
            };
            miClear.Click += (s, e) =>
            {
                if (_canvas == null) return;
                _canvas.ClearAllShapes();
                _canvas.PanMode = false;
                _canvas.Cursor = Cursors.Default;
                _canvas.Focus();
            };

            ctxImage.Items.AddRange(new ToolStripItem[] { miPointer, miPan, miFit, miColorMap, new ToolStripSeparator(), miClear });
            ctxImage.Opening += (s, e) =>
            {
                bool hasImg = (_canvas != null && _canvas.Image != null);
                bool hasShapes = (_canvas != null && _canvas.HasAnyShape);
                miPointer.Enabled = hasImg;
                miPan.Enabled = hasImg;
                miFit.Enabled = hasImg;
                miColorMap.Enabled = hasImg;
                miClear.Enabled = hasShapes;


                var scrPt = _canvas.PointToClient(Control.MousePosition);
                var imgPt = _canvas.Transform.ScreenToImage(scrPt);
                _lastCtxImgPointImg = imgPt;


                if (ctxImage.Items.Contains(_miAddVertex))
                    ctxImage.Items.Remove(_miAddVertex);
                _miAddVertex.Tag = null;


                PolygonShape target = null;
                if (hasImg && hasShapes && _canvas.Shapes != null)
                {
                    for (int i = _canvas.Shapes.Count - 1; i >= 0; --i)
                    {
                        var poly = _canvas.Shapes[i] as PolygonShape;
                        if (poly == null) continue;
                        if (poly.PointsImg == null || poly.PointsImg.Count < 3) continue;
                        if (poly.Contains(imgPt)) { target = poly; break; }
                    }
                }


                if (target != null)
                {
                    _miAddVertex.Tag = target;
                    int idxInsert = ctxImage.Items.IndexOf(miClear);
                    if (idxInsert < 0) idxInsert = ctxImage.Items.Count;
                    ctxImage.Items.Insert(idxInsert, _miAddVertex);
                }
            };
            _canvas.ContextMenuStrip = ctxImage;


            _canvasLayer.Resize += (s, e) =>
            {
                UpdateViewerBounds();
                if (_canvas != null && _canvas.Image != null)
                    _canvas.ZoomToFit();

                UpdateLogLayout();
            };

            CreateHotkeyPanel();
            UpdateViewerBounds();


            _canvasHost.Resize += (s, e) =>
            {
                UpdateSideRailsLayout();
                UpdateLogLayout();
            };
            this.Resize += (s, e) =>
            {
                UpdateSideRailsLayout();
                UpdateLogLayout();
            };
            UpdateSideRailsLayout();

            this.LocationChanged += (s, e) => RepositionBrushWindow();
            this.SizeChanged += (s, e) => RepositionBrushWindow();
            this.VisibleChanged += (s, e) => RepositionBrushWindow();


            HighlightTool(_btnPointer, true, RIGHT_ICON_PX);
            SetHotkeyPanelText(

            "Label | " +
            "Ctrl+S: Labeling 저장, " +
            "Ctrl+C: Labeling 복사, " +
            "Ctrl+V: Labeling 붙여넣기, " +
            "Ctrl+Z: Labeling 취소, " +
            "Ctrl+Click: Labeling 그룹 선택, " +
            "Ctrl+A: Labeling 전체 선택, " +
            "Delete: Labeling 삭제, " +
            "Shift+↑/↓: Labeling 선택 크기 확대/축소(균일 비율), " +
            "←/→/↑/↓: Labeling 선택 이동, " +
            "Ctrl+←/→/↑/↓: Labeling 선택 이동" +
            "\n" +
            "Canvas | " +
            "Ctrl+↑ / Ctrl+↓: (영역 선택 없을 시) 이전 / 다음 이미지," +
            "Enter: AI 프리폼 확정 (AI Tool 전용), " +
            "Esc: AI 프리폼 취소 (AI Tool 전용), " +
            "Ctrl+E: 폴더 일괄 라벨링 (AI ROI Tool 전용), " +
            "Ctrl+D: ROI 즉시 분할 + 폴리곤 확정 (AI ROI Tool전용), " +
            "Ctrl+O: 색상맵 바탕화면 저장, " +
            "Ctrl+U: 색상맵 만 레이블링 적용"
            );

            LoadLastExportZipPath();
            CreateLogPanel();
        }
        #endregion
        #region 5) UI Helpers (유틸/파일/레이아웃 보조)

        private void CreateLogPanel()
        {
            _logPanel = new Guna.UI2.WinForms.Guna2Panel
            {
                BorderRadius = HOTKEY_PANEL_RADIUS,
                FillColor = Color.White,
                BorderColor = HOTKEY_PANEL_BORDER,
                BorderThickness = 2,
                BackColor = Color.Transparent,
                // ↓ 위치/크기는 여기서 설정하지 않음 (동적 배치)
                ShadowDecoration = { Parent = _logPanel }
            };

            _logListBox = new ListBox
            {
                Location = new Point(5, 5),   // 패널 내부 여백
                BorderStyle = BorderStyle.None,
                Font = new Font("Segoe UI", 9f, FontStyle.Regular),
                BackColor = Color.White,
                ForeColor = Color.Black
                // ↓ Size는 UpdateLogLayout에서 패널 크기에 맞춰 조정
            };

            _logPanel.Controls.Add(_logListBox);
            this.Controls.Add(_logPanel);
            _logPanel.BringToFront();

            UpdateLogLayout();
        }

        private void UpdateLogLayout()
        {
            if (_logPanel == null || _canvas == null) return;

            // 1) 기준: 이미지 디스플레이(_canvas)
            int left = _canvas.Left + LOG_X;
            int width = _canvas.Width + LOG_W;
            int top = _canvas.Bottom + LOG_GAP;

            // 2) 하단 기준: 우측/좌측 레일이 있더라도 작업영역 하단과 맞춤
            int bottom = this.ClientSize.Height - OUTER_MARGIN;
            // (상단의 TopBar/TITLE 등의 Dock 영역은 WinForms 레이아웃에서 이미 제외됨)

            int height = Math.Max(60, bottom - top);  // 최소 높이 60 보장
            if (height < 60) height = 60;

            _logPanel.SetBounds(left, top, width, height);

            // 내부 ListBox 크기(패널 내부 여백 5px 반영)
            int lbW = Math.Max(10, width - 10);
            int lbH = Math.Max(10, height - 10);
            _logListBox.Size = new Size(lbW, lbH);

            _logPanel.BringToFront();
        }


        private void AddLog(string message)
        {
            if (_logListBox.InvokeRequired)
            {
                _logListBox.Invoke(new Action<string>(AddLog), message);
                return;
            }

            string timestamp = DateTime.Now.ToString("HH:mm:ss");
            _logListBox.Items.Add($"[{timestamp}] {message}");
            _logListBox.TopIndex = _logListBox.Items.Count - 1;
        }


        private async Task TryAutoLoadDefaultModelAsync()
        {

            var path = DEFAULT_MODEL_PATH;


            if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
            {
                _onnxSession = null;
                SetModelHeader("UNKNOWN");
                SetRuntypeHeader("CPU");
                return;
            }

            try
            {
                using (var overlay = new ProgressOverlay(this, "모델 준비 중…"))
                {
                    var progress = new Progress<(int, string)>(p => overlay.Report(p.Item1, p.Item2));
                    var newSession = await Task.Run(() => SmartLabelingApp.YoloSegOnnx.EnsureSession(path, progress));

                    _onnxSession = newSession;

                    SetModelHeader(System.IO.Path.GetFileName(path));
                    SetRuntypeHeader(_currentRunTypeName);
                }
            }
            catch
            {
                _onnxSession = null;
                SetModelHeader("UNKNOWN");
                SetRuntypeHeader("CPU");
            }
        }

        private void CreateModelHeaderPanel(string initialName)
        {
            if (_modelHeaderPanel != null && !_modelHeaderPanel.IsDisposed) return;

            _modelHeaderPanel = new Guna.UI2.WinForms.Guna2Panel
            {

                Parent = _leftRail,
                Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right,
                Height = MODEL_HEADER_H,
                BorderRadius = 8,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.CornflowerBlue,
                Padding = new Padding(8, 4, 8, 4),
                BackColor = Color.Transparent
            };

            _modelHeaderLabel = new Label
            {
                AutoSize = false,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                ForeColor = Color.Black,
                BackColor = Color.Transparent
            };
            _modelHeaderPanel.Controls.Add(_modelHeaderLabel);

            SetModelHeader(initialName);
            _modelHeaderPanel.BringToFront();
        }

        private void CreateRunTypeHeaderPanel(string initialName)
        {
            if (_runtypeHeaderPanel != null && !_runtypeHeaderPanel.IsDisposed) return;

            _runtypeHeaderPanel = new Guna.UI2.WinForms.Guna2Panel
            {

                Parent = _rightRail,
                Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right,
                Height = RUNTYPE_HEADER_H,
                BorderRadius = 8,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.LightCyan,
                Padding = new Padding(8, 4, 8, 4),
                BackColor = Color.Transparent
            };

            _runtypeHeaderLabel = new Label
            {
                AutoSize = false,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                ForeColor = Color.Black,
                BackColor = Color.Transparent
            };
            _runtypeHeaderPanel.Controls.Add(_runtypeHeaderLabel);

            SetRuntypeHeader(initialName);
            _runtypeHeaderPanel.BringToFront();
        }
        private void SetRuntypeHeader(string modelName)
        {
            _currentRunTypeName = string.IsNullOrWhiteSpace(modelName) ? "CPU" : modelName.Trim();
            if (_runtypeHeaderLabel != null)
                _runtypeHeaderLabel.Text = $"{_currentRunTypeName}";
        }

        private void SetModelHeader(string modelName)
        {
            _currentModelName = string.IsNullOrWhiteSpace(modelName) ? "UNKNOWN" : modelName.Trim();
            if (_modelHeaderLabel != null)
                _modelHeaderLabel.Text = $"DL Model : {_currentModelName}";

            UpdateModelDependentControls();
        }

        private void UpdateModelDependentControls()
        {
            bool hasModel = !string.IsNullOrWhiteSpace(_currentModelName)
                            && !_currentModelName.Equals("UNKNOWN", StringComparison.OrdinalIgnoreCase);


            if (_btnInfer != null)
                _btnInfer.Enabled = hasModel;


            if (_slotToggle != null)
            {
                _slotToggle.Enabled = hasModel;
                _slotToggle.Visible = hasModel;
            }
        }

        private void CreateHotkeyPanel()
        {
            if (_hotkeyPanel != null && !_hotkeyPanel.IsDisposed) return;

            _hotkeyPanel = new Guna.UI2.WinForms.Guna2Panel
            {
                BorderRadius = HOTKEY_PANEL_RADIUS,
                BorderThickness = 2,
                BorderColor = HOTKEY_PANEL_BORDER,
                FillColor = HOTKEY_PANEL_FILL,
                BackColor = Color.Transparent
            };


            _hotkeyPanel.Parent = _canvasLayer;
            _hotkeyPanel.ShadowDecoration.Parent = _hotkeyPanel;

            _hotkeyLabel = new System.Windows.Forms.Label
            {
                AutoSize = false,
                Dock = DockStyle.Fill,
                BackColor = Color.Transparent,
                ForeColor = Color.Black,
                Padding = new Padding(8, 6, 8, 6),
                Font = new Font("Segoe UI", 9f, FontStyle.Regular),
                TextAlign = ContentAlignment.TopLeft,
                UseMnemonic = false,
                Text = "Ctrl+S: Labeling 저장\nCtrl+A: 전체 선택"
            };

            _hotkeyPanel.Controls.Add(_hotkeyLabel);
            _hotkeyPanel.BringToFront();
        }

        private void UpdateHotkeyPanelBounds()
        {
            if (_hotkeyPanel == null || _canvas == null) return;

            _hotkeyPanel.SetBounds(
                _canvas.Left + HOTKEY_PANEL_X,
                _canvas.Top + HOTKEY_PANEL_Y,
                HOTKEY_PANEL_W,
                HOTKEY_PANEL_H
            );

            _hotkeyPanel.BringToFront();
        }

        private void SetHotkeyPanelText(string text)
        {
            if (_hotkeyLabel != null)
                _hotkeyLabel.Text = text ?? string.Empty;
        }

        private void OnToggleClick(object sender, EventArgs e)
        {
            _toggleOn = !_toggleOn;


            _btnToggle.Image = MakeNearWhiteTransparent(_toggleOn ? Properties.Resources.Toggleon2 : Properties.Resources.Toggleoff2, 248);


            if (_slotToggle != null)
            {
                if (_toggleOn)
                {
                    _slotToggle.BorderColor = Color.MediumSeaGreen;
                    _slotToggle.FillColor = Color.FromArgb(235, 248, 239);
                }
                else
                {
                    _slotToggle.BorderColor = Color.Transparent;
                    _slotToggle.FillColor = Color.Transparent;
                }
            }

            if (_toggleOn)
            {
                if (_currentModelName.Contains("PatchCore"))
                {
                    _ = AutoPatchCoreInferIfEnabledAsync();

                }
                else
                {
                    _ = AutoInferIfEnabledAsync();
                }
            }

            if (_canvas != null && !_canvas.Focused) _canvas.Focus();
        }

        private async void OnPrevClick(object sender, EventArgs e)
        {
            _canvas.ClearInferenceOverlays();
            _canvas.Invalidate();

            try { NavigateImage(-1); } catch { }
            if (_currentModelName.Contains("PatchCore"))
                await AutoPatchCoreInferIfEnabledAsync();
            else
                await AutoInferIfEnabledAsync();
            _canvas?.Focus();
        }
        private async void OnNextClick(object sender, EventArgs e)
        {
            _canvas.ClearInferenceOverlays();
            _canvas.Invalidate();

            try { NavigateImage(+1); } catch { }
            if (_currentModelName.Contains("PatchCore"))
                await AutoPatchCoreInferIfEnabledAsync();
            else
                await AutoInferIfEnabledAsync();

            _canvas?.Focus();
        }

        private void NavigateImage(int delta)
        {

            string cur = GetSelectedImagePathFromTree();
            string folder = GetCurrentImageFolder();
            if (string.IsNullOrEmpty(folder)) return;

            var list = EnumerateTopImagesInFolder(folder)
                       .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                       .ToList();
            if (list.Count == 0) return;

            int idx = string.IsNullOrEmpty(cur) ? -1 : list.FindIndex(p => string.Equals(p, cur, StringComparison.OrdinalIgnoreCase));
            if (idx < 0) idx = 0;

            int next = (idx + delta) % list.Count;
            if (next < 0) next += list.Count;

            string path = list[next];
            if (System.IO.File.Exists(path))
            {

                LoadImageAtPath(path);
                SelectTreeNodeByPath(path);
            }
        }


        private void SelectTreeNodeByPath(string path)
        {
            if (_fileTree == null || _fileTree.Nodes.Count == 0 || string.IsNullOrEmpty(path)) return;

            TreeNode found = null;
            var stack = new Stack<TreeNode>();
            foreach (TreeNode n in _fileTree.Nodes) stack.Push(n);
            while (stack.Count > 0)
            {
                var n = stack.Pop();
                string tagPath = n.Tag as string;
                if (string.IsNullOrEmpty(tagPath)) tagPath = n.ToolTipText;
                if (string.IsNullOrEmpty(tagPath)) tagPath = n.Text;

                if (string.Equals(tagPath, path, StringComparison.OrdinalIgnoreCase))
                {
                    found = n; break;
                }
                foreach (TreeNode ch in n.Nodes) stack.Push(ch);
            }
            if (found != null)
            {
                _fileTree.SelectedNode = found;
                found.EnsureVisible();
            }
        }

        private static string GetLastExportZipPathFile()
        {
            var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                    "SmartLabelingApp");
            Directory.CreateDirectory(root);
            return Path.Combine(root, "last_export_zip.txt");
        }
        void LayoutNavRow()
        {
            int w = Math.Max(LABEL_CHIP_MIN_W, _rightToolDock3.ClientSize.Width - _rightToolDock3.Padding.Horizontal);
            _navRow.Width = w;
            int half = (w - 6) / 2;

            _btnPrev.Size = new Size(half, RIGHT_SLOT_H);
            _btnNext.Size = new Size(half, RIGHT_SLOT_H);

            _btnPrev.Location = new Point(0, 0);
            _btnNext.Location = new Point(half + 6, 0);
        }

        private void LoadLastExportZipPath()
        {
            try
            {
                var f = GetLastExportZipPathFile();
                if (File.Exists(f))
                    _lastExportZipPath = File.ReadAllText(f).Trim();
            }
            catch { }
        }

        private void SaveLastExportZipPath()
        {
            try
            {
                File.WriteAllText(GetLastExportZipPathFile(), _lastExportZipPath ?? "");
            }
            catch { }
        }

        private void TryBindAnnotationRootNear(string folder)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder)) return;
                var annotationRoot = Path.Combine(folder, "AnnotationData");
                var modelRoot = Path.Combine(annotationRoot, "Model");
                if (Directory.Exists(modelRoot) && File.Exists(Path.Combine(modelRoot, "classes.txt")))
                {
                    _lastYoloExportRoot = modelRoot;
                    LabelStatusService.SetStorageRoot(annotationRoot);
                }
            }
            catch { }
        }

        private void EnsureBrushWindow()
        {
            if (_brushWin == null || _brushWin.IsDisposed)
            {
                _brushWin = new BrushSizeWindow
                {
                    StartPosition = FormStartPosition.Manual,
                    ShowInTaskbar = false,
                    TopMost = true,
                    MinimumPx = 2,
                    MaximumPx = 256
                };
                _brushWin.BrushSizeChanged += OnBrushSizeChanged;
                _brushWin.FormClosed += (s, e) => { _brushWin = null; };
            }
            _brushWin.ValuePx = _brushDiameterPx;
        }
        private void HideBrushWindow()
        {
            if (_brushWin != null && !_brushWin.IsDisposed && _brushWin.Visible)
                _brushWin.Hide();
        }
        #region 6) Event Handlers (버튼/메뉴/키/마우스)
        #endregion
        private void OnBrushSizeChanged(int px)
        {
            _brushDiameterPx = px;
            if (_canvas != null) _canvas.SetBrushDiameter(_brushDiameterPx);
            if (_canvas != null && !_canvas.Focused) _canvas.Focus();
        }
        private void ShowBrushWindowNear(Control anchor)
        {
            if (anchor == null || anchor.IsDisposed) return;
            EnsureBrushWindow();
            _brushAnchorBtn = anchor;

            Control refCtrl = anchor;
            if (anchor.Parent is Guna2Panel) refCtrl = anchor.Parent;

            Point pScreen = refCtrl.PointToScreen(new Point(0, 0));
            Rectangle wa = Screen.FromControl(this).WorkingArea;

            int x = pScreen.X - _brushWin.Width - 12;
            int y = pScreen.Y + (refCtrl.Height / 2) - (_brushWin.Height / 2);

            if (x < wa.Left) x = wa.Left + 8;
            if (y < wa.Top) y = wa.Top + 8;
            if (y + _brushWin.Height > wa.Bottom) y = wa.Bottom - _brushWin.Height - 8;

            _brushWin.Location = new Point(x, y);

            if (!_brushWin.Visible) _brushWin.Show(this);
            else _brushWin.BringToFront();
        }
        private void RepositionBrushWindow()
        {
            if (_brushWin != null && _brushWin.Visible && _brushAnchorBtn != null && !_brushAnchorBtn.IsDisposed)
                ShowBrushWindowNear(_brushAnchorBtn);
        }


        private static Bitmap MakeNearWhiteTransparent(Image img, byte threshold = 248)
        {
            var src = new Bitmap(img);
            var bmp = new Bitmap(src.Width, src.Height, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(bmp)) g.DrawImage(src, Point.Empty);

            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    var c = bmp.GetPixel(x, y);
                    if (c.R >= threshold && c.G >= threshold && c.B >= threshold)
                        bmp.SetPixel(x, y, Color.FromArgb(0, c));
                }
            }
            src.Dispose();
            return bmp;
        }
        private Guna2ImageButton CreateToolIcon(Image img, string tooltipText, int edge, int iconPx)
        {
            var btn = new Guna2ImageButton();
            var clean = (img == null) ? new Bitmap(1, 1, PixelFormat.Format32bppArgb) : MakeNearWhiteTransparent(img, 248);

            btn.Image = clean;
            btn.ImageSize = new Size(iconPx, iconPx);
            btn.HoverState.ImageSize = new Size(iconPx + 2, iconPx + 2);
            btn.PressedState.ImageSize = new Size(iconPx, iconPx);
            btn.UseTransparentBackground = true;
            btn.BackColor = Color.Transparent;
            btn.Size = new Size(edge, edge);
            btn.Margin = new Padding(0);
            btn.Cursor = Cursors.Hand;

            var tip = new Guna2HtmlToolTip
            {
                TitleForeColor = Color.Black,
                ForeColor = Color.Black,
                BackColor = Color.White
            };
            tip.SetToolTip(btn, tooltipText);
            return btn;
        }
        private void SetTool(ToolMode mode, Guna2ImageButton clicked)
        {
            _canvas.Mode = mode;
            if (!_canvas.Focused) _canvas.Focus();
            this.ActiveControl = _canvas;

            int iconPx = RIGHT_ICON_PX;
            HighlightTool(_btnPointer, clicked == _btnPointer, iconPx);
            HighlightTool(_btnBox, clicked == _btnBox, iconPx);
            HighlightTool(_btnPolygon, clicked == _btnPolygon, iconPx);
            HighlightTool(_btnNgon, clicked == _btnNgon, iconPx);
            HighlightTool(_btnCircle, clicked == _btnCircle, iconPx);
            HighlightTool(_btnTriangle, clicked == _btnTriangle, iconPx);
            HighlightTool(_btnBrush, clicked == _btnBrush, iconPx);
            HighlightTool(_btnEraser, clicked == _btnEraser, iconPx);
            HighlightTool(_btnMask, clicked == _btnMask, iconPx);
            HighlightTool(_btnAI, clicked == _btnAI, iconPx);

            if (clicked != _btnBrush && clicked != _btnEraser)
            {
                if (_brushWin != null && _brushWin.Visible) _brushWin.Hide();
            }
            if (mode == ToolMode.Pointer && _canvas.PanMode)
                DisablePanMode();
        }
        private void HighlightTool(Guna2ImageButton btn, bool active, int baseIconPx)
        {
            if (active)
            {
                btn.ImageSize = new Size(baseIconPx + 4, baseIconPx + 4);
                btn.HoverState.ImageSize = new Size(baseIconPx + 6, baseIconPx + 6);
            }
            else
            {
                btn.ImageSize = new Size(baseIconPx, baseIconPx);
                btn.HoverState.ImageSize = new Size(baseIconPx + 2, baseIconPx + 2);
            }

            var slot = btn.Parent as Guna2Panel;
            if (slot != null)
            {
                slot.BorderColor = active ? Color.LimeGreen : Color.Transparent;
                slot.FillColor = active ? Color.FromArgb(245, 245, 245) : Color.Transparent;
            }
        }


        private static Bitmap MakeRainbowBitmap(Size size)
        {
            if (size.Width < 2 || size.Height < 2)
                size = new Size(Math.Max(2, size.Width), Math.Max(2, size.Height));

            var bmp = new Bitmap(size.Width, size.Height);
            using (var g = Graphics.FromImage(bmp))
            using (var br = new System.Drawing.Drawing2D.LinearGradientBrush(
                new Rectangle(Point.Empty, size),
                Color.Red, Color.Violet, 0f))
            {
                var cb = new System.Drawing.Drawing2D.ColorBlend
                {
                    Positions = new[] { 0f, 0.2f, 0.4f, 0.6f, 0.8f, 1f },
                    Colors = new[]
                    {
                Color.FromArgb(255, 255,   0,   0),
                Color.FromArgb(255, 255, 165,   0),
                Color.FromArgb(255, 255, 255,   0),
                Color.FromArgb(255,   0, 128,   0),
                Color.FromArgb(255,   0,   0, 255),
                Color.FromArgb(255, 128,   0, 128),
            }
                };
                br.InterpolationColors = cb;
                g.FillRectangle(br, 0, 0, size.Width, size.Height);
            }
            return bmp;
        }


        private void ApplyAiButtonRainbowTint(bool enable)
        {
            var slot = _btnAI?.Parent as Guna.UI2.WinForms.Guna2Panel;
            if (slot == null) return;

            if (enable)
            {

                if (_aiRainbowBg != null) { _aiRainbowBg.Dispose(); _aiRainbowBg = null; }
                _aiRainbowBg = MakeRainbowBitmap(slot.ClientSize);


                slot.UseTransparentBackground = true;
                slot.FillColor = Color.Transparent;





                slot.BackgroundImage = _aiRainbowBg;
                slot.BackgroundImageLayout = ImageLayout.Stretch;


                _btnAI.BackColor = Color.Transparent;
            }
            else
            {

                slot.BackgroundImage = null;

                slot.UseTransparentBackground = false;

                if (_aiRainbowBg != null) { _aiRainbowBg.Dispose(); _aiRainbowBg = null; }


                _btnAI.BackColor = Color.Transparent;

            }
        }


        private void EnterAiFreeformMode()
        {

            SetTool(ToolMode.AI, _btnAI);
            _aiSubMode = AiSubMode.Free;


            HighlightTool(_btnAI, true, RIGHT_ICON_PX);
            ApplyAiButtonRainbowTint(false);


            var ai = _canvas.GetTool(ToolMode.AI) as AITool;
            ai?.DisableRoiMode(_canvas);

            if (_brushWin != null && _brushWin.Visible) _brushWin.Hide();
            if (!_canvas.Focused) _canvas.Focus();
        }

        private void EnterAiRoiMode()
        {

            SetTool(ToolMode.AI, _btnAI);
            _aiSubMode = AiSubMode.Roi;


            HighlightTool(_btnAI, true, RIGHT_ICON_PX);
            ApplyAiButtonRainbowTint(true);


            var slot = _btnAI.Parent as Guna2Panel;
            if (slot != null)
            {
                slot.BorderColor = Color.Transparent;
                slot.FillColor = Color.FromArgb(240, 242, 255);
            }


            var ai = _canvas.GetTool(ToolMode.AI) as AITool;
            ai?.EnableRoiMode(_canvas, _lastRoiNorm);

            if (_brushWin != null && _brushWin.Visible) _brushWin.Hide();
            if (!_canvas.Focused) _canvas.Focus();
        }

        private Guna2Panel WrapToolSlot(Guna2ImageButton btn, int width, int height)
        {
            var slot = new Guna2Panel
            {
                Size = new Size(width, height),
                Padding = new Padding(RIGHT_ICON_PAD),
                BorderRadius = 8,
                BorderThickness = 2,
                BorderColor = Color.Transparent,
                FillColor = Color.Transparent,
                BackColor = Color.Transparent,
                Margin = new Padding(0, 0, 0, RIGHT_ICON_GAP)
            };
            btn.Dock = DockStyle.Fill;
            slot.Controls.Add(btn);
            return slot;
        }
        private void DehighlightAllTools()
        {
            int px = RIGHT_ICON_PX;
            HighlightTool(_btnPointer, false, px);
            HighlightTool(_btnBox, false, px);
            HighlightTool(_btnPolygon, false, px);
            HighlightTool(_btnNgon, false, px);
            HighlightTool(_btnCircle, false, px);
            HighlightTool(_btnTriangle, false, px);
            HighlightTool(_btnBrush, false, px);
            HighlightTool(_btnEraser, false, px);
            HighlightTool(_btnMask, false, px);
            HighlightTool(_btnAI, false, px);
        }
        private void EnablePanMode()
        {
            if (_canvas == null || _canvas.Image == null) return;
            _canvas.PanMode = true;
            DehighlightAllTools();
            if (_canvas.Selection != null) _canvas.Selection.Clear();
            _canvas.Cursor = Cursors.Hand;
            if (!_canvas.Focused) _canvas.Focus();
        }
        private void DisablePanMode()
        {
            if (_canvas == null) return;
            _canvas.PanMode = false;
            _canvas.Cursor = Cursors.Default;
            if (_btnPointer != null) SetTool(ToolMode.Pointer, _btnPointer);
        }


        private Rectangle GetFrameRect()
        {
            if (_canvasLayer == null)
                return new Rectangle(FRAME_X, FRAME_Y, Math.Min(FRAME_W, VIEWER_MAX_W), FRAME_H);

            int rawW = _canvasLayer.ClientSize.Width - FRAME_X - FRAME_X_OFFSET;
            int rawH = _canvasLayer.ClientSize.Height - FRAME_Y - FRAME_Y_OFFSET;

            int w = Math.Max(2 * FRAME_BORDER + 2, Math.Min(rawW, VIEWER_MAX_W));
            int h = Math.Max(2 * FRAME_BORDER + 2, rawH);

            return new Rectangle(FRAME_X, FRAME_Y, w, h);
        }
        private void DrawPlaceholder(Graphics g)
        {
            using (var pen = new Pen(Color.Silver, FRAME_BORDER))
            using (var br = new SolidBrush(Color.FromArgb(20, Color.Gray)))
            {
                var r = GetFrameRect();
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.FillRectangle(br, r);
                g.DrawRectangle(pen, r);
            }
        }
        private void UpdateViewerBounds()
        {
            if (_canvasLayer == null || _canvas == null) return;
            var r = GetFrameRect();
            var inner = Rectangle.Inflate(r, -FRAME_BORDER, -FRAME_BORDER);
            _canvas.Bounds = inner;

            UpdateHotkeyPanelBounds();
        }

        private void UpdateSideRailsLayout()
        {
            if (_canvasLayer == null) return;

            int topPad = Math.Max(0, FRAME_Y);


            if (_rightRail != null)
            {
                _rightRail.Padding = new Padding(0, topPad - RIGHT_DOCK_T, 0, 2);
                _rightRail.Width = RIGHT_DOCK_W;

                int clientW = _rightRail.ClientSize.Width - _rightRail.Padding.Horizontal;
                int leftX = _rightRail.Padding.Left;
                int topY = _rightRail.Padding.Top;


                if (_rightToolDock != null)
                {
                    _rightToolDock.Left = leftX;
                    _rightToolDock.Top = topY;
                    _rightToolDock.Width = clientW;
                    _rightToolDock.Height = RIGHT_BAR1_H;
                }


                if (_rightToolDock2 != null)
                {
                    _rightToolDock2.Left = leftX;
                    _rightToolDock2.Width = clientW;
                    _rightToolDock2.Height = Math.Max(RIGHT_BAR_MIN_H, RIGHT_BAR2_H);
                    _rightToolDock2.Top = _rightToolDock.Bottom + RIGHT_BAR_GAP;
                }


                if (_rightToolDock3 != null)
                {
                    _rightToolDock3.Left = leftX;
                    _rightToolDock3.Width = clientW;
                    _rightToolDock3.Top = _rightToolDock2.Bottom + RIGHT_BAR_GAP;

                    int h3 = Math.Max(RIGHT_BAR_MIN_H, RIGHT_BAR3_H);

                    if (RIGHT_BAR3_SNAP_TO_VIEWER)
                    {
                        int viewerBottom = _rightRail.Padding.Top + this.ClientSize.Height - FRAME_BORDER + RIGHT_BAR3_TAIL;
                        int desired = viewerBottom - _rightToolDock3.Top;


                        if (desired < RIGHT_BAR_MIN_H)
                        {
                            int need = RIGHT_BAR_MIN_H - desired;
                            if (_rightToolDock2 != null && _rightToolDock2.Height > RIGHT_BAR_MIN_H)
                            {
                                int canReduce = _rightToolDock2.Height - RIGHT_BAR_MIN_H;
                                int reduce = Math.Min(canReduce, need);
                                _rightToolDock2.Height -= reduce;
                                _rightToolDock3.Top = _rightToolDock2.Bottom + RIGHT_BAR_GAP;
                                desired = viewerBottom - _rightToolDock3.Top;
                            }
                            desired = Math.Max(RIGHT_BAR_MIN_H, desired);
                        }


                        int maxAllow = Math.Max(RIGHT_BAR_MIN_H, viewerBottom - _rightToolDock3.Top);
                        h3 = Math.Min(Math.Max(RIGHT_BAR_MIN_H, desired), maxAllow);
                    }

                    _rightToolDock3.Height = h3;

                    int bottomSpace = _rightRail.ClientSize.Height - _rightToolDock3.Top;
                    _rightToolDock3.Height = bottomSpace;
                }


                if (_runtypeHeaderPanel != null)
                {
                    int x = _rightRail.Padding.Left;
                    int y = _rightRail.Padding.Top + RUNTYPE_HEADER_Y;
                    int w = Math.Max(20, _rightRail.ClientSize.Width - _rightRail.Padding.Horizontal);
                    int h = RUNTYPE_HEADER_H;

                    _runtypeHeaderPanel.SetBounds(x, y, w, h);
                    _runtypeHeaderPanel.BringToFront();
                }


                AdjustLabelChipWidths();
            }

            if (_leftRail != null)
            {
                _leftRail.Padding = new Padding(0, topPad - RIGHT_DOCK_T, 0, 2);


                if (_modelHeaderPanel != null)
                {
                    int x = _leftRail.Padding.Left;
                    int y = _leftRail.Padding.Top + MODEL_HEADER_Y;
                    int w = Math.Max(20, _leftRail.ClientSize.Width - _leftRail.Padding.Horizontal);
                    int h = MODEL_HEADER_H;

                    _modelHeaderPanel.SetBounds(x, y, w, h);
                    _modelHeaderPanel.BringToFront();
                }


                if (_leftDock != null)
                {
                    int dockLeft = 0;
                    int dockTop = (_modelHeaderPanel?.Bottom ?? _leftRail.Padding.Top) + MODEL_HEADER_GAP;
                    int dockWidth = _leftRail.ClientSize.Width;
                    int dockHeight = _leftRail.ClientSize.Height - dockTop - _leftRail.Padding.Bottom;

                    if (dockHeight < 1) dockHeight = 1;
                    _leftDock.SetBounds(dockLeft, dockTop, dockWidth, dockHeight);
                }
            }

            _canvasLayer.Invalidate();
        }

        private void ToggleMaximizeRestore()
        {
            WindowState = (WindowState == FormWindowState.Maximized)
                ? FormWindowState.Normal
                : FormWindowState.Maximized;
        }


        private void OnAddClick(object sender, EventArgs e)
        {
            EnsureLabelWindow();

            _labelAnchorBtn = (_btnAdd.Parent != null) ? (Control)_btnAdd.Parent : (Control)_btnAdd;
            Point pScreen = _labelAnchorBtn.PointToScreen(new Point(0, 0));
            Rectangle wa = Screen.FromControl(this).WorkingArea;

            int x = pScreen.X - _labelWin.Width - 12;
            int y = pScreen.Y + (_labelAnchorBtn.Height / 2) - (_labelWin.Height / 2);

            if (x < wa.Left) x = wa.Left + 8;
            if (y < wa.Top) y = wa.Top + 8;
            if (y + _labelWin.Height > wa.Bottom) y = wa.Bottom - _labelWin.Height - 8;

            _labelWin.StartPosition = FormStartPosition.Manual;
            _labelWin.Location = new Point(x, y);

            var result = _labelWin.ShowDialog(this);
            if (result == DialogResult.OK)
            {
                string name = _labelWin.LabelName;
                if (string.IsNullOrWhiteSpace(name))
                {
                    name = "Label" + _labelSeq.ToString();
                    _labelSeq++;
                }
                Color col = _labelWin.SelectedColor;
                AddLabelChip(name, col);
            }

            if (_canvas != null && !_canvas.Focused) _canvas.Focus();
        }

        private void OnSaveClick(object sender, EventArgs e)
        {
            try
            {
                string savePath = !string.IsNullOrEmpty(_savePathOverride) ? _savePathOverride : _currentImagePath;
                _savePathOverride = null; // 한번 쓰고 즉시 해제

                if (_canvas == null || _canvas.Image == null)
                {
                    MessageBox.Show(this, "이미지가 없습니다.", "SAVE", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                SaveDatasetYoloWithImages(savePath);
                bool keepAiRoi = (_aiSubMode == AiSubMode.Roi) && _canvas != null && _canvas.Mode == ToolMode.AI;


                if (keepAiRoi)
                {

                    if (_canvas.Selection != null) _canvas.Selection.Clear();
                    _canvas.ClearSelectionButKeepMode();
                }
                else
                {

                    _canvas.ClearSelectionAndResetEditing();
                }


                _canvas?.Focus();
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "저장 중 오류: " + ex.Message, "SAVE", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (_canvas != null && !_canvas.Focused) _canvas.Focus();
            }
        }

        private static string MakeLabelSavePathFromSource(string srcPath)
        {
            string dir = Path.Combine(Path.GetDirectoryName(srcPath), "_labels");
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            string stem = Path.GetFileNameWithoutExtension(srcPath);
            return Path.Combine(dir, stem + "_label.png");
        }

        private string PickAnnotationDataFolderWithCommonDialog(string initialDir = null)
        {
            using (var dlg = new FolderBrowserDialog())
            {
                dlg.Description = "AnnotationData 폴더를 선택하세요";
                if (!string.IsNullOrEmpty(initialDir) && Directory.Exists(initialDir))
                    dlg.SelectedPath = initialDir;

                if (dlg.ShowDialog(this) != DialogResult.OK) return null;

                var di = new DirectoryInfo(dlg.SelectedPath);
                if (!di.Name.Equals("AnnotationData", StringComparison.OrdinalIgnoreCase))
                {
                    MessageBox.Show(this, "선택한 폴더의 이름이 'AnnotationData'가 아닙니다.",
                                    "EXPORT", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return null;
                }
                return di.FullName;
            }
        }

        private string GetSelectedImagePathFromTree()
        {
            if (_fileTree?.SelectedNode == null) return null;
            var n = _fileTree.SelectedNode;

            string path = n.Tag as string;
            if (string.IsNullOrEmpty(path)) path = n.ToolTipText;
            if (string.IsNullOrEmpty(path)) path = n.Text;
            return path;
        }


        private string GetDatasetRootFolderOfSelected()
        {
            if (_fileTree?.SelectedNode == null) return null;
            TreeNode root = _fileTree.SelectedNode;
            while (root.Parent != null) root = root.Parent;

            string rootPath = root.Tag as string;
            if (string.IsNullOrEmpty(rootPath)) rootPath = root.ToolTipText;
            if (string.IsNullOrEmpty(rootPath)) rootPath = root.Text;

            if (!string.IsNullOrEmpty(rootPath) && System.IO.File.Exists(rootPath))
                rootPath = System.IO.Path.GetDirectoryName(rootPath);
            return rootPath;
        }


        private string GetCurrentImageFolder()
        {
            var sel = GetSelectedImagePathFromTree();
            return string.IsNullOrEmpty(sel) ? null : System.IO.Path.GetDirectoryName(sel);
        }


        private IEnumerable<string> EnumerateTopImagesInFolder(string folder)
        {
            if (string.IsNullOrEmpty(folder) || !System.IO.Directory.Exists(folder))
                yield break;

            foreach (var p in System.IO.Directory.EnumerateFiles(folder, "*.*", System.IO.SearchOption.TopDirectoryOnly))
            {
                if (IsImageFile(p)) yield return p;
            }
        }








        private void CollectImagePathsRecursive(TreeNode node, List<string> acc)
        {
            if (node == null) return;


            string path = node.Tag as string;
            if (string.IsNullOrEmpty(path)) path = node.ToolTipText;
            if (string.IsNullOrEmpty(path)) path = node.Text;

            if (!string.IsNullOrEmpty(path) && System.IO.File.Exists(path) && IsImageFile(path))
                acc.Add(path);

            foreach (TreeNode ch in node.Nodes)
                CollectImagePathsRecursive(ch, acc);
        }


        private async void StartAutoLabelAllImagesAsync()
        {
            try
            {
                RectangleF? roiSeed = null;
                {
                    var aiSeed = _canvas.GetTool(ToolMode.AI) as AITool;
                    if (aiSeed != null)
                        roiSeed = aiSeed.GetRoiNormalized(_canvas.Transform.ImageSize.ToSize());
                }

                var baseDir = GetCurrentImageFolder();
                var paths = EnumerateTopImagesInFolder(baseDir).OrderBy(p => p, StringComparer.OrdinalIgnoreCase).ToList();

                if (paths.Count == 0)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Auto Label",
                        Text = "라벨링할 이미지가 없습니다.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return;
                }

                int total = paths.Count;
                int labeled = 0, skipped = 0, failed = 0;

                using (var overlay = new ProgressOverlay(this, "Auto Labeling...", true))
                {
                    for (int i = 0; i < total; i++)
                    {
                        string path = paths[i];

                        overlay.Report((i * 100) / total, $"{i + 1}/{total} - {System.IO.Path.GetFileName(path)}");

                        try
                        {

                            LoadImageAtPath(path);


                            var ai = _canvas.GetTool(ToolMode.AI) as AITool;
                            ai?.EnsureRoiForCurrentImage(_canvas, roiSeed);


                            if (_canvas != null && _canvas.Shapes != null && _canvas.Shapes.Count > 0)
                            {
                                skipped++;
                                continue;
                            }


                            if (ai == null)
                            {
                                failed++;
                                continue;
                            }
                            bool ok = await ai.AutoLabelCurrentRoiAndCommitAsync(_canvas);
                            if (!ok)
                            {
                                failed++;
                                continue;
                            }


                            OnSaveClick(_btnSave, null);

                            labeled++;
                        }
                        catch
                        {
                            failed++;

                        }
                    }

                    overlay.Report(100, "완료");
                }


                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "Auto Label 완료",
                    Text = $"총 {total}개\n라벨링: {labeled}\n스킵(기존 도형 있음): {skipped}\n실패: {failed}",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Information,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
            }
            catch (Exception ex)
            {
                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "Auto Label",
                    Text = "오류: " + ex.Message,
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
            }
        }

        string pickedPath = "";
        private async void OnExportClick(object sender, EventArgs e)
        {
            try
            {
                pickedPath = PickAnnotationDataFolderWithCommonDialog(null);
                if (string.IsNullOrEmpty(pickedPath)) return;

                var modelDir = Path.Combine(pickedPath, "Model");
                var imagesDir = Path.Combine(modelDir, "images");
                var labelsDir = Path.Combine(modelDir, "labels");
                var classesPath = Path.Combine(modelDir, "classes.txt");
                var notesPath = Path.Combine(modelDir, "notes.json");

                if (!Directory.Exists(modelDir) || !Directory.Exists(imagesDir) || !Directory.Exists(labelsDir)
                    || !File.Exists(classesPath) || !File.Exists(notesPath))
                {
                    MessageBox.Show(this, "AnnotationData\\Model 폴더에 images, labels, classes.txt, notes.json이 있는지 확인하세요.",
                                    "EXPORT", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                using (var splitDlg = new ExportSplitDialog(90, 5, 5))
                {
                    if (splitDlg.ShowDialog(this) != DialogResult.OK) return;
                    int pTrain = splitDlg.TrainPercent;
                    int pVal = splitDlg.ValPercent;
                    int pTest = splitDlg.TestPercent;

                    if (pTrain + pVal + pTest != 100)
                    {
                        MessageBox.Show(this, "세 비율의 합이 100%가 아닙니다.", "EXPORT", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }

                    var resultRoot = Path.Combine(pickedPath, "Result");

                    int total = 0, nTrain = 0, nVal = 0, nTest = 0;
                    string zipPath = null;

                    using (var overlay = new ProgressOverlay(this, "Model Exporting..."))
                    {
                        overlay.Report(0, "Preparing...");
                        await Task.Run(() =>
                        {
                            overlay.Report(5, "Exporting dataset...");
                            DoYoloSegExport(modelDir, resultRoot, pTrain, pVal, pTest, out total, out nTrain, out nVal, out nTest);
                            overlay.Report(40, "Packaging to ZIP...");
                            zipPath = CreateResultZip(resultRoot, null, (p, msg) => overlay.Report(40 + (int)((p * 60L) / 100), msg)
                            );
                        });

                        overlay.Report(100, "완료");
                    }

                    using (var dlg = new ExportResultDialog(total, nTrain, nVal, nTest, resultRoot, zipPath))
                    {
                        dlg.ShowDialog(this);
                    }

                    if (!string.IsNullOrEmpty(zipPath))
                    {
                        _lastExportZipPath = zipPath;
                        SaveLastExportZipPath();
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "내보내기 중 오류: " + ex.Message, "EXPORT",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (this.ActiveControl != null && !(this.ActiveControl is Form)) this.ActiveControl.Focus();
            }
        }

        private string CreateResultZip(string resultRoot, string zipFileName = null, Action<int, string> onProgress = null)
        {
            if (string.IsNullOrWhiteSpace(resultRoot) || !Directory.Exists(resultRoot))
                throw new DirectoryNotFoundException("Result 폴더가 없습니다: " + resultRoot);

            var dirInfo = new DirectoryInfo(resultRoot);
            var finalZip = Path.Combine(resultRoot, zipFileName ?? (dirInfo.Name + ".zip"));


            var tempZip = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".zip");


            Func<string, string, string> relPath = (root, path) =>
            {
                var r = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                var p = Path.GetFullPath(path);
                if (!p.StartsWith(r, StringComparison.OrdinalIgnoreCase)) return Path.GetFileName(path);
                var rel = p.Substring(r.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                return rel.Replace('\\', '/');
            };


            var allFiles = Directory
                .GetFiles(resultRoot, "*", SearchOption.AllDirectories)
                .Where(f => !string.Equals(f, finalZip, StringComparison.OrdinalIgnoreCase))
                .ToList();

            int total = allFiles.Count;
            int done = 0;

            using (var zip = ZipFile.Open(tempZip, ZipArchiveMode.Create))
            {
                foreach (var file in allFiles)
                {
                    var entryName = relPath(resultRoot, file);
                    zip.CreateEntryFromFile(file, entryName, CompressionLevel.Optimal);

                    done++;
                    if (onProgress != null)
                    {
                        int percent = total == 0 ? 100 : (int)((done * 100L) / total);
                        onProgress(percent, Path.GetFileName(file));
                    }
                }
            }


            if (File.Exists(finalZip)) File.Delete(finalZip);
            File.Move(tempZip, finalZip);

            return finalZip;
        }




        private void DoYoloSegExport(string modelDir, string resultRoot, int pctTrain, int pctVal, int pctTest,
                                     out int total, out int nTrain, out int nVal, out int nTest)
        {
            var imagesDir = Path.Combine(modelDir, "images");
            var labelsDir = Path.Combine(modelDir, "labels");
            var classesPath = Path.Combine(modelDir, "classes.txt");
            var notesPath = Path.Combine(modelDir, "notes.json");

            if (!Directory.Exists(imagesDir) || !Directory.Exists(labelsDir) || !File.Exists(classesPath))
                throw new InvalidOperationException("Model 폴더 구조가 올바르지 않습니다.");


            var classNames = File.ReadAllLines(classesPath, Encoding.UTF8)
                                 .Select(s => (s ?? "").Trim())
                                 .Where(s => s.Length > 0)
                                 .ToList();
            if (classNames.Count == 0) classNames.Add("Default");


            var images = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var labels = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            foreach (var imgPath in Directory.EnumerateFiles(imagesDir, "*.*", SearchOption.AllDirectories))
            {
                var ext = Path.GetExtension(imgPath);
                if (string.IsNullOrEmpty(ext) || !_imgExts.Contains(ext)) continue;
                images[Path.GetFileNameWithoutExtension(imgPath).ToLowerInvariant()] = imgPath;
            }
            foreach (var labPath in Directory.EnumerateFiles(labelsDir, "*.txt", SearchOption.AllDirectories))
            {
                var name = Path.GetFileName(labPath);
                if (name.Equals("classes.txt", StringComparison.OrdinalIgnoreCase)) continue;
                labels[Path.GetFileNameWithoutExtension(labPath).ToLowerInvariant()] = labPath;
            }

            var pairs = new List<Tuple<string, string>>();
            foreach (var kv in images)
            {
                string lab;
                if (labels.TryGetValue(kv.Key, out lab))
                    pairs.Add(Tuple.Create(kv.Value, lab));
            }
            if (pairs.Count == 0) throw new InvalidOperationException("이미지-라벨 쌍을 찾지 못했습니다.");


            Shuffle(pairs, 0);
            total = pairs.Count;
            nVal = (int)Math.Round(total * (pctVal / 100.0));
            nTest = (int)Math.Round(total * (pctTest / 100.0));
            nTrain = total - nVal - nTest;
            if (nTrain < 0) nTrain = 0;

            var trainSet = pairs.Take(nTrain).ToList();
            var valSet = pairs.Skip(nTrain).Take(nVal).ToList();
            var testSet = pairs.Skip(nTrain + nVal).Take(nTest).ToList();


            var subDirs = new[]
            {
                Path.Combine(resultRoot, "images", "train"),
                Path.Combine(resultRoot, "images", "val"),
                Path.Combine(resultRoot, "images", "test"),
                Path.Combine(resultRoot, "labels", "train"),
                Path.Combine(resultRoot, "labels", "val"),
                Path.Combine(resultRoot, "labels", "test"),
            };
            foreach (var d in subDirs) Directory.CreateDirectory(d);


            CopyPairs(trainSet, Path.Combine(resultRoot, "images", "train"), Path.Combine(resultRoot, "labels", "train"));
            CopyPairs(valSet, Path.Combine(resultRoot, "images", "val"), Path.Combine(resultRoot, "labels", "val"));
            CopyPairs(testSet, Path.Combine(resultRoot, "images", "test"), Path.Combine(resultRoot, "labels", "test"));


            var sb = new StringBuilder();
            sb.AppendLine("path: " + QuoteYamlPath(resultRoot));
            sb.AppendLine("train: images/train");
            sb.AppendLine("val: images/val");
            sb.AppendLine("test: images/test");
            sb.AppendLine("names:");
            for (int i = 0; i < classNames.Count; i++)
                sb.AppendLine($"  {i}: {EscapeYaml(classNames[i])}");
            sb.AppendLine("task: seg");
            File.WriteAllText(Path.Combine(resultRoot, "data.yaml"), sb.ToString(), Encoding.UTF8);
        }

        private static string QuoteYamlPath(string p)
        {
            if (string.IsNullOrEmpty(p)) return "''";

            if (p.IndexOfAny(new[] { ' ', ':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '<', '>', '=', '!', '%', '@', '\\' }) >= 0)
                return "'" + p.Replace("'", "''") + "'";
            return p;
        }

        private static string EscapeYaml(string s)
        {
            if (s == null) return "''";

            if (s.IndexOfAny(new[] { ':', '#', '-', '?', '{', '}', ',', '&', '*', '!', '|', '>', '\'', '\"', '%', '@', '`' }) >= 0 || s.Contains(" "))
                return "'" + s.Replace("'", "''") + "'";
            return s;
        }

        private static void CopyPairs(List<Tuple<string, string>> pairs, string imgDstDir, string labDstDir)
        {
            foreach (var t in pairs)
            {
                var img = t.Item1; var lab = t.Item2;
                var imgName = Path.GetFileName(img);
                var labName = Path.GetFileName(lab);
                File.Copy(img, Path.Combine(imgDstDir, imgName), true);
                File.Copy(lab, Path.Combine(labDstDir, labName), true);
            }
        }

        private static void Shuffle<T>(IList<T> list, int seed)
        {
            var rnd = new Random(seed);
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rnd.Next(i + 1);
                var tmp = list[i];
                list[i] = list[j];
                list[j] = tmp;
            }
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == (Keys.Control | Keys.U))
            {
                OnAutoLabelFromColormap_LessThanThreshold(500);

                return true;
            }
            if (keyData == (Keys.Control | Keys.O))
            {
                RunColorMapBatchToDesktop();
                return true;
            }
            if (keyData == (Keys.Control | Keys.S))
            {
                OnSaveClick(_btnSave, null);
                return true;
            }
            if (keyData == (Keys.Control | Keys.E))
            {

                if (_canvas == null || _canvas.Mode != ToolMode.AI)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Auto Label",
                        Text = "AI 도구를 활성화한 상태에서만 실행할 수 있습니다.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return true;
                }


                if (string.IsNullOrWhiteSpace(_canvas.ActiveLabelName))
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Auto Label",
                        Text = "활성 라벨이 없습니다. 오른쪽 라벨 목록에서 라벨을 선택해 주세요.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return true;
                }

                var baseDir = GetCurrentImageFolder();
                var rootDir = GetDatasetRootFolderOfSelected();
                if (string.IsNullOrEmpty(baseDir) || string.IsNullOrEmpty(rootDir))
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Auto Label",
                        Text = "현재 선택된 항목에서 기준 폴더를 알 수 없습니다.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return true;
                }


                if (!string.Equals(baseDir, rootDir, StringComparison.OrdinalIgnoreCase))
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Auto Label",
                        Text = "하위 폴더에서는 전체 자동 라벨링을 실행하지 않습니다.\n루트 폴더(최상위)에서 이미지를 선택하고 다시 시도해 주세요.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return true;
                }

                var aiChk = _canvas.GetTool(ToolMode.AI) as AITool;
                if (aiChk == null || !aiChk.HasActiveRoi)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Auto Label",
                        Text = "현재 ROI가 없습니다. AI 도구(우클릭)로 ROI를 지정한 뒤 다시 시도해 주세요.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return true;
                }


                var confirm = new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "Auto Label",
                    Text = "현재 데이터셋의 모든 이미지에 AI Labeling 을 적용하시겠습니까?",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.YesNo,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Question,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();

                if (confirm == DialogResult.Yes)
                {
                    StartAutoLabelAllImagesAsync();
                }
                return true;
            }
            if (keyData == (Keys.Control | Keys.Up) || keyData == (Keys.Control | Keys.Down))
            {
                if (_canvas != null && !_canvas.HasSelection)
                {
                    if ((keyData & Keys.KeyCode) == Keys.Up)
                        NavigateToPreviousImage();
                    else
                        NavigateToNextImage();
                    return true;
                }

            }

            return base.ProcessCmdKey(ref msg, keyData);
        }

        private void EnsureLabelWindow()
        {
            if (_labelWin == null || _labelWin.IsDisposed)
            {
                _labelWin = new LabelCreateWindow();
            }
            _labelWin.ResetForNewLabel();
        }


        private void AddLabelChip(string labelName, Color color)
        {
            if (_rightTools2 == null) return;

            int chipW = Math.Max(LABEL_CHIP_MIN_W, _btnAdd.Width);
            int chipH = 24;

            var chip = MakeLabelChip(labelName, color, chipW, chipH);

            _rightTools2.Controls.Add(chip);
            SelectOnly(chip);

            chip.MouseDown += (s, e) => { if (!_canvas.Focused) _canvas.Focus(); };
            chip.Click += (s, e) => { if (!_canvas.Focused) _canvas.Focus(); };

            AdjustLabelChipWidths();
        }

        // LabelInfo는 chip.Tag로 쓰고 계신 기존 타입 그대로 사용 (Name, Color 등 포함 가정)
        private Guna2Panel MakeLabelChip(string labelName, Color color, int width, int height)
        {
            // Label Chip UI
            var chip = new Guna2Panel();
            chip.Tag = new LabelInfo(labelName, color);
            chip.FillColor = Color.White;
            chip.BorderColor = Color.Silver;
            chip.BorderThickness = 2;
            chip.BorderRadius = 10;
            chip.Size = new Size(Math.Max(120, width), height);
            chip.Margin = new Padding(0, 0, 0, 8);
            chip.Cursor = Cursors.Hand;

            chip.ShadowDecoration.Enabled = false;
            chip.ShadowDecoration.Color = Color.LimeGreen;
            chip.ShadowDecoration.Shadow = new Padding(4);
            chip.ShadowDecoration.BorderRadius = chip.BorderRadius;
            chip.ShadowDecoration.Parent = chip;

            // Color swatch
            var swatch = new Guna2Panel();
            swatch.Name = "__LabelSwatch";
            swatch.Size = new Size(18, 18);
            swatch.BorderRadius = 4;
            swatch.FillColor = color;
            swatch.BorderColor = Color.Silver;
            swatch.BorderThickness = 1;
            swatch.Location = new Point(2, (chip.Height - swatch.Height) / 2);

            // Text label
            var nameLbl = new Label();
            nameLbl.Name = "__LabelText";
            nameLbl.BackColor = Color.Transparent;
            nameLbl.AutoSize = true;
            nameLbl.Text = labelName;
            nameLbl.Location = new Point(swatch.Right + 2, (chip.Height - nameLbl.Height) / 2);

            // Click(좌클릭) 동작 유지
            chip.Click += OnChipClick_Simple;
            swatch.Click += OnChipClick_Simple;
            nameLbl.Click += OnChipClick_Simple;

            // 컨텍스트 메뉴 구성
            var menu = new ContextMenuStrip();
            menu.Items.Add("Rename", null, (s, e) => BeginInlineRename(chip, nameLbl, swatch));
            menu.Items.Add("Delete", null, (s, e) => DeleteChip(chip));

            // 우클릭 어디서든 같은 메뉴
            chip.ContextMenuStrip = menu;
            swatch.ContextMenuStrip = menu;
            nameLbl.ContextMenuStrip = menu;

            chip.Controls.Add(swatch);
            chip.Controls.Add(nameLbl);

            chip.Resize += (s, e) =>
            {
                swatch.Location = new Point(10, (chip.Height - swatch.Height) / 2);
                nameLbl.Location = new Point(swatch.Right + 2, (chip.Height - nameLbl.Height) / 2);
            };

            return chip;
        }

        // --- Rename: 칩 위 인라인 편집 TextBox ---
        private void BeginInlineRename(Guna2Panel chip, Label nameLbl, Control swatch)
        {
            // 이미 편집 중이면 무시
            if (chip.Controls.Find("__EditBox", false).FirstOrDefault() is TextBox) return;

            var tb = new TextBox();
            tb.Name = "__EditBox";
            tb.BorderStyle = BorderStyle.None;
            tb.Font = nameLbl.Font;
            tb.Text = nameLbl.Text;
            tb.Width = Math.Max(80, nameLbl.Width + 20);
            tb.Location = new Point(nameLbl.Left - 1, (chip.Height - tb.PreferredHeight) / 2);
            tb.TabIndex = 0;

            // 시각적 구분선(밑줄 느낌)
            var underline = new Panel
            {
                Height = 1,
                Width = tb.Width,
                BackColor = Color.Silver,
                Left = tb.Left,
                Top = tb.Bottom + 2
            };

            // 기존 Label 가리기
            nameLbl.Visible = false;

            // 저장/취소 로직
            void Commit()
            {
                var newName = tb.Text?.Trim();
                if (string.IsNullOrEmpty(newName))
                {
                    // 빈 이름 방지: 원래 이름 유지
                    Cancel();
                    return;
                }

                // 필요하면 여기서 이름 중복 체크 로직 추가 가능 (FindExistingLabel(newName) 등)

                nameLbl.Text = newName;
                nameLbl.AutoSize = true;

                // Tag(LabelInfo)도 갱신
                if (chip.Tag is LabelInfo info)
                {
                    info.Name = newName; // LabelInfo에 Name 세터가 있다고 가정
                    chip.Tag = info;
                }

                // 위치 재정렬
                nameLbl.Location = new Point(swatch.Right + 2, (chip.Height - nameLbl.Height) / 2);

                // 편집 박스 제거
                chip.Controls.Remove(tb);
                tb.Dispose();
                chip.Controls.Remove(underline);
                underline.Dispose();

                nameLbl.Visible = true;

                // 외부가 필요하면 여기서 "라벨 이름 변경됨" 이벤트/콜백 호출
                // OnLabelRenamed?.Invoke(oldName, newName);
            }

            void Cancel()
            {
                chip.Controls.Remove(tb);
                tb.Dispose();
                chip.Controls.Remove(underline);
                underline.Dispose();
                nameLbl.Visible = true;
            }

            tb.KeyDown += (s, e) =>
            {
                if (e.KeyCode == Keys.Enter) { e.Handled = true; e.SuppressKeyPress = true; Commit(); }
                else if (e.KeyCode == Keys.Escape) { e.Handled = true; e.SuppressKeyPress = true; Cancel(); }
            };
            tb.Leave += (s, e) => Commit();

            chip.Controls.Add(tb);
            chip.Controls.Add(underline);
            tb.BringToFront();
            underline.BringToFront();
            tb.Focus();
            tb.SelectAll();
        }

        // --- Delete: 칩 제거(+확인) ---
        private void DeleteChip(Guna2Panel chip)
        {
            if (chip.Tag is LabelInfo info)
            {
                var confirm = new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "Delete Label",
                    Text = "현재 Label을 삭제 하시겠습니까?",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.YesNo,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Question,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();

                if (confirm == DialogResult.Yes)
                {
                    // 부모 컨테이너에서 제거
                    var parent = chip.Parent;
                    chip.Visible = false;
                    parent?.Controls.Remove(chip);
                    chip.Dispose();
                }
                else
                {
                    return;
                }
            }
        }


        private void OnChipClick_Simple(object sender, System.EventArgs e)
        {
            var chip = FindChipFrom(sender as Control);
            if (chip != null)
                SelectOnly(chip);
            if (_canvas != null && !_canvas.Focused) _canvas.Focus();
        }

        private void UpdateActiveLabelFromChip(Guna2Panel chip)
        {
            if (chip == null || _canvas == null) return;

            string name = null;
            Color stroke = Color.DeepSkyBlue;

            if (chip.Tag is LabelInfo)
            {
                var li = (LabelInfo)chip.Tag;
                name = li.Name;
                stroke = li.Color;
            }
            else
            {
                var swatch = chip.Controls.OfType<Guna2Panel>().FirstOrDefault();
                var nameLbl = chip.Controls.OfType<Guna2HtmlLabel>().FirstOrDefault();
                if (nameLbl != null) name = nameLbl.Text;
                if (swatch != null) stroke = swatch.FillColor;
            }

            var fill = Color.FromArgb(72, stroke);
            _canvas.SetActiveLabel(name ?? "Default", stroke, fill);
        }

        private Guna2Panel FindChipFrom(Control c)
        {
            while (c != null)
            {
                var gp = c as Guna2Panel;
                if (gp != null)
                {
                    if (gp.Tag is LabelInfo || object.Equals(gp.Tag, "LabelChip"))
                        return gp;
                }
                c = c.Parent;
            }
            return null;
        }

        private void SelectOnly(Guna2Panel target)
        {
            if (_rightTools2 == null) return;

            for (int i = 0; i < _rightTools2.Controls.Count; i++)
            {
                var chip = _rightTools2.Controls[i] as Guna2Panel;
                if (chip != null && (chip.Tag is LabelInfo || object.Equals(chip.Tag, "LabelChip")))
                {
                    bool selected = (chip == target);
                    SetSelected(chip, selected);
                    if (selected) UpdateActiveLabelFromChip(chip);
                }
            }
        }

        private void AddDefaultLabelIfMissing()
        {
            bool hasAny = _rightTools2.Controls.OfType<Guna2Panel>()
                           .Any(p => p.Tag is LabelInfo);
            if (hasAny) return;

            AddLabelChip("Default", Color.DeepSkyBlue);
        }

        private void SetSelected(Guna2Panel chip, bool selected)
        {
            chip.ShadowDecoration.Enabled = selected;
            chip.BorderColor = selected ? Color.FromArgb(60, 180, 75) : Color.Silver;
            chip.BorderThickness = 2;
        }

        private void AdjustLabelChipWidths()
        {
            if (_rightToolDock2 == null || _rightTools2 == null || _btnAdd == null) return;
            int targetW = Math.Max(LABEL_CHIP_MIN_W, _btnAdd.Width);

            for (int i = 0; i < _rightTools2.Controls.Count; i++)
            {
                var chip = _rightTools2.Controls[i] as Guna2Panel;
                if (chip != null && (chip.Tag is LabelInfo || object.Equals(chip.Tag, "LabelChip")))
                {
                    chip.Width = targetW;
                }
            }
        }



        private void DisposeCurrentModel()
        {
            try
            {
                _onnxSession?.Dispose();
            }
            catch { /* ignore */ }
            finally { _onnxSession = null; }

            try
            {
                _engineSession?.Dispose();
            }
            catch { /* ignore */ }
            finally { _engineSession = null; }

            try
            {
                _patchcoreSession?.Dispose();
            }
            catch { /* ignore */ }
            finally { _patchcoreSession = null; }
            _patchcoreArtifacts = null;

            _currentModelName = null;
        }

        private void DisposeCurrentPatchCore()
        {
            if (_patchcoreSession != null)
            {
                _patchcoreSession.Dispose();
                _patchcoreSession = null;
            }
            if (_patchcoreArtifacts != null)
            {
                _patchcoreArtifacts = null;
            }
        }

        // 모델 로더 (ONNX)
        private Task LoadOnnxAsync(string onnxPath, IProgress<(int, string)> progress)
        {
            return Task.Run(() =>
            {
                if (_onnxSession == null)
                {
                    // 필요 시 내부에서 warmup 수행 가능
                    _onnxSession = YoloSegOnnx.EnsureSession(onnxPath, progress);
                    _currentModelName = onnxPath;
                }
            });
        }

        // 모델 로더 (TensorRT)
        private Task LoadEngineAsync(string enginePath)
        {
            return Task.Run(() =>
            {
                if (_engineSession == null)
                {
                    _engineSession = new YoloSegEngine(enginePath, deviceId: 0);
                    _currentModelName = enginePath;
                }

                // Warmup 부분 구현 필요
            });
        }

        private async void OnOpenClick(object sender, EventArgs e)
        {
            using (var dlg = new OpenFileDialog())
            {
                dlg.Title = "이미지 파일 or 폴더 or 딥러닝 모델 열기";
                dlg.Filter = "Image/Model|*.png;*.jpg;*.jpeg;*.bmp;*.onnx;*.engine;|모든 파일|*.*";
                dlg.Multiselect = false;
                dlg.CheckFileExists = false;   // 폴더 선택 허용
                dlg.ValidateNames = false;     // "폴더를 선택하려면..." dummy 파일명 허용
                dlg.FileName = "폴더를 선택하려면 이 항목을 클릭하세요";

                if (dlg.ShowDialog(this) != DialogResult.OK) return;

                var chosen = dlg.FileName;

                // ───────────────────────────────────────────────────────────────
                // 1) 파일 선택
                // ───────────────────────────────────────────────────────────────
                if (File.Exists(chosen))
                {
                    var ext = Path.GetExtension(chosen)?.ToLowerInvariant();

                    // wrn50_l3.onnx를 직접 선택 → PatchCore 폴더로 처리
                    bool isWrn = string.Equals(Path.GetFileName(chosen), "wrn50_l3.onnx", StringComparison.OrdinalIgnoreCase);

                    if (ext == ".onnx" || ext == ".engine" || isWrn)
                    {
                        using (var overlay = new ProgressOverlay(this, "Loading model"))
                        {
                            var progress = new Progress<(int, string)>(p => overlay.Report(p.Item1, p.Item2));

                            try
                            {
                                if (ext == ".engine")
                                {
                                    await LoadEngineAsync(chosen).ConfigureAwait(true);
                                    SetModelHeader(Path.GetFileName(chosen));
                                }
                                else if (ext == ".onnx" && !isWrn)
                                {
                                    // 일반 .onnx (예: 기존 분류/세그 모델)
                                    await LoadOnnxAsync(chosen, progress).ConfigureAwait(true);
                                    SetModelHeader(Path.GetFileName(chosen));
                                }
                                else
                                {
                                    DisposeCurrentPatchCore();

                                    // .patchcore 선택 or wrn50_l3.onnx 직접 선택
                                    var folder = Path.GetDirectoryName(chosen) ?? chosen;
                                    await LoadPatchCoreAsync(folder, progress).ConfigureAwait(true);

                                    // 헤더에는 폴더명 표시
                                    var hdr = Path.GetFileName(folder.TrimEnd(Path.DirectorySeparatorChar));
                                    SetModelHeader(string.IsNullOrEmpty(hdr) ? folder : hdr);
                                }
                            }
                            catch (Exception ex)
                            {
                                DisposeCurrentModel();
                                new Guna.UI2.WinForms.Guna2MessageDialog
                                {
                                    Parent = this,
                                    Caption = "오류",
                                    Text = $"모델 열기 실패:\n{ex.Message}",
                                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error
                                }.Show();
                                return;
                            }
                        }

                        UpdateModelDependentControls();
                        return;
                    }

                    // 이미지 파일
                    if (IsImageFile(chosen))
                    {
                        try
                        {
                            LoadImageAtPath(chosen);
                            var folder = Path.GetDirectoryName(chosen);
                            if (!string.IsNullOrEmpty(folder) && Directory.Exists(folder))
                            {
                                PopulateTreeFromFolder(folder);
                                SelectNodeByPath(chosen);
                            }

                            if (_toggleOn)
                            {
                                if (_currentModelName.Contains("PatchCore"))
                                {
                                    _ = AutoPatchCoreInferIfEnabledAsync();

                                }
                                else
                                {
                                    _ = AutoInferIfEnabledAsync();
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show(this, "이미지 로드 오류: " + ex.Message, "오류",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                        }
                        return;
                    }

                    // 그 외: 같은 폴더를 트리로 로드
                    var parent = Path.GetDirectoryName(chosen);
                    if (!string.IsNullOrEmpty(parent) && Directory.Exists(parent))
                    {
                        PopulateTreeFromFolder(parent);
                    }
                    return;
                }

                // ───────────────────────────────────────────────────────────────
                // 2) 폴더 선택 (CheckFileExists=false + ValidateNames=false)
                // ───────────────────────────────────────────────────────────────
                if (Directory.Exists(chosen))
                {
                    // 폴더 안에 PatchCore 4종이 모두 있으면 PatchCore로 처리
                    if (IsPatchCoreFolder(chosen))
                    {
                        using (var overlay = new ProgressOverlay(this, "Loading PatchCore"))
                        {
                            var progress = new Progress<(int, string)>(p => overlay.Report(p.Item1, p.Item2));
                            try
                            {
                                DisposeCurrentModel();
                                await LoadPatchCoreAsync(chosen, progress).ConfigureAwait(true);
                                var hdr = Path.GetFileName(chosen.TrimEnd(Path.DirectorySeparatorChar));
                                SetModelHeader(string.IsNullOrEmpty(hdr) ? chosen : hdr);
                                UpdateModelDependentControls();
                            }
                            catch (Exception ex)
                            {
                                DisposeCurrentModel();
                                new Guna.UI2.WinForms.Guna2MessageDialog
                                {
                                    Parent = this,
                                    Caption = "오류",
                                    Text = $"PatchCore 열기 실패:\n{ex.Message}",
                                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error
                                }.Show();
                            }
                        }
                        return;
                    }

                    // 일반 폴더: 트리 로드
                    PopulateTreeFromFolder(chosen);
                    return;
                }

                // 3) 가짜 파일명 → 폴더 추정
                var maybeFolder = Path.GetDirectoryName(chosen);
                if (!string.IsNullOrEmpty(maybeFolder) && Directory.Exists(maybeFolder))
                {
                    // PatchCore 폴더 인지 한번 더 체크
                    if (IsPatchCoreFolder(maybeFolder))
                    {
                        using (var overlay = new ProgressOverlay(this, "Loading PatchCore"))
                        {
                            var progress = new Progress<(int, string)>(p => overlay.Report(p.Item1, p.Item2));
                            try
                            {
                                await LoadPatchCoreAsync(maybeFolder, progress).ConfigureAwait(true);
                                var hdr = Path.GetFileName(maybeFolder.TrimEnd(Path.DirectorySeparatorChar));
                                SetModelHeader(string.IsNullOrEmpty(hdr) ? maybeFolder : hdr);
                                UpdateModelDependentControls();
                            }
                            catch (Exception ex)
                            {
                                DisposeCurrentModel();
                                new Guna.UI2.WinForms.Guna2MessageDialog
                                {
                                    Parent = this,
                                    Caption = "오류",
                                    Text = $"PatchCore 열기 실패:\n{ex.Message}",
                                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error
                                }.Show();
                            }
                        }
                        return;
                    }

                    PopulateTreeFromFolder(maybeFolder);
                }
            }
        }
        private static bool IsPatchCoreFolder(string folder)
        {
            return File.Exists(Path.Combine(folder, "wrn50_l3.onnx")) &&
                   File.Exists(Path.Combine(folder, "gallery_f32.bin")) &&
                   File.Exists(Path.Combine(folder, "threshold.json")) &&
                   File.Exists(Path.Combine(folder, "meta.json"));
        }

        private async Task LoadPatchCoreAsync(string folder, IProgress<(int, string)> progress)
        {
            progress?.Report((10, "Checking files..."));

            string onnx = Path.Combine(folder, "wrn50_l3.onnx");
            string gallery = Path.Combine(folder, "gallery_f32.bin");
            string thr = Path.Combine(folder, "threshold.json");
            string meta = Path.Combine(folder, "meta.json");

            if (!File.Exists(onnx) || !File.Exists(gallery) || !File.Exists(thr) || !File.Exists(meta))
                throw new FileNotFoundException("PatchCore 모델 구성 요소(.onnx, .bin, .json) 중 일부가 없습니다.");

            progress?.Report((40, "Loading artifacts..."));
            if (_patchcoreArtifacts == null)
            {
                _patchcoreArtifacts = await Task.Run(() => Artifacts.Load(folder)).ConfigureAwait(false);
            }

            progress?.Report((70, "Creating ONNX session..."));
            if (_patchcoreSession == null)
            {
                _patchcoreSession = await Task.Run(() => PatchCoreOnnx.CreateSession(_patchcoreArtifacts.OnnxPath)).ConfigureAwait(false);
            }

            progress?.Report((100, "PatchCore ready."));
        }


        private async Task AutoInferIfEnabledAsync()
        {
            if (!_toggleOn) return;
            if (_onnxSession == null || _sourceImage == null) return;

            // 이전 작업 취소
            _autoInferCts?.Cancel();
            _autoInferCts?.Dispose();
            _autoInferCts = new CancellationTokenSource();

            try
            {
                await RunInferenceAndApplyAsync(_autoInferCts.Token);
            }
            catch (OperationCanceledException)
            {
                // 취소는 무시
            }
        }

        private async Task AutoPatchCoreInferIfEnabledAsync()
        {
            if (!_toggleOn) return;
            if (_patchcoreSession == null || _sourceImage == null) return;

            // 이전 작업 취소
            _autoInferCts?.Cancel();
            _autoInferCts?.Dispose();
            _autoInferCts = new CancellationTokenSource();

            try
            {
                await RunPatchCoreInferenceAsync(_autoInferCts.Token);
            }
            catch (OperationCanceledException)
            {
                // 취소는 무시
            }
        }


        private bool TryLoadYoloForCurrentImage()
        {
            try
            {
                if (_canvas == null || _canvas.Image == null || string.IsNullOrEmpty(_currentImagePath)) return false;

                var root = FindDatasetRootForImage(_currentImagePath);
                if (root == null) return false;

                var classesPath = Path.Combine(root, "classes.txt");
                var labelsPath = Path.Combine(root, "labels", Path.GetFileNameWithoutExtension(_currentImagePath) + ".txt");
                if (!File.Exists(classesPath) || !File.Exists(labelsPath)) return false;


                var classes = ParseClassesTxt(classesPath);
                if (classes == null || classes.Count == 0) classes = new List<string> { "Default" };


                LoadClassColorsFromNotesJson(root, classes);
                RebuildClassColorMapFromChips();


                LoadYoloLabelFile(labelsPath, classes);

                return _canvas.Shapes != null && _canvas.Shapes.Count > 0;
            }
            catch { return false; }
        }

        private void RebuildClassColorMapFromChips()
        {
            _classColorMap.Clear();
            if (_rightTools2 == null) return;

            foreach (Control c in _rightTools2.Controls)
            {
                var pnl = c as Guna2Panel;
                if (pnl == null || !(pnl.Tag is LabelInfo)) continue;

                var li = (LabelInfo)pnl.Tag;


                string name = null;
                try
                {
                    var nprop = li.GetType().GetProperty("Name");
                    if (nprop != null) name = nprop.GetValue(li, null) as string;
                }
                catch { }
                if (string.IsNullOrWhiteSpace(name))
                    name = pnl.Text;
                if (string.IsNullOrWhiteSpace(name))
                    continue;


                Color baseColor = Color.Empty;
                try
                {
                    var pColor = li.GetType().GetProperty("Color");
                    if (pColor != null)
                    {
                        var v = pColor.GetValue(li, null);
                        if (v is Color col) baseColor = col;
                    }
                    if (baseColor.IsEmpty)
                    {
                        var pFill = li.GetType().GetProperty("FillColor");
                        if (pFill != null)
                        {
                            var v = pFill.GetValue(li, null);
                            if (v is Color col2) baseColor = col2;
                        }
                    }
                    if (baseColor.IsEmpty)
                    {
                        var pStroke = li.GetType().GetProperty("StrokeColor");
                        if (pStroke != null)
                        {
                            var v = pStroke.GetValue(li, null);
                            if (v is Color col3) baseColor = col3;
                        }
                    }
                }
                catch { }

                if (baseColor.IsEmpty)
                    baseColor = ColorFromNameDeterministic(name);

                if (!_classColorMap.ContainsKey(name))
                    _classColorMap.Add(name, baseColor);
            }
        }


        private Color ColorFromNameDeterministic(string name)
        {
            unchecked
            {
                int h = 23;
                for (int i = 0; i < name.Length; i++)
                    h = h * 31 + name[i];


                int r = 128 + ((h) & 63) * 2;
                int g = 128 + ((h >> 6) & 63) * 2;
                int b = 128 + ((h >> 12) & 63) * 2;

                r = Math.Max(32, Math.Min(240, r));
                g = Math.Max(32, Math.Min(240, g));
                b = Math.Max(32, Math.Min(240, b));
                return Color.FromArgb(r, g, b);
            }
        }


        private void GetColorsForClass(string labelName, out Color stroke, out Color fill)
        {
            if (string.IsNullOrWhiteSpace(labelName)) labelName = "Default";

            Color baseColor;
            if (!_classColorMap.TryGetValue(labelName, out baseColor))
                baseColor = ColorFromNameDeterministic(labelName);

            stroke = baseColor;
            fill = Color.FromArgb(72, baseColor);
        }


        private string FindDatasetRootForImage(string imagePath)
        {
            var dir = new DirectoryInfo(Path.GetDirectoryName(imagePath));
            var baseName = Path.GetFileNameWithoutExtension(imagePath);

            if (!string.IsNullOrEmpty(_lastYoloExportRoot))
            {
                var cp = Path.Combine(_lastYoloExportRoot, "classes.txt");
                var lp = Path.Combine(_lastYoloExportRoot, "labels", baseName + ".txt");
                if (File.Exists(cp) && File.Exists(lp))
                    return _lastYoloExportRoot;
            }

            if (dir != null && dir.Name.Equals("images", StringComparison.OrdinalIgnoreCase) && dir.Parent != null)
            {
                var root = dir.Parent.FullName;
                var classesOk = File.Exists(Path.Combine(root, "classes.txt"));
                var labelOk = File.Exists(Path.Combine(root, "labels", baseName + ".txt"));
                if (classesOk && labelOk) return root;
            }



            var walk = dir;
            for (int i = 0; i < 3 && walk != null; i++, walk = walk.Parent)
            {
                var candidate = walk.FullName;
                if (File.Exists(Path.Combine(candidate, "classes.txt")) &&
                    File.Exists(Path.Combine(candidate, "labels", baseName + ".txt")))
                    return candidate;
            }
            return null;
        }

        private List<string> ParseClassesTxt(string classesPath)
        {
            var list = new List<string>();
            foreach (var raw in File.ReadAllLines(classesPath, Encoding.UTF8))
            {
                var s = (raw ?? "").Trim();
                if (s.Length > 0) list.Add(s);
            }
            if (list.Count == 0) list.Add("Default");
            return list;
        }

        private void LoadYoloLabelFile(string labelPath, List<string> classes)
        {
            var img = _canvas.Image; int W = img.Width, H = img.Height;
            var ci = CultureInfo.InvariantCulture;

            _canvas.ClearAllShapes();

            foreach (var raw in File.ReadAllLines(labelPath))
            {
                var line = (raw ?? "").Trim();
                if (line.Length == 0 || line.StartsWith("#")) continue;

                var tok = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                if (tok.Length < 5) continue;

                int cls;
                if (!int.TryParse(tok[0], out cls)) continue;
                string labelName = (cls >= 0 && cls < classes.Count) ? classes[cls] : "cls_" + cls.ToString();


                if (tok.Length == 5)
                {
                    float cx = float.Parse(tok[1], ci) * W;
                    float cy = float.Parse(tok[2], ci) * H;
                    float ww = float.Parse(tok[3], ci) * W;
                    float hh = float.Parse(tok[4], ci) * H;

                    var r = new RectangleF(cx - ww * 0.5f, cy - hh * 0.5f, ww, hh);

                    Color stroke, fill;
                    GetColorsForClass(labelName, out stroke, out fill);
                    _canvas.AddBox(r, labelName, stroke, fill);
                    continue;
                }


                if (((tok.Length - 1) % 2) == 0 && (tok.Length - 1) >= 6)
                {
                    var pts = new List<PointF>();
                    for (int i = 1; i < tok.Length; i += 2)
                    {
                        float nx = float.Parse(tok[i], ci);
                        float ny = float.Parse(tok[i + 1], ci);
                        pts.Add(new PointF(nx * W, ny * H));
                    }

                    Color stroke2, fill2;
                    GetColorsForClass(labelName, out stroke2, out fill2);
                    _canvas.AddPolygon(pts, labelName, stroke2, fill2);
                }
            }
        }


        private static bool IsImageFile(string path)
        {
            try
            {
                var ext = Path.GetExtension(path);
                return !string.IsNullOrEmpty(ext) && _imgExts.Contains(ext);
            }
            catch { return false; }
        }
        private void PopulateTreeFromFolder(string rootPath)
        {
            if (string.IsNullOrWhiteSpace(rootPath) || !Directory.Exists(rootPath)) return;

            _currentFolder = rootPath;
            TryBindAnnotationRootNear(rootPath);
            _fileTree.BeginUpdate();
            _fileTree.Nodes.Clear();

            var root = CreateFolderNode(rootPath);
            root.Text = new DirectoryInfo(rootPath).Name;
            _fileTree.Nodes.Add(root);
            root.Expand();

            _fileTree.EndUpdate();
        }
        private TreeNode CreateFolderNode(string folder)
        {
            var node = new TreeNode(Path.GetFileName(folder)) { Tag = folder };

            try
            {
                foreach (var sub in Directory.GetDirectories(folder))
                {
                    var subNode = CreateFolderNode(sub);
                    node.Nodes.Add(subNode);
                }
            }
            catch { }

            try
            {
                var files = Directory.GetFiles(folder)
                                     .Where(IsImageFile)
                                     .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                                     .ToArray();
                foreach (var f in files)
                {
                    var fn = Path.GetFileName(f);
                    var imgNode = new TreeNode(fn) { Tag = f };
                    LabelStatusService.ApplyNodeState(imgNode, f, _lastYoloExportRoot, showCountSuffix: true);
                    node.Nodes.Add(imgNode);
                }
            }
            catch { }

            return node;
        }
        private void SelectNodeByPath(string fullPath)
        {
            if (_fileTree.Nodes.Count == 0) return;

            TreeNode found = null;
            var q = new Queue<TreeNode>();
            foreach (TreeNode n in _fileTree.Nodes) q.Enqueue(n);

            while (q.Count > 0)
            {
                var n = q.Dequeue();
                var tag = n.Tag as string;
                if (string.Equals(tag, fullPath, StringComparison.OrdinalIgnoreCase))
                {
                    found = n; break;
                }
                foreach (TreeNode c in n.Nodes) q.Enqueue(c);
            }

            if (found != null)
            {
                _fileTree.SelectedNode = found;
                found.EnsureVisible();
            }
        }
        private void LoadImageAtPath(string path)
        {
            try
            {
                _sourceImage?.Dispose();

                using (var temp = Image.FromFile(path))
                {
                    _sourceImage = new Bitmap(temp);
                }

                var viewBmp = (Bitmap)_sourceImage.Clone();
                _canvas.LoadImage(viewBmp);
                _canvas.ClearInferenceOverlays();
                _canvas.ZoomToFit();

                _currentImagePath = path;
                if (!_canvas.Focused) _canvas.Focus();
                _canvasLayer.Invalidate();
                TryBindAnnotationRootNear(Path.GetDirectoryName(_currentImagePath));
                TryLoadYoloForCurrentImage();

                if (_aiSubMode == AiSubMode.Roi)
                {
                    var ai = _canvas.GetTool(ToolMode.AI) as AITool;
                    ai?.EnsureRoiForCurrentImage(_canvas, _lastRoiNorm);
                }

                BeginInvoke(new Action(() =>
                {
                    if (_canvas != null && _canvas.CanFocus) _canvas.Focus();
                }));
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, "이미지 로드 오류: " + ex.Message, "오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }



        private bool IsImageNode(TreeNode n)
        {
            if (n == null) return false;
            var path = n.Tag as string;
            return !string.IsNullOrEmpty(path) && File.Exists(path) && IsImageFile(path);
        }

        private TreeNode FindCurrentImageNode()
        {
            var sel = (_fileTree != null) ? _fileTree.SelectedNode : null;
            if (IsImageNode(sel)) return sel;

            if (!string.IsNullOrEmpty(_currentImagePath))
            {
                var byPath = FindNodeByImagePath(_currentImagePath);
                if (IsImageNode(byPath)) return byPath;
            }
            return null;
        }

        private List<TreeNode> GetAllImageNodes()
        {
            var list = new List<TreeNode>();
            if (_fileTree == null || _fileTree.Nodes.Count == 0) return list;

            var stack = new Stack<TreeNode>();
            foreach (TreeNode n in _fileTree.Nodes) stack.Push(n);


            while (stack.Count > 0)
            {
                var node = stack.Pop();
                if (IsImageNode(node)) list.Add(node);


                for (int i = node.Nodes.Count - 1; i >= 0; i--)
                    stack.Push(node.Nodes[i]);
            }
            return list;
        }

        private void OpenImageFromNode(TreeNode node)
        {
            if (node == null) return;
            var path = node.Tag as string;
            if (string.IsNullOrEmpty(path)) return;

            if (_fileTree != null) _fileTree.SelectedNode = node;
            else LoadImageAtPath(path);
        }

        private void NavigateToNextImage()
        {
            var nodes = GetAllImageNodes();
            if (nodes.Count == 0) return;

            var cur = FindCurrentImageNode();
            int idx = (cur != null) ? nodes.IndexOf(cur) : -1;
            if (idx < 0) idx = -1;

            if (idx + 1 < nodes.Count)
                OpenImageFromNode(nodes[idx + 1]);
        }

        private void NavigateToPreviousImage()
        {
            var nodes = GetAllImageNodes();
            if (nodes.Count == 0) return;

            var cur = FindCurrentImageNode();
            int idx = (cur != null) ? nodes.IndexOf(cur) : -1;

            if (idx > 0)
                OpenImageFromNode(nodes[idx - 1]);
        }

        #endregion

        #region 6) Export / Import (YOLO Segmentation)

        private List<string> GetCurrentClasses()
        {
            var classes = new List<string>();
            if (_rightTools2 != null)
            {
                for (int i = 0; i < _rightTools2.Controls.Count; i++)
                {
                    var chip = _rightTools2.Controls[i] as Guna2Panel;
                    if (chip != null && chip.Tag is LabelInfo)
                    {
                        var li = (LabelInfo)chip.Tag;
                        if (!string.IsNullOrWhiteSpace(li.Name) && !classes.Contains(li.Name))
                            classes.Add(li.Name);
                    }
                }
            }
            if (classes.Count == 0) classes.Add("Default");
            return classes;
        }
        // 기존: private void SaveDatasetYoloWithImages()
        private void SaveDatasetYoloWithImages(string srcImagePath = null)
        {
            // 1) 유효 경로 결정 (인자 우선, 없으면 _currentImagePath)
            string imagePath = !string.IsNullOrEmpty(srcImagePath) ? srcImagePath : _currentImagePath;

            if (_canvas == null || _canvas.Image == null)
                throw new InvalidOperationException("이미지가 없습니다.");

            if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
                throw new InvalidOperationException("현재 이미지 경로를 찾을 수 없습니다.");

            // 2) 이하 모든 _currentImagePath 참조를 imagePath로 교체
            var baseDir = Path.GetDirectoryName(imagePath);
            var annotationRoot = Path.Combine(baseDir, "AnnotationData");
            var rootDir = Path.Combine(annotationRoot, "Model");

            LabelStatusService.SetStorageRoot(annotationRoot);

            var imagesDir = Path.Combine(rootDir, "images");
            var labelsDir = Path.Combine(rootDir, "labels");
            var masksDir = Path.Combine(rootDir, "masks");
            Directory.CreateDirectory(imagesDir);
            Directory.CreateDirectory(labelsDir);
            Directory.CreateDirectory(masksDir);

            var classes = GetCurrentClasses();
            File.WriteAllLines(Path.Combine(rootDir, "classes.txt"), classes, Encoding.UTF8);
            SaveNotesJson(Path.Combine(rootDir, "notes.json"), classes);
            _lastYoloExportRoot = rootDir;

            var srcExt = Path.GetExtension(imagePath);
            var baseName = Path.GetFileNameWithoutExtension(imagePath);
            var dstImagePath = Path.Combine(imagesDir, baseName + srcExt);
            File.Copy(imagePath, dstImagePath, true);

            var dstLabelPath = Path.Combine(labelsDir, baseName + ".txt");
            WriteYoloLabelForCurrentImage(dstLabelPath, classes);

            var dstMaskPath = Path.Combine(masksDir, baseName + ".png");
            using (var mask = _canvas.RenderBinaryMask8bpp())
            {
                if (mask != null) mask.Save(dstMaskPath, ImageFormat.Png);
            }

            LabelStatusService.MarkLabeled(imagePath, _canvas.Shapes.Count);

            try
            {
                var node = FindNodeByImagePath(imagePath);
                if (node != null)
                    LabelStatusService.ApplyNodeState(node, imagePath, _lastYoloExportRoot, showCountSuffix: true);
            }
            catch { }
        }


        private TreeNode FindNodeByImagePath(string fullPath)
        {
            if (string.IsNullOrEmpty(fullPath)) return null;
            return FindNodeByImagePathRec(_fileTree.Nodes, fullPath);
        }
        private TreeNode FindNodeByImagePathRec(TreeNodeCollection nodes, string fullPath)
        {
            foreach (TreeNode n in nodes)
            {
                var tagPath = n.Tag as string;
                if (!string.IsNullOrEmpty(tagPath) &&
                    tagPath.Equals(fullPath, StringComparison.OrdinalIgnoreCase))
                    return n;

                if (n.Nodes.Count > 0)
                {
                    var hit = FindNodeByImagePathRec(n.Nodes, fullPath);
                    if (hit != null) return hit;
                }
            }
            return null;
        }

        private void SaveNotesJson(string path, List<string> classes)
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");

            sb.AppendLine("  \"categories\": [");
            for (int i = 0; i < classes.Count; i++)
            {
                sb.Append("    { \"id\": ").Append(i)
                  .Append(", \"name\": \"").Append(EscapeJson(classes[i])).Append("\" }");
                if (i < classes.Count - 1) sb.Append(",");
                sb.AppendLine();
            }
            sb.AppendLine("  ],");


            sb.AppendLine("  \"colors\": {");
            for (int i = 0; i < classes.Count; i++)
            {
                var name = classes[i];
                var c = GetBaseColorForClass(name);
                sb.Append("    \"").Append(EscapeJson(name)).Append("\": \"").Append(ToHexRgb(c)).Append("\"");
                if (i < classes.Count - 1) sb.Append(",");
                sb.AppendLine();
            }
            sb.AppendLine("  },");

            sb.AppendLine("  \"info\": {");
            sb.Append("    \"year\": ").Append(DateTime.Now.Year).AppendLine(",");
            sb.AppendLine("    \"version\": \"1.0\",");
            sb.AppendLine("    \"contributor\": \"SmartLabelingApp\"");
            sb.AppendLine("  }");
            sb.AppendLine("}");
            File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
        }
        private static string EscapeJson(string s)
        {
            if (s == null) return "";
            return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }


        private Color GetBaseColorForClass(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) name = "Default";

            if (_classColorMap.TryGetValue(name, out var c) && c != Color.Empty) return c;


            if (_rightTools2 != null)
            {
                foreach (Control cc in _rightTools2.Controls)
                {
                    var pnl = cc as Guna2Panel;
                    if (pnl == null || !(pnl.Tag is LabelInfo)) continue;
                    var li = (LabelInfo)pnl.Tag;
                    if (string.Equals(li.Name, name, StringComparison.OrdinalIgnoreCase))
                        return li.Color;
                }
            }

            return ColorFromNameDeterministic(name);
        }

        private static string ToHexRgb(Color c) => $"#{c.R:X2}{c.G:X2}{c.B:X2}";

        private static bool TryParseHexColor(string s, out Color color)
        {
            color = Color.Empty;
            if (string.IsNullOrWhiteSpace(s)) return false;
            s = s.Trim();
            if (s.StartsWith("#")) s = s.Substring(1);
            if (s.Length == 6)
            {
                try
                {
                    int r = int.Parse(s.Substring(0, 2), System.Globalization.NumberStyles.HexNumber);
                    int g = int.Parse(s.Substring(2, 2), System.Globalization.NumberStyles.HexNumber);
                    int b = int.Parse(s.Substring(4, 2), System.Globalization.NumberStyles.HexNumber);
                    color = Color.FromArgb(r, g, b);
                    return true;
                }
                catch { return false; }
            }
            return false;
        }


        private void ApplyColorToChip(Guna2Panel chip, Color color)
        {
            if (chip == null) return;

            if (chip.Tag is LabelInfo li)
                chip.Tag = new LabelInfo(li.Name, color);

            var swatch = chip.Controls.OfType<Guna2Panel>().FirstOrDefault(p => p.Name == "__LabelSwatch");
            if (swatch != null) swatch.FillColor = color;
        }


        private Guna2Panel FindChipByName(string name)
        {
            if (_rightTools2 == null || string.IsNullOrWhiteSpace(name)) return null;
            foreach (Control cc in _rightTools2.Controls)
            {
                var pnl = cc as Guna2Panel;
                if (pnl == null || !(pnl.Tag is LabelInfo)) continue;
                var li = (LabelInfo)pnl.Tag;
                if (string.Equals(li.Name, name, StringComparison.OrdinalIgnoreCase))
                    return pnl;
            }
            return null;
        }





        private void LoadClassColorsFromNotesJson(string rootDir, List<string> classes)
        {
            try
            {
                var notesPath = Path.Combine(rootDir, "notes.json");
                if (!File.Exists(notesPath)) return;

                string json = File.ReadAllText(notesPath, Encoding.UTF8);

                var colors = new Dictionary<string, Color>(StringComparer.OrdinalIgnoreCase);


                int idx = json.IndexOf("colors", StringComparison.OrdinalIgnoreCase);
                if (idx >= 0)
                {
                    int brace = json.IndexOf('{', idx);
                    if (brace >= 0)
                    {
                        int depth = 0;
                        int end = -1;
                        for (int i = brace; i < json.Length; i++)
                        {
                            if (json[i] == '{') depth++;
                            else if (json[i] == '}')
                            {
                                depth--;
                                if (depth == 0) { end = i; break; }
                            }
                        }
                        if (end > brace)
                        {
                            string obj = json.Substring(brace + 1, end - brace - 1);

                            var parts = obj.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                            foreach (var part in parts)
                            {
                                var kv = part.Split(new char[] { ':' }, 2);
                                if (kv.Length == 2)
                                {
                                    string key = kv[0].Trim().Trim('"');
                                    string val = kv[1].Trim().Trim('"');
                                    if (TryParseHexColor(val, out var col))
                                        if (!colors.ContainsKey(key)) colors.Add(key, col);
                                }
                            }
                        }
                    }
                }

                if (colors.Count > 0)
                {

                    foreach (var kv in colors)
                        _classColorMap[kv.Key] = kv.Value;


                    foreach (var name in classes)
                    {
                        var want = GetBaseColorForClass(name);
                        var chip = FindChipByName(name);
                        if (chip == null)
                        {

                            AddLabelChip(name, want);
                            chip = FindChipByName(name);
                        }
                        else
                        {
                            ApplyColorToChip(chip, want);
                        }
                    }
                }
            }
            catch { }
        }

        private static float Clamp01(float v) => (v < 0f) ? 0f : (v > 1f ? 1f : v);

        private static float SignedArea(IList<PointF> poly)
        {
            double a = 0;
            for (int i = 0, j = poly.Count - 1; i < poly.Count; j = i++)
                a += (double)(poly[j].X * poly[i].Y - poly[i].X * poly[j].Y);
            return (float)(0.5 * a);
        }

        private static void RemoveConsecutiveDuplicates(List<PointF> pts, float eps2 = 1e-12f)
        {
            if (pts == null || pts.Count < 2) return;
            var outPts = new List<PointF>(pts.Count);
            PointF prev = pts[0];
            outPts.Add(prev);
            for (int i = 1; i < pts.Count; i++)
            {
                float dx = pts[i].X - prev.X, dy = pts[i].Y - prev.Y;
                if (dx * dx + dy * dy > eps2) { outPts.Add(pts[i]); prev = pts[i]; }
            }
            pts.Clear(); pts.AddRange(outPts);
        }

        private static List<PointF> DownsampleByIndex(List<PointF> pts, int maxCount)
        {
            if (pts == null) return null;
            if (pts.Count <= maxCount) return pts;
            var outPts = new List<PointF>(maxCount);
            for (int i = 0; i < maxCount; i++)
            {
                int idx = (int)Math.Round((double)i * (pts.Count - 1) / (maxCount - 1));
                outPts.Add(pts[idx]);
            }
            return outPts;
        }
        private async void OnTrainClick(object sender, EventArgs e)
        {
            try
            {

                string weightsPath = PathHelper.ResolvePretrainedPath();


                if (!File.Exists(weightsPath))
                {

                    var dr = PretrainedWeightsDialog.ShowForMissingDefault(this, weightsPath);

                    if (dr != DialogResult.OK)
                    {

                        return;
                    }


                    weightsPath = PathHelper.ResolvePretrainedPath();


                    if (!File.Exists(weightsPath))
                    {
                        bool ready = await WaitForFileReadyAsync(
                            weightsPath,
                            TimeSpan.FromSeconds(5),
                            (pct, status) => { }
                        );

                        if (!ready)
                        {




                            return;
                        }
                    }
                }


                StartTraining(weightsPath);
            }
            catch (Exception ex)
            {
                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "오류",
                    Text = ex.Message,
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
            }
        }
        // 공용: 원본(_sourceImage)로 추론 + 오버레이 생성 + 화면 적용까지 한 번에 수행
        private async Task<bool> RunInferenceAndApplyAsync(CancellationToken token = default)
        {
            var overlays = new List<SmartLabelingApp.ImageCanvas.OverlayItem>();

            // 공통 로그 편의를 위해
            void Log(string msg) => AddLog(msg);

            // 엔진이 하나도 없으면 종료
            if (_onnxSession == null && _engineSession == null)
            {
                MessageBox.Show(this, "모델이 로드되지 않았습니다. 먼저 .onnx 또는 .engine 파일을 여세요.",
                    "안내", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return false;
            }

            // 결과 비트맵 (두 경로 모두 공통 렌더러가 돌려주는 Bitmap)
            Bitmap onnxOverlay = null;
            Bitmap engineOverlay = null;

            using (var srcCopy = (Bitmap)_sourceImage.Clone())
            {
                await Task.Run(() =>
                {
                    if (token.IsCancellationRequested) return;

                    if (_onnxSession != null)
                    {
                        // ---------------- ONNX 경로 ----------------
                        var res = SegmentationInfer.InferOnnx(_onnxSession, srcCopy, conf: 0.1f, iou: 0.1f);

                        var swOverlay = System.Diagnostics.Stopwatch.StartNew();
                        // 공통 오버레이: ONNX 스타일의 그림을 두 백엔드 공통으로
                        onnxOverlay = OverlayRendererFast.RenderEx(srcCopy, res, overlaysOut: overlays, maskThr: 0.8f, alpha: 0.3f);
                        swOverlay.Stop();

                        Log($"[ONNX] Inference 완료: ClassCount={res.Dets.Count}개, pre={res.PreMs:F0}ms, infer={res.InferMs:F0}ms, post={res.PostMs:F0}ms");
                        Log($"[ONNX] DrawOverlay 완료: {swOverlay.Elapsed.TotalMilliseconds:F0}ms");
                        Log($"[ONNX] 총합 ≈ {(res.PreMs + res.InferMs + res.PostMs + swOverlay.Elapsed.TotalMilliseconds):F0}ms");
                    }
                    else if (_engineSession != null)
                    {
                        // ---------------- TensorRT 경로 ----------------
                        var swAll = System.Diagnostics.Stopwatch.StartNew();

                        // 1) 추론 (SegResult 반환, ProtoFlat=KHW 보장)
                        var res = SegmentationInfer.Infer(_engineSession, srcCopy);
                        var inferMs = res.InferMs;

                        // 2) 공통 오버레이 (이전 _engineSession.OverlayFast 제거)
                        var swOverlay = System.Diagnostics.Stopwatch.StartNew();
                        engineOverlay = OverlayRendererFast.RenderEx(srcCopy, res, overlaysOut: overlays);
                        swOverlay.Stop();

                        swAll.Stop();
                        Log($"[TRT] Inference 완료: ClassCount={res.Dets.Count}개, pre={res.PreMs:F0}ms, infer={res.InferMs:F0}ms, post={res.PostMs:F0}ms");
                        Log($"[TRT] DrawOverlay 완료: {swOverlay.Elapsed.TotalMilliseconds:F0}ms");
                        Log($"[TRT] 총합 ≈ {(res.PreMs + res.InferMs + res.PostMs + swOverlay.Elapsed.TotalMilliseconds):F0}ms");
                    }
                }, token).ConfigureAwait(true);
            }

            if (token.IsCancellationRequested)
                return false;

            // 결과 적용 (두 경로 모두 비트맵만 세팅)
            if (_onnxSession != null)
            {
                if (onnxOverlay == null) return false;
                _canvas.SetImageAndOverlays(onnxOverlay, overlays);
                return true;
            }
            else if (_engineSession != null)
            {
                if (engineOverlay == null) return false;
                _canvas.SetImageAndOverlays(engineOverlay, overlays);
                return true;
            }

            return false;
        }




        private async void OnInferClick(object sender, EventArgs e)
        {
            if (_canvas == null || _canvas.Image == null)
            {
                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "알림",
                    Text = "이미지가 없습니다. 먼저 이미지를 Open으로 불러오세요.",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Information
                }.Show();
                return;
            }

            try
            {
                if (_currentModelName.Contains("PatchCore"))
                {
                    await RunPatchCoreInferenceAsync(CancellationToken.None);
                }
                else
                {
                    await RunInferenceAndApplyAsync(CancellationToken.None);
                }
            }
            catch (Exception ex)
            {
                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "오류",
                    Text = $"추론 실패:\n{ex.Message}",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error
                }.Show();
            }
        }

        // 클래스 상단이나 필드로 추가 (사용자가 값 조정 가능)
        private const float USER_THRESHOLD = 0.27f;

        private async Task<bool> RunPatchCoreInferenceAsync(CancellationToken token = default)
        {
            if (_patchcoreSession == null || _patchcoreArtifacts == null)
            {
                MessageBox.Show(this, "PatchCore 모델이 로드되지 않았습니다. 먼저 .patchcore를 여세요.",
                    "안내", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return false;
            }

            var overlays = new List<SmartLabelingApp.ImageCanvas.OverlayItem>();
            Bitmap resultBmp = null;
            float imgScore = 0;
            bool isNg = false;

            using (var srcCopy = (Bitmap)_sourceImage.Clone())
            {
                await Task.Run(() =>
                {
                    if (token.IsCancellationRequested) return;

                    var swAll = System.Diagnostics.Stopwatch.StartNew();

                    // 1) 전처리
                    var swPre = System.Diagnostics.Stopwatch.StartNew();

                    DenseTensor<float> input;
                    if (_patchcoreArtifacts.InputSize == 224)
                    {
                        input = ImagePreprocessor.PreprocessToCHW(srcCopy, new PreprocessConfig
                        {
                            resize = 256,
                            crop = _patchcoreArtifacts.InputSize,
                        });
                    }
                    else
                    {
                        input = ImagePreprocessor.PreprocessToCHW(srcCopy, new PreprocessConfig
                        {
                            resize = 384,
                            crop = _patchcoreArtifacts.InputSize,
                        });
                    }

                    swPre.Stop();

                    // 2) ONNX 실행 (layer3)
                    var swInfer = System.Diagnostics.Stopwatch.StartNew();
                    var results = PatchCoreOnnx.Run(_patchcoreSession, input);
                    var tL3 = results.First(r => r.Name == "layer3").AsTensor<float>();
                    swInfer.Stop();

                    // 3) L3-only 패치 임베딩
                    var swEmbed = System.Diagnostics.Stopwatch.StartNew();
                    var pe = PatchEmbedder.BuildFromL3(tL3);
                    swEmbed.Stop();

                    // 4) 거리 계산 (cosine, k=1)
                    var swDist = System.Diagnostics.Stopwatch.StartNew();
                    var patchMin = new float[pe.Patches];
                    bool usedMKL = false, usedSIMD = false;

                    try
                    {
                        Distance.RowwiseMinDistancesIP_MKL(
                            pe.RowsRowMajor, pe.Patches, pe.Dim,
                            _patchcoreArtifacts.Gallery, _patchcoreArtifacts.GalleryRows,
                            patchMin);
                        usedMKL = true;
                    }
                    catch (TypeLoadException) { }

                    if (!usedMKL)
                    {
                        try
                        {
                            Distance.RowwiseMinDistancesIP_Optimized(
                                pe.RowsRowMajor, pe.Patches, pe.Dim,
                                _patchcoreArtifacts.Gallery, _patchcoreArtifacts.GalleryRows,
                                patchMin);
                            usedSIMD = true;
                        }
                        catch (MissingMethodException) { }
                        catch (EntryPointNotFoundException) { }
                    }

                    if (!usedMKL && !usedSIMD)
                    {
                        Distance.RowwiseMinDistances(
                            pe.RowsRowMajor, pe.Patches, pe.Dim,
                            _patchcoreArtifacts.Gallery, _patchcoreArtifacts.GalleryRows,
                            "ip", patchMin);
                    }

                    imgScore = patchMin.Max();
                    isNg = imgScore > USER_THRESHOLD; // 기준점: 낮으면 NG
                    swDist.Stop();

                    // 5) OK/NG 결과 처리
                    var swOverlay = System.Diagnostics.Stopwatch.StartNew();
                    if (isNg)
                    {
                        // NG → 히트맵 + NG 라벨
                        var heat = HeatmapOverlay.MakeOverlay(srcCopy, patchMin, _patchcoreArtifacts.GridH, _patchcoreArtifacts.GridW, alphaMin: 0f, alphaMax: 0.6f, gamma: 2.5f);
                        resultBmp = UiOverlayUtils.DrawStatusFrameFromAnomaly(heat, true, imgScore);
                    }
                    else
                    {
                        // OK → 원본 + OK 라벨
                        resultBmp = UiOverlayUtils.DrawStatusFrameFromAnomaly(srcCopy, false, imgScore);
                    }
                    swOverlay.Stop();

                    swAll.Stop();

                    var sumMs = swPre.Elapsed.TotalMilliseconds + swInfer.Elapsed.TotalMilliseconds + swEmbed.Elapsed.TotalMilliseconds + swDist.Elapsed.TotalMilliseconds;

                    // 로그
                    AddLog($"[PatchCore] pre={swPre.Elapsed.TotalMilliseconds:F0}ms, " +
                           $"infer={swInfer.Elapsed.TotalMilliseconds:F0}ms, " +
                           $"embed={swEmbed.Elapsed.TotalMilliseconds:F0}ms, " +
                           $"dist={swDist.Elapsed.TotalMilliseconds:F0}ms, 총 Inference Time: {sumMs:F0}ms, ({(usedMKL ? "MKL" : usedSIMD ? "SIMD" : "Naive")})");
                    AddLog($"[PatchCore] DrawOverlay : {swOverlay.Elapsed.TotalMilliseconds:F0}ms");
                    AddLog($"[PatchCore] 총합 ≈ {swAll.Elapsed.TotalMilliseconds:F0}ms");
                }, token).ConfigureAwait(true);
            }

            if (token.IsCancellationRequested || resultBmp == null)
                return false;

            _canvas.SetImageAndOverlays(resultBmp, overlays);
            return true;
        }




        private async Task<bool> WaitForFileReadyAsync(string path, TimeSpan timeout, Action<int, string> progress = null)
        {
            var start = DateTime.UtcNow;
            long lastSize = -1;
            DateTime lastChange = DateTime.UtcNow;

            while (DateTime.UtcNow - start < timeout)
            {
                try
                {
                    if (File.Exists(path))
                    {
                        long size = new FileInfo(path).Length;

                        if (size > 0)
                        {
                            if (size != lastSize)
                            {
                                lastSize = size;
                                lastChange = DateTime.UtcNow;
                                progress?.Invoke(80, "다운로드 중...");
                            }
                            else
                            {

                                if ((DateTime.UtcNow - lastChange).TotalSeconds >= 1.0)
                                {
                                    progress?.Invoke(100, "가중치 준비 완료");
                                    return true;
                                }
                            }
                        }
                    }
                }
                catch { }


                double frac = (DateTime.UtcNow - start).TotalMilliseconds / timeout.TotalMilliseconds;
                int pct = Math.Min(95, Math.Max(5, (int)Math.Round(frac * 95)));
                progress?.Invoke(pct, "가중치 확인 중...");

                await Task.Delay(500);
            }
            return File.Exists(path);
        }

        private async void StartTraining(string pretrainedWeightsPath)
        {

            const string baseDir = @"D:\SmartLabelingApp";
            string venvDir = Path.Combine(baseDir, ".venv");
            string pythonExe = Path.Combine(venvDir, "Scripts", "python.exe");
            string yoloExe = Path.Combine(venvDir, "Scripts", "yolo.exe");

            bool cudaUsable = false;

            if (string.IsNullOrEmpty(pretrainedWeightsPath) || !File.Exists(pretrainedWeightsPath))
            {
                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "Train",
                    Text = $"프리트레인 가중치(.pt)를 찾을 수 없습니다.\n경로: {pretrainedWeightsPath ?? "(null)"}",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
                return;
            }

            string zipPath;
            using (var ofd = new OpenFileDialog())
            {
                ofd.Title = "학습 데이터 ZIP 선택";
                ofd.Filter = "Zip Archives (*.zip)|*.zip";
                ofd.CheckFileExists = true;
                ofd.CheckPathExists = true;
                ofd.Multiselect = false;
                ofd.RestoreDirectory = true;

                string initialDir;
                if (!string.IsNullOrEmpty(_lastExportZipPath) && File.Exists(_lastExportZipPath))
                {
                    initialDir = Path.GetDirectoryName(_lastExportZipPath);

                    ofd.FileName = Path.GetFileName(_lastExportZipPath);
                }
                else
                {
                    initialDir = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
                }
                ofd.InitialDirectory = initialDir;

                if (ofd.ShowDialog(this) != DialogResult.OK) return;
                zipPath = ofd.FileName;
            }

            using (var overlay = new ProgressOverlay(this, "환경 준비 (10분 ~ 20분 소요)", true))
            {
                try
                {
                    overlay.Report(0, "시작 준비 중...");
                    await EnvSetup.EnsureVenvAndUltralyticsAsync(
                        baseDir, venvDir, pythonExe,
                        (pct, status) => overlay.Report(pct, status)
                    );

                    // ★ 여기서 CUDA 커널 가용성 사전 점검(5090 등의 커널 미지원 시 자동 CPU 폴백)
                    overlay.Report(92, "CUDA 가용성 점검..."); // ★ (표시만 추가, 기존 흐름 영향 없음)
                    try
                    {
                        cudaUsable = GpuDetector.CanUseCudaForKernels(pythonExe, baseDir); // ★
                    }
                    catch
                    {
                        cudaUsable = false;
                    }

                    overlay.Report(100, "환경 준비 완료");
                }
                catch (Exception ex)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Train",
                        Text = $"가상환경 준비 실패:\n{ex.Message}",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return;
                }
            }

            string extractRoot = Path.Combine(baseDir, "Result");
            string dataYamlPath = null;
            string datasetRoot = null;
            int trainImg = 0, trainLbl = 0, valImg = 0, valLbl = 0;

            using (var overlay = new ProgressOverlay(this, "데이터 준비", true))
            {
                try
                {
                    overlay.Report(0, "Result 폴더 정리...");
                    await ZipDatasetUtils.ExtractZipWithProgressAsync(
                        zipPath, extractRoot, (pct, msg) => overlay.Report(pct, msg));

                    overlay.Report(86, "data.yaml 탐색...");
                    dataYamlPath = DataYamlPatcher.FindDataYaml(extractRoot);
                    if (string.IsNullOrEmpty(dataYamlPath))
                        throw new Exception("data.yaml을 찾을 수 없습니다. ZIP 구조를 확인하세요.");

                    datasetRoot = Path.GetDirectoryName(dataYamlPath) ?? extractRoot;

                    overlay.Report(90, "디렉토리 구조 검증...");
                    DataYamlPatcher.ValidateRequiredDirs(datasetRoot);

                    overlay.Report(94, "파일 개수 점검...");
                    (trainImg, trainLbl) = DataYamlPatcher.CountPair(datasetRoot, @"images\train", @"labels\train");
                    (valImg, valLbl) = DataYamlPatcher.CountPair(datasetRoot, @"images\val", @"labels\val");

                    if (trainImg == 0 || trainLbl == 0) throw new Exception("train 이미지/라벨이 비어 있습니다.");
                    if (valImg == 0 || valLbl == 0) throw new Exception("val 이미지/라벨이 비어 있습니다.");

                    DataYamlPatcher.FixDataYamlForExtractedDataset(dataYamlPath, datasetRoot);
                    overlay.Report(100, "데이터 준비 완료");
                }
                catch (Exception ex)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Train",
                        Text = $"데이터 준비 실패:\n{ex.Message}",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return;
                }
            }

            string projectDir = Path.Combine(baseDir, "runs");
            Directory.CreateDirectory(projectDir);
            string runName = "finetune_" + DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string bestOut = Path.Combine(projectDir, runName, "weights", "best.pt");

            int epochs = TrainingConfig.Epochs;
            int imgsz = TrainingConfig.ImgSize;

            int batch = cudaUsable ? TrainingConfig.BatchGpu : TrainingConfig.BatchCpu;
            string device = cudaUsable ? TrainingConfig.DeviceGpu : TrainingConfig.DeviceCpu;

            string args = string.Join(" ",
                "segment", "train",
                "model=" + YoloCli.Quote(pretrainedWeightsPath),
                "data=" + YoloCli.Quote(dataYamlPath),
                "epochs=" + epochs,
                "imgsz=" + imgsz,
                "batch=" + batch,
                "device=" + device,
                "project=" + YoloCli.Quote(projectDir),
                "retina_masks=True",
                "overlap_mask=True",
                "name=" + YoloCli.Quote(runName)
            );

            string bestCopy = null;
            string onnxPath = null;

            using (var overlay2 = new ProgressOverlay(this, "학습 실행", true))
            {
                try
                {
                    overlay2.Report(0, "YOLO 준비 중...");
                    var cli = YoloCli.GetYoloCli(yoloExe, pythonExe);

                    int exit = await YoloTrainer.RunYoloTrainWithEpochProgressAsync(
                        cli.fileName, cli.argumentsPrefix + " " + args, baseDir,
                        (pct, status) => overlay2.Report(pct, status),
                        0, 96, expectedTotalEpochs: epochs);

                    if (exit != 0)
                        throw new Exception($"YOLO 학습 프로세스가 실패했습니다. (exit={exit})");

                    overlay2.Report(98, "결과 수집...");
                    if (!File.Exists(bestOut))
                        throw new Exception($"best.pt를 찾을 수 없습니다.\n경로: {bestOut}");
                    string saveTo = Path.Combine(baseDir, "weights", "finetuned", runName);
                    Directory.CreateDirectory(saveTo);
                    bestCopy = Path.Combine(saveTo, "best.pt");
                    File.Copy(bestOut, bestCopy, true);

                    overlay2.Report(100, "학습 완료");
                }
                catch (Exception ex)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "Train",
                        Text = $"학습 실행 실패:\n{ex.Message}",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return;
                }
            }

            if (!string.IsNullOrEmpty(bestCopy) && File.Exists(bestCopy))
            {
                using (var ov = new ProgressOverlay(this, "Export: ONNX", true))
                {
                    try
                    {
                        ov.Report(2, "onnx/onnxsim 의존성 확인...");
                        var pipEnv = EnvSetup.GetPipEnv(baseDir);
                        int ec = ProcessRunner.RunProcessProgress(
                            pythonExe,
                            "-m pip install --upgrade --no-cache-dir --prefer-binary onnx onnxsim --timeout 180 --retries 2",
                            baseDir, 2, 20, (p, s) => ov.Report(p, s), "pip (onnx)", pipEnv);
                        if (ec != 0) throw new Exception("onnx/onnxsim 설치 실패");

                        ov.Report(22, "ONNX 변환 준비...");
                        var cli = YoloCli.GetYoloCli(yoloExe, pythonExe);
                        string exportArgs = string.Join(" ",
                            "export",
                            "model=" + YoloCli.Quote(bestCopy),
                            "format=onnx",
                            "opset=12",
                            "dynamic=True",
                            "simplify=True",
                            "imgsz=" + imgsz
                        );

                        ov.Report(25, "ONNX 변환 중...");
                        ec = ProcessRunner.RunProcessProgress(
                            cli.fileName, cli.argumentsPrefix + " " + exportArgs, baseDir,
                            25, 95, (p, s) => ov.Report(p, "ONNX 변환 중..."), "yolo export", null);
                        if (ec != 0) throw new Exception("ONNX 변환 실패");

                        ov.Report(97, "결과 확인...");
                        string searchStart = Path.GetDirectoryName(bestCopy) ?? baseDir;
                        onnxPath = Directory.EnumerateFiles(searchStart, "*.onnx", SearchOption.AllDirectories)
                                            .OrderByDescending(p => new FileInfo(p).LastWriteTimeUtc)
                                            .FirstOrDefault();

                        if (string.IsNullOrEmpty(onnxPath) || !File.Exists(onnxPath))
                            throw new Exception(".onnx 파일을 찾지 못했습니다.");

                        ov.Report(100, "Export 완료");
                    }
                    catch (Exception ex)
                    {
                        new Guna.UI2.WinForms.Guna2MessageDialog
                        {
                            Parent = this,
                            Caption = "Export",
                            Text = "ONNX 내보내기 실패:\n" + ex.Message,
                            Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                            Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                            Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                        }.Show();
                    }
                }
            }

            {
                var msg = "학습이 완료되었습니다.\n\n" +
                          $"runs 경로: {Path.Combine(projectDir, runName)}" +
                          (string.IsNullOrEmpty(bestCopy) ? "" : $"\nPT 복사본: {bestCopy}") +
                          (string.IsNullOrEmpty(onnxPath) ? "" : $"\nONNX: {onnxPath}");

                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "Train",
                    Text = msg,
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Information,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
            }
        }

        private static List<PointF> ResampleClosedByArcLen(List<PointF> closed, int target)
        {
            if (closed == null || closed.Count < 3) return closed ?? new List<PointF>();
            var poly = new List<PointF>(closed);
            if (poly[0] != poly[poly.Count - 1]) poly.Add(poly[0]);

            int n = poly.Count;
            var cum = new double[n];
            cum[0] = 0;
            double total = 0;
            for (int i = 1; i < n; i++)
            {
                double dx = poly[i].X - poly[i - 1].X, dy = poly[i].Y - poly[i - 1].Y;
                double seg = Math.Sqrt(dx * dx + dy * dy);
                total += seg; cum[i] = total;
            }
            if (total <= 1e-9) return new List<PointF> { poly[0] };

            target = Math.Max(8, target);
            var outPts = new List<PointF>(target);
            int segIdx = 1;
            for (int i = 0; i < target; i++)
            {
                double s = (i * total) / target;
                while (segIdx < n && cum[segIdx] < s) segIdx++;
                if (segIdx >= n) segIdx = n - 1;

                var a = poly[segIdx - 1];
                var b = poly[segIdx];
                double segStart = cum[segIdx - 1];
                double segLen = Math.Max(1e-9, cum[segIdx] - segStart);
                double t = (s - segStart) / segLen;

                outPts.Add(new PointF(
                    (float)(a.X + (b.X - a.X) * t),
                    (float)(a.Y + (b.Y - a.Y) * t)
                ));
            }
            return outPts;
        }


        private static List<PointF> GetLargestClosedOutline(GraphicsPath gp)
        {
            if (gp == null) return null;
            gp.Flatten();
            var pts = gp.PathPoints;
            var types = gp.PathTypes;
            if (pts == null || types == null || pts.Length < 3) return null;

            byte mask = (byte)PathPointType.PathTypeMask;
            int start = 0;
            List<PointF> best = null;
            double bestArea = 0;

            for (int i = 0; i < types.Length; i++)
            {
                bool isStart = ((types[i] & mask) == (byte)PathPointType.Start);
                bool isLast = (i == types.Length - 1);
                bool nextIsStart = !isLast && ((types[i + 1] & mask) == (byte)PathPointType.Start);

                if (isStart) start = i;
                if (isLast || nextIsStart)
                {
                    int len = i - start + 1;
                    if (len >= 3)
                    {
                        var seg = new List<PointF>(len + 1);
                        for (int k = 0; k < len; k++) seg.Add(pts[start + k]);
                        if (seg[0] != seg[seg.Count - 1]) seg.Add(seg[0]);


                        double area = 0;
                        for (int a = 0, b = seg.Count - 1; a < seg.Count; b = a++)
                            area += (double)(seg[b].X * seg[a].Y - seg[a].X * seg[b].Y);
                        area = Math.Abs(area) * 0.5;

                        if (area > bestArea) { bestArea = area; best = seg; }
                    }
                }
            }
            if (best == null) return null;
            if (best.Count >= 2 && best[0] == best[best.Count - 1]) best.RemoveAt(best.Count - 1);
            return best;
        }


        private static bool AppendSegLine(List<string> lines, int cls, IList<PointF> ptsImg, int W, int H, IFormatProvider ci)
        {
            if (ptsImg == null || ptsImg.Count < 3) return false;

            var poly = new List<PointF>(ptsImg);


            if (poly.Count >= 2)
            {
                var a = poly[0]; var b = poly[poly.Count - 1];
                if (Math.Abs(a.X - b.X) < 1e-6f && Math.Abs(a.Y - b.Y) < 1e-6f)
                    poly.RemoveAt(poly.Count - 1);
            }
            RemoveConsecutiveDuplicates(poly);
            if (poly.Count < 3) return false;


            if (SignedArea(poly) < 0) poly.Reverse();

            var sb = new System.Text.StringBuilder();
            sb.Append(cls);
            for (int i = 0; i < poly.Count; i++)
            {
                float nx = Clamp01(poly[i].X / W);
                float ny = Clamp01(poly[i].Y / H);
                sb.Append(' ').Append(nx.ToString(ci)).Append(' ').Append(ny.ToString(ci));
            }
            lines.Add(sb.ToString());
            return true;
        }


        private void WriteYoloLabelForCurrentImage(string labelFilePath, List<string> classes)
        {
            var img = _canvas.Image;
            int W = img.Width, H = img.Height;
            var ci = System.Globalization.CultureInfo.InvariantCulture;


            int CIRCLE_SAMPLES_DEFAULT = Math.Max(8, EditorUIConfig.CircleSegVertexCount);
            int BRUSH_MAX_PTS = 256;
            int POLY_MAX_PTS = 512;

            var lines = new List<string>();

            for (int i = 0; i < _canvas.Shapes.Count; i++)
            {
                var s = _canvas.Shapes[i];
                string lbl = GetShapeLabel(s);
                int cls = GetOrAppendClassId(lbl, classes);


                if (s is RectangleShape rs)
                {
                    var r = rs.RectImg;
                    var rectPoly = new List<PointF>
            {
                new PointF(r.Left,  r.Top),
                new PointF(r.Right, r.Top),
                new PointF(r.Right, r.Bottom),
                new PointF(r.Left,  r.Bottom),
            };
                    if (rectPoly.Count > POLY_MAX_PTS) rectPoly = DownsampleByIndex(rectPoly, POLY_MAX_PTS);
                    AppendSegLine(lines, cls, rectPoly, W, H, ci);
                    continue;
                }


                if (s is CircleShape cs)
                {
                    var r = cs.RectImg;
                    float cx = r.X + r.Width * 0.5f;
                    float cy = r.Y + r.Height * 0.5f;
                    float rad = Math.Max(r.Width, r.Height) * 0.5f;

                    int n = (cs.VertexCount > 0) ? cs.VertexCount : CIRCLE_SAMPLES_DEFAULT;
                    n = Math.Max(8, Math.Min(n, POLY_MAX_PTS));

                    var pts = new List<PointF>(n);
                    for (int k = 0; k < n; k++)
                    {
                        double th = 2.0 * Math.PI * k / n;
                        pts.Add(new PointF(
                            cx + (float)(rad * Math.Cos(th)),
                            cy + (float)(rad * Math.Sin(th))
                        ));
                    }
                    AppendSegLine(lines, cls, pts, W, H, ci);
                    continue;
                }


                if (s is TriangleShape ts && ts.PointsImg != null && ts.PointsImg.Count >= 3)
                {
                    var pts = ts.PointsImg;
                    if (pts.Count > POLY_MAX_PTS) pts = DownsampleByIndex(new List<PointF>(pts), POLY_MAX_PTS);
                    AppendSegLine(lines, cls, pts, W, H, ci);
                    continue;
                }
                if (s is PolygonShape ps && ps.PointsImg != null && ps.PointsImg.Count >= 3)
                {
                    var pts = ps.PointsImg;
                    if (pts.Count > POLY_MAX_PTS) pts = DownsampleByIndex(new List<PointF>(pts), POLY_MAX_PTS);
                    AppendSegLine(lines, cls, pts, W, H, ci);
                    continue;
                }


                if (s is BrushStrokeShape bs)
                {
                    using (var gp = bs.GetAreaPathImgClone())
                    {
                        var outer = GetLargestClosedOutline(gp);
                        if (outer != null && outer.Count >= 3)
                        {
                            int target = Math.Min(BRUSH_MAX_PTS, Math.Max(32, outer.Count));
                            var res = ResampleClosedByArcLen(outer, target);
                            AppendSegLine(lines, cls, res, W, H, ci);
                        }
                    }
                    continue;
                }


                var b = s.GetBoundsImg();
                if (!b.IsEmpty)
                {
                    var boxPoly = new List<PointF>
            {
                new PointF(b.Left,  b.Top),
                new PointF(b.Right, b.Top),
                new PointF(b.Right, b.Bottom),
                new PointF(b.Left,  b.Bottom),
            };
                    AppendSegLine(lines, cls, boxPoly, W, H, ci);
                }
            }

            System.IO.File.WriteAllLines(labelFilePath, lines, System.Text.Encoding.ASCII);
        }

        private int GetOrAppendClassId(string label, List<string> classes)
        {
            if (string.IsNullOrWhiteSpace(label)) label = "Default";
            int idx = classes.IndexOf(label);
            if (idx < 0)
            {
                classes.Add(label);
                idx = classes.Count - 1;
            }
            return idx;
        }

        private string GetShapeLabel(object shape)
        {

            var t = shape.GetType();
            var p = t.GetProperty("LabelName") ?? t.GetProperty("Name");
            if (p != null)
            {
                var v = p.GetValue(shape, null);
                if (v != null) return v.ToString();
            }
            return "Default";
        }
        #endregion

        #region 7) ColorMap 관련 기능 구현

        #region Fields
        private Bitmap _colorMapOriginal;      // 원본
        private bool _srcIs16;                 // 비트 깊이 (실제 처리 기준)
        private byte[] _src8;                  // 8bit intensity
        private ushort[] _src16;               // 16bit intensity

        private Bitmap _dst32;                 // 표시용 버퍼
        private Bitmap _preview32;             // 프리뷰 버퍼
        private uint[] _palette256;            // LUT (256 팔레트)
        private int _lastMinFull = -1, _lastMaxFull = -1;
        private int _lastMinPrev = -1, _lastMaxPrev = -1;

        private ColorMapWindow _colorMapWin;
        private System.Windows.Forms.Timer _liveThrottle;
        private volatile ushort _pendingMin, _pendingMax;
        private volatile bool _isDragging;
        private volatile bool _cmBusy;
        private int _previewStep = 4;          // 기본 다운샘플 배수 (16bit)
        private Image _lastBoundSource;

        private uint[] _lut8_colors;          // 길이 256, val(0..255) → ARGB 색
        private ushort _lut8_min, _lut8_max;  // 캐시된 LUT의 min/max
        #endregion

        #region Functions
        private static void Log(string msg) => Trace.WriteLine($"[ColorMap] {DateTime.Now:HH:mm:ss.fff} {msg}");
        private void OnColorMapClick(object sender = null, EventArgs e = null)
        {
            if (_canvas?.Image == null) return;
            _canvas.Image.Tag = _currentImagePath;

            var cur = _canvas.Image as Bitmap;

            if (_colorMapOriginal == null
                || !_colorMapOriginal.Size.Equals(cur.Size)
                || !ReferenceEquals(_lastBoundSource, _canvas.Image))
            {
                ResetColorMapStateForNewSource(_canvas.Image);
                _lastBoundSource = _canvas.Image;
            }

            if (_colorMapWin == null || _colorMapWin.IsDisposed)
            {
                _colorMapWin = new ColorMapWindow();
                _colorMapWin.FormClosing += (s, ev) => { ev.Cancel = true; _colorMapWin.Hide(); };

                _colorMapWin.OnLiveUpdate = (min, max) => { _pendingMin = min; _pendingMax = max; _isDragging = true; };
                _colorMapWin.OnColorChanged = _ => { _isDragging = false; ApplyColorMapFull(_colorMapWin.SelectedMin, _colorMapWin.SelectedMax); };

                _liveThrottle = new System.Windows.Forms.Timer { Interval = 50 };
                _liveThrottle.Tick += (s, ev) => { if (_isDragging) ApplyColorMapPreview(_pendingMin, _pendingMax); };
                _liveThrottle.Start();
            }

            _colorMapWin.ConfigureForBitDepth(_srcIs16 ? 16 : 8);
            _colorMapWin.SetInitialRange(0, (ushort)(_srcIs16 ? 65535 : 255));

            ApplyColorMapFull(_colorMapWin.SelectedMin, _colorMapWin.SelectedMax);
            _colorMapWin.Show();
            _colorMapWin.Activate();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private uint[] GetOrBuildLut8(ushort minU, ushort maxU)
        {
            // 캐시 히트
            if (_lut8_colors != null && _lut8_min == minU && _lut8_max == maxU)
                return _lut8_colors;

            EnsurePalette256(); // _palette256 준비

            // min/max 보정
            int min = minU;
            int max = Math.Max(min + 1, (int)maxU);
            int range = max - min;

            // 고정소수점 스케일 (16비트 정밀도)
            int scaleMul = (int)((255.0 / range) * 65536.0);
            long scaleAdd = -((long)min * scaleMul);

            var lut = _lut8_colors ?? new uint[256];

            for (int v = 0; v < 256; v++)
            {
                int idx = (int)(((long)v * scaleMul + scaleAdd) >> 16);

                // 코그넥스 규칙: 윈도우 밖(≤min, ≥max)은 항상 검정
                if (idx <= 0 || idx >= 255)
                    lut[v] = 0xFF000000u;
                else
                    lut[v] = _palette256[idx];
            }

            _lut8_colors = lut;
            _lut8_min = minU;
            _lut8_max = maxU;
            return lut;
        }
        private void BuildIntensityCache(Bitmap src)
        {
            int w = src.Width, h = src.Height;

            // 1) WIC 경로 시도: Gray16이면 ushort[] 로 로딩
            if (!string.IsNullOrEmpty(_currentImagePath) &&
                TryLoadGray16WithWIC(_currentImagePath, out var buf16, out int ww, out int hh))
            {
                _srcIs16 = true;
                _src16 = buf16;
                w = ww; h = hh;

                // 표시 버퍼는 src.Size에 맞추지만, 로딩 크기와 다르면 맞춰줌
                if (_dst32 == null || _dst32.Width != w || _dst32.Height != h)
                {
                    BuildOrResizeDst(w, h);
                    BuildOrResizePreview(w, h, _previewStep);
                }
                EnsurePalette256();
                return;
            }

            // 2) 실패 → 8bit 경로(G 채널)로 폴백
            _srcIs16 = false;

            var r = new Rectangle(Point.Empty, src.Size);
            BitmapData bd = src.LockBits(r, ImageLockMode.ReadOnly, src.PixelFormat);
            int bpp = Image.GetPixelFormatSize(src.PixelFormat) / 8;

            try
            {
                _src8 = new byte[w * h];
                unsafe
                {
                    byte* sbase = (byte*)bd.Scan0;
                    Parallel.For(0, h, y =>
                    {
                        byte* srow = sbase + y * bd.Stride;
                        int ofs = y * w;

                        for (int x = 0; x < w; x++)
                        {
                            // B,G,R 순서(Format32bppArgb / 24bppRgb 모두 BGR 순)
                            byte b = srow[x * bpp + 0];
                            byte g = srow[x * bpp + 1];
                            byte r8 = srow[x * bpp + 2];

                            // BT.601 luma (정수): (77*R + 150*G + 29*B + 128) >> 8
                            _src8[ofs + x] = (byte)((r8 * 77 + g * 150 + b * 29 + 128) >> 8);
                        }
                    });
                }
            }
            finally { src.UnlockBits(bd); }

            EnsurePalette256();
        }
        private bool TryLoadGray16WithWIC(string path, out ushort[] data, out int width, out int height)
        {
            data = null; width = height = 0;

            try
            {
                using (var fs = System.IO.File.OpenRead(path))
                {
                    var dec = System.Windows.Media.Imaging.BitmapDecoder.Create(
                        fs,
                        System.Windows.Media.Imaging.BitmapCreateOptions.PreservePixelFormat,
                        System.Windows.Media.Imaging.BitmapCacheOption.OnLoad);

                    // ★ 공통 기반형으로 선언 (BitmapSource)
                    System.Windows.Media.Imaging.BitmapSource src = dec.Frames[0];

                    // Gray16 여부 확인
                    bool isGray16 = src.Format == System.Windows.Media.PixelFormats.Gray16;

                    // 16bpp면 Gray16으로 강제 변환 시도
                    if (!isGray16 && src.Format.BitsPerPixel == 16)
                    {
                        try
                        {
                            src = new System.Windows.Media.Imaging.FormatConvertedBitmap(
                                src,
                                System.Windows.Media.PixelFormats.Gray16,
                                null, 0);
                            isGray16 = true;
                        }
                        catch { isGray16 = false; }
                    }

                    if (!isGray16) return false;

                    width = src.PixelWidth;
                    height = src.PixelHeight;
                    int stride = (width * 16 + 7) / 8; // 2 * width 정렬

                    var bytes = new byte[stride * height];
                    src.CopyPixels(bytes, stride, 0);

                    // byte[] → ushort[] (little-endian)
                    data = new ushort[width * height];
                    System.Buffer.BlockCopy(bytes, 0, data, 0, width * height * sizeof(ushort));
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }
        private void BuildOrResizeDst(int w, int h)
        {
            bool needNew = true;
            if (_dst32 != null)
            {
                try { needNew = !(_dst32.Width == w && _dst32.Height == h); }
                catch { needNew = true; } // 핸들 손상 시
            }
            if (!needNew) return;

            // ✅ Canvas에서 먼저 분리 후 Dispose (중요)
            if (ReferenceEquals(_canvas?.Image, _dst32)) _canvas.Image = null;
            try { _dst32?.Dispose(); } catch { /* ignore */ }

            _dst32 = new Bitmap(w, h, PixelFormat.Format32bppArgb);

            // 최초 생성 시 바인딩
            if (_canvas != null && _canvas.Image == null)
                _canvas.Image = _dst32;
        }
        private void BuildOrResizePreview(int w, int h, int step)
        {
            int pw = Math.Max(1, w / step);
            int ph = Math.Max(1, h / step);

            bool needNew = true;
            if (_preview32 != null)
            {
                try { needNew = !(_preview32.Width == pw && _preview32.Height == ph); }
                catch { needNew = true; }
            }
            if (!needNew) return;

            try { _preview32?.Dispose(); } catch { }
            _preview32 = new Bitmap(pw, ph, PixelFormat.Format32bppArgb);
        }
        private void EnsurePalette256()
        {
            if (_palette256 != null) return;
            _palette256 = new uint[256];
            for (int i = 0; i < 256; i++)
            {
                double t = i / 255.0;
                double r = MathUtils.Clamp(1.5 - Math.Abs(4 * t - 3));
                double g = MathUtils.Clamp(1.5 - Math.Abs(4 * t - 2));
                double b = MathUtils.Clamp(1.5 - Math.Abs(4 * t - 1));
                _palette256[i] = 0xFF000000u
                               | ((uint)(r * 255) << 16)
                               | ((uint)(g * 255) << 8)
                               | (uint)(b * 255);
            }
        }
        private void ResetColorMapStateForNewSource(Image newSrc)
        {
            // Canvas에서 분리 후 버퍼 폐기
            if (ReferenceEquals(_canvas?.Image, _dst32)) _canvas.Image = null;

            try { _dst32?.Dispose(); } catch { }
            try { _preview32?.Dispose(); } catch { }
            try { _colorMapOriginal?.Dispose(); } catch { }

            _dst32 = null;
            _preview32 = null;
            _colorMapOriginal = null;

            _src8 = null;
            _src16 = null;

            _palette256 = null;
            _lastMinFull = _lastMaxFull = _lastMinPrev = _lastMaxPrev = -1;

            _cmBusy = false;
            _isDragging = false;

            // 새 원본 채택 + 캐시 재구축
            _colorMapOriginal = (Bitmap)newSrc.Clone();
            BuildIntensityCache(_colorMapOriginal); // 내부에서 _srcIs16 판정/로딩 수행
            BuildOrResizeDst(_colorMapOriginal.Width, _colorMapOriginal.Height);
            BuildOrResizePreview(_colorMapOriginal.Width, _colorMapOriginal.Height, _previewStep);

            // 화면 바인딩(최초 1회만)
            if (!ReferenceEquals(_canvas.Image, _dst32))
                _canvas.Image = _dst32;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ComputeScale16_16(ushort minU, ushort maxU, bool is16bit, out int scaleMul, out long scaleAdd)
        {
            int min = minU;
            int max = Math.Max(min + 1, (int)maxU);
            int range = max - min;

            // ✅ LUT 인덱스는 항상 0~255
            // 16bit는 입력값이 크므로 그대로 정규화,
            // 8bit는 입력이 이미 0~255라 단위 보정 필요.
            double scaleFactor = is16bit ? (255.0 / range) : (255.0 / range);
            scaleMul = (int)(scaleFactor * 65536.0);
            scaleAdd = -((long)min * scaleMul);
        }

        private void ApplyColorMapPreview(ushort minU, ushort maxU)
        {
            if (_dst32 == null || _preview32 == null) return;

            // 8bit는 더 거칠게: step 6 (FPS↑), 16bit는 설정값 유지
            int step = _srcIs16 ? _previewStep : Math.Max(_previewStep, 6);
            if (_preview32.Width != Math.Max(1, _dst32.Width / step)
                || _preview32.Height != Math.Max(1, _dst32.Height / step))
            {
                BuildOrResizePreview(_dst32.Width, _dst32.Height, step);
            }

            int dMin = Math.Abs((int)minU - _lastMinPrev);
            int dMax = Math.Abs((int)maxU - _lastMaxPrev);
            int threshold = _srcIs16 ? 32 : 1;
            if (dMin < threshold && dMax < threshold) return;
            if (_cmBusy) return;
            _cmBusy = true;

            try
            {
                EnsurePalette256();
                ComputeScale16_16(minU, maxU, _srcIs16, out int scaleMul, out long scaleAdd);

                int pw = _preview32.Width, ph = _preview32.Height;
                var pr = new Rectangle(Point.Empty, _preview32.Size);
                BitmapData pbd = _preview32.LockBits(pr, ImageLockMode.WriteOnly, _preview32.PixelFormat);

                try
                {
                    unsafe
                    {
                        if (_srcIs16)
                        {
                            var src16 = _src16;
                            for (int y = 0; y < ph; y++)
                            {
                                uint* drow = (uint*)(pbd.Scan0 + y * pbd.Stride);
                                int sy = Math.Min(_dst32.Height - 1, y * step);
                                int srcOfs = sy * _dst32.Width;
                                for (int x = 0; x < pw; x++)
                                {
                                    int sx = Math.Min(_dst32.Width - 1, x * step);
                                    ushort val = src16[srcOfs + sx];
                                    int idx = (int)(((long)val * scaleMul + scaleAdd) >> 16);
                                    drow[x] = (idx <= 0 || idx >= 255) ? 0xFF000000u : _palette256[idx];
                                }
                            }
                        }
                        else
                        {
                            var src8 = _src8;
                            var lut = GetOrBuildLut8(minU, maxU); // ★ 8-bit 성능 핵심

                            // 미세한 이미지라도 드래그 동안 부하가 있으므로 병렬 허용
                            Parallel.For(0, ph, y =>
                            {
                                uint* drow = (uint*)(pbd.Scan0 + y * pbd.Stride);
                                int sy = Math.Min(_dst32.Height - 1, y * step);
                                int srcOfs = sy * _dst32.Width;
                                for (int x = 0; x < pw; x++)
                                {
                                    int sx = Math.Min(_dst32.Width - 1, x * step);
                                    drow[x] = lut[src8[srcOfs + sx]];
                                }
                            });
                        }
                    }
                }
                finally { _preview32.UnlockBits(pbd); }

                // 업스케일 Blit
                using (var g = Graphics.FromImage(_dst32))
                {
                    g.CompositingMode = CompositingMode.SourceCopy;
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;
                    g.PixelOffsetMode = PixelOffsetMode.Half;
                    g.DrawImage(_preview32, new Rectangle(0, 0, _dst32.Width, _dst32.Height));
                }

                _canvas.Invalidate();
                _lastMinPrev = minU; _lastMaxPrev = maxU;
            }
            finally { _cmBusy = false; }
        }

        private void ApplyColorMapFull(ushort minU, ushort maxU)
        {
            if (_dst32 == null) return;
            try { var _ = _dst32.Width; } // 핸들 검증
            catch
            {
                Log("⚠️ _dst32 손상 → 재생성");
                BuildOrResizeDst(_colorMapOriginal.Width, _colorMapOriginal.Height);
                return;
            }

            if (_lastMinFull == (int)minU && _lastMaxFull == (int)maxU) return;
            if (_cmBusy) return;
            _cmBusy = true;

            try
            {
                EnsurePalette256();
                ComputeScale16_16(minU, maxU, _srcIs16, out int scaleMul, out long scaleAdd);

                int w = _dst32.Width, h = _dst32.Height;
                var r = new Rectangle(Point.Empty, _dst32.Size);
                BitmapData dbd = _dst32.LockBits(r, ImageLockMode.WriteOnly, _dst32.PixelFormat);

                try
                {
                    unsafe
                    {
                        if (_srcIs16)
                        {
                            var src16 = _src16;
                            Parallel.For(0, h, y =>
                            {
                                uint* drow = (uint*)(dbd.Scan0 + y * dbd.Stride);
                                int ofs = y * w;
                                for (int x = 0; x < w; x++)
                                {
                                    ushort val = src16[ofs + x];
                                    int idx = (int)(((long)val * scaleMul + scaleAdd) >> 16);
                                    drow[x] = (idx <= 0 || idx >= 255) ? 0xFF000000u : _palette256[idx];
                                }
                            });
                        }
                        else
                        {
                            var src8 = _src8;
                            var lut = GetOrBuildLut8(minU, maxU); // ★ 8-bit 성능 핵심

                            // 8-bit 풀 적용은 단일스레드가 빠른 경우도 있지만,
                            // 큰 해상도 대비를 위해 병렬 허용(필요 시 취향에 맞게 단일 루프로 변경 가능)
                            Parallel.For(0, h, y =>
                            {
                                uint* drow = (uint*)(dbd.Scan0 + y * dbd.Stride);
                                int ofs = y * w;
                                for (int x = 0; x < w; x++)
                                {
                                    drow[x] = lut[src8[ofs + x]];
                                }
                            });
                        }
                    }
                }
                finally { _dst32.UnlockBits(dbd); }

                _canvas.Invalidate();
                _lastMinFull = minU; _lastMaxFull = maxU;
                _lastMinPrev = minU; _lastMaxPrev = maxU;
            }
            finally { _cmBusy = false; }
        }

        private void RunColorMapBatchToDesktop()
        {
            try
            {
                var baseDir = GetCurrentImageFolder();
                if (string.IsNullOrEmpty(baseDir) || !Directory.Exists(baseDir))
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "ColorMap Export",
                        Text = "현재 이미지의 기준 폴더를 찾을 수 없습니다.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return;
                }

                var outDir = GetDesktopResultFolder();

                var paths = EnumerateTopImagesInFolder(baseDir)
                            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                            .ToList();

                if (paths.Count == 0)
                {
                    new Guna.UI2.WinForms.Guna2MessageDialog
                    {
                        Parent = this,
                        Caption = "ColorMap Export",
                        Text = "처리할 이미지가 없습니다.",
                        Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                        Icon = Guna.UI2.WinForms.MessageDialogIcon.Warning,
                        Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                    }.Show();
                    return;
                }

                int total = paths.Count;
                int okCnt = 0, failCnt = 0;

                using (var overlay = new ProgressOverlay(this, "ColorMap Export...", true))
                {
                    for (int i = 0; i < total; i++)
                    {
                        string src = paths[i];
                        string nameNoExt = Path.GetFileNameWithoutExtension(src);
                        string dst = Path.Combine(outDir, nameNoExt + ".png");

                        // === 개별 파일 기준으로 8/16bit 판정 (PixelFormat 사용 X, WIC만 사용) ===
                        bool is16bit = false;
                        try
                        {
                            if (TryLoadGray16WithWIC(src, out var tmp16, out int w0, out int h0))
                            {
                                is16bit = true;
                            }
                        }
                        catch
                        {
                            is16bit = false;
                        }

                        ushort min, max;
                        if (is16bit)
                        {
                            // 16bit: 0~65535 전체 범위
                            min = COLORMAP_TARGET_START_VALUE_16;
                            max = COLORMAP_TARGET_END_VALUE_16;
                        }
                        else
                        {
                            // 8bit or 컬러: 0~255 전체 범위
                            min = COLORMAP_TARGET_START_VALUE_8;
                            max = COLORMAP_TARGET_END_VALUE_8;
                        }

                        overlay.Report(
                            (i * 100) / Math.Max(1, total),
                            $"{i + 1}/{total} - {Path.GetFileName(src)}"
                            + (is16bit ? " (16bit, 0~65535)" : " (8bit, 0~255)")
                        );

                        try
                        {
                            // 내부에서 Gray16/8bit에 맞게 컬러맵 적용
                            bool ok = ApplyColorMapAndSaveOne(src, dst, min, max);
                            if (ok) okCnt++; else failCnt++;
                        }
                        catch
                        {
                            failCnt++;
                        }
                    }

                    overlay.Report(100, "완료");
                }

                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "ColorMap Export",
                    Text = $"총 {total}개\n성공: {okCnt}\n실패: {failCnt}\n출력 폴더: {outDir}",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Information,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
            }
            catch (Exception ex)
            {
                new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "ColorMap Export",
                    Text = "오류: " + ex.Message,
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                }.Show();
            }
        }





        private bool ApplyColorMapAndSaveOne(string srcPath, string dstPath, ushort min, ushort max)
        {
            // 공용 팔레트/LUT 빌드
            EnsurePalette256();

            // 1) Gray16(WIC) 시도
            if (TryLoadGray16WithWIC(srcPath, out var buf16, out int w16, out int h16))
            {
                // 16-bit 경로
                ComputeScale16_16(min, max, true, out int scaleMul, out long scaleAdd);
                using (var dst = new Bitmap(w16, h16, System.Drawing.Imaging.PixelFormat.Format32bppArgb))
                {
                    var rect = new Rectangle(0, 0, w16, h16);
                    var bd = dst.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, dst.PixelFormat);
                    try
                    {
                        unsafe
                        {
                            Parallel.For(0, h16, y =>
                            {
                                uint* drow = (uint*)((byte*)bd.Scan0 + y * bd.Stride); // ← 각 스레드 지역 변수
                                int ofs = y * w16;                                  // ← 각 스레드 지역 변수
                                for (int x = 0; x < w16; x++)
                                {
                                    int idx = (int)(((long)buf16[ofs + x] * scaleMul + scaleAdd) >> 16);
                                    drow[x] = (idx <= 0 || idx >= 255) ? 0xFF000000u : _palette256[idx];
                                }
                            });
                        }
                    }
                    finally { dst.UnlockBits(bd); }

                    dst.Save(dstPath, System.Drawing.Imaging.ImageFormat.Png);
                }
                return true;
            }

            // 2) 8-bit 경로 (luma)
            if (!LoadAs8BitLuma(srcPath, out var buf8, out int w, out int h))
                return false;

            // LUT 캐시 활용 (min/max가 동일하면 재사용)
            var lut = GetOrBuildLut8(min, max);

            using (var dst = new Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format32bppArgb))
            {
                var rect = new Rectangle(0, 0, w, h);
                var bd = dst.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, dst.PixelFormat);
                try
                {
                    unsafe
                    {
                        Parallel.For(0, h, y =>
                        {
                            uint* drow = (uint*)((byte*)bd.Scan0 + y * bd.Stride); // ← 각 스레드 지역 변수
                            int ofs = y * w;                                    // ← 각 스레드 지역 변수
                            for (int x = 0; x < w; x++)
                            {
                                drow[x] = lut[buf8[ofs + x]];
                            }
                        });
                    }
                }
                finally { dst.UnlockBits(bd); }

                dst.Save(dstPath, System.Drawing.Imaging.ImageFormat.Png);
            }
            return true;
        }

        private bool LoadAs8BitLuma(string path, out byte[] buf, out int w, out int h)
        {
            buf = null; w = 0; h = 0;

            try
            {
                using (var src = new Bitmap(path))
                {
                    // 람다에서 캡처하지 않을 로컬 변수
                    int width = src.Width;
                    int height = src.Height;
                    var tmp = new byte[width * height];

                    var rect = new Rectangle(0, 0, width, height);
                    var bd = src.LockBits(rect, ImageLockMode.ReadOnly, src.PixelFormat);

                    try
                    {
                        int bpp = Image.GetPixelFormatSize(src.PixelFormat) / 8;
                        if (bpp < 1) return false;

                        unsafe
                        {
                            byte* sbase = (byte*)bd.Scan0;

                            if (bpp == 1)
                            {
                                // 8bpp 인덱스/그레이: 그대로 복사
                                for (int y = 0; y < height; y++)
                                {
                                    byte* srow = sbase + y * bd.Stride;
                                    int ofs = y * width;
                                    for (int x = 0; x < width; x++)
                                        tmp[ofs + x] = srow[x];
                                }
                            }
                            else
                            {
                                // 24/32bpp 등: BGR[ A ] 가정 → BT.601 luma
                                Parallel.For(0, height, y =>
                                {
                                    byte* srow = sbase + y * bd.Stride;
                                    int ofs = y * width;

                                    for (int x = 0; x < width; x++)
                                    {
                                        int p = x * bpp; // 픽셀 시작 인덱스
                                        byte b = srow[p + 0];
                                        byte g = srow[p + 1];
                                        byte r = srow[p + 2];
                                        tmp[ofs + x] = (byte)((r * 77 + g * 150 + b * 29 + 128) >> 8);
                                    }
                                });
                            }
                        }
                    }
                    finally
                    {
                        src.UnlockBits(bd);
                    }

                    // 병렬 처리 후 out 변수에 대입 (람다 밖)
                    buf = tmp;
                    w = width;
                    h = height;
                    return true;
                }
            }
            catch
            {
                buf = null; w = h = 0;
                return false;
            }
        }

        private static string GetDesktopResultFolder()
        {
            string desktop = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            if (string.IsNullOrEmpty(desktop))
                desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop); // 호환 폴백

            string dir = Path.Combine(desktop, "ResultColorMap");
            try { Directory.CreateDirectory(dir); } catch { /* 이미 존재/권한 이슈 무시 */ }
            return dir;
        }

        // Ctrl+U 핫키 핸들러

        // 8-이웃 팽창: radius=1 -> 상하좌우+대각 1픽셀 확장
        private static bool[,] DilateMaskDirectional(bool[,] src, int left, int right, int top, int bottom)
        {
            int h = src.GetLength(0);
            int w = src.GetLength(1);
            bool[,] dst = new bool[h, w];

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if (!src[y, x]) continue;

                    // 확장/축소 대상 범위 계산
                    int x0 = Math.Max(0, x + left);
                    int x1 = Math.Min(w - 1, x + right);
                    int y0 = Math.Max(0, y + top);
                    int y1 = Math.Min(h - 1, y + bottom);

                    for (int yy = y0; yy <= y1; yy++)
                    {
                        for (int xx = x0; xx <= x1; xx++)
                        {
                            dst[yy, xx] = true;
                        }
                    }
                }
            }
            return dst;
        }
        // (h,w) → (h+2, w+2)로 False 패딩한 새 마스크 생성
        private static bool[,] PadFalseBorder(bool[,] src)
        {
            int h = src.GetLength(0), w = src.GetLength(1);
            bool[,] dst = new bool[h + 2, w + 2];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    dst[y + 1, x + 1] = src[y, x];
            return dst;
        }

        // 패딩 좌표를 원래 이미지 좌표로 되돌림 (pad=1이면 -1,-1 이동)
        private static List<List<PointF>> UnpadContours(List<List<PointF>> contours, float padX, float padY)
        {
            var outList = new List<List<PointF>>(contours.Count);
            for (int i = 0; i < contours.Count; i++)
            {
                var c = contours[i];
                if (c == null) continue;
                var cc = new List<PointF>(c.Count);
                for (int k = 0; k < c.Count; k++)
                    cc.Add(new PointF(c[k].X - padX, c[k].Y - padY));
                outList.Add(cc);
            }
            return outList;
        }

        private void OnAutoLabelFromColormap_LessThanThreshold(int minAreaPixels = 0)
        {
            Task.Run(() =>
            {
                try
                {
                    if (_canvas == null || _canvas.Image == null)
                    {
                        this.BeginInvoke((Action)(() => AddLog("⚠️ 이미지가 없습니다.")));
                        return;
                    }

                    // ========================= 1) 원본 경로 확인 =========================
                    if (string.IsNullOrEmpty(_currentImagePath))
                    {
                        // 원본 경로가 없으면 WIC로 비트깊이 판정 불가 → 8bit로만 처리
                        // (필요하면 여기서 바로 return 해도 되고, 일단 8bit로 진행)
                        AddLog("ℹ️ _currentImagePath 없음 → 8bit 그레이 기준으로 레이블링합니다.");
                    }

                    bool use16 = false;
                    int w = 0, h = 0;
                    ushort[] buf16 = null;
                    byte[] buf8 = null;
                    Bitmap gray8 = null;

                    // ========================= 2) WIC로 Gray16 시도 =========================
                    if (!string.IsNullOrEmpty(_currentImagePath) &&
                        TryLoadGray16WithWIC(_currentImagePath, out buf16, out w, out h))
                    {
                        // ★ 여기서가 핵심: "원본 파일을 WIC로 Gray16으로 읽을 수 있냐 없냐"로 결정
                        use16 = true;

                        // 캔버스 이미지와 크기가 다르면 좌표가 어긋날 수 있으므로 체크
                        if (w != _canvas.Image.Width || h != _canvas.Image.Height)
                        {
                            AddLog($"⚠️ Gray16 로드 크기({w}x{h})와 Canvas 크기({_canvas.Image.Width}x{_canvas.Image.Height})가 달라 8bit 기준으로 다시 처리합니다.");
                            use16 = false;
                            buf16 = null;
                        }
                    }

                    // ========================= 3) 8bit 버퍼 준비 (fallback 포함) =========================
                    if (!use16)
                    {
                        // 캔버스 이미지를 8bit 그레이로 변환
                        gray8 = GetCurrentImageGray8();
                        if (gray8 == null)
                        {
                            this.BeginInvoke((Action)(() => AddLog("⚠️ 8bit 그레이 변환 실패.")));
                            return;
                        }

                        w = gray8.Width;
                        h = gray8.Height;
                        buf8 = LockBitsToArray(gray8); // 너가 이미 쓰던 helper
                        if (buf8 == null || buf8.Length < w * h)
                        {
                            gray8.Dispose();
                            this.BeginInvoke((Action)(() => AddLog("⚠️ 8bit 버퍼 길이가 이미지 크기와 맞지 않습니다.")));
                            return;
                        }
                    }
                    else
                    {
                        // 16bit 경로에서 w, h, buf16은 이미 세팅되어 있음
                        if (buf16 == null || buf16.Length < w * h)
                        {
                            this.BeginInvoke((Action)(() => AddLog("⚠️ 16bit 버퍼 길이가 이미지 크기와 맞지 않습니다.")));
                            return;
                        }
                    }

                    // ========================= 4) 바이너리 마스크 생성 =========================
                    bool[,] bin = new bool[h, w];
                    int trueCount = 0;

                    if (use16)
                    {
                        // ----- 16bit: WIC Gray16 기준 -----
                        int idx = 0;
                        for (int y = 0; y < h; y++)
                        {
                            for (int x = 0; x < w; x++, idx++)
                            {
                                if (idx >= buf16.Length) break;
                                ushort v = buf16[idx]; // 0~65535
                                bool isFg = (v >= COLORMAP_TARGET_START_VALUE_16 && v <= COLORMAP_TARGET_END_VALUE_16);
                                bin[y, x] = isFg;
                                if (isFg) trueCount++;
                            }
                        }
                    }
                    else
                    {
                        // ----- 8bit: GetCurrentImageGray8() 기준 -----
                        int idx = 0;
                        for (int y = 0; y < h; y++)
                        {
                            for (int x = 0; x < w; x++, idx++)
                            {
                                byte v = buf8[idx]; // 0~255
                                bool isFg = (v >= COLORMAP_TARGET_START_VALUE_8 && v <= COLORMAP_TARGET_END_VALUE_8);
                                bin[y, x] = isFg;
                                if (isFg) trueCount++;
                            }
                        }
                    }

                    if (gray8 != null)
                    {
                        gray8.Dispose();
                        gray8 = null;
                    }

                    // ========================= 5) fill ratio 체크 =========================
                    double fillRatio = (double)trueCount / (w * h);
                    if (fillRatio > 0.95)
                    {
                        this.BeginInvoke((Action)(() =>
                            AddLog("⚠️ 거의 전체가 foreground로 판정되어 레이블링을 건너뜁니다. (ratio=" +
                                   fillRatio.ToString("0.000") + ")")));
                        return;
                    }

                    // ========================= 6) 마스크 팽창 + 시프트 =========================
                    bin = DilateMaskDirectionalShifted(
                        bin,
                        left: -1,
                        right: 0,
                        top: 0,
                        bottom: 5,
                        shiftX: +1,
                        shiftY: 0
                    );

                    // ========================= 7) 패딩 → 컨투어 추출 =========================
                    bool[,] binPadded = PadFalseBorder(bin);
                    List<List<PointF>> contoursPadded = TraceContours(binPadded);
                    List<List<PointF>> contours = UnpadContours(contoursPadded, 1f, 1f);

                    if (contours == null || contours.Count == 0)
                    {
                        this.BeginInvoke((Action)(() => AddLog("ℹ️ 감지된 컨투어가 없습니다.")));
                        return;
                    }

                    // ========================= 8) 노이즈 필터 =========================
                    List<List<PointF>> filtered = new List<List<PointF>>(contours.Count);
                    foreach (var c in contours)
                    {
                        if (c == null) continue;

                        int n = c.Count;
                        if (n < 10) continue;     // 너무 적은 점수(이건 그대로 둬도 좋음)
                        if (n > 20000) continue;  // 비정상적으로 큰 컨투어 방어

                        // --- 면적 계산 (Shoelace formula) ---
                        double area = Math.Abs(ComputePolygonArea(c)); // 픽셀^2 단위 정도

                        // minAreaPixels가 0보다 크면, 그 값보다 작은 라벨은 버림
                        if (minAreaPixels > 0 && area < minAreaPixels)
                            continue;

                        filtered.Add(c);
                    }

                    // ========================= 9) 폴리곤 생성 및 추가 =========================
                    int added = 0;
                    foreach (List<PointF> polyPts in filtered)
                    {
                        if (CreatePolygonAndAdd(polyPts))
                            added++;
                    }

                    // ========================= 10) UI 갱신 =========================
                    this.BeginInvoke((Action)(() =>
                    {
                        _canvas.Invalidate();
                        string modeText = use16
                            ? "16bit(WIC Gray16) 기준"
                            : "8bit(Gray 변환) 기준";

                        AddLog($"Ctrl+U → {added}개 라벨링 완료 ({modeText})");
                    }));
                }
                catch (Exception ex)
                {
                    this.BeginInvoke((Action)(() =>
                        AddLog("❌ Ctrl+U 실패: " + ex.Message)));
                }
            });
        }


        private static double ComputePolygonArea(IList<PointF> pts)
        {
            int n = pts.Count;
            if (n < 3) return 0.0;

            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                var p0 = pts[i];
                var p1 = pts[(i + 1) % n];
                sum += (double)p0.X * p1.Y - (double)p1.X * p0.Y;
            }
            return 0.5 * sum; // 호출할 때 Math.Abs()로 부호 제거
        }


        /// <summary>
        /// 마스크를 방향별로 확장/축소하고, 전체를 지정한 만큼 평행이동한다.
        /// left/right/top/bottom: 각 방향 확장(+)/축소(-)
        /// shiftX/shiftY: 전체 마스크 이동 (양수→오른쪽/아래쪽, 음수→왼쪽/위쪽)
        /// </summary>
        private static bool[,] DilateMaskDirectionalShifted(
            bool[,] src,
            int left, int right, int top, int bottom,
            int shiftX, int shiftY)
        {
            int h = src.GetLength(0);
            int w = src.GetLength(1);
            bool[,] tmp = new bool[h, w];

            // 1️⃣ 먼저 방향별 팽창/축소
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if (!src[y, x]) continue;

                    int x0 = Math.Max(0, x + left);
                    int x1 = Math.Min(w - 1, x + right);
                    int y0 = Math.Max(0, y + top);
                    int y1 = Math.Min(h - 1, y + bottom);

                    for (int yy = y0; yy <= y1; yy++)
                    {
                        for (int xx = x0; xx <= x1; xx++)
                        {
                            tmp[yy, xx] = true;
                        }
                    }
                }
            }

            // 2️⃣ 전체 평행 이동
            bool[,] dst = new bool[h, w];
            for (int y = 0; y < h; y++)
            {
                int yy = y + shiftY;
                if (yy < 0 || yy >= h) continue;

                for (int x = 0; x < w; x++)
                {
                    int xx = x + shiftX;
                    if (xx < 0 || xx >= w) continue;

                    if (tmp[y, x])
                        dst[yy, xx] = true;
                }
            }

            return dst;
        }
        private Bitmap GetCurrentImageGray()
        {
            if (_canvas == null || _canvas.Image == null)
                return null;

            if (!(_canvas.Image is Bitmap src))
                return null;

            PixelFormat fmt = src.PixelFormat;
            int bpp = Image.GetPixelFormatSize(fmt);

            if (fmt == PixelFormat.Format8bppIndexed ||
                fmt == PixelFormat.Format16bppGrayScale ||
                fmt == PixelFormat.Format32bppArgb)
            {
                return (Bitmap)src.Clone();
            }

            return GetCurrentImageGray8();  // 이미 구현돼 있다고 가정
        }

        // ===== 현재 캔버스 이미지 → 8비트 그레이 변환 =====
        private Bitmap GetCurrentImageGray8()
        {
            Bitmap src = _canvas.Image as Bitmap;
            if (src == null) return null;

            // 24bpp로 그린 뒤 Gray로 변환(플랫폼 호환성/안정성 우선)
            Bitmap tmp = new Bitmap(src.Width, src.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(tmp)) g.DrawImage(src, 0, 0, src.Width, src.Height);

            Bitmap gray = new Bitmap(src.Width, src.Height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            SetGrayPalette(gray);

            Rectangle rect = new Rectangle(0, 0, tmp.Width, tmp.Height);
            BitmapData bd24 = tmp.LockBits(rect, ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            BitmapData bd8 = gray.LockBits(rect, ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);

            try
            {
                unsafe
                {
                    byte* p24 = (byte*)bd24.Scan0;
                    byte* p8 = (byte*)bd8.Scan0;
                    int s24 = bd24.Stride;
                    int s8 = bd8.Stride;

                    for (int y = 0; y < tmp.Height; y++)
                    {
                        byte* row24 = p24 + y * s24;
                        byte* row8 = p8 + y * s8;
                        for (int x = 0; x < tmp.Width; x++)
                        {
                            byte b = row24[x * 3 + 0];
                            byte g2 = row24[x * 3 + 1];
                            byte r = row24[x * 3 + 2];
                            int yv = (r * 299 + g2 * 587 + b * 114 + 500) / 1000; // BT.601 근사
                            if (yv < 0) yv = 0; else if (yv > 255) yv = 255;
                            row8[x] = (byte)yv;
                        }
                    }
                }
            }
            finally
            {
                tmp.UnlockBits(bd24);
                gray.UnlockBits(bd8);
                tmp.Dispose();
            }
            return gray;
        }

        private static void SetGrayPalette(Bitmap bmp8)
        {
            ColorPalette pal = bmp8.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp8.Palette = pal;
        }

        private static byte[] LockBitsToArray(Bitmap gray8)
        {
            Rectangle rect = new Rectangle(0, 0, gray8.Width, gray8.Height);
            BitmapData bd = gray8.LockBits(rect, ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            try
            {
                int h = gray8.Height, w = gray8.Width;
                byte[] buf = new byte[h * w];
                int stride = bd.Stride;
                unsafe
                {
                    byte* p = (byte*)bd.Scan0;
                    int dst = 0;
                    for (int y = 0; y < h; y++)
                    {
                        Marshal.Copy(new IntPtr(p + y * stride), buf, dst, w);
                        dst += w;
                    }
                }
                return buf;
            }
            finally
            {
                gray8.UnlockBits(bd);
            }
        }

        // ===== 외곽선(모어 이웃) =====
        private static readonly int[,] Moore = new int[,]
        {
    {  1,  0 }, { 1,  1 }, { 0,  1 }, { -1,  1 },
    { -1, 0 }, { -1,-1 }, { 0, -1 }, {  1, -1 }
        };

        private static List<List<PointF>> TraceContours(bool[,] bin)
        {
            int h = bin.GetLength(0), w = bin.GetLength(1);
            bool[,] visited = new bool[h, w];
            List<List<PointF>> contours = new List<List<PointF>>();

            for (int y = 1; y < h - 1; y++)
            {
                for (int x = 1; x < w - 1; x++)
                {
                    if (!bin[y, x] || visited[y, x]) continue;

                    bool isBorder = false;
                    for (int k = 0; k < 8; k++)
                    {
                        int nx = x + Moore[k, 0];
                        int ny = y + Moore[k, 1];
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h || !bin[ny, nx])
                        {
                            isBorder = true; break;
                        }
                    }
                    if (!isBorder) { visited[y, x] = true; continue; }

                    List<PointF> c = FollowBorder(bin, visited, x, y);
                    if (c != null && c.Count >= 3) contours.Add(c);
                }
            }
            return contours;
        }

        private static List<PointF> FollowBorder(bool[,] bin, bool[,] visited, int sx, int sy)
        {
            int h = bin.GetLength(0), w = bin.GetLength(1);
            List<PointF> pts = new List<PointF>(128);

            int cx = sx, cy = sy;
            int dir = 0; // 진행 시작각

            pts.Add(new PointF(cx + 0.5f, cy + 0.5f));
            visited[cy, cx] = true;

            int maxIter = w * h * 4;
            int iter = 0;

            while (iter++ < maxIter)
            {
                bool moved = false;
                for (int i = 0; i < 8; i++)
                {
                    int k = (dir + i) % 8;
                    int nx = cx + Moore[k, 0];
                    int ny = cy + Moore[k, 1];
                    if (nx <= 0 || ny <= 0 || nx >= w - 1 || ny >= h - 1) continue;

                    if (bin[ny, nx])
                    {
                        if (pts.Count == 1 || pts[pts.Count - 1].X != nx + 0.5f || pts[pts.Count - 1].Y != ny + 0.5f)
                            pts.Add(new PointF(nx + 0.5f, ny + 0.5f));

                        cx = nx; cy = ny;
                        visited[cy, cx] = true;

                        dir = (k + 6) % 8; // 내부를 타이트하게
                        moved = true;
                        break;
                    }
                }
                if (!moved) break;
                if (cx == sx && cy == sy && pts.Count > 3) break;
            }

            if (pts.Count >= 3)
            {
                if (pts[0] != pts[pts.Count - 1]) pts.Add(pts[0]);
                return pts;
            }
            return null;
        }

        // ===== 기하 유틸 =====
        private static float PolygonArea(List<PointF> p)
        {
            double a = 0;
            for (int i = 0; i < p.Count - 1; i++)
                a += (double)p[i].X * p[i + 1].Y - (double)p[i + 1].X * p[i].Y;
            return (float)(a * 0.5);
        }
        private static float PolyPerimeter(List<PointF> p)
        {
            double s = 0;
            for (int i = 0; i < p.Count - 1; i++)
            {
                double dx = p[i + 1].X - p[i].X;
                double dy = p[i + 1].Y - p[i].Y;
                s += Math.Sqrt(dx * dx + dy * dy);
            }
            return (float)s;
        }

        // ===== RDP 단순화 =====
        private static List<PointF> RdpSimplify(List<PointF> pts, float eps)
        {
            if (pts == null || pts.Count <= 3) return pts;

            bool closed = (pts[0] == pts[pts.Count - 1]);
            List<PointF> src = closed ? new List<PointF>(pts.GetRange(0, pts.Count - 1)) : new List<PointF>(pts);

            bool[] keep = new bool[src.Count];
            for (int i = 0; i < keep.Length; i++) keep[i] = false;
            keep[0] = true; keep[src.Count - 1] = true;

            RdpRec(src, 0, src.Count - 1, eps, keep);

            List<PointF> outp = new List<PointF>();
            for (int i = 0; i < src.Count; i++) if (keep[i]) outp.Add(src[i]);
            if (closed && outp.Count > 0) outp.Add(outp[0]);
            return outp;
        }
        private static void RdpRec(List<PointF> pts2, int s, int e, float eps2, bool[] keep2)
        {
            float dmax = 0f; int idx = -1;
            PointF A = pts2[s]; PointF B = pts2[e];

            for (int i = s + 1; i < e; i++)
            {
                float d = PerpDist(pts2[i], A, B);
                if (d > dmax) { dmax = d; idx = i; }
            }
            if (dmax > eps2 && idx > 0)
            {
                keep2[idx] = true;
                RdpRec(pts2, s, idx, eps2, keep2);
                RdpRec(pts2, idx, e, eps2, keep2);
            }
        }
        private static float PerpDist(PointF P, PointF A, PointF B)
        {
            float dx = B.X - A.X, dy = B.Y - A.Y;
            if (dx == 0f && dy == 0f) return CalcDist(P, A);
            float t = ((P.X - A.X) * dx + (P.Y - A.Y) * dy) / (dx * dx + dy * dy);
            float xx = A.X + t * dx, yy = A.Y + t * dy;
            return CalcDist(P, new PointF(xx, yy));
        }
        private static float CalcDist(PointF a, PointF b)
        {
            float dx = a.X - b.X, dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        // ===== 실제 폴리곤 추가(프로젝트 API에 맞게 필요 시 수정) =====
        private bool CreatePolygonAndAdd(List<PointF> ptsImg)
        {
            try
            {
                if (_canvas == null || ptsImg == null || ptsImg.Count < 3)
                    return false;

                // PolygonShape는 IEnumerable<PointF> 생성자를 가짐
                var poly = new PolygonShape(ptsImg);
                poly.LabelName = _labelWin.LabelName;   // (프로젝트에 맞게) 현재 활성 라벨명 할당
                poly.StrokeColor = _labelWin.SelectedColor;

                if (_canvas.Shapes == null)
                    return false;

                _canvas.Shapes.Add(poly);
                return true;
            }
            catch
            {
                return false;
            }
        }



        #endregion

        #endregion

        private void MainForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (_aiRainbowBg != null) { _aiRainbowBg.Dispose(); _aiRainbowBg = null; }
            DisposeCurrentModel();
        }
    }
}