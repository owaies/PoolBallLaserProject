"""
Graphical User Interface for PoolBot Laser Calibration System.

A complete Tkinter desktop application featuring:
- Virtual pool table drawn to exact scale & aspect ratio with interactive target selection.
- Real-time pan/tilt prediction using replaceable interpolation.
- Safety controls (Simulation mode, Soft limits, Explicit move confirmation, STOP).
- ESP32 hardware integration with verified endpoints.
- Calibration node management (CRUD + JSON save/load).
- Phone-camera image verification with 4-corner homography and laser spot measurement.
- Session error statistics (ex, ey, Euclidean, Mean, Median, Max, RMS) and CSV export.
"""

import math
import os
import sys
import threading
import time
from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw

from laser_calibration.table_geometry import TABLE, TableCanvasTransformer, get_pocket_locations
from laser_calibration.calibration_data import CalibrationDataset, CalibrationNode, get_default_nine_samples
from laser_calibration.interpolator import (
    BaseAngleInterpolator,
    BilinearGridInterpolator,
    ScatteredInterpolator,
    GeometricModelInterpolator,
)
from laser_calibration.esp32_client import ESP32Client, ServoSoftLimits, ESP32Status
from laser_calibration.camera_homography import CameraHomography, ImageCornerSelection, detect_red_laser_spot
from laser_calibration.spot_measurement import MeasurementSession, SpotMeasurementRecord


# Theme Colors
BG_DARK = "#0f172a"
BG_CARD = "#1e293b"
BG_CARD_LIGHT = "#334155"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_BLUE = "#0284c7"
ACCENT_BLUE_HOVER = "#0369a1"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
ACCENT_AMBER = "#f59e0b"
FELT_COLOR = "#0c5432"
FELT_BORDER = "#073b22"
RAIL_WOOD = "#5c3317"
POCKET_COLOR = "#090d16"


class PoolBotCalibrationApp(tk.Tk):
    def __init__(self, default_json_path: Optional[str] = None):
        super().__init__()

        self.title("PoolBot AI Laser Calibration System")
        self.geometry("1380x880")
        self.minsize(1100, 750)
        self.configure(bg=BG_DARK)

        # File paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_json_path = default_json_path or os.path.join(base_dir, "calibration_points.json")

        # Core State
        self.dataset = self._load_initial_dataset()
        self.interpolators = {
            "Bilinear Grid": BilinearGridInterpolator(self.dataset.nodes),
            "Scattered IDW": ScatteredInterpolator(self.dataset.nodes),
            "Geometric Kinematic": GeometricModelInterpolator(),
        }
        self.current_interpolator_name = "Bilinear Grid"
        self.active_interpolator: BaseAngleInterpolator = self.interpolators[self.current_interpolator_name]

        # Hardware Client
        self.soft_limits = ServoSoftLimits(pan_min=15, pan_max=175, tilt_min=25, tilt_max=155)
        self.esp32 = ESP32Client(
            base_url="http://192.168.4.1",
            timeout=2.0,
            simulation_mode=True,  # Default to SIMULATION for safety!
            soft_limits=self.soft_limits,
        )

        # Phone Camera & Homography State
        self.camera_homography = CameraHomography()
        self.current_image_path: Optional[str] = None
        self.raw_pil_image: Optional[Image.Image] = None
        self.display_pil_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.corner_points: List[Tuple[float, float]] = []  # Image pixels
        self.corner_step_index = 0
        self.measuring_spot_mode = False
        self.img_scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

        # Selected Target on Virtual Table
        self.target_x_mm: float = TABLE.INNER_LENGTH_MM / 2.0
        self.target_y_mm: float = TABLE.INNER_BREADTH_MM / 2.0
        self.predicted_pan: float = 125.0
        self.predicted_tilt: float = 84.0
        self.prediction_info: dict = {}

        # Session & Logging
        self.session = MeasurementSession()

        # Canvas Transformer
        self.table_transformer = TableCanvasTransformer(canvas_width=720, canvas_height=420)

        # Setup GUI Components
        self._configure_styles()
        self._build_header()
        self._build_main_layout()
        self._build_status_bar()

        # Initial prediction
        self._update_prediction()

        # Start periodic background status polling
        self.running = True
        self.poll_thread = threading.Thread(target=self._esp32_polling_loop, daemon=True)
        self.poll_thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_initial_dataset(self) -> CalibrationDataset:
        if os.path.exists(self.default_json_path):
            try:
                return CalibrationDataset.load_from_file(self.default_json_path)
            except Exception as e:
                print(f"Error loading {self.default_json_path}: {e}")
        return CalibrationDataset()

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=TEXT_MAIN, font=("Segoe UI", 10))
        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=BG_CARD,
            foreground=TEXT_MUTED,
            padding=[16, 8],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", BG_CARD_LIGHT), ("active", "#2d3748")],
            foreground=[("selected", TEXT_MAIN), ("active", TEXT_MAIN)],
        )

        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Treeview",
            background=BG_CARD,
            foreground=TEXT_MAIN,
            fieldbackground=BG_CARD,
            rowheight=26,
            font=("Segoe UI", 9)
        )
        style.configure("Treeview.Heading",
            background=BG_CARD_LIGHT,
            foreground=TEXT_MAIN,
            font=("Segoe UI", 9, "bold")
        )
        style.map("Treeview", background=[("selected", ACCENT_BLUE)])

    def _build_header(self):
        header = tk.Frame(self, bg=BG_CARD, height=64, padx=16, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)

        # Title & Subtitle
        title_frame = tk.Frame(header, bg=BG_CARD)
        title_frame.pack(side=tk.LEFT)
        lbl_title = tk.Label(
            title_frame,
            text="PoolBot Laser Calibration & Verification System",
            font=("Segoe UI", 14, "bold"),
            fg=TEXT_MAIN,
            bg=BG_CARD,
        )
        lbl_title.pack(anchor="w")
        lbl_sub = tk.Label(
            title_frame,
            text="AI-Based Pool Ball Identification & Laser Pan-Tilt Calibration",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        lbl_sub.pack(anchor="w")

        # Right header controls (Emergency Stop, Center, Status, Sim Mode)
        actions_frame = tk.Frame(header, bg=BG_CARD)
        actions_frame.pack(side=tk.RIGHT)

        # Simulation Mode Toggle Checkbox
        self.sim_mode_var = tk.BooleanVar(value=True)
        self.chk_sim = tk.Checkbutton(
            actions_frame,
            text="SIMULATION MODE",
            variable=self.sim_mode_var,
            command=self._on_toggle_simulation,
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_AMBER,
            bg=BG_CARD,
            activebackground=BG_CARD,
            activeforeground=ACCENT_AMBER,
            selectcolor=BG_DARK,
        )
        self.chk_sim.pack(side=tk.LEFT, padx=10)

        # ESP32 Status Badge
        self.lbl_esp_status = tk.Label(
            actions_frame,
            text="ESP32: [SIMULATED]",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD_LIGHT,
            padx=10,
            pady=4,
            relief=tk.FLAT,
        )
        self.lbl_esp_status.pack(side=tk.LEFT, padx=8)

        # Center Servos Button
        btn_center = tk.Button(
            actions_frame,
            text="CENTER (90°)",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_BLUE,
            fg="white",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_center_clicked,
        )
        btn_center.pack(side=tk.LEFT, padx=6)

        # EMERGENCY STOP Button
        btn_stop = tk.Button(
            actions_frame,
            text="EMERGENCY STOP",
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT_RED,
            fg="white",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            cursor="hand2",
            command=self._on_emergency_stop_clicked,
        )
        btn_stop.pack(side=tk.LEFT, padx=6)

    def _build_main_layout(self):
        main_paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_DARK, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left Container: Virtual Pool Table and Live Target Readout
        left_frame = tk.Frame(main_paned, bg=BG_CARD, padx=12, pady=12)
        main_paned.add(left_frame, minsize=520, width=680)

        # Table Section Header
        tbl_header = tk.Frame(left_frame, bg=BG_CARD)
        tbl_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            tbl_header,
            text="Virtual Playing Surface (Exact 2341.9 × 1168.4 mm Scale)",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_MAIN,
            bg=BG_CARD,
        ).pack(side=tk.LEFT)

        self.lbl_cursor_coords = tk.Label(
            tbl_header,
            text="Cursor: X: --- mm | Y: --- mm",
            font=("Consolas", 10),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        self.lbl_cursor_coords.pack(side=tk.RIGHT)

        # Pool Table Canvas
        self.table_canvas = tk.Canvas(
            left_frame,
            bg=BG_DARK,
            highlightthickness=1,
            highlightbackground=BG_CARD_LIGHT,
            cursor="crosshair",
        )
        self.table_canvas.pack(fill=tk.BOTH, expand=True)

        self.table_canvas.bind("<Configure>", self._on_table_canvas_resize)
        self.table_canvas.bind("<Motion>", self._on_table_mouse_move)
        self.table_canvas.bind("<Button-1>", self._on_table_click)

        # Target Coordinates & Predicted Angles readout banner
        target_banner = tk.Frame(left_frame, bg=BG_CARD_LIGHT, pady=8, padx=10)
        target_banner.pack(fill=tk.X, pady=(8, 0))

        # Target X/Y display
        col1 = tk.Frame(target_banner, bg=BG_CARD_LIGHT)
        col1.pack(side=tk.LEFT, padx=10)
        tk.Label(col1, text="Selected Target (xr, yr):", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD_LIGHT).pack(anchor="w")
        self.lbl_target_val = tk.Label(
            col1,
            text=f"X: {self.target_x_mm:.1f} mm | Y: {self.target_y_mm:.1f} mm",
            font=("Consolas", 11, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD_LIGHT,
        )
        self.lbl_target_val.pack(anchor="w")

        # Predicted Pan/Tilt display
        col2 = tk.Frame(target_banner, bg=BG_CARD_LIGHT)
        col2.pack(side=tk.LEFT, padx=20)
        tk.Label(col2, text="Predicted Servo Angles:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD_LIGHT).pack(anchor="w")
        self.lbl_pred_angles = tk.Label(
            col2,
            text=f"Pan: {self.predicted_pan:.1f}° | Tilt: {self.predicted_tilt:.1f}°",
            font=("Consolas", 12, "bold"),
            fg=ACCENT_BLUE,
            bg=BG_CARD_LIGHT,
        )
        self.lbl_pred_angles.pack(anchor="w")

        # Interpolation method info
        col3 = tk.Frame(target_banner, bg=BG_CARD_LIGHT)
        col3.pack(side=tk.RIGHT, padx=10)
        tk.Label(col3, text="Active Model:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD_LIGHT).pack(anchor="e")
        self.lbl_model_info = tk.Label(
            col3,
            text=self.current_interpolator_name,
            font=("Segoe UI", 10, "italic"),
            fg=TEXT_MAIN,
            bg=BG_CARD_LIGHT,
        )
        self.lbl_model_info.pack(anchor="e")

        # Right Container: Tabs for Servo Targeting, Calibration Nodes, Camera Homography, Test Logs
        right_frame = tk.Frame(main_paned, bg=BG_CARD)
        main_paned.add(right_frame, minsize=480)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Servo Control & Targeting
        self.tab_servo = tk.Frame(self.notebook, bg=BG_CARD, padx=14, pady=12)
        self.notebook.add(self.tab_servo, text="  Servo Control  ")
        self._build_tab_servo()

        # Tab 2: Calibration Nodes
        self.tab_nodes = tk.Frame(self.notebook, bg=BG_CARD, padx=14, pady=12)
        self.notebook.add(self.tab_nodes, text="  Calibration Nodes  ")
        self._build_tab_nodes()

        # Tab 3: Phone Camera Verification
        self.tab_camera = tk.Frame(self.notebook, bg=BG_CARD, padx=14, pady=12)
        self.notebook.add(self.tab_camera, text="  Camera Verification  ")
        self._build_tab_camera()

        # Tab 4: Results & CSV Export
        self.tab_results = tk.Frame(self.notebook, bg=BG_CARD, padx=14, pady=12)
        self.notebook.add(self.tab_results, text="  Test Results & CSV  ")
        self._build_tab_results()

    def _build_status_bar(self):
        self.status_bar = tk.Frame(self, bg="#0b1120", height=24, padx=12, pady=3)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status_msg = tk.Label(
            self.status_bar,
            text="Ready. Simulation mode enabled. Servos safe.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg="#0b1120",
        )
        self.lbl_status_msg.pack(side=tk.LEFT)

        self.lbl_limits_msg = tk.Label(
            self.status_bar,
            text=f"Soft Limits: Pan [{self.soft_limits.pan_min}°-{self.soft_limits.pan_max}°] | Tilt [{self.soft_limits.tilt_min}°-{self.soft_limits.tilt_max}°]",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg="#0b1120",
        )
        self.lbl_limits_msg.pack(side=tk.RIGHT)

    # =========================================================================
    # TAB 1: SERVO CONTROL & TARGETING
    # =========================================================================
    def _build_tab_servo(self):
        # Frame for Manual Target Coordinates Entry
        grp_target = tk.LabelFrame(
            self.tab_servo,
            text="Target Position (mm)",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        grp_target.pack(fill=tk.X, pady=(0, 10))

        t_row = tk.Frame(grp_target, bg=BG_CARD)
        t_row.pack(fill=tk.X)

        tk.Label(t_row, text="X (mm):", bg=BG_CARD, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 4))
        self.ent_x = tk.Entry(t_row, width=8, font=("Consolas", 10))
        self.ent_x.insert(0, f"{self.target_x_mm:.1f}")
        self.ent_x.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(t_row, text="Y (mm):", bg=BG_CARD, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 4))
        self.ent_y = tk.Entry(t_row, width=8, font=("Consolas", 10))
        self.ent_y.insert(0, f"{self.target_y_mm:.1f}")
        self.ent_y.pack(side=tk.LEFT, padx=(0, 14))

        btn_apply_coords = tk.Button(
            t_row,
            text="Apply Target",
            command=self._on_manual_coords_applied,
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
        )
        btn_apply_coords.pack(side=tk.LEFT)

        # Commanded Servos Group
        grp_cmd = tk.LabelFrame(
            self.tab_servo,
            text="Command Servos",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        grp_cmd.pack(fill=tk.X, pady=(0, 10))

        # Confirmation Checkbox (Safety requirement)
        self.var_confirm_move = tk.BooleanVar(value=True)
        self.chk_confirm = tk.Checkbutton(
            grp_cmd,
            text="Require explicit confirmation before physical movement",
            variable=self.var_confirm_move,
            bg=BG_CARD,
            fg=TEXT_MAIN,
            selectcolor=BG_DARK,
            activebackground=BG_CARD,
            activeforeground=TEXT_MAIN,
            font=("Segoe UI", 9),
        )
        self.chk_confirm.pack(anchor="w", pady=(0, 8))

        # Main Command Button
        self.btn_send_servos = tk.Button(
            grp_cmd,
            text="SEND PREDICTED ANGLES TO SERVOS",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT_BLUE,
            fg="white",
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
            command=self._on_send_predicted_clicked,
        )
        self.btn_send_servos.pack(fill=tk.X, pady=(0, 6))

        # Feedback label
        self.lbl_cmd_feedback = tk.Label(
            grp_cmd,
            text="Servos ready.",
            font=("Segoe UI", 9, "italic"),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        self.lbl_cmd_feedback.pack(anchor="w")

        # Current ESP32 Telemetry Group
        grp_telemetry = tk.LabelFrame(
            self.tab_servo,
            text="Live ESP32 Telemetry",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        grp_telemetry.pack(fill=tk.X, pady=(0, 10))

        t_grid = tk.Frame(grp_telemetry, bg=BG_CARD)
        t_grid.pack(fill=tk.X)

        tk.Label(t_grid, text="Current Pan:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_telem_pan = tk.Label(t_grid, text="90°", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_BLUE)
        self.lbl_telem_pan.grid(row=0, column=1, sticky="w", padx=10, pady=2)

        tk.Label(t_grid, text="Current Tilt:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=2, sticky="w", pady=2)
        self.lbl_telem_tilt = tk.Label(t_grid, text="90°", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_BLUE)
        self.lbl_telem_tilt.grid(row=0, column=3, sticky="w", padx=10, pady=2)

        tk.Label(t_grid, text="Laser Status:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_telem_laser = tk.Label(t_grid, text="OFF", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_AMBER)
        self.lbl_telem_laser.grid(row=1, column=1, sticky="w", padx=10, pady=2)

        tk.Label(t_grid, text="Emergency Stop:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=2, sticky="w", pady=2)
        self.lbl_telem_stopped = tk.Label(t_grid, text="NORMAL", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_GREEN)
        self.lbl_telem_stopped.grid(row=1, column=3, sticky="w", padx=10, pady=2)

        # Jog Fine-Tuning Controls
        grp_jog = tk.LabelFrame(
            self.tab_servo,
            text="Manual Jog Controls (±1°, ±5°)",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        grp_jog.pack(fill=tk.X, pady=(0, 10))

        # Pan Jog
        j_pan_row = tk.Frame(grp_jog, bg=BG_CARD)
        j_pan_row.pack(fill=tk.X, pady=2)
        tk.Label(j_pan_row, text="Jog Pan: ", width=10, bg=BG_CARD, fg=TEXT_MUTED).pack(side=tk.LEFT)
        for delta in [-5, -1, 1, 5]:
            txt = f"{delta:+d}°"
            b = tk.Button(
                j_pan_row,
                text=txt,
                width=5,
                bg=BG_CARD_LIGHT,
                fg=TEXT_MAIN,
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda d=delta: self._jog_servo("pan", d),
            )
            b.pack(side=tk.LEFT, padx=3)

        # Tilt Jog
        j_tilt_row = tk.Frame(grp_jog, bg=BG_CARD)
        j_tilt_row.pack(fill=tk.X, pady=2)
        tk.Label(j_tilt_row, text="Jog Tilt:", width=10, bg=BG_CARD, fg=TEXT_MUTED).pack(side=tk.LEFT)
        for delta in [-5, -1, 1, 5]:
            txt = f"{delta:+d}°"
            b = tk.Button(
                j_tilt_row,
                text=txt,
                width=5,
                bg=BG_CARD_LIGHT,
                fg=TEXT_MAIN,
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda d=delta: self._jog_servo("tilt", d),
            )
            b.pack(side=tk.LEFT, padx=3)

        # Manual Laser Safety Control Group
        grp_laser = tk.LabelFrame(
            self.tab_servo,
            text="Manual Laser Safety Switch (OFF by Default)",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        grp_laser.pack(fill=tk.X)

        tk.Label(
            grp_laser,
            text="WARNING: Laser must remain OFF by default.\nNever point laser towards eyes or reflective surfaces.",
            font=("Segoe UI", 8),
            fg=ACCENT_AMBER,
            bg=BG_CARD,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 6))

        l_btn_row = tk.Frame(grp_laser, bg=BG_CARD)
        l_btn_row.pack(fill=tk.X)

        self.btn_laser_toggle = tk.Button(
            l_btn_row,
            text="ACTIVATE LASER (SAFETY ACK REQUIRED)",
            bg=BG_CARD_LIGHT,
            fg=ACCENT_AMBER,
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_toggle_laser_manual,
        )
        self.btn_laser_toggle.pack(side=tk.LEFT, padx=(0, 8))

        btn_laser_off = tk.Button(
            l_btn_row,
            text="TURN LASER OFF",
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_laser_off_forced,
        )
        btn_laser_off.pack(side=tk.LEFT)

    # =========================================================================
    # TAB 2: CALIBRATION NODES MANAGEMENT
    # =========================================================================
    def _build_tab_nodes(self):
        top_bar = tk.Frame(self.tab_nodes, bg=BG_CARD)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        # Model Selector
        tk.Label(top_bar, text="Interpolation Model:", bg=BG_CARD, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 6))
        self.cmb_model = ttk.Combobox(
            top_bar,
            values=list(self.interpolators.keys()),
            state="readonly",
            width=20,
        )
        self.cmb_model.set(self.current_interpolator_name)
        self.cmb_model.bind("<<ComboboxSelected>>", self._on_model_changed)
        self.cmb_model.pack(side=tk.LEFT, padx=(0, 14))

        # Reset button
        btn_reset = tk.Button(
            top_bar,
            text="Reset to 9 Defaults",
            command=self._on_reset_nodes_clicked,
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            relief=tk.FLAT,
            padx=8,
            pady=2,
            cursor="hand2",
        )
        btn_reset.pack(side=tk.RIGHT)

        # Treeview for Nodes
        cols = ("id", "label", "x_mm", "y_mm", "pan_deg", "tilt_deg", "notes")
        self.tree_nodes = ttk.Treeview(self.tab_nodes, columns=cols, show="headings", height=10)

        self.tree_nodes.heading("id", text="ID")
        self.tree_nodes.heading("label", text="Label")
        self.tree_nodes.heading("x_mm", text="X (mm)")
        self.tree_nodes.heading("y_mm", text="Y (mm)")
        self.tree_nodes.heading("pan_deg", text="Pan (°)")
        self.tree_nodes.heading("tilt_deg", text="Tilt (°)")
        self.tree_nodes.heading("notes", text="Notes")

        self.tree_nodes.column("id", width=60, anchor="center")
        self.tree_nodes.column("label", width=100, anchor="w")
        self.tree_nodes.column("x_mm", width=75, anchor="e")
        self.tree_nodes.column("y_mm", width=75, anchor="e")
        self.tree_nodes.column("pan_deg", width=65, anchor="e")
        self.tree_nodes.column("tilt_deg", width=65, anchor="e")
        self.tree_nodes.column("notes", width=150, anchor="w")

        tree_scroll = ttk.Scrollbar(self.tab_nodes, orient=tk.VERTICAL, command=self.tree_nodes.yview)
        self.tree_nodes.configure(yscrollcommand=tree_scroll.set)

        self.tree_nodes.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, before=self.tree_nodes)

        # Populate treeview
        self._refresh_nodes_treeview()

        # Action Buttons (Add, Edit, Delete, Save JSON, Load JSON)
        btn_bar = tk.Frame(self.tab_nodes, bg=BG_CARD)
        btn_bar.pack(fill=tk.X, pady=(10, 0))

        btn_add = tk.Button(
            btn_bar,
            text="+ Add Node",
            command=self._on_add_node_dialog,
            bg=ACCENT_BLUE,
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_add.pack(side=tk.LEFT, padx=(0, 6))

        btn_edit = tk.Button(
            btn_bar,
            text="Edit Selected",
            command=self._on_edit_node_dialog,
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_edit.pack(side=tk.LEFT, padx=6)

        btn_del = tk.Button(
            btn_bar,
            text="Delete Selected",
            command=self._on_delete_node_clicked,
            bg=BG_CARD_LIGHT,
            fg=ACCENT_RED,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_del.pack(side=tk.LEFT, padx=6)

        btn_save = tk.Button(
            btn_bar,
            text="Save to JSON",
            command=self._on_save_nodes_json,
            bg=ACCENT_GREEN,
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_save.pack(side=tk.RIGHT, padx=6)

        btn_load = tk.Button(
            btn_bar,
            text="Load JSON",
            command=self._on_load_nodes_json,
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_load.pack(side=tk.RIGHT, padx=6)

    # =========================================================================
    # TAB 3: PHONE CAMERA VERIFICATION & HOMOGRAPHY
    # =========================================================================
    def _build_tab_camera(self):
        # Guide banner
        self.lbl_camera_guide = tk.Label(
            self.tab_camera,
            text="Step 1: Load a photo of the table. Fixed camera position required.",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT_BLUE,
            bg=BG_CARD,
        )
        self.lbl_camera_guide.pack(anchor="w", pady=(0, 6))

        # Image action buttons
        img_btn_bar = tk.Frame(self.tab_camera, bg=BG_CARD)
        img_btn_bar.pack(fill=tk.X, pady=(0, 8))

        btn_load_img = tk.Button(
            img_btn_bar,
            text="1. Load Photo...",
            command=self._on_load_photo_clicked,
            bg=ACCENT_BLUE,
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_load_img.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_calib_corners = tk.Button(
            img_btn_bar,
            text="2. Mark 4 Inner Corners",
            command=self._on_start_corner_calibration,
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        self.btn_calib_corners.pack(side=tk.LEFT, padx=6)

        self.btn_pick_spot = tk.Button(
            img_btn_bar,
            text="3. Click Laser Spot on Photo",
            command=self._on_toggle_spot_pick_mode,
            bg=BG_CARD_LIGHT,
            fg=ACCENT_GREEN,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        self.btn_pick_spot.pack(side=tk.LEFT, padx=6)

        btn_autodetect_spot = tk.Button(
            img_btn_bar,
            text="Auto-Detect Red Spot",
            command=self._on_autodetect_spot_clicked,
            bg=BG_CARD_LIGHT,
            fg=TEXT_MAIN,
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        btn_autodetect_spot.pack(side=tk.LEFT, padx=6)

        # Image Canvas
        self.cam_canvas = tk.Canvas(
            self.tab_camera,
            bg=BG_DARK,
            highlightthickness=1,
            highlightbackground=BG_CARD_LIGHT,
            cursor="crosshair",
        )
        self.cam_canvas.pack(fill=tk.BOTH, expand=True)

        self.cam_canvas.bind("<Configure>", self._on_cam_canvas_resize)
        self.cam_canvas.bind("<Button-1>", self._on_cam_canvas_click)

        # Verification result panel
        res_card = tk.LabelFrame(
            self.tab_camera,
            text="Measured Spot Coordinates & Error (mm)",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=6,
        )
        res_card.pack(fill=tk.X, pady=(8, 0))

        r_grid = tk.Frame(res_card, bg=BG_CARD)
        r_grid.pack(fill=tk.X)

        tk.Label(r_grid, text="Measured Spot (xm, ym):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w")
        self.lbl_spot_measured = tk.Label(r_grid, text="--- mm, --- mm", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        self.lbl_spot_measured.grid(row=0, column=1, sticky="w", padx=10)

        tk.Label(r_grid, text="Requested Target (xr, yr):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=2, sticky="w")
        self.lbl_spot_requested = tk.Label(r_grid, text="--- mm, --- mm", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        self.lbl_spot_requested.grid(row=0, column=3, sticky="w", padx=10)

        tk.Label(r_grid, text="Error (ex, ey):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.lbl_spot_exy = tk.Label(r_grid, text="ex: --- mm | ey: --- mm", font=("Consolas", 10), bg=BG_CARD, fg=TEXT_MUTED)
        self.lbl_spot_exy.grid(row=1, column=1, sticky="w", padx=10, pady=(4, 0))

        tk.Label(r_grid, text="Euclidean Error (e):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=2, sticky="w", pady=(4, 0))
        self.lbl_spot_euclidean = tk.Label(r_grid, text="--- mm", font=("Consolas", 11, "bold"), bg=BG_CARD, fg=ACCENT_AMBER)
        self.lbl_spot_euclidean.grid(row=1, column=3, sticky="w", padx=10, pady=(4, 0))

        # Log measurement button
        self.btn_log_meas = tk.Button(
            res_card,
            text="Record Spot Measurement to Session Log",
            command=self._on_record_measurement_clicked,
            bg=ACCENT_GREEN,
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.btn_log_meas.pack(fill=tk.X, pady=(8, 2))

    # =========================================================================
    # TAB 4: RESULTS & CSV EXPORT
    # =========================================================================
    def _build_tab_results(self):
        # Summary statistics banner
        stats_frame = tk.LabelFrame(
            self.tab_results,
            text="Summary Error Statistics Across Session",
            bg=BG_CARD,
            fg=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
        )
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        s_grid = tk.Frame(stats_frame, bg=BG_CARD)
        s_grid.pack(fill=tk.X)

        tk.Label(s_grid, text="Test Samples:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w")
        self.lbl_stat_count = tk.Label(s_grid, text="0", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        self.lbl_stat_count.grid(row=0, column=1, sticky="w", padx=10)

        tk.Label(s_grid, text="Mean Error:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=2, sticky="w")
        self.lbl_stat_mean = tk.Label(s_grid, text="0.0 mm", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_BLUE)
        self.lbl_stat_mean.grid(row=0, column=3, sticky="w", padx=10)

        tk.Label(s_grid, text="Median Error:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=4, sticky="w")
        self.lbl_stat_median = tk.Label(s_grid, text="0.0 mm", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_BLUE)
        self.lbl_stat_median.grid(row=0, column=5, sticky="w", padx=10)

        tk.Label(s_grid, text="RMS Error:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.lbl_stat_rms = tk.Label(s_grid, text="0.0 mm", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_AMBER)
        self.lbl_stat_rms.grid(row=1, column=1, sticky="w", padx=10, pady=(4, 0))

        tk.Label(s_grid, text="Max Error:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=2, sticky="w", pady=(4, 0))
        self.lbl_stat_max = tk.Label(s_grid, text="0.0 mm", font=("Consolas", 10, "bold"), bg=BG_CARD, fg=ACCENT_RED)
        self.lbl_stat_max.grid(row=1, column=3, sticky="w", padx=10, pady=(4, 0))

        # Treeview for logged measurements
        cols = ("id", "timestamp", "target", "commanded", "measured", "ex", "ey", "euclidean")
        self.tree_results = ttk.Treeview(self.tab_results, columns=cols, show="headings", height=12)

        self.tree_results.heading("id", text="ID")
        self.tree_results.heading("timestamp", text="Time")
        self.tree_results.heading("target", text="Target (X, Y mm)")
        self.tree_results.heading("commanded", text="Commanded (Pan, Tilt)")
        self.tree_results.heading("measured", text="Measured (X, Y mm)")
        self.tree_results.heading("ex", text="ex (mm)")
        self.tree_results.heading("ey", text="ey (mm)")
        self.tree_results.heading("euclidean", text="Error e (mm)")

        self.tree_results.column("id", width=55, anchor="center")
        self.tree_results.column("timestamp", width=125, anchor="center")
        self.tree_results.column("target", width=120, anchor="w")
        self.tree_results.column("commanded", width=120, anchor="w")
        self.tree_results.column("measured", width=120, anchor="w")
        self.tree_results.column("ex", width=65, anchor="e")
        self.tree_results.column("ey", width=65, anchor="e")
        self.tree_results.column("euclidean", width=80, anchor="e")

        res_scroll = ttk.Scrollbar(self.tab_results, orient=tk.VERTICAL, command=self.tree_results.yview)
        self.tree_results.configure(yscrollcommand=res_scroll.set)

        self.tree_results.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        res_scroll.pack(side=tk.RIGHT, fill=tk.Y, before=self.tree_results)

        # Action Buttons
        res_btn_bar = tk.Frame(self.tab_results, bg=BG_CARD)
        res_btn_bar.pack(fill=tk.X, pady=(10, 0))

        btn_export = tk.Button(
            res_btn_bar,
            text="Export Session Results to CSV...",
            command=self._on_export_csv_clicked,
            bg=ACCENT_GREEN,
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        btn_export.pack(side=tk.LEFT)

        btn_clear = tk.Button(
            res_btn_bar,
            text="Clear Session Logs",
            command=self._on_clear_results_clicked,
            bg=BG_CARD_LIGHT,
            fg=ACCENT_RED,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
        )
        btn_clear.pack(side=tk.RIGHT)

    # =========================================================================
    # VIRTUAL POOL TABLE RENDERING & INTERACTION
    # =========================================================================
    def _on_table_canvas_resize(self, event):
        w = max(100, event.width)
        h = max(100, event.height)
        self.table_transformer.resize(w, h)
        self._redraw_virtual_table()

    def _redraw_virtual_table(self):
        c = self.table_canvas
        c.delete("all")

        # 1. Draw outer wood frame
        ox0, oy0, ox1, oy1 = self.table_transformer.get_outer_frame_rect()
        c.create_rectangle(ox0, oy0, ox1, oy1, fill=RAIL_WOOD, outline="#381d0c", width=2)

        # 2. Draw cushions / felt border
        ix0, iy0, ix1, iy1 = self.table_transformer.get_inner_felt_rect()
        c.create_rectangle(ix0 - 4, iy0 - 4, ix1 + 4, iy1 + 4, fill=FELT_BORDER, outline="")

        # 3. Draw inner playing felt
        c.create_rectangle(ix0, iy0, ix1, iy1, fill=FELT_COLOR, outline="#147a4b", width=1)

        # 4. Draw table grid lines (subtle dashed)
        # Center line
        cx, _ = self.table_transformer.table_mm_to_canvas_px(TABLE.INNER_LENGTH_MM / 2.0, 0)
        c.create_line(cx, iy0, cx, iy1, fill="#0f6b40", dash=(3, 3))
        _, cy = self.table_transformer.table_mm_to_canvas_px(0, TABLE.INNER_BREADTH_MM / 2.0)
        c.create_line(ix0, cy, ix1, cy, fill="#0f6b40", dash=(3, 3))

        # 5. Draw 6 pockets
        pockets = get_pocket_locations()
        for p in pockets:
            px, py = self.table_transformer.table_mm_to_canvas_px(p.x_mm, p.y_mm)
            r = p.radius_mm * self.table_transformer.scale
            c.create_oval(px - r, py - r, px + r, py + r, fill=POCKET_COLOR, outline="#2b3748", width=1)

        # 6. Draw calibration nodes
        for node in self.dataset.nodes:
            nx, ny = self.table_transformer.table_mm_to_canvas_px(node.x_mm, node.y_mm)
            # Pin circle
            c.create_oval(nx - 5, ny - 5, nx + 5, ny + 5, fill=ACCENT_AMBER, outline="white", width=1.5)
            # Label
            c.create_text(
                nx + 8,
                ny - 8,
                text=f"{node.label}\n({node.pan_deg:.0f}°, {node.tilt_deg:.0f}°)",
                anchor="sw",
                fill="white",
                font=("Segoe UI", 7, "bold"),
            )

        # 7. Draw selected target reticle
        tx, ty = self.table_transformer.table_mm_to_canvas_px(self.target_x_mm, self.target_y_mm)
        # Reticle circle
        c.create_oval(tx - 8, ty - 8, tx + 8, ty + 8, outline=ACCENT_RED, width=2)
        c.create_line(tx - 12, ty, tx + 12, ty, fill=ACCENT_RED, width=1.5)
        c.create_line(tx, ty - 12, tx, ty + 12, fill=ACCENT_RED, width=1.5)
        # Inner dot
        c.create_oval(tx - 2, ty - 2, tx + 2, ty + 2, fill=ACCENT_RED, outline="")

        # 8. Draw coordinate axes indicators at top-left
        c.create_text(
            ix0 + 6,
            iy0 + 6,
            text="Origin (0, 0)\n+X → Length (2341.9 mm)\n+Y ↓ Breadth (1168.4 mm)",
            anchor="nw",
            fill="#a1a1aa",
            font=("Consolas", 8),
        )

    def _on_table_mouse_move(self, event):
        x_mm, y_mm = self.table_transformer.canvas_px_to_table_mm(event.x, event.y)
        if (0 <= x_mm <= TABLE.INNER_LENGTH_MM) and (0 <= y_mm <= TABLE.INNER_BREADTH_MM):
            self.lbl_cursor_coords.config(text=f"Cursor: X: {x_mm:.1f} mm | Y: {y_mm:.1f} mm")
        else:
            self.lbl_cursor_coords.config(text="Cursor: [Outside Felt Playing Area]")

    def _on_table_click(self, event):
        x_mm, y_mm = self.table_transformer.canvas_px_to_table_mm(event.x, event.y)
        # Clamp to table bounds
        x_clamped = max(0.0, min(TABLE.INNER_LENGTH_MM, x_mm))
        y_clamped = max(0.0, min(TABLE.INNER_BREADTH_MM, y_mm))

        self.target_x_mm = round(x_clamped, 1)
        self.target_y_mm = round(y_clamped, 1)

        # Update entry inputs
        self.ent_x.delete(0, tk.END)
        self.ent_x.insert(0, f"{self.target_x_mm:.1f}")
        self.ent_y.delete(0, tk.END)
        self.ent_y.insert(0, f"{self.target_y_mm:.1f}")

        self._update_prediction()
        self._redraw_virtual_table()

    def _on_manual_coords_applied(self):
        try:
            x = float(self.ent_x.get())
            y = float(self.ent_y.get())
            if not (0 <= x <= TABLE.INNER_LENGTH_MM and 0 <= y <= TABLE.INNER_BREADTH_MM):
                messagebox.showwarning(
                    "Coordinates Out of Range",
                    f"X must be in [0, {TABLE.INNER_LENGTH_MM:.1f}] mm\nY must be in [0, {TABLE.INNER_BREADTH_MM:.1f}] mm",
                )
                return
            self.target_x_mm = x
            self.target_y_mm = y
            self._update_prediction()
            self._redraw_virtual_table()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid floating-point coordinate numbers.")

    def _update_prediction(self):
        pan, tilt, info = self.active_interpolator.predict(self.target_x_mm, self.target_y_mm)
        self.predicted_pan = pan
        self.predicted_tilt = tilt
        self.prediction_info = info

        # Update labels
        self.lbl_target_val.config(text=f"X: {self.target_x_mm:.1f} mm | Y: {self.target_y_mm:.1f} mm")
        self.lbl_pred_angles.config(text=f"Pan: {self.predicted_pan:.1f}° | Tilt: {self.predicted_tilt:.1f}°")
        self.lbl_model_info.config(text=self.current_interpolator_name)

        if not self.soft_limits.is_pan_safe(pan) or not self.soft_limits.is_tilt_safe(tilt):
            self.lbl_pred_angles.config(fg=ACCENT_AMBER)
            self.lbl_cmd_feedback.config(
                text="Warning: Predicted angles exceed soft limits and will be safely clamped.",
                fg=ACCENT_AMBER,
            )
        else:
            self.lbl_pred_angles.config(fg=ACCENT_BLUE)
            self.lbl_cmd_feedback.config(text="Angles within safe travel limits.", fg=TEXT_MUTED)

    # =========================================================================
    # ESP32 HARDWARE INTEGRATION & SAFETY CONTROLS
    # =========================================================================
    def _on_toggle_simulation(self):
        is_sim = self.sim_mode_var.get()
        self.esp32.set_simulation_mode(is_sim)
        if is_sim:
            self.lbl_esp_status.config(text="ESP32: [SIMULATED]", fg=ACCENT_AMBER)
            self.lbl_status_msg.config(text="Simulation mode enabled. Commands are simulated locally without network calls.")
        else:
            self.lbl_esp_status.config(text="ESP32: Connecting...", fg=TEXT_MUTED)
            self.lbl_status_msg.config(text="Physical hardware mode enabled. Verifying connection to 192.168.4.1...")

    def _on_send_predicted_clicked(self):
        req_confirm = self.var_confirm_move.get()
        if req_confirm and not self.esp32.simulation_mode:
            answer = messagebox.askyesno(
                "Confirm Servo Movement",
                f"Are you sure you want to move physical servos?\n\n"
                f"Target: ({self.target_x_mm:.1f}, {self.target_y_mm:.1f}) mm\n"
                f"Pan: {self.predicted_pan:.1f}°\n"
                f"Tilt: {self.predicted_tilt:.1f}°\n\n"
                f"Ensure laser is pointed safely away from people and eyes.",
                icon="warning",
            )
            if not answer:
                self.lbl_cmd_feedback.config(text="Movement cancelled by user.", fg=TEXT_MUTED)
                return

        success, msg = self.esp32.move_servos(
            self.predicted_pan,
            self.predicted_tilt,
            require_confirmation=False,
            confirmed=True,
        )
        color = ACCENT_GREEN if success else ACCENT_RED
        self.lbl_cmd_feedback.config(text=msg, fg=color)
        self.lbl_status_msg.config(text=msg)

    def _on_center_clicked(self):
        success, msg = self.esp32.center_servos()
        self.lbl_status_msg.config(text=msg)
        messagebox.showinfo("Servos Centered", msg)

    def _on_emergency_stop_clicked(self):
        success, msg = self.esp32.stop_servos()
        self.lbl_status_msg.config(text=msg)
        messagebox.showwarning(
            "Emergency Stop Triggered",
            f"{msg}\n\n"
            f"NOTE: The laser has been turned OFF and servo target movement halted.\n"
            f"Use 'CENTER' to clear stop status and resume operation.",
        )

    def _jog_servo(self, axis: str, delta: int):
        status = self.esp32.get_status()
        current_angle = status.pan if axis == "pan" else status.tilt
        target_angle = current_angle + delta
        if axis == "pan":
            success, msg = self.esp32.move_servos(target_angle, status.tilt, confirmed=True)
        else:
            success, msg = self.esp32.move_servos(status.pan, target_angle, confirmed=True)
        self.lbl_cmd_feedback.config(text=msg, fg=ACCENT_GREEN if success else ACCENT_RED)

    def _on_toggle_laser_manual(self):
        current_status = self.esp32.get_status()
        if current_status.laser:
            # Turn it off
            self.esp32.set_laser(False)
            self.btn_laser_toggle.config(text="ACTIVATE LASER (SAFETY ACK REQUIRED)", fg=ACCENT_AMBER)
        else:
            # Safety confirmation required
            ack = messagebox.askyesno(
                "CRITICAL LASER SAFETY WARNING",
                "Laser Activation Notice:\n\n"
                "1. Laser must remain OFF by default during calibration.\n"
                "2. Ensure eye protection is worn and no people are in the beam path.\n"
                "3. Ensure the beam points onto a safe, non-reflective target surface.\n\n"
                "Do you acknowledge and explicitly request turning the laser ON?",
                icon="warning",
            )
            if ack:
                success, msg = self.esp32.set_laser(True, safety_acknowledgment=True)
                if success:
                    self.btn_laser_toggle.config(text="DEACTIVATE LASER [ON]", fg=ACCENT_RED)
                messagebox.showinfo("Laser State", msg)

    def _on_laser_off_forced(self):
        self.esp32.set_laser(False)
        self.btn_laser_toggle.config(text="ACTIVATE LASER (SAFETY ACK REQUIRED)", fg=ACCENT_AMBER)
        self.lbl_status_msg.config(text="Laser deactivated.")

    def _esp32_polling_loop(self):
        """Background thread polling ESP32 /status every 1.5 seconds."""
        while self.running:
            try:
                status = self.esp32.get_status()
                self._update_telemetry_ui(status)
            except Exception as e:
                pass
            time.sleep(1.5)

    def _update_telemetry_ui(self, status: ESP32Status):
        if not self.running:
            return

        def _apply():
            if status.is_simulation:
                self.lbl_esp_status.config(text="ESP32: [SIMULATION MODE]", fg=ACCENT_AMBER)
            elif status.connected:
                latency = f"{status.latency_ms:.0f}ms"
                self.lbl_esp_status.config(text=f"ESP32: Connected ({latency})", fg=ACCENT_GREEN)
            else:
                self.lbl_esp_status.config(text="ESP32: Disconnected", fg=ACCENT_RED)

            self.lbl_telem_pan.config(text=f"{status.pan}°")
            self.lbl_telem_tilt.config(text=f"{status.tilt}°")
            self.lbl_telem_laser.config(
                text="ON" if status.laser else "OFF",
                fg=ACCENT_RED if status.laser else ACCENT_AMBER,
            )
            self.lbl_telem_stopped.config(
                text="STOPPED" if status.stopped else "NORMAL",
                fg=ACCENT_RED if status.stopped else ACCENT_GREEN,
            )

        self.after(0, _apply)

    # =========================================================================
    # CALIBRATION NODES TAB ACTIONS
    # =========================================================================
    def _refresh_nodes_treeview(self):
        for item in self.tree_nodes.get_children():
            self.tree_nodes.delete(item)
        for n in self.dataset.nodes:
            self.tree_nodes.insert(
                "",
                tk.END,
                values=(n.id, n.label, f"{n.x_mm:.1f}", f"{n.y_mm:.1f}", f"{n.pan_deg:.1f}", f"{n.tilt_deg:.1f}", n.notes),
            )

    def _on_model_changed(self, event=None):
        name = self.cmb_model.get()
        if name in self.interpolators:
            self.current_interpolator_name = name
            self.active_interpolator = self.interpolators[name]
            self.active_interpolator.fit(self.dataset.nodes)
            self._update_prediction()

    def _on_reset_nodes_clicked(self):
        if messagebox.askyesno("Reset Calibration", "Reset all calibration nodes back to the 9 default empirical samples?"):
            self.dataset.reset_to_defaults()
            self._reinit_interpolators()
            self._refresh_nodes_treeview()
            self._redraw_virtual_table()
            self._update_prediction()

    def _reinit_interpolators(self):
        for interp in self.interpolators.values():
            interp.fit(self.dataset.nodes)

    def _on_add_node_dialog(self):
        self._show_node_dialog(title="Add Calibration Node", node=None)

    def _on_edit_node_dialog(self):
        selected = self.tree_nodes.selection()
        if not selected:
            messagebox.showinfo("Select Node", "Please select a node to edit.")
            return
        node_id = self.tree_nodes.item(selected[0])["values"][0]
        node = next((n for n in self.dataset.nodes if n.id == node_id), None)
        if node:
            self._show_node_dialog(title="Edit Calibration Node", node=node)

    def _show_node_dialog(self, title: str, node: Optional[CalibrationNode]):
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.geometry("380x320")
        dlg.configure(bg=BG_CARD)
        dlg.transient(self)
        dlg.grab_set()

        tk.Label(dlg, text="Label:", bg=BG_CARD, fg=TEXT_MUTED).grid(row=0, column=0, sticky="w", padx=16, pady=8)
        ent_lbl = tk.Entry(dlg, width=20)
        ent_lbl.grid(row=0, column=1, padx=16, pady=8)
        ent_lbl.insert(0, node.label if node else "Custom Point")

        tk.Label(dlg, text=f"X mm (0 - {TABLE.INNER_LENGTH_MM:.1f}):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=1, column=0, sticky="w", padx=16, pady=8)
        ent_x = tk.Entry(dlg, width=20)
        ent_x.grid(row=1, column=1, padx=16, pady=8)
        ent_x.insert(0, f"{node.x_mm:.1f}" if node else f"{self.target_x_mm:.1f}")

        tk.Label(dlg, text=f"Y mm (0 - {TABLE.INNER_BREADTH_MM:.1f}):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=2, column=0, sticky="w", padx=16, pady=8)
        ent_y = tk.Entry(dlg, width=20)
        ent_y.grid(row=2, column=1, padx=16, pady=8)
        ent_y.insert(0, f"{node.y_mm:.1f}" if node else f"{self.target_y_mm:.1f}")

        tk.Label(dlg, text="Pan (°):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=3, column=0, sticky="w", padx=16, pady=8)
        ent_pan = tk.Entry(dlg, width=20)
        ent_pan.grid(row=3, column=1, padx=16, pady=8)
        ent_pan.insert(0, f"{node.pan_deg:.1f}" if node else f"{self.predicted_pan:.1f}")

        tk.Label(dlg, text="Tilt (°):", bg=BG_CARD, fg=TEXT_MUTED).grid(row=4, column=0, sticky="w", padx=16, pady=8)
        ent_tilt = tk.Entry(dlg, width=20)
        ent_tilt.grid(row=4, column=1, padx=16, pady=8)
        ent_tilt.insert(0, f"{node.tilt_deg:.1f}" if node else f"{self.predicted_tilt:.1f}")

        def _save():
            try:
                x = float(ent_x.get())
                y = float(ent_y.get())
                pan = float(ent_pan.get())
                tilt = float(ent_tilt.get())
                lbl = ent_lbl.get()

                if node is None:
                    new_node = CalibrationNode(
                        id=f"node_{len(self.dataset.nodes)+1}",
                        label=lbl,
                        x_mm=x,
                        y_mm=y,
                        pan_deg=pan,
                        tilt_deg=tilt,
                        notes="User added point",
                    )
                    self.dataset.add_node(new_node)
                else:
                    self.dataset.update_node(node.id, x, y, pan, tilt, label=lbl)

                self._reinit_interpolators()
                self._refresh_nodes_treeview()
                self._redraw_virtual_table()
                self._update_prediction()
                dlg.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

        btn_save = tk.Button(dlg, text="Save Node", command=_save, bg=ACCENT_GREEN, fg="white", relief=tk.FLAT, padx=12, pady=4)
        btn_save.grid(row=5, column=0, columnspan=2, pady=16)

    def _on_delete_node_clicked(self):
        selected = self.tree_nodes.selection()
        if not selected:
            messagebox.showinfo("Select Node", "Please select a node to delete.")
            return
        node_id = self.tree_nodes.item(selected[0])["values"][0]
        if messagebox.askyesno("Delete Node", f"Are you sure you want to delete node {node_id}?"):
            if not self.dataset.delete_node(node_id):
                messagebox.showwarning("Cannot Delete", "At least 3 calibration nodes must remain.")
                return
            self._reinit_interpolators()
            self._refresh_nodes_treeview()
            self._redraw_virtual_table()
            self._update_prediction()

    def _on_save_nodes_json(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="calibration_points.json",
        )
        if filepath:
            self.dataset.save_to_file(filepath)
            messagebox.showinfo("Saved", f"Calibration data successfully saved to:\n{filepath}")

    def _on_load_nodes_json(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.dataset = CalibrationDataset.load_from_file(filepath)
                self._reinit_interpolators()
                self._refresh_nodes_treeview()
                self._redraw_virtual_table()
                self._update_prediction()
                messagebox.showinfo("Loaded", f"Loaded {len(self.dataset.nodes)} nodes from:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Load Failed", f"Could not load JSON file: {e}")

    # =========================================================================
    # TAB 3: PHONE CAMERA HOMOGRAPHY & SPOT MEASUREMENT
    # =========================================================================
    def _on_load_photo_clicked(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            self.current_image_path = filepath
            self.raw_pil_image = Image.open(filepath).convert("RGB")
            self.corner_points = []
            self.corner_step_index = 0
            self.lbl_camera_guide.config(
                text="Photo loaded. Click '2. Mark 4 Inner Corners' to calibrate table boundaries.",
                fg=TEXT_MAIN,
            )
            self._render_camera_canvas()
        except Exception as e:
            messagebox.showerror("Error Opening Image", f"Failed to load image: {e}")

    def _on_start_corner_calibration(self):
        if not self.raw_pil_image:
            messagebox.showinfo("No Photo Loaded", "Please load a photo of the table first.")
            return
        self.corner_points = []
        self.corner_step_index = 0
        self.measuring_spot_mode = False
        self._update_corner_step_instruction()

    def _update_corner_step_instruction(self):
        steps = [
            "Click Corner 1: TOP-LEFT inner felt corner",
            "Click Corner 2: TOP-RIGHT inner felt corner",
            "Click Corner 3: BOTTOM-RIGHT inner felt corner",
            "Click Corner 4: BOTTOM-LEFT inner felt corner",
        ]
        if self.corner_step_index < 4:
            self.lbl_camera_guide.config(
                text=f"CALIBRATION IN PROGRESS: {steps[self.corner_step_index]}",
                fg=ACCENT_AMBER,
            )
        else:
            self.lbl_camera_guide.config(
                text="Homography computed! Now use '3. Click Laser Spot on Photo' to measure accuracy.",
                fg=ACCENT_GREEN,
            )
        self._render_camera_canvas()

    def _on_toggle_spot_pick_mode(self):
        if not self.camera_homography.is_calibrated():
            messagebox.showwarning(
                "Homography Not Calibrated",
                "Please calibrate the 4 table corners first before picking the laser spot.",
            )
            return
        self.measuring_spot_mode = True
        self.lbl_camera_guide.config(
            text="SPOT MEASUREMENT MODE: Click directly on the laser spot in the photo.",
            fg=ACCENT_GREEN,
        )

    def _on_autodetect_spot_clicked(self):
        if not self.current_image_path or not self.camera_homography.is_calibrated():
            messagebox.showwarning(
                "Prerequisites Missing",
                "Ensure image is loaded and 4 corners are calibrated.",
            )
            return
        # Load image via OpenCV
        img_bgr = cv2.imread(self.current_image_path)
        spot = detect_red_laser_spot(img_bgr)
        if spot:
            u_px, v_px, conf = spot
            self._process_measured_laser_spot(u_px, v_px)
            messagebox.showinfo(
                "Spot Detected",
                f"Red spot detected at pixel ({u_px:.1f}, {v_px:.1f}) with {conf*100:.0f}% confidence.",
            )
        else:
            messagebox.showwarning(
                "Spot Not Found",
                "Could not automatically isolate red spot. Please use manual click mode.",
            )

    def _on_cam_canvas_resize(self, event):
        self._render_camera_canvas()

    def _render_camera_canvas(self):
        c = self.cam_canvas
        c.delete("all")
        if not self.raw_pil_image:
            c.create_text(
                c.winfo_width() / 2,
                c.winfo_height() / 2,
                text="No photo loaded.\nClick '1. Load Photo...' to begin camera verification.",
                fill=TEXT_MUTED,
                font=("Segoe UI", 12),
                justify=tk.CENTER,
            )
            return

        cw = max(50, c.winfo_width())
        ch = max(50, c.winfo_height())
        iw, ih = self.raw_pil_image.size

        # Fit image preserving aspect ratio
        scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2

        self.img_scale_factor = scale
        self.img_offset_x = ox
        self.img_offset_y = oy

        resized = self.raw_pil_image.resize((nw, nh), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(resized)

        # Draw selected corners and connecting quad
        pts = []
        for i, (u, v) in enumerate(self.corner_points):
            su = int(u * scale)
            sv = int(v * scale)
            pts.append((su, sv))
            # Draw circle
            draw.ellipse([su - 6, sv - 6, su + 6, sv + 6], fill="#10b981", outline="white", width=2)
            draw.text((su + 8, sv - 8), f"C{i+1}", fill="white")

        if len(pts) > 1:
            draw.line(pts, fill="#10b981", width=2)
            if len(pts) == 4:
                draw.line([pts[3], pts[0]], fill="#10b981", width=2)

        self.photo_image = ImageTk.PhotoImage(resized)
        c.create_image(ox, oy, anchor="nw", image=self.photo_image)

    def _on_cam_canvas_click(self, event):
        if not self.raw_pil_image:
            return

        # Convert canvas click to original image pixel coordinates
        scale = self.img_scale_factor
        if scale <= 0:
            return
        u_px = (event.x - self.img_offset_x) / scale
        v_px = (event.y - self.img_offset_y) / scale

        iw, ih = self.raw_pil_image.size
        if not (0 <= u_px <= iw and 0 <= v_px <= ih):
            return

        # If in corner calibration mode:
        if self.corner_step_index < 4:
            self.corner_points.append((u_px, v_px))
            self.corner_step_index += 1
            if self.corner_step_index == 4:
                # Compute Homography
                corners = ImageCornerSelection(
                    top_left=self.corner_points[0],
                    top_right=self.corner_points[1],
                    bottom_right=self.corner_points[2],
                    bottom_left=self.corner_points[3],
                )
                success = self.camera_homography.compute_homography(corners)
                if success:
                    messagebox.showinfo(
                        "Homography Computed",
                        "Perspective homography matrix successfully computed!\n"
                        "Now point laser at a target and click the laser spot on the image to measure error.",
                    )
                else:
                    messagebox.showerror("Error", "Failed to compute homography matrix. Please retry.")
                    self.corner_points = []
                    self.corner_step_index = 0
            self._update_corner_step_instruction()

        elif self.measuring_spot_mode:
            self._process_measured_laser_spot(u_px, v_px)

    def _process_measured_laser_spot(self, u_px: float, v_px: float):
        mapped = self.camera_homography.image_to_table(u_px, v_px)
        if not mapped:
            messagebox.showerror("Error", "Coordinate projection failed. Homography might be degenerate.")
            return

        xm, ym = mapped
        xr, yr = self.target_x_mm, self.target_y_mm
        ex = xm - xr
        ey = ym - yr
        e = math.hypot(ex, ey)

        # Update verification panel
        self.lbl_spot_measured.config(text=f"X: {xm:.1f} mm | Y: {ym:.1f} mm")
        self.lbl_spot_requested.config(text=f"X: {xr:.1f} mm | Y: {yr:.1f} mm")
        self.lbl_spot_exy.config(text=f"ex: {ex:+.1f} mm | ey: {ey:+.1f} mm")
        self.lbl_spot_euclidean.config(text=f"{e:.1f} mm")

        # Stash current measurement candidate
        self.last_spot_measurement = {
            "target_x": xr,
            "target_y": yr,
            "pan": self.predicted_pan,
            "tilt": self.predicted_tilt,
            "measured_x": xm,
            "measured_y": ym,
            "ex": ex,
            "ey": ey,
            "e": e,
        }
        self.btn_log_meas.config(state=tk.NORMAL)

        # Draw spot marker on canvas
        self._render_camera_canvas()
        c = self.cam_canvas
        su = int(u_px * self.img_scale_factor) + self.img_offset_x
        sv = int(v_px * self.img_scale_factor) + self.img_offset_y
        c.create_oval(su - 8, sv - 8, su + 8, sv + 8, outline=ACCENT_RED, width=2)
        c.create_text(su + 12, sv, text=f"Spot: e={e:.1f}mm", fill=ACCENT_RED, font=("Segoe UI", 9, "bold"), anchor="w")

    def _on_record_measurement_clicked(self):
        if not hasattr(self, "last_spot_measurement"):
            return
        m = self.last_spot_measurement
        rec = SpotMeasurementRecord.create(
            target_x_mm=m["target_x"],
            target_y_mm=m["target_y"],
            commanded_pan_deg=m["pan"],
            commanded_tilt_deg=m["tilt"],
            measured_x_mm=m["measured_x"],
            measured_y_mm=m["measured_y"],
            notes=f"Model: {self.current_interpolator_name}",
            image_file=os.path.basename(self.current_image_path or ""),
        )
        self.session.add_record(rec)
        self._refresh_results_tab()
        self.btn_log_meas.config(state=tk.DISABLED)
        messagebox.showinfo("Logged", f"Measurement recorded into session!\nError: {rec.euclidean_error_mm:.1f} mm")

    # =========================================================================
    # TAB 4: RESULTS & CSV ACTIONS
    # =========================================================================
    def _refresh_results_tab(self):
        for item in self.tree_results.get_children():
            self.tree_results.delete(item)

        for r in self.session.records:
            self.tree_results.insert(
                "",
                tk.END,
                values=(
                    r.id,
                    r.timestamp.split(" ")[-1],
                    f"({r.target_x_mm:.1f}, {r.target_y_mm:.1f})",
                    f"({r.commanded_pan_deg:.1f}°, {r.commanded_tilt_deg:.1f}°)",
                    f"({r.measured_x_mm:.1f}, {r.measured_y_mm:.1f})",
                    f"{r.error_x_mm:+.1f}",
                    f"{r.error_y_mm:+.1f}",
                    f"{r.euclidean_error_mm:.1f}",
                ),
            )

        # Update stats
        stats = self.session.calculate_statistics()
        self.lbl_stat_count.config(text=str(stats.count))
        self.lbl_stat_mean.config(text=f"{stats.mean_error_mm:.1f} mm")
        self.lbl_stat_median.config(text=f"{stats.median_error_mm:.1f} mm")
        self.lbl_stat_rms.config(text=f"{stats.rms_error_mm:.1f} mm")
        self.lbl_stat_max.config(text=f"{stats.max_error_mm:.1f} mm")

    def _on_export_csv_clicked(self):
        if self.session.count() == 0:
            messagebox.showinfo("No Data", "There are no recorded measurements to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="laser_calibration_results.csv",
        )
        if filepath:
            self.session.export_to_csv(filepath)
            messagebox.showinfo("Export Successful", f"Exported {self.session.count()} measurements to:\n{filepath}")

    def _on_clear_results_clicked(self):
        if messagebox.askyesno("Clear Logs", "Clear all recorded session results?"):
            self.session.clear()
            self._refresh_results_tab()

    def _on_close(self):
        self.running = False
        self.destroy()


def run_app():
    app = PoolBotCalibrationApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
