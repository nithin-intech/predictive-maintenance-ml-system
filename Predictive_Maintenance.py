import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import warnings
import threading
import math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────
BG_DEEP      = "#0A0A0F"
BG_CARD      = "#111118"
BG_CARD2     = "#14141E"
BG_PANEL     = "#0D0D15"
ACCENT_BLUE  = "#00AAFF"
ACCENT_CYAN  = "#00E5FF"
ACCENT_GREEN = "#00FF88"
ACCENT_RED   = "#FF3355"
ACCENT_WARN  = "#FFAA00"
TEXT_PRIMARY = "#E8EAED"
TEXT_MUTED   = "#7A8499"
TEXT_DIM     = "#4A5168"
BORDER_DIM   = "#1E2030"
BORDER_GLOW  = "#003366"
GLOW_BLUE    = "#002244"

FONT_TITLE  = ("Consolas", 22, "bold")
FONT_HEAD   = ("Consolas", 14, "bold")
FONT_SUB    = ("Consolas", 11, "bold")
FONT_BODY   = ("Consolas", 10)
FONT_SMALL  = ("Consolas", 9)
FONT_LABEL  = ("Consolas", 9)
FONT_METRIC = ("Consolas", 20, "bold")
FONT_BTN    = ("Consolas", 10, "bold")

PLT_STYLE = {
    "figure.facecolor": BG_DEEP,
    "axes.facecolor":   BG_CARD,
    "axes.edgecolor":   BORDER_DIM,
    "axes.labelcolor":  TEXT_MUTED,
    "text.color":       TEXT_PRIMARY,
    "xtick.color":      TEXT_MUTED,
    "ytick.color":      TEXT_MUTED,
    "grid.color":       BORDER_DIM,
    "grid.linewidth":   0.5,
    "axes.grid":        True,
    "legend.facecolor": BG_CARD2,
    "legend.edgecolor": BORDER_DIM,
    "legend.labelcolor": TEXT_PRIMARY,
}

# ─────────────────────────────────────────────
#  3D ARCHITECTURE DATA
# ─────────────────────────────────────────────
ARCH_NODES = [
    # (label,               x,    y,    z,   layer_color)
    ("Sensor Data",          0,    0,    0,   "#00E5FF"),   # L1 bottom
    ("Data Preprocessing",  -1.2,  0,    1.6, "#00AAFF"),   # L2
    ("Feature Engineering",  1.2,  0,    1.6, "#00AAFF"),   # L2
    ("Random Forest Model",  0,    0,    3.2, "#7B2FFF"),   # L3 centre
    ("Failure Risk\nPrediction", -1.2, 0, 4.8, "#00FF88"),  # L4
    ("Health Index\nEngine",  1.2,  0,    4.8, "#00FF88"),  # L4
    ("Maintenance\nDecision", -1.0, 0,   6.4, "#FF9500"),   # L5
    ("Report\nGenerator",     1.0,  0,    6.4, "#FF9500"),  # L5
]

ARCH_EDGES = [
    (0, 1), (0, 2),        # Sensor → Preprocessing, Feature Eng
    (1, 3), (2, 3),        # Both L2 → Random Forest
    (3, 4), (3, 5),        # RF → Failure Pred, Health Index
    (4, 6), (5, 6),        # L4 → Maintenance Decision
    (4, 7), (5, 7),        # L4 → Report Generator
]


# ─────────────────────────────────────────────
#  SPLASH SCREEN
# ─────────────────────────────────────────────
class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.win  = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg=BG_DEEP)

        W, H = 700, 420
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        self.win.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.win.lift()
        self.win.attributes("-topmost", True)

        self._draw_border()
        self._build_content()

        self.messages = [
            "Initializing ML Engine...",
            "Loading Predictive Analytics Core...",
            "Configuring Model Parameters...",
            "Preparing Health Index Module...",
            "Building 3D Architecture View...",
            "System Ready.",
        ]
        self.progress = 0.0
        self._animate(0)

    def _draw_border(self):
        c = tk.Canvas(self.win, bg=BG_DEEP, highlightthickness=0, width=700, height=420)
        c.place(x=0, y=0)
        for i in range(4):
            shade = int(40 + i * 15)
            c.create_rectangle(i, i, 700-i, 420-i,
                                outline=f"#00{shade:02X}{shade*2:02X}", width=1)
        c.create_line(560,  0, 700,  0, fill=ACCENT_BLUE, width=2)
        c.create_line(700,  0, 700, 80, fill=ACCENT_BLUE, width=2)
        c.create_line(560,  0, 700, 80, fill=ACCENT_BLUE, width=1)
        c.create_line(  0, 340, 140, 420, fill=ACCENT_BLUE, width=1)
        c.create_line(  0, 340,   0, 420, fill=ACCENT_BLUE, width=2)
        c.create_line(  0, 420, 140, 420, fill=ACCENT_BLUE, width=2)
        self.canvas = c

    def _build_content(self):
        f = tk.Frame(self.win, bg=BG_DEEP)
        f.place(x=40, y=30, width=620, height=360)

        tk.Label(f, text="◈  INDUSTRIAL PREDICTIVE MAINTENANCE  ◈",
                 bg=BG_DEEP, fg=ACCENT_CYAN,
                 font=("Consolas", 15, "bold")).pack(pady=(20, 4))
        tk.Label(f, text="SYSTEM",
                 bg=BG_DEEP, fg=ACCENT_BLUE,
                 font=("Consolas", 38, "bold")).pack(pady=(0, 4))
        tk.Label(f, text="Machine Learning-Based Industrial Failure Prediction",
                 bg=BG_DEEP, fg=TEXT_MUTED,
                 font=("Consolas", 10)).pack(pady=(0, 20))

        detail_f = tk.Frame(f, bg=BG_CARD)
        detail_f.pack(fill="x", padx=20, pady=(0, 20))
        for i, (k, v) in enumerate([
            ("VERSION", "3.0"),
            ("ENGINE",  "Random Forest"),
            ("DATASET", "AI4I 2020"),
            ("MODULE",  "Health Index"),
        ]):
            col = tk.Frame(detail_f, bg=BG_CARD)
            col.grid(row=0, column=i, padx=18, pady=12)
            tk.Label(col, text=k, bg=BG_CARD, fg=TEXT_DIM,
                     font=("Consolas", 8, "bold")).pack()
            tk.Label(col, text=v, bg=BG_CARD, fg=ACCENT_CYAN,
                     font=("Consolas", 10, "bold")).pack()
        detail_f.columnconfigure(list(range(4)), weight=1)

        tk.Frame(f, bg=BORDER_GLOW, height=1).pack(fill="x", padx=20, pady=(0, 15))

        self.msg_var = tk.StringVar(value="Initializing ML Engine...")
        tk.Label(f, textvariable=self.msg_var,
                 bg=BG_DEEP, fg=ACCENT_GREEN,
                 font=("Consolas", 10)).pack(pady=(0, 8))

        pb_frame = tk.Frame(f, bg=BG_DEEP)
        pb_frame.pack(fill="x", padx=20, pady=(0, 4))
        self.pb_bg = tk.Canvas(pb_frame, height=8, bg=BG_CARD,
                               highlightthickness=1,
                               highlightbackground=BORDER_DIM)
        self.pb_bg.pack(fill="x")
        self.pb_fill = self.pb_bg.create_rectangle(0, 0, 0, 8,
                                                    fill=ACCENT_BLUE, width=0)
        self.pct_var = tk.StringVar(value="0%")
        tk.Label(f, textvariable=self.pct_var,
                 bg=BG_DEEP, fg=TEXT_MUTED,
                 font=("Consolas", 8)).pack()
        tk.Label(f,
                 text="Industrial Predictive Maintenance System  |  AI-Powered Analytics",
                 bg=BG_DEEP, fg=TEXT_DIM, font=("Consolas", 8)).pack(side="bottom", pady=8)

    def _animate(self, step):
        total = 140
        if step <= total:
            pct = step / total
            self.pb_bg.update_idletasks()
            w = self.pb_bg.winfo_width()
            self.pb_bg.coords(self.pb_fill, 0, 0, int(w * pct), 8)
            color = (ACCENT_GREEN if pct > 0.9 else
                     ACCENT_CYAN  if pct > 0.6 else ACCENT_BLUE)
            self.pb_bg.itemconfig(self.pb_fill, fill=color)
            idx = min(int(pct * len(self.messages)), len(self.messages) - 1)
            self.msg_var.set(self.messages[idx])
            self.pct_var.set(f"{int(pct * 100)}%")
            self.win.after(22, lambda: self._animate(step + 1))
        else:
            self.win.after(400, self.close)

    def close(self):
        self.win.destroy()


# ─────────────────────────────────────────────
#  HELPER WIDGETS
# ─────────────────────────────────────────────
class MetricCard(tk.Frame):
    def __init__(self, parent, label, value="—", unit="", accent=ACCENT_BLUE, **kw):
        super().__init__(parent, bg=BG_CARD,
                         highlightthickness=1,
                         highlightbackground=BORDER_DIM, **kw)
        self.accent = accent
        self.configure(cursor="hand2")

        tk.Label(self, text=label.upper(), bg=BG_CARD, fg=TEXT_DIM,
                 font=("Consolas", 8, "bold")).pack(pady=(10, 2), padx=12)
        self.val_label = tk.Label(self, text=value, bg=BG_CARD, fg=accent,
                                  font=("Consolas", 18, "bold"))
        self.val_label.pack(padx=12)
        if unit:
            tk.Label(self, text=unit, bg=BG_CARD, fg=TEXT_DIM,
                     font=("Consolas", 8)).pack(pady=(0, 8), padx=12)
        else:
            tk.Label(self, text="", bg=BG_CARD).pack(pady=(0, 6))

        for w in [self] + list(self.winfo_children()):
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)

    def _on_enter(self, e):
        self.configure(highlightbackground=self.accent, bg=BG_CARD2)
        for w in self.winfo_children():
            w.configure(bg=BG_CARD2)

    def _on_leave(self, e):
        self.configure(highlightbackground=BORDER_DIM, bg=BG_CARD)
        for w in self.winfo_children():
            w.configure(bg=BG_CARD)

    def update_value(self, value):
        self.val_label.configure(text=str(value))


def styled_button(parent, text, command=None, accent=ACCENT_BLUE, width=18):
    btn = tk.Button(parent, text=text, command=command,
                    bg=BG_CARD2, fg=accent, activebackground=accent,
                    activeforeground=BG_DEEP, font=FONT_BTN, relief="flat", bd=0,
                    padx=14, pady=8, width=width,
                    highlightthickness=1, highlightbackground=accent, cursor="hand2")
    btn.bind("<Enter>", lambda e: btn.configure(bg=accent, fg=BG_DEEP))
    btn.bind("<Leave>", lambda e: btn.configure(bg=BG_CARD2, fg=accent))
    return btn


def section_label(parent, text):
    f = tk.Frame(parent, bg=BG_PANEL)
    tk.Frame(f, bg=ACCENT_BLUE, width=3).pack(side="left", fill="y", padx=(0, 8))
    tk.Label(f, text=text, bg=BG_PANEL, fg=ACCENT_CYAN,
             font=FONT_SUB).pack(side="left", pady=6)
    return f


# ─────────────────────────────────────────────
#  3D ARCHITECTURE VISUALIZATION
# ─────────────────────────────────────────────
class ArchitectureView3D:
    """
    Self-contained rotating 3D network embedded into a Tkinter frame.
    Uses only matplotlib + mpl_toolkits (no external deps).
    """

    # Layer descriptors shown in the legend panel
    LAYER_META = [
        ("Layer 1 — Input",          "#00E5FF", "Sensor Data ingestion"),
        ("Layer 2 — Preprocessing",  "#00AAFF", "Cleaning · Feature Engineering"),
        ("Layer 3 — ML Core",        "#7B2FFF", "Random Forest Classifier"),
        ("Layer 4 — Inference",      "#00FF88", "Risk Prediction · Health Engine"),
        ("Layer 5 — Output",         "#FF9500", "Decisions · Reporting"),
    ]

    def __init__(self, parent_frame):
        self.parent   = parent_frame
        self._angle   = 0
        self._anim    = None
        self._running = False
        self._fig     = None
        self._canvas  = None
        self._ax      = None
        self._build()

    # ── node geometry (rotate around Z) ───────
    @staticmethod
    def _rot_y(nodes, deg):
        """Rotate all node x-coords by deg around the vertical (Z) axis."""
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = []
        for label, x, y, z, color in nodes:
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            out.append((label, xr, yr, z, color))
        return out

    # ── build UI ──────────────────────────────
    def _build(self):
        # top control strip
        ctrl = tk.Frame(self.parent, bg=BG_PANEL)
        ctrl.pack(fill="x", padx=16, pady=(10, 4))

        section_label(ctrl, "INDUSTRIAL AI SYSTEM ARCHITECTURE").pack(side="left")

        self._btn_toggle = styled_button(
            ctrl, "⏸  PAUSE ROTATION", self._toggle_anim,
            accent=ACCENT_CYAN, width=20)
        self._btn_toggle.pack(side="right", padx=4)

        styled_button(ctrl, "⟳  RESET VIEW", self._reset_view,
                      accent=ACCENT_BLUE, width=14).pack(side="right", padx=4)

        # main area: 3D canvas + legend
        main = tk.Frame(self.parent, bg=BG_DEEP)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        # 3D figure
        fig_frame = tk.Frame(main, bg=BG_DEEP,
                             highlightthickness=1,
                             highlightbackground=BORDER_DIM)
        fig_frame.pack(side="left", fill="both", expand=True)

        self._fig = plt.Figure(figsize=(9, 7), facecolor="#0e1117")
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._style_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # legend panel
        legend_frame = tk.Frame(main, bg=BG_CARD,
                                highlightthickness=1,
                                highlightbackground=BORDER_DIM,
                                width=220)
        legend_frame.pack(side="right", fill="y", padx=(8, 0))
        legend_frame.pack_propagate(False)
        self._build_legend(legend_frame)

        # subtitle bar
        sub = tk.Frame(self.parent, bg=GLOW_BLUE)
        sub.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(sub,
                 text="  Random Forest ML Pipeline  ·  AI4I 2020 Dataset  ·  Real-Time Health Computation",
                 bg=GLOW_BLUE, fg=ACCENT_CYAN, font=("Consolas", 9)).pack(
                     side="left", pady=4, padx=8)
        self._node_count_var = tk.StringVar(
            value=f"Nodes: {len(ARCH_NODES)}  |  Edges: {len(ARCH_EDGES)}")
        tk.Label(sub, textvariable=self._node_count_var,
                 bg=GLOW_BLUE, fg=TEXT_MUTED, font=("Consolas", 9)).pack(
                     side="right", pady=4, padx=8)

        # initial draw + start animation
        self._draw(0)
        self._start_anim()

    def _build_legend(self, frame):
        tk.Label(frame, text="ARCHITECTURE LEGEND",
                 bg=BG_CARD, fg=ACCENT_CYAN,
                 font=("Consolas", 9, "bold")).pack(pady=(14, 6), padx=10)

        tk.Frame(frame, bg=BORDER_DIM, height=1).pack(fill="x", padx=10, pady=(0, 10))

        for layer_name, color, desc in self.LAYER_META:
            row = tk.Frame(frame, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=3)
            dot = tk.Canvas(row, width=12, height=12, bg=BG_CARD,
                            highlightthickness=0)
            dot.create_oval(2, 2, 11, 11, fill=color, outline="")
            dot.pack(side="left", padx=(0, 6))
            col = tk.Frame(row, bg=BG_CARD)
            col.pack(side="left")
            tk.Label(col, text=layer_name, bg=BG_CARD, fg=TEXT_PRIMARY,
                     font=("Consolas", 8, "bold"), anchor="w").pack(anchor="w")
            tk.Label(col, text=desc, bg=BG_CARD, fg=TEXT_DIM,
                     font=("Consolas", 7), anchor="w").pack(anchor="w")

        tk.Frame(frame, bg=BORDER_DIM, height=1).pack(fill="x", padx=10, pady=(14, 8))

        # node list
        tk.Label(frame, text="PIPELINE NODES",
                 bg=BG_CARD, fg=ACCENT_CYAN,
                 font=("Consolas", 8, "bold")).pack(pady=(0, 6), padx=10)

        for idx, (label, *_, color) in enumerate(ARCH_NODES):
            short = label.replace("\n", " ")
            row = tk.Frame(frame, bg=BG_CARD)
            row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=f"{idx+1:02d}",
                     bg=BG_CARD, fg=color,
                     font=("Consolas", 8, "bold"), width=3).pack(side="left")
            tk.Label(row, text=short, bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Consolas", 7), anchor="w").pack(side="left")

        tk.Frame(frame, bg=BORDER_DIM, height=1).pack(fill="x", padx=10, pady=(12, 8))

        # edge count
        for label, val in [("Total Nodes", str(len(ARCH_NODES))),
                            ("Total Edges", str(len(ARCH_EDGES))),
                            ("Layers",      "5")]:
            row = tk.Frame(frame, bg=BG_CARD)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label, bg=BG_CARD, fg=TEXT_DIM,
                     font=("Consolas", 7), width=13, anchor="w").pack(side="left")
            tk.Label(row, text=val, bg=BG_CARD, fg=ACCENT_CYAN,
                     font=("Consolas", 8, "bold")).pack(side="left")



    # ── axes styling ──────────────────────────
    def _style_axes(self):
        ax = self._ax
        # Slightly lighter deep-blue background so neon elements pop
        ax.set_facecolor("#0b1a2a")
        self._fig.patch.set_facecolor("#0e1117")
        # Disable all grid lines
        ax.grid(False)
        # Make all pane walls fully transparent — no foggy boxes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Pane edge lines — keep very subtle dark border
        ax.xaxis.pane.set_edgecolor("#0d1f35")
        ax.yaxis.pane.set_edgecolor("#0d1f35")
        ax.zaxis.pane.set_edgecolor("#0d1f35")
        # Hide all axis ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # ── draw cube frame helper ────────────────
    @staticmethod
    def _draw_cube_frame(ax, x_range, y_range, z_range):
        """Draw a bounding box / cage with neon cyan glow — double-pass technique."""
        xs, xe = x_range
        ys, ye = y_range
        zs, ze = z_range

        # All 12 edges of a rectangular box
        corners = [
            # bottom face
            ([xs, xe], [ys, ys], [zs, zs]),
            ([xs, xe], [ye, ye], [zs, zs]),
            ([xs, xs], [ys, ye], [zs, zs]),
            ([xe, xe], [ys, ye], [zs, zs]),
            # top face
            ([xs, xe], [ys, ys], [ze, ze]),
            ([xs, xe], [ye, ye], [ze, ze]),
            ([xs, xs], [ys, ye], [ze, ze]),
            ([xe, xe], [ys, ye], [ze, ze]),
            # vertical pillars
            ([xs, xs], [ys, ys], [zs, ze]),
            ([xe, xe], [ys, ys], [zs, ze]),
            ([xs, xs], [ye, ye], [zs, ze]),
            ([xe, xe], [ye, ye], [zs, ze]),
        ]

        NEON_CYAN = "#00ffff"
        for (ex, ey, ez) in corners:
            # Pass 1 — thick soft glow halo
            ax.plot(ex, ey, ez,
                    color=NEON_CYAN, lw=5, alpha=0.10, zorder=1)
            # Pass 2 — crisp bright core line
            ax.plot(ex, ey, ez,
                    color=NEON_CYAN, lw=2.5, alpha=0.90, zorder=2)

    # ── draw frame ────────────────────────────
    def _draw(self, angle):
        ax = self._ax
        ax.cla()
        self._style_axes()

        nodes = self._rot_y(ARCH_NODES, angle)
        pos   = {i: (n[1], n[2], n[3]) for i, n in enumerate(nodes)}

        # ── cube bounding frame ────────────────
        # Encloses the entire node network with a neon cyan wireframe cage
        self._draw_cube_frame(ax,
                              x_range=(-1.8,  1.8),
                              y_range=(-1.8,  1.8),
                              z_range=(-0.3,  6.7))

        # ── pipeline edges — layered glow ──────
        for (src, dst) in ARCH_EDGES:
            xs = [pos[src][0], pos[dst][0]]
            ys = [pos[src][1], pos[dst][1]]
            zs = [pos[src][2], pos[dst][2]]
            # outer glow halo
            ax.plot(xs, ys, zs, color="#003366", lw=5.0, alpha=0.10, zorder=3)
            # mid glow
            ax.plot(xs, ys, zs, color="#0077CC", lw=2.5, alpha=0.40, zorder=4)
            # core bright line
            ax.plot(xs, ys, zs, color="#00AAFF", lw=1.2, alpha=0.90, zorder=5)

            # animated flow dot — travels 0→1 along each edge
            t   = (math.sin(math.radians(angle * 3 + src * 47)) * 0.5 + 0.5)
            dot = [xs[0] + t * (xs[1] - xs[0]),
                   ys[0] + t * (ys[1] - ys[0]),
                   zs[0] + t * (zs[1] - zs[0])]
            ax.scatter(*dot, s=28, c="#00E5FF", alpha=0.95,
                       depthshade=False, zorder=11)

        # ── nodes ─────────────────────────────
        for i, (label, xr, yr, zr, color) in enumerate(nodes):
            # glow halo rings — three concentric scatter layers
            for s, a in [(380, 0.05), (200, 0.12), (95, 0.25)]:
                ax.scatter(xr, yr, zr, s=s, c=color, alpha=a,
                           depthshade=False, zorder=6)
            # core node dot
            ax.scatter(xr, yr, zr, s=52, c=color, alpha=1.0,
                       depthshade=False, zorder=7,
                       edgecolors="white", linewidths=0.8)

            # label — nudge outward, above node
            offset_x = 0.20 if xr >= 0 else -0.20
            ha        = "left" if xr >= 0 else "right"
            ax.text(xr + offset_x, yr, zr + 0.25,
                    label, fontsize=7.5, color=color,
                    ha=ha, va="bottom",
                    fontfamily="monospace", fontweight="bold",
                    zorder=8)

        # ── title ─────────────────────────────
        ax.set_title("INDUSTRIAL AI SYSTEM ARCHITECTURE",
                     color="#00ffff", fontsize=11, fontweight="bold",
                     fontfamily="monospace", pad=12)

        # ── layer ring guides ─────────────────
        # Subtle horizontal circles at each layer Z level
        layer_z = [0.0, 1.6, 3.2, 4.8, 6.4]
        for z in layer_z:
            theta = np.linspace(0, 2 * np.pi, 72)
            r     = 1.82
            # glow pass
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    [z] * 72, color="#00ffff", lw=1.5, alpha=0.08, zorder=0)
            # core pass
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    [z] * 72, color="#004466", lw=0.7, alpha=0.55, zorder=0)

        # ── view ──────────────────────────────
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_zlim(-0.5, 7.2)
        ax.view_init(elev=18, azim=angle)
        self._canvas.draw_idle()

    # ── animation ─────────────────────────────
    def _tick(self, frame):
        self._angle = (self._angle + 0.6) % 360
        self._draw(self._angle)

    def _start_anim(self):
        self._running = True
        self._anim = FuncAnimation(
            self._fig, self._tick,
            interval=55, cache_frame_data=False)
        self._canvas.draw()
        self._btn_toggle.configure(text="⏸  PAUSE ROTATION")

    def _toggle_anim(self):
        if self._running:
            self._anim.event_source.stop()
            self._running = False
            self._btn_toggle.configure(text="▶  RESUME ROTATION")
        else:
            self._anim.event_source.start()
            self._running = True
            self._btn_toggle.configure(text="⏸  PAUSE ROTATION")

    def _reset_view(self):
        self._angle = 0
        self._draw(0)

    def stop(self):
        if self._anim:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class PredictiveMaintenanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Predictive Maintenance System v3.0")
        self.root.configure(bg=BG_DEEP)
        self.root.state("zoomed")
        self.root.minsize(1100, 720)

        # ML state
        self.df         = None
        self.model      = None
        self.scaler     = None
        self.le         = None
        self.feature_cols = None
        self.X_test     = None
        self.y_test     = None
        self.y_pred     = None
        self.accuracy   = None
        self.health_index = None

        self.status_var = tk.StringVar(value="◉  READY  |  Load a dataset to begin")
        self._arch_view = None

        self._apply_ttk_theme()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        if self._arch_view:
            self._arch_view.stop()
        plt.close("all")
        self.root.destroy()

    # ── TTK THEME ────────────────────────────
    def _apply_ttk_theme(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG_DEEP, foreground=TEXT_PRIMARY,
                    fieldbackground=BG_CARD, font=FONT_BODY,
                    bordercolor=BORDER_DIM, troughcolor=BG_CARD)
        s.configure("TNotebook", background=BG_PANEL, borderwidth=0,
                    tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab",
                    background=BG_CARD, foreground=TEXT_MUTED,
                    font=FONT_SUB, padding=[18, 10], borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", BG_PANEL)],
              foreground=[("selected", ACCENT_CYAN)],
              expand=[("selected", [0, 0, 0, 2])])
        s.configure("TCombobox", selectbackground=BG_CARD,
                    selectforeground=TEXT_PRIMARY)
        s.configure("TProgressbar", background=ACCENT_BLUE,
                    troughcolor=BG_CARD, borderwidth=0, thickness=6)
        s.configure("Vertical.TScrollbar",
                    background=BG_CARD, troughcolor=BG_DEEP, arrowcolor=TEXT_MUTED)
        s.configure("Horizontal.TScrollbar",
                    background=BG_CARD, troughcolor=BG_DEEP, arrowcolor=TEXT_MUTED)
        s.configure("Treeview",
                    background=BG_CARD, fieldbackground=BG_CARD,
                    foreground=TEXT_PRIMARY, rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading",
                    background=BG_CARD2, foreground=ACCENT_CYAN,
                    font=("Consolas", 9, "bold"), borderwidth=0)
        s.map("Treeview", background=[("selected", GLOW_BLUE)])

    # ── UI BUILD ─────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_status_bar()
        self._build_metrics_strip()
        self._build_notebook()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_PANEL,
                       highlightthickness=1, highlightbackground=BORDER_GLOW)
        hdr.pack(fill="x", pady=(0, 1))

        left = tk.Frame(hdr, bg=BG_PANEL)
        left.pack(side="left", padx=20, pady=12)
        tk.Label(left, text="⬡", bg=BG_PANEL, fg=ACCENT_BLUE,
                 font=("Consolas", 28, "bold")).pack(side="left", padx=(0, 12))
        titles = tk.Frame(left, bg=BG_PANEL)
        titles.pack(side="left")
        tk.Label(titles,
                 text="INDUSTRIAL PREDICTIVE MAINTENANCE SYSTEM",
                 bg=BG_PANEL, fg=ACCENT_CYAN,
                 font=("Consolas", 16, "bold")).pack(anchor="w")
        tk.Label(titles,
                 text="Machine Learning-Based Industrial Failure Prediction  ·  v3.0  ·  Random Forest Engine",
                 bg=BG_PANEL, fg=TEXT_MUTED, font=("Consolas", 9)).pack(anchor="w")

        right = tk.Frame(hdr, bg=BG_PANEL)
        right.pack(side="right", padx=20)
        for t, c in [("AI-POWERED", ACCENT_GREEN), ("ANALYTICS", ACCENT_GREEN),
                     ("PLATFORM", TEXT_DIM)]:
            tk.Label(right, text=t, bg=BG_PANEL, fg=c,
                     font=("Consolas", 9 if t != "PLATFORM" else 8,
                           "bold" if t != "PLATFORM" else "normal")).pack(anchor="e")

    def _build_status_bar(self):
        sb = tk.Frame(self.root, bg=GLOW_BLUE, height=28)
        sb.pack(fill="x")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self.status_var,
                 bg=GLOW_BLUE, fg=ACCENT_CYAN, font=("Consolas", 9)).pack(
                     side="left", padx=16, pady=4)
        self.clock_var = tk.StringVar()
        tk.Label(sb, textvariable=self.clock_var,
                 bg=GLOW_BLUE, fg=TEXT_MUTED, font=("Consolas", 9)).pack(
                     side="right", padx=16)
        self._tick_clock()

    def _tick_clock(self):
        import datetime
        self.clock_var.set(datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _build_metrics_strip(self):
        strip = tk.Frame(self.root, bg=BG_DEEP)
        strip.pack(fill="x", padx=16, pady=(10, 4))

        self.card_accuracy = MetricCard(strip, "Model Accuracy", "—", "%",     accent=ACCENT_CYAN)
        self.card_samples  = MetricCard(strip, "Dataset Size",   "—", "rows",  accent=ACCENT_BLUE)
        self.card_failures = MetricCard(strip, "Failures",       "—", "detected", accent=ACCENT_RED)
        self.card_health   = MetricCard(strip, "Health Index",   "—", "/ 100", accent=ACCENT_GREEN)
        self.card_risk     = MetricCard(strip, "Risk Level",     "—", "",      accent=ACCENT_WARN)

        for c in [self.card_accuracy, self.card_samples, self.card_failures,
                  self.card_health, self.card_risk]:
            c.pack(side="left", expand=True, fill="both", padx=4)

    def _build_notebook(self):
        nb_wrap = tk.Frame(self.root, bg=BG_DEEP)
        nb_wrap.pack(fill="both", expand=True, padx=16, pady=(4, 0))

        self.nb = ttk.Notebook(nb_wrap)
        self.nb.pack(fill="both", expand=True)

        self.tab_data   = tk.Frame(self.nb, bg=BG_PANEL)
        self.tab_train  = tk.Frame(self.nb, bg=BG_PANEL)
        self.tab_pred   = tk.Frame(self.nb, bg=BG_PANEL)
        self.tab_health = tk.Frame(self.nb, bg=BG_PANEL)
        self.tab_viz    = tk.Frame(self.nb, bg=BG_PANEL)
        self.tab_report = tk.Frame(self.nb, bg=BG_PANEL)
        self.tab_arch   = tk.Frame(self.nb, bg=BG_PANEL)

        self.nb.add(self.tab_data,   text="  ⊞  DATA LOADER  ")
        self.nb.add(self.tab_train,  text="  ⚙  MODEL TRAINING  ")
        self.nb.add(self.tab_pred,   text="  ▶  PREDICTION  ")
        self.nb.add(self.tab_health, text="  ♡  HEALTH INDEX  ")
        self.nb.add(self.tab_viz,    text="  ◎  VISUALIZATIONS  ")
        self.nb.add(self.tab_report, text="  ≡  REPORT  ")
        self.nb.add(self.tab_arch,   text="  ◈  3D Architecture  ")

        self._build_tab_data()
        self._build_tab_train()
        self._build_tab_pred()
        self._build_tab_health()
        self._build_tab_viz()
        self._build_tab_report()
        self._build_tab_arch()

    def _build_footer(self):
        ft = tk.Frame(self.root, bg=BG_PANEL,
                      highlightthickness=1, highlightbackground=BORDER_DIM)
        ft.pack(fill="x", side="bottom")
        tk.Label(ft,
                 text="Industrial Predictive Maintenance System  |  AI-Powered Analytics  |  Final Year Project",
                 bg=BG_PANEL, fg=TEXT_DIM, font=("Consolas", 8)).pack(pady=5)

    # ── TAB: DATA ────────────────────────────
    def _build_tab_data(self):
        p = self.tab_data
        section_label(p, "DATASET MANAGEMENT").pack(fill="x", padx=16, pady=(14, 8))

        ctrl = tk.Frame(p, bg=BG_PANEL)
        ctrl.pack(fill="x", padx=16, pady=4)
        styled_button(ctrl, "⊕  Load CSV Dataset",
                      self.load_csv, accent=ACCENT_BLUE).pack(side="left", padx=(0, 8))
        styled_button(ctrl, "⊘  Clear Data",
                      self.clear_data, accent=ACCENT_RED, width=14).pack(side="left")

        self.data_info_var = tk.StringVar(value="No dataset loaded.")
        tk.Label(p, textvariable=self.data_info_var,
                 bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_BODY,
                 justify="left").pack(padx=16, pady=4, anchor="w")

        tbl_frame = tk.Frame(p, bg=BG_PANEL)
        tbl_frame.pack(fill="both", expand=True, padx=16, pady=(4, 12))
        section_label(tbl_frame, "DATASET PREVIEW").pack(fill="x", pady=(0, 6))

        cols_frame = tk.Frame(tbl_frame, bg=BG_PANEL)
        cols_frame.pack(fill="both", expand=True)

        sy = ttk.Scrollbar(cols_frame, orient="vertical")
        sx = ttk.Scrollbar(cols_frame, orient="horizontal")
        self.data_tree = ttk.Treeview(cols_frame,
                                      yscrollcommand=sy.set,
                                      xscrollcommand=sx.set,
                                      show="headings", height=14)
        sy.config(command=self.data_tree.yview)
        sx.config(command=self.data_tree.xview)
        sy.pack(side="right",  fill="y")
        sx.pack(side="bottom", fill="x")
        self.data_tree.pack(fill="both", expand=True)

    # ── TAB: TRAIN ───────────────────────────
    def _build_tab_train(self):
        p = self.tab_train
        section_label(p, "MODEL CONFIGURATION").pack(fill="x", padx=16, pady=(14, 10))

        cfg = tk.Frame(p, bg=BG_PANEL)
        cfg.pack(fill="x", padx=16)

        params_f = tk.Frame(cfg, bg=BG_CARD,
                             highlightthickness=1, highlightbackground=BORDER_DIM)
        params_f.pack(side="left", padx=(0, 16), pady=4, fill="y")

        for i, (lbl, attr, default) in enumerate([
            ("N Estimators", "n_est_var",     "100"),
            ("Max Depth",    "max_depth_var", "10"),
            ("Test Split %", "test_split_var","20"),
            ("Random Seed",  "seed_var",      "42"),
        ]):
            row = tk.Frame(params_f, bg=BG_CARD)
            row.grid(row=i, column=0, padx=16, pady=8, sticky="w")
            tk.Label(row, text=lbl, bg=BG_CARD, fg=TEXT_MUTED,
                     font=FONT_LABEL, width=14, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            tk.Entry(row, textvariable=var, width=8,
                     bg=BG_CARD2, fg=ACCENT_CYAN, insertbackground=ACCENT_CYAN,
                     font=FONT_BODY, relief="flat",
                     highlightthickness=1, highlightbackground=BORDER_DIM).pack(side="left")

        right_f = tk.Frame(cfg, bg=BG_PANEL)
        right_f.pack(side="left", fill="both", expand=True)

        styled_button(right_f, "⚙  TRAIN MODEL",
                      self.train_model, accent=ACCENT_GREEN, width=20).pack(pady=(0, 8))

        section_label(right_f, "TRAINING LOG").pack(fill="x", pady=(6, 4))
        self.train_log = tk.Text(right_f, height=14, bg=BG_CARD,
                                 fg=ACCENT_GREEN, font=("Consolas", 9),
                                 relief="flat", state="disabled",
                                 insertbackground=ACCENT_GREEN,
                                 highlightthickness=1, highlightbackground=BORDER_DIM,
                                 wrap="word")
        self.train_log.pack(fill="both", expand=True, pady=(0, 8))

        res_f = tk.Frame(p, bg=BG_PANEL)
        res_f.pack(fill="both", expand=True, padx=16, pady=(6, 12))
        section_label(res_f, "CLASSIFICATION REPORT").pack(fill="x", pady=(0, 4))
        self.results_text = tk.Text(res_f, height=10, bg=BG_CARD,
                                    fg=TEXT_PRIMARY, font=("Consolas", 9),
                                    relief="flat", state="disabled",
                                    highlightthickness=1, highlightbackground=BORDER_DIM,
                                    wrap="word")
        self.results_text.pack(fill="both", expand=True)

    # ── TAB: PREDICTION ──────────────────────
    def _build_tab_pred(self):
        p = self.tab_pred
        section_label(p, "MANUAL INPUT — REAL-TIME PREDICTION").pack(
            fill="x", padx=16, pady=(14, 10))

        outer = tk.Frame(p, bg=BG_PANEL)
        outer.pack(fill="both", expand=True, padx=16)

        left_col = tk.Frame(outer, bg=BG_CARD,
                             highlightthickness=1, highlightbackground=BORDER_DIM)
        left_col.pack(side="left", fill="y", padx=(0, 12), pady=4)

        self.pred_vars = {}
        fields = [
            ("Air Temperature [K]",     "air_temp",  "298.1"),
            ("Process Temperature [K]", "proc_temp", "308.6"),
            ("Rotational Speed [rpm]",  "rot_speed", "1551"),
            ("Torque [Nm]",             "torque",    "42.8"),
            ("Tool Wear [min]",         "tool_wear", "0"),
            ("Machine Type",            "type",      "M"),
        ]
        for i, (lbl, key, default) in enumerate(fields):
            row = tk.Frame(left_col, bg=BG_CARD)
            row.grid(row=i, column=0, padx=16, pady=8, sticky="w")
            tk.Label(row, text=lbl, bg=BG_CARD, fg=TEXT_MUTED,
                     font=FONT_LABEL, width=24, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self.pred_vars[key] = var
            if key == "type":
                ttk.Combobox(row, textvariable=var,
                             values=["L", "M", "H"], width=10,
                             state="readonly").pack(side="left")
            else:
                tk.Entry(row, textvariable=var, width=12,
                         bg=BG_CARD2, fg=ACCENT_CYAN,
                         insertbackground=ACCENT_CYAN,
                         font=FONT_BODY, relief="flat",
                         highlightthickness=1,
                         highlightbackground=BORDER_DIM).pack(side="left")

        styled_button(left_col, "▶  RUN PREDICTION",
                      self.run_prediction, accent=ACCENT_GREEN,
                      width=22).grid(row=len(fields), column=0,
                                     padx=16, pady=16, sticky="w")

        right_col = tk.Frame(outer, bg=BG_PANEL)
        right_col.pack(side="left", fill="both", expand=True)

        section_label(right_col, "PREDICTION RESULT").pack(fill="x", pady=(4, 8))
        self.pred_result_frame = tk.Frame(right_col, bg=BG_CARD,
                                          highlightthickness=2,
                                          highlightbackground=BORDER_DIM)
        self.pred_result_frame.pack(fill="both", expand=True, pady=(0, 8))
        self.pred_result_label = tk.Label(self.pred_result_frame,
                                          text="—", bg=BG_CARD, fg=TEXT_DIM,
                                          font=("Consolas", 32, "bold"))
        self.pred_result_label.pack(expand=True)

        self.pred_prob_var = tk.StringVar(value="")
        tk.Label(right_col, textvariable=self.pred_prob_var,
                 bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_BODY).pack()

        section_label(right_col, "AI INSIGHTS").pack(fill="x", pady=(10, 4))
        self.insights_text = tk.Text(right_col, height=9, bg=BG_CARD,
                                     fg=ACCENT_CYAN, font=("Consolas", 9),
                                     relief="flat", state="disabled",
                                     highlightthickness=1,
                                     highlightbackground=BORDER_DIM, wrap="word")
        self.insights_text.pack(fill="both", expand=True)

    # ── TAB: HEALTH ──────────────────────────
    def _build_tab_health(self):
        p = self.tab_health
        section_label(p, "MACHINE HEALTH INDEX COMPUTATION").pack(
            fill="x", padx=16, pady=(14, 10))

        top = tk.Frame(p, bg=BG_PANEL)
        top.pack(fill="x", padx=16, pady=(0, 8))
        styled_button(top, "⟳  COMPUTE HEALTH INDEX",
                      self.compute_health_index,
                      accent=ACCENT_CYAN, width=24).pack(side="left", padx=(0, 12))
        self.health_status_var = tk.StringVar(value="Load and train model first.")
        tk.Label(top, textvariable=self.health_status_var,
                 bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_BODY).pack(side="left")

        gauge_frame = tk.Frame(p, bg=BG_PANEL)
        gauge_frame.pack(fill="x", padx=16, pady=(0, 8))
        self.health_canvas = tk.Canvas(gauge_frame, width=320, height=180,
                                       bg=BG_DEEP, highlightthickness=1,
                                       highlightbackground=BORDER_DIM)
        self.health_canvas.pack(side="left", padx=(0, 16))
        self._draw_gauge(None)

        hcf = tk.Frame(gauge_frame, bg=BG_PANEL)
        hcf.pack(side="left", fill="both", expand=True)
        self.hi_card  = MetricCard(hcf, "Health Index", "—", "/ 100", accent=ACCENT_GREEN)
        self.hi_card.pack(side="left", expand=True, fill="both", padx=4)
        self.hi_risk  = MetricCard(hcf, "Risk Level",   "—", "",      accent=ACCENT_WARN)
        self.hi_risk.pack(side="left", expand=True, fill="both", padx=4)
        self.hi_avail = MetricCard(hcf, "Availability", "—", "%",     accent=ACCENT_CYAN)
        self.hi_avail.pack(side="left", expand=True, fill="both", padx=4)

        section_label(p, "HEALTH METRICS BREAKDOWN").pack(fill="x", padx=16, pady=(4, 4))
        self.health_text = tk.Text(p, height=10, bg=BG_CARD,
                                   fg=TEXT_PRIMARY, font=("Consolas", 9),
                                   relief="flat", state="disabled",
                                   highlightthickness=1, highlightbackground=BORDER_DIM,
                                   wrap="word")
        self.health_text.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    def _draw_gauge(self, value):
        c = self.health_canvas
        c.delete("all")
        cx, cy, r = 160, 155, 110
        c.create_arc(cx-r, cy-r, cx+r, cy+r, start=0, extent=180,
                     outline=BORDER_DIM, width=16, style="arc")
        if value is not None:
            pct   = max(0, min(100, value)) / 100
            color = (ACCENT_GREEN if value >= 70 else
                     ACCENT_WARN  if value >= 40 else ACCENT_RED)
            c.create_arc(cx-r, cy-r, cx+r, cy+r, start=0,
                         extent=int(180*pct), outline=color,
                         width=14, style="arc")
            c.create_text(cx, cy-30, text=f"{value:.1f}",
                          fill=color, font=("Consolas", 28, "bold"))
            c.create_text(cx, cy-8, text="HEALTH INDEX",
                          fill=TEXT_DIM, font=("Consolas", 8))
        else:
            c.create_text(cx, cy-30, text="—",
                          fill=TEXT_DIM, font=("Consolas", 28, "bold"))
            c.create_text(cx, cy-8, text="HEALTH INDEX",
                          fill=TEXT_DIM, font=("Consolas", 8))

    # ── TAB: VISUALIZATIONS ──────────────────
    def _build_tab_viz(self):
        p = self.tab_viz
        ctrl = tk.Frame(p, bg=BG_PANEL)
        ctrl.pack(fill="x", padx=16, pady=(12, 8))
        section_label(ctrl, "ANALYTICS VISUALIZATIONS").pack(side="left")

        for lbl, cmd in [
            ("Confusion Matrix",    self.plot_confusion_matrix),
            ("Feature Importance",  self.plot_feature_importance),
            ("Failure Distribution",self.plot_failure_distribution),
            ("Correlation Heatmap", self.plot_correlation),
        ]:
            styled_button(ctrl, lbl, cmd, accent=ACCENT_BLUE, width=20).pack(
                side="left", padx=4)

        self.fig_frame = tk.Frame(p, bg=BG_DEEP,
                                   highlightthickness=1,
                                   highlightbackground=BORDER_DIM)
        self.fig_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        tk.Label(self.fig_frame, text="Select a visualization above",
                 bg=BG_DEEP, fg=TEXT_DIM,
                 font=("Consolas", 13)).pack(expand=True)

    # ── TAB: REPORT ──────────────────────────
    def _build_tab_report(self):
        p = self.tab_report
        section_label(p, "MAINTENANCE REPORT").pack(fill="x", padx=16, pady=(14, 8))

        ctrl = tk.Frame(p, bg=BG_PANEL)
        ctrl.pack(fill="x", padx=16, pady=(0, 8))
        styled_button(ctrl, "⟳  GENERATE REPORT",
                      self.generate_report, accent=ACCENT_CYAN, width=20).pack(
            side="left", padx=(0, 8))
        styled_button(ctrl, "⊕  EXPORT TO FILE",
                      self.export_report, accent=ACCENT_GREEN, width=18).pack(side="left")

        self.report_text = tk.Text(p, bg=BG_CARD, fg=TEXT_PRIMARY,
                                   font=("Consolas", 9), relief="flat",
                                   state="disabled",
                                   highlightthickness=1,
                                   highlightbackground=BORDER_DIM, wrap="word")
        scroll = ttk.Scrollbar(p, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y", padx=(0, 16))
        self.report_text.pack(fill="both", expand=True, padx=(16, 0), pady=(0, 12))

    # ── TAB: 3D ARCHITECTURE ─────────────────
    def _build_tab_arch(self):
        p = self.tab_arch
        self._arch_view = ArchitectureView3D(p)

    # ─────────────────────────────────────────
    #  ML LOGIC
    # ─────────────────────────────────────────
    def set_status(self, msg):
        self.status_var.set(f"◉  {msg}")
        self.root.update_idletasks()

    def log_train(self, msg):
        self.train_log.configure(state="normal")
        self.train_log.insert("end", msg + "\n")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")
        self.root.update_idletasks()

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            self.set_status(f"Dataset loaded: {path.split('/')[-1]}")
            self.data_info_var.set(
                f"  Rows: {len(self.df)}    Columns: {len(self.df.columns)}    "
                f"File: {path.split('/')[-1]}")
            self.card_samples.update_value(len(self.df))
            if "Machine failure" in self.df.columns:
                self.card_failures.update_value(int(self.df["Machine failure"].sum()))
            self._populate_tree()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _populate_tree(self):
        self.data_tree.delete(*self.data_tree.get_children())
        cols = list(self.df.columns)
        self.data_tree["columns"] = cols
        for c in cols:
            self.data_tree.heading(c, text=c)
            self.data_tree.column(c, width=110, anchor="center")
        for _, row in self.df.head(200).iterrows():
            self.data_tree.insert("", "end", values=list(row))

    def clear_data(self):
        self.df = self.model = None
        self.data_tree.delete(*self.data_tree.get_children())
        self.data_info_var.set("No dataset loaded.")
        for card in [self.card_samples, self.card_failures,
                     self.card_accuracy, self.card_health, self.card_risk]:
            card.update_value("—")
        self.set_status("Data cleared.")

    def train_model(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        try:
            n_est   = int(self.n_est_var.get())
            max_dep = int(self.max_depth_var.get())
            split   = float(self.test_split_var.get()) / 100
            seed    = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Parameter Error", "Invalid model parameters.")
            return

        def _train():
            self.log_train("━" * 52)
            self.log_train("[ TRAINING SESSION START ]")
            self.log_train(f"  N Estimators : {n_est}")
            self.log_train(f"  Max Depth    : {max_dep}")
            self.log_train(f"  Test Split   : {split*100:.0f}%")
            self.log_train(f"  Random Seed  : {seed}")
            self.log_train("━" * 52)

            df     = self.df.copy()
            target = "Machine failure"
            if target not in df.columns:
                self.root.after(0, lambda: messagebox.showerror(
                    "Column Error", "Column 'Machine failure' not found."))
                return

            self.le = LabelEncoder()
            if "Type" in df.columns:
                df["Type"] = self.le.fit_transform(df["Type"].astype(str))

            drop_cols = [target, "UDI", "Product ID",
                         "TWF", "HDF", "PWF", "OSF", "RNF"]
            self.feature_cols = [c for c in df.columns
                                  if c not in drop_cols and df[c].dtype != object]
            self.log_train(f"  Features: {self.feature_cols}")

            X = df[self.feature_cols].fillna(0)
            y = df[target]

            self.scaler = StandardScaler()
            X_s = self.scaler.fit_transform(X)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_s, y, test_size=split, random_state=seed)
            self.X_test = X_te
            self.y_test = y_te

            self.log_train("  Training Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_dep,
                random_state=seed, n_jobs=-1)
            self.model.fit(X_tr, y_tr)

            self.y_pred  = self.model.predict(X_te)
            self.accuracy = accuracy_score(y_te, self.y_pred)
            report = classification_report(y_te, self.y_pred,
                                           target_names=["No Failure", "Failure"])
            self.log_train(f"\n  Accuracy: {self.accuracy*100:.2f}%")
            self.log_train("  Training complete.")
            self.log_train("━" * 52)
            self.root.after(0, lambda: self._post_train(report))

        threading.Thread(target=_train, daemon=True).start()

    def _post_train(self, report):
        acc = f"{self.accuracy*100:.2f}"
        self.card_accuracy.update_value(acc)
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", report)
        self.results_text.configure(state="disabled")
        self.set_status(f"Model trained. Accuracy: {acc}%")

    def run_prediction(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        try:
            air_temp  = float(self.pred_vars["air_temp"].get())
            proc_temp = float(self.pred_vars["proc_temp"].get())
            rot_speed = float(self.pred_vars["rot_speed"].get())
            torque    = float(self.pred_vars["torque"].get())
            tool_wear = float(self.pred_vars["tool_wear"].get())
            mtype     = self.pred_vars["type"].get()

            type_enc = self.le.transform([mtype])[0] if self.le else 0
            sample   = {
                "Type":                    type_enc,
                "Air temperature [K]":     air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]":  rot_speed,
                "Torque [Nm]":             torque,
                "Tool wear [min]":         tool_wear,
            }
            X      = np.array([[sample.get(f, 0) for f in self.feature_cols]])
            X_s    = self.scaler.transform(X)
            pred   = self.model.predict(X_s)[0]
            proba  = self.model.predict_proba(X_s)[0]
            fp     = proba[1] * 100 if len(proba) > 1 else proba[0] * 100
            nfp    = proba[0] * 100

            if pred == 1:
                self.pred_result_label.configure(text="⚠  FAILURE PREDICTED", fg=ACCENT_RED)
                self.pred_result_frame.configure(highlightbackground=ACCENT_RED)
                risk = "HIGH RISK"
            else:
                self.pred_result_label.configure(text="✔  NORMAL OPERATION", fg=ACCENT_GREEN)
                self.pred_result_frame.configure(highlightbackground=ACCENT_GREEN)
                risk = "LOW RISK"

            self.pred_prob_var.set(
                f"Failure Probability: {fp:.1f}%  |  Normal: {nfp:.1f}%")
            self.card_risk.update_value(risk)
            self._generate_insights(air_temp, proc_temp, rot_speed, torque,
                                    tool_wear, pred, fp)
            self.set_status(
                f"Prediction: {'FAILURE' if pred else 'NORMAL'}  | Probability: {fp:.1f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _generate_insights(self, air_temp, proc_temp, rot_speed,
                           torque, tool_wear, pred, fail_prob):
        insights = []
        td = proc_temp - air_temp
        if td > 10:
            insights.append(f"⚠ High temp differential ({td:.1f}K): thermal failure risk.")
        if rot_speed < 1200 or rot_speed > 2800:
            insights.append(f"⚠ Rotational speed {rot_speed:.0f} rpm outside optimal range.")
        if torque > 60:
            insights.append(f"⚠ High torque ({torque:.1f} Nm): possible overload.")
        if tool_wear > 150:
            insights.append(f"⚠ Tool wear ({tool_wear:.0f} min): replacement recommended.")
        power = 2 * math.pi * rot_speed * torque / 60
        if power > 9000:
            insights.append(f"⚠ High power ({power:.0f} W): thermal stress risk.")
        if not insights:
            insights.append("✔ All parameters within acceptable operating range.")
        insights.append(
            "🔴 URGENT: Immediate inspection recommended." if fail_prob > 70 else
            "🟡 CAUTION: Schedule preventive maintenance soon." if fail_prob > 40 else
            "🟢 System stable. Continue standard monitoring.")
        self.insights_text.configure(state="normal")
        self.insights_text.delete("1.0", "end")
        for ins in insights:
            self.insights_text.insert("end", ins + "\n\n")
        self.insights_text.configure(state="disabled")

    def compute_health_index(self):
        if self.df is None or self.model is None:
            messagebox.showwarning("Not Ready", "Load data and train model first.")
            return
        try:
            df = self.df.copy()
            if "Type" in df.columns and self.le is not None:
                df["Type"] = self.le.transform(
                    df["Type"].astype(str).map(
                        lambda x: x if x in self.le.classes_ else self.le.classes_[0]))
            X  = df[self.feature_cols].fillna(0)
            Xs = self.scaler.transform(X)
            probs      = self.model.predict_proba(Xs)
            fail_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            hs         = (1 - fail_probs) * 100
            self.health_index = hs.mean()

            tw_col  = next((c for c in df.columns if "tool wear" in c.lower()), None)
            tmp_col = next((c for c in df.columns if "air temp"  in c.lower()), None)
            tor_col = next((c for c in df.columns if "torque"    in c.lower()), None)
            tw_h  = (1 - df[tw_col].clip(0,300)/300).mean()*100 if tw_col  else 75.0
            tmp_h = (1 - ((df[tmp_col]-295).abs().clip(0,15)/15)).mean()*100 if tmp_col else 80.0
            tor_h = (1 - (df[tor_col].clip(0,80)/80)).mean()*100 if tor_col else 70.0
            avail = min(100, self.health_index + 5)
            risk  = ("LOW" if self.health_index >= 75 else
                     "MEDIUM" if self.health_index >= 50 else "HIGH")

            self.card_health.update_value(f"{self.health_index:.1f}")
            self.card_risk.update_value(risk)
            self.hi_card.update_value(f"{self.health_index:.1f}")
            self.hi_risk.update_value(risk)
            self.hi_avail.update_value(f"{avail:.1f}")
            self._draw_gauge(self.health_index)
            self.health_status_var.set(f"Computed on {len(df)} machines.")

            rpt = (
                f"  Overall Health Index  : {self.health_index:.2f} / 100\n"
                f"  Risk Level            : {risk}\n"
                f"  Estimated Availability: {avail:.1f}%\n\n"
                f"  Sub-System Scores\n  {'─'*40}\n"
                f"  Tool Wear Health      : {tw_h:.2f}%\n"
                f"  Temperature Health    : {tmp_h:.2f}%\n"
                f"  Torque Health         : {tor_h:.2f}%\n"
                f"  ML-Based Score        : {self.health_index:.2f}%\n\n"
                f"  Failure Probability Distribution\n  {'─'*40}\n"
                f"  Mean Failure Prob     : {fail_probs.mean()*100:.2f}%\n"
                f"  Max Failure Prob      : {fail_probs.max()*100:.2f}%\n"
                f"  Min Failure Prob      : {fail_probs.min()*100:.2f}%\n"
                f"  Std Deviation         : {fail_probs.std()*100:.2f}%\n"
            )
            self.health_text.configure(state="normal")
            self.health_text.delete("1.0", "end")
            self.health_text.insert("end", rpt)
            self.health_text.configure(state="disabled")
            self.set_status(f"Health Index: {self.health_index:.1f} | Risk: {risk}")
        except Exception as e:
            messagebox.showerror("Health Index Error", str(e))

    # ── VISUALIZATIONS ───────────────────────
    def _clear_fig_frame(self):
        for w in self.fig_frame.winfo_children():
            w.destroy()

    def _embed_fig(self, fig):
        self._clear_fig_frame()
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _apply_plt_style(self):
        plt.rcParams.update(PLT_STYLE)

    def plot_confusion_matrix(self):
        if self.y_test is None:
            messagebox.showwarning("No Results", "Train model first.")
            return
        self._apply_plt_style()
        cm  = confusion_matrix(self.y_test, self.y_pred)
        fig = Figure(figsize=(6, 5), facecolor=BG_DEEP)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        im  = ax.imshow(cm, cmap="Blues", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Failure", "Failure"], color=TEXT_MUTED)
        ax.set_yticklabels(["No Failure", "Failure"], color=TEXT_MUTED)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=TEXT_PRIMARY, fontsize=16,
                        fontweight="bold", fontfamily="Consolas")
        ax.set_title("Confusion Matrix", color=ACCENT_CYAN,
                     fontfamily="Consolas", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", color=TEXT_MUTED, fontfamily="Consolas")
        ax.set_ylabel("Actual",    color=TEXT_MUTED, fontfamily="Consolas")
        fig.tight_layout()
        self._embed_fig(fig)

    def plot_feature_importance(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train model first.")
            return
        self._apply_plt_style()
        imp  = self.model.feature_importances_
        idx  = np.argsort(imp)[::-1]
        fs   = [self.feature_cols[i] for i in idx]
        iv   = imp[idx]
        fig  = Figure(figsize=(8, 5), facecolor=BG_DEEP)
        ax   = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        bars = ax.barh(fs, iv,
                       color=[ACCENT_BLUE if v < 0.3 else ACCENT_CYAN for v in iv],
                       edgecolor="none")
        for bar, val in zip(bars, iv):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", color=TEXT_PRIMARY,
                    fontsize=8, fontfamily="Consolas")
        ax.set_title("Feature Importance", color=ACCENT_CYAN,
                     fontfamily="Consolas", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score", color=TEXT_MUTED, fontfamily="Consolas")
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        fig.tight_layout()
        self._embed_fig(fig)

    def plot_failure_distribution(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load data first.")
            return
        self._apply_plt_style()
        target = "Machine failure"
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Column '{target}' not found.")
            return
        counts = self.df[target].value_counts()
        fig  = Figure(figsize=(8, 5), facecolor=BG_DEEP)
        ax   = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        vals = [counts.get(0, 0), counts.get(1, 0)]
        bars = ax.bar(["No Failure", "Failure"], vals,
                      color=[ACCENT_GREEN, ACCENT_RED],
                      edgecolor="none", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", color=TEXT_PRIMARY,
                    fontsize=11, fontweight="bold", fontfamily="Consolas")
        ax.set_title("Failure Distribution", color=ACCENT_CYAN,
                     fontfamily="Consolas", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count", color=TEXT_MUTED, fontfamily="Consolas")
        ax.tick_params(colors=TEXT_MUTED)
        fig.tight_layout()
        self._embed_fig(fig)

    def plot_correlation(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load data first.")
            return
        self._apply_plt_style()
        num_df = self.df.select_dtypes(include=[np.number])
        corr   = num_df.corr()
        fig    = Figure(figsize=(9, 7), facecolor=BG_DEEP)
        ax     = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        im     = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right",
                           color=TEXT_MUTED, fontsize=7, fontfamily="Consolas")
        ax.set_yticklabels(corr.columns, color=TEXT_MUTED,
                           fontsize=7, fontfamily="Consolas")
        ax.set_title("Correlation Heatmap", color=ACCENT_CYAN,
                     fontfamily="Consolas", fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._embed_fig(fig)

    # ── REPORT ───────────────────────────────
    def generate_report(self):
        import datetime
        if self.model is None:
            messagebox.showwarning("No Model", "Train model first.")
            return
        now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hi_str  = f"{self.health_index:.2f}" if self.health_index else "N/A"
        acc_str = f"{self.accuracy*100:.2f}%" if self.accuracy else "N/A"
        risk_str = (
            "HIGH"   if self.health_index and self.health_index < 50 else
            "MEDIUM" if self.health_index and self.health_index < 75 else "LOW"
        ) if self.health_index else "N/A"

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║   INDUSTRIAL PREDICTIVE MAINTENANCE SYSTEM — ANALYSIS REPORT     ║
╚══════════════════════════════════════════════════════════════════╝

  Report Generated  : {now}
  ML Engine         : Random Forest Classifier
  Dataset           : AI4I 2020 Predictive Maintenance Dataset

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model Accuracy    : {acc_str}
  N Estimators      : {self.n_est_var.get()}
  Max Depth         : {self.max_depth_var.get()}
  Test Split        : {self.test_split_var.get()}%
  Features Used     : {', '.join(self.feature_cols) if self.feature_cols else 'N/A'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MACHINE HEALTH ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Overall Health Index : {hi_str} / 100
  Risk Level           : {risk_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Records     : {len(self.df) if self.df is not None else 'N/A'}
  Total Features    : {len(self.df.columns) if self.df is not None else 'N/A'}
  Failure Cases     : {int(self.df['Machine failure'].sum()) if self.df is not None and 'Machine failure' in self.df.columns else 'N/A'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SYSTEM ARCHITECTURE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pipeline Nodes    : {len(ARCH_NODES)} (8 processing nodes)
  Pipeline Edges    : {len(ARCH_EDGES)} (data flow connections)
  Architecture      : 5-layer hierarchical ML pipeline
  Visualization     : Live 3D rotating network diagram

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MAINTENANCE RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {"⚠ URGENT: Schedule immediate inspection — health index critical." if self.health_index and self.health_index < 50 else "✔ Continue standard monitoring schedule."}
  Perform regular tool wear checks every 100 operational hours.
  Monitor temperature differential between air and process streams.
  Ensure rotational speed stays within 1200–2800 rpm operating band.
  Log torque anomalies above 60 Nm for root cause analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Industrial Predictive Maintenance System  |  AI-Powered Analytics
  Final Year Project  —  Version 3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", report)
        self.report_text.configure(state="disabled")
        self.set_status("Report generated.")

    def export_report(self):
        content = self.report_text.get("1.0", "end").strip()
        if not content:
            messagebox.showwarning("Empty Report", "Generate report first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Report")
        if path:
            with open(path, "w") as f:
                f.write(content)
            messagebox.showinfo("Exported", f"Report saved to:\n{path}")
            self.set_status(f"Report exported: {path.split('/')[-1]}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.withdraw()

    splash = SplashScreen(root)

    def launch_main():
        root.deiconify()
        PredictiveMaintenanceSystem(root)

    root.after(3800, launch_main)
    root.mainloop()


if __name__ == "__main__":
    main()
