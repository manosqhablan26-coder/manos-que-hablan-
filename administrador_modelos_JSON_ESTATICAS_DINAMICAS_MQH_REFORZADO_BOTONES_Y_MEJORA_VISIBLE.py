import cv2
import mediapipe as mp
import numpy as np
import json
import os
import shutil
import subprocess
import sys
import time
import threading
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from pathlib import Path
from collections import deque


APP_TITLE = "Administrador de modelos JSON · Manos que Hablan"
MODEL_VERSION = 1
DYNAMIC_MIN_FRAMES = 8
DYNAMIC_MIN_MOTION = 0.030
DYNAMIC_SEQUENCE_STEPS = 16


def default_models_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home())
        return root / "ManosQueHablan" / "modelos_entrenados"
    return Path(__file__).resolve().parent / "modelos_entrenados"


def sanitize_label(name: str) -> str:
    name = str(name or "").strip().upper()
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name)
    safe = safe.strip("._-")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "SENA"


def is_dynamic(sample) -> bool:
    return isinstance(sample, dict) and (
        str(sample.get("type", "")).lower() == "dynamic"
        or isinstance(sample.get("frames"), list)
    )


def validate_hand(hand) -> bool:
    if not isinstance(hand, dict):
        return False
    points = hand.get("landmarks")
    if not isinstance(points, list) or len(points) != 21:
        return False
    try:
        for lm in points:
            if isinstance(lm, dict):
                float(lm["x"]); float(lm["y"]); float(lm["z"])
            else:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def validate_static(sample) -> bool:
    if not isinstance(sample, dict) or is_dynamic(sample):
        return False
    hands = sample.get("hands")
    return isinstance(hands, list) and 1 <= len(hands) <= 2 and all(validate_hand(h) for h in hands)


# ==========================================================
# REFUERZO SINTÉTICO DE MUESTRAS ESTÁTICAS
# ==========================================================
# Estas variantes NO sustituyen las capturas reales. Solo añaden pequeñas
# tolerancias geométricas para que una misma seña no dependa de repetir
# exactamente la postura usada durante el entrenamiento.
AUGMENTATION_VERSION = 1
AUGMENT_ROTATION_DEG = 1.3
AUGMENT_AXIS_SCALE = 0.005
AUGMENT_LOCAL_JITTER = 0.004
AUGMENT_Z_JITTER = 0.003


def _clamp(value, low, high):
    return max(low, min(high, value))


def _augment_hand_landmarks(hand):
    """Crea una variante muy suave conservando muñeca, mano y handedness."""
    if not validate_hand(hand):
        return None

    points = hand.get("landmarks", [])
    wrist = points[0]
    wx, wy, wz = float(wrist["x"]), float(wrist["y"]), float(wrist["z"])

    # Escala aproximada de la mano. El ruido queda expresado como una fracción
    # del tamaño real detectado, no como un desplazamiento fijo en pantalla.
    hand_scale = max(
        (
            ((float(lm["x"]) - wx) ** 2 +
             (float(lm["y"]) - wy) ** 2 +
             (float(lm["z"]) - wz) ** 2) ** 0.5
            for lm in points
        ),
        default=0.0,
    )
    if hand_scale < 1e-6:
        return None

    angle = math.radians(random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    scale_x = 1.0 + random.uniform(-AUGMENT_AXIS_SCALE, AUGMENT_AXIS_SCALE)
    scale_y = 1.0 + random.uniform(-AUGMENT_AXIS_SCALE, AUGMENT_AXIS_SCALE)
    scale_z = 1.0 + random.uniform(-AUGMENT_AXIS_SCALE, AUGMENT_AXIS_SCALE)

    new_points = []
    for idx, lm in enumerate(points):
        x, y, z = float(lm["x"]), float(lm["y"]), float(lm["z"])
        if idx == 0:
            # La muñeca se conserva como ancla. El reconocedor ya elimina
            # traslación y tamaño global al vectorizar la mano.
            nx, ny, nz = wx, wy, wz
        else:
            dx = (x - wx) * scale_x
            dy = (y - wy) * scale_y
            dz = (z - wz) * scale_z

            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a

            jxy = hand_scale * AUGMENT_LOCAL_JITTER
            jz = hand_scale * AUGMENT_Z_JITTER
            nx = wx + rx + random.uniform(-jxy, jxy)
            ny = wy + ry + random.uniform(-jxy, jxy)
            nz = wz + dz + random.uniform(-jz, jz)

            # x/y de MediaPipe son coordenadas normalizadas de imagen.
            nx = _clamp(nx, 0.0, 1.0)
            ny = _clamp(ny, 0.0, 1.0)

        new_points.append({"x": float(nx), "y": float(ny), "z": float(nz)})

    return {
        "handedness": str(hand.get("handedness", "Unknown")),
        "landmarks": new_points,
    }


def augment_static_sample(sample, variant_index=1):
    """Genera una muestra estática sintética compatible con el JSON actual."""
    if not validate_static(sample):
        return None

    new_hands = []
    for hand in sample.get("hands", []):
        augmented = _augment_hand_landmarks(hand)
        if augmented is None:
            return None
        new_hands.append(augmented)

    output = dict(sample)
    output["hands"] = new_hands
    output["timestamp"] = time.time()
    output["_mqh_augmented"] = True
    output["_mqh_augmentation_version"] = AUGMENTATION_VERSION
    output["_mqh_variant"] = int(variant_index)
    # No heredamos un marcador previo si el JSON fue manipulado manualmente.
    output.pop("frames", None)
    output.pop("type", None)
    return output


def validate_dynamic(sample) -> bool:
    if not is_dynamic(sample):
        return False
    frames = sample.get("frames")
    if not isinstance(frames, list) or len(frames) < DYNAMIC_MIN_FRAMES:
        return False
    valid_frames = 0
    expected_hands = None
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        hands = frame.get("hands")
        if not isinstance(hands, list) or not hands:
            continue
        hands = [h for h in hands if validate_hand(h)]
        if not hands:
            continue
        if expected_hands is None:
            expected_hands = min(2, len(hands))
        if len(hands) >= expected_hands:
            valid_frames += 1
    return valid_frames >= DYNAMIC_MIN_FRAMES


def load_model(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("El JSON raíz no es un objeto")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("El JSON no contiene una lista 'samples'")
    label = str(data.get("label", path.stem)).strip().upper() or path.stem.upper()
    return {"version": int(data.get("version", MODEL_VERSION)), "label": label, "samples": samples}


def atomic_save_model(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = str(data.get("label", path.stem)).strip().upper() or path.stem.upper()
    samples = []
    for sample in data.get("samples", []):
        if not isinstance(sample, dict):
            continue
        item = dict(sample)
        item["label"] = label
        samples.append(item)
    output = {"version": MODEL_VERSION, "label": label, "samples": samples}
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    temp.replace(path)


def backup_file(path: Path):
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = path.parent / "respaldos_admin"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"{path.stem}_{stamp}.json"
    shutil.copy2(path, dst)
    return dst


def order_motion_hands(hands_data):
    prepared = []
    for index, hand in enumerate(hands_data or []):
        if not validate_hand(hand):
            continue
        try:
            points = [(float(lm["x"]), float(lm["y"]), float(lm["z"])) for lm in hand["landmarks"]]
        except Exception:
            continue
        handedness = str(hand.get("handedness", "Unknown"))
        order = 0 if handedness == "Left" else 1 if handedness == "Right" else 2
        prepared.append(((order, points[0][0], index), handedness, points))
    prepared.sort(key=lambda item: item[0])
    return [(handedness, points) for _, handedness, points in prepared[:2]]


def motion_energy(sequence):
    if not sequence or len(sequence) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for prev, cur in zip(sequence, sequence[1:]):
        if len(prev) != len(cur) or not cur:
            continue
        squared = sum((a - b) * (a - b) for a, b in zip(cur, prev))
        total += (squared / len(cur)) ** 0.5
        pairs += 1
    return total / max(1, pairs)


def vectorize_motion(frames, target_steps=DYNAMIC_SEQUENCE_STEPS):
    raw_frames = []
    for frame in frames or []:
        hands_data = frame.get("hands", []) if isinstance(frame, dict) else frame
        ordered = order_motion_hands(hands_data)
        if ordered:
            raw_frames.append(ordered)
    if len(raw_frames) < 2:
        return None

    hand_count = len(raw_frames[0])
    raw_frames = [frame for frame in raw_frames if len(frame) == hand_count]
    if len(raw_frames) < 2:
        return None

    bases = []
    for _, points in raw_frames[0]:
        wx, wy, wz = points[0]
        scale = max(
            (((x - wx) ** 2 + (y - wy) ** 2 + (z - wz) ** 2) ** 0.5 for x, y, z in points),
            default=0.0,
        )
        if scale < 1e-6:
            return None
        bases.append(((wx, wy, wz), scale))

    sequence = []
    for frame in raw_frames:
        vector = []
        for hand_index, (_, points) in enumerate(frame):
            (bx, by, bz), scale = bases[hand_index]
            for x, y, z in points:
                vector.extend(((x - bx) / scale, (y - by) / scale, (z - bz) / scale))
        sequence.append(vector)

    if target_steps and len(sequence) != target_steps:
        last = len(sequence) - 1
        if target_steps <= 1:
            sequence = [sequence[-1]]
        else:
            indices = [round(i * last / (target_steps - 1)) for i in range(target_steps)]
            sequence = [sequence[i] for i in indices]

    return {"hand_count": hand_count, "sequence": sequence, "motion": motion_energy(sequence)}


class MQHModelAdmin:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        # Ajuste SOLO visual: la ventana nunca excede el área útil aproximada
        # de la pantalla. Se conserva intacta la lógica/callbacks de captura.
        screen_w = max(800, int(self.root.winfo_screenwidth()))
        screen_h = max(600, int(self.root.winfo_screenheight()))
        # La ventana se adapta a la pantalla real y nunca queda más ancha/alta
        # que el área disponible. Cambio únicamente visual.
        window_w = min(1320, max(820, screen_w - 40), screen_w)
        window_h = min(820, max(560, screen_h - 90), screen_h)
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(min(820, window_w), min(560, window_h))

        self.models_dir = default_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = None
        self.current_data = None

        self.cap = None
        self.camera_running = False
        self.camera_index = 0
        self.last_frame = None
        self.latest_hands = []
        self.latest_frame_id = 0
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands_detector = None

        self.capture_mode = None
        self.capture_after_id = None
        self.capture_countdown_until = 0.0
        self.capture_started_at = 0.0
        self.capture_last_saved = 0.0
        self.capture_target = 0
        self.capture_saved = 0
        # Último frame usado por el lote estático. Evita guardar varias veces
        # exactamente el mismo resultado de MediaPipe.
        self.capture_static_last_frame_id = -1
        self.capture_frames = []
        self.capture_dynamic_rep = 0
        self.capture_dynamic_total = 0
        self.pending_dynamic_samples = []
        self.pending_static_samples = []

        self.status_var = tk.StringVar(value="Listo")
        self.folder_var = tk.StringVar(value=str(self.models_dir))
        self.train_label_var = tk.StringVar(value="")
        self.train_type_var = tk.StringVar(value="Estática")
        self.hand_filter_var = tk.StringVar(value="Todas")
        self.static_count_var = tk.IntVar(value=30)
        self.static_interval_var = tk.DoubleVar(value=0.08)
        self.dynamic_duration_var = tk.DoubleVar(value=1.20)
        self.dynamic_reps_var = tk.IntVar(value=5)
        self.camera_index_var = tk.IntVar(value=0)

        self._build_ui()
        self.refresh_models()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Carpeta de modelos:").pack(side="left")
        ttk.Entry(top, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Elegir carpeta", command=self.choose_folder).pack(side="left", padx=3)
        ttk.Button(top, text="Abrir carpeta", command=self.open_folder).pack(side="left", padx=3)
        ttk.Button(top, text="Actualizar", command=self.refresh_models).pack(side="left", padx=3)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=2)
        body.add(right, weight=3)

        ttk.Label(left, text="Modelos JSON", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 5))
        columns = ("label", "static", "dynamic", "total", "state")
        self.model_tree = ttk.Treeview(left, columns=columns, show="headings", height=16, selectmode="browse")
        for col, text, width in [
            ("label", "Seña", 145), ("static", "Estáticas", 75), ("dynamic", "Dinámicas", 80),
            ("total", "Total", 60), ("state", "Estado", 100)
        ]:
            self.model_tree.heading(col, text=text)
            self.model_tree.column(col, width=width, anchor="center" if col != "label" else "w")
        self.model_tree.pack(fill="both", expand=True)
        self.model_tree.bind("<<TreeviewSelect>>", self.on_model_select)

        model_buttons = ttk.Frame(left)
        model_buttons.pack(fill="x", pady=(7, 0))
        model_actions = [
            ("Nueva", self.new_model), ("Renombrar", self.rename_model),
            ("Eliminar", self.delete_model), ("Importar JSON", self.import_json),
            ("Exportar", self.export_json),
        ]
        for col in range(3):
            model_buttons.columnconfigure(col, weight=1)
        for i, (text, cmd) in enumerate(model_actions):
            ttk.Button(model_buttons, text=text, command=cmd).grid(
                row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2
            )

        # Refuerzo visible justo debajo de las acciones del modelo.
        # Solo cambia la ubicación visual; la lógica de refuerzo permanece intacta.
        reinforce = ttk.LabelFrame(left, text="Mejorar / reforzar seña seleccionada", padding=(6, 4))
        reinforce.pack(fill="x", pady=(7, 0))
        ttk.Label(
            reinforce,
            text="Crea variantes suaves de las capturas reales (no modifica dinámicas).",
        ).pack(anchor="w", pady=(0, 4))
        reinforce_buttons = ttk.Frame(reinforce)
        reinforce_buttons.pack(fill="x")
        for col in range(2):
            reinforce_buttons.columnconfigure(col, weight=1)
        ttk.Button(reinforce_buttons, text="Mejorar x2", command=lambda: self.reinforce_current_model(2)).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(reinforce_buttons, text="Mejorar x3", command=lambda: self.reinforce_current_model(3)).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(reinforce_buttons, text="Mejorar x5", command=lambda: self.reinforce_current_model(5)).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(reinforce_buttons, text="Todos x3", command=lambda: self.reinforce_all_models(3)).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        detail = ttk.LabelFrame(left, text="Contenido del archivo", padding=6)
        detail.pack(fill="both", expand=True, pady=(8, 0))
        sample_cols = ("n", "type", "hands", "frames", "valid")
        self.sample_tree = ttk.Treeview(detail, columns=sample_cols, show="headings", height=10, selectmode="extended")
        for col, text, width in [
            ("n", "#", 35), ("type", "Tipo", 80), ("hands", "Manos", 55),
            ("frames", "Frames", 60), ("valid", "Válida", 55)
        ]:
            self.sample_tree.heading(col, text=text)
            self.sample_tree.column(col, width=width, anchor="center")
        self.sample_tree.pack(fill="both", expand=True)
        sb = ttk.Frame(detail)
        sb.pack(fill="x", pady=(5, 0))
        for col in range(2):
            sb.columnconfigure(col, weight=1)
        ttk.Button(sb, text="Eliminar muestras", command=self.delete_samples).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(sb, text="Eliminar estáticas", command=lambda: self.delete_samples_by_type("static")).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(sb, text="Eliminar dinámicas", command=lambda: self.delete_samples_by_type("dynamic")).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(sb, text="Validar", command=self.validate_current).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(right, text="Entrenamiento / captura", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 5))
        train_top = ttk.Frame(right)
        train_top.pack(fill="x", pady=(0, 5))
        ttk.Label(train_top, text="Seña:").pack(side="left")
        ttk.Entry(train_top, textvariable=self.train_label_var, width=20).pack(side="left", padx=(4, 10))
        ttk.Label(train_top, text="Tipo:").pack(side="left")
        ttk.Radiobutton(train_top, text="Estática", variable=self.train_type_var, value="Estática").pack(side="left", padx=3)
        ttk.Radiobutton(train_top, text="Dinámica", variable=self.train_type_var, value="Dinámica").pack(side="left", padx=3)
        ttk.Label(train_top, text="Mano:").pack(side="left", padx=(10, 0))
        ttk.Combobox(train_top, textvariable=self.hand_filter_var, values=("Todas", "Izquierda", "Derecha"), state="readonly", width=10).pack(side="left", padx=4)

        camera_bar = ttk.Frame(right)
        camera_bar.pack(fill="x", pady=(0, 5))
        camera_bar.columnconfigure(2, weight=1)
        camera_bar.columnconfigure(3, weight=1)
        ttk.Label(camera_bar, text="Cámara:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(camera_bar, from_=0, to=10, textvariable=self.camera_index_var, width=4).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(camera_bar, text="Iniciar cámara", command=self.start_camera).grid(row=0, column=2, sticky="ew", padx=3)
        ttk.Button(camera_bar, text="Detener", command=self.stop_camera).grid(row=0, column=3, sticky="ew", padx=3)
        ttk.Button(camera_bar, text="Usar modelo seleccionado", command=self.use_selected_label).grid(row=1, column=0, columnspan=4, sticky="ew", padx=3, pady=(4, 0))

        preview_shell = ttk.Frame(right, relief="sunken", borderwidth=1)
        preview_shell.pack(fill="both", expand=True)
        self.preview = ttk.Label(preview_shell, text="Cámara detenida", anchor="center")
        self.preview.pack(fill="both", expand=True)

        options = ttk.LabelFrame(right, text="Opciones de entrenamiento", padding=8)
        options.pack(fill="x", pady=(8, 0))

        row1 = ttk.Frame(options)
        row1.pack(fill="x", pady=2)
        row1.columnconfigure(1, weight=1)
        row1.columnconfigure(3, weight=1)
        ttk.Label(row1, text="Estáticas: cantidad").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row1, from_=1, to=500, textvariable=self.static_count_var, width=6).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(row1, text="intervalo (s)").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Spinbox(row1, from_=0.0, to=1.0, increment=0.01, textvariable=self.static_interval_var, width=6).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Button(row1, text="Capturar 1 estática", command=self.capture_one_static).grid(row=1, column=0, columnspan=2, sticky="ew", padx=3, pady=(4, 0))
        ttk.Button(row1, text="Capturar lote estático", command=self.start_static_batch).grid(row=1, column=2, columnspan=2, sticky="ew", padx=3, pady=(4, 0))

        row2 = ttk.Frame(options)
        row2.pack(fill="x", pady=2)
        row2.columnconfigure(1, weight=1)
        row2.columnconfigure(3, weight=1)
        ttk.Label(row2, text="Dinámica: duración (s)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row2, from_=0.5, to=4.0, increment=0.1, textvariable=self.dynamic_duration_var, width=6).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(row2, text="repeticiones").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Spinbox(row2, from_=1, to=50, textvariable=self.dynamic_reps_var, width=6).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Button(row2, text="Capturar dinámicas", command=self.start_dynamic_capture).grid(row=1, column=0, columnspan=4, sticky="ew", padx=3, pady=(4, 0))

        row3 = ttk.Frame(options)
        row3.pack(fill="x", pady=(6, 0))
        ttk.Button(row3, text="Cancelar captura", command=self.cancel_capture).pack(side="left")
        self.capture_progress = ttk.Progressbar(row3, mode="determinate", maximum=100)
        self.capture_progress.pack(side="left", fill="x", expand=True, padx=8)
        self.capture_label = ttk.Label(row3, text="Sin captura")
        self.capture_label.pack(side="right")

        bottom = ttk.Frame(self.root, padding=(10, 4))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(fill="x")

    def set_status(self, text):
        self.status_var.set(str(text))

    def choose_folder(self):
        path = filedialog.askdirectory(initialdir=str(self.models_dir), title="Carpeta modelos_entrenados")
        if not path:
            return
        self.models_dir = Path(path)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.folder_var.set(str(self.models_dir))
        self.current_path = None
        self.current_data = None
        self.refresh_models()

    def open_folder(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(self.models_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.models_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.models_dir)])
        except Exception as exc:
            self.set_status(f"No pude abrir la carpeta: {exc}")

    def model_stats(self, path):
        try:
            data = load_model(path)
            static_count = sum(1 for s in data["samples"] if validate_static(s))
            dynamic_count = sum(1 for s in data["samples"] if validate_dynamic(s))
            invalid = len(data["samples"]) - static_count - dynamic_count
            state = "OK" if invalid == 0 else f"{invalid} inválida(s)"
            return data["label"], static_count, dynamic_count, len(data["samples"]), state
        except Exception:
            return path.stem.upper(), 0, 0, 0, "JSON dañado"

    def refresh_models(self):
        try:
            typed = Path(self.folder_var.get()).expanduser()
            self.models_dir = typed
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        selected_path = str(self.current_path) if self.current_path else None
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        target_iid = None
        for path in sorted(self.models_dir.glob("*.json"), key=lambda p: p.name.lower()):
            label, st, dy, total, state = self.model_stats(path)
            iid = self.model_tree.insert("", "end", values=(label, st, dy, total, state), tags=(str(path),))
            if selected_path and str(path) == selected_path:
                target_iid = iid
        if target_iid:
            self.model_tree.selection_set(target_iid)
            self.model_tree.focus(target_iid)
            self.model_tree.see(target_iid)
            self.on_model_select()
        else:
            self.refresh_samples()
        self.set_status(f"{len(self.model_tree.get_children())} archivo(s) JSON en {self.models_dir}")

    def selected_model_path(self):
        sel = self.model_tree.selection()
        if not sel:
            return None
        tags = self.model_tree.item(sel[0], "tags")
        if not tags:
            return None
        return Path(tags[0])

    def on_model_select(self, *_):
        path = self.selected_model_path()
        if not path:
            return
        try:
            data = load_model(path)
        except Exception as exc:
            self.current_path = path
            self.current_data = None
            self.refresh_samples()
            self.set_status(f"No se pudo abrir {path.name}: {exc}")
            return
        self.current_path = path
        self.current_data = data
        self.train_label_var.set(data["label"])
        self.refresh_samples()
        self.set_status(f"Seleccionado: {path.name}")

    def refresh_samples(self):
        for item in self.sample_tree.get_children():
            self.sample_tree.delete(item)
        if not self.current_data:
            return
        for idx, sample in enumerate(self.current_data.get("samples", [])):
            dyn = is_dynamic(sample)
            if dyn:
                frames = sample.get("frames", []) if isinstance(sample, dict) else []
                hands = 0
                for fr in frames:
                    if isinstance(fr, dict) and isinstance(fr.get("hands"), list) and fr["hands"]:
                        hands = len(fr["hands"])
                        break
                valid = validate_dynamic(sample)
                values = (idx + 1, "Dinámica", hands, len(frames), "Sí" if valid else "No")
            else:
                hands = len(sample.get("hands", [])) if isinstance(sample, dict) and isinstance(sample.get("hands"), list) else 0
                valid = validate_static(sample)
                values = (idx + 1, "Estática", hands, 1, "Sí" if valid else "No")
            self.sample_tree.insert("", "end", iid=str(idx), values=values)

    def ensure_label(self):
        label = self.train_label_var.get().strip().upper()
        if not label:
            messagebox.showwarning("Falta nombre", "Escribe o selecciona el nombre de la seña.", parent=self.root)
            return None
        self.train_label_var.set(label)
        return label

    def path_for_label(self, label):
        return self.models_dir / f"{sanitize_label(label)}.json"

    def load_or_create_label(self, label):
        path = self.path_for_label(label)
        if path.exists():
            data = load_model(path)
            if data["label"] != label:
                data["label"] = label
            return path, data
        return path, {"version": MODEL_VERSION, "label": label, "samples": []}

    def save_data(self, path, data, make_backup=True):
        if make_backup and path.exists():
            backup_file(path)
        atomic_save_model(path, data)
        self.current_path = path
        self.current_data = load_model(path)
        self.refresh_models()

    def new_model(self):
        name = simpledialog.askstring("Nueva seña", "Nombre de la seña:", parent=self.root)
        if not name:
            return
        label = name.strip().upper()
        path = self.path_for_label(label)
        if path.exists():
            messagebox.showinfo("Ya existe", f"Ya existe {path.name}", parent=self.root)
            return
        atomic_save_model(path, {"version": MODEL_VERSION, "label": label, "samples": []})
        self.current_path = path
        self.train_label_var.set(label)
        self.refresh_models()
        self.set_status(f"Creado {path.name}. Ya puedes entrenarlo.")

    def rename_model(self):
        path = self.selected_model_path()
        if not path:
            return
        try:
            data = load_model(path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        name = simpledialog.askstring("Renombrar seña", "Nuevo nombre:", initialvalue=data["label"], parent=self.root)
        if not name:
            return
        new_label = name.strip().upper()
        new_path = self.path_for_label(new_label)
        if new_path.exists() and new_path.resolve() != path.resolve():
            messagebox.showerror("Ya existe", f"Ya existe {new_path.name}", parent=self.root)
            return
        backup_file(path)
        data["label"] = new_label
        atomic_save_model(new_path, data)
        if new_path.resolve() != path.resolve() and path.exists():
            path.unlink()
        self.current_path = new_path
        self.train_label_var.set(new_label)
        self.refresh_models()
        self.set_status(f"Renombrado a {new_path.name}")

    def delete_model(self):
        path = self.selected_model_path()
        if not path:
            return
        if not messagebox.askyesno("Eliminar modelo", f"¿Eliminar {path.name}?\nSe guardará un respaldo primero.", parent=self.root):
            return
        try:
            backup = backup_file(path)
            path.unlink()
            self.current_path = None
            self.current_data = None
            self.refresh_models()
            self.set_status(f"Eliminado. Respaldo: {backup.name if backup else 'no creado'}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)

    def import_json(self):
        src = filedialog.askopenfilename(title="Importar modelo JSON", filetypes=[("JSON", "*.json")])
        if not src:
            return
        try:
            data = load_model(Path(src))
            label = data["label"]
            dst = self.path_for_label(label)
            if dst.exists() and not messagebox.askyesno("Reemplazar", f"{dst.name} ya existe. ¿Reemplazarlo?", parent=self.root):
                return
            if dst.exists():
                backup_file(dst)
            atomic_save_model(dst, data)
            self.current_path = dst
            self.refresh_models()
            self.set_status(f"Importado: {dst.name}")
        except Exception as exc:
            messagebox.showerror("JSON no válido", str(exc), parent=self.root)

    def export_json(self):
        path = self.selected_model_path()
        if not path:
            return
        dst = filedialog.asksaveasfilename(initialfile=path.name, defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not dst:
            return
        shutil.copy2(path, dst)
        self.set_status(f"Exportado a {dst}")

    def delete_samples(self):
        if not self.current_path or not self.current_data:
            return
        selected = self.sample_tree.selection()
        if not selected:
            return
        indices = sorted((int(i) for i in selected), reverse=True)
        if not messagebox.askyesno("Eliminar muestras", f"¿Eliminar {len(indices)} muestra(s)?", parent=self.root):
            return
        backup_file(self.current_path)
        samples = self.current_data.get("samples", [])
        for idx in indices:
            if 0 <= idx < len(samples):
                del samples[idx]
        atomic_save_model(self.current_path, self.current_data)
        self.current_data = load_model(self.current_path)
        self.refresh_models()
        self.set_status(f"Eliminadas {len(indices)} muestra(s)")

    def delete_samples_by_type(self, kind):
        if not self.current_path or not self.current_data:
            return
        pretty = "estáticas" if kind == "static" else "dinámicas"
        old = self.current_data.get("samples", [])
        keep = [s for s in old if (is_dynamic(s) if kind == "static" else not is_dynamic(s))]
        removed = len(old) - len(keep)
        if removed <= 0:
            self.set_status(f"No hay muestras {pretty} para eliminar.")
            return
        if not messagebox.askyesno("Eliminar muestras", f"¿Eliminar las {removed} muestras {pretty}?", parent=self.root):
            return
        backup_file(self.current_path)
        self.current_data["samples"] = keep
        atomic_save_model(self.current_path, self.current_data)
        self.current_data = load_model(self.current_path)
        self.refresh_models()
        self.set_status(f"Eliminadas {removed} muestras {pretty}")

    def _build_reinforced_samples(self, data, factor):
        """Devuelve variantes SOLO de capturas estáticas originales.

        factor=3 significa: 1 captura real + 2 variantes. Las muestras que ya
        tienen _mqh_augmented se ignoran como fuente para evitar crecimiento
        exponencial al pulsar el botón varias veces.
        """
        factor = max(2, min(5, int(factor)))
        originals = [
            sample for sample in data.get("samples", [])
            if validate_static(sample) and not bool(sample.get("_mqh_augmented"))
        ]
        generated = []
        for sample in originals:
            for variant_index in range(1, factor):
                variant = augment_static_sample(sample, variant_index=variant_index)
                if variant is not None:
                    generated.append(variant)
        return originals, generated

    def reinforce_current_model(self, factor=3):
        if not self.current_path or not self.current_data:
            messagebox.showinfo(
                "Refuerzo",
                "Selecciona primero un modelo JSON.",
                parent=self.root,
            )
            return

        originals, generated = self._build_reinforced_samples(self.current_data, factor)
        if not originals:
            messagebox.showinfo(
                "Refuerzo",
                "Este modelo no tiene capturas estáticas originales válidas para reforzar.",
                parent=self.root,
            )
            return

        # Reemplazamos refuerzos antiguos del mismo archivo y regeneramos desde
        # las capturas reales. Así x2 -> x5 produce exactamente x5, no x10.
        existing = self.current_data.get("samples", [])
        kept = [sample for sample in existing if not bool(sample.get("_mqh_augmented"))]
        old_augmented = len(existing) - len(kept)

        if not messagebox.askyesno(
            "Reforzar modelo",
            f"Seña: {self.current_data['label']}\n\n"
            f"Capturas estáticas reales: {len(originals)}\n"
            f"Factor: x{factor}\n"
            f"Variantes nuevas: {len(generated)}\n"
            f"Refuerzos anteriores a reemplazar: {old_augmented}\n\n"
            "Las capturas reales y las dinámicas se conservarán. ¿Continuar?",
            parent=self.root,
        ):
            return

        backup_file(self.current_path)
        self.current_data["samples"] = kept + generated
        atomic_save_model(self.current_path, self.current_data)
        self.current_data = load_model(self.current_path)
        self.refresh_models()
        self.refresh_samples()
        self.set_status(
            f"{self.current_data['label']}: refuerzo x{factor} listo · "
            f"{len(originals)} reales + {len(generated)} variantes"
        )

    def reinforce_all_models(self, factor=3):
        paths = sorted(self.models_dir.glob("*.json"), key=lambda p: p.name.lower())
        if not paths:
            messagebox.showinfo("Refuerzo", "No hay modelos JSON en esta carpeta.", parent=self.root)
            return

        summary = []
        total_generated = 0
        eligible = 0
        for path in paths:
            try:
                data = load_model(path)
                originals, generated = self._build_reinforced_samples(data, factor)
                if originals:
                    eligible += 1
                    summary.append((path, data, originals, generated))
                    total_generated += len(generated)
            except Exception:
                continue

        if not summary:
            messagebox.showinfo(
                "Refuerzo",
                "No encontré capturas estáticas originales válidas para reforzar.",
                parent=self.root,
            )
            return

        if not messagebox.askyesno(
            "Reforzar todos",
            f"Modelos con estáticas válidas: {eligible}\n"
            f"Factor: x{factor}\n"
            f"Variantes que se crearán: {total_generated}\n\n"
            "Cada JSON tendrá respaldo antes del cambio. ¿Continuar?",
            parent=self.root,
        ):
            return

        changed = 0
        for path, data, originals, generated in summary:
            try:
                existing = data.get("samples", [])
                kept = [sample for sample in existing if not bool(sample.get("_mqh_augmented"))]
                backup_file(path)
                data["samples"] = kept + generated
                atomic_save_model(path, data)
                changed += 1
            except Exception:
                continue

        # Recarga el modelo seleccionado si sigue existiendo.
        if self.current_path and self.current_path.exists():
            try:
                self.current_data = load_model(self.current_path)
            except Exception:
                self.current_data = None
        self.refresh_models()
        self.refresh_samples()
        self.set_status(
            f"Refuerzo x{factor}: {changed}/{eligible} modelos actualizados · "
            f"{total_generated} variantes creadas"
        )

    def validate_current(self):
        if not self.current_data:
            return
        total = len(self.current_data.get("samples", []))
        st = sum(1 for s in self.current_data["samples"] if validate_static(s))
        dy = sum(1 for s in self.current_data["samples"] if validate_dynamic(s))
        bad = total - st - dy
        messagebox.showinfo(
            "Validación",
            f"Seña: {self.current_data['label']}\n\n"
            f"Estáticas válidas: {st}\nDinámicas válidas: {dy}\nInválidas: {bad}\nTotal: {total}",
            parent=self.root,
        )

    def use_selected_label(self):
        if self.current_data:
            self.train_label_var.set(self.current_data["label"])

    def start_camera(self):
        if self.camera_running:
            return

        # SOLO CÁMARA: detectar dispositivos disponibles y preguntar cuál usar.
        # No modifica MediaPipe, captura estática/dinámica ni modelos JSON.
        try:
            current_idx = int(self.camera_index_var.get())
        except Exception:
            current_idx = 0

        detected = []

        if sys.platform.startswith("linux"):
            # En Linux usamos los dispositivos reales /dev/videoN para evitar
            # asumir que la webcam siempre corresponde al índice 0.
            def _video_num(path):
                suffix = path.name.replace("video", "", 1)
                return int(suffix) if suffix.isdigit() else 9999

            for dev in sorted(Path("/dev").glob("video*"), key=_video_num):
                suffix = dev.name.replace("video", "", 1)
                if not suffix.isdigit():
                    continue
                idx = int(suffix)
                name = "Cámara"
                try:
                    name_path = Path("/sys/class/video4linux") / dev.name / "name"
                    if name_path.exists():
                        name = name_path.read_text(encoding="utf-8", errors="ignore").strip() or name
                except Exception:
                    pass
                # Solo mostrar dispositivos que realmente entregan un frame.
                probe = cv2.VideoCapture(str(dev), cv2.CAP_V4L2)
                try:
                    if probe.isOpened():
                        ok, frame = probe.read()
                        if ok and frame is not None:
                            detected.append((idx, f"{name} ({dev})"))
                finally:
                    probe.release()
        else:
            # Windows/macOS no exponen /dev/videoN. Probamos índices pequeños
            # únicamente para construir la lista que se mostrará al usuario.
            probe_backends = [cv2.CAP_ANY]
            if os.name == "nt":
                probe_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

            for idx in range(0, 6):
                found = False
                for backend in probe_backends:
                    probe = cv2.VideoCapture(idx, backend)
                    try:
                        if probe.isOpened():
                            ok, frame = probe.read()
                            if ok and frame is not None:
                                found = True
                                break
                    finally:
                        probe.release()
                if found:
                    detected.append((idx, f"Cámara {idx}"))

        if detected:
            lines = [f"{idx}: {name}" for idx, name in detected]
            prompt = (
                "Cámaras detectadas:\n\n"
                + "\n".join(lines)
                + "\n\nEscribe el número de la cámara que quieres usar:"
            )
            valid_indices = {idx for idx, _ in detected}
            initial = current_idx if current_idx in valid_indices else detected[0][0]
            idx = simpledialog.askinteger(
                "Elegir cámara",
                prompt,
                parent=self.root,
                initialvalue=initial,
                minvalue=0,
                maxvalue=99,
            )
            if idx is None:
                self.set_status("Selección de cámara cancelada")
                return
            if idx not in valid_indices:
                messagebox.showwarning(
                    "Cámara",
                    f"La cámara {idx} no aparece entre las cámaras detectadas.\n"
                    "Selecciona uno de los números mostrados.",
                    parent=self.root,
                )
                return
        else:
            # Si el sistema no deja enumerarlas, todavía permitimos elegir un
            # índice manualmente en vez de abortar directamente.
            idx = simpledialog.askinteger(
                "Elegir cámara",
                "No pude enumerar automáticamente las cámaras.\n\n"
                "Escribe el número de cámara que quieres probar (normalmente 0, 1 o 2):",
                parent=self.root,
                initialvalue=current_idx,
                minvalue=0,
                maxvalue=99,
            )
            if idx is None:
                self.set_status("Selección de cámara cancelada")
                return

        self.camera_index_var.set(idx)

        backends = []
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform.startswith("linux"):
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        cap = None
        for backend in backends:
            # En Linux abrimos /dev/videoN directamente cuando existe. Esto
            # evita depender de cómo OpenCV haya enumerado los índices.
            source = idx
            if sys.platform.startswith("linux"):
                dev_path = Path(f"/dev/video{idx}")
                if dev_path.exists() and backend == cv2.CAP_V4L2:
                    source = str(dev_path)

            test_cap = cv2.VideoCapture(source, backend)
            if test_cap.isOpened():
                ok, frame = test_cap.read()
                if ok and frame is not None:
                    cap = test_cap
                    break
            test_cap.release()

        if cap is None:
            messagebox.showerror(
                "Cámara",
                f"La cámara {idx} fue seleccionada, pero no pudo entregar imagen.\n\n"
                "Cierra el navegador, OBS, Cheese u otra aplicación que pueda estar usando "
                "la webcam y vuelve a intentarlo.",
                parent=self.root,
            )
            self.set_status(f"No se pudo abrir la cámara {idx}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.camera_running = True
        self.latest_hands = []
        self.set_status(f"Cámara {idx} iniciada")
        self._camera_loop()

    def stop_camera(self):
        self.cancel_capture(silent=True)
        self.camera_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.hands_detector is not None:
            try:
                self.hands_detector.close()
            except Exception:
                pass
            self.hands_detector = None
        self.latest_hands = []
        self.preview.configure(image="", text="Cámara detenida")
        self.preview.image = None
        self.set_status("Cámara detenida")

    def filter_hands(self, hands):
        f = self.hand_filter_var.get()
        if f == "Todas":
            return hands
        target = "left" if f == "Izquierda" else "right"
        return [h for h in hands if str(h.get("handedness", "")).lower() == target]

    def copy_hands(self, results):
        output = []
        if not results or not results.multi_hand_landmarks:
            return output
        for i, hand_lms in enumerate(results.multi_hand_landmarks[:2]):
            handedness = "Unknown"
            if results.multi_handedness and i < len(results.multi_handedness):
                try:
                    handedness = results.multi_handedness[i].classification[0].label
                except Exception:
                    pass
            output.append({
                "handedness": handedness,
                "landmarks": [
                    {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                    for lm in hand_lms.landmark
                ],
            })
        return output

    def _camera_loop(self):
        if not self.camera_running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.set_status("No llegó un frame de la cámara")
            self.root.after(30, self._camera_loop)
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb) if self.hands_detector is not None else None
        self.latest_hands = self.copy_hands(results)
        self.latest_frame_id += 1
        self.last_frame = frame

        if results and results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        self._capture_tick()

        h, w = frame.shape[:2]
        max_w, max_h = 760, 470
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok_enc, buf = cv2.imencode(".ppm", frame)
        if ok_enc:
            try:
                photo = tk.PhotoImage(data=buf.tobytes())
                self.preview.configure(image=photo, text="")
                self.preview.image = photo
            except tk.TclError:
                pass
        self.root.after(10, self._camera_loop)

    def sample_now(self, label):
        hands = self.filter_hands(self.latest_hands)
        if not hands:
            return None
        return {
            "label": label,
            "timestamp": time.time(),
            "hands": json.loads(json.dumps(hands)),
        }

    def capture_one_static(self):
        label = self.ensure_label()
        if not label:
            return
        if not self.camera_running:
            messagebox.showwarning("Cámara", "Inicia la cámara primero.", parent=self.root)
            return
        sample = self.sample_now(label)
        if sample is None:
            self.set_status("No detecté la mano seleccionada")
            return
        path, data = self.load_or_create_label(label)
        data["samples"].append(sample)
        atomic_save_model(path, data)
        self.current_path = path
        self.current_data = load_model(path)
        self.refresh_models()
        self.set_status(f"1 muestra estática guardada en {path.name}")

    def start_static_batch(self):
        """Inicia una serie de muestras estáticas sin modificar cámara/MediaPipe."""
        label = self.ensure_label()
        if not label:
            return
        if not self.camera_running or self.cap is None:
            messagebox.showwarning("Cámara", "Inicia la cámara primero.", parent=self.root)
            return

        # Los Spinbox pueden quedar temporalmente vacíos mientras se editan.
        # En vez de dejar que Tk lance una excepción, usamos valores seguros.
        try:
            target = int(self.static_count_var.get())
        except (TypeError, ValueError, tk.TclError):
            target = 30
            self.static_count_var.set(target)
        target = max(1, min(500, target))

        try:
            interval = float(self.static_interval_var.get())
        except (TypeError, ValueError, tk.TclError):
            interval = 0.08
            self.static_interval_var.set(interval)
        interval = max(0.0, min(1.0, interval))

        self.cancel_capture(silent=True)
        self.capture_mode = "static"
        self.capture_target = target
        self.capture_saved = 0
        self.capture_last_saved = 0.0
        self.capture_static_last_frame_id = -1
        self.pending_static_samples = []
        self.capture_countdown_until = time.perf_counter() + 2.0
        self.capture_progress["value"] = 0
        self.capture_label.configure(text=f"0/{self.capture_target}")
        self.set_status(
            f"Lote estático listo: {self.capture_target} muestras · "
            f"intervalo {interval:.2f}s · comienza en 2 segundos"
        )

    def start_dynamic_capture(self):
        label = self.ensure_label()
        if not label or not self.camera_running:
            if label and not self.camera_running:
                messagebox.showwarning("Cámara", "Inicia la cámara primero.", parent=self.root)
            return
        self.cancel_capture(silent=True)
        self.capture_mode = "dynamic_wait"
        self.capture_dynamic_total = max(1, int(self.dynamic_reps_var.get()))
        self.capture_dynamic_rep = 0
        self.pending_dynamic_samples = []
        self.capture_countdown_until = time.perf_counter() + 2.0
        self.capture_progress["value"] = 0
        self.capture_label.configure(text=f"0/{self.capture_dynamic_total}")
        self.set_status("Prepárate: primera seña dinámica en 2 segundos")

    def _capture_tick(self):
        if not self.capture_mode:
            return
        now = time.perf_counter()
        label = self.train_label_var.get().strip().upper()
        if not label:
            self.cancel_capture(silent=True)
            return

        if self.capture_mode in ("static", "dynamic_wait") and now < self.capture_countdown_until:
            remain = max(0.0, self.capture_countdown_until - now)
            self.capture_progress["value"] = max(0, min(100, (2.0 - remain) / 2.0 * 100.0))
            self.capture_label.configure(text=f"Empieza en {remain:.1f}s")
            return

        if self.capture_mode == "static":
            # Solo usamos un resultado nuevo de la cámara. Así el lote no duplica
            # artificialmente el mismo frame cuando la interfaz actualiza más rápido.
            if self.latest_frame_id == self.capture_static_last_frame_id:
                return
            self.capture_static_last_frame_id = self.latest_frame_id

            sample = self.sample_now(label)
            try:
                interval = float(self.static_interval_var.get())
            except (TypeError, ValueError, tk.TclError):
                interval = 0.08
            interval = max(0.0, min(1.0, interval))

            if sample is None:
                self.set_status(
                    f"Lote estático {self.capture_saved}/{self.capture_target}: "
                    "esperando la mano seleccionada"
                )
                return

            if self.capture_last_saved > 0 and now - self.capture_last_saved < interval:
                return

            # Durante el lote guardamos en RAM y escribimos el JSON una sola vez
            # al terminar para no provocar tirones de disco en la cámara.
            self.pending_static_samples.append(sample)
            self.capture_saved += 1
            self.capture_last_saved = now
            self.capture_progress["value"] = 100.0 * self.capture_saved / self.capture_target
            self.capture_label.configure(text=f"{self.capture_saved}/{self.capture_target}")
            self.set_status(
                f"Capturando lote estático: {self.capture_saved}/{self.capture_target}"
            )

            if self.capture_saved >= self.capture_target:
                path, data = self.load_or_create_label(label)
                data["samples"].extend(self.pending_static_samples)
                atomic_save_model(path, data)
                self.capture_mode = None
                self.current_path = path
                self.current_data = load_model(path)
                self.refresh_models()
                self.capture_progress["value"] = 100
                self.capture_label.configure(text=f"{self.capture_saved}/{self.capture_target} ✓")
                self.set_status(
                    f"Lote terminado: {self.capture_saved} muestras estáticas guardadas en {path.name}"
                )
            return

        if self.capture_mode == "dynamic_wait":
            if self.sample_now(label) is None:
                self.set_status("Dinámica: esperando la mano seleccionada")
                return
            self.capture_mode = "dynamic_record"
            self.capture_started_at = now
            self.capture_last_saved = 0.0
            self.capture_frames = []
            self.set_status(
                f"Grabando dinámica {self.capture_dynamic_rep + 1}/{self.capture_dynamic_total}: haz la seña completa"
            )
            return

        if self.capture_mode == "dynamic_record":
            duration = max(0.5, float(self.dynamic_duration_var.get()))
            sample = self.sample_now(label)
            if sample is not None and (self.capture_last_saved <= 0 or now - self.capture_last_saved >= 0.030):
                self.capture_frames.append({
                    "t": now - self.capture_started_at,
                    "hands": sample["hands"],
                })
                self.capture_last_saved = now
            elapsed = now - self.capture_started_at
            fraction = max(0.0, min(1.0, elapsed / duration))
            base = self.capture_dynamic_rep / self.capture_dynamic_total
            self.capture_progress["value"] = 100.0 * (base + fraction / self.capture_dynamic_total)
            self.capture_label.configure(text=f"Dinámica {self.capture_dynamic_rep + 1}/{self.capture_dynamic_total} · {fraction*100:.0f}%")
            if elapsed < duration:
                return

            frames = list(self.capture_frames)
            feature = vectorize_motion(frames)
            valid = len(frames) >= DYNAMIC_MIN_FRAMES and feature is not None and feature["motion"] >= DYNAMIC_MIN_MOTION
            if valid:
                self.pending_dynamic_samples.append({
                    "label": label,
                    "timestamp": time.time(),
                    "type": "dynamic",
                    "frames": frames,
                })
                self.capture_dynamic_rep += 1
                self.capture_label.configure(text=f"{self.capture_dynamic_rep}/{self.capture_dynamic_total}")
                if self.capture_dynamic_rep >= self.capture_dynamic_total:
                    path, data = self.load_or_create_label(label)
                    data["samples"].extend(self.pending_dynamic_samples)
                    atomic_save_model(path, data)
                    self.capture_mode = None
                    self.current_path = path
                    self.current_data = load_model(path)
                    self.refresh_models()
                    self.capture_progress["value"] = 100
                    self.set_status(f"Dinámicas guardadas: {len(self.pending_dynamic_samples)} en {path.name}")
                else:
                    self.capture_mode = "dynamic_wait"
                    self.capture_countdown_until = now + 1.5
                    self.set_status(
                        f"Dinámica {self.capture_dynamic_rep} válida. Prepárate para la siguiente."
                    )
            else:
                motion = feature["motion"] if feature is not None else 0.0
                self.capture_mode = "dynamic_wait"
                self.capture_countdown_until = now + 1.5
                self.set_status(
                    f"Repetición descartada: pocos frames o poco movimiento ({motion:.3f}). Repítela."
                )

    def cancel_capture(self, silent=False):
        was = self.capture_mode
        self.capture_mode = None
        self.capture_frames = []
        self.capture_static_last_frame_id = -1
        self.pending_dynamic_samples = []
        self.pending_static_samples = []
        self.capture_progress["value"] = 0
        self.capture_label.configure(text="Sin captura")
        if was and not silent:
            self.set_status("Captura cancelada. No se guardó la serie incompleta.")

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = MQHModelAdmin(root)
    root.mainloop()


if __name__ == "__main__":
    main()
