import cv2
import mediapipe as mp
import threading
import os
import tkinter as tk
from tkinter import ttk, messagebox
import time
import json
import urllib.request
import urllib.error
import queue
from pathlib import Path


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

# Versión instalada de la aplicación.
# Al publicar una nueva Release, actualiza este valor (por ejemplo 1.0.2).
APP_VERSION = "1.0.1"

# GitHub Releases se usa como servidor de actualizaciones.
GITHUB_OWNER = "manosqhablan26-coder"
GITHUB_REPO = "manos-que-hablan-"
GITHUB_LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# Información de la última versión encontrada.
latest_release_info = None

# Resolución SOLO para el procesamiento de MediaPipe.
# La cámara visible conserva la resolución que entregue el dispositivo.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# Tamaño máximo de la vista previa dentro de la interfaz.
# Esto NO cambia la resolución capturada por la cámara.
PREVIEW_MAX_WIDTH = 900
PREVIEW_MAX_HEIGHT = 620


# ==========================================================
# MEDIAPIPE HANDS
# ==========================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================================
# VARIABLES GLOBALES
# ==========================================================

cap = None
capture_thread = None
processing_thread = None

# Último frame capturado por la cámara.
latest_frame = None
latest_frame_id = 0
latest_frame_capture_time = 0.0

# Último resultado procesado por MediaPipe.
latest_processed_frame = None
latest_processed_frame_id = -1
latest_hand_count = 0
latest_processing_latency_ms = 0.0

running = False

# Un lock corto protege únicamente el intercambio de referencias.
lock = threading.Lock()

camera_items = []
selected_camera_id = None

# Métricas suavizadas.
camera_fps = 0.0
mediapipe_fps = 0.0
display_fps = 0.0

last_capture_time = 0.0
last_process_time = 0.0
last_display_time = 0.0

# ==========================================================
# ESTABILIZACIÓN DE LANDMARKS
# ==========================================================
stabilization_mode = "Baja"
landmark_history = {}


# ==========================================================
# BUSCAR CÁMARAS
# ==========================================================

def buscar_camaras():
    camaras = []

    for i in range(10):
        dispositivo = f"/dev/video{i}"

        if not os.path.exists(dispositivo):
            continue

        cap_test = cv2.VideoCapture(i)

        if cap_test.isOpened():
            ret, frame = cap_test.read()

            if ret and frame is not None:
                alto, ancho = frame.shape[:2]

                camaras.append({
                    "id": i,
                    "dispositivo": dispositivo,
                    "ancho": ancho,
                    "alto": alto
                })

        cap_test.release()

    return camaras


# ==========================================================
# CAPTURA DEL FRAME MÁS RECIENTE
# ==========================================================

def capture_frames():
    global latest_frame
    global latest_frame_id
    global latest_frame_capture_time
    global running
    global camera_fps
    global last_capture_time

    while running and cap is not None:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        now = time.perf_counter()

        # FPS reales de captura, suavizados para que no "salten".
        if last_capture_time > 0:
            delta = now - last_capture_time

            if delta > 0:
                instant_fps = 1.0 / delta
                camera_fps = (
                    instant_fps
                    if camera_fps == 0
                    else camera_fps * 0.88 + instant_fps * 0.12
                )

        last_capture_time = now

        with lock:
            # No existe una cola.
            # El frame anterior se reemplaza inmediatamente por el más nuevo.
            latest_frame = frame
            latest_frame_id += 1
            latest_frame_capture_time = now


# ==========================================================
# ESTABILIZACIÓN ADAPTATIVA DE LANDMARKS
# ==========================================================

def set_stabilization_mode(event=None):
    global stabilization_mode

    if "stabilization_var" in globals():
        value = stabilization_var.get()
    elif "stabilization_combo" in globals():
        value = stabilization_combo.get()
    else:
        value = stabilization_mode

    if value in ("OFF", "Baja", "Media"):
        stabilization_mode = value
        landmark_history.clear()

        if "stabilization_status" in globals():
            stabilization_status.configure(
                text=f"Estabilización: {value}"
            )


def stabilize_hand_landmarks(hand_landmarks, hand_key):
    if stabilization_mode == "OFF":
        return hand_landmarks

    if stabilization_mode == "Baja":
        base_alpha = 0.72
        fast_alpha = 0.92
        motion_threshold = 0.018
    else:  # Media
        base_alpha = 0.52
        fast_alpha = 0.88
        motion_threshold = 0.015

    previous = landmark_history.get(hand_key)

    current_points = [
        (lm.x, lm.y, lm.z)
        for lm in hand_landmarks.landmark
    ]

    if previous is None or len(previous) != len(current_points):
        landmark_history[hand_key] = current_points
        return hand_landmarks

    smoothed_points = []

    for index, lm in enumerate(hand_landmarks.landmark):
        px, py, pz = previous[index]

        dx = lm.x - px
        dy = lm.y - py
        movement = (dx * dx + dy * dy) ** 0.5

        if movement >= motion_threshold:
            alpha = fast_alpha
        else:
            ratio = movement / motion_threshold if motion_threshold > 0 else 1.0
            ratio = max(0.0, min(1.0, ratio))
            alpha = base_alpha + (fast_alpha - base_alpha) * ratio

        sx = alpha * lm.x + (1.0 - alpha) * px
        sy = alpha * lm.y + (1.0 - alpha) * py
        sz = alpha * lm.z + (1.0 - alpha) * pz

        lm.x = sx
        lm.y = sy
        lm.z = sz

        smoothed_points.append((sx, sy, sz))

    landmark_history[hand_key] = smoothed_points

    return hand_landmarks


# ==========================================================
# PROCESAMIENTO MEDIAPIPE EN HILO INDEPENDIENTE
# ==========================================================

def process_frames():
    global latest_processed_frame
    global latest_processed_frame_id
    global latest_hand_count
    global latest_processing_latency_ms
    global mediapipe_fps
    global last_process_time
    global running

    processed_id = -1

    while running:
        # Tomamos una referencia al frame MÁS RECIENTE.
        # Si MediaPipe se atrasó, los frames intermedios se descartan.
        with lock:
            if latest_frame is None or latest_frame_id == processed_id:
                frame = None
            else:
                frame = latest_frame.copy()
                frame_id = latest_frame_id
                capture_time = latest_frame_capture_time

        if frame is None:
            time.sleep(0.001)
            continue

        process_start = time.perf_counter()

        # Espejo antes de MediaPipe para que landmarks e imagen coincidan.
        frame = cv2.flip(frame, 1)

        original_height, original_width = frame.shape[:2]

        # COPIA PEQUEÑA SOLO PARA MEDIAPIPE.
        small_frame = cv2.resize(
            frame,
            (PROCESS_WIDTH, PROCESS_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_small.flags.writeable = False

        results = hands.process(rgb_small)

        rgb_small.flags.writeable = True

        hand_count = 0

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            current_hand_keys = set()

            # Los landmarks normalizados se pueden dibujar directamente
            # sobre el frame ORIGINAL.
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

                hand_label = f"hand_{hand_index}"

                if (
                    results.multi_handedness
                    and hand_index < len(results.multi_handedness)
                ):
                    try:
                        hand_label = (
                            results.multi_handedness[hand_index]
                            .classification[0]
                            .label
                        )
                    except Exception:
                        pass

                hand_key = f"{hand_label}_{hand_index}"
                current_hand_keys.add(hand_key)

                hand_landmarks = stabilize_hand_landmarks(
                    hand_landmarks,
                    hand_key
                )

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 220, 255),
                        thickness=2,
                        circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(80, 255, 140),
                        thickness=2
                    )
                )

            stale_keys = [
                key for key in list(landmark_history.keys())
                if key not in current_hand_keys
            ]

            for key in stale_keys:
                landmark_history.pop(key, None)

        else:
            landmark_history.clear()

        cv2.putText(
            frame,
            f"{original_width}x{original_height}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        process_end = time.perf_counter()

        # FPS reales del hilo de MediaPipe.
        if last_process_time > 0:
            process_delta = process_end - last_process_time

            if process_delta > 0:
                instant_process_fps = 1.0 / process_delta
                mediapipe_fps = (
                    instant_process_fps
                    if mediapipe_fps == 0
                    else mediapipe_fps * 0.85 + instant_process_fps * 0.15
                )

        last_process_time = process_end

        # Latencia extremo-a-extremo desde que se capturó ESE frame
        # hasta que MediaPipe terminó de procesarlo.
        latency_ms = (process_end - capture_time) * 1000.0

        with lock:
            latest_processed_frame = frame
            latest_processed_frame_id = frame_id
            latest_hand_count = hand_count
            latest_processing_latency_ms = latency_ms

        processed_id = frame_id


# ==========================================================
# INICIAR CÁMARA
# ==========================================================

def iniciar_camara():
    global cap
    global capture_thread
    global processing_thread
    global running
    global latest_frame
    global latest_frame_id
    global latest_frame_capture_time
    global latest_processed_frame
    global latest_processed_frame_id
    global selected_camera_id
    global camera_fps
    global mediapipe_fps
    global display_fps
    global last_capture_time
    global last_process_time
    global last_display_time

    detener_camara()

    seleccion = camera_combo.current()

    if seleccion < 0 or seleccion >= len(camera_items):
        set_status("Selecciona una cámara primero.", error=True)
        return

    selected_camera_id = camera_items[seleccion]["id"]

    set_status(f"Abriendo /dev/video{selected_camera_id}...")

    cap = cv2.VideoCapture(selected_camera_id)

    if not cap.isOpened():
        cap = None
        set_status(
            f"No se pudo abrir /dev/video{selected_camera_id}.",
            error=True
        )
        return

    # Intentamos mantener el buffer al mínimo.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # NO fijamos CAP_PROP_FRAME_WIDTH ni CAP_PROP_FRAME_HEIGHT.
    # De esta forma no obligamos a la cámara a una resolución concreta.

    with lock:
        latest_frame = None
        latest_processed_frame = None
        latest_frame_id = 0
        latest_frame_capture_time = 0.0
        latest_processed_frame = None
        latest_processed_frame_id = -1

    camera_fps = 0.0
    mediapipe_fps = 0.0
    display_fps = 0.0

    landmark_history.clear()

    last_capture_time = 0.0
    last_process_time = 0.0
    last_display_time = 0.0

    running = True

    capture_thread = threading.Thread(
        target=capture_frames,
        daemon=True,
        name="CameraCapture"
    )

    processing_thread = threading.Thread(
        target=process_frames,
        daemon=True,
        name="MediaPipeProcessing"
    )

    capture_thread.start()
    processing_thread.start()

    # Iniciar permanece visible incluso con la cámara encendida.
    start_button.configure(text="▶ Iniciar")
    stop_button.configure(state="normal")

    theme = THEMES.get(current_theme_name, THEMES["Oscuro"])
    translation_value.configure(
        text="Esperando una seña...",
        fg=theme["muted"]
    )

    set_status(f"Cámara activa: /dev/video{selected_camera_id}")


# ==========================================================
# DETENER CÁMARA
# ==========================================================

def detener_camara():
    global cap
    global running
    global capture_thread
    global processing_thread
    global latest_frame
    global latest_processed_frame

    running = False

    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join(timeout=0.5)

    capture_thread = None

    if processing_thread is not None and processing_thread.is_alive():
        processing_thread.join(timeout=0.7)

    processing_thread = None

    if cap is not None:
        cap.release()
        cap = None

    with lock:
        latest_frame = None

    if "stop_button" in globals():
        stop_button.configure(state="disabled")

    if "start_button" in globals():
        start_button.configure(text="▶ Iniciar")

    if "camera_label" in globals():
        camera_label.configure(image="", text="Cámara detenida")

    if "resolution_value" in globals():
        resolution_value.configure(text="--")

    if "hands_value" in globals():
        hands_value.configure(text="0")

    if "camera_fps_value" in globals():
        camera_fps_value.configure(text="--")

    if "mediapipe_fps_value" in globals():
        mediapipe_fps_value.configure(text="--")

    if "latency_value" in globals():
        latency_value.configure(text="--")

    if "fps_value" in globals():
        fps_value.configure(text="--")

    if "detection_value" in globals():
        detection_value.configure(text="En espera")


# ==========================================================
# ESTADO
# ==========================================================

def set_status(text, error=False):
    theme = THEMES.get(current_theme_name, THEMES["Oscuro"])
    status_label.configure(
        text=text,
        fg=theme["danger"] if error else theme["muted"]
    )


# ==========================================================
# CONVERTIR FRAME DE OPENCV PARA TKINTER
# ==========================================================

def frame_to_photo(frame):
    # Muestra TODO el frame original, sin recortar ni deformar.
    # La imagen se escala proporcionalmente para caber dentro del cuadro fijo.
    alto, ancho = frame.shape[:2]

    if "CAMERA_VIEW_WIDTH" in globals() and "CAMERA_VIEW_HEIGHT" in globals():
        disponible_w = max(1, CAMERA_VIEW_WIDTH)
        disponible_h = max(1, CAMERA_VIEW_HEIGHT)
    elif "camera_label" in globals():
        disponible_w = max(1, camera_label.winfo_width())
        disponible_h = max(1, camera_label.winfo_height())
    else:
        disponible_w = PREVIEW_MAX_WIDTH
        disponible_h = PREVIEW_MAX_HEIGHT

    if disponible_w <= 2:
        disponible_w = PREVIEW_MAX_WIDTH
    if disponible_h <= 2:
        disponible_h = PREVIEW_MAX_HEIGHT

    # "Contain": conserva la relación de aspecto y muestra el encuadre completo.
    # En modo normal no agrandamos por encima de la resolución original.
    # En modo video-pantalla-completa sí permitimos ampliar para aprovechar
    # todo el espacio disponible sin deformar la imagen.
    escala_disponible = min(
        disponible_w / ancho,
        disponible_h / alto
    )

    if globals().get("video_fullscreen_active", False):
        escala = escala_disponible
    else:
        escala = min(escala_disponible, 1.0)

    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))

    if nuevo_ancho != ancho or nuevo_alto != alto:
        interpolation = (
            cv2.INTER_LINEAR
            if escala > 1.0
            else cv2.INTER_AREA
        )

        vista = cv2.resize(
            frame,
            (nuevo_ancho, nuevo_alto),
            interpolation=interpolation
        )
    else:
        vista = frame

    ok, buffer = cv2.imencode(".ppm", vista)

    if not ok:
        return None

    return tk.PhotoImage(data=buffer.tobytes())


# ==========================================================
# PROCESAR Y MOSTRAR
# ==========================================================

last_displayed_processed_id = -1


def actualizar_video():
    global last_displayed_processed_id
    global last_display_time
    global display_fps

    if running:
        with lock:
            if (
                latest_processed_frame is not None
                and latest_processed_frame_id != last_displayed_processed_id
            ):
                frame = latest_processed_frame.copy()
                processed_id = latest_processed_frame_id
                hand_count = latest_hand_count
                latency_ms = latest_processing_latency_ms
            else:
                frame = None

        if frame is not None:
            last_displayed_processed_id = processed_id

            original_height, original_width = frame.shape[:2]

            now = time.perf_counter()

            if last_display_time > 0:
                delta = now - last_display_time

                if delta > 0:
                    instant_fps = 1.0 / delta
                    display_fps = (
                        instant_fps
                        if display_fps == 0
                        else display_fps * 0.85 + instant_fps * 0.15
                    )

            last_display_time = now

            resolution_value.configure(
                text=f"{original_width} × {original_height}"
            )

            hands_value.configure(
                text=str(hand_count)
            )

            camera_fps_value.configure(
                text=f"{camera_fps:.0f} FPS"
            )

            mediapipe_fps_value.configure(
                text=f"{mediapipe_fps:.0f} FPS"
            )

            latency_value.configure(
                text=f"{latency_ms:.0f} ms"
            )

            fps_value.configure(
                text=f"{display_fps:.0f} FPS"
            )

            if hand_count > 0:
                detection_value.configure(
                    text="Detectando",
                    fg=THEMES.get(current_theme_name, THEMES["Oscuro"])["ok"]
                )

                if "translation_status_value" in globals():
                    translation_status_value.configure(
                        text="Detectando",
                        fg=THEMES.get(current_theme_name, THEMES["Oscuro"])["ok"]
                    )

                translation_value.configure(
                    text="Mano detectada",
                    fg=THEMES.get(current_theme_name, THEMES["Oscuro"])["text"]
                )
            else:
                detection_value.configure(
                    text="En espera",
                    fg=THEMES.get(current_theme_name, THEMES["Oscuro"])["muted"]
                )

                if "translation_status_value" in globals():
                    translation_status_value.configure(
                        text="En espera",
                        fg=THEMES.get(current_theme_name, THEMES["Oscuro"])["muted"]
                    )

                translation_value.configure(
                    text="Esperando una seña...",
                    fg=THEMES.get(current_theme_name, THEMES["Oscuro"])["muted"]
                )

            photo = frame_to_photo(frame)

            if photo is not None:
                camera_label.configure(
                    image=photo,
                    text="",
                    width=1,
                    height=1
                )
                camera_label.image = photo

    # La UI solo pinta resultados ya procesados.
    # Nunca llama hands.process(), por eso Tkinter no se bloquea.
    root.after(1, actualizar_video)


# ==========================================================
# ACTUALIZAR LISTA DE CÁMARAS
# ==========================================================

def actualizar_camaras():
    global camera_items

    set_status("Buscando cámaras...")

    detener_camara()

    camera_items = buscar_camaras()

    valores = []

    for camara in camera_items:
        valores.append(
            f"{camara['dispositivo']}  ·  "
            f"{camara['ancho']}x{camara['alto']}"
        )

    camera_combo["values"] = valores

    if valores:
        camera_combo.current(0)

        set_status(
            f"{len(valores)} cámara(s) encontrada(s)."
        )
    else:
        camera_combo.set("")
        set_status(
            "No se encontró ninguna cámara.",
            error=True
        )


# ==========================================================
# PANTALLA COMPLETA SOLO DEL VIDEO
# ==========================================================

video_fullscreen_active = False
_video_layout_state = {}

def toggle_video_fullscreen(event=None):
    global video_fullscreen_active, _video_layout_state

    if not video_fullscreen_active:
        video_fullscreen_active = True

        # Guardamos cómo estaba colocado el panel central.
        _video_layout_state["camera_panel_manager"] = camera_panel.winfo_manager()

        # Ocultamos temporalmente la interfaz alrededor.
        topbar.pack_forget()
        footer.pack_forget()
        left_panel.grid_remove()
        side_panel.grid_remove()

        # Quitamos el panel de cámara del grid central y lo hacemos ocupar
        # toda el área disponible de la ventana.
        camera_panel.grid_remove()
        camera_panel.pack(in_=main, fill="both", expand=True, padx=0, pady=0)

        # Ocultamos elementos que no son video.
        camera_header.pack_forget()
        controls.pack_forget()

        # El contenedor de cámara ocupa completamente el panel.
        camera_container.pack_forget()
        camera_container.pack(fill="both", expand=True, padx=0, pady=0)

        camera_image_frame.pack_forget()
        camera_image_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Recalculamos el tamaño real disponible después de expandir
        # el área del video para que el frame se adapte inmediatamente.
        root.update_idletasks()
        actualizar_dimensiones_video()
        root.after(50, actualizar_dimensiones_video)
        root.after(150, actualizar_dimensiones_video)

        draw_fullscreen_icon()

    else:
        salir_video_fullscreen()

def salir_video_fullscreen(event=None):
    global video_fullscreen_active

    if not video_fullscreen_active:
        return

    video_fullscreen_active = False

    # Restaurar estructura original.
    camera_panel.pack_forget()

    topbar.pack(fill="x", padx=0, pady=0, before=main)
    footer.pack(fill="x")

    left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)
    side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

    camera_header.pack(fill="x", padx=14, pady=(12, 8), before=camera_container)

    camera_container.pack_forget()
    camera_container.pack(fill="both", expand=True, padx=14, pady=(0, 10))

    camera_image_frame.pack_forget()
    camera_image_frame.pack(fill="both", expand=True, padx=10, pady=5)

    controls.pack(fill="x", padx=14, pady=(0, 8))

    # IMPORTANTE: al salir de pantalla completa respetamos la sección
    # que estaba activa antes de entrar. Si seguimos en "Inicio",
    # el panel derecho debe continuar oculto y la cámara debe conservar
    # todo el espacio libre.
    if sidebar_active == "Inicio":
        side_panel.grid_remove()
        main.grid_columnconfigure(0, weight=2, uniform="")
        main.grid_columnconfigure(1, weight=8, uniform="")
        main.grid_columnconfigure(2, weight=0, minsize=0, uniform="")
    else:
        # En Traducir/otras vistas recuperamos la distribución normal.
        side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        main.grid_columnconfigure(0, weight=2, uniform="main_cols")
        main.grid_columnconfigure(1, weight=6, uniform="main_cols")
        main.grid_columnconfigure(2, weight=3, minsize=0, uniform="main_cols")

    draw_fullscreen_icon()

    # Recalcular área visible para mantener el video correctamente escalado.
    root.update_idletasks()
    actualizar_dimensiones_video()
    root.after(50, actualizar_dimensiones_video)
    root.after(120, actualizar_dimensiones_video)

def actualizar_dimensiones_video():
    global CAMERA_VIEW_WIDTH, CAMERA_VIEW_HEIGHT
    if "camera_label" in globals():
        root.update_idletasks()
        CAMERA_VIEW_WIDTH = max(1, camera_label.winfo_width())
        CAMERA_VIEW_HEIGHT = max(1, camera_label.winfo_height())

# ==========================================================
# CERRAR APLICACIÓN
# ==========================================================

def cerrar_app():
    detener_camara()
    hands.close()
    root.destroy()


# ==========================================================
# GESTOR DE ACTUALIZACIONES · GITHUB RELEASES
# ==========================================================

def _version_tuple(version):
    """Convierte v1.2.3 / 1.2.3 en una tupla comparable de enteros."""
    version = str(version).strip().lower().lstrip("v")
    parts = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:4])


def _set_update_status(text, kind="muted"):
    """Actualiza el texto del gestor sin bloquear la interfaz."""
    if "update_status_label" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    color = c["muted"]
    if kind == "ok":
        color = c["ok"]
    elif kind == "error":
        color = c["danger"]
    elif kind == "text":
        color = c["text"]

    update_status_label.configure(text=text, fg=color)


def _find_windows_asset(release_data):
    """Busca primero el ZIP de Windows y, si no, cualquier ZIP de la Release."""
    assets = release_data.get("assets") or []

    for asset in assets:
        name = str(asset.get("name", ""))
        lower = name.lower()
        if lower.endswith(".zip") and "manosquehablan" in lower and "windows" in lower:
            return asset

    for asset in assets:
        name = str(asset.get("name", ""))
        if name.lower().endswith(".zip"):
            return asset

    return None


def buscar_actualizaciones_app():
    """Consulta la Release más reciente de GitHub sin bloquear Tkinter."""
    global latest_release_info

    if "update_check_button" in globals():
        update_check_button.configure(state="disabled", text="Buscando...")
    if "update_download_button" in globals():
        update_download_button.pack_forget()

    _set_update_status("Consultando GitHub Releases...", "muted")

    result_queue = queue.Queue()

    def worker():
        try:
            request = urllib.request.Request(
                GITHUB_LATEST_RELEASE_API,
                headers={
                    "User-Agent": "ManosQueHablan-Updater/1.0",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()

            data = json.loads(raw.decode("utf-8"))
            tag = str(data.get("tag_name", "")).strip()

            if not tag:
                raise RuntimeError("GitHub no devolvió una versión válida.")

            latest = tag.lstrip("vV")
            asset = _find_windows_asset(data)

            result_queue.put((
                "ok",
                {
                    "version": latest,
                    "tag": tag,
                    "notes": str(data.get("body") or "").strip(),
                    "page_url": str(data.get("html_url") or "").strip(),
                    "asset": asset,
                },
            ))

        except urllib.error.HTTPError as exc:
            result_queue.put(("error", f"GitHub respondió con error {exc.code}."))
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            detalle = f" ({reason})" if reason else ""
            result_queue.put(("error", f"Sin conexión a GitHub{detalle}."))
        except TimeoutError:
            result_queue.put(("error", "GitHub tardó demasiado en responder."))
        except Exception as exc:
            result_queue.put(("error", f"No se pudo comprobar: {exc}"))

    def revisar_resultado():
        global latest_release_info

        try:
            kind, payload = result_queue.get_nowait()
        except queue.Empty:
            # Esta función sí corre en el hilo principal de Tkinter.
            root.after(100, revisar_resultado)
            return

        if kind == "error":
            _update_error(payload)
            return

        latest_release_info = payload
        latest = payload["version"]
        tag = payload["tag"]
        asset = payload["asset"]

        update_check_button.configure(state="normal", text="Buscar actualizaciones")

        if _version_tuple(latest) > _version_tuple(APP_VERSION):
            _set_update_status(f"Nueva versión disponible: {tag}", "ok")

            if asset:
                update_download_button.configure(
                    state="normal",
                    text=f"Descargar {tag}",
                    command=descargar_actualizacion_app,
                )
            else:
                update_download_button.configure(
                    state="normal",
                    text="Ver Release",
                    command=abrir_release_actualizacion,
                )

            update_download_button.pack(fill="x", pady=(7, 0))
        else:
            _set_update_status(f"Estás al día · v{APP_VERSION}", "ok")

    threading.Thread(target=worker, daemon=True, name="UpdateCheck").start()
    root.after(100, revisar_resultado)


def _update_error(message):
    if "update_check_button" in globals():
        update_check_button.configure(state="normal", text="Reintentar")
    _set_update_status(message, "error")


def abrir_release_actualizacion():
    """Abre la página de la última Release si no hay ZIP descargable."""
    import webbrowser

    if latest_release_info and latest_release_info.get("page_url"):
        webbrowser.open(latest_release_info["page_url"])


def _safe_download_path(filename):
    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    target = downloads / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 2
    while True:
        candidate = downloads / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def descargar_actualizacion_app():
    """Descarga el ZIP de la nueva versión sin congelar Tkinter."""
    if not latest_release_info:
        return

    asset = latest_release_info.get("asset")
    if not asset:
        abrir_release_actualizacion()
        return

    url = str(asset.get("browser_download_url") or "").strip()
    filename = str(asset.get("name") or "ManosQueHablan-Windows.zip").strip()

    if not url:
        _set_update_status("La Release no tiene un archivo descargable.", "error")
        return

    update_download_button.configure(state="disabled", text="Descargando... 0%")
    _set_update_status("Descargando actualización...", "text")

    def worker():
        target = _safe_download_path(filename)
        temp_target = target.with_suffix(target.suffix + ".part")

        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ManosQueHablan-Updater"},
            )

            with urllib.request.urlopen(request, timeout=30) as response, open(temp_target, "wb") as f:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0

                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        percent = min(100, int(downloaded * 100 / total))
                        root.after(
                            0,
                            lambda p=percent: update_download_button.configure(
                                text=f"Descargando... {p}%"
                            ),
                        )

            temp_target.replace(target)

            def finish():
                update_download_button.configure(
                    state="normal",
                    text="Abrir carpeta de descargas",
                    command=lambda: abrir_carpeta_descargas(target.parent),
                )
                _set_update_status(
                    f"Descargada: {target.name}",
                    "ok",
                )
                messagebox.showinfo(
                    "Actualización descargada",
                    "La nueva versión se descargó correctamente.\n\n"
                    "Cierra Manos que Hablan antes de reemplazar la versión actual.",
                )

            root.after(0, finish)

        except Exception as exc:
            try:
                if temp_target.exists():
                    temp_target.unlink()
            except OSError:
                pass

            msg = f"Error al descargar: {exc}"
            root.after(0, lambda: _download_error(msg))

    threading.Thread(target=worker, daemon=True, name="UpdateDownload").start()


def _download_error(message):
    if "update_download_button" in globals():
        update_download_button.configure(
            state="normal",
            text="Reintentar descarga",
            command=descargar_actualizacion_app,
        )
    _set_update_status(message, "error")


def abrir_carpeta_descargas(folder):
    """Abre la carpeta donde quedó el ZIP en Windows, Linux o macOS."""
    try:
        folder = str(folder)
        if os.name == "nt":
            os.startfile(folder)
        elif os.sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as exc:
        messagebox.showerror("No se pudo abrir la carpeta", str(exc))


# ==========================================================
# INTERFAZ GRÁFICA · ESTILO DASHBOARD
# ==========================================================

import subprocess

THEMES = {
    "Oscuro": {
        # Paleta basada en la referencia: azul profundo + azul intenso.
        "bg": "#01132B",
        "topbar": "#00152E",
        "panel": "#031A34",
        "panel_alt": "#06213F",
        "card": "#082746",
        "border": "#164D86",
        "text": "#F4F8FF",
        "muted": "#8FB2D9",
        "accent": "#004097",
        "accent_text": "#FFFFFF",
        "button": "#062A52",
        "button_active": "#0A4D9C",
        "camera_bg": "#001021",
        "ok": "#58D68D",
        "danger": "#FF6B6B",
        "metric_line": "#2C72C8",
    },
    "Claro": {
        # Versión clara de la misma identidad azul.
        "bg": "#EEF5FF",
        "topbar": "#F7FAFF",
        "panel": "#FFFFFF",
        "panel_alt": "#EDF4FC",
        "card": "#F7FAFF",
        "border": "#BDD1EB",
        "text": "#0B1B33",
        "muted": "#58708E",
        "accent": "#004097",
        "accent_text": "#FFFFFF",
        "button": "#E1ECFA",
        "button_active": "#C9DCF3",
        "camera_bg": "#001021",
        "ok": "#258C42",
        "danger": "#C74747",
        "metric_line": "#2C72C8",
    },
}

current_theme_name = "Oscuro"
theme_widgets = []
metric_widgets = []
last_system_theme = None


def detectar_tema_sistema():
    """Intenta detectar si Linux usa tema claro u oscuro."""
    gtk_theme = os.environ.get("GTK_THEME", "").lower()
    if "dark" in gtk_theme:
        return "Oscuro"

    # GNOME / escritorios compatibles con gsettings
    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=0.35,
        )
        if "dark" in result.stdout.lower():
            return "Oscuro"
        if result.stdout.strip():
            return "Claro"
    except Exception:
        pass

    # KDE Plasma: revisa el nombre del esquema de color.
    kdeglobals = os.path.expanduser("~/.config/kdeglobals")
    try:
        if os.path.exists(kdeglobals):
            with open(kdeglobals, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read().lower()
            for line in data.splitlines():
                if line.startswith("colorscheme="):
                    value = line.split("=", 1)[1]
                    if "dark" in value or "breeze dark" in value:
                        return "Oscuro"
                    return "Claro"
    except Exception:
        pass

    # Si no se puede detectar, mantenemos el dashboard oscuro.
    return "Oscuro"


def register_theme(widget, role):
    theme_widgets.append((widget, role))
    return widget


def apply_theme(theme_name=None):
    global current_theme_name, last_system_theme

    selected = theme_var.get() if "theme_var" in globals() else "Sistema"
    if theme_name is None:
        theme_name = detectar_tema_sistema() if selected == "Sistema" else selected

    if theme_name not in THEMES:
        theme_name = "Oscuro"

    current_theme_name = theme_name
    last_system_theme = theme_name
    c = THEMES[theme_name]

    root.configure(bg=c["bg"])

    # ttk Combobox
    style.configure(
        "Dashboard.TCombobox",
        fieldbackground=c["button"],
        background=c["button"],
        foreground=c["text"],
        arrowcolor=c["text"],
        bordercolor=c["border"],
        lightcolor=c["border"],
        darkcolor=c["border"],
        padding=7,
    )
    style.map(
        "Dashboard.TCombobox",
        fieldbackground=[("readonly", c["button"])],
        foreground=[("readonly", c["text"])],
        selectbackground=[("readonly", c["button"])],
        selectforeground=[("readonly", c["text"])],
    )

    for widget, role in theme_widgets:
        try:
            if role == "bg":
                widget.configure(bg=c["bg"])
            elif role == "topbar":
                widget.configure(bg=c["topbar"])
            elif role == "panel":
                widget.configure(bg=c["panel"], highlightbackground=c["border"])
            elif role == "panel_alt":
                widget.configure(bg=c["panel_alt"], highlightbackground=c["border"])
            elif role == "card":
                widget.configure(bg=c["card"], highlightbackground=c["border"])
            elif role == "text_top":
                widget.configure(bg=c["topbar"], fg=c["text"])
            elif role == "muted_top":
                widget.configure(bg=c["topbar"], fg=c["muted"])
            elif role == "text_panel":
                widget.configure(bg=c["panel"], fg=c["text"])
            elif role == "muted_panel":
                widget.configure(bg=c["panel"], fg=c["muted"])
            elif role == "text_card":
                widget.configure(bg=c["card"], fg=c["text"])
            elif role == "muted_card":
                widget.configure(bg=c["card"], fg=c["muted"])
            elif role == "camera":
                # Fondo igual al panel para evitar barras negras visuales
                # cuando la proporción de la cámara no coincide con el cuadro.
                widget.configure(bg=c["panel_alt"], fg="#b5b5b5")
            elif role == "primary_button":
                widget.configure(
                    bg=c["accent"], fg=c["accent_text"],
                    activebackground=c["text"], activeforeground=c["bg"]
                )
            elif role == "button":
                widget.configure(
                    bg=c["button"], fg=c["text"],
                    activebackground=c["button_active"], activeforeground=c["text"]
                )
        except tk.TclError:
            pass

    # Algunos estados dinámicos necesitan colores especiales.
    if "status_dot" in globals():
        status_dot.configure(bg=c["panel_alt"], fg=c["ok"])
    if "status_label" in globals():
        status_label.configure(bg=c["panel_alt"], fg=c["muted"])
    if "detection_value" in globals():
        detection_value.configure(bg=c["card"])
    if "translation_value" in globals():
        translation_value.configure(bg=c["card"])

    if "settings_panel" in globals():
        settings_panel.configure(bg=c["panel"], highlightbackground=c["border"])
        appearance_label.configure(bg=c["panel"], fg=c["muted"])
        stabilization_label.configure(bg=c["panel"], fg=c["muted"])
        appearance_row.configure(bg=c["panel"])
        stabilization_row.configure(bg=c["panel"])
        settings_separator.configure(bg=c["border"])
        if "updates_separator" in globals():
            updates_separator.configure(bg=c["border"])
        if "updates_label" in globals():
            updates_label.configure(bg=c["panel"], fg=c["muted"])
        if "update_version_label" in globals():
            update_version_label.configure(bg=c["panel"], fg=c["text"])
        if "update_status_label" in globals():
            update_status_label.configure(bg=c["panel"])
        if "updates_actions" in globals():
            updates_actions.configure(bg=c["panel"])
        if "update_check_button" in globals():
            update_check_button.configure(
                bg=c["button"], fg=c["text"],
                activebackground=c["button_active"], activeforeground=c["text"],
                highlightbackground=c["border"],
            )
        if "update_download_button" in globals():
            update_download_button.configure(
                bg=c["accent"], fg=c["accent_text"],
                activebackground=c["button_active"], activeforeground=c["text"],
                highlightbackground=c["border"],
            )
        update_settings_controls()

    if "logo_box" in globals():
        # El logo usa fondo limpio dentro de la barra superior.
        logo_box.configure(
            bg=c["topbar"],
            fg=c["text"],
            activebackground=c["topbar"],
            highlightbackground=c["topbar"],
        )

    # Estilo especial de la cabecera, inspirado en la referencia.
    if "brand_manos" in globals():
        brand_manos.configure(
            bg=c["topbar"],
            fg="#F4F7FB" if theme_name == "Oscuro" else "#102238",
        )
    if "brand_que_hablan" in globals():
        brand_que_hablan.configure(
            bg=c["topbar"],
            fg="#11A8FF" if theme_name == "Oscuro" else "#0077C8",
        )
    if "brand_title" in globals():
        brand_title.configure(
            bg=c["topbar"],
            fg="#F0F3F8" if theme_name == "Oscuro" else "#26384D",
        )
    if "brand_subtitle" in globals():
        brand_subtitle.configure(
            bg=c["topbar"],
            fg="#16B7FF" if theme_name == "Oscuro" else "#0088CC",
        )
    if "brand_separator" in globals():
        brand_separator.configure(
            bg="#1D4D78" if theme_name == "Oscuro" else "#B7CDE3",
        )

    # Colores del menú lateral.
    if "sidebar_nav" in globals():
        sidebar_nav.configure(bg=c["panel"])
    if "sidebar_brand" in globals():
        sidebar_brand.configure(bg=c["panel"])
    if "sidebar_hand" in globals():
        sidebar_hand.configure(bg=c["panel"])
    if "sidebar_brand_title1" in globals():
        sidebar_brand_title1.configure(bg=c["panel"], fg=c["text"])
    if "sidebar_brand_title2" in globals():
        sidebar_brand_title2.configure(bg=c["panel"], fg="#12AFFF")
    if "sidebar_heart" in globals():
        sidebar_heart.configure(bg=c["panel"], fg="#168EFF")
    if "sidebar_slogan" in globals():
        sidebar_slogan.configure(bg=c["panel"], fg=c["muted"])
    if "sidebar_buttons" in globals():
        update_sidebar_style()
    if "features_panel" in globals():
        update_features_panel_theme()

    # El icono de cuenta no queda seleccionado permanentemente.
    # draw_account_icon() decide el color según el hover real.
    if "account_button" in globals():
        draw_account_icon()

    if "settings_button" in globals():
        draw_settings_gear()

    if "account_button" in globals():
        draw_account_icon()

    if "notification_button" in globals():
        draw_notification_icon()

    if "fullscreen_button" in globals():
        draw_fullscreen_icon()

    # Borde visible alrededor del área donde se muestra la cámara.
    if "camera_image_frame" in globals():
        camera_image_frame.configure(
            bg=c["panel"],
            highlightbackground=c["border"],
            highlightcolor=c["border"],
            bd=1,
            relief="solid"
        )


def on_theme_change(event=None):
    apply_theme()


def monitor_system_theme():
    global last_system_theme
    if "theme_var" in globals() and theme_var.get() == "Sistema":
        detected = detectar_tema_sistema()
        if detected != last_system_theme:
            apply_theme(detected)
    root.after(2500, monitor_system_theme)


root = tk.Tk()
root.title("Manos que Hablan · Traductor de Señas")

try:
    root.state("zoomed")
except tk.TclError:
    root.geometry("1280x800")

root.minsize(980, 650)

style = ttk.Style()
style.theme_use("clam")

# Fuente con aspecto técnico parecida a la referencia.
FONT = "DejaVu Sans Mono"

# ---------------- LOGO EMBEBIDO ----------------
# Imagen de manos añadida como icono de la aplicación.
# Está dentro del propio .py, así que no necesita archivos externos.
_LOGO_MANOS_DATA = """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"""
logo_manos_image = tk.PhotoImage(data=_LOGO_MANOS_DATA)

# ---------------- BARRA SUPERIOR ----------------

topbar = register_theme(tk.Frame(root, highlightthickness=0), "topbar")
topbar.pack(fill="x", padx=0, pady=0)

brand = register_theme(tk.Frame(topbar), "topbar")
brand.pack(side="left", padx=12, pady=5)

# Logo pequeño a la izquierda.
logo_box = register_theme(
    tk.Label(
        brand,
        image=logo_manos_image,
        relief="flat",
        bd=0,
        padx=2,
        pady=2,
    ),
    "text_top",
)
logo_box.image = logo_manos_image
logo_box.pack(side="left", padx=(0, 8))

# Nombre de la app en dos líneas, como en la referencia.
brand_name = register_theme(tk.Frame(brand), "topbar")
brand_name.pack(side="left", padx=(0, 12))

brand_manos = tk.Label(
    brand_name,
    text="MANOS",
    font=("DejaVu Sans", 13, "bold"),
    anchor="w",
)
brand_manos.pack(anchor="w")

brand_que_hablan = tk.Label(
    brand_name,
    text="QUE HABLAN",
    font=("DejaVu Sans", 13, "bold"),
    anchor="w",
)
brand_que_hablan.pack(anchor="w")

# Separador vertical fino.
brand_separator = tk.Frame(
    brand,
    width=1,
    height=36,
    bd=0,
)
brand_separator.pack(side="left", padx=(0, 14), pady=2)

# Título y lema a la derecha del separador.
brand_text = register_theme(tk.Frame(brand), "topbar")
brand_text.pack(side="left")

brand_title = tk.Label(
    brand_text,
    text="Traductor inteligente de lengua de señas",
    font=("DejaVu Sans", 8),
    anchor="w",
)
brand_title.pack(anchor="w")

brand_subtitle = tk.Label(
    brand_text,
    text="Entiende sin barreras, conecta sin límites.",
    font=("DejaVu Sans", 8, "italic"),
    anchor="w",
)
brand_subtitle.pack(anchor="w", pady=(2, 0))

header_controls = register_theme(tk.Frame(topbar), "topbar")
header_controls.pack(side="right", padx=14, pady=5)

theme_var = tk.StringVar(value="Sistema")
stabilization_var = tk.StringVar(value="Baja")
settings_panel_visible = False
settings_theme_buttons = {}
settings_stab_buttons = {}

# Estado visual de hover para iconos dibujados manualmente.
hovered_header_icon = None
fullscreen_hovered = False


def update_settings_controls():
    """Pinta como activo el ajuste seleccionado dentro del panel flotante."""
    if "settings_panel" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])

    for value, button in settings_theme_buttons.items():
        selected = theme_var.get() == value
        button.configure(
            bg=c["accent"] if selected else c["button"],
            fg=c["accent_text"] if selected else c["text"],
            activebackground=c["accent"] if selected else c["button_active"],
            activeforeground=c["accent_text"] if selected else c["text"],
            highlightbackground=c["border"],
        )

    for value, button in settings_stab_buttons.items():
        selected = stabilization_var.get() == value
        button.configure(
            bg=c["accent"] if selected else c["button"],
            fg=c["accent_text"] if selected else c["text"],
            activebackground=c["accent"] if selected else c["button_active"],
            activeforeground=c["accent_text"] if selected else c["text"],
            highlightbackground=c["border"],
        )


def seleccionar_tema(value):
    theme_var.set(value)
    on_theme_change()
    update_settings_controls()


def seleccionar_estabilizacion(value):
    stabilization_var.set(value)
    set_stabilization_mode()
    update_settings_controls()


def cerrar_panel_ajustes():
    global settings_panel_visible
    settings_panel.place_forget()
    settings_panel_visible = False


def toggle_panel_ajustes():
    global settings_panel_visible

    if settings_panel_visible:
        cerrar_panel_ajustes()
        return

    # Tarjeta flotante dentro de la propia ventana, alineada al engranaje.
    settings_panel.place(
        relx=1.0,
        x=-18,
        y=68,
        anchor="ne",
        width=330,
    )
    settings_panel.lift()
    settings_panel_visible = True
    update_settings_controls()


def draw_settings_gear(event=None):
    """Dibuja el engranaje manualmente para que siempre sea visible."""
    if "settings_button" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    is_hovered = globals().get("hovered_header_icon") == "settings"
    settings_button.configure(
        bg=c["button_active"] if is_hovered else c["button"],
        highlightbackground=c["accent"] if is_hovered else c["border"],
        highlightcolor=c["accent"] if is_hovered else c["border"],
    )
    settings_button.delete("all")

    import math

    cx, cy = 16, 16
    gear_color = c["text"]

    for i in range(8):
        angle = math.radians(i * 45)
        x1 = cx + math.cos(angle) * 7
        y1 = cy + math.sin(angle) * 7
        x2 = cx + math.cos(angle) * 11
        y2 = cy + math.sin(angle) * 11
        settings_button.create_line(
            x1, y1, x2, y2,
            fill=gear_color,
            width=2,
            capstyle=tk.ROUND,
        )

    settings_button.create_oval(
        cx - 8, cy - 8,
        cx + 8, cy + 8,
        outline=gear_color,
        width=2,
    )

    settings_button.create_oval(
        cx - 2.5, cy - 2.5,
        cx + 2.5, cy + 2.5,
        outline=gear_color,
        width=3,
    )


settings_button = tk.Canvas(
    header_controls,
    width=32,
    height=32,
    bd=0,
    highlightthickness=1,
    cursor="hand2",
)
settings_button.pack(side="right")
settings_button.bind("<Button-1>", lambda event: toggle_panel_ajustes())
settings_button.bind("<Configure>", draw_settings_gear)
settings_button.bind("<Enter>", lambda event: set_header_icon_hover("settings"))
settings_button.bind("<Leave>", lambda event: set_header_icon_hover(None))


def draw_account_icon(event=None):
    """Icono de cuenta refinado: limpio, redondo, centrado y equilibrado."""
    if "account_button" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    is_hovered = globals().get("hovered_header_icon") == "account"

    # En reposo usa exactamente el mismo fondo que los otros iconos.
    # Solo se resalta cuando el cursor está realmente encima.
    account_button.configure(
        bg=c["button_active"] if is_hovered else c["button"],
        highlightbackground=c["accent"] if is_hovered else c["border"],
        highlightcolor=c["accent"] if is_hovered else c["border"],
    )
    account_button.delete("all")

    icon_color = c["accent_text"]

    # Canvas 34x34. Todo el símbolo queda centrado en x=17.
    cx = 17

    # Cabeza: ligeramente más pequeña para que no se vea pesada.
    account_button.create_oval(
        cx - 4.6, 6.2,
        cx + 4.6, 15.4,
        outline=icon_color,
        width=2,
    )

    # Separación pequeña entre cabeza y hombros.
    # Hombros tipo icono de perfil moderno: una sola curva limpia.
    account_button.create_arc(
        7.0, 16.0,
        27.0, 32.0,
        start=18,
        extent=144,
        style=tk.ARC,
        outline=icon_color,
        width=2,
    )

    # Laterales cortos y redondeados para dar una silueta más natural.
    account_button.create_line(
        7.5, 25.0, 7.2, 27.8,
        fill=icon_color,
        width=2,
        capstyle=tk.ROUND,
    )
    account_button.create_line(
        26.5, 25.0, 26.8, 27.8,
        fill=icon_color,
        width=2,
        capstyle=tk.ROUND,
    )

def draw_notification_icon(event=None):
    """Campana visible con pequeño indicador de notificación."""
    if "notification_button" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    is_hovered = globals().get("hovered_header_icon") == "notification"
    notification_button.configure(
        bg=c["button_active"] if is_hovered else c["button"],
        highlightbackground=c["accent"] if is_hovered else c["border"],
        highlightcolor=c["accent"] if is_hovered else c["border"],
    )
    notification_button.delete("all")

    icon_color = c["text"]

    # Campana.
    notification_button.create_arc(
        9, 7, 25, 25,
        start=0,
        extent=180,
        style=tk.ARC,
        outline=icon_color,
        width=2,
    )
    notification_button.create_line(
        9, 16, 9, 23, 25, 23, 25, 16,
        fill=icon_color,
        width=2,
        smooth=True,
    )
    notification_button.create_line(
        7, 24, 27, 24,
        fill=icon_color,
        width=2,
    )
    notification_button.create_oval(
        15, 25, 19, 29,
        fill=icon_color,
        outline=icon_color,
    )

    # Punto azul para que el icono destaque.
    notification_button.create_oval(
        23, 5, 30, 12,
        fill=c["accent"],
        outline=c["accent_text"],
        width=1,
    )


def set_header_icon_hover(icon_name=None):
    """Ilumina suavemente el icono bajo el cursor."""
    global hovered_header_icon
    hovered_header_icon = icon_name

    if "settings_button" in globals():
        draw_settings_gear()
    if "account_button" in globals():
        draw_account_icon()
    if "notification_button" in globals():
        draw_notification_icon()


# Cuenta: queda inmediatamente al lado izquierdo del engranaje.
account_button = tk.Canvas(
    header_controls,
    width=34,
    height=34,
    bd=0,
    highlightthickness=1,
    cursor="hand2",
)
account_button.pack(side="right", padx=(7, 7))
account_button.bind("<Configure>", draw_account_icon)
account_button.bind("<Enter>", lambda event: set_header_icon_hover("account"))
account_button.bind("<Leave>", lambda event: set_header_icon_hover(None))

# Notificaciones: a la izquierda del icono de cuenta.
notification_button = tk.Canvas(
    header_controls,
    width=34,
    height=34,
    bd=0,
    highlightthickness=1,
    cursor="hand2",
)
notification_button.pack(side="right")
notification_button.bind("<Configure>", draw_notification_icon)
notification_button.bind("<Enter>", lambda event: set_header_icon_hover("notification"))
notification_button.bind("<Leave>", lambda event: set_header_icon_hover(None))


# ---------------- PANEL FLOTANTE DE AJUSTES ----------------
settings_panel = tk.Frame(
    root,
    highlightthickness=1,
    bd=0,
)

appearance_label = tk.Label(
    settings_panel,
    text="APARIENCIA",
    anchor="w",
    font=("DejaVu Sans", 8, "bold"),
)
appearance_label.pack(fill="x", padx=16, pady=(14, 7))

appearance_row = tk.Frame(settings_panel)
appearance_row.pack(fill="x", padx=16, pady=(0, 14))

for option in ("Sistema", "Oscuro", "Claro"):
    btn = tk.Button(
        appearance_row,
        text=option,
        command=lambda value=option: seleccionar_tema(value),
        relief="flat",
        bd=0,
        highlightthickness=1,
        font=("DejaVu Sans", 9, "bold"),
        padx=10,
        pady=7,
        cursor="hand2",
    )
    btn.pack(side="left", expand=True, fill="x", padx=(0, 5) if option != "Claro" else 0)
    settings_theme_buttons[option] = btn

    def _theme_enter(event, b=btn, value=option):
        c = THEMES.get(current_theme_name, THEMES["Oscuro"])
        if theme_var.get() != value:
            b.configure(bg=c["button_active"], highlightbackground=c["accent"])

    def _theme_leave(event, b=btn, value=option):
        update_settings_controls()

    btn.bind("<Enter>", _theme_enter)
    btn.bind("<Leave>", _theme_leave)

settings_separator = tk.Frame(settings_panel, height=1)
settings_separator.pack(fill="x", padx=16, pady=(0, 14))

stabilization_label = tk.Label(
    settings_panel,
    text="ESTABILIZACIÓN",
    anchor="w",
    font=("DejaVu Sans", 8, "bold"),
)
stabilization_label.pack(fill="x", padx=16, pady=(0, 7))

stabilization_row = tk.Frame(settings_panel)
stabilization_row.pack(fill="x", padx=16, pady=(0, 16))

for option in ("OFF", "Baja", "Media"):
    btn = tk.Button(
        stabilization_row,
        text=option,
        command=lambda value=option: seleccionar_estabilizacion(value),
        relief="flat",
        bd=0,
        highlightthickness=1,
        font=("DejaVu Sans", 9, "bold"),
        padx=10,
        pady=7,
        cursor="hand2",
    )
    btn.pack(side="left", expand=True, fill="x", padx=(0, 5) if option != "Media" else 0)
    settings_stab_buttons[option] = btn

    def _stab_enter(event, b=btn, value=option):
        c = THEMES.get(current_theme_name, THEMES["Oscuro"])
        if stabilization_var.get() != value:
            b.configure(bg=c["button_active"], highlightbackground=c["accent"])

    def _stab_leave(event, b=btn, value=option):
        update_settings_controls()

    btn.bind("<Enter>", _stab_enter)
    btn.bind("<Leave>", _stab_leave)

# ---------------- ACTUALIZACIONES ----------------
updates_separator = tk.Frame(settings_panel, height=1)
updates_separator.pack(fill="x", padx=16, pady=(0, 14))

updates_label = tk.Label(
    settings_panel,
    text="ACTUALIZACIONES",
    anchor="w",
    font=("DejaVu Sans", 8, "bold"),
)
updates_label.pack(fill="x", padx=16, pady=(0, 7))

update_version_label = tk.Label(
    settings_panel,
    text=f"Versión instalada: v{APP_VERSION}",
    anchor="w",
    font=("DejaVu Sans", 9, "bold"),
)
update_version_label.pack(fill="x", padx=16)

update_status_label = tk.Label(
    settings_panel,
    text="Puedes buscar una versión nueva en GitHub.",
    anchor="w",
    justify="left",
    wraplength=290,
    font=("DejaVu Sans", 8),
)
update_status_label.pack(fill="x", padx=16, pady=(4, 8))

updates_actions = tk.Frame(settings_panel)
updates_actions.pack(fill="x", padx=16, pady=(0, 16))

update_check_button = tk.Button(
    updates_actions,
    text="Buscar actualizaciones",
    command=buscar_actualizaciones_app,
    relief="flat",
    bd=0,
    highlightthickness=1,
    font=("DejaVu Sans", 9, "bold"),
    padx=10,
    pady=7,
    cursor="hand2",
)
update_check_button.pack(fill="x")

update_download_button = tk.Button(
    updates_actions,
    text="Descargar actualización",
    command=descargar_actualizacion_app,
    relief="flat",
    bd=0,
    highlightthickness=1,
    font=("DejaVu Sans", 9, "bold"),
    padx=10,
    pady=7,
    cursor="hand2",
)
# Se muestra solo cuando GitHub informa que existe una versión nueva.

# Estas dos filas deben seguir el color del panel flotante.
# Se actualizan desde apply_theme().

# Se conserva esta variable porque set_stabilization_mode() la actualiza.
stabilization_status = register_theme(
    tk.Label(header_controls, text="", font=(FONT, 8)),
    "muted_top",
)

online_badge = register_theme(
    tk.Label(
        header_controls,
        text="■ SISTEMA LISTO",
        font=(FONT, 9, "bold"),
        padx=14,
        pady=8,
        relief="solid",
        bd=1,
    ),
    "text_top",
)
# Oculto por petición: "SISTEMA LISTO"
# online_badge.pack(side="left")

# ---------------- ÁREA PRINCIPAL ----------------

main = register_theme(tk.Frame(root), "bg")
main.pack(fill="both", expand=True, padx=8, pady=10)

# Layout visual de tres columnas:
# rectángulo izquierdo | cámara centrada | rectángulo derecho
# Solo cambia la interfaz; no modifica la lógica de captura ni MediaPipe.
# Columna izquierda convertida en menú de navegación.
# Solo cambia la interfaz; la cámara y MediaPipe conservan su lógica.
main.grid_columnconfigure(0, weight=2, uniform="main_cols")
main.grid_columnconfigure(1, weight=6, uniform="main_cols")
main.grid_columnconfigure(2, weight=3, uniform="main_cols")
main.grid_rowconfigure(0, weight=1)

# Rectángulo lateral izquierdo
left_panel = register_theme(
    tk.Frame(main, highlightthickness=1),
    "panel",
)
left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
left_panel.grid_columnconfigure(0, weight=1)
left_panel.grid_rowconfigure(1, weight=1)

# ---------------- MENÚ LATERAL ----------------
# Este bloque es exclusivamente visual. No cambia funciones de cámara,
# procesamiento, MediaPipe ni los comandos existentes.
sidebar_nav = register_theme(tk.Frame(left_panel), "panel")
sidebar_nav.grid(row=0, column=0, sticky="new", padx=10, pady=(12, 4))

sidebar_buttons = {}
sidebar_active = "Inicio"

def draw_sidebar_icon(canvas, kind, active=False):
    """Dibuja iconos sencillos para evitar depender de emojis/fuentes."""
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    canvas.delete("all")
    color = "#FFFFFF" if active else ("#27AFFF" if current_theme_name == "Oscuro" else "#087AC1")
    canvas.configure(bg=c["accent"] if active else c["panel"], highlightthickness=0)

    if kind == "home":
        canvas.create_polygon(4, 11, 12, 4, 20, 11, outline=color, fill="", width=1.6)
        canvas.create_rectangle(7, 10, 17, 20, outline=color, width=1.6)
        canvas.create_rectangle(11, 14, 14, 20, outline=color, width=1.2)
    elif kind == "hand":
        canvas.create_line(7,19,6,11,8,10,9,15,9,7,11,7,12,14,12,6,14,6,15,14,15,8,17,8,18,16,20,14,
                           fill=color, width=1.5, smooth=True)
        canvas.create_line(7,19,12,21,17,18,20,14, fill=color, width=1.5, smooth=True)
    elif kind == "chat":
        canvas.create_rectangle(5,6,20,17, outline=color, width=1.5)
        canvas.create_line(9,17,8,21,13,17, fill=color, width=1.5)
        for x in (9,13,17):
            canvas.create_oval(x-1,10,x+1,12, fill=color, outline=color)
    elif kind == "learn":
        canvas.create_polygon(4,10,12,5,21,10,12,15, outline=color, fill="", width=1.5)
        canvas.create_line(7,12,7,17,12,20,18,17,18,12, fill=color, width=1.5)
        canvas.create_line(21,10,21,17, fill=color, width=1.2)
    elif kind == "history":
        canvas.create_oval(5,5,20,20, outline=color, width=1.5)
        canvas.create_line(12.5,8,12.5,13,17,16, fill=color, width=1.5)
    elif kind == "heart":
        canvas.create_line(4,10,6,6,10,5,13,8,16,5,20,6,22,10,21,14,13,21,5,14,4,10,
                           fill=color, width=1.5, smooth=True)
    elif kind == "gear":
        canvas.create_oval(7,7,19,19, outline=color, width=1.5)
        canvas.create_oval(11,11,15,15, outline=color, width=1.5)
        for x1,y1,x2,y2 in ((13,3,13,7),(13,19,13,23),(3,13,7,13),(19,13,23,13),
                             (6,6,9,9),(17,17,20,20),(20,6,17,9),(6,20,9,17)):
            canvas.create_line(x1,y1,x2,y2,fill=color,width=1.5)

def set_sidebar_active(name):
    global sidebar_active
    sidebar_active = name
    update_sidebar_style()

    # Al pulsar "Inicio", ocultamos COMPLETAMENTE el panel del lado derecho
    # y dejamos que la cámara use todo el espacio liberado.
    # No se modifica la lógica de captura ni MediaPipe.
    if name == "Inicio":
        side_panel.grid_remove()

        # Quitamos el reparto uniforme mientras el panel derecho está oculto.
        # Así la columna de la cámara puede crecer de verdad.
        main.grid_columnconfigure(0, weight=2, uniform="")
        main.grid_columnconfigure(1, weight=8, uniform="")
        main.grid_columnconfigure(2, weight=0, minsize=0, uniform="")

        # Recalculamos el área real de video después de que Tkinter
        # haya expandido la columna central.
        root.update_idletasks()
        actualizar_dimensiones_video()
        root.after(40, actualizar_dimensiones_video)
        root.after(120, actualizar_dimensiones_video)

    elif name == "Traducir":
        # Al volver a Traducir, restauramos el panel derecho y
        # las proporciones normales del dashboard.
        side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        main.grid_columnconfigure(0, weight=2, uniform="main_cols")
        main.grid_columnconfigure(1, weight=6, uniform="main_cols")
        main.grid_columnconfigure(2, weight=3, minsize=0, uniform="main_cols")

        # Volvemos a calcular el tamaño visible de la cámara para que
        # regrese exactamente a su espacio normal.
        root.update_idletasks()
        actualizar_dimensiones_video()
        root.after(40, actualizar_dimensiones_video)
        root.after(120, actualizar_dimensiones_video)

def update_sidebar_style():
    if "sidebar_buttons" not in globals():
        return
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    for name, data in sidebar_buttons.items():
        frame, icon, label, kind = data
        active = name == sidebar_active
        bg = c["accent"] if active else c["panel"]
        frame.configure(bg=bg)
        label.configure(
            bg=bg,
            fg="#FFFFFF" if active else c["text"],
            font=("DejaVu Sans", 9, "bold" if active else "normal"),
        )
        draw_sidebar_icon(icon, kind, active)

def sidebar_enter(name):
    if name == sidebar_active:
        return
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    frame, icon, label, kind = sidebar_buttons[name]
    frame.configure(bg=c["button_active"])
    label.configure(bg=c["button_active"])
    icon.configure(bg=c["button_active"])

def sidebar_leave(name):
    update_sidebar_style()

menu_items = [
    ("Inicio", "home"),
    ("Traducir", "hand"),
    ("Historial", "history"),
]

for menu_name, menu_icon in menu_items:
    item = tk.Frame(sidebar_nav, height=38, cursor="hand2")
    item.pack(fill="x", pady=2)
    item.pack_propagate(False)

    icon_canvas = tk.Canvas(item, width=26, height=26, bd=0, highlightthickness=0, cursor="hand2")
    icon_canvas.pack(side="left", padx=(8, 7), pady=6)

    item_label = tk.Label(
        item,
        text=menu_name,
        anchor="w",
        font=("DejaVu Sans", 9),
        cursor="hand2",
    )
    item_label.pack(side="left", fill="both", expand=True)

    sidebar_buttons[menu_name] = (item, icon_canvas, item_label, menu_icon)

    # Solo selección visual. No se conectan comandos nuevos a la lógica existente.
    for widget in (item, icon_canvas, item_label):
        widget.bind("<Button-1>", lambda event, n=menu_name: set_sidebar_active(n))
        widget.bind("<Enter>", lambda event, n=menu_name: sidebar_enter(n))
        widget.bind("<Leave>", lambda event, n=menu_name: sidebar_leave(n))

# Icono de mano embebido para la marca inferior del menú lateral.
# No necesita un archivo PNG externo al ejecutar la aplicación.
_SIDEBAR_HAND_ICON_DATA = """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"""
sidebar_hand_image = tk.PhotoImage(data=_SIDEBAR_HAND_ICON_DATA)

# Marca inferior inspirada en la referencia.
sidebar_brand = register_theme(tk.Frame(left_panel), "panel")
sidebar_brand.grid(row=1, column=0, sticky="sew", padx=12, pady=(8, 16))

sidebar_hand = tk.Label(
    sidebar_brand,
    image=sidebar_hand_image,
    bd=0,
    relief="flat",
)
# Mantener referencia para que Tkinter no elimine la imagen.
sidebar_hand.image = sidebar_hand_image
sidebar_hand.pack(pady=(4, 2))

sidebar_brand_title1 = tk.Label(
    sidebar_brand,
    text="MANOS",
    font=("DejaVu Sans", 13, "bold"),
)
sidebar_brand_title1.pack()

sidebar_brand_title2 = tk.Label(
    sidebar_brand,
    text="QUE HABLAN",
    font=("DejaVu Sans", 9, "bold"),
)
sidebar_brand_title2.pack()

sidebar_heart = tk.Label(
    sidebar_brand,
    text="—  ♥  —",
    font=("DejaVu Sans", 10, "bold"),
)
sidebar_heart.pack(pady=(6, 5))

sidebar_slogan = tk.Label(
    sidebar_brand,
    text="Traducimos señas,\nconectamos personas.",
    justify="center",
    font=("DejaVu Sans", 7),
)
sidebar_slogan.pack()

# Panel cámara, colocado físicamente en la columna central
camera_panel = register_theme(
    tk.Frame(main, highlightthickness=1),
    "panel",
)
camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)

camera_header = register_theme(tk.Frame(camera_panel), "panel")
camera_header.pack(fill="x", padx=14, pady=(12, 8))

# Encabezado "Cámara de señas" oculto por diseño.

live_badge = register_theme(
    tk.Label(camera_header, text="● LIVE", font=(FONT, 8, "bold")),
    "muted_panel",
)
# Oculto por petición: indicador "LIVE"
# live_badge.pack(side="right")

camera_container = register_theme(
    tk.Frame(camera_panel, highlightthickness=1),
    "panel_alt",
)
camera_container.pack(fill="both", expand=True, padx=14, pady=(0, 10))
# Evita que la imagen de la cámara pueda cambiar el tamaño solicitado
# del contenedor y, por rebote, ensanchar el panel.
camera_container.pack_propagate(False)

# Marco visible alrededor de la imagen de la cámara.
camera_image_frame = tk.Frame(
    camera_container,
    bd=1,
    relief="solid",
    highlightthickness=1,
)
camera_image_frame.pack(fill="both", expand=True, padx=10, pady=10)
camera_image_frame.pack_propagate(False)

camera_label = register_theme(
    tk.Label(camera_image_frame, text="Cámara apagada", font=(FONT, 13)),
    "camera",
)
# Centra visualmente la vista de la cámara dentro de su marco.
# No modifica la captura, MediaPipe, resolución ni lógica de procesamiento.
camera_label.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)

# Botón discreto de pantalla completa en la esquina superior derecha
# del área de la cámara. Solo cambia la visualización de la ventana.
# Botón de pantalla completa dibujado manualmente.
# No depende de emojis ni de que la fuente tenga el símbolo.
fullscreen_button = tk.Canvas(
    camera_image_frame,
    width=38,
    height=38,
    bd=0,
    highlightthickness=1,
    cursor="hand2",
)

def draw_fullscreen_icon(event=None):
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])

    is_hovered = globals().get("fullscreen_hovered", False)
    fullscreen_button.configure(
        bg=c["button_active"] if is_hovered else c["button"],
        highlightbackground=c["accent"] if is_hovered else c["border"],
        highlightcolor=c["accent"] if is_hovered else c["border"],
    )
    fullscreen_button.delete("all")

    color = c["text"]
    w = 38
    h = 38
    m = 10
    l = 7

    # Esquina superior izquierda
    fullscreen_button.create_line(m, m + l, m, m, m + l, m,
                                  fill=color, width=2)

    # Esquina superior derecha
    fullscreen_button.create_line(w - m - l, m, w - m, m, w - m, m + l,
                                  fill=color, width=2)

    # Esquina inferior izquierda
    fullscreen_button.create_line(m, h - m - l, m, h - m, m + l, h - m,
                                  fill=color, width=2)

    # Esquina inferior derecha
    fullscreen_button.create_line(w - m - l, h - m, w - m, h - m,
                                  w - m, h - m - l,
                                  fill=color, width=2)

fullscreen_button.place(relx=1.0, x=-12, y=12, anchor="ne")
def set_fullscreen_hover(value):
    global fullscreen_hovered
    fullscreen_hovered = value
    draw_fullscreen_icon()


fullscreen_button.bind("<Button-1>", lambda event: toggle_video_fullscreen())
fullscreen_button.bind("<Configure>", draw_fullscreen_icon)
fullscreen_button.bind("<Enter>", lambda event: set_fullscreen_hover(True))
fullscreen_button.bind("<Leave>", lambda event: set_fullscreen_hover(False))
draw_fullscreen_icon()

def add_button_hover(button, role="button"):
    """Aclara el botón mientras el cursor está encima, sin alterar su comando."""
    def on_enter(event=None):
        c = THEMES.get(current_theme_name, THEMES["Oscuro"])
        if str(button.cget("state")) != "disabled":
            button.configure(
                bg=c["button_active"] if role == "button" else c["button_active"],
                highlightbackground=c["accent"],
            )

    def on_leave(event=None):
        c = THEMES.get(current_theme_name, THEMES["Oscuro"])
        if role == "primary_button":
            button.configure(
                bg=c["accent"],
                highlightbackground=c["border"],
            )
        else:
            button.configure(
                bg=c["button"],
                highlightbackground=c["border"],
            )

    button.bind("<Enter>", on_enter, add="+")
    button.bind("<Leave>", on_leave, add="+")


# Controles de cámara estilo barra técnica
controls = register_theme(tk.Frame(camera_panel), "panel")
controls.pack(fill="x", padx=14, pady=(0, 8))
controls.grid_columnconfigure(0, weight=1)

camera_combo = ttk.Combobox(
    controls,
    state="readonly",
    style="Dashboard.TCombobox",
    font=(FONT, 9),
)
camera_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))

start_button = register_theme(
    tk.Button(
        controls,
        text="▶ Iniciar",
        command=iniciar_camara,
        relief="flat",
        bd=0,
        font=(FONT, 9, "bold"),
        padx=14,
        pady=8,
        cursor="hand2",
    ),
    "primary_button",
)
start_button.grid(row=0, column=1, padx=(8, 4))
add_button_hover(start_button, "primary_button")

stop_button = register_theme(
    tk.Button(
        controls,
        text="■ Detener",
        command=detener_camara,
        relief="flat",
        bd=0,
        font=(FONT, 9, "bold"),
        padx=14,
        pady=8,
        cursor="hand2",
        state="disabled",
    ),
    "button",
)
stop_button.grid(row=0, column=2, padx=(4, 4))
add_button_hover(stop_button, "button")

# Icono de refresh embebido en el propio archivo.
# No depende de fuentes, emojis ni archivos externos.
_REFRESH_ICON_DATA = """iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAhklEQVR4nO1VSQ7AIAgcm76C/7+Nb9iTCW1kMeClcRIvgjOAAYADAMzcNVvLkBBR+9rGXVjAis4DETVVIEM8yAElA6sckSCkryswI075M3MfxyPWRCSuFRILWuQvgdXSRFCWwRGoEZAfm+3kqcAObO/kO/p4ZRZJlE7TWRAl+8AqY8lG+zceBCpWk3RORvoAAAAASUVORK5CYII="""
refresh_icon = tk.PhotoImage(data=_REFRESH_ICON_DATA)

refresh_button = register_theme(
    tk.Button(
        controls,
        image=refresh_icon,
        command=actualizar_camaras,
        relief="flat",
        bd=0,
        padx=8,
        pady=5,
        cursor="hand2",
    ),
    "button",
)
# Mantener una referencia al icono para evitar que Tkinter lo elimine.
refresh_button.image = refresh_icon

# Botón para volver a detectar/refrescar las cámaras disponibles.
refresh_button.grid(row=0, column=3, padx=(4, 0))
add_button_hover(refresh_button, "button")

# El selector de estabilización fue trasladado a la barra superior.
# Se deja libre este espacio debajo de los controles de cámara.


# Panel derecho: traducción + telemetría
side_panel = register_theme(
    tk.Frame(main, highlightthickness=1),
    "panel",
)
side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

# Estado de cámara colocado en el panel derecho.
# La lógica de set_status() sigue usando exactamente status_label.
status_box = register_theme(
    tk.Frame(side_panel, highlightthickness=1),
    "panel_alt",
)
# Oculto: caja superior "Cámara activa..."
# status_box.pack(fill="x", padx=14, pady=(12, 8))

status_dot = tk.Label(status_box, text="●", font=(FONT, 9))
status_dot.pack(side="left", padx=(10, 6), pady=8)

status_label = tk.Label(status_box, text="Preparando...", font=(FONT, 8))
status_label.pack(side="left", pady=8)


side_header = register_theme(tk.Frame(side_panel), "panel")
side_header.pack(fill="x", padx=14, pady=(12, 8))
# Título "Estado del sistema" oculto para dejar esta zona limpia.
# Se conserva side_header para no alterar la estructura general de la interfaz.
translation_card = register_theme(
    tk.Frame(side_panel, highlightthickness=1),
    "card",
)
# translation_card.pack(...) oculto: reemplazado por panel de funciones

register_theme(
    tk.Label(translation_card, text="SEÑA DETECTADA", font=(FONT, 8, "bold")),
    "muted_card",
)
# Oculto visualmente: "SEÑA DETECTADA"

translation_value = register_theme(
    tk.Label(
        translation_card,
        text="Esperando una seña...",
        justify="left",
        wraplength=390,
        font=(FONT, 22, "bold"),
    ),
    "text_card",
)
# No se empaqueta translation_value: la variable sigue existiendo para
# mantener intacta la lógica interna, pero el cuadro queda visualmente vacío.

# Pequeño bloque de actividad visual tipo gráfico de la referencia.
activity_card = register_theme(
    tk.Frame(side_panel, highlightthickness=1),
    "card",
)
# Más compacto: conserva todo el ancho del panel, pero ocupa menos altura.
# activity_card.pack(...) oculto: cuadro de actividad visual eliminado de la interfaz

register_theme(
    tk.Label(activity_card, text="ACTIVIDAD DE DETECCIÓN", font=(FONT, 8, "bold")),
    "muted_card",
).pack(anchor="w", padx=12, pady=(10, 5))

activity_canvas = tk.Canvas(activity_card, height=88, highlightthickness=0, bd=0)
activity_canvas.pack(fill="x", padx=12, pady=(0, 8))
register_theme(activity_canvas, "card")

# ---------------- PANEL DE TRADUCCIÓN REDISEÑADO ----------------
# Solo cambia la interfaz del panel derecho. La cámara y MediaPipe siguen intactos.

features_panel = register_theme(tk.Frame(side_panel), "panel")
features_panel.pack(fill="both", expand=True, padx=12, pady=(4, 12))

# Encabezado del modo Traducir.
translate_header = register_theme(tk.Frame(features_panel), "panel")
translate_header.pack(fill="x", pady=(2, 10))

translate_title = register_theme(
    tk.Label(
        translate_header,
        text="TRADUCCIÓN",
        font=("DejaVu Sans", 15, "bold"),
        anchor="w",
    ),
    "text_panel",
)
translate_title.pack(anchor="w")

translate_subtitle = register_theme(
    tk.Label(
        translate_header,
        text="La seña reconocida aparecerá aquí.",
        font=("DejaVu Sans", 8),
        anchor="w",
    ),
    "muted_panel",
)
translate_subtitle.pack(anchor="w", pady=(3, 0))

# Tarjeta principal: reutiliza translation_value para no alterar la lógica existente.
translation_card.pack(fill="x", pady=(0, 10))

translation_card_label = register_theme(
    tk.Label(
        translation_card,
        text="RESULTADO",
        font=("DejaVu Sans", 8, "bold"),
        anchor="w",
    ),
    "muted_card",
)
translation_card_label.pack(fill="x", padx=14, pady=(13, 4))

translation_value.configure(
    justify="left",
    anchor="w",
    wraplength=330,
    font=("DejaVu Sans", 20, "bold"),
)
translation_value.pack(fill="x", padx=14, pady=(4, 16))

# Estado / confianza. Por ahora no se inventa un porcentaje: queda en -- hasta
# que el modelo de reconocimiento entregue una confianza real.
translation_info = register_theme(
    tk.Frame(features_panel, highlightthickness=1),
    "card",
)
translation_info.pack(fill="x", pady=(0, 10))

translation_info_top = register_theme(tk.Frame(translation_info), "card")
translation_info_top.pack(fill="x", padx=12, pady=(11, 4))

translation_status_title = register_theme(
    tk.Label(
        translation_info_top,
        text="Estado de detección",
        font=("DejaVu Sans", 8, "bold"),
        anchor="w",
    ),
    "text_card",
)
translation_status_title.pack(side="left")

translation_status_value = register_theme(
    tk.Label(
        translation_info_top,
        text="En espera",
        font=("DejaVu Sans", 8, "bold"),
        anchor="e",
    ),
    "muted_card",
)
translation_status_value.pack(side="right")

translation_confidence_row = register_theme(tk.Frame(translation_info), "card")
translation_confidence_row.pack(fill="x", padx=12, pady=(3, 11))

translation_confidence_title = register_theme(
    tk.Label(
        translation_confidence_row,
        text="Confianza del reconocimiento",
        font=("DejaVu Sans", 8),
        anchor="w",
    ),
    "muted_card",
)
translation_confidence_title.pack(side="left")

translation_confidence_value = register_theme(
    tk.Label(
        translation_confidence_row,
        text="--",
        font=("DejaVu Sans", 9, "bold"),
        anchor="e",
    ),
    "text_card",
)
translation_confidence_value.pack(side="right")

# Acciones rápidas.
translate_actions = register_theme(tk.Frame(features_panel), "panel")
translate_actions.pack(fill="x", pady=(0, 10))
for col in range(3):
    translate_actions.grid_columnconfigure(col, weight=1)


def copiar_traduccion():
    texto = translation_value.cget("text").strip()
    if not texto or texto == "Esperando una seña...":
        set_status("Todavía no hay una traducción para copiar.")
        return
    root.clipboard_clear()
    root.clipboard_append(texto)
    root.update()
    set_status("Traducción copiada al portapapeles.")


def escuchar_traduccion():
    texto = translation_value.cget("text").strip()
    if not texto or texto == "Esperando una seña...":
        set_status("Todavía no hay una traducción para escuchar.")
        return

    # Usa un lector del sistema si existe, sin añadir dependencias al proyecto.
    import shutil
    try:
        if shutil.which("spd-say"):
            subprocess.Popen(["spd-say", texto])
            set_status("Reproduciendo traducción por voz.")
        elif shutil.which("espeak"):
            subprocess.Popen(["espeak", texto])
            set_status("Reproduciendo traducción por voz.")
        else:
            set_status("No se encontró un lector de voz del sistema.")
    except Exception:
        set_status("No se pudo reproducir la traducción por voz.", error=True)


def limpiar_traduccion():
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    translation_value.configure(text="Esperando una seña...", fg=c["muted"])
    translation_status_value.configure(text="En espera", fg=c["muted"])
    translation_confidence_value.configure(text="--")
    set_status("Traducción limpiada.")


def make_translate_action(parent, text, command, column, primary=False):
    role = "primary_button" if primary else "button"
    button = register_theme(
        tk.Button(
            parent,
            text=text,
            command=command,
            relief="flat",
            bd=0,
            padx=7,
            pady=8,
            cursor="hand2",
            font=("DejaVu Sans", 8, "bold"),
        ),
        role,
    )
    button.grid(
        row=0,
        column=column,
        sticky="ew",
        padx=(0, 4) if column == 0 else ((4, 4) if column == 1 else (4, 0)),
    )
    add_button_hover(button, role)
    return button


copy_translation_button = make_translate_action(
    translate_actions, "⧉  Copiar", copiar_traduccion, 0
)
listen_translation_button = make_translate_action(
    translate_actions, "▶  Escuchar", escuchar_traduccion, 1, primary=True
)
clear_translation_button = make_translate_action(
    translate_actions, "×  Limpiar", limpiar_traduccion, 2
)

# Mensaje inferior discreto.
translate_hint = register_theme(
    tk.Label(
        features_panel,
        text="Coloca tu mano frente a la cámara para comenzar.",
        justify="center",
        wraplength=320,
        font=("DejaVu Sans", 8),
    ),
    "muted_panel",
)
translate_hint.pack(fill="x", pady=(5, 0))


def update_features_panel_theme():
    """Compatibilidad con apply_theme(): repinta el nuevo panel Traducir."""
    if "features_panel" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    features_panel.configure(bg=c["panel"])

    # Mantener el estado dinámico con el color correspondiente.
    if "translation_status_value" in globals():
        current = translation_status_value.cget("text")
        translation_status_value.configure(
            fg=c["ok"] if current == "Detectando" else c["muted"]
        )

# ---------------- MÉTRICAS EN PANEL DERECHO ----------------

# Solo cambia su ubicación visual; la lógica permanece igual.

metrics_right = register_theme(tk.Frame(side_panel), "panel")
# metrics_right.pack(...) oculto: reemplazado por panel de funciones

for i in range(2):
    metrics_right.grid_columnconfigure(i, weight=1)


def make_metric(parent, title, icon="◇"):
    box = register_theme(tk.Frame(parent, highlightthickness=1), "card")
    head = register_theme(tk.Frame(box), "card")
    head.pack(fill="x", padx=10, pady=(9, 2))

    register_theme(
        tk.Label(head, text=icon, font=(FONT, 11)),
        "muted_card",
    ).pack(side="left", padx=(0, 6))

    value = register_theme(
        tk.Label(head, text="--", font=(FONT, 13, "bold")),
        "text_card",
    )
    value.pack(side="left")

    register_theme(
        tk.Label(box, text=title, font=(FONT, 7)),
        "muted_card",
    ).pack(anchor="w", padx=10, pady=(1, 9))

    metric_widgets.append((box, value))
    return box, value


resolution_box, resolution_value = make_metric(metrics_right, "Resolución", "▣")
# Oculto por diseño: resolution_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 8))

hands_box, hands_value = make_metric(metrics_right, "Manos detectadas", "♧")
# Oculto por diseño: hands_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 8))

camera_fps_box, camera_fps_value = make_metric(metrics_right, "FPS cámara", "↗")
# Oculto por diseño: camera_fps_box.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(0, 8))

mediapipe_fps_box, mediapipe_fps_value = make_metric(metrics_right, "FPS MediaPipe", "⌁")
# Oculto por diseño: mediapipe_fps_box.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(0, 8))

latency_box, latency_value = make_metric(metrics_right, "Latencia", "◴")
# Oculto por diseño: latency_box.grid(row=2, column=0, sticky="nsew", padx=(0, 4), pady=(0, 8))

fps_box, fps_value = make_metric(metrics_right, "FPS interfaz", "∿")
# Oculto por diseño: fps_box.grid(row=2, column=1, sticky="nsew", padx=(4, 0), pady=(0, 8))

detection_box, detection_value = make_metric(metrics_right, "Estado", "◎")
# Oculto por diseño: detection_box.grid(row=3, column=0, columnspan=2, sticky="nsew")

resolution_value.configure(text="--")
hands_value.configure(text="0")
camera_fps_value.configure(text="--")
mediapipe_fps_value.configure(text="--")
latency_value.configure(text="--")
fps_value.configure(text="--")
detection_value.configure(text="En espera")


# Footer mínimo estilo consola/dashboard
footer = register_theme(tk.Frame(root), "topbar")
footer.pack(fill="x")
register_theme(
    tk.Label(
        footer,
        text="Manos que Hablan  •  MediaPipe Hands  •  Sistema de reconocimiento en tiempo real",
        font=(FONT, 7),
    ),
    "muted_top",
).pack(pady=5)



# Estado interno del ECG. Solo afecta al dibujo del panel.
ecg_history = []
ecg_last_processed_id = -1
ecg_phase = 0.0
ecg_point_pulse = 0


def _ecg_wave(phase):
    """Forma P-QRS-T aproximada, phase entre 0 y 1."""
    import math

    # Pulso P pequeño
    p = 0.13 * math.exp(-((phase - 0.18) / 0.045) ** 2)

    # Complejo QRS: Q pequeña, R alta y S negativa
    q = -0.18 * math.exp(-((phase - 0.355) / 0.018) ** 2)
    r = 1.00 * math.exp(-((phase - 0.395) / 0.014) ** 2)
    s = -0.34 * math.exp(-((phase - 0.435) / 0.022) ** 2)

    # Onda T más ancha
    t = 0.28 * math.exp(-((phase - 0.68) / 0.075) ** 2)

    return p + q + r + s + t


def update_activity_graph():
    """
    ECG técnico:
    - La señal SOLO avanza cuando MediaPipe termina un frame nuevo.
    - El punto final representa el último frame procesado.
    - El color del punto representa la latencia de procesamiento.
    """
    global ecg_history, ecg_last_processed_id, ecg_phase, ecg_point_pulse

    if "activity_canvas" not in globals():
        return

    c = THEMES[current_theme_name]
    activity_canvas.configure(bg=c["card"])
    activity_canvas.delete("all")

    w = max(80, activity_canvas.winfo_width())
    h = max(60, activity_canvas.winfo_height())

    top_margin = 30
    bottom_margin = 10
    graph_h = max(30, h - top_margin - bottom_margin)
    mid = top_margin + graph_h * 0.52
    amplitude = max(10.0, min(28.0, graph_h * 0.34))

    # Tomamos juntos los datos del último resultado REAL de MediaPipe.
    with lock:
        hc = latest_hand_count
        processed_id = latest_processed_frame_id
        latency_ms = latest_processing_latency_ms

    # Línea central muy tenue.
    activity_canvas.create_line(
        10, mid, w - 10, mid,
        fill=c["border"],
        width=1
    )

    # La señal SOLO genera una muestra nueva cuando cambia el ID procesado.
    new_processed_frame = (
        processed_id >= 0
        and processed_id != ecg_last_processed_id
    )

    if new_processed_frame:
        if hc > 0:
            fps = max(1.0, mediapipe_fps)

            # Forma ECG visual estable.
            # El avance ocurre una vez por cada nuevo resultado procesado.
            beats_per_second = 1.20
            ecg_phase = (ecg_phase + beats_per_second / fps) % 1.0
            value = _ecg_wave(ecg_phase)
        else:
            ecg_phase = 0.0
            value = 0.0

        ecg_history.append(value)
        ecg_last_processed_id = processed_id

        # El punto "late" brevemente cada vez que llega un frame nuevo.
        ecg_point_pulse = 3
    else:
        ecg_point_pulse = max(0, ecg_point_pulse - 1)

    # Conservamos solo lo necesario para cubrir el ancho.
    pixels_per_sample = 3
    max_samples = max(20, int((w - 24) / pixels_per_sample))

    if len(ecg_history) > max_samples:
        ecg_history = ecg_history[-max_samples:]

    x_start = w - 12 - (len(ecg_history) - 1) * pixels_per_sample

    points = []
    for i, value in enumerate(ecg_history):
        x = x_start + i * pixels_per_sample
        y = mid - value * amplitude
        points.extend((x, y))

    # Color del indicador según latencia real.
    # Verde = respuesta rápida
    # Ámbar = respuesta media
    # Rojo = latencia alta
    if latency_ms <= 55:
        point_color = c["ok"]
        latency_state = "RÁPIDA"
    elif latency_ms <= 110:
        point_color = "#d6a84b"
        latency_state = "MEDIA"
    else:
        point_color = c["danger"]
        latency_state = "ALTA"

    if len(points) >= 4:
        activity_canvas.create_line(
            *points,
            fill=c["ok"] if hc > 0 else c["metric_line"],
            width=2 if hc > 0 else 1,
            smooth=False,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND
        )

        # Este punto ES el último frame procesado mostrado por el gráfico.
        last_x = points[-2]
        last_y = points[-1]

        # Se agranda apenas durante unos refrescos cuando entra un frame nuevo.
        radius = 3.8 if ecg_point_pulse > 0 else 2.8

        activity_canvas.create_oval(
            last_x - radius, last_y - radius,
            last_x + radius, last_y + radius,
            fill=point_color if hc > 0 else c["metric_line"],
            outline=""
        )
    else:
        activity_canvas.create_line(
            12, mid, w - 12, mid,
            fill=c["metric_line"],
            width=1
        )

    # El punto y la señal quedan como único indicador visual.

    root.after(15, update_activity_graph)


def bloquear_ancho_paneles():
    """Congela las dimensiones iniciales para que nada cambie el layout."""
    root.update_idletasks()

    camera_width = max(1, camera_panel.winfo_width())
    camera_height = max(1, camera_panel.winfo_height())
    side_width = max(1, side_panel.winfo_width())
    side_height = max(1, side_panel.winfo_height())

    camera_panel.configure(width=camera_width, height=camera_height)
    side_panel.configure(width=side_width, height=side_height)

    # IMPORTANTE:
    # Los hijos de estos paneles están colocados con pack(), no con grid().
    # Por eso usamos pack_propagate(False) para impedir que la imagen de
    # la cámara o los botones alteren el tamaño del panel.
    camera_panel.pack_propagate(False)
    side_panel.pack_propagate(False)

    # El contenedor de video tampoco puede adoptar el tamaño de la imagen.
    camera_container.pack_propagate(False)


# Aplicar tema inicial respetando el sistema.
apply_theme()

# Fijamos el layout tal como se ve antes de encender la cámara.
root.update_idletasks()
bloquear_ancho_paneles()

# La aplicación ARRANCA directamente en la vista "Inicio".
# Esto oculta el panel derecho desde el primer momento y deja que
# la cámara ocupe automáticamente el espacio liberado, sin que el
# usuario tenga que pulsar el botón Inicio.
set_sidebar_active("Inicio")

# Dimensiones iniciales del área visible de la cámara, ya con la
# distribución de Inicio aplicada.
root.update_idletasks()
CAMERA_VIEW_WIDTH = max(1, camera_label.winfo_width())
CAMERA_VIEW_HEIGHT = max(1, camera_label.winfo_height())


# ==========================================================
# ARRANQUE
# ==========================================================

root.protocol("WM_DELETE_WINDOW", cerrar_app)
root.bind("<Escape>", salir_video_fullscreen)
root.bind("<F11>", toggle_video_fullscreen)

actualizar_camaras()
actualizar_video()
update_activity_graph()
monitor_system_theme()

root.mainloop()
