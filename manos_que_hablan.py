import cv2
import mediapipe as mp
import threading
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import json
import urllib.request
import urllib.error
import urllib.parse
import queue
import hashlib
import webbrowser
import base64
from collections import deque
from pathlib import Path


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

# Versión instalada de la aplicación.
# Al publicar una nueva Release, actualiza este valor (por ejemplo 1.0.2).
APP_VERSION = "1.0.7"

# GitHub Releases se usa como servidor de actualizaciones.
GITHUB_OWNER = "manosqhablan26-coder"
GITHUB_REPO = "manos-que-hablan-"
GITHUB_PROJECT_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
GITHUB_MODELS_PAGE_URL = f"{GITHUB_PROJECT_URL}/tree/main/modelos"
DRIVE_MODELS_PAGE_URL = "https://drive.google.com/drive/folders/1hrsKKJV7UN7mhhnBvljzUQSSfOdFbWaQ?usp=drive_link"
GITHUB_LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)

# ==========================================================
# MODELO DE SEÑAS EN GITHUB · SIN DELAY EN LA CÁMARA
# ==========================================================
# GitHub se usa SOLO para comprobar/descargar el archivo del modelo.
# MediaPipe y reconocer_sena() siguen funcionando 100 % localmente.
MODEL_GITHUB_BRANCH = "main"
MODEL_GITHUB_FOLDER = "modelos"
MODEL_MANIFEST_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/"
    f"{MODEL_GITHUB_BRANCH}/{MODEL_GITHUB_FOLDER}/manifest.json"
)

# El último modelo descargado queda guardado aquí para poder abrir la app
# incluso si en ese momento no hay Internet.
MODEL_CACHE_DIR = Path.home() / ".manos_que_hablan" / "modelos"
MODEL_CACHE_FILE = MODEL_CACHE_DIR / "modelo_senas.json"
MODEL_CACHE_MANIFEST = MODEL_CACHE_DIR / "manifest.json"

# Cada seña entrenada localmente se guarda en su propio JSON.
# Ejemplo: HOLA.json, A.json, GRACIAS.json, etc.
# En Windows usamos LOCALAPPDATA para que los modelos sobrevivan al empaquetado
# como .exe (especialmente PyInstaller one-file, que usa una carpeta temporal).
if os.name == "nt":
    _windows_appdata = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    LOCAL_TRAINED_MODELS_DIR = _windows_appdata / "ManosQueHablan" / "modelos_entrenados"
else:
    LOCAL_TRAINED_MODELS_DIR = Path(__file__).resolve().parent / "modelos_entrenados"

# Historial persistente de palabras y oraciones creadas con señas.
# Se guarda en la carpeta del usuario para que sobreviva a actualizaciones
# y también funcione cuando la app se empaquete como ejecutable.
HISTORY_DIR = Path.home() / ".manos_que_hablan"
HISTORY_FILE = HISTORY_DIR / "historial_traducciones.json"
history_entries = []


def cargar_historial_local():
    global history_entries
    try:
        if HISTORY_FILE.exists():
            with HISTORY_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                history_entries = [item for item in data if isinstance(item, dict)]
            else:
                history_entries = []
        else:
            history_entries = []
    except Exception:
        history_entries = []
    return history_entries


def guardar_historial_local():
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        temporal = HISTORY_FILE.with_suffix(".json.part")
        with temporal.open("w", encoding="utf-8") as fh:
            json.dump(history_entries[-500:], fh, ensure_ascii=False, indent=2)
        temporal.replace(HISTORY_FILE)
    except Exception:
        pass


def agregar_historial(texto, tipo="Palabra", entry_id=None):
    texto = str(texto or "").strip()
    if not texto:
        return None
    entry_id = entry_id or str(time.time_ns())
    history_entries.append({
        "id": entry_id,
        "tipo": str(tipo),
        "texto": texto,
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if len(history_entries) > 500:
        del history_entries[:-500]
    guardar_historial_local()
    if "refrescar_historial_ui" in globals():
        try:
            refrescar_historial_ui()
        except Exception:
            pass
    return entry_id


def actualizar_historial(entry_id, texto, tipo="Oración"):
    texto = str(texto or "").strip()
    if not entry_id or not texto:
        return None
    for item in history_entries:
        if item.get("id") == entry_id:
            item["texto"] = texto
            item["tipo"] = str(tipo)
            item["fecha"] = time.strftime("%Y-%m-%d %H:%M:%S")
            guardar_historial_local()
            if "refrescar_historial_ui" in globals():
                try:
                    refrescar_historial_ui()
                except Exception:
                    pass
            return entry_id
    return agregar_historial(texto, tipo, entry_id=entry_id)


def eliminar_historial(entry_id):
    if not entry_id:
        return
    history_entries[:] = [item for item in history_entries if item.get("id") != entry_id]
    guardar_historial_local()


# Recupera el historial de sesiones anteriores desde el arranque.
cargar_historial_local()

model_online_version = None
model_sync_in_progress = False

# Información de la última versión encontrada.
latest_release_info = None
# La campanita solo muestra el punto cuando existe una versión más nueva.
update_notification_unread = False

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
# MEDIAPIPE FACE MESH · OPCIONAL Y SIN INTERFERIR CON MANOS
# ==========================================================
# La cara NO se usa para adivinar emociones. Solo extraemos rasgos visibles
# (apertura de ojos/boca, posición de cejas, labios e inclinación) para que
# una seña entrenada pueda usar la expresión facial como información adicional.
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    FACE_MESH_AVAILABLE = True
except Exception:
    mp_face_mesh = None
    face_mesh = None
    FACE_MESH_AVAILABLE = False

# Se procesa a baja resolución y en un hilo independiente. Así, si Face Mesh
# no está disponible o no se necesita, toda la lógica original de manos sigue.
FACE_PROCESS_WIDTH = 320
FACE_PROCESS_HEIGHT = 180
FACE_PROCESS_INTERVAL = 0.090   # ~11 FPS; suficiente para gestos faciales.
FACE_LIVE_MAX_AGE = 0.35
FACE_STATIC_WEIGHT = 0.15       # Manos siguen teniendo el mayor peso.
FACE_DYNAMIC_WEIGHT = 0.12
FACE_MAX_DISTANCE = 0.40

# Vista ligera de la cara: solo unos pocos puntos/segmentos para no llenar
# la cámara de marcas ni afectar demasiado el rendimiento.
SHOW_HAND_POINTS = True
SHOW_FACE_POINTS = True
FACE_DISPLAY_INDICES = [
    33, 133, 159, 145,      # ojo izquierdo
    362, 263, 386, 374,     # ojo derecho
    105, 107, 334, 336,     # cejas
    61, 13, 14, 291,        # boca
]
FACE_DISPLAY_CONNECTIONS = [
    (33, 159), (159, 145), (145, 133),
    (362, 386), (386, 374), (374, 263),
    (105, 107), (334, 336),
    (61, 13), (13, 14), (14, 291),
]
FACE_DISPLAY_POINT_RADIUS = 1
FACE_DISPLAY_POINT_COLOR = (255, 210, 110)
FACE_DISPLAY_LINE_COLOR = (140, 200, 255)


# ==========================================================
# VARIABLES GLOBALES
# ==========================================================

cap = None
capture_thread = None
processing_thread = None
face_thread = None

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
# MODELO DE RECONOCIMIENTO
# ==========================================================
# El archivo JSON cargado se convierte a vectores normalizados una sola vez.
# Así la comparación durante la cámara es ligera y no modifica la captura.
loaded_recognition_model_path = None
loaded_recognition_model_data = None

# El reconocedor final combina dos fuentes en memoria:
# 1) modelos externos (GitHub o cargados manualmente)
# 2) todos los JSON de modelos_entrenados/
recognition_external_samples = []
recognition_local_samples = []
recognition_local_files = []
# Contiene tanto muestras estáticas como secuencias dinámicas ya preparadas.
recognition_model_samples = []

# Reconocimiento temporal para señas que dependen del movimiento (por ejemplo J/Z
# o palabras cuyo significado está en la trayectoria). El buffer conserva solo
# landmarks; nunca guarda imágenes, por lo que el coste de memoria es pequeño.
DYNAMIC_SEQUENCE_STEPS = 16
DYNAMIC_BUFFER_MAX_FRAMES = 48
DYNAMIC_MIN_FRAMES = 8
DYNAMIC_MIN_MOTION = 0.030
DYNAMIC_MAX_DISTANCE = 0.62
DYNAMIC_RECOGNITION_EVERY = 3
dynamic_recognition_buffer = deque(maxlen=DYNAMIC_BUFFER_MAX_FRAMES)
dynamic_recognition_tick = 0
dynamic_last_result = (None, 0.0)

# El reconocimiento de SEÑAS arranca desactivado. MediaPipe puede seguir
# detectando la mano para la cámara y para Entrenar modelo, pero no se
# clasifica ninguna seña hasta cargar un modelo o guardar entrenamiento.
recognition_enabled = False
latest_recognized_sign = None
latest_recognition_confidence = 0.0

# Últimos landmarks ya procesados por el hilo principal de MediaPipe.
# La ventana de entrenamiento reutiliza estos datos para capturar muestras
# a la misma velocidad del procesamiento, sin crear otro detector por muestra.
latest_recognition_hands_data = []

# Últimos rasgos faciales. No contienen imagen ni identidad: solo un pequeño
# vector numérico normalizado. Se actualiza únicamente cuando hace falta.
latest_face_features = None
latest_face_frame_id = -1
latest_face_time = 0.0
latest_face_overlay_points = []
latest_face_overlay_frame_id = -1
face_training_requested = False


def _distancia_2d(puntos, a, b):
    ax, ay, _ = puntos[a]
    bx, by, _ = puntos[b]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _extraer_vector_facial(face_landmarks):
    """Convierte Face Mesh a rasgos geométricos normalizados y observables."""
    try:
        puntos = [(float(lm.x), float(lm.y), float(lm.z)) for lm in face_landmarks.landmark]
    except Exception:
        return None

    if len(puntos) < 455:
        return None

    face_width = _distancia_2d(puntos, 234, 454)
    face_height = _distancia_2d(puntos, 10, 152)
    left_eye_width = _distancia_2d(puntos, 33, 133)
    right_eye_width = _distancia_2d(puntos, 362, 263)
    mouth_width = _distancia_2d(puntos, 61, 291)

    if min(face_width, face_height, left_eye_width, right_eye_width) < 1e-6:
        return None

    # Aperturas relativas: invariantes a acercarse/alejarse de la cámara.
    eye_left_open = _distancia_2d(puntos, 159, 145) / left_eye_width
    eye_right_open = _distancia_2d(puntos, 386, 374) / right_eye_width
    mouth_open = _distancia_2d(puntos, 13, 14) / face_width
    outer_mouth_open = _distancia_2d(puntos, 0, 17) / face_width
    mouth_width_norm = mouth_width / face_width

    # Distancia ceja-ojo para registrar cejas arriba/abajo sin inferir emoción.
    brow_left = _distancia_2d(puntos, 105, 159) / face_width
    brow_right = _distancia_2d(puntos, 334, 386) / face_width
    brow_inner_width = _distancia_2d(puntos, 107, 336) / face_width

    # Curvatura vertical de las comisuras respecto del centro de los labios.
    mouth_center_y = (puntos[13][1] + puntos[14][1]) * 0.5
    corner_shape = (
        (mouth_center_y - puntos[61][1]) +
        (mouth_center_y - puntos[291][1])
    ) * 0.5 / face_width

    # Inclinación aproximada de la cabeza usando el eje de los ojos.
    left_eye_center = (
        (puntos[33][0] + puntos[133][0]) * 0.5,
        (puntos[33][1] + puntos[133][1]) * 0.5,
    )
    right_eye_center = (
        (puntos[362][0] + puntos[263][0]) * 0.5,
        (puntos[362][1] + puntos[263][1]) * 0.5,
    )
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    # Evitamos importar math: esta razón acotada es suficiente como rasgo.
    head_tilt = max(-1.0, min(1.0, dy / max(abs(dx), 1e-6)))

    vector = [
        eye_left_open, eye_right_open,
        brow_left, brow_right, brow_inner_width,
        mouth_open, outer_mouth_open, mouth_width_norm, corner_shape,
        head_tilt,
    ]

    # Protección contra valores absurdos por detecciones parciales.
    if any((value != value) or abs(value) > 10.0 for value in vector):
        return None

    return {
        "version": 1,
        "vector": [float(value) for value in vector],
    }


def _vector_facial_desde_dato(face_data):
    if not isinstance(face_data, dict):
        return None
    vector = face_data.get("vector")
    if not isinstance(vector, list) or not vector:
        return None
    try:
        result = [float(value) for value in vector]
    except (TypeError, ValueError):
        return None
    if any((value != value) or abs(value) > 10.0 for value in result):
        return None
    return result


def _distancia_vectorial(vector_a, vector_b):
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return None
    squared = sum((a - b) * (a - b) for a, b in zip(vector_a, vector_b))
    return (squared / len(vector_a)) ** 0.5


def _modelos_usan_cara():
    for sample in recognition_model_samples:
        if not isinstance(sample, dict):
            continue
        if sample.get("face_vector") or sample.get("face_sequence"):
            return True
    return False


def _cara_actual_reciente():
    with lock:
        face = latest_face_features
        face_time = latest_face_time
    if face is None or face_time <= 0:
        return None
    if time.perf_counter() - face_time > FACE_LIVE_MAX_AGE:
        return None
    return {
        "version": int(face.get("version", 1)),
        "vector": list(face.get("vector", [])),
    }


def _extraer_puntos_overlay_facial(face_landmarks):
    """Extrae solo unos pocos puntos visibles para mostrar la cara sin saturar la cámara."""
    overlay = []
    try:
        total = len(face_landmarks.landmark)
        for idx in FACE_DISPLAY_INDICES:
            if 0 <= idx < total:
                lm = face_landmarks.landmark[idx]
                overlay.append((int(idx), float(lm.x), float(lm.y)))
    except Exception:
        return []
    return overlay


def _dibujar_puntos_faciales(frame):
    if not SHOW_FACE_POINTS:
        return

    with lock:
        overlay = list(latest_face_overlay_points or [])
        face_time = latest_face_time

    if not overlay or face_time <= 0:
        return
    if time.perf_counter() - face_time > FACE_LIVE_MAX_AGE:
        return

    alto, ancho = frame.shape[:2]
    point_map = {}
    for idx, x, y in overlay:
        px = int(round(x * ancho))
        py = int(round(y * alto))
        if 0 <= px < ancho and 0 <= py < alto:
            point_map[int(idx)] = (px, py)

    if not point_map:
        return

    for a, b in FACE_DISPLAY_CONNECTIONS:
        if a in point_map and b in point_map:
            cv2.line(
                frame,
                point_map[a],
                point_map[b],
                FACE_DISPLAY_LINE_COLOR,
                1,
                lineType=cv2.LINE_AA,
            )

    for px, py in point_map.values():
        cv2.circle(
            frame,
            (px, py),
            FACE_DISPLAY_POINT_RADIUS,
            FACE_DISPLAY_POINT_COLOR,
            -1,
            lineType=cv2.LINE_AA,
        )


def _vector_reconocimiento(hands_data):
    """Convierte 1 o 2 manos a un vector comparable, independiente de posición/tamaño."""
    if not hands_data:
        return None

    prepared = []

    for hand in hands_data[:2]:
        landmarks = hand.get("landmarks") if isinstance(hand, dict) else None
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            return None

        points = []
        try:
            for lm in landmarks:
                if isinstance(lm, dict):
                    points.append((float(lm["x"]), float(lm["y"]), float(lm["z"])))
                else:
                    points.append((float(lm.x), float(lm.y), float(lm.z)))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

        wx, wy, wz = points[0]
        centered = [(x - wx, y - wy, z - wz) for x, y, z in points]

        # Normalizamos el tamaño para que acercarse o alejarse de la cámara
        # cambie lo menos posible el reconocimiento.
        scale = max(
            ((x * x + y * y + z * z) ** 0.5 for x, y, z in centered),
            default=0.0,
        )
        if scale < 1e-6:
            return None

        vector = []
        for x, y, z in centered:
            vector.extend((x / scale, y / scale, z / scale))

        handedness = str(hand.get("handedness", "Unknown")) if isinstance(hand, dict) else "Unknown"
        handedness_order = 0 if handedness == "Left" else 1 if handedness == "Right" else 2
        prepared.append(((handedness_order, wx), vector))

    prepared.sort(key=lambda item: item[0])

    final_vector = []
    for _, vector in prepared:
        final_vector.extend(vector)

    return {
        "hand_count": len(prepared),
        "vector": final_vector,
    }


def _ordenar_manos_movimiento(hands_data):
    """Devuelve manos ordenadas de forma estable con sus 21 puntos numéricos."""
    prepared = []
    for index, hand in enumerate(hands_data or []):
        if not isinstance(hand, dict):
            continue
        landmarks = hand.get("landmarks")
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            continue
        try:
            points = [
                (float(lm["x"]), float(lm["y"]), float(lm["z"]))
                if isinstance(lm, dict)
                else (float(lm.x), float(lm.y), float(lm.z))
                for lm in landmarks
            ]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

        handedness = str(hand.get("handedness", "Unknown"))
        order = 0 if handedness == "Left" else 1 if handedness == "Right" else 2
        prepared.append(((order, points[0][0], index), handedness, points))

    prepared.sort(key=lambda item: item[0])
    return [(handedness, points) for _, handedness, points in prepared[:2]]


def _energia_movimiento(sequence):
    """Movimiento medio entre frames de una secuencia ya normalizada."""
    if not sequence or len(sequence) < 2:
        return 0.0
    total = 0.0
    pares = 0
    for anterior, actual in zip(sequence, sequence[1:]):
        if len(anterior) != len(actual) or not actual:
            continue
        squared = sum((a - b) * (a - b) for a, b in zip(actual, anterior))
        total += (squared / len(actual)) ** 0.5
        pares += 1
    return total / max(1, pares)


def _vectorizar_secuencia_movimiento(frames, target_steps=DYNAMIC_SEQUENCE_STEPS):
    """
    Convierte una secuencia de landmarks a un número fijo de pasos.

    A diferencia del reconocedor estático, el origen queda fijado en la muñeca
    del PRIMER frame. Así se conserva la trayectoria de la mano en el tiempo,
    pero se elimina la posición inicial y se normaliza el tamaño de la mano.
    """
    raw_frames = []
    for frame in frames or []:
        hands_data = frame.get("hands", []) if isinstance(frame, dict) else frame
        ordered = _ordenar_manos_movimiento(hands_data)
        if ordered:
            raw_frames.append(ordered)

    if len(raw_frames) < 2:
        return None

    # Tomamos el número de manos del primer frame y descartamos pérdidas breves.
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

    if not sequence:
        return None

    # Remuestreo temporal: todas las ejecuciones terminan con el mismo número
    # de pasos aunque el usuario haga la seña algo más rápido o más lento.
    if target_steps and len(sequence) != target_steps:
        if target_steps <= 1:
            sequence = [sequence[-1]]
        else:
            last = len(sequence) - 1
            indices = [round(i * last / (target_steps - 1)) for i in range(target_steps)]
            sequence = [sequence[index] for index in indices]

    return {
        "hand_count": hand_count,
        "sequence": sequence,
        "motion": _energia_movimiento(sequence),
    }


def _vectorizar_secuencia_facial(frames, target_steps=DYNAMIC_SEQUENCE_STEPS):
    """Extrae y remuestrea la secuencia facial presente en un clip dinámico."""
    sequence = []
    for frame in frames or []:
        if not isinstance(frame, dict):
            continue
        vector = _vector_facial_desde_dato(frame.get("face"))
        if vector is not None:
            sequence.append(vector)

    if len(sequence) < 2:
        return None

    dims = len(sequence[0])
    sequence = [item for item in sequence if len(item) == dims]
    if len(sequence) < 2:
        return None

    if target_steps and len(sequence) != target_steps:
        if target_steps <= 1:
            sequence = [sequence[-1]]
        else:
            last = len(sequence) - 1
            indices = [round(i * last / (target_steps - 1)) for i in range(target_steps)]
            sequence = [sequence[index] for index in indices]

    return sequence


def _preparar_muestras_reconocimiento(data):
    """Valida JSON antiguos y añade cara solo cuando esa muestra la contiene."""
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        return []

    prepared = []
    for sample in data["samples"]:
        if not isinstance(sample, dict):
            continue

        label = str(sample.get("label", "")).strip().upper()
        if not label:
            continue

        # Formato nuevo para una seña dinámica.
        if str(sample.get("type", "")).lower() == "dynamic" or isinstance(sample.get("frames"), list):
            feature = _vectorizar_secuencia_movimiento(sample.get("frames", []))
            if feature is None:
                continue
            item = {
                "label": label,
                "kind": "dynamic",
                "hand_count": feature["hand_count"],
                "sequence": feature["sequence"],
                "motion": feature["motion"],
            }
            face_sequence = _vectorizar_secuencia_facial(sample.get("frames", []))
            if face_sequence is not None:
                item["face_sequence"] = face_sequence
            prepared.append(item)
            continue

        # Formato original: una pose de un único frame.
        feature = _vector_reconocimiento(sample.get("hands", []))
        if feature is None:
            continue
        item = {
            "label": label,
            "kind": "static",
            "hand_count": feature["hand_count"],
            "vector": feature["vector"],
        }
        face_vector = _vector_facial_desde_dato(sample.get("face"))
        if face_vector is not None:
            item["face_vector"] = face_vector
        prepared.append(item)

    return prepared


def reconocer_sena_dinamica(frames):
    """Reconoce trayectorias comparando ventanas recientes con secuencias entrenadas."""
    dynamic_samples = [
        sample for sample in recognition_model_samples
        if isinstance(sample, dict) and sample.get("kind") == "dynamic"
    ]
    if not dynamic_samples or not frames:
        return None, 0.0

    # Probamos varias longitudes recientes para tolerar distintas velocidades.
    total_frames = len(frames)
    candidate_lengths = []
    for length in (12, 18, 24, 32, total_frames):
        length = min(total_frames, length)
        if length >= DYNAMIC_MIN_FRAMES and length not in candidate_lengths:
            candidate_lengths.append(length)

    best_by_label = {}
    for length in candidate_lengths:
        feature = _vectorizar_secuencia_movimiento(list(frames)[-length:])
        if feature is None or feature["motion"] < DYNAMIC_MIN_MOTION:
            continue

        current_sequence = feature["sequence"]
        current_motion = feature["motion"]
        current_face_sequence = _vectorizar_secuencia_facial(list(frames)[-length:])

        for sample in dynamic_samples:
            if sample.get("hand_count") != feature["hand_count"]:
                continue
            reference_sequence = sample.get("sequence") or []
            if len(reference_sequence) != len(current_sequence):
                continue

            squared = 0.0
            dims = 0
            for current_frame, reference_frame in zip(current_sequence, reference_sequence):
                if len(current_frame) != len(reference_frame):
                    squared = None
                    break
                for a, b in zip(current_frame, reference_frame):
                    diff = a - b
                    squared += diff * diff
                    dims += 1
            if squared is None or dims == 0:
                continue

            distance = (squared / dims) ** 0.5

            # Penaliza secuencias con una cantidad de movimiento muy diferente.
            reference_motion = max(1e-6, float(sample.get("motion", 0.0)))
            motion_ratio = min(current_motion, reference_motion) / max(current_motion, reference_motion)
            adjusted_distance = distance + (1.0 - motion_ratio) * 0.12

            # La cara es complementaria: solo participa si ESA muestra fue
            # entrenada con rostro y el rostro actual está disponible.
            reference_face_sequence = sample.get("face_sequence")
            if current_face_sequence and reference_face_sequence:
                if len(current_face_sequence) == len(reference_face_sequence):
                    face_squared = 0.0
                    face_dims = 0
                    face_valid = True
                    for current_face, reference_face in zip(
                        current_face_sequence, reference_face_sequence
                    ):
                        if len(current_face) != len(reference_face):
                            face_valid = False
                            break
                        for a, b in zip(current_face, reference_face):
                            diff = a - b
                            face_squared += diff * diff
                            face_dims += 1
                    if face_valid and face_dims:
                        face_distance = (face_squared / face_dims) ** 0.5
                        face_normalized = min(1.5, face_distance / FACE_MAX_DISTANCE)
                        hand_normalized = min(1.5, adjusted_distance / DYNAMIC_MAX_DISTANCE)
                        combined = (
                            hand_normalized * (1.0 - FACE_DYNAMIC_WEIGHT)
                            + face_normalized * FACE_DYNAMIC_WEIGHT
                        )
                        adjusted_distance = combined * DYNAMIC_MAX_DISTANCE

            label = sample["label"]
            previous = best_by_label.get(label)
            if previous is None or adjusted_distance < previous:
                best_by_label[label] = adjusted_distance

    if not best_by_label:
        return None, 0.0

    ordered = sorted((distance, label) for label, distance in best_by_label.items())
    nearest_distance, winner = ordered[0]
    second_distance = ordered[1][0] if len(ordered) > 1 else DYNAMIC_MAX_DISTANCE

    similarity = max(0.0, min(1.0, 1.0 - nearest_distance / DYNAMIC_MAX_DISTANCE))
    separation = max(0.0, min(1.0, (second_distance - nearest_distance) / max(0.10, second_distance)))
    confidence = (similarity * 0.82 + separation * 0.18) * 100.0

    if nearest_distance > DYNAMIC_MAX_DISTANCE or confidence < 58.0:
        return None, confidence

    return winner, min(99.0, confidence)

def _sanitizar_nombre_modelo(nombre):
    """Convierte el nombre visible de una seña en un nombre de archivo seguro."""
    nombre = str(nombre or "").strip().upper()
    seguro = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_"
        for ch in nombre
    )
    seguro = seguro.strip("._-")
    while "__" in seguro:
        seguro = seguro.replace("__", "_")
    return seguro or "SENA"


def _ruta_modelo_local(nombre):
    return LOCAL_TRAINED_MODELS_DIR / f"{_sanitizar_nombre_modelo(nombre)}.json"


def _reconstruir_muestras_reconocimiento():
    """Une modelos externos + modelos locales sin tocar el hilo de cámara."""
    global recognition_model_samples
    recognition_model_samples = list(recognition_external_samples) + list(recognition_local_samples)
    return recognition_model_samples


def cargar_modelos_locales_entrenados(mostrar_estado=False):
    """Carga todos los JSON de modelos_entrenados/ como un solo modelo en memoria."""
    global recognition_local_samples
    global recognition_local_files

    LOCAL_TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    preparados = []
    archivos_validos = []

    for ruta in sorted(LOCAL_TRAINED_MODELS_DIR.glob("*.json")):
        try:
            datos = _leer_json_modelo(ruta)
            muestras = _preparar_muestras_reconocimiento(datos)
            if not muestras:
                continue
            preparados.extend(muestras)
            archivos_validos.append(ruta)
        except Exception:
            # Un archivo dañado no impide cargar el resto de señas.
            continue

    recognition_local_samples = preparados
    recognition_local_files = archivos_validos
    _reconstruir_muestras_reconocimiento()

    if mostrar_estado:
        labels = sorted({item["label"] for item in recognition_local_samples})
        try:
            set_status(
                f"Modelos locales: {len(labels)} seña(s), "
                f"{len(recognition_local_samples)} muestra(s)."
            )
        except Exception:
            pass

    return archivos_validos, preparados


def _activar_modelo_online(datos_modelo, ruta_modelo, version=None):
    """Activa un modelo externo y lo combina con todos los modelos locales."""
    global loaded_recognition_model_path
    global loaded_recognition_model_data
    global recognition_external_samples
    global model_online_version

    prepared = _preparar_muestras_reconocimiento(datos_modelo)
    if not prepared:
        return False

    loaded_recognition_model_data = datos_modelo
    loaded_recognition_model_path = str(ruta_modelo)
    recognition_external_samples = prepared
    _reconstruir_muestras_reconocimiento()

    if version:
        model_online_version = str(version)

    return True


def _leer_json_modelo(ruta):
    with Path(ruta).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _leer_manifest_local_modelo():
    if not MODEL_CACHE_MANIFEST.exists():
        return {}
    try:
        data = _leer_json_modelo(MODEL_CACHE_MANIFEST)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _url_archivo_modelo(nombre_archivo):
    # Path(...).name evita que un nombre del manifest pueda salir de /modelos/.
    nombre_seguro = Path(str(nombre_archivo)).name or "modelo_senas.json"
    return (
        f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/"
        f"{MODEL_GITHUB_BRANCH}/{MODEL_GITHUB_FOLDER}/{nombre_seguro}"
    )


def _descargar_modelo_bytes(url, timeout=10):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ManosQueHablan-ModelSync/1.0",
            "Accept": "application/json, application/octet-stream;q=0.9, */*;q=0.1",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _guardar_modelo_atomico(ruta, contenido):
    """Evita dejar un JSON a medias si se corta una descarga."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".part")

    with temporal.open("wb") as fh:
        fh.write(contenido)

    temporal.replace(ruta)


def sincronizar_modelo_github():
    """
    Carga el último modelo guardado y luego revisa GitHub en un hilo aparte.

    IMPORTANTE: esta función jamás se ejecuta dentro de process_frames().
    Por eso Internet no añade latencia al seguimiento de la mano.
    """
    global model_sync_in_progress

    if model_sync_in_progress:
        return

    model_sync_in_progress = True

    def worker():
        global model_sync_in_progress

        # ------------------------------------------------------
        # 1. CARGAR CACHÉ LOCAL PRIMERO
        # ------------------------------------------------------
        # Esto permite reconocer aunque GitHub esté lento o no haya Internet.
        if MODEL_CACHE_FILE.exists():
            try:
                datos_cache = _leer_json_modelo(MODEL_CACHE_FILE)
                if _preparar_muestras_reconocimiento(datos_cache):
                    manifest_cache = _leer_manifest_local_modelo()
                    version_cache = str(manifest_cache.get("version", "")).strip() or None

                    def aplicar_cache(data=datos_cache, version=version_cache):
                        _activar_modelo_online(
                            data,
                            MODEL_CACHE_FILE,
                            version=version,
                        )

                    root.after(0, aplicar_cache)
            except Exception:
                # Si la caché está dañada, simplemente intentamos bajar una nueva.
                pass

        # ------------------------------------------------------
        # 2. REVISAR GITHUB EN SEGUNDO PLANO
        # ------------------------------------------------------
        try:
            manifest_raw = _descargar_modelo_bytes(MODEL_MANIFEST_URL, timeout=6)
            manifest = json.loads(manifest_raw.decode("utf-8"))

            if not isinstance(manifest, dict):
                raise ValueError("manifest.json no es válido")

            version_remota = str(manifest.get("version", "")).strip()
            archivo_remoto = str(manifest.get("file", "modelo_senas.json")).strip()
            sha256_esperado = str(manifest.get("sha256", "")).strip().lower()

            if not version_remota:
                raise ValueError("manifest.json no contiene la versión del modelo")

            manifest_local = _leer_manifest_local_modelo()
            version_local = str(manifest_local.get("version", "")).strip()

            # Solo descargamos el modelo cuando falta o cambia la versión.
            if MODEL_CACHE_FILE.exists() and version_local == version_remota:
                return

            modelo_url = str(manifest.get("url", "")).strip()
            if not modelo_url:
                modelo_url = _url_archivo_modelo(archivo_remoto)

            modelo_raw = _descargar_modelo_bytes(modelo_url, timeout=15)

            # Si en el futuro rellenas sha256 en manifest.json, también se verifica.
            if sha256_esperado:
                sha256_real = hashlib.sha256(modelo_raw).hexdigest().lower()
                if sha256_real != sha256_esperado:
                    raise ValueError("El SHA-256 del modelo descargado no coincide")

            datos_nuevos = json.loads(modelo_raw.decode("utf-8"))
            if not _preparar_muestras_reconocimiento(datos_nuevos):
                raise ValueError("El modelo descargado no contiene muestras válidas")

            # Primero validamos; recién después reemplazamos la caché anterior.
            _guardar_modelo_atomico(MODEL_CACHE_FILE, modelo_raw)
            _guardar_modelo_atomico(
                MODEL_CACHE_MANIFEST,
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )

            def aplicar_nuevo(data=datos_nuevos, version=version_remota):
                if _activar_modelo_online(
                    data,
                    MODEL_CACHE_FILE,
                    version=version,
                ):
                    try:
                        labels = sorted({item["label"] for item in recognition_model_samples})
                        set_status(
                            f"Modelo de GitHub v{version} listo · "
                            f"{len(labels)} seña(s), {len(recognition_model_samples)} muestra(s)."
                        )
                    except Exception:
                        pass

            root.after(0, aplicar_nuevo)

        except Exception:
            # Sin Internet no bloqueamos la aplicación ni la cámara.
            # Si existía caché, ya se programó su carga arriba.
            pass
        finally:
            model_sync_in_progress = False

    threading.Thread(
        target=worker,
        daemon=True,
        name="GitHubModelSync",
    ).start()


def reconocer_sena(hands_data, face_data=None):
    """Reconoce manos y usa la cara solo en modelos entrenados con ella."""
    model_samples = recognition_model_samples
    if not model_samples:
        return None, 0.0

    feature = _vector_reconocimiento(hands_data)
    if feature is None:
        return None, 0.0

    current = feature["vector"]
    hand_count = feature["hand_count"]
    current_face_vector = _vector_facial_desde_dato(face_data)
    distances = []

    for sample in model_samples:
        if sample.get("kind", "static") != "static":
            continue
        if sample["hand_count"] != hand_count:
            continue

        reference = sample["vector"]
        if len(reference) != len(current):
            continue

        # Distancia RMS entre landmarks normalizados.
        squared = 0.0
        for a, b in zip(current, reference):
            diff = a - b
            squared += diff * diff
        hand_distance = (squared / max(1, len(current))) ** 0.5
        distance = hand_distance

        reference_face_vector = sample.get("face_vector")
        if current_face_vector is not None and reference_face_vector is not None:
            face_distance = _distancia_vectorial(current_face_vector, reference_face_vector)
            if face_distance is not None:
                # Normalizamos las dos distancias antes de mezclarlas para que
                # la cara ayude, pero nunca domine a las manos.
                hand_normalized = min(1.5, hand_distance / 0.42)
                face_normalized = min(1.5, face_distance / FACE_MAX_DISTANCE)
                combined = (
                    hand_normalized * (1.0 - FACE_STATIC_WEIGHT)
                    + face_normalized * FACE_STATIC_WEIGHT
                )
                distance = combined * 0.42

        distances.append((distance, sample["label"]))

    if not distances:
        return None, 0.0

    distances.sort(key=lambda item: item[0])
    nearest = distances[: min(5, len(distances))]

    weights = {}
    total_weight = 0.0
    for distance, label in nearest:
        weight = 1.0 / (distance + 1e-6)
        weights[label] = weights.get(label, 0.0) + weight
        total_weight += weight

    winner = max(weights, key=weights.get)
    winner_distances = [distance for distance, label in nearest if label == winner]
    nearest_distance = min(winner_distances) if winner_distances else nearest[0][0]
    agreement = weights[winner] / total_weight if total_weight > 0 else 0.0

    # La confianza combina cercanía geométrica y acuerdo entre vecinos.
    similarity = max(0.0, min(1.0, 1.0 - (nearest_distance / 0.55)))
    confidence = (similarity * 0.65 + agreement * 0.35) * 100.0

    # Rechazamos poses demasiado alejadas para no inventar una traducción.
    if nearest_distance > 0.42 or confidence < 50.0:
        return None, confidence

    return winner, min(99.0, confidence)


# ==========================================================
# BUSCAR CÁMARAS
# ==========================================================

def buscar_camaras():
    """Busca cámaras en Linux y Windows sin cambiar la lógica de MediaPipe."""
    camaras = []

    for i in range(10):
        if os.name == "nt":
            # En Windows probamos DirectShow primero y MSMF como respaldo.
            # Guardamos el backend que realmente funcionó para reutilizarlo
            # cuando el usuario pulse Iniciar.
            cap_test = None
            backend_usado = None

            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                prueba = cv2.VideoCapture(i, backend)

                if prueba.isOpened():
                    ret, frame = prueba.read()

                    if ret and frame is not None:
                        cap_test = prueba
                        backend_usado = backend
                        break

                prueba.release()

            if cap_test is None:
                continue

            dispositivo = f"Cámara {i}"
            alto, ancho = frame.shape[:2]

            camaras.append({
                "id": i,
                "dispositivo": dispositivo,
                "ancho": ancho,
                "alto": alto,
                "backend": backend_usado
            })

            cap_test.release()

        else:
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
                        "alto": alto,
                        "backend": None
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
# PROCESAMIENTO FACIAL EN HILO INDEPENDIENTE
# ==========================================================

def process_face_frames():
    global latest_face_features
    global latest_face_frame_id
    global latest_face_time
    global latest_face_overlay_points
    global latest_face_overlay_frame_id

    if not FACE_MESH_AVAILABLE or face_mesh is None:
        return

    processed_id = -1
    last_run = 0.0

    while running:
        # Si ningún modelo necesita cara, el usuario no la está entrenando y
        # tampoco se pidió mostrar puntos, este hilo queda casi dormido.
        if (not SHOW_FACE_POINTS) and (not face_training_requested) and (not _modelos_usan_cara()):
            with lock:
                latest_face_features = None
                latest_face_frame_id = -1
                latest_face_time = 0.0
                latest_face_overlay_points = []
                latest_face_overlay_frame_id = -1
            time.sleep(0.05)
            continue

        now = time.perf_counter()
        remaining = FACE_PROCESS_INTERVAL - (now - last_run)
        if remaining > 0:
            time.sleep(min(remaining, 0.02))
            continue

        with lock:
            if latest_frame is None or latest_frame_id == processed_id:
                source = None
            else:
                source = latest_frame.copy()
                source_id = latest_frame_id

        if source is None:
            time.sleep(0.003)
            continue

        # Igual que manos: espejo primero para trabajar con la misma orientación.
        source = cv2.flip(source, 1)
        small = cv2.resize(
            source,
            (FACE_PROCESS_WIDTH, FACE_PROCESS_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        try:
            result = face_mesh.process(rgb)
        except Exception:
            result = None

        features = None
        overlay_points = []
        if result is not None and result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            features = _extraer_vector_facial(face_landmarks)
            overlay_points = _extraer_puntos_overlay_facial(face_landmarks)

        stamp = time.perf_counter()
        with lock:
            latest_face_features = features
            latest_face_frame_id = source_id
            latest_face_overlay_points = overlay_points
            latest_face_overlay_frame_id = source_id if overlay_points else -1
            latest_face_time = stamp if (features is not None or overlay_points) else 0.0

        processed_id = source_id
        last_run = stamp


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
    global latest_recognized_sign
    global latest_recognition_confidence
    global latest_recognition_hands_data
    global dynamic_recognition_tick
    global dynamic_last_result

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
        recognition_hands_data = []

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

                # Copia ligera de los 21 puntos para el modelo de reconocimiento.
                recognition_hands_data.append({
                    "handedness": hand_label,
                    "landmarks": [
                        {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                        for lm in hand_landmarks.landmark
                    ],
                })

                # Esta opción solo oculta/muestra los puntos; los landmarks
                # siguen entrando al reconocimiento aunque no se dibujen.
                if SHOW_HAND_POINTS:
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

        # Dibujamos solo unos pocos puntos de la cara para mantener la cámara limpia.
        _dibujar_puntos_faciales(frame)

        # ------------------------------------------------------
        # RECONOCIMIENTO HÍBRIDO: manos + movimiento + cara opcional
        # ------------------------------------------------------
        face_snapshot = _cara_actual_reciente()

        if recognition_hands_data:
            dynamic_frame = {
                "t": time.perf_counter(),
                "hands": recognition_hands_data,
            }
            if face_snapshot is not None:
                dynamic_frame["face"] = face_snapshot
            dynamic_recognition_buffer.append(dynamic_frame)
        else:
            dynamic_recognition_buffer.clear()
            dynamic_last_result = (None, 0.0)

        if recognition_enabled and recognition_hands_data and recognition_model_samples:
            static_sign, static_confidence = reconocer_sena(
                recognition_hands_data,
                face_snapshot,
            )

            dynamic_recognition_tick += 1
            if dynamic_recognition_tick >= DYNAMIC_RECOGNITION_EVERY:
                dynamic_recognition_tick = 0
                dynamic_last_result = reconocer_sena_dinamica(dynamic_recognition_buffer)

            dynamic_sign, dynamic_confidence = dynamic_last_result

            # Cuando una trayectoria dinámica encaja con suficiente seguridad,
            # tiene prioridad sobre una pose intermedia que podría parecer estática.
            if dynamic_sign and dynamic_confidence >= max(62.0, static_confidence - 4.0):
                recognized_sign = dynamic_sign
                recognition_confidence = dynamic_confidence
            else:
                recognized_sign = static_sign
                recognition_confidence = static_confidence
        else:
            recognized_sign, recognition_confidence = None, 0.0

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
            latest_recognized_sign = recognized_sign
            latest_recognition_confidence = recognition_confidence
            latest_recognition_hands_data = recognition_hands_data

        processed_id = frame_id


# ==========================================================
# INICIAR CÁMARA
# ==========================================================

def iniciar_camara():
    global cap
    global capture_thread
    global processing_thread
    global face_thread
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
    nombre_camara = camera_items[seleccion]["dispositivo"]
    backend_camara = camera_items[seleccion].get("backend")

    set_status(f"Abriendo {nombre_camara}...")

    if os.name == "nt" and backend_camara is not None:
        cap = cv2.VideoCapture(selected_camera_id, backend_camara)
    else:
        cap = cv2.VideoCapture(selected_camera_id)

    if not cap.isOpened():
        cap = None
        set_status(
            f"No se pudo abrir {nombre_camara}.",
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
        globals()["latest_face_features"] = None
        globals()["latest_face_frame_id"] = -1
        globals()["latest_face_time"] = 0.0

    camera_fps = 0.0
    mediapipe_fps = 0.0
    display_fps = 0.0

    landmark_history.clear()
    dynamic_recognition_buffer.clear()

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

    face_thread = None
    if FACE_MESH_AVAILABLE:
        face_thread = threading.Thread(
            target=process_face_frames,
            daemon=True,
            name="MediaPipeFaceProcessing",
        )

    capture_thread.start()
    processing_thread.start()
    if face_thread is not None:
        face_thread.start()

    # Iniciar permanece visible incluso con la cámara encendida.
    start_button.configure(text="▶ Iniciar")
    stop_button.configure(state="normal")

    theme = THEMES.get(current_theme_name, THEMES["Oscuro"])
    translation_value.configure(
        text="Esperando una seña...",
        fg=theme["muted"]
    )

    set_status(f"Cámara activa: {nombre_camara}")


# ==========================================================
# DETENER CÁMARA
# ==========================================================

def detener_camara():
    global cap
    global running
    global capture_thread
    global processing_thread
    global face_thread
    global latest_frame
    global latest_processed_frame
    global latest_face_features
    global latest_face_frame_id
    global latest_face_time
    global latest_face_overlay_points
    global latest_face_overlay_frame_id

    running = False

    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join(timeout=0.5)

    capture_thread = None

    if processing_thread is not None and processing_thread.is_alive():
        processing_thread.join(timeout=0.7)

    processing_thread = None

    if face_thread is not None and face_thread.is_alive():
        face_thread.join(timeout=0.5)
    face_thread = None

    if cap is not None:
        cap.release()
        cap = None

    with lock:
        latest_frame = None
        latest_face_features = None
        latest_face_frame_id = -1
        latest_face_time = 0.0
        latest_face_overlay_points = []
        latest_face_overlay_frame_id = -1

    dynamic_recognition_buffer.clear()

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
                recognized_sign = latest_recognized_sign
                recognition_confidence = latest_recognition_confidence
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

            c = THEMES.get(current_theme_name, THEMES["Oscuro"])

            if hand_count > 0:
                detection_value.configure(text="Detectando", fg=c["ok"])

                if recognition_model_samples:
                    if recognized_sign:
                        if "translation_status_value" in globals():
                            translation_status_value.configure(
                                text="Reconocida",
                                fg=c["ok"],
                            )
                        translation_value.configure(
                            text=recognized_sign,
                            fg=c["text"],
                        )
                        if "translation_confidence_value" in globals():
                            translation_confidence_value.configure(
                                text=f"{recognition_confidence:.0f}%"
                            )
                        if "actualizar_oracion_detectada" in globals():
                            actualizar_oracion_detectada(recognized_sign, confidence=recognition_confidence)
                    else:
                        if "actualizar_oracion_detectada" in globals():
                            actualizar_oracion_detectada(None)
                        if "translation_status_value" in globals():
                            translation_status_value.configure(
                                text="Sin coincidencia",
                                fg=c["muted"],
                            )
                        translation_value.configure(
                            text="Seña no reconocida",
                            fg=c["muted"],
                        )
                        if "translation_confidence_value" in globals():
                            translation_confidence_value.configure(
                                text=(
                                    f"{recognition_confidence:.0f}%"
                                    if recognition_confidence > 0
                                    else "--"
                                )
                            )
                else:
                    if "actualizar_oracion_detectada" in globals():
                        actualizar_oracion_detectada(None)
                    if "translation_status_value" in globals():
                        translation_status_value.configure(
                            text="Modelo no cargado",
                            fg=c["muted"],
                        )
                    translation_value.configure(
                        text="Carga un modelo para reconocer",
                        fg=c["muted"],
                    )
                    if "translation_confidence_value" in globals():
                        translation_confidence_value.configure(text="--")
            else:
                detection_value.configure(text="En espera", fg=c["muted"])

                if "actualizar_oracion_detectada" in globals():
                    actualizar_oracion_detectada(None, sin_manos=True)

                if "translation_status_value" in globals():
                    translation_status_value.configure(
                        text="En espera",
                        fg=c["muted"],
                    )

                translation_value.configure(
                    text="Esperando una seña...",
                    fg=c["muted"],
                )
                if "translation_confidence_value" in globals():
                    translation_confidence_value.configure(text="--")

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
        main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
        main.grid_columnconfigure(1, weight=1, uniform="")
        main.grid_columnconfigure(2, weight=0, minsize=0, uniform="")
    else:
        # En Traducir/otras vistas recuperamos la distribución normal.
        side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
        main.grid_columnconfigure(1, weight=6, uniform="main_content")
        main.grid_columnconfigure(2, weight=3, minsize=0, uniform="main_content")

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
    if face_mesh is not None:
        try:
            face_mesh.close()
        except Exception:
            pass
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
    """Busca un paquete de Windows (.zip, .exe o .msi) en la Release."""
    assets = release_data.get("assets") or []
    extensiones = (".zip", ".exe", ".msi")

    # Primero intentamos localizar claramente el paquete de Manos que Hablan para Windows.
    for asset in assets:
        name = str(asset.get("name", ""))
        lower = name.lower().replace("_", "").replace("-", "")
        es_app = "manosquehablan" in lower
        es_windows = any(tag in lower for tag in ("windows", "win64", "win32", "win"))
        if es_app and es_windows and name.lower().endswith(extensiones):
            return asset

    # Si el nombre no tiene Windows, aceptamos un instalador/paquete compatible.
    for asset in assets:
        name = str(asset.get("name", ""))
        if name.lower().endswith(extensiones):
            return asset

    return None


def buscar_actualizaciones_app():
    """Consulta la Release más reciente de GitHub sin bloquear Tkinter."""
    global latest_release_info
    global update_notification_unread

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
        global update_notification_unread

        try:
            kind, payload = result_queue.get_nowait()
        except queue.Empty:
            # Esta función sí corre en el hilo principal de Tkinter.
            root.after(100, revisar_resultado)
            return

        if kind == "error":
            _update_error(payload)
            if "_refrescar_panel_notificaciones" in globals():
                _refrescar_panel_notificaciones()
            return

        latest_release_info = payload
        latest = payload["version"]
        tag = payload["tag"]
        asset = payload["asset"]

        update_check_button.configure(state="normal", text="Buscar actualizaciones")

        if _version_tuple(latest) > _version_tuple(APP_VERSION):
            update_notification_unread = True
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
            update_notification_unread = False
            _set_update_status(f"Estás al día · v{APP_VERSION}", "ok")

        if "notification_button" in globals():
            draw_notification_icon()
        if "_refrescar_panel_notificaciones" in globals():
            _refrescar_panel_notificaciones()

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

    if "account_panel" in globals():
        account_panel.configure(bg=c["panel"], highlightbackground=c["border"])
        account_title.configure(bg=c["panel"], fg=c["text"])
        account_message.configure(bg=c["panel"], fg=c["muted"])
        account_url.configure(
            bg=c["panel"],
            fg="#11A8FF" if theme_name == "Oscuro" else "#0077C8",
        )

    if "settings_panel" in globals():
        settings_panel.configure(bg=c["panel"], highlightbackground=c["border"])
        appearance_label.configure(bg=c["panel"], fg=c["muted"])
        stabilization_label.configure(bg=c["panel"], fg=c["muted"])
        appearance_row.configure(bg=c["panel"])
        stabilization_row.configure(bg=c["panel"])
        settings_separator.configure(bg=c["border"])
        if "landmarks_separator" in globals():
            landmarks_separator.configure(bg=c["border"])
        if "landmarks_label" in globals():
            landmarks_label.configure(bg=c["panel"], fg=c["muted"])
        if "landmarks_options" in globals():
            landmarks_options.configure(bg=c["panel"])
        if "show_hand_points_check" in globals():
            show_hand_points_check.configure(
                bg=c["panel"], fg=c["text"],
                activebackground=c["panel"], activeforeground=c["text"],
                selectcolor=c["button"],
            )
        if "show_face_points_check" in globals():
            show_face_points_check.configure(
                bg=c["panel"], fg=c["text"],
                activebackground=c["panel"], activeforeground=c["text"],
                selectcolor=c["button"],
            )
        if "landmarks_help" in globals():
            landmarks_help.configure(bg=c["panel"], fg=c["muted"])
        if "training_separator" in globals():
            training_separator.configure(bg=c["border"])
        if "training_label" in globals():
            training_label.configure(bg=c["panel"], fg=c["muted"])
        if "training_actions" in globals():
            training_actions.configure(bg=c["panel"])
        if "training_button" in globals():
            training_button.configure(
                bg=c["button"], fg=c["text"],
                activebackground=c["button_active"], activeforeground=c["text"],
                highlightbackground=c["border"],
            )
        if "load_model_button" in globals():
            load_model_button.configure(
                bg=c["button"], fg=c["text"],
                activebackground=c["button_active"], activeforeground=c["text"],
                highlightbackground=c["border"],
            )
        if "model_extra_actions" in globals():
            model_extra_actions.configure(bg=c["panel"])
        if "search_models_button" in globals():
            search_models_button.configure(
                bg=c["button"], fg=c["text"],
                activebackground=c["button_active"], activeforeground=c["text"],
                highlightbackground=c["border"],
            )
        if globals().get("delete_models_button") is not None:
            delete_models_button.configure(
                bg=c["button"], fg=c["text"],
                activebackground=c["button_active"], activeforeground=c["text"],
                highlightbackground=c["border"],
            )
        if "models_pick_box" in globals():
            models_pick_box.configure(bg=c["panel_alt"], highlightbackground=c["border"])
            models_pick_header.configure(bg=c["panel_alt"])
            models_pick_title.configure(bg=c["panel_alt"], fg=c["muted"])
            models_selected_count_label.configure(bg=c["panel_alt"], fg=c["muted"])
            models_folder_value.configure(bg=c["panel_alt"], fg=c["muted"])
            models_checklist_outer.configure(bg=c["panel_alt"])
            models_checklist_canvas.configure(bg=c["panel_alt"])
            models_checklist_inner.configure(bg=c["panel_alt"])
            models_pick_actions.configure(bg=c["panel_alt"])
            for child in models_checklist_inner.winfo_children():
                try:
                    child.configure(bg=c["panel_alt"], fg=c["text"])
                except tk.TclError:
                    try:
                        child.configure(bg=c["panel_alt"])
                    except tk.TclError:
                        pass
                if isinstance(child, tk.Frame):
                    for sub in child.winfo_children():
                        try:
                            if isinstance(sub, tk.Checkbutton):
                                sub.configure(
                                    bg=c["panel_alt"], fg=c["text"],
                                    activebackground=c["panel_alt"], activeforeground=c["text"],
                                    selectcolor=c["button"],
                                )
                            else:
                                sub.configure(bg=c["panel_alt"], fg=c["text"])
                        except tk.TclError:
                            pass
            # Botones de “Modelos a usar” con contraste fuerte para que
            # se distingan claramente tanto en tema oscuro como claro.
            for b in (models_select_all_button, models_clear_all_button):
                b.configure(
                    bg=c["card"], fg=c["text"],
                    activebackground=c["button_active"], activeforeground=c["text"],
                    relief="solid", bd=1,
                    highlightbackground=c["border"],
                )
            if "remove_selected_models_button" in globals():
                remove_selected_models_button.configure(
                    bg=c["danger"], fg=c["accent_text"],
                    activebackground=c["danger"], activeforeground=c["accent_text"],
                    relief="solid", bd=1,
                    highlightbackground=c["danger"],
                )
            apply_selected_models_button.configure(
                bg=c["accent"], fg=c["accent_text"],
                activebackground=c["button_active"], activeforeground=c["accent_text"],
                disabledforeground=c["muted"],
                relief="solid", bd=1,
                highlightbackground=c["accent"],
            )
        if "updates_separator" in globals():
            updates_separator.configure(bg=c["border"])
        if "updates_label" in globals():
            updates_label.configure(bg=c["panel"], fg=c["muted"])
        if "update_version_label" in globals():
            update_version_label.configure(bg=c["panel"], fg=c["text"])
        if "update_status_label" in globals():
            # Tarjeta más clara para destacar el mensaje del gestor de actualizaciones.
            update_status_label.configure(
                bg=c["card"],
                highlightbackground=c["border"],
                highlightcolor=c["border"],
            )
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

    # Vista Texto a señas: colores de entrada/canvas y re-renderizado.
    if "texto_senas_entry" in globals():
        try:
            texto_senas_entry.configure(
                bg=c["button"],
                fg=c["text"],
                insertbackground=c["text"],
                highlightbackground=c["border"],
                highlightcolor=c["accent"],
            )
            texto_senas_result_canvas.configure(bg=c["panel_alt"])
            texto_senas_result_inner.configure(bg=c["panel_alt"])
            if sidebar_active == "Comunicar con señas":
                root.after(0, convertir_texto_a_senas)
        except tk.TclError:
            pass

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
_LOGO_MANOS_DATA = """iVBORw0KGgoAAAANSUhEUgAAACoAAAAkCAYAAAD/yagrAAAIR0lEQVR42uWYXYhd1RXHf2vvfc65853vRFOjVopWIUhroZSCyYsPxYKFJtCHaoVSbbGo1fapMBkKQrW0WsXgS61gSzvpB33p6yQg9MNKKZiCtCjGjkmcSTKf956Pvdfqw7kzmTSTqJlAKT1wOffCuXv/z/qv9V//teH/9bJxHMA7z7Lv3Ivyp4WXw4nTh3nm9XFyM8RA/usgx8dxBvLWj7hh9ifurP1xl9V//Zja1CZ79zm+CzA1TriStd3VBHoInIB5z91br802q6d0Tms2bVFfyBcA9h0iXTWgU+MEmyJMjROuhCrvGMbMqKOzpA7D4TbG+Lo07J8gMnEFq+0DJkAgICI4MTEEMYQri+RFQA0EAxHs9LPcNzTIHd2Kvx3dxosHDqAICNgHKighR4Q+GYaBGHp1ImqICHr6MId33DL2ICMjDM2f47NvLH9ahK/aOI6J9wF6tL2JUeAEQ0AEzDCzDQFdzVER9NQP+Wg+mj+oW7c12hno6fBIzIfc/dNPs0cm0BXpWee6IAHFtUBFLnji6gAFUNjic4+piJZVZjgbGQ1OhFsBuO2ShWUXgBXy9lc/om2UU18aZMNAozFXV5owHOKQ4DQfCuD5OADH19/EQGwcee3aVXjFedwGGGobK6YLgKbEEmo9FCcihjjIAzj2miFswds4wcYJZv1KmcQLmEygdzxAY5N4M4rzvPfvZnbV5GlwmSXTtOywYRXBVIQsJ2TcJoIB1erDEzB5AC8HSb//JsX+m7nubMmcHGT21PNkCKwFa0K8akCP/Z3evj22KOhOxANJzHlccHve/r5eWwxy98AQt8eKN0vHT3c/wOyp5/jS4Kj7XpNnuzfFtHT6+fikCQFx5yNqa3T0UPuSG9JROUI6fZctoAo+YCbOfKa+CGPFaPOHnbdu3cOmEeguMnv87Jenn9bHs5Hw85Ebx4jqNDjbxvTCkwtn6m5dob6QIGKxn2MKcOTIRoupn3OozaGRPtUQgklwxcjukT3s2F6p5BWj26vha4b3iuOX2UihikRrEkklMVBEcQyaa9uR9fmP2kb0wDrS9kHa9Hnq+29qyhwpYta2mH4zMjKftE5ByygUwSiyKJ7NQGqLDzATTUnMUFntTC31TvrUb0dsEt9XEJUJdFUfJvEAcvBihTgPdEV6jHPEuNoFEPCdnGa5dB3BzAkmAiIOI2k0WSHECWg0itGALwJmgqwsa6hN4mX/xUVldxLkGHEF4EqE17bs8J+GAsc5UkTM2qeS4ouMer6L1VEkeGNle4e07bFFAg5NSjZUIHjWCpKAk4Okfz3B1mKIz0jOnZbJXoLffUYkP3O/zlOl403Db+Qhfve+7smMmhTBEojDouKzDIuKLi/jRjdBVCQ2fd1p4bQt3TA1HAGSa9nwCJqZCXunf8zhoS3ZPcPbh3b5zQPQ6YDrQ0gRet1PMjt/78nny1+p577dD9C9JFAHHSzhUmPqCzAIPpi4ILowj/O5oGpSVYjDVNfopYEpiGWgAeeEaq7nl6fPWWfT4E1j14/dZINDaAiNKmqKEGP7JxPwhcm2MXahX5x+s/4z8NQlgarRQRPaNIIrMCeIM3EhIy4uU2QzpCRC06zWmWBtYPtOScSDKt2ZZWnKyMCOUYptwzEhSjJnjXozPB51ThJmYrHx1LU5mkQAgR3rU390jaFQhaZGfMSyANZYZ/OALL3bxVHiPHS70BkeJdUlsUqETqBerEm1EQYDqZNJKMYYGOqAE7RJDjO3UiaSS3R1lcvScmjmepTdiBMjdynMnU2vpoynL+/wTYyoECuoKyQMoiEn34oNuy1Snu1itZENDzI4NEJvaZGldxeR4EiNMrxzDLc5N80DGYJFFUutMRGHiWAC6s7M5+dOzE3Xi/FnGjnWRE57Je8MottneE0mLlSHsM4YcZYINDVGiXiPdAbQvCDf5i0bGxSiIgoWa+t0BiRPOTFFQifDF97UGqgVnJgg5y2gtr6XU7P5zNuLLzddHt39OLPrOrLWyNslqY/J3rDGQa7gKqg9iCBFB/U5ODVyhaSrzcBJTu4dJEV7S+A8zpuZtkooTtp9heTOzOWn3lp86ZqH+AqATRGYwTiOHen73QMH0bUg/5N6BbDEK/ML1owECSKN4WpWDIbkResB1In5VTffKq4EpF5CYo2FHFupZAckQUTU11WYObH4VlPx9ZVpYb0GcPlRpD9qXPcI/6h6vOLNRKImayqINdQV1GVbZJpstWlY/6MJqi7EElLCYhQ0QoxYTCKp1ji76MqePbbnMXpHwcnEBx9PLiymfuij2RP1stsfhgFimwKr7lrBOXDOnHNm1vcv1mobsUF8hREwaUtHvEa31CtOn2x+secRfmuTeDn44fzpRa7FDuDlCOnkM/LSro/IvUkoycjIMggF+AxzDsSdz3ZrtZTYYMsLrZUNOYbHOaKrm2L2RP3PpZ596oY5FjjUVv/GTkom+9Nmbd+YOWmveqNDQ0PZWEt9iTQVNBVWV9CsSYtYtaJfmVlZIXUZXbdXzE/XZ7pdu+fGR5nrt1vb0My0ssgh4Jpvs3y2tM/NvWfHfKJDxEkdI2WpVvZaYHUJ/e9WllBWJlEVRaVBfa3F/Cl7Z27R7rr+Wxy3SfyHycvLUr/2+FAm0L+8QHZ9IxOdjjw6vIVOm7WmLkhaOQVZdU8qzkEQB/USLC/x65muPnzzY0z38/KKJ9HLOuu1ojv9A27pDMjXQpDPS7CbhgdFxPdX6Fd+WUFV2XvaMJUae2H7w0ytfemNDHfvOwIYCJOtlwQ4+RRDxRCf0ORuT9iNCCMeKkymTeV1rdKrO7/DqdVD3SsonP/p69/0PWVYbBNhdgAAAABJRU5ErkJggg=="""
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

# Lema secundario retirado por diseño. Se conserva la variable para que
# apply_theme() siga siendo compatible sin afectar ninguna lógica.
brand_subtitle = tk.Label(
    brand_text,
    text="",
    font=("DejaVu Sans", 8, "italic"),
    anchor="w",
)

header_controls = register_theme(tk.Frame(topbar), "topbar")
header_controls.pack(side="right", padx=14, pady=5)

theme_var = tk.StringVar(value="Sistema")
stabilization_var = tk.StringVar(value="Baja")
show_hand_points_var = tk.BooleanVar(value=SHOW_HAND_POINTS)
show_face_points_var = tk.BooleanVar(value=SHOW_FACE_POINTS)
settings_panel_visible = False
account_panel_visible = False
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


def actualizar_visibilidad_landmarks():
    """Cambia solo la visualización; reconocimiento y entrenamiento siguen activos."""
    global SHOW_HAND_POINTS, SHOW_FACE_POINTS

    try:
        SHOW_HAND_POINTS = bool(show_hand_points_var.get())
    except Exception:
        SHOW_HAND_POINTS = True

    try:
        SHOW_FACE_POINTS = bool(show_face_points_var.get())
    except Exception:
        SHOW_FACE_POINTS = True


def cerrar_panel_ajustes():
    global settings_panel_visible
    settings_panel.place_forget()
    settings_panel_visible = False


def cerrar_panel_cuenta():
    """Oculta el panel de contribución del icono de Cuenta."""
    global account_panel_visible

    if "account_panel" in globals():
        account_panel.place_forget()

    account_panel_visible = False


def abrir_github_contribucion():
    """Abre el repositorio del proyecto en el navegador predeterminado."""
    try:
        webbrowser.open_new_tab(GITHUB_PROJECT_URL)
    except Exception as exc:
        messagebox.showerror(
            "No se pudo abrir GitHub",
            f"No se pudo abrir el enlace.\n\n{GITHUB_PROJECT_URL}\n\n{exc}",
        )


def toggle_panel_cuenta():
    """Muestra u oculta la tarjeta para contribuir al proyecto."""
    global account_panel_visible

    if account_panel_visible:
        cerrar_panel_cuenta()
        return

    # Evita que Cuenta y Ajustes aparezcan superpuestos.
    if globals().get("settings_panel_visible", False):
        cerrar_panel_ajustes()

    if "account_panel" not in globals():
        return

    account_panel.place(
        relx=1.0,
        x=-58,
        y=68,
        anchor="ne",
        width=370,
    )
    account_panel.lift()
    account_panel_visible = True


def abrir_vista_configuracion():
    """Muestra los ajustes como una vista del menú lateral, no como panel del engranaje."""
    global settings_panel_visible

    if globals().get("account_panel_visible", False):
        cerrar_panel_cuenta()

    if "main" not in globals() or "settings_panel" not in globals():
        return

    # Dejamos que Tk calcule primero el tamaño real del dashboard.
    root.update_idletasks()

    # El panel de configuración ocupa únicamente el área de contenido,
    # respetando el menú lateral izquierdo.
    content_x = main.winfo_x() + SIDEBAR_FIXED_WIDTH + 6
    content_y = main.winfo_y()
    content_width = max(420, main.winfo_width() - SIDEBAR_FIXED_WIDTH - 6)
    content_height = max(360, main.winfo_height())

    settings_panel.place(
        x=content_x,
        y=content_y,
        width=content_width,
        height=content_height,
    )
    settings_panel.lift()
    settings_panel_visible = True
    update_settings_controls()


def abrir_vista_cuenta():
    """Muestra el contenido de Cuenta dentro del área principal."""
    global account_panel_visible

    if globals().get("settings_panel_visible", False):
        cerrar_panel_ajustes()

    if "main" not in globals() or "account_panel" not in globals():
        return

    root.update_idletasks()

    content_x = main.winfo_x() + SIDEBAR_FIXED_WIDTH + 6
    content_y = main.winfo_y()
    content_width = max(420, main.winfo_width() - SIDEBAR_FIXED_WIDTH - 6)
    content_height = max(360, main.winfo_height())

    account_panel.place(
        x=content_x,
        y=content_y,
        width=content_width,
        height=content_height,
    )
    account_panel.lift()
    account_panel_visible = True

    # Al ser ahora una vista amplia, el texto aprovecha mejor el espacio.
    try:
        account_message.configure(wraplength=max(330, content_width - 70))
        account_url.configure(wraplength=max(330, content_width - 70))
    except tk.TclError:
        pass


def toggle_panel_ajustes():
    """Compatibilidad interna: lleva a la sección Configuración."""
    if globals().get("sidebar_active") == "Configuración":
        return
    if "set_sidebar_active" in globals():
        set_sidebar_active("Configuración")




def buscar_modelos_internet():
    """Abre en Google Drive la carpeta pública donde se distribuyen los modelos."""
    # No ocultamos Configuración: al volver del navegador el usuario
    # permanece exactamente en la misma sección de la aplicación.
    try:
        webbrowser.open_new_tab(DRIVE_MODELS_PAGE_URL)
        set_status("Abriendo modelos disponibles en Google Drive...")
    except Exception as exc:
        messagebox.showerror(
            "No se pudo abrir Google Drive",
            f"No se pudo abrir la página de modelos.\n\n{DRIVE_MODELS_PAGE_URL}\n\n{exc}",
            parent=root,
        )


def eliminar_modelos_reconocimiento():
    """Permite borrar señas locales individualmente y quitar modelos externos/caché."""
    # Configuración queda visible detrás de esta ventana modal.
    LOCAL_TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cargar_modelos_locales_entrenados()

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    win = tk.Toplevel(root)
    win.title("Eliminar modelos · Manos que Hablan")
    win.geometry("520x430")
    win.minsize(460, 360)
    win.configure(bg=c["bg"])
    win.transient(root)
    win.grab_set()

    shell = tk.Frame(
        win, bg=c["panel"], highlightthickness=1,
        highlightbackground=c["border"],
    )
    shell.pack(fill="both", expand=True, padx=16, pady=16)

    tk.Label(
        shell, text="Eliminar modelos", bg=c["panel"], fg=c["text"],
        font=("DejaVu Sans", 15, "bold"), anchor="w",
    ).pack(fill="x", padx=16, pady=(15, 3))

    tk.Label(
        shell,
        text="Selecciona una o varias señas entrenadas localmente. Cada seña corresponde a un archivo JSON independiente.",
        bg=c["panel"], fg=c["muted"], font=("DejaVu Sans", 9),
        anchor="w", justify="left", wraplength=450,
    ).pack(fill="x", padx=16, pady=(0, 10))

    list_frame = tk.Frame(shell, bg=c["panel_alt"])
    list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        list_frame,
        selectmode="extended",
        yscrollcommand=scrollbar.set,
        bg=c["button"], fg=c["text"],
        selectbackground=c["accent"], selectforeground=c["accent_text"],
        relief="flat", bd=0, highlightthickness=1,
        highlightbackground=c["border"],
        font=("DejaVu Sans", 10),
    )
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.configure(command=listbox.yview)

    rutas = list(recognition_local_files)
    for ruta in rutas:
        try:
            datos = _leer_json_modelo(ruta)
            cantidad = len(datos.get("samples", [])) if isinstance(datos, dict) else 0
        except Exception:
            cantidad = 0
        listbox.insert("end", f"{ruta.stem}   ·   {cantidad} muestras")

    if not rutas:
        listbox.insert("end", "No hay señas locales guardadas")
        listbox.configure(state="disabled")

    def borrar_seleccionados():
        indices = list(listbox.curselection())
        if not indices or not rutas:
            messagebox.showinfo(
                "Selecciona un modelo",
                "Selecciona al menos una seña de la lista.",
                parent=win,
            )
            return

        nombres = [rutas[i].stem for i in indices if i < len(rutas)]
        if not messagebox.askyesno(
            "Confirmar eliminación",
            "¿Eliminar estos modelos?\n\n" + "\n".join(f"• {n}" for n in nombres) +
            "\n\nEsta acción borra sus archivos JSON de modelos_entrenados/.",
            parent=win,
        ):
            return

        errores = []
        for i in sorted(indices, reverse=True):
            if i >= len(rutas):
                continue
            try:
                rutas[i].unlink(missing_ok=True)
            except Exception as exc:
                errores.append(f"{rutas[i].name}: {exc}")

        cargar_modelos_locales_entrenados(mostrar_estado=True)
        if errores:
            messagebox.showwarning(
                "Eliminación parcial",
                "Algunos archivos no se pudieron borrar:\n\n" + "\n".join(errores),
                parent=win,
            )
        else:
            messagebox.showinfo(
                "Modelos eliminados",
                "Los modelos seleccionados fueron eliminados.",
                parent=win,
            )
        win.destroy()

    def quitar_externos():
        global loaded_recognition_model_path
        global loaded_recognition_model_data
        global recognition_external_samples
        global model_online_version
        global latest_recognized_sign
        global latest_recognition_confidence

        hay_externo = bool(recognition_external_samples or loaded_recognition_model_path)
        hay_cache = MODEL_CACHE_FILE.exists() or MODEL_CACHE_MANIFEST.exists()
        if not hay_externo and not hay_cache:
            messagebox.showinfo(
                "Sin modelo externo",
                "No hay un modelo externo activo ni caché de GitHub para quitar.",
                parent=win,
            )
            return

        if not messagebox.askyesno(
            "Quitar modelo externo",
            "Se quitará el modelo cargado manualmente/GitHub y su caché.\n\n"
            "Tus JSON de modelos_entrenados/ NO se borrarán.",
            parent=win,
        ):
            return

        loaded_recognition_model_path = None
        loaded_recognition_model_data = None
        recognition_external_samples = []
        model_online_version = None
        latest_recognized_sign = None
        latest_recognition_confidence = 0.0
        _reconstruir_muestras_reconocimiento()

        errores = []
        for ruta_cache in (MODEL_CACHE_FILE, MODEL_CACHE_MANIFEST):
            try:
                ruta_cache.unlink(missing_ok=True)
            except Exception as exc:
                errores.append(f"{ruta_cache.name}: {exc}")

        if errores:
            messagebox.showwarning(
                "Caché parcial",
                "El modelo se quitó, pero algunos archivos no pudieron borrarse:\n\n" + "\n".join(errores),
                parent=win,
            )
        else:
            messagebox.showinfo(
                "Modelo externo quitado",
                "El modelo externo fue quitado. Los modelos locales siguen activos.",
                parent=win,
            )
        set_status("Modelo externo quitado; modelos locales conservados.")

    buttons = tk.Frame(shell, bg=c["panel"])
    buttons.pack(fill="x", padx=16, pady=(0, 15))

    tk.Button(
        buttons, text="Eliminar seleccionados", command=borrar_seleccionados,
        relief="flat", bd=0, bg=c["accent"], fg=c["accent_text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        font=("DejaVu Sans", 9, "bold"), padx=10, pady=9, cursor="hand2",
    ).pack(side="left", fill="x", expand=True, padx=(0, 5))

    tk.Button(
        buttons, text="Quitar externo / caché", command=quitar_externos,
        relief="flat", bd=0, bg=c["button"], fg=c["text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        font=("DejaVu Sans", 9, "bold"), padx=10, pady=9, cursor="hand2",
    ).pack(side="left", fill="x", expand=True, padx=(5, 0))


def _cargar_rutas_modelo_reconocimiento(rutas, mostrar_aviso=True):
    """Carga las rutas JSON elegidas y las combina con modelos_entrenados/."""
    global loaded_recognition_model_path
    global loaded_recognition_model_data
    global recognition_external_samples
    global recognition_enabled

    if not rutas:
        return

    preparados_externos = []
    muestras_raw = []
    nombres_cargados = []
    errores = []

    try:
        local_dir = LOCAL_TRAINED_MODELS_DIR.resolve()
    except Exception:
        local_dir = LOCAL_TRAINED_MODELS_DIR

    for ruta in rutas:
        ruta_modelo = Path(ruta)
        if ruta_modelo.suffix.lower() != ".json":
            errores.append(f"{ruta_modelo.name}: formato no compatible")
            continue

        try:
            datos_modelo = _leer_json_modelo(ruta_modelo)
            prepared = _preparar_muestras_reconocimiento(datos_modelo)
            if not prepared:
                errores.append(f"{ruta_modelo.name}: no contiene muestras válidas")
                continue

            # Los archivos dentro de modelos_entrenados/ ya se cargan automáticamente,
            # así evitamos duplicar su peso al seleccionarlos manualmente.
            try:
                es_local = ruta_modelo.resolve().parent == local_dir
            except Exception:
                es_local = False

            if not es_local:
                preparados_externos.extend(prepared)
                if isinstance(datos_modelo, dict):
                    muestras_raw.extend(datos_modelo.get("samples", []))
            nombres_cargados.append(ruta_modelo.name)
        except Exception as exc:
            errores.append(f"{ruta_modelo.name}: {exc}")

    cargar_modelos_locales_entrenados()

    if not nombres_cargados:
        messagebox.showerror(
            "Modelos no válidos",
            "No se pudo cargar ningún modelo válido.\n\n" + "\n".join(errores),
            parent=root,
        )
        return

    recognition_external_samples = preparados_externos
    loaded_recognition_model_data = {"version": 1, "samples": muestras_raw}
    loaded_recognition_model_path = "; ".join(nombres_cargados)
    _reconstruir_muestras_reconocimiento()

    recognition_enabled = bool(recognition_model_samples)

    labels = sorted({item["label"] for item in recognition_model_samples})
    set_status(
        f"Modelos listos: {len(labels)} seña(s), {len(recognition_model_samples)} muestra(s)."
    )

    aviso = (
        f"Modelos seleccionados: {len(nombres_cargados)}\n"
        f"Señas activas totales: {len(labels)}\n"
        f"Muestras activas totales: {len(recognition_model_samples)}"
    )
    if errores:
        aviso += "\n\nAlgunos archivos se omitieron:\n" + "\n".join(errores[:8])

    if mostrar_aviso:
        messagebox.showinfo(
            "Modelos cargados",
            aviso + "\n\nLos modelos locales y externos se combinan en memoria para traducir.",
            parent=root,
        )


# Estado del selector compacto de modelos dentro de Configuración.
config_model_folder = None
config_model_vars = {}
config_model_paths = []


def _actualizar_contador_modelos_config():
    """Actualiza el contador de modelos marcados dentro de Configuración."""
    cantidad = sum(1 for var in config_model_vars.values() if var.get())
    if "models_selected_count_label" in globals():
        texto = f"{cantidad} marcado" if cantidad == 1 else f"{cantidad} marcados"
        models_selected_count_label.configure(text=texto)

    if "apply_selected_models_button" in globals():
        # Lo mantenemos visible/activo siempre; si no hay selección, la función
        # ya muestra un aviso. Además indicamos cuántos se aplicarán.
        apply_selected_models_button.configure(
            state="normal",
            text=(f"✓  Usar marcados ({cantidad})" if cantidad else "✓  Usar marcados"),
        )

    if "remove_selected_models_button" in globals():
        remove_selected_models_button.configure(
            text=(f"−  Quitar marcados ({cantidad})" if cantidad else "−  Quitar marcados")
        )


def _mostrar_modelos_en_configuracion(carpeta):
    """Muestra los JSON de una carpeta como casillas dentro del panel Configuración."""
    global config_model_folder, config_model_vars, config_model_paths

    if "models_checklist_inner" not in globals():
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    carpeta = Path(carpeta)
    config_model_folder = carpeta

    for child in models_checklist_inner.winfo_children():
        child.destroy()

    config_model_vars = {}
    config_model_paths = []

    try:
        archivos = sorted(
            (ruta for ruta in carpeta.iterdir()
             if ruta.is_file() and ruta.suffix.lower() == ".json"),
            key=lambda ruta: ruta.name.lower(),
        )
    except Exception as exc:
        archivos = []
        messagebox.showerror(
            "Modelos",
            f"No se pudo leer la carpeta seleccionada.\n\n{exc}",
            parent=root,
        )

    if "models_folder_value" in globals():
        models_folder_value.configure(
            text=str(carpeta) if archivos else "No se encontraron archivos .json"
        )

    if not archivos:
        tk.Label(
            models_checklist_inner,
            text="No hay modelos .json en esta carpeta.",
            bg=c["panel_alt"], fg=c["muted"],
            font=("DejaVu Sans", 9),
            anchor="center",
            pady=18,
        ).pack(fill="x")
        _actualizar_contador_modelos_config()
        return

    # Si algunos de esos archivos ya estaban cargados, aparecen marcados al volver.
    cargados = {
        nombre.strip()
        for nombre in str(loaded_recognition_model_path or "").split(";")
        if nombre.strip()
    }

    for index, ruta in enumerate(archivos):
        row = tk.Frame(models_checklist_inner, bg=c["panel_alt"])
        row.pack(fill="x", padx=5, pady=(4 if index == 0 else 1, 1))

        marcado = ruta.name in cargados or str(ruta) in cargados
        var = tk.BooleanVar(value=marcado)
        config_model_vars[str(ruta)] = var
        config_model_paths.append(ruta)

        checkbox = tk.Checkbutton(
            row,
            variable=var,
            command=_actualizar_contador_modelos_config,
            bg=c["panel_alt"],
            activebackground=c["panel_alt"],
            fg=c["text"],
            activeforeground=c["text"],
            selectcolor=c["button"],
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            padx=2,
        )
        checkbox.pack(side="left", padx=(3, 5), pady=5)

        name_label = tk.Label(
            row,
            text=ruta.name,
            bg=c["panel_alt"], fg=c["text"],
            font=("DejaVu Sans", 9),
            anchor="w",
            cursor="hand2",
        )
        name_label.pack(side="left", fill="x", expand=True, pady=5)

        # También se puede pulsar el nombre completo, no solo el cuadrito.
        def alternar(event=None, v=var):
            v.set(not v.get())
            _actualizar_contador_modelos_config()

        name_label.bind("<Button-1>", alternar)
        row.bind("<Button-1>", alternar)

    _actualizar_contador_modelos_config()

    if "models_checklist_canvas" in globals():
        root.after_idle(
            lambda: models_checklist_canvas.configure(
                scrollregion=models_checklist_canvas.bbox("all")
            )
        )


def cargar_modelo_reconocimiento():
    """Elige una carpeta y muestra sus modelos dentro de Configuración."""
    inicial = config_model_folder
    if inicial is None or not Path(inicial).exists():
        downloads = Path.home() / "Downloads"
        inicial = downloads if downloads.exists() else Path.home()

    carpeta = filedialog.askdirectory(
        parent=root,
        title="Selecciona la carpeta donde están tus modelos JSON",
        initialdir=str(inicial),
        mustexist=True,
    )

    if not carpeta:
        return

    _mostrar_modelos_en_configuracion(carpeta)

    # Mantiene al usuario exactamente en Configuración.
    if not globals().get("settings_panel_visible", False):
        abrir_vista_configuracion()


def seleccionar_todos_modelos_config():
    for var in config_model_vars.values():
        var.set(True)
    _actualizar_contador_modelos_config()


def quitar_todos_modelos_config():
    for var in config_model_vars.values():
        var.set(False)
    _actualizar_contador_modelos_config()


def aplicar_modelos_marcados_config():
    """Carga únicamente los archivos marcados en el cuadrito de Configuración."""
    seleccionados = [
        ruta
        for ruta, var in config_model_vars.items()
        if var.get()
    ]

    if not seleccionados:
        messagebox.showwarning(
            "Selecciona modelos",
            "Marca al menos un archivo .json dentro de ‘Modelos a usar’.",
            parent=root,
        )
        return

    _cargar_rutas_modelo_reconocimiento(seleccionados)

    # Después del mensaje de confirmación seguimos en Configuración.
    if not globals().get("settings_panel_visible", False):
        abrir_vista_configuracion()


def quitar_modelos_marcados_config():
    """Quita del reconocimiento los modelos externos marcados sin borrar sus JSON."""
    global loaded_recognition_model_path
    global loaded_recognition_model_data
    global recognition_external_samples
    global recognition_enabled
    global latest_recognized_sign
    global latest_recognition_confidence

    marcados = [
        Path(ruta)
        for ruta, var in config_model_vars.items()
        if var.get()
    ]
    if not marcados:
        messagebox.showinfo(
            "Quitar modelos",
            "Marca en la lista los modelos que quieres quitar del reconocimiento.",
            parent=root,
        )
        return

    # Solo se desactivan modelos externos cargados manualmente.
    # Los archivos no se borran del disco y los modelos entrenados localmente
    # siguen protegidos por la lógica original de modelos_entrenados/.
    activos = {
        nombre.strip()
        for nombre in str(loaded_recognition_model_path or "").split(";")
        if nombre.strip()
    }
    marcados_nombres = {ruta.name for ruta in marcados}
    quedan_nombres = activos - marcados_nombres

    # Reconstruimos únicamente con los archivos que continúan activos y que
    # pertenecen a la carpeta visible en el selector.
    restantes = [
        ruta for ruta in config_model_paths
        if ruta.name in quedan_nombres and ruta.exists()
    ]

    if restantes:
        _cargar_rutas_modelo_reconocimiento([str(ruta) for ruta in restantes], mostrar_aviso=False)
    else:
        recognition_external_samples = []
        loaded_recognition_model_path = None
        loaded_recognition_model_data = None
        latest_recognized_sign = None
        latest_recognition_confidence = 0.0
        cargar_modelos_locales_entrenados()
        _reconstruir_muestras_reconocimiento()
        recognition_enabled = bool(recognition_model_samples)
        set_status("Modelos externos quitados; los archivos JSON se conservaron.")

    for ruta, var in config_model_vars.items():
        if Path(ruta).name in marcados_nombres:
            var.set(False)

    _actualizar_contador_modelos_config()
    _mostrar_modelos_en_configuracion(config_model_folder)

    messagebox.showinfo(
        "Modelos quitados",
        "Los modelos marcados se quitaron del reconocimiento.\n\n"
        "Sus archivos .json NO fueron eliminados de tu computadora.",
        parent=root,
    )


def abrir_ventana_entrenamiento():
    """Abre una ventana con cámara en vivo y opciones avanzadas de captura."""
    # Configuración se conserva abierta detrás de la ventana de entrenamiento.
    # Así, al cerrar Entrenar modelo, el usuario vuelve directamente a ella.

    # Evita abrir varias ventanas de entrenamiento al mismo tiempo.
    existente = globals().get("training_window")
    try:
        if existente is not None and existente.winfo_exists():
            existente.lift()
            existente.focus_force()
            return
    except tk.TclError:
        pass

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    win = tk.Toplevel(root)
    globals()["training_window"] = win
    win.title("Entrenar modelo · Manos que Hablan")
    win.geometry("1180x720")
    win.minsize(980, 640)
    win.configure(bg=c["bg"])
    win.transient(root)

    container = tk.Frame(
        win,
        bg=c["panel"],
        highlightthickness=1,
        highlightbackground=c["border"],
    )
    container.pack(fill="both", expand=True, padx=18, pady=18)

    title = tk.Label(
        container,
        text="Entrenar modelo",
        bg=c["panel"], fg=c["text"],
        font=("DejaVu Sans", 16, "bold"), anchor="w",
    )
    title.pack(fill="x", padx=20, pady=(18, 2))

    subtitle = tk.Label(
        container,
        text="Mira la cámara en vivo y registra ejemplos de cada seña con la cantidad y velocidad que prefieras.",
        bg=c["panel"], fg=c["muted"],
        font=("DejaVu Sans", 9), anchor="w", justify="left",
    )
    subtitle.pack(fill="x", padx=20, pady=(0, 10))

    # ----------------------------------------------------------
    # DISTRIBUCIÓN DE ENTRENAMIENTO
    # Controles a la izquierda + cámara grande a la derecha.
    # ----------------------------------------------------------
    workspace = tk.Frame(container, bg=c["panel"])
    workspace.pack(fill="both", expand=True, padx=20, pady=(0, 14))

    controls_panel = tk.Frame(
        workspace,
        bg=c["panel"],
        width=380,
    )
    controls_panel.pack(side="left", fill="y", padx=(0, 14))
    controls_panel.pack_propagate(False)

    # ----------------------------------------------------------
    # CÁMARA EN VIVO DE ENTRENAMIENTO
    # Reutiliza el frame YA procesado por la cámara principal.
    # No abre una segunda cámara ni un segundo MediaPipe.
    # ----------------------------------------------------------
    preview_shell = tk.Frame(
        workspace,
        bg=c["panel_alt"],
        highlightthickness=1,
        highlightbackground=c["border"],
    )
    preview_shell.pack(side="right", fill="both", expand=True)

    preview_label = tk.Label(
        preview_shell,
        text="Iniciando cámara de entrenamiento...",
        bg=c["camera_bg"],
        fg=c["muted"],
        font=("DejaVu Sans", 10),
        anchor="center",
    )
    preview_label.pack(fill="both", expand=True, padx=8, pady=(8, 0))

    preview_info_var = tk.StringVar(value="CÁMARA EN VIVO")
    preview_info = tk.Label(
        preview_shell,
        textvariable=preview_info_var,
        bg=c["panel_alt"], fg=c["muted"],
        font=("DejaVu Sans", 8, "bold"), anchor="w",
    )
    preview_info.pack(fill="x", padx=10, pady=7)

    # ----------------------------------------------------------
    # NOMBRE + CONTADOR
    # ----------------------------------------------------------
    form_row = tk.Frame(controls_panel, bg=c["panel"])
    form_row.pack(fill="x", pady=(0, 10))

    name_box = tk.Frame(form_row, bg=c["panel"])
    name_box.pack(fill="x")

    name_label = tk.Label(
        name_box,
        text="NOMBRE DE LA SEÑA",
        bg=c["panel"], fg=c["muted"],
        font=("DejaVu Sans", 8, "bold"), anchor="w",
    )
    name_label.pack(fill="x", pady=(0, 5))

    sign_name_var = tk.StringVar()
    sign_entry = tk.Entry(
        name_box,
        textvariable=sign_name_var,
        bg=c["button"], fg=c["text"], insertbackground=c["text"],
        relief="flat", bd=0, highlightthickness=1,
        highlightbackground=c["border"], highlightcolor=c["accent"],
        font=("DejaVu Sans", 11),
    )
    sign_entry.pack(fill="x", ipady=8)
    sign_entry.focus_set()

    info_frame = tk.Frame(
        form_row,
        bg=c["panel_alt"],
        highlightthickness=1,
        highlightbackground=c["border"],
    )
    info_frame.pack(fill="x", pady=(10, 0))

    samples_var = tk.StringVar(value="Muestras guardadas: 0")
    samples_label = tk.Label(
        info_frame,
        textvariable=samples_var,
        bg=c["panel_alt"], fg=c["text"],
        font=("DejaVu Sans", 9, "bold"), anchor="w",
    )
    samples_label.pack(fill="x", padx=12, pady=(9, 2))

    LOCAL_TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    file_label = tk.Label(
        info_frame,
        text="Archivo: escribe el nombre de la seña",
        bg=c["panel_alt"], fg=c["muted"],
        font=("DejaVu Sans", 8), anchor="w",
    )
    file_label.pack(fill="x", padx=12, pady=(0, 9))

    # ----------------------------------------------------------
    # OPCIONES DE CAPTURA
    # ----------------------------------------------------------
    options_shell = tk.Frame(
        controls_panel,
        bg=c["panel_alt"],
        highlightthickness=1,
        highlightbackground=c["border"],
    )
    options_shell.pack(fill="x", pady=(0, 10))

    options_title = tk.Label(
        options_shell,
        text="OPCIONES DE CAPTURA",
        bg=c["panel_alt"], fg=c["muted"],
        font=("DejaVu Sans", 8, "bold"), anchor="w",
    )
    options_title.pack(fill="x", padx=12, pady=(9, 7))

    options_row = tk.Frame(options_shell, bg=c["panel_alt"])
    options_row.pack(fill="x", padx=12, pady=(0, 11))

    cantidad_var = tk.StringVar(value="30")
    velocidad_var = tk.StringVar(value="Máxima")
    countdown_var = tk.StringVar(value="0 s")
    hand_filter_var = tk.StringVar(value="Todas")

    def crear_opcion(parent, titulo_opcion, variable, valores, pad=(0, 8)):
        box = tk.Frame(parent, bg=c["panel_alt"])
        box.pack(fill="x", pady=(0, 7))
        lbl = tk.Label(
            box,
            text=titulo_opcion,
            bg=c["panel_alt"], fg=c["muted"],
            font=("DejaVu Sans", 8, "bold"), anchor="w",
        )
        lbl.pack(fill="x", pady=(0, 4))
        combo = ttk.Combobox(
            box,
            textvariable=variable,
            values=valores,
            state="readonly",
            style="Dashboard.TCombobox",
        )
        combo.pack(fill="x")
        return combo

    cantidad_combo = crear_opcion(
        options_row, "CANTIDAD", cantidad_var,
        ("10", "30", "50", "100", "200"),
        (0, 6),
    )
    velocidad_combo = crear_opcion(
        options_row, "VELOCIDAD", velocidad_var,
        ("Máxima", "Rápida", "Normal"),
        (6, 6),
    )
    countdown_combo = crear_opcion(
        options_row, "CUENTA REGRESIVA", countdown_var,
        ("0 s", "3 s", "5 s"),
        (6, 6),
    )
    hand_filter_combo = crear_opcion(
        options_row, "MANOS", hand_filter_var,
        ("Todas", "Izquierda", "Derecha"),
        (6, 0),
    )

    include_face_var = tk.BooleanVar(value=False)

    face_option_box = tk.Frame(options_row, bg=c["panel_alt"])
    face_option_box.pack(fill="x", pady=(8, 0))
    face_check = tk.Checkbutton(
        face_option_box,
        text="Incluir gestos faciales",
        variable=include_face_var,
        bg=c["panel_alt"],
        fg=c["text"],
        activebackground=c["panel_alt"],
        activeforeground=c["text"],
        selectcolor=c["button"],
        font=("DejaVu Sans", 9, "bold"),
        anchor="w",
        cursor="hand2",
    )
    face_check.pack(fill="x")
    face_help = tk.Label(
        face_option_box,
        text=(
            "Guarda cejas, ojos, boca e inclinación como parte de la seña. "
            "No intenta adivinar emociones."
        ),
        bg=c["panel_alt"],
        fg=c["muted"],
        font=("DejaVu Sans", 8),
        anchor="w",
        justify="left",
        wraplength=330,
    )
    face_help.pack(fill="x", pady=(2, 0))

    if not FACE_MESH_AVAILABLE:
        face_check.configure(state="disabled")
        face_help.configure(
            text="Face Mesh no está disponible; el reconocimiento de manos continúa normalmente."
        )

    def actualizar_solicitud_cara(*_):
        global face_training_requested
        face_training_requested = bool(include_face_var.get()) and FACE_MESH_AVAILABLE

    include_face_var.trace_add("write", actualizar_solicitud_cara)

    status_var = tk.StringVar(
        value="Escribe el nombre de la seña, elige las opciones y mantén la mano visible."
    )
    status_label_training = tk.Label(
        controls_panel,
        textvariable=status_var,
        bg=c["panel"], fg=c["muted"],
        font=("DejaVu Sans", 9),
        anchor="w", justify="left", wraplength=360,
    )
    status_label_training.pack(fill="x", pady=(0, 4))

    progress_var = tk.StringVar(value="Listo para capturar")
    progress_label = tk.Label(
        controls_panel,
        textvariable=progress_var,
        bg=c["panel"], fg=c["text"],
        font=("DejaVu Sans", 9, "bold"),
        anchor="w",
    )
    progress_label.pack(fill="x", pady=(0, 8))

    def cargar_dataset(nombre=None):
        nombre = (nombre or sign_name_var.get()).strip().upper()
        if not nombre:
            return {"version": 1, "label": "", "samples": []}

        ruta = _ruta_modelo_local(nombre)
        if not ruta.exists():
            return {"version": 1, "label": nombre, "samples": []}

        try:
            data = _leer_json_modelo(ruta)
            if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
                return {"version": 1, "label": nombre, "samples": []}

            # Cada archivo debe representar una sola seña.
            samples = [
                item for item in data["samples"]
                if isinstance(item, dict) and str(item.get("label", "")).strip().upper() == nombre
            ]
            return {"version": int(data.get("version", 1)), "label": nombre, "samples": samples}
        except Exception:
            return {"version": 1, "label": nombre, "samples": []}

    def guardar_dataset(data, nombre=None):
        global recognition_enabled
        nombre = (nombre or sign_name_var.get()).strip().upper()
        if not nombre:
            raise ValueError("La seña no tiene nombre")

        ruta = _ruta_modelo_local(nombre)
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # Forzamos que este JSON contenga únicamente muestras de esta seña.
        samples = []
        for sample in data.get("samples", []):
            if not isinstance(sample, dict):
                continue
            sample = dict(sample)
            sample["label"] = nombre
            samples.append(sample)

        salida = {
            "version": 1,
            "label": nombre,
            "samples": samples,
        }
        temporal = ruta.with_suffix(".json.part")
        with temporal.open("w", encoding="utf-8") as fh:
            json.dump(salida, fh, ensure_ascii=False, indent=2)
        temporal.replace(ruta)

        # Refresca todos los JSON locales en memoria; la cámara no se reinicia.
        cargar_modelos_locales_entrenados()

        # Al guardar al menos una muestra desde Entrenar modelo, el usuario ya
        # ha preparado un modelo de forma explícita y habilitamos reconocimiento.
        if samples and recognition_model_samples:
            recognition_enabled = True

        return ruta

    def actualizar_contador(data=None):
        nombre = sign_name_var.get().strip().upper()
        if not nombre:
            samples_var.set("Muestras guardadas: 0")
            file_label.configure(text="Archivo: escribe el nombre de la seña")
            return

        ruta = _ruta_modelo_local(nombre)
        file_label.configure(text=f"Archivo: modelos_entrenados/{ruta.name}")
        if data is None:
            data = cargar_dataset(nombre)
        total = len(data.get("samples", []))
        samples_var.set(f"Muestras de {nombre}: {total}")

    def copiar_manos(hands_data):
        """Copia únicamente los datos necesarios de los landmarks ya procesados."""
        copied = []
        for hand in hands_data or []:
            landmarks = hand.get("landmarks", [])
            if len(landmarks) != 21:
                continue
            copied.append({
                "handedness": str(hand.get("handedness", "Unknown")),
                "landmarks": [
                    {
                        "x": float(lm["x"]),
                        "y": float(lm["y"]),
                        "z": float(lm["z"]),
                    }
                    for lm in landmarks
                ],
            })
        return copied

    def filtrar_manos(hands_snapshot, filtro):
        if filtro == "Todas":
            return hands_snapshot

        objetivo = "left" if filtro == "Izquierda" else "right"
        return [
            hand for hand in hands_snapshot
            if str(hand.get("handedness", "")).strip().lower() == objetivo
        ]

    def obtener_muestra_actual(nombre, filtro="Todas", incluir_cara=False):
        """Toma manos del último frame y, opcionalmente, el rostro más reciente."""
        with lock:
            frame_id = latest_processed_frame_id
            hands_snapshot = copiar_manos(latest_recognition_hands_data)

        hands_snapshot = filtrar_manos(hands_snapshot, filtro)

        if frame_id < 0 or not hands_snapshot:
            return None, frame_id

        sample = {
            "label": nombre,
            "timestamp": time.time(),
            "hands": hands_snapshot,
        }

        if incluir_cara:
            face_snapshot = _cara_actual_reciente()
            if face_snapshot is not None:
                sample["face"] = face_snapshot

        return sample, frame_id

    # ----------------------------------------------------------
    # PREVIEW RÁPIDO: solo redibuja cuando existe un frame nuevo.
    # ----------------------------------------------------------
    preview_state = {"last_frame_id": -1, "after_id": None}

    def convertir_preview(frame):
        if frame is None:
            return None

        h, w = frame.shape[:2]
        available_w = max(320, preview_label.winfo_width() - 4)
        available_h = max(220, preview_label.winfo_height() - 4)
        scale = min(available_w / max(1, w), available_h / max(1, h), 1.0)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        if nw != w or nh != h:
            view = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            view = frame

        ok, buffer = cv2.imencode(".ppm", view)
        if not ok:
            return None
        return tk.PhotoImage(data=buffer.tobytes())

    def actualizar_preview():
        try:
            if not win.winfo_exists():
                return
        except tk.TclError:
            return

        with lock:
            frame_id = latest_processed_frame_id
            frame = (
                None
                if latest_processed_frame is None or frame_id == preview_state["last_frame_id"]
                else latest_processed_frame.copy()
            )
            hand_count = latest_hand_count

        if frame is not None:
            preview_state["last_frame_id"] = frame_id
            photo = convertir_preview(frame)
            if photo is not None:
                preview_label.configure(image=photo, text="")
                preview_label.image = photo
            face_text = ""
            if include_face_var.get():
                face_text = "  •  rostro ✓" if _cara_actual_reciente() is not None else "  •  rostro --"
            preview_info_var.set(
                f"CÁMARA EN VIVO  •  {hand_count} mano(s){face_text}  •  {mediapipe_fps:.0f} FPS"
            )
        elif not running:
            preview_label.configure(image="", text="La cámara está detenida")
            preview_label.image = None
            preview_info_var.set("CÁMARA EN VIVO  •  cámara detenida")

        preview_state["after_id"] = win.after(15, actualizar_preview)

    # ----------------------------------------------------------
    # CAPTURA AVANZADA
    # ----------------------------------------------------------
    capture_job = {
        "running": False,
        "mode": "",
        "target": 0,
        "captured": 0,
        "last_frame_id": -1,
        "last_saved_time": 0.0,
        "min_interval": 0.0,
        "label": "",
        "hand_filter": "Todas",
        "include_face": False,
        "data": None,
        "after_id": None,
        "countdown_after_id": None,
        "motion_frames": [],
        "motion_started_at": 0.0,
        "motion_duration": 1.20,
    }

    def intervalo_seleccionado():
        # Máxima = guarda cada nuevo frame que publique MediaPipe.
        # Las otras opciones separan un poco las muestras para dar más variedad temporal.
        return {
            "Máxima": 0.0,
            "Rápida": 0.035,
            "Normal": 0.080,
        }.get(velocidad_var.get(), 0.0)

    def segundos_countdown():
        try:
            return int(countdown_var.get().split()[0])
        except (ValueError, IndexError):
            return 0

    def actualizar_texto_lote(*_):
        if not capture_job["running"]:
            try:
                capture_many.configure(text=f"Capturar {int(cantidad_var.get())} muestras")
            except Exception:
                try:
                    capture_many.configure(text="Capturar lote")
                except Exception:
                    pass

    def capturar_muestra_unica():
        if capture_job["running"]:
            status_var.set("Detén la captura actual antes de guardar una muestra individual.")
            return False

        nombre = sign_name_var.get().strip().upper()
        if not nombre:
            status_var.set("Primero escribe el nombre de la seña.")
            return False

        filtro = hand_filter_var.get()
        incluir_cara = bool(include_face_var.get())
        sample, frame_id = obtener_muestra_actual(nombre, filtro, incluir_cara=incluir_cara)
        if sample is None:
            if not running:
                status_var.set("La cámara está detenida. Iníciala para registrar la seña.")
            else:
                status_var.set(
                    f"No detecté la mano seleccionada ({filtro.lower()}) en el último frame."
                )
            return False

        if incluir_cara and "face" not in sample:
            status_var.set(
                "No detecté el rostro con claridad. Mira hacia la cámara y vuelve a capturar."
            )
            return False

        try:
            data = cargar_dataset(nombre)
            data["samples"].append(sample)
            guardar_dataset(data, nombre)
            actualizar_contador(data)
            progress_var.set("1 muestra guardada")
            status_var.set(f"Muestra de {nombre} guardada · frame {frame_id}.")
            return True
        except Exception as exc:
            status_var.set(f"No se pudo guardar la muestra: {exc}")
            return False

    def restaurar_botones():
        try:
            capture_many.configure(text=f"Capturar {int(cantidad_var.get())} muestras")
            capture_continuous.configure(text="Captura continua")
            capture_one.configure(state="normal")
            try:
                capture_motion.configure(text="🎥 Capturar seña con movimiento", state="normal")
            except (NameError, tk.TclError):
                pass
        except tk.TclError:
            pass

    def terminar_captura(cancelado=False):
        if not capture_job["running"] and not cancelado:
            return

        # Cancela una cuenta regresiva pendiente si existe.
        countdown_after_id = capture_job.get("countdown_after_id")
        if countdown_after_id:
            try:
                win.after_cancel(countdown_after_id)
            except tk.TclError:
                pass
            capture_job["countdown_after_id"] = None

        data = capture_job.get("data")
        captured = capture_job.get("captured", 0)
        nombre = capture_job.get("label", "")
        mode = capture_job.get("mode", "")
        capture_job["running"] = False

        try:
            if data is not None and captured > 0:
                guardar_dataset(data, nombre)
                actualizar_contador(data)
        except Exception as exc:
            status_var.set(f"No se pudo guardar la captura: {exc}")
            restaurar_botones()
            return

        restaurar_botones()
        if cancelado:
            status_var.set(f"Captura detenida. Se guardaron {captured} muestras de {nombre}.")
        elif mode == "motion":
            if captured:
                status_var.set(
                    f"Movimiento de {nombre} guardado. Repite la captura varias veces desde posiciones ligeramente distintas."
                )
            # Si captured == 0, capturar_paso ya dejó un mensaje explicando el problema.
        elif mode == "continuous":
            status_var.set(f"Captura continua terminada: {captured} muestras de {nombre}.")
        else:
            status_var.set(f"Serie terminada: {captured} muestras de {nombre} guardadas.")
        progress_var.set(f"Guardadas en esta serie: {captured}")

    def capturar_paso():
        if not capture_job["running"]:
            return

        # Una muestra dinámica es un CLIP completo, no una colección de
        # fotogramas independientes. Guardamos el orden temporal de los frames.
        if capture_job["mode"] == "motion":
            nombre = capture_job["label"]
            sample, frame_id = obtener_muestra_actual(
                nombre,
                capture_job["hand_filter"],
                incluir_cara=capture_job.get("include_face", False),
            )

            if frame_id != capture_job["last_frame_id"]:
                capture_job["last_frame_id"] = frame_id
                now = time.perf_counter()

                if sample is not None:
                    if capture_job["motion_started_at"] == 0.0:
                        capture_job["motion_started_at"] = now
                    elapsed_save = now - capture_job["last_saved_time"]
                    if capture_job["last_saved_time"] == 0.0 or elapsed_save >= 0.030:
                        motion_frame = {
                            "t": now - capture_job["motion_started_at"],
                            "hands": sample["hands"],
                        }
                        if sample.get("face") is not None:
                            motion_frame["face"] = sample["face"]
                        capture_job["motion_frames"].append(motion_frame)
                        capture_job["last_saved_time"] = now

                    elapsed = now - capture_job["motion_started_at"]
                    progress_var.set(
                        f"Grabando movimiento: {min(100, int(elapsed / capture_job['motion_duration'] * 100))}%"
                    )
                    status_var.set(
                        "Haz la seña completa: posición inicial → movimiento → posición final."
                    )

                    if elapsed >= capture_job["motion_duration"]:
                        frames = list(capture_job["motion_frames"])
                        feature = _vectorizar_secuencia_movimiento(frames)
                        if feature is None or len(frames) < DYNAMIC_MIN_FRAMES:
                            status_var.set(
                                "No se pudo guardar: faltaron frames. Mantén la mano visible durante todo el movimiento."
                            )
                            capture_job["captured"] = 0
                        elif feature["motion"] < DYNAMIC_MIN_MOTION:
                            status_var.set(
                                "Detecté muy poco movimiento. Repite la seña recorriendo su trayectoria completa."
                            )
                            capture_job["captured"] = 0
                        elif capture_job.get("include_face", False) and (
                            sum(1 for item in frames if item.get("face") is not None)
                            < max(2, int(len(frames) * 0.40))
                        ):
                            status_var.set(
                                "No vi el rostro durante suficiente parte del movimiento. "
                                "Mira hacia la cámara y repite la seña."
                            )
                            capture_job["captured"] = 0
                        else:
                            capture_job["data"]["samples"].append({
                                "label": nombre,
                                "timestamp": time.time(),
                                "type": "dynamic",
                                "frames": frames,
                            })
                            capture_job["captured"] = 1
                        terminar_captura(cancelado=False)
                        return
                else:
                    progress_var.set("Esperando mano para iniciar el movimiento")
                    status_var.set(
                        f"Coloca la mano seleccionada ({capture_job['hand_filter'].lower()}) frente a la cámara."
                    )

            capture_job["after_id"] = win.after(1, capturar_paso)
            return

        if capture_job["mode"] == "batch" and capture_job["captured"] >= capture_job["target"]:
            terminar_captura(cancelado=False)
            return

        nombre = capture_job["label"]
        sample, frame_id = obtener_muestra_actual(
            nombre,
            capture_job["hand_filter"],
            incluir_cara=capture_job.get("include_face", False),
        )

        # Nunca repetimos un mismo frame procesado.
        if frame_id == capture_job["last_frame_id"]:
            capture_job["after_id"] = win.after(1, capturar_paso)
            return

        capture_job["last_frame_id"] = frame_id
        now = time.perf_counter()

        if sample is not None:
            if capture_job.get("include_face", False) and "face" not in sample:
                progress_var.set("Esperando rostro visible para guardar la muestra")
                capture_job["after_id"] = win.after(5, capturar_paso)
                return

            elapsed = now - capture_job["last_saved_time"]
            if capture_job["last_saved_time"] == 0.0 or elapsed >= capture_job["min_interval"]:
                capture_job["data"]["samples"].append(sample)
                capture_job["captured"] += 1
                capture_job["last_saved_time"] = now

                if capture_job["mode"] == "batch":
                    progress_var.set(
                        f"Capturando: {capture_job['captured']}/{capture_job['target']}"
                    )
                    status_var.set(
                        f"{velocidad_var.get()} · {capture_job['hand_filter']} · "
                        f"faltan {max(0, capture_job['target'] - capture_job['captured'])}"
                    )
                else:
                    progress_var.set(
                        f"Captura continua: {capture_job['captured']} muestras"
                    )
                    status_var.set(
                        f"Capturando sin límite · {velocidad_var.get()} · "
                        "pulsa Detener cuando tengas suficientes muestras."
                    )
        else:
            progress_var.set(f"Capturadas: {capture_job['captured']} · esperando mano")
            status_var.set(
                f"Esperando la mano seleccionada: {capture_job['hand_filter'].lower()}..."
            )

        capture_job["after_id"] = win.after(1, capturar_paso)

    def comenzar_captura_real():
        if not capture_job["running"]:
            return

        capture_job["countdown_after_id"] = None
        capture_job["last_frame_id"] = -1
        capture_job["last_saved_time"] = 0.0
        capture_job["motion_frames"] = []
        capture_job["motion_started_at"] = 0.0
        if capture_job["mode"] == "motion":
            progress_var.set("Movimiento: esperando mano")
            extra = " Mantén también el rostro visible." if capture_job.get("include_face") else ""
            status_var.set(
                "Cuando aparezca la mano, tendrás ~1.2 s para realizar la seña completa." + extra
            )
        elif capture_job["mode"] == "batch":
            progress_var.set(f"Capturando: 0/{capture_job['target']}")
            status_var.set(
                "Capturando frames nuevos. Mantén la seña y haz pequeños cambios de posición."
            )
        else:
            progress_var.set("Captura continua: 0 muestras")
            status_var.set("Captura continua iniciada. Pulsa Detener cuando quieras terminar.")
        capturar_paso()

    def ejecutar_countdown(restante):
        if not capture_job["running"]:
            return
        if restante <= 0:
            progress_var.set("¡Capturando!")
            comenzar_captura_real()
            return

        progress_var.set(f"Comienza en {restante}...")
        status_var.set("Prepara la seña frente a la cámara.")
        capture_job["countdown_after_id"] = win.after(
            1000, lambda: ejecutar_countdown(restante - 1)
        )

    def preparar_captura(mode):
        if capture_job["running"]:
            terminar_captura(cancelado=True)
            return

        nombre = sign_name_var.get().strip().upper()
        if not nombre:
            status_var.set("Primero escribe el nombre de la seña.")
            return
        if not running:
            status_var.set("La cámara está detenida. Iníciala para comenzar la captura.")
            return

        try:
            target = int(cantidad_var.get()) if mode == "batch" else (1 if mode == "motion" else 0)
        except ValueError:
            target = 30 if mode == "batch" else (1 if mode == "motion" else 0)

        capture_job["running"] = True
        capture_job["mode"] = mode
        capture_job["target"] = target
        capture_job["captured"] = 0
        capture_job["last_frame_id"] = -1
        capture_job["last_saved_time"] = 0.0
        capture_job["min_interval"] = intervalo_seleccionado()
        capture_job["label"] = nombre
        capture_job["hand_filter"] = hand_filter_var.get()
        capture_job["include_face"] = bool(include_face_var.get()) and FACE_MESH_AVAILABLE
        capture_job["data"] = cargar_dataset(nombre)
        capture_job["motion_frames"] = []
        capture_job["motion_started_at"] = 0.0

        capture_one.configure(state="disabled")
        try:
            capture_motion.configure(state="disabled")
        except (NameError, tk.TclError):
            pass
        if mode == "batch":
            capture_many.configure(text="Detener captura")
            capture_continuous.configure(text="Captura continua")
        elif mode == "continuous":
            capture_many.configure(text=f"Capturar {cantidad_var.get()} muestras")
            capture_continuous.configure(text="Detener captura")
        else:  # motion
            capture_many.configure(text=f"Capturar {cantidad_var.get()} muestras")
            capture_continuous.configure(text="Captura continua")
            try:
                capture_motion.configure(text="Detener movimiento", state="normal")
            except (NameError, tk.TclError):
                pass

        countdown = segundos_countdown()
        if countdown > 0:
            ejecutar_countdown(countdown)
        else:
            comenzar_captura_real()

    def capturar_lote():
        preparar_captura("batch")

    def capturar_continuo():
        preparar_captura("continuous")

    def capturar_movimiento():
        """Graba una ejecución completa de una seña dinámica (~1.2 s)."""
        preparar_captura("motion")

    def guardar_modelo_como():
        """Exporta una copia del modelo entrenado a la ruta/nombre elegidos."""
        if capture_job["running"]:
            status_var.set("Detén la captura actual antes de usar Guardar como…")
            return

        nombre = sign_name_var.get().strip().upper()
        if not nombre:
            messagebox.showinfo(
                "Nombre de la seña",
                "Primero escribe el nombre de la seña que quieres guardar.",
                parent=win,
            )
            return

        data = cargar_dataset(nombre)
        muestras = data.get("samples", []) if isinstance(data, dict) else []
        if not muestras:
            messagebox.showinfo(
                "Sin muestras",
                "Primero captura al menos una muestra de esta seña.",
                parent=win,
            )
            return

        ruta = filedialog.asksaveasfilename(
            parent=win,
            title="Guardar modelo de reconocimiento como",
            initialfile=f"{nombre}.json",
            defaultextension=".json",
            filetypes=[
                ("Modelo de señas JSON", "*.json"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta:
            return

        destino = Path(ruta)
        if destino.suffix.lower() != ".json":
            destino = destino.with_suffix(".json")

        salida = {
            "version": int(data.get("version", 1)),
            "label": nombre,
            "samples": muestras,
        }
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            temporal = destino.with_suffix(destino.suffix + ".part")
            with temporal.open("w", encoding="utf-8") as fh:
                json.dump(salida, fh, ensure_ascii=False, indent=2)
            temporal.replace(destino)
            status_var.set(f"Modelo guardado como: {destino.name}")
            messagebox.showinfo(
                "Modelo guardado",
                f"Se guardaron {len(muestras)} muestras de {nombre}.\n\n{destino}",
                parent=win,
            )
        except Exception as exc:
            messagebox.showerror(
                "No se pudo guardar",
                f"No se pudo guardar el modelo:\n\n{exc}",
                parent=win,
            )

    # ----------------------------------------------------------
    # GUARDAR MODELO - SIEMPRE VISIBLE ARRIBA
    # Se crea aquí porque guardar_modelo_como ya está definida, pero se
    # coloca visualmente antes de "Opciones de captura".
    # ----------------------------------------------------------
    save_as_button = tk.Button(
        controls_panel,
        text="💾  GUARDAR MODELO COMO…",
        command=guardar_modelo_como,
        relief="flat", bd=0, highlightthickness=1,
        bg=c["accent"], fg=c["accent_text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        highlightbackground=c["border"],
        font=("DejaVu Sans", 10, "bold"),
        padx=12, pady=11, cursor="hand2",
    )
    save_as_button.pack(
        fill="x",
        pady=(0, 10),
        before=options_shell,
    )

    save_help = tk.Label(
        controls_panel,
        text="Elige dónde guardar el .json y qué nombre tendrá.",
        bg=c["panel"],
        fg=c["muted"],
        font=("DejaVu Sans", 8),
        anchor="w",
    )
    save_help.pack(
        fill="x",
        pady=(0, 8),
        before=options_shell,
    )

    # ----------------------------------------------------------
    # SEÑA CON MOVIMIENTO - SIEMPRE VISIBLE ARRIBA
    # Se coloca antes de las opciones para que no quede recortada
    # en pantallas de menor altura.
    # ----------------------------------------------------------
    capture_motion = tk.Button(
        controls_panel,
        text="🎥  CAPTURAR SEÑA CON MOVIMIENTO",
        command=capturar_movimiento,
        relief="flat", bd=0, highlightthickness=1,
        bg=c["accent"], fg=c["accent_text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        highlightbackground=c["border"],
        font=("DejaVu Sans", 9, "bold"),
        padx=12, pady=10, cursor="hand2",
    )
    capture_motion.pack(
        fill="x",
        pady=(0, 5),
        before=options_shell,
    )

    motion_help = tk.Label(
        controls_panel,
        text="Graba una ejecución completa (~1.2 s). Repite 8–15 veces para mejorar precisión.",
        bg=c["panel"], fg=c["muted"],
        font=("DejaVu Sans", 8),
        anchor="w", justify="left", wraplength=350,
    )
    motion_help.pack(
        fill="x",
        pady=(0, 8),
        before=options_shell,
    )

    # ----------------------------------------------------------
    # BOTONES DE CAPTURA
    # ----------------------------------------------------------
    buttons = tk.Frame(controls_panel, bg=c["panel"])
    buttons.pack(fill="x", pady=(0, 4))

    capture_one = tk.Button(
        buttons,
        text="Capturar 1 muestra",
        command=capturar_muestra_unica,
        relief="flat", bd=0, highlightthickness=1,
        bg=c["button"], fg=c["text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        highlightbackground=c["border"],
        font=("DejaVu Sans", 9, "bold"),
        padx=12, pady=9, cursor="hand2",
    )
    capture_one.pack(fill="x", pady=(0, 6))

    capture_many = tk.Button(
        buttons,
        text="Capturar 30 muestras",
        command=capturar_lote,
        relief="flat", bd=0, highlightthickness=1,
        bg=c["accent"], fg=c["accent_text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        highlightbackground=c["border"],
        font=("DejaVu Sans", 9, "bold"),
        padx=12, pady=9, cursor="hand2",
    )
    capture_many.pack(fill="x", pady=(0, 6))

    capture_continuous = tk.Button(
        buttons,
        text="Captura continua",
        command=capturar_continuo,
        relief="flat", bd=0, highlightthickness=1,
        bg=c["button"], fg=c["text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        highlightbackground=c["border"],
        font=("DejaVu Sans", 9, "bold"),
        padx=12, pady=9, cursor="hand2",
    )
    capture_continuous.pack(fill="x", pady=(0, 6))


    def cerrar_entrenamiento():
        global face_training_requested
        # Si había una serie en curso, guarda lo ya capturado antes de cerrar.
        if capture_job["running"]:
            terminar_captura(cancelado=True)
        capture_job["running"] = False
        for key in ("after_id", "countdown_after_id"):
            after_id = capture_job.get(key)
            if after_id:
                try:
                    win.after_cancel(after_id)
                except tk.TclError:
                    pass
        preview_after = preview_state.get("after_id")
        if preview_after:
            try:
                win.after_cancel(preview_after)
            except tk.TclError:
                pass
        globals()["training_window"] = None
        face_training_requested = False
        try:
            win.destroy()
        except tk.TclError:
            pass

    win.protocol("WM_DELETE_WINDOW", cerrar_entrenamiento)
    sign_name_var.trace_add("write", lambda *_: actualizar_contador())
    cantidad_var.trace_add("write", actualizar_texto_lote)
    actualizar_contador()

    # Si ya hay una cámara seleccionada, la inicia automáticamente para que
    # la ventana de entrenamiento abra directamente con imagen en vivo.
    if not running:
        try:
            if "camera_combo" in globals() and camera_combo.current() >= 0:
                iniciar_camara()
        except Exception:
            pass

    actualizar_preview()


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


# La tuerquita superior fue retirada. Los ajustes ahora viven únicamente
# en la opción "Configuración" del menú lateral.

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

    # El punto aparece únicamente cuando hay una actualización nueva pendiente.
    if globals().get("update_notification_unread", False):
        notification_button.create_oval(
            23, 5, 30, 12,
            fill=c["accent"],
            outline=c["accent_text"],
            width=1,
        )


def _refrescar_panel_notificaciones():
    """Actualiza el contenido de la ventana de notificaciones si está abierta."""
    win = globals().get("notification_window")
    try:
        if win is None or not win.winfo_exists():
            return
    except tk.TclError:
        return

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    info = globals().get("latest_release_info")

    title = globals().get("notification_title_label")
    status = globals().get("notification_status_label")
    notes_box = globals().get("notification_notes_text")
    action = globals().get("notification_action_button")

    if not info:
        if title is not None:
            title.configure(text="Actualizaciones", bg=c["panel"], fg=c["text"])
        if status is not None:
            status.configure(
                text=f"Versión instalada: v{APP_VERSION}\nTodavía no se ha comprobado GitHub.",
                bg=c["panel"], fg=c["muted"],
            )
        if notes_box is not None:
            notes_box.configure(state="normal", bg=c["button"], fg=c["muted"])
            notes_box.delete("1.0", "end")
            notes_box.insert("1.0", "Pulsa ‘Buscar ahora’ para comprobar si existe una versión nueva.")
            notes_box.configure(state="disabled")
        if action is not None:
            action.configure(text="Buscar ahora", command=buscar_actualizaciones_app, state="normal")
        return

    latest = str(info.get("version", "")).strip()
    tag = str(info.get("tag", latest or "")).strip()
    hay_nueva = bool(latest and _version_tuple(latest) > _version_tuple(APP_VERSION))
    notas = str(info.get("notes") or "").strip()
    if not notas:
        notas = "Esta versión no incluye notas de cambios publicadas en GitHub."

    if title is not None:
        title.configure(
            text=f"Nueva versión {tag}" if hay_nueva else "Manos que Hablan está actualizado",
            bg=c["panel"], fg=c["text"],
        )
    if status is not None:
        status.configure(
            text=(
                f"Tienes instalada la v{APP_VERSION} y está disponible {tag}."
                if hay_nueva
                else f"Versión instalada: v{APP_VERSION}. No hay una versión más nueva."
            ),
            bg=c["panel"], fg=c["ok"] if not hay_nueva else c["accent"],
        )
    if notes_box is not None:
        notes_box.configure(state="normal", bg=c["button"], fg=c["text"])
        notes_box.delete("1.0", "end")
        notes_box.insert("1.0", notas)
        notes_box.configure(state="disabled")

    if action is not None:
        if hay_nueva:
            if info.get("asset"):
                action.configure(text=f"Descargar {tag}", command=descargar_actualizacion_app, state="normal")
            else:
                action.configure(text="Ver versión en GitHub", command=abrir_release_actualizacion, state="normal")
        else:
            action.configure(text="Buscar de nuevo", command=buscar_actualizaciones_app, state="normal")


def abrir_notificaciones_actualizacion(event=None):
    """Muestra recordatorios de actualización y las notas de la versión."""
    global update_notification_unread

    update_notification_unread = False
    draw_notification_icon()

    existente = globals().get("notification_window")
    try:
        if existente is not None and existente.winfo_exists():
            existente.lift()
            existente.focus_force()
            _refrescar_panel_notificaciones()
            return
    except tk.TclError:
        pass

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    win = tk.Toplevel(root)
    globals()["notification_window"] = win
    win.title("Notificaciones · Manos que Hablan")
    win.geometry("560x390")
    win.minsize(500, 340)
    win.configure(bg=c["bg"])
    win.transient(root)

    shell = tk.Frame(
        win, bg=c["panel"], highlightthickness=1, highlightbackground=c["border"]
    )
    shell.pack(fill="both", expand=True, padx=14, pady=14)

    title = tk.Label(
        shell, text="Actualizaciones", bg=c["panel"], fg=c["text"],
        anchor="w", font=("DejaVu Sans", 14, "bold")
    )
    title.pack(fill="x", padx=16, pady=(15, 4))
    globals()["notification_title_label"] = title

    status = tk.Label(
        shell, text="", bg=c["panel"], fg=c["muted"], anchor="w", justify="left",
        wraplength=500, font=("DejaVu Sans", 9)
    )
    status.pack(fill="x", padx=16, pady=(0, 10))
    globals()["notification_status_label"] = status

    tk.Label(
        shell, text="CAMBIOS DE LA VERSIÓN", bg=c["panel"], fg=c["muted"],
        anchor="w", font=("DejaVu Sans", 8, "bold")
    ).pack(fill="x", padx=16, pady=(2, 5))

    notes = tk.Text(
        shell, height=10, wrap="word", relief="flat", bd=0,
        highlightthickness=1, highlightbackground=c["border"],
        bg=c["button"], fg=c["text"], insertbackground=c["text"],
        font=("DejaVu Sans", 9), padx=10, pady=9
    )
    notes.pack(fill="both", expand=True, padx=16, pady=(0, 10))
    globals()["notification_notes_text"] = notes

    action = tk.Button(
        shell, text="Buscar ahora", command=buscar_actualizaciones_app,
        relief="flat", bd=0, highlightthickness=1,
        bg=c["accent"], fg=c["accent_text"],
        activebackground=c["button_active"], activeforeground=c["text"],
        font=("DejaVu Sans", 9, "bold"), padx=10, pady=8, cursor="hand2"
    )
    action.pack(fill="x", padx=16, pady=(0, 15))
    globals()["notification_action_button"] = action

    _refrescar_panel_notificaciones()

    # Si todavía no hay información, comprobamos automáticamente.
    if latest_release_info is None:
        buscar_actualizaciones_app()


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


# El acceso de Cuenta de la barra superior también se retiró. Todo su contenido
# ahora se abre desde la nueva opción "Cuenta" del menú lateral.

# Campanita de notificaciones de actualización en la barra superior.
notification_button = tk.Canvas(
    header_controls,
    width=34, height=34, bd=0, highlightthickness=1, cursor="hand2"
)
notification_button.pack(side="left", padx=(0, 2))
notification_button.bind("<Button-1>", abrir_notificaciones_actualizacion)
notification_button.bind("<Enter>", lambda event: set_header_icon_hover("notification"))
notification_button.bind("<Leave>", lambda event: set_header_icon_hover(None))
draw_notification_icon()

# ---------------- VISTA DE CUENTA / CONTRIBUCIÓN ----------------
account_panel = register_theme(
    tk.Frame(
        root,
        highlightthickness=1,
        bd=0,
    ),
    "panel",
)

account_title = register_theme(
    tk.Label(
        account_panel,
        text="CONTRIBUYE AL PROYECTO",
        anchor="w",
        font=("DejaVu Sans", 9, "bold"),
    ),
    "text_panel",
)
account_title.pack(fill="x", padx=18, pady=(16, 8))

account_message = register_theme(
    tk.Label(
        account_panel,
        text=(
            "¿Quieres contribuir a la mejora de Manos que Hablan? "
            "Puedes apoyar el proyecto, proponer mejoras o colaborar desde GitHub."
        ),
        anchor="w",
        justify="left",
        wraplength=330,
        font=("DejaVu Sans", 9),
    ),
    "muted_panel",
)
account_message.pack(fill="x", padx=18, pady=(0, 10))

account_url = tk.Label(
    account_panel,
    text=GITHUB_PROJECT_URL,
    anchor="w",
    justify="left",
    wraplength=330,
    font=("DejaVu Sans", 8, "underline"),
    cursor="hand2",
)
account_url.pack(fill="x", padx=18, pady=(0, 12))
account_url.bind("<Button-1>", lambda event: abrir_github_contribucion())

account_open_button = register_theme(
    tk.Button(
        account_panel,
        text="Abrir GitHub",
        command=abrir_github_contribucion,
        relief="flat",
        bd=0,
        highlightthickness=0,
        font=("DejaVu Sans", 9, "bold"),
        padx=12,
        pady=8,
        cursor="hand2",
    ),
    "primary_button",
)
account_open_button.pack(fill="x", padx=18, pady=(0, 16))

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

# ---------------- VISUALIZACIÓN DE LANDMARKS ----------------
landmarks_separator = tk.Frame(settings_panel, height=1)
landmarks_separator.pack(fill="x", padx=16, pady=(0, 14))

landmarks_label = tk.Label(
    settings_panel,
    text="PUNTOS EN CÁMARA",
    anchor="w",
    font=("DejaVu Sans", 8, "bold"),
)
landmarks_label.pack(fill="x", padx=16, pady=(0, 7))

landmarks_options = tk.Frame(settings_panel)
landmarks_options.pack(fill="x", padx=16, pady=(0, 16))

show_hand_points_check = tk.Checkbutton(
    landmarks_options,
    text="Mostrar puntos de las manos",
    variable=show_hand_points_var,
    command=actualizar_visibilidad_landmarks,
    anchor="w",
    font=("DejaVu Sans", 9, "bold"),
    cursor="hand2",
    bd=0,
    highlightthickness=0,
)
show_hand_points_check.pack(fill="x", pady=(0, 5))

show_face_points_check = tk.Checkbutton(
    landmarks_options,
    text="Mostrar puntos faciales",
    variable=show_face_points_var,
    command=actualizar_visibilidad_landmarks,
    anchor="w",
    font=("DejaVu Sans", 9, "bold"),
    cursor="hand2",
    bd=0,
    highlightthickness=0,
)
show_face_points_check.pack(fill="x")

landmarks_help = tk.Label(
    settings_panel,
    text="Estas opciones solo cambian lo que ves en la cámara; no desactivan la detección.",
    anchor="w",
    justify="left",
    font=("DejaVu Sans", 8),
)
landmarks_help.pack(fill="x", padx=16, pady=(0, 14))

# ---------------- ENTRENAMIENTO DE MODELO ----------------
training_separator = tk.Frame(settings_panel, height=1)
training_separator.pack(fill="x", padx=16, pady=(0, 14))

training_label = tk.Label(
    settings_panel,
    text="GESTION DEL MODELO DE RECONOCIMIENTO",
    anchor="w",
    font=("DejaVu Sans", 8, "bold"),
)
training_label.pack(fill="x", padx=16, pady=(0, 7))

# Las 4 acciones usan una sola cuadrícula 2x2. Así todos los botones tienen
# exactamente el mismo ancho y alto aunque sus textos tengan longitudes distintas.
training_actions = tk.Frame(settings_panel)
training_actions.pack(fill="x", padx=16, pady=(0, 16))
training_actions.grid_columnconfigure(0, weight=1, uniform="model_action")
training_actions.grid_columnconfigure(1, weight=1, uniform="model_action")
training_actions.grid_rowconfigure(0, weight=1, uniform="model_action_row")
training_actions.grid_rowconfigure(1, weight=1, uniform="model_action_row")

training_button = tk.Button(
    training_actions,
    text="Entrenar modelo",
    command=abrir_ventana_entrenamiento,
    relief="flat",
    bd=0,
    highlightthickness=1,
    font=("DejaVu Sans", 9, "bold"),
    padx=8,
    pady=8,
    cursor="hand2",
)
training_button.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))

load_model_button = tk.Button(
    training_actions,
    text="Cargar modelo",
    command=cargar_modelo_reconocimiento,
    relief="flat",
    bd=0,
    highlightthickness=1,
    font=("DejaVu Sans", 9, "bold"),
    padx=8,
    pady=8,
    cursor="hand2",
)
load_model_button.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))

# Se conserva este nombre como alias por compatibilidad con apply_theme().
model_extra_actions = training_actions

search_models_button = tk.Button(
    training_actions,
    text="Buscar modelos por internet",
    command=buscar_modelos_internet,
    relief="flat",
    bd=0,
    highlightthickness=1,
    font=("DejaVu Sans", 9, "bold"),
    padx=8,
    pady=8,
    cursor="hand2",
)
search_models_button.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=(5, 0))

# El botón de quitar modelos vive ahora junto a la lista de casillas,
# porque allí es donde el usuario ve exactamente qué modelos está marcando.
delete_models_button = None

# ---------------- MODELOS A USAR · SELECTOR COMPACTO ----------------
# La selección se hace aquí mismo dentro de Configuración, sin abrir otra ventana.
models_pick_box = tk.Frame(
    settings_panel,
    highlightthickness=1,
    bd=0,
)
models_pick_box.pack(fill="x", padx=16, pady=(0, 14))

models_pick_header = tk.Frame(models_pick_box)
models_pick_header.pack(fill="x", padx=9, pady=(8, 4))

models_pick_title = tk.Label(
    models_pick_header,
    text="MODELOS A USAR",
    anchor="w",
    font=("DejaVu Sans", 8, "bold"),
)
models_pick_title.pack(side="left")

models_selected_count_label = tk.Label(
    models_pick_header,
    text="0 marcados",
    anchor="e",
    font=("DejaVu Sans", 8, "bold"),
)
models_selected_count_label.pack(side="right")

models_folder_value = tk.Label(
    models_pick_box,
    text="Pulsa ‘Cargar modelo’ para elegir una carpeta con archivos .json",
    anchor="w",
    justify="left",
    font=("DejaVu Sans", 8),
)
models_folder_value.pack(fill="x", padx=9, pady=(0, 5))

models_checklist_outer = tk.Frame(models_pick_box)
models_checklist_outer.pack(fill="x", padx=9, pady=(0, 6))

# Un poco más de alto evita que la última casilla quede cortada.
# La lista sigue usando scroll para no hacer enorme Configuración.
models_checklist_canvas = tk.Canvas(
    models_checklist_outer,
    height=106,
    bd=0,
    highlightthickness=0,
)
models_checklist_scroll = ttk.Scrollbar(
    models_checklist_outer,
    orient="vertical",
    command=models_checklist_canvas.yview,
)
models_checklist_inner = tk.Frame(models_checklist_canvas)
models_checklist_window = models_checklist_canvas.create_window(
    (0, 0), window=models_checklist_inner, anchor="nw"
)
models_checklist_canvas.configure(yscrollcommand=models_checklist_scroll.set)

models_checklist_canvas.pack(side="left", fill="x", expand=True)
models_checklist_scroll.pack(side="right", fill="y")


def _ajustar_modelos_scroll(event=None):
    models_checklist_canvas.configure(scrollregion=models_checklist_canvas.bbox("all"))


def _ajustar_modelos_ancho(event):
    models_checklist_canvas.itemconfigure(models_checklist_window, width=event.width)


models_checklist_inner.bind("<Configure>", _ajustar_modelos_scroll)
models_checklist_canvas.bind("<Configure>", _ajustar_modelos_ancho)


def _scroll_modelos_config(event):
    """Scroll cómodo dentro de la lista en Windows y Linux."""
    if getattr(event, "delta", 0):
        pasos = -1 if event.delta > 0 else 1
    elif getattr(event, "num", None) == 4:
        pasos = -1
    elif getattr(event, "num", None) == 5:
        pasos = 1
    else:
        return
    models_checklist_canvas.yview_scroll(pasos, "units")


def _activar_scroll_modelos(event=None):
    models_checklist_canvas.bind_all("<MouseWheel>", _scroll_modelos_config)
    models_checklist_canvas.bind_all("<Button-4>", _scroll_modelos_config)
    models_checklist_canvas.bind_all("<Button-5>", _scroll_modelos_config)


def _desactivar_scroll_modelos(event=None):
    models_checklist_canvas.unbind_all("<MouseWheel>")
    models_checklist_canvas.unbind_all("<Button-4>")
    models_checklist_canvas.unbind_all("<Button-5>")


models_checklist_canvas.bind("<Enter>", _activar_scroll_modelos)
models_checklist_canvas.bind("<Leave>", _desactivar_scroll_modelos)
models_checklist_inner.bind("<Enter>", _activar_scroll_modelos)
models_checklist_inner.bind("<Leave>", _desactivar_scroll_modelos)

models_empty_hint = tk.Label(
    models_checklist_inner,
    text="Aquí aparecerán los modelos para que marques ☑ los que necesitas.",
    font=("DejaVu Sans", 9),
    anchor="center",
    pady=18,
)
models_empty_hint.pack(fill="x")

models_pick_actions = tk.Frame(models_pick_box)
# Los botones van ANTES de la lista para que nunca queden escondidos debajo
# de los últimos modelos, incluso en pantallas pequeñas.
models_pick_actions.pack(fill="x", padx=9, pady=(2, 8), before=models_checklist_outer)

# Dos filas de botones: así los últimos botones nunca quedan apretados
# o cortados cuando el panel de Configuración es estrecho (Windows/Linux).
for col in range(2):
    models_pick_actions.grid_columnconfigure(col, weight=1, uniform="models_actions")

models_select_all_button = tk.Button(
    models_pick_actions,
    text="☑  Seleccionar todos",
    command=seleccionar_todos_modelos_config,
    relief="solid",
    bd=1,
    highlightthickness=0,
    font=("DejaVu Sans", 9, "bold"),
    padx=7,
    pady=8,
    cursor="hand2",
)
models_select_all_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=(0, 7))

models_clear_all_button = tk.Button(
    models_pick_actions,
    text="☐  Quitar selección",
    command=quitar_todos_modelos_config,
    relief="solid",
    bd=1,
    highlightthickness=0,
    font=("DejaVu Sans", 9, "bold"),
    padx=7,
    pady=8,
    cursor="hand2",
)
models_clear_all_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=(0, 7))

remove_selected_models_button = tk.Button(
    models_pick_actions,
    text="−  Quitar marcados",
    command=quitar_modelos_marcados_config,
    relief="solid",
    bd=1,
    highlightthickness=0,
    font=("DejaVu Sans", 9, "bold"),
    padx=7,
    pady=8,
    cursor="hand2",
)
remove_selected_models_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))

apply_selected_models_button = tk.Button(
    models_pick_actions,
    text="✓  Usar marcados",
    command=aplicar_modelos_marcados_config,
    state="normal",
    relief="solid",
    bd=1,
    highlightthickness=0,
    font=("DejaVu Sans", 9, "bold"),
    padx=7,
    pady=8,
    cursor="hand2",
)
apply_selected_models_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))

# ---------------- ACTUALIZACIONES ----------------
updates_separator = tk.Frame(settings_panel, height=1)
updates_separator.pack(fill="x", padx=16, pady=(0, 14))

updates_label = tk.Label(
    settings_panel,
    text="GESTOR DE ACTUALIZACIONES",
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
    wraplength=620,
    font=("DejaVu Sans", 8),
    padx=12,
    pady=10,
    relief="flat",
    bd=0,
    highlightthickness=1,
)
update_status_label.pack(fill="x", padx=16, pady=(4, 10))

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

# IMPORTANTE: el gestor se creó después de los modelos por organización del código,
# pero visualmente lo colocamos ANTES del entrenamiento/modelos. Así el botón
# “Buscar actualizaciones” siempre queda visible incluso en pantallas de menor altura.
for _widget in (
    updates_separator,
    updates_label,
    update_version_label,
    update_status_label,
    updates_actions,
):
    _widget.pack_forget()

updates_separator.pack(fill="x", padx=16, pady=(0, 14), before=training_separator)
updates_label.pack(fill="x", padx=16, pady=(0, 7), before=training_separator)
update_version_label.pack(fill="x", padx=16, before=training_separator)
update_status_label.pack(fill="x", padx=16, pady=(4, 10), before=training_separator)
updates_actions.pack(fill="x", padx=16, pady=(0, 16), before=training_separator)

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
        font=(FONT, 8, "bold"),
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
SIDEBAR_FIXED_WIDTH = 250
main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
main.grid_columnconfigure(1, weight=6, uniform="main_content")
main.grid_columnconfigure(2, weight=3, uniform="main_content")
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
    elif kind == "account":
        canvas.create_oval(9,5,17,13, outline=color, width=1.5)
        canvas.create_arc(5,12,21,24, start=20, extent=140, style=tk.ARC, outline=color, width=1.5)

def set_sidebar_active(name):
    global sidebar_active
    sidebar_active = name
    update_sidebar_style()

    # Configuración ahora tiene su propia vista. Al cambiar a cualquier otra
    # sección, ocultamos esa vista para recuperar el dashboard normal.
    if name != "Configuración" and globals().get("settings_panel_visible", False):
        cerrar_panel_ajustes()

    if name != "Cuenta" and globals().get("account_panel_visible", False):
        cerrar_panel_cuenta()

    # La vista Historial es únicamente visual. Los paneles reales NO se quitan
    # del grid: solo se coloca una capa vacía encima. Así sus dimensiones
    # permanecen exactamente iguales y no se achican los cuadros inferiores.
    if "history_blank_panel" in globals():
        history_blank_panel.place_forget()
    if "history_side_blank_panel" in globals():
        history_side_blank_panel.place_forget()
    if "history_combined_blank_panel" in globals():
        history_combined_blank_panel.place_forget()
    if "text_to_sign_panel" in globals():
        text_to_sign_panel.place_forget()

    # Al pulsar "Inicio", ocultamos COMPLETAMENTE el panel del lado derecho
    # y dejamos que la cámara use todo el espacio liberado.
    # No se modifica la lógica de captura ni MediaPipe.
    if name == "Inicio":
        camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        side_panel.grid_remove()

        # Quitamos el reparto uniforme mientras el panel derecho está oculto.
        # Así la columna de la cámara puede crecer de verdad.
        main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
        main.grid_columnconfigure(1, weight=1, uniform="")
        main.grid_columnconfigure(2, weight=0, minsize=0, uniform="")

        # Recalculamos el área real de video después de que Tkinter
        # haya expandido la columna central.
        root.update_idletasks()
        actualizar_dimensiones_video()
        root.after(40, actualizar_dimensiones_video)
        root.after(120, actualizar_dimensiones_video)

    elif name == "Traducir":
        camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)

        # Al volver a Traducir, restauramos el panel derecho y
        # las proporciones normales del dashboard.
        side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
        main.grid_columnconfigure(1, weight=6, uniform="main_content")
        main.grid_columnconfigure(2, weight=3, minsize=0, uniform="main_content")

        # Volvemos a calcular el tamaño visible de la cámara para que
        # regrese exactamente a su espacio normal.
        root.update_idletasks()
        actualizar_dimensiones_video()
        root.after(40, actualizar_dimensiones_video)
        root.after(120, actualizar_dimensiones_video)

    elif name in (
        "Comunicar con señas",
        "Configuración",
        "Cuenta",
    ):
        # Estas opciones nuevas se agregan al menú sin alterar la lógica existente.
        # Mientras se desarrollan sus vistas, conservan la distribución normal
        # del dashboard para que al venir desde Inicio o Historial no quede
        # ningún panel oculto accidentalmente.
        camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
        main.grid_columnconfigure(1, weight=6, uniform="main_content")
        main.grid_columnconfigure(2, weight=3, minsize=0, uniform="main_content")

        root.update_idletasks()
        actualizar_dimensiones_video()
        root.after(40, actualizar_dimensiones_video)
        root.after(120, actualizar_dimensiones_video)

        # Configuración tiene una vista propia dentro del área principal.
        # Aquí viven los mismos controles que antes aparecían desde la
        # tuerquita superior: apariencia, estabilización, modelos y actualizaciones.
        if name == "Comunicar con señas":
            abrir_vista_texto_a_senas()
        elif name == "Configuración":
            abrir_vista_configuracion()
        elif name == "Cuenta":
            abrir_vista_cuenta()

    elif name == "Historial":
        # Conservamos EXACTAMENTE los mismos paneles y el mismo grid que Traducir.
        # En vez de sustituirlos (lo que hacía recalcular alturas), ponemos una
        # capa azul vacía encima de cada uno. El layout no cambia ni un píxel.
        camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        main.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform="")
        main.grid_columnconfigure(1, weight=6, uniform="main_content")
        main.grid_columnconfigure(2, weight=3, minsize=0, uniform="main_content")

        # Un único rectángulo vacío cubre cámara + panel derecho.
        # Se usa place() sobre main para NO tocar el grid ni sus proporciones.
        root.update_idletasks()
        x = camera_panel.winfo_x()
        y = min(camera_panel.winfo_y(), side_panel.winfo_y())
        right = side_panel.winfo_x() + side_panel.winfo_width()
        bottom = max(
            camera_panel.winfo_y() + camera_panel.winfo_height(),
            side_panel.winfo_y() + side_panel.winfo_height(),
        )
        history_combined_blank_panel.place(
            x=x,
            y=y,
            width=max(1, right - x),
            height=max(1, bottom - y),
        )
        history_combined_blank_panel.lift()
        if "refrescar_historial_ui" in globals():
            refrescar_historial_ui()

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
    # (nombre interno, texto visible, icono, altura)
    ("Inicio", "Inicio", "home", 38),
    ("Traducir", "Señas a texto", "hand", 38),
    (
        "Comunicar con señas",
        "Texto a señas",
        "chat",
        38,
    ),
    ("Historial", "Historial", "history", 38),
    ("Configuración", "Configuración", "gear", 38),
    ("Cuenta", "Cuenta", "account", 38),
]

for menu_name, menu_text, menu_icon, menu_height in menu_items:
    item = tk.Frame(sidebar_nav, height=menu_height, cursor="hand2")
    item.pack(fill="x", pady=2)
    item.pack_propagate(False)

    icon_canvas = tk.Canvas(item, width=26, height=26, bd=0, highlightthickness=0, cursor="hand2")
    icon_canvas.pack(side="left", padx=(8, 7), pady=max(4, (menu_height - 26) // 2))

    item_label = tk.Label(
        item,
        text=menu_text,
        anchor="w",
        justify="left",
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

# Eslogan inferior retirado por diseño. Se conserva la variable únicamente
# por compatibilidad con apply_theme().
sidebar_slogan = tk.Label(
    sidebar_brand,
    text="",
    justify="center",
    font=("DejaVu Sans", 7),
)

# Panel cámara, colocado físicamente en la columna central
camera_panel = register_theme(
    tk.Frame(main, highlightthickness=1),
    "panel",
)
camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)

# Capas vacías de Historial. Son HIJAS de los paneles reales y solo los cubren.
# De esta forma no participan en el grid principal y nunca cambian sus tamaños.
history_blank_panel = register_theme(
    tk.Frame(camera_panel, highlightthickness=0),
    "panel",
)
history_blank_panel.place_forget()

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

# Capa vacía de Historial para el panel derecho.
# Se crea después de side_panel y no participa en el grid principal.
history_side_blank_panel = register_theme(
    tk.Frame(side_panel, highlightthickness=0),
    "panel",
)
history_side_blank_panel.place_forget()

# Capa única de Historial: cubre visualmente cámara + panel derecho como
# un solo rectángulo, sin modificar el grid ni los tamaños originales.
history_combined_blank_panel = register_theme(
    tk.Frame(main, highlightthickness=1),
    "panel",
)
history_combined_blank_panel.place_forget()

# Contenido real del Historial. La capa sigue usando place() para conservar
# exactamente el tamaño original de cámara + panel derecho.
history_header = register_theme(tk.Frame(history_combined_blank_panel), "panel")
history_header.pack(fill="x", padx=22, pady=(20, 10))

history_title = register_theme(
    tk.Label(
        history_header,
        text="Historial",
        font=(FONT, 18, "bold"),
        anchor="w",
    ),
    "text_panel",
)
history_title.pack(fill="x")

history_subtitle = register_theme(
    tk.Label(
        history_header,
        text="Palabras y oraciones creadas con el reconocimiento de señas.",
        font=(FONT, 9),
        anchor="w",
    ),
    "muted_panel",
)
history_subtitle.pack(fill="x", pady=(3, 0))

history_list_shell = register_theme(
    tk.Frame(history_combined_blank_panel, highlightthickness=1),
    "card",
)
history_list_shell.pack(fill="both", expand=True, padx=22, pady=(0, 12))

history_scroll = tk.Scrollbar(history_list_shell)
history_scroll.pack(side="right", fill="y", padx=(0, 8), pady=10)

history_listbox = tk.Listbox(
    history_list_shell,
    yscrollcommand=history_scroll.set,
    relief="flat",
    bd=0,
    highlightthickness=0,
    font=(FONT, 10),
    activestyle="none",
)
history_listbox.pack(side="left", fill="both", expand=True, padx=12, pady=10)
history_scroll.configure(command=history_listbox.yview)

history_visible_ids = []

def refrescar_historial_ui():
    if "history_listbox" not in globals():
        return
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    try:
        history_listbox.configure(
            bg=c["card"], fg=c["text"],
            selectbackground=c["accent"], selectforeground=c["accent_text"],
        )
        history_listbox.delete(0, "end")
        history_visible_ids.clear()
        for item in reversed(history_entries):
            texto = str(item.get("texto", "")).strip()
            if not texto:
                continue
            tipo = str(item.get("tipo", "Palabra")).upper()
            fecha = str(item.get("fecha", ""))
            fecha_corta = fecha[5:16] if len(fecha) >= 16 else fecha
            history_listbox.insert("end", f"{fecha_corta}   ·   {tipo}   ·   {texto}")
            history_visible_ids.append(item.get("id"))
        if not history_visible_ids:
            history_listbox.insert("end", "Todavía no hay palabras ni oraciones guardadas.")
    except tk.TclError:
        pass

def copiar_historial_seleccionado():
    seleccion = history_listbox.curselection()
    if not seleccion or not history_visible_ids:
        set_status("Selecciona un elemento del historial para copiar.")
        return
    idx = seleccion[0]
    if idx >= len(history_visible_ids):
        return
    entry_id = history_visible_ids[idx]
    item = next((x for x in history_entries if x.get("id") == entry_id), None)
    if not item:
        return
    root.clipboard_clear()
    root.clipboard_append(str(item.get("texto", "")))
    root.update()
    set_status("Texto del historial copiado.")

def borrar_historial_seleccionado():
    seleccion = history_listbox.curselection()
    if not seleccion or not history_visible_ids:
        set_status("Selecciona un elemento del historial para borrar.")
        return
    idx = seleccion[0]
    if idx >= len(history_visible_ids):
        return
    eliminar_historial(history_visible_ids[idx])
    refrescar_historial_ui()
    set_status("Elemento eliminado del historial.")

def limpiar_historial_completo():
    if not history_entries:
        set_status("El historial ya está vacío.")
        return
    if not messagebox.askyesno(
        "Limpiar historial",
        "¿Quieres borrar todas las palabras y oraciones guardadas?",
        parent=root,
    ):
        return
    history_entries.clear()
    guardar_historial_local()
    refrescar_historial_ui()
    set_status("Historial limpiado.")

history_actions = register_theme(tk.Frame(history_combined_blank_panel), "panel")
history_actions.pack(fill="x", padx=22, pady=(0, 20))
for _col in range(3):
    history_actions.grid_columnconfigure(_col, weight=1, uniform="history_action")

def _history_button(text, command, column, primary=False):
    role = "primary_button" if primary else "button"
    btn = register_theme(
        tk.Button(
            history_actions, text=text, command=command, relief="flat", bd=0,
            padx=10, pady=9, cursor="hand2", font=(FONT, 9, "bold"),
        ),
        role,
    )
    btn.grid(row=0, column=column, sticky="ew", padx=(0, 5) if column == 0 else ((5, 5) if column == 1 else (5, 0)))
    add_button_hover(btn, role)
    return btn

history_copy_button = _history_button("⧉  Copiar", copiar_historial_seleccionado, 0, True)
history_delete_button = _history_button("⌫  Borrar", borrar_historial_seleccionado, 1)
history_clear_button = _history_button("×  Limpiar historial", limpiar_historial_completo, 2)


# ==========================================================
# TEXTO A SEÑAS · USA LOS MODELOS JSON YA ENTRENADOS/CARGADOS
# ==========================================================

text_to_sign_panel = register_theme(
    tk.Frame(main, highlightthickness=1),
    "panel",
)
text_to_sign_panel.place_forget()

texto_senas_after_id = None


def _clave_texto_sena(valor):
    """Convierte texto visible a la misma forma usada por las etiquetas JSON."""
    valor = str(valor or "").strip().upper()
    resultado = []
    ultimo_sep = False
    for ch in valor:
        if ch.isalnum() or ch == "Ñ":
            resultado.append(ch)
            ultimo_sep = False
        elif ch in (" ", "_", "-"):
            if resultado and not ultimo_sep:
                resultado.append("_")
                ultimo_sep = True
    clave = "".join(resultado).strip("_")
    while "__" in clave:
        clave = clave.replace("__", "_")
    return clave


def _letra_sin_tilde(ch):
    """Para deletreo: Á -> A, É -> E, etc.; conserva Ñ."""
    if ch.upper() == "Ñ":
        return "Ñ"
    equivalencias = {
        "Á": "A", "À": "A", "Ä": "A", "Â": "A",
        "É": "E", "È": "E", "Ë": "E", "Ê": "E",
        "Í": "I", "Ì": "I", "Ï": "I", "Î": "I",
        "Ó": "O", "Ò": "O", "Ö": "O", "Ô": "O",
        "Ú": "U", "Ù": "U", "Ü": "U", "Û": "U",
    }
    return equivalencias.get(ch.upper(), ch.upper())


def _muestra_cruda_valida(sample):
    if not isinstance(sample, dict):
        return False
    manos = sample.get("hands")
    if not isinstance(manos, list) or not manos:
        return False
    for mano in manos:
        if not isinstance(mano, dict):
            continue
        puntos = mano.get("landmarks")
        if isinstance(puntos, list) and len(puntos) == 21:
            return True
    return False


def _agregar_muestras_a_biblioteca(datos, biblioteca):
    if not isinstance(datos, dict):
        return
    muestras = datos.get("samples")
    if not isinstance(muestras, list):
        return
    for sample in muestras:
        if not _muestra_cruda_valida(sample):
            continue
        label = str(sample.get("label", "")).strip().upper()
        if label and label not in biblioteca:
            biblioteca[label] = sample


def obtener_biblioteca_texto_a_senas():
    """
    Obtiene una muestra representativa de cada seña disponible.
    No modifica recognition_model_samples ni la cámara.
    """
    biblioteca = {}

    # Modelos entrenados localmente.
    try:
        LOCAL_TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for ruta in sorted(LOCAL_TRAINED_MODELS_DIR.glob("*.json")):
            try:
                _agregar_muestras_a_biblioteca(_leer_json_modelo(ruta), biblioteca)
            except Exception:
                pass
    except Exception:
        pass

    # Modelos externos cargados manualmente o desde GitHub.
    try:
        _agregar_muestras_a_biblioteca(loaded_recognition_model_data, biblioteca)
    except Exception:
        pass

    # Caché local de GitHub, si existe.
    try:
        if MODEL_CACHE_FILE.exists():
            _agregar_muestras_a_biblioteca(
                _leer_json_modelo(MODEL_CACHE_FILE),
                biblioteca,
            )
    except Exception:
        pass

    return biblioteca


def _palabras_limpias_texto_senas(texto):
    palabras = []
    actual = []
    for ch in str(texto or ""):
        if ch.isalnum() or ch in "ÑñÁÉÍÓÚÜáéíóúü":
            actual.append(ch)
        else:
            if actual:
                palabras.append("".join(actual))
                actual = []
    if actual:
        palabras.append("".join(actual))
    return palabras


def crear_plan_texto_a_senas(texto, biblioteca):
    """
    Busca primero señas de palabras/frases completas.
    Si no existen, intenta deletrear usando modelos A, B, C...
    """
    palabras = _palabras_limpias_texto_senas(texto)
    if not palabras:
        return []

    items = []
    i = 0
    max_span = 4

    while i < len(palabras):
        encontrado = None

        # Preferimos la coincidencia más larga: BUENOS_DIAS antes que BUENOS.
        for span in range(min(max_span, len(palabras) - i), 0, -1):
            grupo = palabras[i:i + span]
            clave = _clave_texto_sena(" ".join(grupo))
            if clave in biblioteca:
                encontrado = {
                    "label": clave,
                    "visible": " ".join(grupo),
                    "sample": biblioteca[clave],
                    "tipo": "seña",
                    "inicio_palabra": True,
                }
                i += span
                break

        if encontrado is not None:
            items.append(encontrado)
            continue

        # No hay modelo para la palabra: intentamos deletrearla.
        palabra = palabras[i]
        primera = True
        for ch in palabra:
            letra = _letra_sin_tilde(ch)
            if not letra.isalnum():
                continue
            if letra in biblioteca:
                items.append({
                    "label": letra,
                    "visible": letra,
                    "sample": biblioteca[letra],
                    "tipo": "letra",
                    "inicio_palabra": primera,
                })
            else:
                items.append({
                    "label": letra,
                    "visible": letra,
                    "sample": None,
                    "tipo": "faltante",
                    "inicio_palabra": primera,
                })
            primera = False
        i += 1

    return items


# ==========================================================
# IMÁGENES PARA TEXTO A SEÑAS · SIN TOCAR LOS MODELOS/RECONOCIMIENTO
# ==========================================================
# Las imágenes personalizadas se pueden colocar junto al .py:
#   imagenes_senas/HOLA.png
#   imagenes_senas/GRACIAS.jpg
#   imagenes_senas/BUENOS_DIAS.png
#   imagenes_senas/A.jpg
#
# También se intentan descargar desde la carpeta `imagenes_senas` del mismo
# repositorio GitHub de Manos que Hablan. Las letras estáticas que no tengan
# imagen propia usan como respaldo el dataset público de alfabeto LSP de
# Expo99 (CC BY-SA 4.0). Todo queda en caché local y NO interviene en cámara,
# MediaPipe, entrenamiento ni reconocimiento.
SIGN_IMAGE_PROJECT_DIR = Path(__file__).resolve().parent / "imagenes_senas"
SIGN_IMAGE_CACHE_DIR = Path.home() / ".manos_que_hablan" / "imagenes_senas"
SIGN_IMAGE_CUSTOM_BASE_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/"
    f"{MODEL_GITHUB_BRANCH}/imagenes_senas"
)
SIGN_IMAGE_LSP_DATASET_BASE_URL = (
    "https://raw.githubusercontent.com/Expo99/"
    "Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet/"
    "master"
)

# El dataset contiene los 24 gestos estáticos. J, Ñ y Z dependen de movimiento.
SIGN_IMAGE_STATIC_LSP_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")
SIGN_IMAGE_DYNAMIC_LSP_LETTERS = {"J", "Ñ", "Z"}
SIGN_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Láminas de referencia proporcionadas por el usuario.
# Se embeben dentro del .py para que la vista Texto a señas funcione
# incluso sin Internet y sin depender de archivos externos.
SIGN_IMAGE_SHEET_DIR = SIGN_IMAGE_CACHE_DIR / "_laminas_referencia"
SIGN_IMAGE_SHEET_ASSETS_DIR = SIGN_IMAGE_CACHE_DIR / "_laminas_recortadas"
texto_senas_sheet_assets_ready = False

_TEXTOSENAS_ALFABETO_SHEET_B64 = """iVBORw0KGgoAAAANSUhEUgAAAWEAAAH/CAIAAAAJ8fuzAAIj40lEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a/a5qqrrrrq+QAIrrrqqqueP4Dgqquuuur5A6hcddVV/1Vs8wJI4n8cgOCqq6666vkDqPxb2QYk8b+TbUn8W9m2LYn/nVprkkop/OtlWgIkgWmtAZL430VgJIEkEC86m9ZaiUDieRlkp3le4vmSZJv/JpJ4/gAq/yar1erChQvDMPC/kKSIKKUcO3ZssVhEBP964zju7e0tl8vM5AEk8b/BNE2zfnb8xPHFYhER/GsITdN0dHR4cHAwjqMkQBL/q9Rat7e3F4vNEuJfSQK4dOnS/sE+z8PmMtvmOUni+ZHEv5Jt/n1KKVtbW9vb25J4/gAqLxrbtiXt7e394A/+4C//8i9fvHhxHEf+F7ItCTh9+vR7vMd7vNmbvdnGxoYk/iXTNNluU/vDP/rDb/iGb9jf3z88PLTNc5LE/2y2M7PWur29/Qqv8Aof8REfcerUqYjgX2QOD4/+/C/+4qu+8quPlofDsF6vV5L4X6hE3d7ZftSjHv0xH/MxN954QynBi8Dm6PDod3/v977+67/x6OhwHIdsmU7xbAZInh9JPD+SJPGvkZn8W0WEpGmaTpw48ZIv+ZIf93Efd+rUqVIKzw1AtnnR2F6tVp/wCZ/wO7/zO9vb2y/7si/7qEc9ShL/q9iW1HXdX//1X//DP/zD/v7+K73SK33Jl3zJsWPH+JeM44j5iq/8ih/9sR9ZzDce8uCHvMzLvEzXdzyAJP43KKU85SlP+dM//dNLly498pGP/PZv//bjx49L4oU6PDz84R/6kW/79u8Y1sODHnzLy7/cy80Xs77vAdv8r/KMZ9z+13/91xcvXrzpppu+5qu/5sEPeZDEv2i1Wn31V33tT/7UT45De+yLPfalXurFNze3nOa5yGCehySeH0n869nm30TSOI7nz5//7d/+7d3d3Yc+9KHf+I3feMMNN/DcAGSbF80wDN/3fd/3lV/5la/6qq/6qZ/6qadOneq6ThL/C9kehuHcuXNf8zVf8zu/8zuf/dmf/TZv8zbZUiGJy8TzsP34xz/hPd7j3W+48bov+sIvvv76G2azWSmF5ySJ/w3Gcbx48eJP/MRPfM/3fM+bvumbfsZnfMZiseAFsLH9p3/6px/3cZ+wtbn9YR/24a/8Ki+/sbGotfK/0ziOh4dHP/3TP/Nd3/ldr/AKr/x5n/fZJ04cQ7wQrbUnPenJ7/3e79N1/ad/+qe//Mu97MbmhiQnDyQBRvyrSOI5jeP4hCc84Rd/8RdPnDhx4cKF93zP97z++uvb1EotgCT+HWxP07S/v/+bv/mbX/ZlX/YyL/Ny3/zN39jP+lDwbACVF9k4jL/zO7+zsbHxWZ/1WTfddBP/yy0Wi62trY//+I//y7/8y2//9m9/q7d6a0VIBgMgnkeb8pu+6Zttf/7nfd6jHvWoWjueH0n8bzCfzxeLxfu+7/v+5m/+5h/+4R+u1+vFYsEL5HEc//iP/vjw8PC93+u93+ANXnc2n4VA/C81n8+3t7ff+Z3e6S/+4i//9E//7E//5C9f9/Veq+uDF2y9Gn7oh354f3//0z/901/ndV6r67oI2WCemyyJ52GbF0ASzykzb7vttg/6oA+qtR4eHn7v937vR3zER2xtbUUEEBH8O9gGtra23viN3/gHfuAH/vZv/vbOO+96yENuQcGzAQQvMuPbb7/9ZV7mZW64/gb+T5B0/Pjx6667/o477t69uCtxmXkBhmH4u7/7u83Nzeuvv77WKkmSJEmSJEmSxL9Da225XP7O7/zO7/3e7z3pSU8ax5H/TLXWnZ2d13u917v33nunaeIFk5St3XvfvYv51ou/+MvMZjMA8b/dseM7j33MY6epPe1pt07TxAuVbk9+8pMi4pVf+ZX6vpMAJBDIyMjIEpJ4fiRJkiRJkiRJkiTxPCQ95jGPufbaa0+dOnX99dd3XfeXf/GXkviPIEmSpM3Nzbd6q7fJjMf9/dOkwnMACF5ktsdxPHHiBGDbNv/LSZLUdZ2Io+WSZzIvwDiOw3qYzWalhMR/BttPecpTzpw589CHPvT2229/2tOeNo4j/5kknT59urVmmxcqMzPbfLZ5bPuURAT/J2hjY1OK1pptzAvXWiul3HTTjZIkAYCEJEmSJCH+o7TWANullHd913f967/5a9v8hyqlnjp1Gmt5lCCeA0DlX0mSQpL4H8D2NE1///d//+d//ufDMFxzzTWPfexjH/GIR/R9z4tAkiQQgMEgXihBgPhPc+HChaOjo5d7uZcrpWxsbHzBF3zB53zO55QoCmVmREjiP5okSbwIBBHR91Xiv5htnsUoxH8E2xLY2QAQ/xJntq7r+S/xd3/3dy/2Yi/WWrN93bXXXX/99bfddtuDH/xg/qMZ2SklCMQzAQT/SpIk8T9Da+1P/uRPPvdzP/fWW2/d29v7gz/4gx/+4R9eLpf8a0iSAmOexTwvgwHxn+kJT3jCYx/72FKKpO3t7Ztuusn23v7ePffcc3BwME2Tbf4TSOJFYEDZsgGY/2LTNC2Xy+Vy2VqzzX8EOxGgNGD+JZJqLRL/BWzv7u4OwxARJUpEvMqrvMrf/u3fOs1/PNnmuQFU/pUk8Z/PtiQgMyVJ4jnZtn3ffff95E/+5Kd8yqc88pGPfNrTnnbs2LFf//Vfb63ZlsS/kngW8fwJgv9o0zRl5tmzZ7uuu3jxYt/3kgDgzd7szQ4PD3/3d3+3tXbNNdc8/OEPv/baa/u+l8R/HEmSeBEZMP9VbLfWDg4O/v7v//5pT3va4eGh7e3t7YjY2dl5qZd6qe3t7WPHjgkhJPGvJwCB+B9G0mq1On/+/A033DBNk9ANN9zwvd/zvW/0Rm/U9z3/FQAq/8PYHsfx/PnzgKTZbHbs2DFJPKfMPDo6+t7v/d4P/uAPfvCDH1xr/f3f//13fdd3veGGG2qt/GtIkoQEgED8CwKC/zjZ8m/+9m/+4R/+YXt7+2//9m9f7/Ver+97oJSyXq+f8YxnvOqrvuqZM2eG9fDzv/Dzr/Iqr3LzzTdL4r+HQCj5L2F7uVw+/vGP/7Vf+7Vjx4699Eu/9PHjxzNzb2/vtttue9w/PO4v/uIvWmsv8zIv8/Iv//I33HBDrZX/Q4Qe9KAH/eZv/ubrvd7rHRwc7O3tnT9//glPfMJTnvKUF3uxF+O/AkDlv09rrZTCZbYl2V4ul3/2Z3/29V//9Zubm7PZ7IYbbnjDN3zDl33Zl+37/sKFC7feemtmAovF4slPfvITn/jEa6+9ttYK7O7uHh4edl1XokjiXyHBvIgc/IeyfeHihb29vfd6r/ey/fjHP54HKKX8xE/8xCd90ifVWm0/9rGPveOOO2688caI4L+NwchY/KexDazX67/6q7/6zM/8zOuuu+793u/9NjY2Dg4OxnHc3Nx8+Zd/+dd//de3vbe39wu/8At/8id/8uZv/uaPfvSjT58+LYn/E2pXr7vuuo/8yI98whOecPz48dd5ndd58IMf/Jqv+Zq/+qu/+tjHPpb/CgCV/ya2bV+8ePG2226zfezYseuvv7619nM/93N//dd//e7v/u4v/dIvLemuu+76vd/7vdtvv/3lXu7lvu/7vu+uu+46ceJEKWVvb++P/uiPPvADP7Dve0nAW7/1W//2b/+27XRiEP8a5tkCzPNlbERAgPiPYPuv//qvX+IlXkKSpBd/8Rfv+34cx2EYSpTTp09funQpIiT1ff+gBz3oN37jN5bL5fb2Nv9tzH+JaZqe+tSn/vRP//S7vdu7LeaL8+fP33bbbefPn7948eJqtVqtVjs7O6/0Sq/0Ui/1Uu/93u+9t7f3+Mc//md+5mde/dVf/ZZbbtnY2OBFIwkJwOZ/mFLKox/96I2NjQ/5kA+59tpra63jOM7n8/d8z/d87/d+75MnT0riPxdA5b+J7TvvvPMbv/Eb77777lrrxsbGW7zFWzz1qU99ylOe8pmf+Zk7OztCiJtuvOmlX/qlP+ETPuG7v/u7r7/++i/+4i/e3t6OiIODg9d//defz+cRwWWPeMQjtre3f/3Xf721hvhXMeJFYAAjJPEf584773yDN3gDLjt+/Likixcv/sZv/EaJ8lIv/VLz+VwSACwWi2c84xmr1Wp7e5v/OLZt8yIxMiQYxH8a27fddtuP/MiPvNd7vdejH/1oLosI26219Wp9eHT45V/+5bu7uz/3cz938803P/rRj361V3u1V3j5V/jCL/rCm2666V3e5V1OnjwZEbwIBGAD5n8USV3X3XzzzefOnbvhhhts931/0003vcEbvMEv/dIvvcu7vEsphf8ABjA2zwOg8t/B9tmzZ7/jO75jY2PjC7/wCxeLxcHBwdd//df/xm/8xjd/8zcfO3YMkASoqFf/8Ic//Jd+6Zfe7/3e78SJExEBHDt2bDab1VoB28BsNrvlllvW6/WlS5eOHz/OC2ZbEvezRYoXhbERAsAg/t1s33vvvRHBZZubm7Zns9n29vbh4eGTnvSk06dPl1IAICIyk/8EtnlRmf98Fy9e/NZv/dZXf/VXf+hDH3pwcLBarS5dupSZtdYLFy4AXe3e4z3eo+u6iLjnnnu+9Eu/9J3e6Z0e+YhHLhaLf/iHf/j6r//6j/zIjzx+/HhE8EIJIYEw/wN1XfeQhzzk9ttvf4mXeAlJQCnloz/6o7/hG74hM0sp/EewbYwNBvFsAMF/FafX6/Xh4eHh4eH58+d/7Md+zPaHf/iH33DDDSdOnLjxxhs/6qM+ant7+/Tp05Ik8QDHjx+fzWaPfexjI4LLIuKN3/iNIwKQJElSRDzsYQ972tOeNo4j/wkMYEmS+LfKzMPDw7Nnz+7t7QG2b731VmBvb+++++4rpbTWdnZ23viN3/iN3uiNHv7wh6/Xa9uZuVwuh/Vw3XXXtdb4P+3o6OiHf/iHH/HwRzzykY/8q7/6q6//+q//ge//gQsXLvR9X0o5duzYX/3VX/3pn/3parXa3d39zu/8zlOnTn3Zl33ZarX6hE/8hL/+67/+qI/6qMc8+jFf/dVffXBwkJm8cEIIMOZ/nr7vP+RDPuS2224DgHEcDw8PM3O1WtnmPx1A5b/E3t7e2bNn//zP//zWW2/d2Nj4i7/4i/l8/vmf//knTpyQBJRSuq7b3Nzkedg+e/bsW73VWz3ykY+UxP3e/M3f/I//+I+ncZrNZlxm++Vf/uW/9Vu/9bGPfey1117Li07m2cy/wJD8m9gex/GXf/mXW2tPf/rTP+ETPsH2crmcpunOO++84447HvKQh9xzzz0333xzROzs7Bw7dmy9XtvOzN/4jd+w/YxnPOPs2bPXXXcd/0etV+s/+7M/A4Zx+LEf+7F3eZd3+biP/bjFYoG4IjMf+tCH7u3t/eEf/uHh4eGLv/iL33bbbY961KPe9E3f9K677nrUox71yZ/8ya/5mq/5x3/8x9/93d/9Ae//AYuNBf9rlVKOHTv2p3/6p2/8xm+8Wq1uu+22xz3ucX3f/8M//AP/FQAq//nGcfyFX/iF7/zO73zP93zPt3zLt3zyk5/84z/+4x/wAR9w4sQJ25K4zPY0TeM48jwuXrz4si/7shEhyTaXvcRLvMT3fd/3TW2yDUiSNJvNzp49+7SnPe3MmTMRwb+aISF4/gwJCQkGg/jXcPquu+56xVd8xWuvvfYnf/Inz58/f+LEiRtuuKGU8uhHP/rRj3607Z/72Z+78cYbIwKwffr0aadLKW/8xm+8Wq329/f39/f57yT+00zTdN/Z+37wB37w9OnTr/CKr/Ce7/mefd+XUoydbq3ZLqXUWk+cOPGmb/qm0zTt7+//3u/9niRJb/3Wb/0RH/ERN91006/92q+9zdu8zfXXX4/43y4iSinv//7vf3BwcPz48c/+7M9+yEMeIqm1ZlsS/4kAKv/JbN9+++1/+Id/+KVf+qWPecxj+r5/+MMfXmv9oz/6o6Ojo/lsLgmQJKnve9s8gO3W2h/90R997Md+rCRAEpf1ff/IRz5yb29vNpt1XYexfdeddz3xiU/84R/+4Rd/8Rff2tqSxIvEyDybeUGUyGAw/3pTmx7/+Me/4iu+YinllV/5lZ/2tKe9/Mu//Ku92qvZlgRIesITn/BGb/xGkrjs0Y9+dD/rAUl937/ma7zmhYsX+G8jLAgQ/9HGcbz33ns/5VM+5bVe67WGYXiN13iN+XxeSuGyw8PDX/7lXy6lvMZrvAbQ973tv/iLv/iZn/mZRz7ykYeHh7PZ7MSJE1/7tV+7v7//KZ/yKfP5/M3e7M1KKbYBSfxvkJmttdZaZkZE13XAyZMnP/IjP/Laa6/d2NjY3NyMiM3NzSc96Ukv/uIvXkrhPxFA5T/Z4eHht3/7t7/My7zMYx7zmL7vJdVaX/mVX/n3f//3f/Znf/Yd3v4duF8pZbFY8JwycxiG1Wp18uRJntM0TefOnfvWb/3WV3mVV3md136ddE7T9AVf+AVPe9rTVqvVpUuXNjc3JfEvM5hnMhjE82dISDD/JrZ3d3d3dnaA66+7/i/+4i9e/uVf/i//8i/f+I3fOCKAaZr29/eBaZqGYZjNZi/1Ui8lSVJEtNaOHTs2m88yUxKXSeK/lPjPcenSpc/5nM95ozd6ozvuuOM93/M9T5w4Ydu2pGEYnvKUp/z6r//6K77iK37Kp3zK+fPnX//1X/+GG2548IMf/EEf9EF/+7d/+7mf+7mPecxjNjc3bf/e7/3ex3zMx7zyK79yKYX/VaZp2t/b/53f/Z3Dw8O9vb2+79/ojd7ozJkzL//yL7+/v/8SL/ESETFNU2vtpV/6pZ/whCe82Iu9GP+5ACr/yc6dO/cXf/EXr/7qr15KiQgAOHHixGu/9mt/z/d8zxu/8RsfP3YcIIgIYJom24AkoE3td37nd17jNV6D5/GkJz7puuuue9M3edPZfFa7Kqnrus/9nM/9lE/9lPPnz5dSJPEiMc8mEP9pJO3s7EQEgKi1TtP0lKc8BbANREREALfeeus3f/M3f8InfMK5c+dOnTo1n88lYWpXn/IPT1ksFhsbG5Ik2QYk8V9BIBAW/3Ey8+jo6Ku+6qtuuOGGv/u7v3ud13md66+/PiKmabr99tu3t7f/7M/+7Ad+4Ac+67M+a3Nz8y/+4i++9Eu/9OEPfzhgG3jsYx/7Fm/xFk9/+tN/+qd/+o3e6I3e6q3eajFf8CIxmP8BbNu+++67P+ZjPubYsWOf/MmffO211953332f9Emf9LVf+7Wv93qv9+u//usRIanWCjz60Y/+9V//9czkPxdA8J9pHMcv//Ivf4VXeIXMnKbJNgBIetSjHjVN06233sr9JAHTOPEAxk94whMe+tCHjuO4Wq1Wq1VrDRjH8Wu+9mte//Vf/5prrjlx4kSttdba1e6WW2551CMftbW11XWdJP4VxH+Jvu8lTdM0DEPf9xFx8uTJruumaZqmyTZg+2lPe9q7vMu7/OIv/uLXfu3XHhwc2LYdJQ4PD//wD/9wuVxKAmzz30D8hxqG4fM+7/Oe8pSn3HTTTY95zGNe6ZVeSRIg6eDg4M3e7M0++qM/er1e11r/+q//+pGPfOTDH/5w7icJKKW8+Iu/+Kd92qe9wiu8wqyfGfOvYP6b2LbdWhuGYX9//9d+9ddsf8u3fMuDHvSg7e3ta6+99q3e6q1+6Zd+abFY2AZsA5Lm8/l6vR7H0bZt/rMAVP7TtNZuu+22cRzf8z3f86/+6q8ODw/n8zn3u/nmm9/nfd7nJ3/iJ1/6pV5aIUBSKWVq0zAMrTVJtjPz8PBwY2Pjt3/7t3/v937P9nu+53s+4hGPuPXWWx/2sIddf/31/ay3LQlQ6GD/4N777v2ET/iE06dP8/zY5rkFLpgHEP8y8a8nKSIy82/+5m/29/d/8zd/8/Ve7/V+53d+ZxzHZzzjGbfeeutrvMZrALZf67Ve60/+5E/e8R3fcbValVIy88KFC+fPn1+v109/+tMP9g9OnTolSRL/m2VmZv7QD/3Qvffe+4Ef+IF/9Vd/9W7v9m7Hjx+XBJRSNjY25vP5Nddc86Ef+qFf8zVf81d/9Vff9V3fZTsieID5fC5Jku3a1cyUxPORAAQABoP572N7GIbHPe5xv/d7v3fvvffeeuuti8UiImqtwPb29pu8yZu8//u//6u+6qs+5SlPWa/Xs9kMsA0cO3aslAIAtgFJ/AcDqPynycwf+IEf+Id/+Ifv//7vf9SjHvWEJzzhVV7lVUopmZmZmVlKuXDxAqK1Nk3Ter3e2NjITOAv/uIv/viP/3h/f/+93uu9hmHouu61X/u1X+d1XiciJAFPf/rTH/awh5VSAMC2JGC9Xl9zzTUPe9jD+FcR9xOIF0YACADxb1JrffmXf/n1ev1Hf/RHrbVpmtbr9cMe9rDrr78+M7e2tkIxn89f5VVepda6s7Nje71e7+/vX3fddRsbG2/yJm9y6zNuffBDHiyJ/+VsHxwc/OzP/uwnfdIn/czP/MyHf/iHX3fddZKAzGyt/emf/ul99933qq/6qo997GP/+I//+Lrrrrvmmmts80JJ4n+81towDD/2Yz+2WCze6Z3eCfiKr/iKBz3oQZK432w2e+M3fuPf+Z3fefzjHw+s12sgIiRtbm7atg1I4j8FQOU/jaSnPvWpH/ABH/A7v/M7v/Ebv3H99dffeOONN99881133fWrv/qrFy9ePHv27OHh4aVLl/7kT/7kx37sx/q+P3fuXGZO03THHXe01l76pV/65MmT58+fB37v935vc3PzZV/2ZUMREQcHB6vVyrZtHuDWW289derU5uYm/zrm2cQLJBAEiH8T2621aZxqV0spkmyv1+tLly71fX/PPfc8+MEP3t3dRQBd1wEnT560vVgsHvrQhwK2H/vYx/7DP/xDZpZSAEn8rxURf/EXf/HyL//y3//93/8hH/Ih11xzDZetVqs777zzz/7sz37sR39M0qu+6qtO0/S3f/u3b/AGbzCbzUopACCJ5ySJ/yUi4uLFi/fee+8Hf/AHLxaL1tprvdZrPeEJTwBsA5L6vn+rt3qrN3/zN3/VV33V7/qu77rjjjtWq9U111xzyy233HbbbefOnbvxxhsl8Z8FoPKfZrlcbm1tvcEbvME7vuM7Pv3pT/+TP/mT3/iN33iXd3mX66677s3f/M13dnYuXLjwaZ/2aYeHh6/xGq/x8Ic//Prrr//Zn/1ZzHw+f+d3fmfbksZx3Nvbu3Tp0oMe9KC/+7u/Ozw8fI1Xfw3gZ3/mZx/xyEdkJs/piU984kMe8pD5fM6/kSEheEEsnkn869m+7777ogQAvPRLv/TZ+85ubGyM41hrfdjDHoYpUYQAScA111yzWq1sS+Ky7e3tJz3pSa/2aq928uRJ/pcbx/Hrv/7r1+v113zN1zz0oQ8FJAG2Dw4OXuzFXuzgTQ9Onjr5Fm/xFufOnbvxxhvf+Z3eudbK/3K2JUlaLpdbW1ubm5uZWUp50IMe9LM/+7PTNNVauV/f92/3dm/3hm/4ho9+9KOBJz3pSeM4juN43333/fIv//J7vdd7dV3HfxaAyn+ao6OjcRwzs+u6Rz7ykQ9/+MO/67u+64/+6I9e93Vf95prrmmt1Vq7rlutVvP5/BGPeMRqtaq1plPSOI4RwWWtte/5nu+ZzWattR/5kR9p2e47e998Mf/QD/3Q2WxmWxIwTZOkJz7xiW/1Vm9Va+VfS+bZzAskHCD+TWwfHh6u1+ta68/93M/9zd/8zXq9fpVXeZXjx4+XUoDWGsKY+83n87/6q7+67rrrZrMZIGlnZ6fv+/V6zf9+P/3TP/2MZzzjq7/6qx/60IeWUrjffD5/iZd4ib29vR/7sR974zd+42PHjv3t3/7tm7/5m29ubfIikMTzFzyb+O+TmcDTnva01pqkWiuwWCyWy+U0TV3XAZk5DmNE3HTTTb/8y7/8Yi/2YpJe7MVejMte4iVe4ou+6Isyk/9EAJX/TOfOnfuCL/gC4J577rn11lt3d3ff8A3f8OVf/uX7vv+DP/iDJzzhCeM4SuKyiLjhhhsy8/u+7/v+/M///NixY6/yKq/yBm/wBsBsNluv17XWvb09Sd/wDd/wLu/yLlubW5K4X6314oWLd999d9d1mRkRvKgM5pkMBvEvEP8mtodhWK1W6/V6HMeHPOQh586du+WWWzY3N7ns7Lmzy+XSNmAbsP0Hf/AHr/d6rzebzbis67rz58//2Z/92Vu8xVtI4r+B+Q/yO7/zO+/6ru/68i//8qUUnsddd911zz33vN/7vd/58+d/5md+5iM/8iMzMyL4Dyb+a2XmX//1X995551PfepT/+7v/u7ee+89c+ZMKeXEiRNnzpy5++67H/rQh67X61/8xV987dd+bUm//Mu//FZv9VaSeIC+72+66SZJgCT+UwBU/tNExOu//uu/4zu+4/HjxyXZvnTp0hd/8Rf/6Z/+6cMe9rC//du/fZ/3eZ8TJ04cHh7+wA/8wE//9E8fO3bsiU984hd/8RcvFosv//Ivb6198zd/80u8xEusVqtf+ZVfeYM3eIM777zzF37hF37+53/+8Y9//Gu91mshJHE/2+fPn9/f35/P5/zrGMwzCcQLJAgIEP8mkra2tlar1fXXX/+O7/iO6/X6677u606cOAFkJtB13T333DMMw6yfGQPTND3jGc8YxzEzJWWm7fV6/Vd/9Vev//qvv1gsJPFfx5BgZCz+HaZpuvvuuzc2Nt77vd97sVjwnDIT+Pu///vrrrtusVh8zdd8zeMe97itrS3+JdM0lVKA1hpQa+WFEf8dnvCEJ+zu7r7pm75pRPzMz/zMj/7oj374h3+47ZMnT37oh37oD/7gD37yJ39y3/Vv/dZvDezu7l66dOlN3/RNbUuyLQmQtLOzc/fdd99yyy2S+E8BEPynmc/nL/uyL7u1tVVKiYhSytbW1gd8wAf86q/+6hOf+MRTp05tb2+/xEu8xGq1+tM//dNv/dZv/ZZv+ZYf+ZEf+eEf/uEbb7ix1lprfcxjHnPvvfc++clP/uEf/uGXfdmXfYM3eIOnP/3pr/zKr/zpn/7pn/qpn/oHf/AH0zTxAN/8Ld9cSjl9+rQk/nOJf5OIOHny5HK5BCSFYrVa/fIv/3IphcuOHTt25513ttYODg+cxtRaT506lZnjON5zzz2/9Vu/9QPf/wNPfOIT9/f3p2mSxH8D8+9WSvnDP/zDD/7gDz62c0wSz0nSwf7BH/7hHx4dHX38x3/8iRMnHvSgBx0cHPAiWC6XP/ADP/D5n//53/zN3/ybv/mb0zTxLzD/tX7lV35luVxKkvTKr/zKT3rSk+65557W2tHR0bFjx57+9KfffffdxpIkRcQtt9xSawVsc5ltLnviE5+YmfxnAaj8p+m67vd+7/ce9ahHzWYzLuv7/sEPfvBLvMRLfOM3fuPW1tajHvWoe+65Z2tr6xGPeMTJkyclHT9+/ODg4PyF84DtYRj++I//+PVe7/W+9Eu/9OLFi09+8pOPHTv2mMc8JiL29/e/4Au+4EM/9EPf5E3eZD6fA8MwPOMZz3j0ox/d970k/hUCChbPJv5zRMSpU6eGYchMAHHy5Mk777xT0jRNERER0zT9/M///BOe8IS777671nry5MmP+IiPAD7t0z7tuuuue+xjH7t/sL9er++9995Lly5tbW1JksR/EaMEg0H8O6zX6z/7sz973dd93dpVLrMNZCaA+Zu//Zu//Mu/vPbaaz/5kz/5+uuv/4RP+IRf//Vf/8AP/EBegNaa7d/7vd/7uZ/7OUmv9mqvtr29/Tu/8zt//ud//p7v+Z7XXnutJP5nsA1ExKVLl9br9Uu91Ev95m/+5su+7Mv++Z//+du93du97du+7Z/8yZ+85Vu+5VOf+tT9/f2bb7657/uLFy+eOHEiM2utrTVA0smTJ4+OjjKT/ywAlf80rbV77rlntVrxAF3XveM7vuNLvuRLPvnJT/6TP/mTO++889577/3wD/9wSVyWmd/2bd/2W7/1W7b/4A/+4MSJE1/5lV85m82uu+6606dPC5VaJH3gB37gM57xjGc84xm/+7u/+9qv/dpd1913333nzp17yEMe0ve9bUm86Mz9BOI/TSnlUY961JOe9CRJgKTXeq3X+s7v/E7gwoUL29vbs9nsZV/2Zc+cOXP27Nk777zz2muvfehDH/q7v/u7f/VXf/Xe7/3eL/ESLzGfzcdpfKd3eqdv/uZv/rVf+7V3eqd32tjYsA1I4n+Pxz/+8S/1Ui+1tbUF2JZ0eHj4/d///efPnz84OLhw4cKtt976Rm/0Ru///u9/4sSJ1Wr1sIc97Kd/+qdf+7Vf+9GPfjTPw/bBwcEv//Ivb25ufs5nf06U6Ps+M1/jNV7j1ltv/bqv+7pP+7RP29jY4H+G9Xr9hCc84WVf9mX//u///jGPeczLv9zLf/O3fPMbvuEbvtM7vVMp5cVf/MU/5EM+5JVe6ZUODg5e7MVe7Cd/8idPnTr1V3/1V6/7uq8biic84QmnTp06derUer3uuo7/XACV/zSLxeLg4ODee++97rrreID5fP5SL/VSL/7iL56ZFy9e/LiP+7hf//Vff9mXfdlSSkRcf/313/AN33DXXXc98YlPfKM3eqNXe7VX29zclAR0Xcf9uq77mI/5mG/7tm/73d/93Vd8xVfc2d558pOfPI7jS7zES/R9L4l/FZlnE/8y828iaWNj48lPfvKrvMqrAF3X3XLLLRcuXHiP93iPzc1N29dee+1v/MZv/Pqv/3pmXnPNNY997GNvvvnmV37lV37Xd33Xn/zJn7z55ptvuOGGPvqTJ0++zdu8zU/8xE9M0wRI4n8Pp23/6q/+6vu8z/vMZjNAErCxsfEu7/Iuf/e3f3d4dPi7v/u7L/7iL/7+7//+XddFxMbGxou92Iv90A/90Nd//dd/wRd8wXw+7/teku1pmoBf//Vf/5Ef+ZEv/MIvvOaaa2qtPMDDH/7wM2fO2OZ/jMz8q7/6q3d913d9ndd5nYjY3t4+ffr0n//5n7/O67wOMJ/Pl8vlJ37iJ37O53zOYrF4ozd6ow/8wA88ffo0kM5f//Vf/4AP+ABgPp+XUnZ3dzOT/ywAlf80tg8ODnZ3d3keERERtk+dOvUJn/AJn//5n/+DP/iD11133TRNr/iKr7ixsfHIRz7yYQ97WESUUiTx/Fx//fUv9mIv9uIv/uJ919v+7u/+7q2treuuuy4i+LczGMTzZzAkmH8r2+M4ArYvXrx49uzZT//0T3+d13mdM2fOfO/3fu/111//Jm/yJvP5/Dd+4zfe+Z3f+eTJk7PZDHD6Dd/wDX/zN3/zdV/3da+99lpJ119//fHjx9frtST+dxFPe9rT/u7v/i4iJHE/oc3NzVd65VcC/u7v/q7rutlsBgCllDd8wzf8nd/5nac//el33HHHYx7zGMB2Zp4/f/77vu/7fvzHf/yd3umdTp8+XWvlOd12223PeMYzbPM/xjiOf/3Xf/1nf/Znb/mWb7ler8+fP//IRz7yj/7oj1791V/9SU960i/90i+99mu/9s/93M/VWp/85Cf/6q/+6tbW1u/+7u++4zu+4xOf+MQf//Eff6d3eqeTJ09GBGCb/0QAwX+aiHjUox71Ez/xE/fee29m8jwkdV33Yi/2Yt/xHd/xaq/2at/5nd/58Ic//MEPfnDXdaWUvu9rrZJ4fiIiIt7kTd7kvvvu6/ruGbc94/bbbv+Mz/iMa665hn8bcT+DeYESJSQYzL9eRFy6dOnuu+/+tm/7ti/+4i+2/R7v8R4PechDdnZ2XvZlX/aWW27Z2tpaLpeSTp8+vbm52XVdrbV29czpM2/wBm/wbd/2bffeey+wsbExDMPf/d3frddr2/wXEQQIxL+V7W/5lm95y7d8y62tLR7AOBSllFLKsWPHdnd3V6uVJEnAzs7Ol3zJl7zru77rh33Yh/393//9arW6/fbbv/Irv/JTP/VT3+qt3upjP/Zjf+/3fq/rOgDIzGy5Xq9/8id/8uM//uPf7u3ebj6f8z/GxsbGu77ru/7N3/zNM57xjHvvvVfSa73Wa911111PfepTf+iHfuhDP/RDP/ADP/Caa6756I/+6Mx8xCMeIelN3uRNPvIjP/KDPuiDPuVTPuXUqVO7F3eHYWhTWy6XkvjPAhD8p7H92q/92k972tPOnj0riReg67pjx45tbm6+8Ru/8TXXXFNKkSSJF0Hf96/+6q++XC6///u/f3Nr82EPe1jXdfyrGQwGwGBeIKOEBIP5NymlPPjBD/7pn/7pd3mXd/msz/qshz/84X3fA621l3mZl/n8z//8UsqLv/iLf/AHf/BsNpMESAJKLceOHfvgD/7gb/qmb/rrv/5r4DGPecxv//Zvr9dr/isZEP8OT33KU++7777XeZ3X6fueB5CEuGJjY+Nv//ZvL126xP0iYnNz83Vf93U3Njbe673e6z3e4z3e7/3eb39//0u/9Esf9KAHvdEbvdHBwYFtABjH8fyF8+/3fu/3d3/3d9/4jd/4Cq/wCqUU/scopbzUS73Uq73aq33/93//OI7Hjx8/derUa7zGa/zkT/7k677u69Za+76fzWYf8REf8bM/+7Mv8RIv8ZVf+ZX33nvve73Xe73xG7/xk5/85Gmavuu7vws4e+7ssWPHJPGfBSD4z3TddddN0zQMAy/UNE17e3uv93qv97jHPa6UwmWSJEmSxPOQJKnruoiYxunP//zPP/zDP/zkyZMRwb+awTyTQLwwBsD8W0VErfUxj3nMsWPHNjc3SymSIgLouu6DP/iDp2mqtW5ubkbEOI7nzp379V//9dYa0HXdqVOnPuIjPuI7vuM7fu3Xfu3aa6+ttbbWWmu2bfOfThAAFv96mZmZf/4Xf/7Kr/zK29vbknhOkiRJ6vv+3nvvPXv2LA8g6dSpUx/3cR8n6RM+4RO+6Zu+6X3e531OnjzZ9/1isXilV3ql3d1d27Yj4tu//dtf/KVe8iM/+qNOX3NNnfWS+B9juVzeddddN9988ziOf/EXf/Ht3/7tP/mTP3lwcLC/v//7v//7T3nKU86dO7e/v3/TTTc9+tGPvuaaa7a3t9/t3d7trrvu+sRP/MQ77rjju77ru+677z4ko+MnTibiPwtA8J8mIk6dOrWzs/P4xz9+HEdeqJ2dnb7vv+7rvs42LzKhcRx/9dd+FXiFV3iFruv4H0/SjTfeuLm5aRuQxGURUUp53dd93fl8/q3f+q3/8A//cHBwcPbs2e///u9/mZd5mVKKJEkRceLEiU//9E8fhuHXfu3Xtra2bHM/2/xXEP8mtler1Q/90A+9/du/fd/3vGC11nEcL1y4wHOqtb74i7/4wx/+8IsXLz7kIQ+54YYbJEnquu7d3/3dn/CEJ9i2/Qu/8AvHjx//8A/9sM3NLYX4HyYifvu3f/vxj3/8Qx7ykJ/+6Z9+2tOedvfdd4/j+IhHPOKv//qvP/RDP/S93uu9JF26dKnv+9baOI7f8R3f8fqv//qz2ezjPu7jMvOWW24BGRTBfyKAyn8O28DW1tbJkyef+tSnjuPY9z0vQCnlmmuuOXv27IULF9brNS8y42c84xnf+Z3f+TVf8zXHjh2TxL+FILB4NvGfJiIe8uCH/O3f/a3tzFwul5m5tbUFZGbf9w972MPe533e5/z58x/6oR/66Ec/ejabSQJsA5Jqrddee+2bvdmbPfShD/293/092xEBSOK/iPg3iYg//IM/fOQjH7lYLCKC5ySJ+9Vap2lar9c8J0lbW1uv8Rqv8XM/93Nv8AZv0Pe9JCAiHvKQh5w+fdr2P/zDP/zQD//wN3/TNy02NgAhIAEI/ke4/vrrNzc3X/d1X1fSm73Zm+3v79s+duzYbDZ7ndd5nQ/90A996EMf+vSnP/26664bx/GzP/uzgfd6r/c6ffq006dOnXqP93rvr/jyL//jP/mTu+66+/ixEzYJQPAfDqDyn6nrug/7sA/7mq/5moODg83NTV4ASZIkrVar7/zO73zlV35lXjTr9fqHfuiHXuIlXuKGG24opfBvJCyeSSD+ZQLxbxIRi43F3t7e3Xff/TM/8zMPe9jDbr/99rvuuuvaa699zGMec+011yL+8i//8k//9E/f+I3f+I3e6I2GYXjyk5/8ki/5krPZDLAtKSLm8/mjH/3oP/7jP37KU57y8i//8hHB/3jTNP3gD/3g+73f+836GS/U7u7uOI7L5ZLnUWvt+/4f/uEfJEnifqWU/f39L/qiL/qHf/iH93zv997a3pbEi0T813qZl3mZP/zDP4yIxWKxsbFx6tQpAFiv16vV6mM/9mPPnTv3iEc84jd+4zfe9V3f9ZGPfGREnDhxAogSwGI+P3ny5Bd8wRfecMMNr/xKr8x/IoDKfw5JtiU9+MEP7rru9ttvv/baa3mhtre33/d93/frvu7r7rj9jhtvulESL1Rm/uZv/ubTnva0L/uyL9vY2JDEv5nMs4kXSBAgnkn860kCfuVXfuXBD37wu73bu/V9P47jOI533333OI57+3vAox71qFd5lVe59tprZ7NZZv7lX/7l3t7eqVOnQqGQbUm25/P5ox/96B/5kR955CMfeeLECf5ns/1rv/Zre3t7L/mSL1m7alsSz8M2cMcdd0zTVGvledRaX+VVXuWHf/iHJbWpIVprT3/607/5m7/5r//6rx/60Id+zud8zou9xEsg8T/V9ddfv7m5OU0T97N9cHDwZ3/2ZxsbG4973OPe9V3f9WM+5mMe9KAH/fVf//UrvdIrZcuIkNRaG4bhB37wh374R350Md/44z/+04/72I8PBf9ZACr/yRaLxcu//Mv/6Z/+6Yu92Iv1fV9KAWxL4jnN5/M3eqM3+s7v/M6f/bmf/eAP/mBJvGCZ+cu//Mtf//Vf/8Vf/MXb29u1Vv7DGMQLIxCIf6uIeOQjH/niL/7itVZgY2MDOH36NNBaAyRJsg2UUl71VV/1K77iKz7ogz5oZ2en7/vFYgFIAl7hFV7haU972qVLl06cOMH/YNM0jeP4Td/0TW/wBm8wn88BSbwA0zStVivbEcHziIjrrrvu2muv3dvb29jYuO0Zt33zN3/zX/7lX1533XUf//Ef/wqv8AonTpyIWkMCzP9EW1tbFy9evHDhwvb2NpetVqu/+Zu/OXHixM/8zM+87/u+7w/8wA98xEd8xIu92Iv9zM/8zN/8zd+84zu+44kTJ7jsD//wD5/61Ke+6qu+6kd8+Id9wAd+8PHjx6IE/1kAgv80kiRFxKu/+qvfeuutFy5ckAQAknh+FovFS73US33f933f2bNnp2kCDOY5DNM0tfakJz3pcz7ncx772Mc+6EEP6vteEv8xDAbzgjiw+HcrpZRSeB6llFKKJKC11loDdnZ2XvzFX/xbv/Vbv/mbv/kZz3gGl0mStLGxceONNz7jGc/gf7aI+Iu/+Iu77rrrbd/2bUspvGC2Dw4O7rrrrtlstrOzw3OyDdRaJT35yU/+sR/7sU/6pE+S9DEf8zFf93Vf98Zv/ManT5+utRZJIJ5NIACDeSZDgsH81+r7/vjx409/+tMz03ab2t/93d/deOONfd8/4hGPmM1mf/u3f3vTTTedOHHiHd7hHQ4PDz/zMz/zR3/0R//u7/5ud3f3Z37mZ3Z3L37Cx3/c9Tfc8AZv8PobG4sICcR/BoDgP1lE3HTTTYvF4i//8i+HYeCF2tjY+PAP/3Dga7/2a+++++7MTGxIMBgMR0eHf/v3f/cpn/IpL/mSL/kRH/ERW5tbEcG/l5EBMJh/gSD4d5B011137e7u8gJIAu66667f+Z3fGYbB9mu+5mu+1mu91lu8xVvcdNNNPEBElFJuvfVW/kuZfyXbP/7jP/6O7/iOZ86ckcQLJunw8PCpT33qq73aqz32sY/lOUlaLpf/8A//cO+99/7AD/zA8ePHP+3TPu1TPuVT3uRN3uTkyZOlFEmSuCx5NoEADAaDwWAw/+W6rnvkIx/5pCc9ablcAogXe7EXu+aaa+644443e7M36/t+Pp/3fR8Rm5ub7/Ve7/Wmb/qm3/RN3/RxH/dxf/7nf/6+7/u+n/WZn3nq5IkSuuVBN88XMwmB+M8AUPlPJmlzc/Mt3/Itv+zLvuyGG2546Zd+6VIKz4+krutuvvnmD/7gD/7e7/3eO+64483e7M1OnTlz+szpfjYTnD137sL587/7O7/zZ3/25yeOHfucz/6cU6dPlVok8e+SYJ5JIF4ggUD8+5RSHvWoR/3VX/3V677u6/L82L7zzjt/5Ed+5H3f932nafr7v//7hz/84bXWBz/4wfP5nAeQVGu99957+S9iMBgZixfZ7bff/pd/+Zff8z3fU0rhhZL0pCc9aRzH13u919vZ2eEBbI/j+Bd/8Rdf/uVf/qqv+qof/uEffurUqVprZpZSuEwS9xP/Q9l+1KMe9Tu/8zv33nvvQx/60Frr1tbWOI4v8RIvsbOzs7u7u7m5ube3Zzsijh8//tqv/dq//Mu/fPPNNz/96U9/rdd6rfl8LintNo4lQjYS/ykAKv/5JD3qkY96xCMe8cM//MMPetCDTp06JYkXYD6fv8M7vMNLvdRL/dzP/dwP//APX9i9eOnSpdJ3ksZhbK0d7O9fe821H/MxH3Pq9Km+7/mvJhAE/w4RceONN955551cZhuQxP2Wy+Wv/dqvvcVbvMXGxsbR4dF8Pp/P5y/5ki8JlFJ4TqWUzOS/jvnX+53f+Z0bb7zx+uuvjwheqGEYfuu3fmtnZ+elX/qlI4IHmKbpzjvv/Lqv+7pXeZVX+eAP/uCdnZ2IAEopPD/iBRHPZv7L2d7Z2bn++ut/8zd/88EPenDtKtD3/Y033gjYvu+++574xCc+6lGPKqVERK31VV/1Vd/8zd/8j/7oj3Z3d6+//nogWzt/7pz4TwUQ/JfY2tr68A//8IODg6/8yq+8++671+t1aw1wOjMzcxiGO+6440d/9EfPnTs3m81e4iVe4uM+7uM++ZM/+dGPetTLvMzLnLvv7N133TVN00u/1EvN5/N3fud3evlXeIVSKyCJfy+BsHg28Z8pIh7+8If/9V//tW0ewLbtcRx/+Zd/+cEPfvBDHvKQrna/9Mu/dOONN3Zdd/LkyY2NDZ5TZnZdN03TOI78VzAyJJgXTWttb2/vJ37iJ1791V89InihMvOee+7567/+69d+7dc+duwYl7XWMnO9Xj/pSU/63M/93Ouuu+4DPuADtre3JfGvJhDPQfx36LruFV/xFZ/85Cdf2rvE/SQBGxsbD3vYw37gB37g0qVLbWpAKeWt3/qtF4vF677u61577bW2M/NpT33am77pm5ZS+E8EUPkvodB111334R/+4Z/0SZ/0wR/8wW/4hm/4Ui/1UjfccEMp5fDw8I477nj605/+m7/5m9ddd90rvuIrnj59Gui67tGPfvQrveIr7e3t1VqP7Rx7/Td8g5d7mZf5uI//hOuuu672nUpg/iMIB88kEP/JIuL6668fxzEzI0ISYJvL/v7v//7uu+9+4zd649lsdvfdd994442LxaKUwguwubm5Wq1Wq1XXdfxXMP8arbXv+q7veuITn/g5n/M5pRReMNt7e3tf/MVfvLu7+wav/waL+cI2YHt3d/dP//RPv/qrv/rYsWOf//mff/LkSUn8W4hnEwAG8V8rIoCHPexhr/Iqr/KUpzzl1KlT3E/SYrH4kA/5kO/+7u/+u7/7u9d4jdcAJM1mMx5gmqZn3PaM136t144I/hMBVP4LPexhD3uDN3iDb/qmb3rSk560vb29s7OTmV3Xrdfr48ePf+AHfuCrvdqr7WzvcFkpZWtr6z3f8z3HcRymqeu6WmspsdhYrIcBACTxH0Lmv1Yp5aabbrrttttuueWWUgr3m6bpj/7oj97rvd6rn/XDMPzhH/7hG7zBG8xnc16AUPR9P01TZtqWxP88P/MzP/PQhz70UY96lCRJQGut1soD2B7H8dd//df/4A/+4HVf93Uf/JAHT21ar9b3nb3vKU95yk/91E/97d/+7S233PI5n/M5N910E/9hxH8HSbb7vn/kIx/5q7/6qy/zMi/T9z33k3T8+PH77rvvp37qp17u5V5ua2uL5zQMw9/93d/ddNNNEcF/LoDKf6HZbHby5Mn5fP6IRzxib2/vtttuu/HGGx/5yEdubW3dfPPNj3jEI44fP879IsI20Pd9N58BgEGSJCH+2xiSf59Symu8xmv81V/91UMe8hAukwSs1+uXe7mX67quRLnnnnv+8i//8k3e5E0QL5DITEmlFP7nsf13f/d3t91226d+6qdGBJeN4zhNUylFEgDY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3nt/53d+5+3e7u1KKZIkzWazt3u7t/uUT/mU937v9/6zP/szSXfeeed11133QR/0QceOHfvWb/3Wu+++e39//0/+5E8kfe/3fu+HfdiHveEbveGnfdqnjePIv5cwLxrxbxWKP/+LP3/lV37l48eP265dvfHGG9/pnd7pF3/xFz/8Iz78y7/8y2+44YZbbrml67pHPOIRv/RLv/SYxzzm3d7t3XZ2dra2tn70R3/0FV7hFR772Mc+6lGP+tZv/dZHPepRT3va09br9ROe8ITt7e33eI/3+IM/+INf/uVf/rAP+zD+9/iDP/iDt3u7t1ssFlx24403vtRLvdTTnva0O++885GPfOTh0eGZONNaK6UMw7C3t3fhwoXf//3ff93Xfd2NjY3f//3ff9VXfdWu62644YYf//Eff6u3eitJ/M+2ubn5kR/xkbfdftvh4eHjH//4r/3ar32lV3qld32Xdz1x8oTtaZqAv/7rv77jjjvuueee3/zN3/yzP/uz137t17755pvf9m3f1vYv/MIv7OzsvMzLvMzLvuzLctldd901TdPLvMzLRASXXbhwoe/72WwmiX8Fg8EIxHMCCP6Ttda+7/u+b3t7+9KlS6/wCq/wCZ/wCV/wBV/wNm/zNr/2a792sH8gicu6rjtz5sw999xTSvnIj/zId3rHd3rCE57w4i/+4n/2Z3/WWnulV3qlv/3bv/3lX/7l1Wr1rd/6rcePHz99+vTNN9/8mMc85p577slM/l0Ewb+a+NdI5+2333733XdzmaRa67u927v96Z/+6XXXXffQhz70sz7rs/78z//8/Pnz0zi98zu/8+Me97izZ8/efvvtrbVbb731m7/5m++8886u62qt+/v7D3nIQ2w//vGPf8xjHtP3/c0333znnXceHBzYts3/BqvVCpAkqe/7JzzhCaWUl3iJl3ijN3qjt3zLt3zqU596cHBw69NvvfXWW8dxlHTttde++Zu/+ebm5jiOP/7jP/6yL/uytvf39//6r/96b2/PNv+zlVJOnDzx13/913/3d3/36Z/+6Q9+8IPf7d3e7fiJ40BE9H2fmX/6p3/61V/91U972tO++Iu/+NM//dPvuuuuT/3UT/2Yj/mYz/iMz/jbv/3bd3mXd4kISZIknT9//lVf9VVPnzrddR2XZaYk/i0MBvPcACr/mWx3XXfLLbe01p70pCd9zdd8zTu/8zsvFotTp06N4/g3f/s3r/qqr1prHcfx8PBwd3f32muvPXHixMbGxjXXXvMqr/IqJ0+e/L7v+77VavXkJz/593//99/zPd/zzd/szY+WR7/zO7/zFV/xFR/8wR98yy23/N7v/d7DHvawxWLBv535tzAvMttd173u677u7bff/uhHPxqQJGlra+uDP/iDv+mbvuljPuZjfviHf/grvuIrtra23u3d3u2Rj3zka77ma371V3/1wcHBy7zMy3zzN3/zH/zBH/zhH/7hzTff/JSnPGVra+td3+Vd+1n/4i/+4r/8y7/8x3/8x6/2aq929uzZ3d3dnZ0d/gezbVsS5pZbbvnzP//z137t1y5RdnZ2fvu3f/thD3vYgx70oK7rtra2dnd3f+EXfuHixYvXXXdd3/dHR0d33XVXREja2Nh45Vd+5fPnz99xxx2/+Iu/+Dqv8zr/8A//8Cqv8ir81xGIfz1Jj3vc4x73uMe927u921u+5Vvu7OxI4n5937/aq73a7/3e733RF31RKWUapzd/8zf/lV/5lVd+5Vd+13d912PHjp04fsK2JC7b3NjMTIV4gIiQxH8YgMp/GknAxYsXX+mVXum3fuu3HvOYx0TEbbfddvfdd588efJjP/Zjf+AHfuCRj3zkzs7OH/7hH/7cz/3c3XffPZ/Pf+/3fu8lX/IlSykv8zIvs729/aVf+qW/8Au/sFgsvuALvuAxj36MpGPdsbd8i7d8uZd7uZ/+6Z++9957n/CEJ7zLu7zLYrHgv4D5N4uIra2tn/7pn37Uox51/fXXA8B8Pn+zN3uzu+6867bbbvuAD/iAN3qjN/rqr/7qL/qiLzp27Ni7vuu7vv/7v/81Z645feZ0ifJmb/ZmT3jCE/70T//0nd/5nV/yJV5yY2MDcerUqXd4h3c4d+7cj/7ojz74wQ/+u7/7u1tuuUUS/4Mtl8tz586dPXv2wQ9+8M///M+/zMu8zMmTJzc2Nj7sQz/sx3/ix9/szd7swQ9+8PHjx2ez2R/+4R9+7Md+7Obm5ud//uffeuutZ+87+9qv89qr1epRj3qU7c/8zM98j/d4j7d/+7ff2tr69m//9pd+6Zfe2Njg3yhwgHgRWfzr2W6t7e7uvuM7vuM7vMM7LBYLwDaXtdYkRURrrbWWmffcc883f/M333HHHV/xFV/xoAc9iOdx8y0333333Xt7e7XWWitgezabdV3HfxiA4D/T1tbWd3/3d5dShmG4/vrrT548+Zqv+Zrv9Z7v9WZv9maPecxj3v7t3/7Xf/3Xn/GMZ3zHd3zHh33Yh73u677u3/zN3/z4j//43Xfffffdd//BH/zBwx/+8Guuueb93//9P+zDPuzFXuzFooRCXHbjjTe+93u996u92qvt7e2N48j/YJK6rnu5l3u5u++++xM+4RO+/du//eDggAd4+CMe/j3f8z3DMPzAD/zAJ37iJ37zN3/zfffd9xmf8Rkf//Efv3+wny2N5/P5y7zMy3zQB33QK73SK21vb3O/Wuv111//Hu/xHk960pN++Zd/eZom/md7whOe8E3f9E2/9Vu/9SVf8iWZ+XM/93NHR0e2b7zpxvd4j/f4nu/5nttuu221Wr3Kq7zKwcHBk570pFrru73bu128ePHN3uzN3uRN3uSVXumV/vzP/7zWeuzYsd/7vd/r+34cxzNnztx99938e4kXhfm3Wa/Xv/RLv3Ty5Mm3ePO3WCwWPIDtiFitVk984hNvuOGG1Wr1tKc97UM+5EN+7dd+7RVe4RUe9KAHYZ5XKeXJT37yz/zMz4zjaBsYhmE2m3Vdx38YgOA/Tba85pprNjY2br311jd90zf9xV/8xe/7vu970pOeNLWplNL3/YMf/OAnPOEJj3vc41prD3vYw975nd/5Hd/xHT/+4z/+S77kS77xG7/xPd7jPR7+8If3fV9KiYiI4DKFFMJsbm6+zuu8zvb29sH+QWvNNv9TbW1tvdmbvdl3f/d3d133pm/6pt/8zd988eLF1tpqufqd3/mdn/mZn3nIQx7yB3/wB9ddd90111zz0Ic+9Gu/9mtf+qVf+tM+7dM+8iM/8tz5c9M0RURERETf9wohJAERIenkyZOf/dmfvV6v9/b2siX/U61Wq1/7tV/77M/+7I/7uI/7ru/6rhtvvPEbv/Ebf+M3fmMcR0mnTp36kA/5kK/92q/9rd/6rYj41E/91N///d//wR/8wW/5lm85ffr0L//KL1+8eLGUcvvtt+/s7PzgD/7gF3/RFz/hCU/41V/91fl8vlqt+LcLCBCIF85A8G8yjuMv/dIvvfM7v/PJUye5nySgtbZcLm+//fY/+qM/+riP+7hP/uRP/rRP+7Rrr732kz7pk6655pqI4AEys7XWWjt79izwHd/xHb/7u7+bmbb7vi+l8K8mEAQW5jkBVP5z2EYE8d7v/d6f/dmffezYsZ2dnWuuueYTPuETvuALvuARj3jEfD4/derUO73TO33wB3/w6dOn9/b2ZrPZ+7zP+/zBH/zB13zN10TEer0+d+5ca62UcubMGUASYJv71VJf6ZVe6fM+//O+9mu/dmtri38j859M0nw+v/XWWz/6oz96Y2Pj7d7u7T7jMz7j4z/+4x//+Mc/+clP/rqv+7phGD7kQz7kL//yL9/qrd5qc3PzIQ95yKMe9ajHPvaxD37wgz/jMz7jcz/3c2ezmaStra2+7yOC55SZOzs7knZ3d08cP8H/VL/yK7/y1m/91qUUoNb6ru/6rrfeeutXf/VXv8IrvML1118v6cYbb/ycz/mcL/7iL/6VX/mVD/uwD/uIj/iIvu8/4AM+QNLtt9/+GZ/xGa212Wz2UR/1Udvb28ePH7/+husf/OAHf+zHfuynfMqn8G9mAAzmP09EHD9+vJQC2AaAaZoO9g8Ojw5//dd//bd/+7c/6zM/6/rrr/+Gb/iGUkpm3nfffb/7u787DEOtFQBsSwKe/vSnf+InfuL7v//7v+IrvuKDH/xgLsvMw8PDcRz51xEIAPHcACr/aSQZ33DDDcMwPOhBD/qoj/qozPzcz/3cD/mQDzl58uQHfuAHvtZrvdaNN974iZ/4iV/wBV/w6Z/+6VtbW3feeeeHfMiHLJfLo6Ojn/iJn/ixH/uxvb29Bz/4wT/xEz8BSOJZAiCIN3iDN/iTP/mT5dFya2uLfzOZ/2SllL7vgY/4iI94xCMeIemDP/iDP+3TPu3ChQv7+/t930/TNI7jh37oh77cy73cE5/4xBtvvHFjsfGN3/CNf/qnf/qe7/meQCnlC7/wC1/qpV6q1hoRkrifpI2Njdd+7df+gz/4g4c+9KG2AUn8D/Nnf/ZnJ06cePCDHyyp1lpK+fiP//gnPOEJ995775kzZyIiIra3t9/3fd/3L/7iL775m7/5+PHjH/ABH7C9vS3pIQ95yLd8y7fwnLquO3Xq1Fu91Vv9+q//+iMf+ci+7yXxryYcEPxn2tzcfJ/3eZ9pmoZh6PsekHTu3LmP+IiPuP322yV9xmd8xi233KKQJCAiNjY2Ll269LSnPe3hD394KQWYpsn2MAxf9VVf9fmf//kPfehD+77nfn3fD8MwDAP/Rua5AVT+k21sbNx0001v8zZvc+bMGeBTPuVTvuRLvuQZz3jGD/3QD33Hd3zHox71qJd/+Zd/x3d8x8c//vF937/VW73V7u7u933f9/3hH/5hRHz0R3/0i73Yi0niBVDommuu2d3dPVoe8e8i/vPN5/NP/MRPfLEXezFJr/d6r3fLLbd8+Zd/+Vu+5Vt+1md91mMf+9hXeqVXet/3fd8///M/39vbe7/3e78Xf/EX39jckPSIRzzi/d7v/V7u5V7u2LFjJ0+elBQRknhOs9nskY985F//9V9P01Rr5X+kV3qlV/qt3/qtc+fOvcVbvAVgu+/7z//8z//SL/3Sz/mcz7n22mtba621Y8eO1Vof+chHrlarkydP8pxsS+J+tdbXeZ3X+cRP/MQnPOEJL/kSL4n4TyTA/JvYftjDHvbd3/XdL/MyL3P8+PGIsL2zvfOFX/iF8/n89ttvf9rTnqYQD9D3/du//dt/x3d8x+d93udFxDAM4zjedtttP/MzP7Ozs/Owhz2s1soDRMRqtVqtVvzrJBiMeB4Alf8ckmxLKqVsbm7u7OxIkvSgBz3oa77ma57xjGf89V//9dbW1k033bRcLre3t6dpOjo6+qM/+qNHPvKRr/3ar/0O7/AO3/M93/Oar/ma4zh+6Id+6I//+I9zP0k8QK314OBgvV7zbxeYfz3xr9F13Su90ivVWiMC6LruMY95zFd+xVc++SlPfpd3fpdbHnTLddddJ+m1X/u1AUmSbNvevbT7Jm/yJpubmz/3cz/3iEc84jGPeQzPo5RSSun7/u///u8z0zb3k8T/GG/yJm/yV3/1V/fee+80TbXWaZp+9Ed/9PVe7/Xe+Z3f+Vd+5Vfe7u3e7uzZsz/7sz87m80e8+jH9H1/eHi4XC67rgNaa+M4TtM062e1q5K4TNLJkyff7/3e72u/9mu/7du+TYj/VDL/JtM0nTp1arVe7e3tnThxAgDmi/nDH/5w4Prrr3/Jl3xJ7mdb0nw+f/3Xf/0nPvGJP/mTP/kGb/AGP/3TP/3Xf/3XL/ESL7G/v3/99deXUiKCB4iIw8PD3b29tJF4TgLxghgMRjwngMp/sq2trePHjx8cHEgCbHdd9/CHP/zhD394ZtrOzJd5mZd5szd9s1LLNE21VmC5XI7jePz4cds/8AM/wAt24viJWiv/Dcy/xmw2e8u3fMu7776b+0naObbzci/3chHRpjaOY0TUWgFJACBpGAbbfd+/3du9ne2IkMTzI+nSpUuHh4fHjx/nf6RSymu+xmu2bH/3d3/3Ui/1UhHxZm/2Zttb28sHL7/sy77s6OjI9ju/8zsvFou/+qu/uvPOO21funRpsVhIWi6XX/u1XzsMw6d92qdJ4gFqrS/5ki9pWxL/U3Vd13Xd9vb2Pffc8+AHPxgAJHFZKWWxWEzTVGvlATY2Nj7ogz7o/d///X/5l3/5Yz7mY97kTd7kxhtvnKbp8PAwInhOq9VqsVicOH4csC2Jfy+Ayn8aSYCkF3uxF1uv15kZEZK4zPYwDL/+67/++Mc//n3e531OnToFdF1nOzPb1BaLhW1Jfd/zPCTZBrq+u/HGG9frdWZGBP8W5j9fROzs7Pz93/+9bUmSbJdSbAPPuO0ZP/dzP3fTjTe9zdu8TZTgMknAH/zBH7z92789IEmSJF6Arusy0zb/Y5lHPPIRf/RHf/SMZzzjH/7hH97xHd/x5MmTd9111w/8wA985md+5td//de/2Iu92B/90R+94Ru+4TOe8YyzZ8/ed999d91118Me9rCIuOWWW17ndV7nxV7sxSTxPBaLxYMe9KDWWq2VfwuDeREpwSD+9V791V99b2+vtVZK4TlFREQAtiVlpm3A9jiO3/zN33z8+PGIALquO378OA+QLdP5q7/6q8ePH9/a3kYS/yEAgv98L/ZiL/b4xz9+tVplJveT1Fo7duzY67/e6x87dsx2ZtoGVqvVT/zkT3Rdx78kRcKxEyc2NjeQ+J8tIn7nd37HNpdJ4jJJi8Xind7pnd7mbd9Gocy0bZvL/uRP/kRSREiSxAu2sbHR9/04jpIk8T+PQqdPn37KU57y9m//9g996EO/+7u/++Dg4PDw8O3f/u1f4RVe4V3e5V3+4R/+YbVa/fIv//Lv/M7vtNZe+ZVfeblcrlaraZouXbrUWnvSk54UEbZ5fnZ3d/k3UKJERvxnu/nmm//2b//28PCQf4mkaZr+6q/+6uzZs5/7uZ97/PjxYRhs83yF9g8Ofv6XfvFrvu5rT58+zX8YgOA/maSbb7757NmzBwcHEQEAkiRtbW295mu+5su+3MvWWg8ODn7sx35sb2/Pdmb+1V/91cMe9jDbvAAJlgCFIoSUdsNp8z+Y7dtvv30YBu4nKTOvv/76a6+9FrD9uMc97tKlS9M02V6tVo997GNrrbwI5vN53/dd13GZJEn8j2Hbdt/3d9111xd+4Rf2ff8Gb/AGn/u5n3vhwoXv+q7vunjx4nw+//iP//itra0LFy680zu90yd+4ifu7Oy80Ru90Yd+6Id+wAd8wNu+7du+5mu+5iu+4ivWWnl+MvO+++7j38KQkGD+k91www3AxYsXeRGsVqtf/uVfvueee178xV8c6LsesM3zSPt3/uD3br7l5pMnT0r8KwkEgYV5TgDBfzLbW1tbmNtvv12SJEk8p3Ecf+EXfuGHf/iHv/Zrv9a2pFLK9vY2/xKDoeEEgn8H85/M9nw+f+M3fuOf/MmfrLXyAJJs25Z0zz33fOu3fusv/uIvLpfLaZr29va2t7dLKfxrSJLE/1QbGxuf+Zmf+Yqv+IoPfehDP+zDPuwpT3nKxYsXv/Ebv/EpT3nKLbfc8uZv/uYf8AEf8AZv8AZd1915552ZaZvnJInnkZmr1Yp/i0QG85+vq52kxz/+8fxLnN7Y2Hind3qnz/mczzl79iyAeIHs7/ru7/7QD/vw7e0dI/OvIhAA4rkBBP/JIqLv+4c/4uF33HFHZvKcWmt7e3u/93u/9yu/8iuv+7qv+4qv+IqZ2Vobx/HChQv8S4SEnM5MjJAk/tWMzH8ySV3XvdRLvdTf/u3fZiYPIEmS7XEcP+mTPum1X/u1b7/99r7va63r9foP/uAPlsulbdv8byZJkqTNzU1MKaXWesvNt7zbu73bt33bt336p336+7//+586dSoiJEXE4eHhE5/4xHvuuQewbZsXqtY6TRP/s6XztV/7tR/3uMft7e0BkiRJkiRJkiRJACIibrzxxjd8wzf8iZ/4CUk8P7bT/uu/+1unH/awh5VSAgSCAPHvBBD8l7jpppt+4Rd+4fz58zyncRx/93d/90d+5Ec+4iM+4k//9E9f/uVfXlIpxfYf/uEf8qJJ53oYQPyPN5/Pb7nllrvvvpvnJOncuXNf/uVf/gEf8AHL5fLDPuzD+r6fpmmxWBwcHKzXa140tvkf70EPelCphcue9vSn/ezP/OwP/dAPPeO2Z5RSJHHZ3/z13/zDP/zD1tbWbbfddnh4yItAEv/jCV133XV/+Zd/+YxnPEOSJEk8P5IiYj6fv/d7v/fP//zP8wK01pbL5Td+4zc+5CEP3pjNBQZAvOgMBsA8N4DKf4mHPvShm5ub6/Wa+915552/8Au/8Gu/9mtv9EZv9KVf+qV/93d/9/qv//onTpyQ1Pf9R37kR378x3/83/7t377kS7xklOB5BAAtc8qmdI4TtgXIIBAvOmHxn09S3/ev9mqvdnBwwP0y88d+7MfW63Up5SM/8iO7rvuLv/iLWitQa93e3n7EIx7x13/916/92q89m80k8YIdHBy01vq+538qScDNN99s+0lPetKv/Mqv7OzsvNEbvdGJEyf6vpcESDp37tzv/O7vvP3bvf2DHvSg06dP33vvvVtbW/xLhmHouo7/2Yx3tneuueaae+6556Ve6qW4TBLPj21JW1tbr/AKr/CkJz3pYQ97WCmF55T2n/35n/3D3//9j/zoj25ubAjEZQYhAMwLZzAkMuI5AVT+S5RSJO3u7t58881cduLEiTd6ozd613d9177vV6vVn/3Zn73TO71TRADjON5+++3f+I3f+GM/9mMPf/jDNzc3JfH8CMh8+tOfvlgsJPFvJP6r1Fpf8RVf8c///M8f85jHcJmkt33bt83MiKi1/tEf/dHrvPbrlFIA4EM/9EO/+Iu/+M/+7M/++I//+OVe7uW2t7cl8QIcHR11XTebzWwDkvgf6eEPf/jnfd7nbWxsfMiHfMjm5mbf95K4n+0//MM/fJmXeZnTZ053XXfixInf/M3fPHXq1PHjx3mh+r7f3Nzkf7ZSyvbO9qlTp+6++27+JZKArute7dVe7R/+4R8e9rCH8Twi4vu+//vf4s3f/OSJE5J4IIP49wEI/kuUUqZpWq1W3G+xWNx8882bm5td1x0dHf31X//1bDbjMknTNF137XUf/MEfvFgsANs8X6ELFy8eLZeSbDtt25h/HYP5LyGp1nrhwgXuJ6nrutls1nWdpDvvvPNRj35UKUVSZkrquu4N3uANXu3VXm1zcxOwzQsg6fTp0xHB/2w33nhj13Wv8PKvsLm5OZvNIoL7ZebTn/70++6776Vf+qW72t10000Pe9jDXuIlXuLcuXO2eaGmaTpx4gT/FoEF4r9EREi6++67h2HgRSBpe3t7HEdJPI8nPPGJ58+d++AP+ZDNjU1xmcFg/iMABP8lIqLWOgwD95Mkicsi4qabbtra2uIBosRsNosISbwAttfrYWNjo+86CYUQ//MdP36cF2A+n0uSBNiWBNRau66LCF6o5XJ58803RwT/s0l6/dd//alNESEJkCQpM+++++7v/M7vfL3Xe72tzS2FpmmS9LIv+7IPfvCDI4IXbJqmv/zLvzx+/Dj/RuK/0Iu92IsdHh4Ow8CLoLX2tKc9bTab2eZ5/PiP/9ibv/mbb25uSuIKgUD8RwAI/kssFoudnZ077rjDNveTJEnSqVOnPvuzP3s+n/OvZf7qr//q+PHji40FEiAkxP9gkm6//XZegDd6ozeaz+eSJNVaedHYvnDhwo/+6I9ubm5GhCRJ/E8VES/2Yi927ty5zOQBLl68+JM/+ZMf/MEffPPNN0cJLrt06VLf913X8YKN4/gXf/EXe3t7tVb+LQQCgfgvMZ/PMzMzeRGM4/i4xz3uxV/8xYV4gMz8rd/57T/6oz96y7d8q77reS4CYTAvCoFAmOcEEPyX6Pv+ZV/2ZZfLZWstM3lOpZTZbCaJfyVJf/93f5etdV0vJMT/eLanaeIF6PteEpfZPjw8bK3xIjg6OnrGM57x8i//8vyPl5knTpwAhmHITNvAMAx/+Id/+GZv9mbXX399rVWSpNlsNo6jbV6oaZr+5E/+5B3e4R34NxIIC/Nf45ZbbimlZCYvmnPnzt1yyy0K8Zy+9Vu/9cVe/MXn87kk7mdISEgwL4oAgTDPAyD4L5GZL/VSL/W0pz3t3LlzmWnbtm3bPI/lcvnDP/zDq9UqM3mhLl68+Ed/9Edf9VVfvbm5aWzMv4VR8l/oz//8z8+ePdtas22bF6C1dv78+fV6bdu2bV6wv//7v7d97Ngx2/wPZltI0kMe8pCIsG3bNvDar/3aD3rQg0opXCZpe3t7d3dXEi9Ua+3pT3/6e7/3e0vi38DCAvFf5Q/+4A/29vYign/JNE333nvvarWqtUriAf7sL/7iGc+47Z3e4R3ms5mxcUKC+TcTzw0g+C8REWfOnOm67q677hKybZsXoJRyxx13/PEf//E0TTyLeV5/+Vd/eebMNSdPnaxd5X+Jruve+I3f+PGPf3xm8kJl5nw+/97v/d7Dw0NeqL29vb/8y7/8yI/8yBMnTvA/mySFgIc+9KG/9mu/NqwHp23XWre3t2utPEBr7YlPfOI0Tbxg0zT96Z/+6du+7dtubW3Z5t9IIP6r3H333ev1epomXijb+/v7H/uxH/u+7/u+PI9v/MZveI/3eI/HPOaxiMzk385gAMxzAwj+S0ja2Ni45ppr/uxP/yydvFBbW1vf+q3f+kM/9EPf9E3ftF6vM9P21CYeoLU2DMNP/sRPvs1bvdXmfBEoUKCAAPGvIhz8V+m67rVf+7X/9E//tJTCC9V13Xd/93efPn360z7t0+67777W2jAMtm1zP6cz85577rlw4cKLvdiL1Vr5X+Kaa655/OMfv1qvEBhJknhOD3/4w6dpOnfunG3bgO1xHKdp4n6ttb/927+99pprucy2bf5nk7SzvdP3PS/UMAy//uu/Xkp5mZd5GUncb71e/+mf/ulTn/KUd3i7t9va3OxqrVECBQQEBAQEBAQEBIgXxGBIZMRzAgj+q0TEG7/xGz/lqU85d+4cL1St9UEPetDnfd7nbW1tfdRHfdS3fuu3PvWpT5XEA0j6m7/5m9ue8YxXfuVX7rpOIBD/NgLxryb+TSJia2srM//oj/6IF0rSiRMn3umd3unjP/7j//iP//j7vu/7nvSkJx0dHfEAxuMwPuEJT7j++ut3dnYASfxv0HXdYx7zmIODg9VqZWyb53H8+PGHP/zhX/3VX/3Xf/3X6/Xatu3Dw8O///u/5362z507p5Bt25Ik8T/ejTfdOJ/PecEy89KlSz/2Yz/2xV/8xRsbGzxA13U//MM//Nqv9do72zsRISQQ/xkAgv9C119//eu//ut/67d+69///d9P08QLFhHXXnvtu73bu33Mx3zM8ePH77nnnmmaeE7f/d3ffezYsY2NDUn8uxjMv5r5t+q67j3f8z1/9md/9q//+q+HYWitZaZtnp+tra2bb775Dd/wDV/xFV/xr//6r3d3d3kASfsH+7/6q7/6xm/8xqUU/ld52Zd92Z/6qZ8ax1GSJJ5H13Wv9Vqv9c7v/M4//3M//8mf/Mm/8Au/8JSnPOULvuALjh8/bts2UEp52Zd92ac+9alARPA/m+3WWq31QQ96UCmFF6C1duHChc/6rM/6oA/6oIc85CERwQN893d/9x//8R9/6Id+aO0q/7kAKv+Faq2v+qqvevbs2S/7si97n/d5n1d4hVfY2NiQJEkSz0nSxsbGwx/+8Ac96EFArZUHCMWJEyee9rSnDcPQWpMkSRL/G0TEiRMnPuzDPuwrv/IrX/3VX/0N3/ANNzc2LUeEJJ6TJGBjY+Mxj3nMwx72sFIKzykihmE4d+6cbUn8b2Db6RtuuEHSxYsXNzc3a608PxsbGy/xEi9x8803/8M//MMP/dAPRcS7v/u7nzlzhvvVWl/zNV/z677u697kTd5EkiT+1YwSkv98QqvVarlcHjt2rLVWa+V52L548eKnfdqnnTlz5lVf9VVLKdzP9p133vlN3/RNb/qmb3r69GlJtiXxnwUg+C8kaWNj463f+q0/4iM+4id/8ic//uM//gu+4Av29/cB27Z5HrXWxWKxsbFRSuEBjD/4gz94Pp//xV/8RWstIiTxv0fXdTfccMNHfuRH3nnnnZ/+6Z/+aZ/+aX/wB38wTZNt2zw/pZT5fN51nSQeYHt7+83f/M2/93u/99y5c5lpm//xJEWJruve4R3e4Su/8isvXbqULbmfbaczs7UGlFJOnjz5yq/8yl/8xV/8eZ/3eS/7si+7ubkpSRIgaWNj48KFC3/wB3/Av40SEhmZ/2TGFy9evOuuu2644QbANs9jf3//S77kS1prH/7hHz7rZ7ZtA8B6vf7qr/7qxz72sR/0QR9UawUk8R9AIBDPDSD4LyQpIjY2Nl7mZV7moz/6ow8PD0+ePNl1XWvNNmCb5yRJEs9D0k033fThH/7h3/3d3314eMj/NpJKKbfccsv7vu/7vsqrvMqlS5de6iVfqrXGCyVJEg8gqeu6V3/1V+/7/pu/+ZuXyyX/q1xzzTWf8imf8iM/8iP7B/u2gcxcLpd7+3ur1YrLJEnq+/748eOnTp2az+c8p67rXvd1X/fHf/zH+TcyJCT/+TLzzjvvPHXy1LXXXhsRgG2e02q1uu+++z73cz/3zOkztrnfMAzf+I3f+LSnPe2zPuuzrr322oiQxH8AgUAYzHMCCP5rSQJKKQ960IPe+Z3f+dZbb53NZq017mebF9krvMIrdF33vd/7vUdHR/zbGSX/tSRJioiNjY1Xf/VXX6/X8/m81sq/ybFjx97+7d/+z/7szy5evMj/KpKuu+66t3zLt/yO7/iOJz/5yRcvXjx37txP//RP/+zP/qykiABs80KVKK/0Sq/0tKc9jX8jI/5rRMRv//Zvv8zLvszW5pZtnp9jx4599Vd/9XXXXWcMAKvV6sKFC9/7vd/7N3/zN5/1WZ/1kIc8pJTCfxjxTOK5AVT+y0WE7Vrrddddt16vx3Hsug6wLYl/ja2trXd6p3f6uq/7und8x3dcLBaS+F9FkqTrrrtO0npYb9ZN/k1KKS/90i/9Mi/zMn/5l395ww03SOJ/CUmSbrzxxtd//df/zd/8zcPDw5MnT+7s7LzO67zOfD4HbPMviYgTJ0485jGP4X+8o6Oj8+fPv/Ebv3GU4AWYzWZ93wOS9vf3/+iP/uiv//qvH/7wh998882f/dmf/eAHP1gS/5HMM5nnBhD897nhhhumadrb25MESOJf7+Ve7uUkLZdL/u2ExX8fSQ960IOe8pSnKGQbsM2/hu2dnZ13fMd3/Mu//MvM5H8bSS/5ki/5Du/wDqdOnXrpl37pN37jNz5x4gTPSRIvwNSmu+66a3d31zb/+Wzzb3Xp0qXlcnnttdcCknihlsvld33Xd/3sz/7s673e673hG77hG7zBG9xyyy2S+A9mMBgZ8ZwAgv8Okmxvbm7efPPN9913HyCJf6XW2v7+/hOe8ITDw0P+XQTBv5r4d5MkqZTyaq/2an/7t3/bpgbYBmzzIsvMS5cu/e7v/u59993H/0KSJJ04ceL1X//1/+Iv/mKaJgCQJEmSJF6A1to999zzZV/2ZY9+9KMl8Z9DkiQusw3Yts2/0n333Xfy5MnFYiGJF0yS7YP9g9///d//1E/91Jd7uZfb3t6WFBGAbf6DGczzAVD5byJpsVi89Eu/9K/8yq88+tGPjojMXB4tj5ZHq9Xqvvvu29/fXy6Xy+USGIYBWCwWtm0D0zQdHh7ecccdf/3Xf/1Gb/RGx48f59/O/FuY/yCSHvawh/3Mz/zMu7zLuxSK05aPjo6Wy+U0TXt7e+fPn9/f3z86OhqGISK2trZ2dnYkAa21/f39vb29v/mbvzl37tx7vdd7SeJ/J6Hjx44/49Zn7O7u7uzsSLJtWxL3sw1EhO3W2sWLF//hH/7hV37lV17u5V7uHd/xHfk3M//ZbAO/93u/98hHPnJjY4N/yTAMP/0zP/0mb/ImN9xwgyT+ewBU/ptI6vv+2muv/cEf/MGzZ8/u7e39wz/8wz/8wz887nGPi4jFYtH3/c7Ojm1gmibbkiIiM20Dkl7sxV7sfd7nfa655pqu6wDbkvhf6MEPfrCke+65p+/7Jz/5yefPn7/tttv+8i//crFYdF23tbUlKTMzE+i6zjaXSQrF9s7227zN2zzsYQ87tnNMEv87KbS1vfXqr/Hqt95660033VRKse10Olu21lprLTOHYZim6eLFi095ylP+8A//cGdn533f930f9KAH9X3Pv40FAeI/2e7u7t/+7d++3du9Xa2Vf8l6vf793//9L/mSL4kInpMk/osAVP5bPfjBD7b9tV/7tX/zN39z3XXXvcqrvMo7vMM7bGxsbG5uzufz+XxeSgFs2+Yy27Zt11r5X8t2ZpZSuN9DHvKQL//yL7/rrrskPepRj3rrt37rt3zLt6y1bm1tbWxsdLVDSGqtZct0ArVWQFJESLLdWpPE/2YPe9jDfuVXfuWWW255xjOecXR0JNRau7R36eDg4Ojo6OLFi7feemtm3nLLLQ996EM//MM//Jprrum6jn8XASD+Mw3D8O3f/u133nnnox/96MyMCF6o1tpisTh9+jT/nQAq/622trY++qM/+klPetKrvuqrvuZrvuaxY8d4fiRJ4gWwDUji3868KMR/CNvL5fL8+fOlFElc9uhHP/rJT37yy7/8y7/RG77RNddeExFcdnR0tFwueU6SbG9tbW1tbUmSBEiqtfK/3Obm5h133PGlX/qlZ8+etb21tbWxsXH8+PGTJ08eO3ZsZ2fnIQ95yCu8wis86EEP2trasi0JkMS/nSCwsHgRSOJf7+jo6C//8i/f+I3f+MyZM5JsA5J4fqZpuvfee6+55ppaK//pBAGBhXlOAJX/Pq21YRhuuummm2++eZqm1Wq1Xq8B29xPEmCb50eS7e3t7fl8Lol/IyPA/MsMBkD8OwzD8PVf//VPe9rTuq7jfsMw9H1/++23f+d3fqcxYBuwHRGSbEvKzIiQtFqtHv3oR3/AB3zA9vY2/z6SbPNfxfbu7u4wDDw/0zS9wRu8webm5sbGxmKxmM/nQIkSJQDM1KaIODw8XC6XGxsbm5ub/HuJF5mkiJBkm3+NxWLxWZ/1WfP5fDabrdfrixcvXrx4EZDEZZK4zPYwDF/+5V/+kIc85PGPe7xCPCdJ3O/aa689ceJERPBvJ0mSQDw3gMq/hm3+g9i+dOnSV3zFV9x3332llHEcSymSgMzkfhEBZCbPT9d1R0dHN9xww6d8yqdsbW1J4t/IyLyoBAKB+DdZr9e//uu//t3f/d2LxYL72Y6IiHiPd33Xf/j7f/jYj//4d37Xd/n1X/u1T/2kT37bt3u7T/2MT7ctybYkSRcvXvymb/qmZzzjGS/+4i/Ov49tSbwIxH+ACxcufOInfmJERATPQ9I0TZJs11ptt9YiQpKkiBiGIRRTm9br9Uu+5Et+xEd8xGKx4N9FsiB4Edi2bZt/pfl8/uhHPxoALl64+Nmf/dkH+wdRQpIkQBJg23ZmrlarJzzhCV/whV8QEZJ4AEncb2Nj4yu/8is3Nzf5t7JtK6KCAAwCC0AAlRdZRNher9f8R7B9zz33LJfLL/iCL5jP5rYVkgTY/o1f//VP+NiPAz74Qz/kgz7kQz7r0z/jZ3/mZ4Dv/6EfeuyLPZb72d7d3X3Xd33X93//99/e3uZfLzOnqYWKFBjECxERCmembQAM4l/PthTXXHNtrYX72ZYElFKAb/mmb3rbt3u7zc1NYDabHT9+nOdUa6213n777S/+4i/Ov8/Fixc3NzcjgheqlJA0Ta21xEL82/z5n/+57c/7vM/b2Njg+Xn84x53x+13RMTrvcHrA7YBSVyWmZJWq9Xu7u7bvd3bvcIrvMJrvdZrRYQk/pVsT9OU2RSWzL9EUilluVzu7+9vb29L4t/kaHl08uTJL/iCLzh16hQgCZDEZbaf/rSnf8kXf9Ff/PlfbCwWr/Yar/5Jn/wpOzs73E8h7vcGb/AGwzBsbm7ybzW14W//9i9LqNawEQ8EELxoJO3s7Gxtbf3hH/7her3m3y0igFrriRMnNrc2jx0/trOzs7W1tbW1tb29PZ/PgYj4u7/9262trb/6q7+KCGBjY2N7e3t7e3t7e3t7e3t7e3t7e7u1No0T/0q2W2uHh4cXLpw/eerk1tYibdu2eQG2tzc3NjYuXLiwPFq3ljbPl23btm3btm3bNvcTCpVSCg8gicuEgMPDw8/4tE8TAiTx/Egax5F/B9sHBwe/8Ru/cfr06VIKL9R1111Xqu6887Y777prmjIn/m1aa7PZbHt7e3t7e3t7e3t7e3t7e3t7e3t7e3t7+/Ve+7U/7mM+5g3f+I1Onzn9si/5Un/4B3+QrR07dmxnZ2dnZ2dnZ+f48ePHjh279tprjx07FhG33XabJEk8P7Zt27Zt27ZtLrM9jtNdd919z7137+xszhc9FF4o21tbW2fPnv2Lv/iLzOTfSlJmllIiIiIkSeJ+58+ff/d3eZff+LVff7VXe7UzZ8782I/86Ae+//sbR4koESUkSZIkab1e829le5qmCxcu3nnXraXjEY+6SWEuU6AAACovMtsv9VIv9Yd/+Ie//uu//jqv8zpd10UE/w6tNduSJPGchICXeMmX/NM/+dOnPvWpt99220u99Ev/zV//tcTzsj21aRxHSbzIpmm6tHvpcz7nc/b391//9V9vY3MhYZsXQnqLt3jz7/zO7/iBH/j+937v99nY2Cyl8Jxs8wJI4rL1em2bF8AYeKmXfunf+s3f3NzcBGzz/Ngex3G9XvMAtnkASbwAmbler3/oh35oHMfHPvaxXdfxgtk+efLkS77ki//Ob//pz//8zz7s4ddsbM76Wce/3jiOtnkBhO65+54v/5Ivff03fAP+JZk5juN6vZbE82Ob5yEJGIfx8Ojw67/ha5eH3HjDw1/11V6sm/HCzWazj//4j3+f93mfz/qsz/rGb/zGG2+8sdZq85wMgCRekKOjo9YaL8Av/cIv3nfffW/1Nm/9FV/1Vbbf+PXf4C//4i+e8PjHP/oxjyml8JxsHx4eRgQgiX+N1tq5c+c+93M/e/fi+h3e/o1uuHlLIcwDAFReZJsbm5/1WZ/1Tu/0Tp/wCZ/wRm/0Rq/zOq8zn88BCJ4tef4E4pkMtn3XXXcdHBzYxtgGJAGAJODlX+EV/uav//q7vuM7IuLlXv7l/+av/1qSJJ6T7T/5kz95+tOfzv2kEOKZjHgu8/n8CU94wi/90i89+clPfumXful3fdd3lrC5LHgBIuIDPuD9f/d3f+e7v/t7f+d3fu+lX/qljx8/bpvnZJvnQ1Jw2Wq12j/Yb1OrXQFs8wBCwKd9xqd/6Ad98M//3M8BkmzzPIZh+NVf/dWnPe1pvECSxPPn3Yu7f/03f/2MZzzjYQ972Ed/9EfP53NesIg4efLkh33Yh/7N3/zNL/3KT/3dP/zJq7zaS/d1AzCAAdv8CwL81Kc+tZSQJInnYTybzX7g+7//xMkTgG3btnketsdx/P3f//1Ll/aAzIaRxL9EIdDtt9/2l3/5lxfOX3rJF3v913ndVz92cqOU4IWaz+aPfvSj3+M93uO7v/u73+3d3u3BD35wrR3mOciAEBLPwdxvb2//wQ9+EC/APffcDdx0002ApJtvueWpT33qnXfe+YhHPrKUwnM6ODj4qI/6KBubCGzbAts8gHkmAQIkofWwvuvOO9fr6eVe5jXe673fdWtrjgEwiMsAKi8y4xtvvPG7v/u7P+MzPuO3f/u3f/u3f3sYBgAHz6Lk+bJAPJORJU3T9Iqv+IqAQjwnY+DRj370zs7OT/zYj7/4S7z4zs4OYPO8+r7/4i/+4q7rJPFMIcQzGZkHsG17GIbZbPamb/qmn/Zpn7ZYLGzzL5HY2dn57u/+7i//8i//9V//rZ//uV9xpiJ4AGPM8yUVMDids76WWnjBTp069Wmf+Rkf85EfBdjmedgex/F3f/cP/uD3/5QXLBQAAvNAtautjQq/0Ru90Sd+4idubmxmZkTwQj3mMY/5vu//rq/6ym/83d/9vZ/4iae2sbMBbAO2eYEMgAC7vfqrvxov2Ou87us+49Zbv+1bvpV/yThOf/Znf/43f/04JGdikHgO5nlIAqZpmvWz13qNN37Hd3jPl3/FR/V9SOKFE/P5/MM//MNf8zVf87u+87v//C/+YrVcg0A8k8GAVHgORgYDwHq9uv766zKT5+eaa64F7rjjDi67/fbbgdOnT0cEzyMzn/a0p0f0koJIEuQEzDMZGcACAYAxzlLKQx7y8Nd/vTd5+3d4mzPX7EjBcwOovMhqrcCjHvWoH/iBH7jtttvuvvvubM40YBsEBlDyQBaI55Fud955x5/8yZ/Y5nkIAaWWV3ilV/yNX/v1l3+FV+QFa6199md/9pkzZwAIANsAAsBgMA+QmZubmzffdPNNN99USuFFU2sFTp48+Xmf93kf+eEX77373NHRaJ6DQOL5UggMHB0dfvbnferB/kGUAMA8QMsGLJer132913vt132d3/7N35qm6ejoiOd0eHhYa/cB7/ehr/DyrwHi+RFIPF9dx3yj3vyg644fP8b9bEviBZP0kAc//Ku/+iv29vbPnT2/t7s+PBhLRQKQeGFkMMo/+qM/eNKTn8gLINR13ed+wee/w9u+Hf+S2ax/p3d619d/nbeSxGUSV9gAtnkeAqQInTl97PS1x7a2Z5IkIAEQiBes67qXe9mXf5mXfrmjo6Pl0QrE85B4DgLMM/nue+7+/u//PqBNjefx+q//Bl/+pV/6Mz/108MwnL3vvqc+5SnXX3/9i7/YSwi1qXGFwCCOHz/+vd/7vVsbO4BTNsg8NwMgns02QNf3x49vScIAiOcEUPnX67ruYQ972MMe9jAhA4ANAsCIF9ETnvCEv/7rvz46Ouq6jue0Xq+B9Xr9ki/5Ur/xa7/+ki/1kk964pOA5XJ5eHjI/Wwvl0vbL/9yL//wRzwcwAKMARAABsA8i0lnKQWwzb9eKeW6G05fe/0pJ89F4vkxQjzT0XL5Yi/26Pd+n/fuuh6wGw8w29h4uVd6xc/67M/ispd7pVd82jNufd/3fV+e03K5XK1Wr/+xr/car/0SYADEv4La1CLE/STxLxMgsbOzffz4MRunbUtCiH+RgIu7Z5/8lCesVitJPA/brbVHPfrR7/yu7/rDP/iDwzCuVuvlcgWAeYD1el1rfeQjH/oqr/ZiEldI4jLbgM3zkkACJGxDAhIgEC8KEUXzxWJzc0MSz0sAGMTzWg+r3d3dZzzjGUdHRzw/n/k5n/0NX/d1v/QLv/jghzzkbd7ubd/0zd/8nvvu4QEklVIODw+zcfzYieMnjvGvZYCppSSeW/JMAcg2/37mmcTzZ56DsH3fffd90Rd90ROe8IS+78dxBGz+NZyZkl791V/9wz7sw06ePAnYYCSeg3gW21wmiX+jBPFMCTaAeSaJ55U8QCZPe9ozvvM7vuvue+4F7MbzZYG4QsnzuOWWm9/rvd7zIQ95MBhJAhDiOQgA89wCsG1bKtxPEv8S24BkECSYF1WAb731GR/5kR914sSJUiqAAxnMZZJ4NoF4JoN5gPPnz7/sy77MR37kR+7sbAMKA0K8qIJnE4h/DduYywQGUIJBIEkgDAISDIBAwGq1+sEf/KHv/M7vvHjxIv8OmfkWb/EWn/VZn7W5sQkgAIwNAEaGBEBIIngm82ziuSXPFIBs8+9nnk08B/P8Caczs2XWWqQAMP9KbpkYRK0FsMFIPAfxn8jYPJsQz0M8gIHWXErYAGCeL4tnkXkuFsK2AARIPB8CwDw3AdjmOUniX2QMEs9k/mUCEgBDrFbLYRhAIP4F4pnMc4qIzc1NJxKXCZD4VxD/ZnbjmQSGRAmAQEIgEAAJCUAAIAjbIEn8O9gGwBIg7mcbgESGBCCEQCAAzLMJBOLZkmcKQLb59zPPh3kO4t9OPH8GsAEknk38P2UQ/z2MzQtmBCQIwAJAPIvMcxAPZADEczPPIvGiEv9xEoDkmQRA8EwCwJAAiGcSiP9IyQtkAMSziecmnj+Ayr+dATCAzHMQCIlnMyT/duL5kgEwgMSziedDPJsAMADi/wzxP5S4n7D4F9kgnot5viT+uwXPJhAAYAAkEM8kEP/xgv8UAJV/pfV63XXdsB5uu/32bGlzmUAABsQzCQABYDCYfxOFeH5sY2wAiWdRiOcmDIAQIamUOHPt8Y3FRu0q/xpOp3N5tHrKU55ytFxinoMkxPNnni8lz8UBgAAsZDBKrrBAPH/i+TPPqeW0vbXz6Mc8su97QBIvGtvTlNM03nH7nefOnTMGY0C2eQ7mWWSexQJAABYyGJkHsgAQz4d5TlLwIpOkQNLx48e3t7dPnTpVaw0FgJD4F63X69baX/3VX/3N3/zNwcEeYCwEAUDwLA6UkJA8k0D8t5LEZX3fP+Lhj3jpl3np66+/fhiGruskAZJ4JoDKv4bto6Ojb/7mb/7FX/zF++67z+Z+BkBcYQEgEAgMiQwG8a8niefHNuA0gCTxLJJ4IAPiMkM2ao1HPfrh7/9+7//mb/HmkniRXdq79C3f8i0/8eM/denSpXTyPCTxr2Awz00grrAAZEieSSD+HWwPw9D3/ebm4j3e4z0+9EM/dD6fc5kkXrDWpqc85alf/dVf99u/9buttVplbBsD2OZ5KXkuDp4twGCUPAdh8UwCgcE8LyHE8yeeD01tBPddv7m1+WZv9qbv+77vd+MNN0aIF0Fr7e///nGf8zmf9YTHP9HOZMIAkiAAHDwHI4PBAAT/rSTxnK45c83nf8Hnv/Zrv3Y2VHhOALLNi8B2a+2+++57z/d8z9tvv/0lX/Ilb7nlltlsZgMWMuLZDIB5IAkD4l9JgMTzECgkhaTWmoTN/cwLZNuttUuXLv3hH/7hOI4f+qEf+qEf+qGlFP4lti/tXnrHd3rHZzzjGS//8q94880311J5NiPb2IDBPLfg+Uuev+DZkmcTiOdgMAgEQPLcgufnb//2r5/whCe8yqu8yjd94zdtbG4AkgBJPKfWmqS/+Zu/+IiP+NiL58dbbnnwi73Yw+aLgpAMgAAwLyrxbOb5E/8CA04BKHlu5rkJBAbGcfrTP/mzu+6689Vf9Y0+7uM/9pGPvrnW4IWyfeedd73LO7/7hYvn3vxN3+Y93uudTp06DkDwAJKAUHCFEswzBRb/jYQkYBzHO+6447d+67d+8Ad/sJTuK770G1/jtV6hFBTi2QAqLxpJwzB82Zd92Z133vlxH/dx7/Ee72G71soLY56D+I9m8zd/8ze33vqM13u919ve3uSBzPMnMJk5jMPe3t4HfdAHfcd3fMfDH/7wN3mTN7GReOG+7du+7fbbb/+wD/uw937v957N5hIgnsn8bzOO45d/+Zf/wA/8wI/86I+827u9W9/3vACllPPnz3/d13393u7R277Nu77f+7/7dTecEEKWeIDkuQX/USyegxGAbQAM5l8Q3G+acm9v7yM/8iP/9M9/9/u/78aP+MgPuO764wrxgh0eHn7BF3z+3Xff897v/f4f9/EfNut7Y2QRtrmfJEASmGcyzxT8z9Bau/baa1/8xV/8ZV7mZT7+4z/p677+a1/8Jb7pzLVbPAeAyotsHMff/u3ffumXfun3fu/37vuef1nybALxQtmexla7inkuCp4vm9/4jd/8y7/8y9d5ndfpul7iRTdfzDc3N7/5m7/5jd/4jX/qp376Dd/gjRClBC/YarX6iZ/8ieuuu+6d3/mdt7e3JfECGcyzBf+lkmcLXrC+7z/qoz7qZ37mZ37zN3/zHd7hHfq+5wW76667/u7vHv+QBz/mDV7/9W+6+XREkQAUPEDy3IJ/B9vTNA3DcOnS3smTp2azXuIFSP5lwf1mMxaL2cd+7Ed+1Ed/2K233vbXf/HU13+Tl+mi8IKN43TXnXcfO3bsrd7qzReLOQC2EySJ+0niuYlnEv8z1FqBjY2Nl3mZlz1+/MRttz39tmecO3PtJgDimQAqLxrbBwcHwzC82qu9Wq2V/wSSxml84hOf+GIv9mK8yKZpOjw8lJD41yqlbG1tXXvtdXfffffe/v6JE8d4oXZ3d1trN9xww2KxkMT/CceOHXvsYx97afdSZvKC2b506VKoXHftLSdPni61gXDwn8l2a+1JT3rSt3/7t9s89rEv9j7v815d1/EfJCJuuvmGzc3NixcvXrq036bsusILYHu1Wl64ePGhD33kIx75YAzi/4CuqxuLxeH+8t57du2HSObZAIIX2TRNrbWdnR1eVAEBAQHiX5KZT3vaUz/10z7F5DgNyMjIyLxAfuVXfsVxXI/jYJt/vYjY2FisV2ObGsZpOyF5fsZxbK099KEPjQj+BYKAgIDgv1pAQEDwL5EEjNNomxcqM0G1i52dTalKKFDwnAICAgICgn+fs2fPfvd3f/d7vud7ftInfeJTnvLkP//zP+cFCggICAgICAgICAgInsesX2xtnnSW1tI2L5ik2Ww2jin6+XyGzGWSJEuWJEkSzyYQCAQC8a+RmbazZWbatj0MQ2byH0pS7SJtuROAeDaAyr+SJP5zSNre3r7zzjuHYej7nhfNjTfeuFqtVquVbS6TxL+GFKYBNiAwL5htSZL4vyUzAUm8cJIACQLMf7LW2t/8zd+8wiu8wmMe85jMfLVXe9V/+Id/eJVXeRX+40hFFCzzIhHCBYAGlf80mTlN091333327Fnbe3t7XddtbW29xEu8hCRJ/EcSBgoG8QAAlf8xJG1tbd18881PecpTHvvYx0riXzKsh63NreVyOU0Tl0ni38AB4qr/Ye66667v/u7vvuGGG172ZV/2mmuuedrTnvaQhzyE/3gB4n+S1tpqtfqWb/mWpz3taW/x5m/x8Ec8/Prrr//t3/7tH//xH3+913u9d37ndz579uxtt932hCc84dSpU4997GMf9KAHHTt2rOs6/u0EMogHAgj+J9nY2Hipl3qpX/3VX5XEi6Dru37W11qHYeDfz1z1P8c999zzrd/6rR/4gR/4gR/4gV/91V/993//93fddddrv/Zr83+d7dbauXPnMvMLvuALXvt1XvuWW27puu6lX/ql3/iN3/hbvuVb3vZt3/auu+56xVd8xXd7t3d7ozd6o62trQ/90A89f/58a41/I6OE5LkBBP+TzGazD/qgD/qFX/iF3d3dzLRt2zYvgKRTp04dO3bsD/7gD9brNVf9O0jif4xLly59wzd8w6VLlx772Mc+5CEP+YiP+Iiv/dqvfbM3e7OdnR3+XQzJMxmS/xgBwb+JbdvTNA3DcPHixXPnzl24cOG22277oA/6oDd+4zfe3t62/cVf/MVv93Zv9zEf8zHf8R3fsbu7+zIv/TKv/EqvfOLEiTNnzpw5c+ZhD3vYF37hF37RF33Rcrnk3yjBkDw3gMr/JBGxvb19yy23/MZv/Mbbvu3b2gYighdAUtd1n/Zpn/Y1X/M1b//2bz+fz7nqf79xHH/xF3/xoQ996L333nv+/PlrrrnmxhtvfJd3eZeDgwP+vQxA8kwG899tHMeLFy/+/M///F/+5V+++qu/+mq1+tmf/dlLly4dO3ZsuVx+/dd//TAM7/Iu7/IGb/AG99133xd8wReUWhBXSKq13nzzzffee+9qtdra2uLfQIZEiQzi2QCC/2EWi8X7v//7f9mXfdndd98tKSL4lzz2sY9drVbnz5/PTNv8qxkl//Nky8zMzEuXLl24cOHChQsXLly4cOHChQsXLly4cHBwMAzDOI6Abdu2+d9vmqZ777333nvvfYu3eIvXfM3X/PVf//Xlcrm5uflKr/RKv/7rv3733XfzH0AgAATiv1VmPu1pT/vSL/3Sm2666ZM+6ZPe4i3e4m3f9m0/4iM+YpqmYRge97jHrddr2+/zPu/ziEc84pVf6ZW/8Au/8Pbbb/+kT/qkW2+9dRonYL1eS3rjN37jw8PDYRhs828kLJ4DQOV/mFLKwx/+8KOjo6OjI140x48fz8wnPvGJN95442Kx4P+KCxcv7O/v/9AP/dATnvCECxcucD9JtheLxXXXXfdmb/ZmD33oQ7e2tnZ2djY2NiTxn0n8p4uIe+6550EPetDOzs6jH/3o3/qt33rqU5/6Ei/xEseOHXvFV3zFX/3VX32f93kf/r3EM5n/bhcvXvyJn/iJD//wD7/++utnsxlg+8Vf/MWvu+661tpdd931hm/4hk9+0pM3NjaA2tWXfImX/KIv+qJP/MRP/JAP+ZD3eI/3eJM3eZPNzc1SyiMf+ci//Mu/fNM3fVNJ/GsZWyCeG0DlfxhJJUrXdRGBQfyLIuKt3/qtf/qnf/rVX/3VF4sF/2rC4pkMhuC/XGstImwPw3DnnXf+yI/8yK233np0dPSGb/iGL/mSL2nbdmstMx//+Mf/zd/8zb333nvPPff82Z/9WSllsVjccMMNJ06cOH369Mu//Ms/4hGP2NjY2Nra2tzcBIBSCi+YJACQBAC2eT4E2MYGAzaAxH8gSbfffvvGxkbXdVtbW2//9m//fd/3fadPn57NZi/xEi/xvd/7vbu7u1tbW7VW/i3Es4n/MAZAvGhs25a0Xq//6q/+ahzGG2+8sasd95vP59dee+3u7m4pZTabPenJT/rFX/zFV3qlVzp+/LhCN9xww/b29gd8wAf81m/91m/8xm+8zdu8zWu8xmu01m677bZpmmazGf9qArB4bgCV/2GWy+X3fO/3vNEbvdEN19+AeBG91mu91q/+6q8eHR0dO3aMfzVB8BzEf7lpmmz//d///Q//8A/ffvvtL/uyL/sBH/ABN99084mTJ2qtEcFlrbU3fMM3XK1WwzDs7e3dfvvte3t7y+XynnvuufXWW//iL/7ij//4j4HZbHb8+PEzZ87cfPPNb//2b3/dddfxQkmSxAslAeJZbCwA8R+rtRYRQNd1N9544+nTp3/gB37glltu6brudV7ndf70T//0dV/3dW0DkvjXEc9B/McwAOJFk5nnz5//xV/8xbvuuutpT3vaOIwXL1685ppruEzSrJ+9z/u8z1Oe8pRTp079xm/8xkd8xEf84A/+4G/91m/t7OxsbW1dunTpHd/xHV/v9V7vVV7lVZ7xjGf8+Z//+Sd+4ic+/OEPH8fRtm1J/GvYOAFxP9uAJKDyP8k4jn/+53/+S7/0S9/6rd86n88l8SKIiNOnT4/jeO7cueuvv55/C/PfwbZt29M0/c3f/M2P//iP//Vf//W7vMu7fNiHfdg111zT930pRRL3sx0RGxsbW1tbtm+44YZHP/rRQGYOwzAMw3q9Hsdxd3f3jjvuuPfee6dp+su/+Msv//Iv//Iv/3L+A4j/fE7bjgigtVZrrbX++Z//+YkTJ970Td80M7/0S7/0lV7plXZ2dvhfKDOBe++990u/9EtPnTr15m/+5qdOnfqB7/+B7/qu7/qYj/mYvu8lAf2sf7EXe7E///M/f6M3eqNv+7Zve9M3fdMP/MAPvP3225/+9KcfHR496tGPeuhDHxqK48eOzx45u+mmm773e7/3d37nd97jPd6DfxvLAOK5AVT+x2itnb3v7Hd+53e+7/u+74Me9CCFeJGdPHny5MmTf/7nf/7whz98sVhI4l/B/DdprV24cOFxj3vcn/7pn956662nTp36nM/5nJd7uZertfICSJIkSRL3K6UsFovFYsFlN95444u92IsBrbW3eeu32b20y/8ekoZhKKU4HRGttWEYXuqlXuot3uItzpw+s1wtX/7lX/6222578Rd/cf4Xsn3bbbd98Rd/8ebm5gd8wAdcc801wGu85mt89Vd/9e233/7whz8cACSVUvq+397eft/3fd9v+7Zv+4SP/4SHPexhj3jEIyRJAlprT37Kk/f398+fP79YLFprZ86c4d9G5vkDCP4HsL1er5/y5Kd87dd97du8zdu89Vu/dSmFf42IeM/3eM+f/MmfPH/+PP+zZWZmjuN4zz33/MiP/MjHfMzHfPM3f/PGxsZHfdRHfdInfdIrvdIrlVJ4ASTxr1FKOXHyxEMe8hD+95C0v7cfEU97+tP+7E//7GlPe9rR0dENN9xwzTXXIObz+Wu8xmv8yq/8SmuNfwuDeSaD+Y8hEC9UZtq+ePHi937v977My7zM+77v+545cwaw/djHPvbGG2/89V//9dYaD7BarWqtr/3ar33ixImv+MqvOHfunCTbXPaMZzzjO77jO/78z//8t3/7tz/7sz/75ptvPnPmTCmFfwujhOS5AVT+B8jMc2fPfflXfPlHfuRHPupRj+r7nn+9hz38YV3X7e/v8z+b7cPDw5/4iZ/45V/+5Xvuuefd3/3dX//1X/+mm24qpUgCIsI2L5gk/u9K53pYb29vP+1pT7vrrrvOnDlzeHh4eHgISJK0tbX1l3/5l/fee+91111XSuFfxzybwWD+A4gXwTiOv/qrv/piL/Zib/RGb7S9vQ3YlrS1tfXO7/zO3/3d3z0Mw2Kx4H5Pf/rTbWfme7/3e7/ne77n53zO53zSJ33STTfdFBEA8KQnPQn4yI/8yGuuueat3/qtz58/z7+FIcEoeW4Alf9u4zieO3fu27/j21/1VV/1EY94RK2Vf5P5fD5N08WLFzOzlMK/gpH5z5eZq+XqqU976g/90A/91V/91Yu/+It/7Md+7Mu93MuVUmzbBiTx/9g4jvfdd9+5c+f++q//+pZbbnnEwx9x/vz5m2666c4771yv1/P53HbXdS/zMi9z4cKF66+/nv89WmvPeMYz7rvvvrd6q7cCbHNZZgqdOXMGODo6WiwWXCaplHJ4ePinf/qn991336u+6qtO0/QN3/ANL/dyL/fQhz702muvPXPmzNd+zdceO35ssVjUWl/1VV/1R3/0R4f1sFgs+LcwzwdA5b/VOI533HHHF33RF73FW7zF67zO6/R9HxH8m2xsbAB//ud//nIv93IRIYl/BfOfbBzHu++++4d/+Id///d//6Ve6qW+/uu//oYbblgsFoBtSdzPNv8vLZfLJz7xiT/0Qz/0si/7sg960IN+5Zd/5VGPetSbv8WbX7x48Ru/8RvvuOOO6667brFY9H1/ww03/OVf/uWLv9iL828knk38lzg6Ovrcz/3cV3zFV/zVX/3VCxcuvOu7vutNN91USpEUESdPnHzxF3/xo6OjkydPApIi4tprr93b23u5l3u5o6OjN3zDN9zc3PzZn/3Zz/iMz3jQgx70qEc96o3e6I3e4A3eYD6fA9M0XXfddXfeeec4jRjEv4ZAECCeG0Dlv4lt2/fee+/nfM7nfOAHfuBLvdRLbWxsSOJfYhuQxPOotWZmtuRfy+I/je3VavW3f/u3X/AFX3DDDTd82Zd92YMf/OCu6zKTB5DE/2wS/3lWq9WTnvSk7//+7//wD//w66+/vpb64Ac/+Md//Mf39va2t7evueaa7/7u7361V3u113/915/NZi/+4i/+pV/6pe/6ru9aVQFJvKjEcxD/MQyAeMGe/vSnr1aru++++3GPe9zTnva01Wr1iZ/4iZubmwCin/WllH/4h3+48YYbFQIknTlzZrVanThxYr1ej+O4Wq22t7eXy+UbvuEbvvu7v/vOzk7f94DtiLjmmmsODg6Wy6WxEP86EsLiuQFU/pvYvnDhwg//8A/fcMMNL/uyLzubzXjRTNMkqdbKcxrHsbX2mMc8pp/1/OsEmP80q9XqN37jN77927/9Td/0Td/5nd95Z2dHElBK4TJJgG1AEv8DSOI52RaS+M9wdHT013/91z/1Uz/1gR/4gddff30tdWrTsWPHTpw48bSnPe3Rj370wcHBy7/8yz/kIQ+ptdq+9tprl8vlNE21Vv51xLOJ/zAGQLxgf/EXf/F2b/d2b/Imb/LkJz/5e77ne57whCfce++9D3nIQwBJtdZrrrnmnnvuadnuu+e+cRyPHz8+m83uvffeP/7jP37KU55yeHj44Ac/+O677661/t7v/d5bv/VbX3vttREBSAL6vj927Njh4WFrrdbKv4YUImxsBLYlAQAQ/Hewvbe397M/+7N/+7d/+9Ef/dGz2YwXyrbtzLznnnvuvPPOiOB57O3t2X7Qgx5USpHE/wzr9fov//Ivv/3bv/1DP/RD3/3d3/3YsWOhkMTzkCSJ/38ODg5+7/d+78///M8//uM/fmtr647b72jZaq0R8eAHP/js2bOr1Soi3uAN3uDhD384AGxsbDzqUY/a3d21zf8GmfnkJz/5IQ95SEQ89rGP/azP+qzXe73X+7M/+7Npmris1nrNNdeM45iZf/d3f/dnf/Zny+Wy1vpjP/Zjj3zkIz/iIz7i0z7t09br9U/8xE/UWl/qpV7qh37oh2xzP0nA+7zP+3zjN37jwcEB/xbi+QAI/svZXi6XP/IjP/Id3/EdH/ZhH3bNNddIkiSJF8D2wcHB0dHR133d1+3t7dnmedx6662v9VqvdeONN0riX8dg/hNk5q233vrFX/zFH/ERH/H6r//6W1tbkhCAJEn8v3d0dPQrv/IrT33qU9/7vd/79OnTR0dH3/8D3394eChpNpudOXPmqU996hOe8IRpmubzeUTYBubz+Xu/93vfcccdkvgfz/Z6vb7rrrt+5qd/5nu/93u/7/u+79d//ddvv/32P/7jP14ul7YlSaq12u77/vVf//Xf5m3eZmtr6/Dw8I/+6I9Onjx54sQJSeM4ZmYp5S3e4i0+4iM+IiJ4AEk33XTT4x73uEu7lzD/SgJAPDeAyn852095ylN++qd/+k3e5E1e/MVfnBeBpJ/+6Z+ez+fnzp07duwY5rnY/u3f/u23e7u3297e5n+McRx/53d+p7X28i//8pL4D3J4eFhr7fueyyTxv9b58+d//ud//gu/8As3NzdtLxaLpzzlKXt7e8eOHbN9/PjxxWLxlKc8BZDE/Wqtp0+f/oEf+IGXf/mXl8T/bLZXq9X+/r7xsWPHHvSgB0m67777nviEJ54/d357exsYx3G1WmVma+3SpUtPfvKTl8vlU57ylGEY9vf377rrrl/+5V/+sz/7s9d6rdfa3d29ePHi5uamJJ7TsWPHJK3Wq3SGgn8Fo4TkuQEE/+V2d3d/9Ed/9LVe67Xe+73f2zYvAttv9EZv9LM/+7P7+/uHh4cK8ZwODw+f/vSnb21tSeJ/jGEY7rzzzjd8wzfc2dnhP84v/uIv/umf/qlt27Zt2+Z/p9/4jd94iZd4CeDs2bO7u7u7u7sbGxs/8iM/cuHChcPDw6c97Wn33nvvNddcM5vN1uv1MAyr1WoYhtZaRNx9992ZaZt/hYQEA5CQ/McICF4ASZm5WCze7V3f7d3e7d1e/dVf/VVf9VXf933f98Ve/MV+/Cd+/OjoaLlc/sM//MO3fuu3nj17dn9//+M+7uNms9nx48d/53d+Z7VaffInf/JHf/RHv9iLvdh3f/d3f8M3fMOHfdiH/fZv/zZg2zYPcObMmWEY/vZv/7a1Zpt/hYQE89wAgv9amfkLv/ALj3vc4x72sId92Id92JOe9CReBJJOnjz5vu/7vn/6p386jiPPyfb58+fX6/Xm5ib/Fgbzn+Do6Oiv//qvH/zgB/Mfapqmc+fOjeNom/+FbAOttb29vdls9od/+Icf93Ef9wVf8AU/93M/d/Hixfd93/d98Rd/8e/8zu/8zd/8zV/91V99wzd8w9/8zd+8/fbb//zP//wnf/Inv//7v//7v//7H//4x+/v7/d9v1wup2niX82QABjMfzJJgKTZfAYAtufz+Wu/9mv/8R//8dmzZ//iL/7iZ3/2Zz/90z/9jjvuuO22297hHd7hpV7qpW666aZjx45tbGz0ff+1X/u1r/zKrzyfz0spr/AKr/DEJz7xqU99amby/Nx+++3TNEniXyEBlMg8B4DKf63Dw8O//uu/ns1mf/zHf3zixIlHPOIRvAgk1Vpf8RVf8ZprrnnKU57yqEc+ajafSeIy25cuXQIiwrYk/rVk/kPZbq095SlPOTg4eNmXfVlJ/Me59tpr//zP//zv/u7vbrrppvl83nVd3/dd1/G/hO3VanXfffd9/dd/fa31wz7sw172ZV92a3NLkkLAcrn867/+6z/+4z8+derUS77kS77CK7zCb/3Wb33xF3/xa7/2a585c2a9Xv/Yj/3YX/3VX0XEer3e3NzkXy0AMADiv4Tt1hoASJL0Mi/zMq/8yq/8u7/7u9ecueZd3+VdNzY3dnd3v+VbvuWt3/qtDw4O9vb29vf3T5069R3f8R2nT58Gaq3AxsbG+77v+37rt37rl3zJl0QEDzAMA7C1tSWJf50A4cDiOQBU/gs5vbe391d/9Vdv8AZvsLu7e8MNN2xubtpurWVmKaWUwgvQWrv77rtrrd/7vd/7eq/3ev2sl8Rlktbr9XXXXce/nfiPtlwu//iP/3hnZ2dzc5PnZHu1WgHz+XyaJkm1Vl5k29vbP/uzP/uDP/iD29vbp06derEXe7E3e7M3e4kXf4nZfNZ1nST+Z5P0G7/xG3/wB3/woR/6oddff32tFThaHvV93/e9pI2Njdls9qd/+qe2x3F8q7d6q/39/c/+7M9+qZd6qVprRNg+f/78d33Xd911110nT57kf4n1em1bkiTb29vbb/3Wb/1pn/ZptdbXeq3X2tzcfJVXeZXVcvX5n//5L/ZiL/aEJzxhvV5/zdd8zcmTJzOzlML93vRN33SxWEQE97Mt6Z577pH0Gq/xGn3X868mng+A4L+SaK2N4/j0pz/9Qz/0Qz/lkz8lWy6Xy1/8xV/8mZ/5GUm8ANM4/dEf/dEHfdAHfdzHfdy5c+d+6qd+apom21zm9K233vpSL/VSm5ub/FsIB/+hJF3avfSbv/mbr/Ear3Hy5Eme0zAM3/RN3/R5n/d5e3t7v/iLv/j7v//7wzDwIvvjP/7j93u/9/vRH/3RT/u0T3uxx77Yn/7pn37wB3/wu7/Hu//UT/3U+fPnp2nif7ZhGH7kR37kwz7sw66//vppmoDM/IRP+IRbb72V+z3ykY9853d+51d7tVe75ZZbfu3Xfu2ee+55+MMfXmuVBEg6derUB3zAB3zd133darXiXyEgeCZB8B8jIXnBSim1VkmSJAGSgIc85CGf//mf/9Iv/dJ33nnndddd98Ef/MEf87Ef883f/M1Pf/rT9/f3v/7rv/5lX/Zla62lFEncLyJe7/Ver9YqifutVquv/MqvfKu3fKubbropSvCvYSvTUpEAJNm2DQDBf62zZ892Xffqr/7qN9100+bWZqnlwoUL3/Vd3/XgBz+YF8D2nXfd+XEf93Gv8iqv8sZv/MallN/+7d8ex5H7KfQbv/EbL/3SL911nST+1cR/NNtPfdpTx3F8vdd7vcwEbE/TNAzDMAwR8VZv9VZ33nnnL/7iL770S730p3zKpzz1qU+1bds2L1RrbXd396Ve6qUe+tCHvuEbvuHnf8Hn/+AP/uCnfuqnrtfrL/zCL/ygD/qgf/iHf8hM/gertZ48efLYsWPAx33cx919992ZaXu1Wtnmskc+8pG2H/vYx952223v+Z7v+T7v8z7z+Xy9Xq/X6/V6zWV93z/kIQ/5oR/6If5ny8zZbFZrPTg44DmVUh7xiEd87Md+7Gd/9me/5mu+Zt/3fd+fPHlyNpt9xVd8xSu8wivYBiTxAjhte7Va3XHHHU960pPe6Z3faXt7m38L8XwAVP4LrVarX/mVX3mbt3mbt3/7twciIjPPnz9v+0EPelBmSgIkcT/bR0dHH/RBH/Rqr/pqH/dxHzdN08Me9rB3eId3qLVK4rLlcnnvvfdub2/XWvm3MJj/UJl57ty5l3iJl3ipl3opLlutVj/5kz/59Kc//ejo6C3e4i1e5mVe5mu/9ms/6qM+6rprr3vv937vr//6r/+ar/maUgpgWxLPzziOt95668WLF2+88cZSiiTg9OnTb//2b//Wb/3Wt95662/91m/9xE/8xCMf+cj5fC6J/5Faa7XWiJjNZu/5nu/5BV/wBW/3dm9n+4/+6I8e+chHLhYLYD6f//Iv//Lx48dba5/+6Z8+TdO1117bWtvb29vY2PiCL/iCiJjP5+/zPu/zUR/1Ue/5nu8ZERHB/0gRsVgsTpw48eu//usv+ZIvubm5yXMqpQC1VmC9Xu/v7584ceJVX/VVI4J/STrdPI7jJ3/yJ7//+73/iRMnbEviX8UCgXhuAJX/Kpn5S7/0S8B7vdd7zedzSbaBa6655kM/9EOPHz8uifvZ5rLDw8OP+qiPepVXeZVP//RPb6097WlP29jYeM3XfM2+77nffffdt7GxceLECf7H2N/f//M/+/N3eZd34X7z+fxt3/Ztf/VXf/VTP/VTDw4OvvzLv/zhD3/4Nddc83Ef/3Gf8zmf85M/+ZNPf/rTH/rQh5ZSeMHuvPPOz/iMz/jQD/3QM2fO8AC11lrrox71qEc84hGS+J+t67pz58611iS96qu+6kMf+tA//uM/fpM3eZOf+7mfe9VXfdWXeImXkPS0pz3tTd7kTd72bd8WU2pprbXWSilPe9rT/vRP/7SUYlvSYrF49Vd/9ac+9amPetSj+J/txV/8xX/8x3/84OBgc3OTF8D2XXfd9T3f8z3v+q7v2nUdL4JSyvnd85/7uZ978uTJN3yjN+y6jn8DGSWY5wYQ/JfIzL/927/9nd/5nfd5n/eZz+dAZmam7ZMnT77Ga7xGrVUS92ut/e3f/u0Tn/jE93qv97rppps+8zM/MyJKKffcc89DH/rQvu8lAbZt//Ef//Ebv/Ebb2xs8D/GarW6/Y7bT58+zf2EVsvVj/3oj33AB3zAl33Zl33lV37ln/zJn2TmYx/72E//9E/f29u7++67JfEC2F4eLb/6q7/6xIkTL/ESLxERkrhMEpdJKqVERERI4n8q2+v1+mlPe1prTdK111z71m/91m/0Rm+0sbExn8+HYRjH8e///u8f8YhHdF3X9V1E1Fpns1nXdcBqtbItSdJsNrv55pt3d3fHccxM27Z5YRKSZzIk/zECghfq2LFjrbWLFy/yAkiKiIODg5//+Z8/deqUbV4ErbVf+qVf+pM/+ZPP+ZzP2dnZqbVK4l8tIVHy3ACC/2S2gac//elf8zVf87Zv+7bXXHNNKUWSpHEcf/VXf/UjPuIjvvzLv/zSpUtOc7/M/N3f/d23e7u3e5d3eZfP+qzP4jJJv/d7v/fSL/3SEQEAmXnnnXd+0zd90yu90ivN53P+7cx/qPvuu++GG2649tpruZ/xelin8wM+4AMiYmNj48Vf/MX39vZe6qVearFY3HvvvU9/+tMzkxdgmqZv+uZvsv0Zn/EZJ06c4H+ziHjXd33XP/zDPzw4OLCtkKQnPelJm5ubBwcHH/ABH/D+7//+t91220Me8hDuJ0mSpPl8/pd/+ZeXLl0CbHdd9xIv8RJ//dd/HRGSeFElJBgSzH+Jl3/5l8/MZzzjGfzHsf3bv/3b3/3d3/3FX/zFx48f59/IKMGQyDwHgOA/maSjo6Mf//Eff5/3eZ9XfdVXlSQJOHv27Fd91Vd9yZd8yenTp3/xF3/xr/7qrxCSJAG11td7vdfruu6nfuqnnvzkJ3PZ0dHRPffc8+qv/uq1VgCIiJ//+Z+/8cYbT548mZm2bfOvZmT+g9gex/EJT3hCRCwWC+63Wq1+8id/8gM/8AM3Nzdba9/+7d++XC6/+Zu/+eM+7uN+4zd+41M+5VP+6q/+qrXG83NwcPBZn/VZf/VXf/XxH//xN9xwA//L2X6DN3iDZzzjGZIASZn5Uz/1U6/4iq/4lKc85X3f933f/M3f/FVf9VWPHz++XC53d3dtS5LEZc94xjO+5Vu+5dKlS9kSOHPmzF133SUJkCSJf1lAgCBA/Jc4duzYzs7O7bffPo6jJEmSJEmSJEmSpJ2dnVOnTt1xxx2ttcy0bZvnJzN/9md/9gu/8As/9EM/9FVf9VXn8zn/RsIBgsDiOQAE/8ls7+7u3nLLLS/90i9da33605/+5Cc/+e677/7sz/7sn/3Zn/3Ij/zI93zP95zP5094whMigvvZnqbpAz/wA9/u7d7uEz/xE7ns9ttv39nZkcT91uv1r/7qr37CJ3zC9vY2/2Ps7+//2Z/92Yu92IvZ5n5HR0e/9Eu/tFqtgGma/vzP//z1Xu/1JEmaz+ePecxj7rvvvojgebTWfu/3fu83f/M3P/zDP/yGG27gfz9Ji8Xi5ptvvnDhgm0gIk6fPr1er3d3d0+cOHHrrbc++tGPjoh/+Id/+OEf/uFxHLlfRLzMy7zM3t7eH/3RHxlLkjSbzSJCEv+DLRaL137t1/7TP/3T/f19XgDbx48ff8VXfMU/+7M/4wWwzWUXLlz4jM/4jMc85jFv9EZv1HUd/17i+QAI/pNJ2t/fn81m8/nc9nq9/s3f/M3v+Z7vufbaa3/hF37hTd/0TU+dOrWxsfEP//APtm3bBjLzaU972mu+5mu+6Zu+6aMf/eif+7mfa6399V//9Ru+4RuWUrjfXXfdJelBD3pQV7uIACTxryYc/Aex/Rd/8Rd/93d/9+qv/uqttdZaa20Yhqc+9amv8Aqv8EM/9EO3335713Wf+qmf+kqv9ErjOGZmZj71qU+dpikzeYBxHFtrt99++zd+4zd+3/d93yu8wiuUUvi/4nVe53V+67d+a71eA621t3/7t/+DP/iDn/zJn/y5n/u5Wustt9wCvNiLvVjf93/0R39km8tms9nrv/7rf87nfM7rv/7r11olSeq6rrVm2zb/goDgmQTBf4yE5IWqtb7Zm73Z+fPnL1682Frj+ZG0s7PzaZ/2aa/4iq9YawVsA7Z5gNZaZn7BF3zB677u637Jl3zJxsaGJEn8W0klCKdsANuSJAFA5T9ZZi4Wi7Nnzw7DUEp5+MMf/vCHP/xXf/VXn/GMZzztaU+bpun8+fP7+/vL5dJpFQHAMAw/+qM/OpvNzp87/6hHP+rVX/3VgT/8wz/8vM/7vIjgfj/1Uz/1uq/7uvP5XCH+7QTiP8jBwcEv/uIvvsRLvMR11123Xq//5E/+5OLFi+v1+kd/9Ee/8Ru/8c/+7M8+4zM+423e5m2uueaaYRjW6/VivoiIP/qjP3r1V391zAN1XXfvvfd+4Ad+4Kd92qc9+MEPjgj+D3nYwx727d/+7W/zNm+zWCwi4pprrvmKr/iKxz3ucXfeeeerv/qrb2xsAIvF4g3e4A2+7Mu+7JZbbrn55pu7rjtz5sxrveZrIUopXGZ7tVplZkTwP5jt06dPHzt27G//9m8f/vCH8wJExObm5uu93uv9wz/8wyMf+chaK89JUmZ+yZd8yVOf+tTv/d7v3djYkMR/gOD5AKj8J5O0sbGxvb3dWgO6rrO9tbX1Ez/xE7/xG78BtNa2trZKKQjAdmuttXbNNdd85md+5mw2q7X2fS/p5ptv3tzclATYHsfx937v9z7v8z5vNptJ4t/OYP6DnDt37ilPeconfdInzWYzp1/u5V7uvvvu+8iP/Mijo6OdnZ03esM3aq398i//8t7e3jAMpRTbly5duvfeez/7sz+76zsAmKbJ9tOf/vQP/MAPfLu3e7uXf/mXr7Xyf0vf94985CPPnz+/tbVVSomI+Xz+0i/90i/1Ui8VEZK47MyZMy/2Yi/28z/38x/5UR8JRATBs9i+7777rr322oiQxP9gEbFYLF7xFV/x93//91/ndV7n+PHjkngBxnH8+q/7+lOnT33yJ3/yxmJDIcDp9bD+h3/4h4/92I89PDz8wR/8wa2tLUn8BxAA4rkBVP6TSZJ08eLFYRgiArANfOiHfujrvd7rAdM0PeEJT/iTP/kTLpMkaX9//+EPf/j29nbXddzv4sWLtgHb6/X6b/7mb9br9XXXXVei8D9G3/e11j//8z9/tVd7NRXt7OzM5/NP+IRP+MIv/MK+72utb/7mb/5Gb/RGQGtNku2f+7mfu/3228+cORMRAGC7tfYlX/Ilr/M6r/Oe7/mes9lMEv/nvN7rvd6P/diPfciHfMjOzg4ARATPaT6fv+EbvuEP//APr9fr2WzGc5qm6SlPecqjH/1oSfyPFxHv8A7v8Imf+Il/8Ad/8AZv8AZ930vieUja3Nz83M/73I/+6I9+t3d7t9d49dd42MMftlqt/uZv/ubxj3/8M57xjBtuuOHLvuzLHvKQh9Ra+Y9hMJjnBhD85ytRnvGMZxwdHXHZpUuXfuu3fuvWW2996lOfeuedd95+++1f9VVfNY7j4eEhlx0dHX3Jl3zJy7/8y0cED3D33Xffc889mXlwcPDN3/zNX/7lX75YLPq+R/zPcebMmdd7vdf7gz/4g3vvvReQ1Pf9ox/96O3t7XvvvVdS13Wbm5ubm5s7OzuLxQL4xm/8xnd4h3eICODixYu33nrrX/3VX33913/9K7/yK3/ER3zE1tZWrZX/cyQ9+EEPvuuuu2677bZpmnh+JEk6efLkxsbG7u4uz2MYhqc+9anXXXddZtq2zb8gIXkmQ/IfIyD4l0ja3t5+q7d6qy/5ki/55m/+5jvuuKO1No7jOI7TNE3TtLe39/d///df93Vf9xd/8Rfv/M7v/PIv//LTND391qf/1m/91pd8yZf88i//cq11GIYv/uIvfqmXeqlaq23+YyRKSJ4bQOU/me2NzY2NjY0777zz5ptvPjo6+v7v//4//MM/vOaaa/7hH/4hIs6fP3/+/Pk//uM/Pn3q9Pt/wPsD6/X6j/7ojz70Qz+0lMIDPPKRj/zt3/7td3zHd3zc4x43juPdd9/9bu/6brPZjP8A5j9I3/cv9VIv9ZM/+ZNPf/rTr732WsD2bDbr+/7SpUs33HBDRHA/SX/3d3937bXX3nDDDYDtD//wD3+f93mfT/mUT3nYwx72Td/0TceOHZPE/0WS+ln/vu/7vl/zNV/z5V/+5ceOHeMFWCwWN99883K55HmsV+vf+q3feo/3eI+I4F8heSaD+S8k6aVe6qUy81u+5Vt+4Ad+4B3e4R1e9VVfNSKAZzzjGd/+7d9+8eLFjY2NX/mVXzl79uzNN9/80Ic+9N57793f37/mmmve673e643e6I2+4Au+4LGPeWw6+Y+UYJTIIJ4NoPKfTFLXdY961KPuueee1WoVEa/+6q/+1m/91js7O7aBu+6667M+67O+6Iu+6NixY7YB4Lrrrluv1621Ugr3e4d3eIeP//iPf9M3fdNxHG+66aaNjY3FxiIi+PcyMv9xHvOYx7zaq73aj/3Yj73My7zMfD6XtFgsXuu1Xut3fud3HvvYx9rmMkmZ+Ud/9EfHjh2LCEnAl33Zlz3lKU9prX3ER3zE9vZ2RPB/2mMe85gbbrjhvvvu29raKqVI4nn0fX/ttdfa5jlN0/SM257x6Ec/emtrC5DEi0ogSBCI/xKSgFLK9vb2Ix/5yEc+8pE/8zM/873f+70/9EM/lJmttdba8ePHd3Z2Sin33Xcf8PCHP/xN3uRNWmutNWBjYyMiJCFqqfxHCgAHFs8BIPjPJ+mlXuqlfuiHfuiuu+6azWYv9VIvddNNNx07duz48ePHjh07ceLE9vZ23/enT5/msvl8/o7v+I5//Md/DNi2zWUPetCDPvmTP/nd3/3dz58//+d//udbW1sRwf8wkk6cOPE6r/06T33qUy9cuGAb6Pv+Dd/wDX/6p3+6tcYD2B7H8SVf8iW72nHZdddd9yd/8icv8RIv8eIv/uKlFP6v67ruIz/yI7/6q7/63nvv5QWLCJ7Her3+ju/4jvd8z/eUJIl/BQEg/suVUhaLxc033/zoRz/6NV/zNbe3t0+dOvUqr/IqH/ZhH/YDP/ADP/7jP/6lX/qlJ06ceKM3eqOIOHHixHw+39nZOXHixIkTJ/q+v+22226++eaIkMR/MPF8AFT+S1x//fU33XTTD/3QD33Ih3zIqVOnJHG/ruv29vae/vSn33LLLVy2WCze9E3f9Cu+4its8wBd173cy73cB3/wB3/jN35jrfV1Xud1Dg8PV6tV3/f8uwgH/3FKKY969KMy8+67777++uu5rNbaWrMtCZAE2F6v19vb2wpx2TRNP/ETP/FN3/RNW1tbkvi/TtKJEyfe533e51d+5Vfe8R3fcXNzE5DEA7TWbGfmarWaz+eA7WmaxnFcLBYPfvCD+VcInk0g/mMkAMGLICKmaZqm6ZM/+ZOXR8u+72fzWdd1XdeVUs6fP/+MZzzjvd/7vX/lV35FUkRwP9t333vPS7/sy1hYFMR/EFs2UkgAkmwDkoDKf4nNzc1P+7RP+6Zv+qYP/dAPzUzu13VdRCyXy7/8y798zdd8TUmApK7r2tQk2ZZkWxLQdd0bv/Ebv9RLvdTFixf//M///KabblosFvx7CcR/qOPHj7/8y7/8D//wD7/sy74sUEoppUi6dOnSmTNneIDW2tOe9rTMLKUAv/1bv11rfdjDHlZK4f+HiHjkIx/567/+6z/8wz/89m//9seOHeM5TdO0u7srqZSys7Nz8uRJYL1ef8M3fMN7vMd7lFL4X0XSOI6Hh4cnTpw4eeKkMQ9ge5qmWiuXSeIBWmuz2QwkxH8oWyCeG0Dlv0Tf96dOnfroj/7oO+6449KlS9yvlLK9vb2/v/81X/M1R0dHW1tbtjOztdayhQIBSAIk2Z7P5w95yEMe9KAHPfaxj42IWir/Acx/qPl8/pqv+Zpf9mVf9nd/93cv/uIvDkREa+3g4ODMmTM8wGw2e8pTniLpe7/3e5/4xCf+6Z/+6Uu+5Ev2fc//D5KAnZ2dD/mQD/mhH/qhpzzlKS/90i9da7W9Wq3+4i/+4olPfOITnvCEra2tRz/60Y973ONe4RVe4cSJE5l5dHS0s7Pz8Ic/PCL4X2Wapv39/bNnz7bWaq1CtgFJGKe7rtva3BrHcWtrKzMdEgIMtm0DxiD+o1gAiOcGUPmvIml7e/vRj340IIkHuPfeeyNitVptbW1Jsv3EJzzRNuK5SOKyiJjP5/zHMP/RIuLRj350Zn7iJ37i+7//+7/O67xOZmZma40HKKW85mu+5g/8wA/ce++9d9555wd+4Af+3M/93Id/+IfXWvn/RNLx48ff+I3f+Od+7uce8YhH7Ozs2I6IG264oeu6V3nlV7n2ums3NjZaa13X2c7Mxz/+8S/90i+9WCz438Z2Zl68eHEcx1orIAkAEMaz2WxzZ6t23XyxaJkiCACDATAW4j+QDIB5bgCV/1qSeB4RAWSmbaC19hu/+Ru11syUxP0k8b+EpO3t7dd93df9uZ/7ucc97nGr1eq1X/u1F4vFiRMneIBSysu93Ms9+tGP/uqv/mpJJ0+enM1mr/7qr15K4f+fa6+9drlc3n333Ts7O5nZdd1DHvKQBz/4wYBtSdxP0sMf/vCnPOUpEcG/jgEQAOY/jHiRtdamaTo6OpqmiechVLtuPptTopF9hCRAyLhEHOwfCAnxH8kowTw3gOB/gMzMTC6zvV6vn/GMZ9iOCP7XWiwWL/7iL37ixIk3eqM3uueee37v937vZV7mZba3t23bBgBJGxsbX/RFX/SkJz3pjjvu+O3f/u3XeI3X2NjYAGzz/8x8Pn+1V3u1X//1X7906VIpRRIgSZIkHiAi+r7//d//ff7VDAaDwZBg/gMIxIvA6YgAMpPnSxAqtc4X88OjoymbhQGQ9KhHPurhj3i4JP4jGRKMEsxzAAj+B8jMcRz39vZst9Zsr1arm266KTP5r2CUAOI/UCnlzJkzkv7wD//wF37hF37+53/+4z7u47qukySJ+0l65CMf+aVf+qXz+fx7vud73uEd3qHrOv5fkvTQhz707/7u73Z3d23zAJIkSZIkSVKt9YlPfGJm8r+K8Xw+f/M3f/NHP/rRXdfx/AhqV3e2d3bPXyySkBAg6ZprrrnlQQ/iP4UBEM8BoPI/w8HBwZd8yZe8zdu8zRu90RvZvuuuu17hFV5BEv+bPeYxj/noj/7oO+644wM/8ANf7dVe7cyZMzw/EfGIRzzi8z7v886dO/foRz+6lML/VydPnjx58uRTn/rUBz3oQbxQpZTjx4/fe++91113nST+dcSzif9CERERb/Imb2K773ueH0lbm1s33njj3/7t3z7yEY/gAQwlCv/BBIIA8dwAKv8zbGxsvOu7vuuXfumXvu7rvu4TnvCEzc3Nhz/s4U4rxH81g0H8u21sbLzWa71WRNjmX3LNNddcd911/E8lCQFCPJMEiP8wkubz+Tu/8zs/7WlPG4ah73tesMV88b7v875/+Zd/+cZv/MalFF5U4jmI/wDiX2k+n9vm+ZFUomzvbH/u53zOk574JIEBEP+pBGDx3ACC/wHGcdzZ2XnoQx/ad/3B/sFXf/VXf87nfM7JUycR/yWExX+CiIgIQJIkXqhSiiRJkiRJ4n8cIUBIIEAC8R/r2muvfcYznrFarSRJkiRJkiRJkiRJ6vruIQ99yM/93M+N48i/gkA8k0D8exgQiH89SZJ4TrZPnTr1Dm//9rXUM2fOvPqrv7okLhMIBAKB+I8lHCCeG0DwP4Okxz3ucXv7e9/9Pd/9Ei/xEi/+4i8uSRL/RcSziaueh21sDAYbm/8cGxsbBwcH6/Waf0nf9zfeeOOf//mf87+NJEASz0nSDTfc8D7v8z4CgfivYgAQzw0g+B/g+PHjr/u6r/ujP/qjs9lsuVx+yId8iCRJkiRJ4j+bEsyziaueh23bNja2bP4zLBaLt3zLt5zP5/xLZv3sAz/gA1tr/JcSCMy/jyRegIgISUYAyJZt27Zt27Zt8x9JPH8Alf8O6/V6b29vvV5zv1d5lVd5yZd8yWEYNhYbR0dHy+XSNs+PJC6TdPr06b7vJfHvYjDPJl4oSbb59xnH8b777uPfp+u648eP933PfwkDKNMgAMS/SWaeO3duHEeen8w8efLk3t7e/v6+bZ4fSZlZaz158uRrvuZr8q9gAMQzGYgISdiYf5EgImwk/s2mabrjjjvuvfdewDYPYJvLJEkCMpMXoJRyyy23nDlzppTCv5OMGhjEswFUXmS2+Y9g++lPf/qXfdmX7ezs2LYN2I4I27aBWus0TYAkSYBtwLYkICIODg7e4R3e4bVe67Vmsxn/XokMIEC8UJnZWrMtiX+TaZp+//d//0d+5Ec2Nzf5t2qtHR0dffzHf/wjH/lI/quEQorMBCT+be65555P+7RP29nZKaXwPCS11iTxQtVa9/b23vzN3/z1Xu/1NjY2+NcxAAZaay0zSvAiEtmm9Xqczyvi32Zvb+/zP//zZ7NZ3/e2bXM/SdxPEmCb5yczp2nq+/4Lv/ALNzY2+HcRgAZiggrimQAq/xq2j46O+Pex/Wd/9mfHjh377M/+bEk8gCTbv/5rv/YJH/txXHbtdde+/hu8wSd80ifZ5gFs//3f//13fdd3veqrvupsNuPfxHZmrlarrutKKYgXru/7iLj11ltt8+8wDMPP/MzPfMiHfMiDH/xgQBL/eq21b/u2b/v93//9Rz7ykfxb2QaWy2UphX+JhHE6Dw+XcIJ/q7/+679+8Rd/8fd8z/eczWY8p9/73d/9qA//CC7b2dl57Iu92Ed+9Ec94pGPjAie0zRNT33qUz/kQz5ka2vrdV7ndXhRCYAEwLbHcTjY39+an5zNZlHECyWp6+s999197z1nH/TgG+wmFf71Dg8PT548+Ymf+ImnT58GbHM/ScB99933WZ/+GX//d383DMMrvOIrftwnfsINN9wwm814gNba0dHRh33Yh2Um/1aZ2VoOwxgqm1sVg3gAgOBFduzYsVrrn/7pn+7v7/OvZTAYjFBm9n2/s7Ozvb29tbW1vb29vb29vb29tbW1vb29WCyAl3ypl/ymb/2WzY3NH/i+73/cP/zD9vb29vb29vb29vb29vb21tbWsZ1jXJaZtm3zr9Rau3Dhwj333POoRz1qa2sLg4XFC3Dy5MnFYvGEJzzh0qVLtvn3OX78+Pb29vb29vb29vb29vb29vb29vb29sHBwed81me96Ru90Zu+0Rt9/ud87v7+/vb29vb29vb29vb29vb29vb29vb29vHjx/u+b63xb2U7M++7776nPOUpp0+fLqXwQphrzlzTst32jGc8+clPHYYpm/m32tra2tnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2djYAB75qEd+7w98/2u85mv+8R/90fd/3/dtbm7u7Ozs7Ozs7Ozs7Ozs7Ozs7OwcP358Z2en1vqMZzyDf5PW2jCOT3zSk8+fv3D8xIkbbrg2VHihau1uvvnGu+++7Wd/5tfWq7HlCOZfzzYQEVwmSZIkSQDw7d/6bb/2q7/6xm/6pu/2Hu/xy7/0S1/wuZ93z9338JxKKaWUiHCafxPbq9Xqnnvuvve+e0+fvubFX/LB45ggQJIkAKi8aCRJeumXfuk//MM//OEf/uF3eqd3ms/npRReROZZbLfW+Jdsb2+/7uu+3td/7dcBwzDwPCTZHsdxHEdJXCaJf4ltYJqmc+fOfdqnfVpEvPEbv3GtFfPC1Vrf/u3f/ru/+7u///u//33f9323trYign+95XI5TRMvwKd84if+we//wdu83duG4id+/Mfvu+++7/re7+H5kTSO49HREf96NnZeunTpe77ne6Zpes3XfM2+73nBFDp56uRLvPiL/ekfP+5P/vT3H/nYa6655vTGxlwS/0rDMPBCbW5uvcIrvuKf/smfAF3X8YLZHsfx6OiI52GbZxLPzeDWprNnz37LN31bif7FX/xRL/lSD+r64IXq++6DPuiD/vZvPvoHfuD7H/mom1/ipR4+n22B+Ffa39+fpokXrJQAfvM3fv3FX/wlvv6bvvGN3+RNeAHGcdzd3R2nkX8l2621u+6665M+6ZO6rrz1W7/l1tZW14nnAFB5kW1ubn7e533eu7zLu3z5l3/57/3e773Wa73WqVOnAEm8YLKEeIDW2j/87T+UrnI/25J4Tn/w+3/wqIc/HHjMYx7z8q/wCrZ5AEmIixcv/uqv/mqtVZJtHsA2L4Dtruv+4R/+4dd//dd3d3ff+q3f+nVf93UlEC+c0Pu+7/v+4i/+4g/8wA/8xm/8xou/+Itvb2/zPCTx3MQVBmKchluf/gzbXCaJ+43j+Ae//wcv+VIv+WVf8RXAU5785D/6oz8chqHruojgAWzb/v3f//1z585xmW3ANs/LwXNq2S5euPh3f/83u7u7L/MyL/NO7/ROfd/zQp0+ffrjP+HjPugDP+Tnf/GHfv8Pf/Exj330qVOnIqLWyr/G7bff/gqv8Aq2bUviefzVX/7lYx7xSODMmTPv/T7vW2uVxHOSJGlvb+/Xfu3X7rzzTp5HZgK2IXhuCdxxxx2Pe9zjh7Vf7dVe8x3f+U22jvX8S2az2Su8wsu953u90/d93/d//Cd87KlTx2azGQTPQxIv2P7+/ou92Ivxgn34R37khfMXfvqnf+oZtz7jF37+59/qrd/6Yz7uY2+6+Waex9Of/vT3fp/3LqXwr9Faa62N47i/vz+bzd72bd/mfd7n3bsuJJ4TQOVFJulBD3rQ93//93/iJ37iE5/4xH/4h39orQGSeMFkCfEApZT1MLzJm7wxYJsHsA3YBl7ypV7yEz/5k4GXeumXns1mPA9J99xzzxd90RfZBmwDkgDbgG2en4iYz+fjOC4Wi0/+5E9+szd7s1oqLwKFdnZ2fvRHf/RHf/RHv/d7v/eP//iPp3Ey5rmJZzMIBICwALsZl1J4HpJKLUL33Xff3qVL/WzWppaZPI/MnKbpL//yLx/3uMdxmW3ANs9H8Jy6rgP6vvvAD/zAd3/3d18sFrwIHvXIR/zAD37PF3/xl/zFX/zFX/3VXwGZmZn8a9h++Zd/eV6wRz7qkS/zsi/7Iz/0w4945CNvvOlGSTw/todh+Nu//dunPvWpPD+2ASwQz6N2teu693vf93nrt36b06ePS+JFMJ/PP+qjPuqN3uiNfuInfuK3f/u3d3f3eH4k8YIdHR2N49ha4wV40hOf+LZv/3Yf8VEf+Su//Mtf9RVf+TM//dPv9C7vfNPNN/M8VqtVZkriX0nSYrF4rdd6rXd6p3d67GMf23UdzwdA5UVWawUe8YhH/PAP//Du7u6lS5eGYbAtiRfCEmAwV1j8zE//9MXdi7xQ29vbr/TKr8wLYNv2Qx7ykI/+6I+ezWaS+NewvbGxce21125tbfGvdPz48Q/8wA98r/d6r3vuuWd/f5/nYplnMRgEArC4bBjWX/t1X9ta43nUWl//9d/gV375lz/3sz77137tV2upr/pqr7ZYLHgepZRSyju/8zu/wzu8A8+feGG8sbF17bXXLBZzSbxoosTNN9/8dV/3tQcHh7u7F5fL5bAeogT/Gn/0R3+UmTw/koCtre3P+bzP+9M//pM//IM/+PZv/daP/fiPr7XyPCSdOHHi7d/+7d/iLd6CF8ICQDyTuWxzc+O666+bz+dYEogXhe1Syou92Iu92Iu92Gd91mdJ4l/vtttu+7qv+7qI4AX4xq//ht/6zd98h3d8x9d5vdcFNjc35/M5zykzgZd4iZf4hm/4hq2tLf5NbAOSeP4AKv96s9ns2muvveaaawBJ/Cs5/Vd/9ZcXLl64ePEi95ME2Ab6vn/FV3rFBz34Ibu7u7xgBwcHm5ubD3/4w48fP86/hu1sGSUk8W/V9/2DHvQg/k2WR8vt7e2LFy/u7OxIksQDfOInf9LW1tbv/M7vtKldd931T33KU/7ub//ummuvmc1mPMA0TdM03XTTTY997GN5vixesEwjIvg3kLS9vbW9vSXJtm3+Ne64446nP/3pBwcHwzBI4gGOjo6AaZqWy+WHfeRHfPzHfOz3f+/3ve3bvf21113Lc8rMo6MjSbfccsuLvdiL8UJYPA9D5hQhsMIgnkn857N9dHS0u7vLC/BJn/opx48f/8M//MMf+9EffejDHvYhH/ahO8eOnT9/ngewffbs2YiQxL+VJF4YANnmv4B5lnQ+7nGP++RP/uRhGLhMkiRAEi+aiFgtV5/6aZ/66q/+6rPZjP+BDOL5GsfxV37lV77yK79yNpsBgG2eH9t7ly7tHDsmiefUWjt+/Pjnfu7nPvrRj+a5WLyIZEAS/4Vuv/32D/uwD9vY2IgISfzrScrMS5cuveqrvuqHfdiH7ezs8DwkAZL4FxjMMwnEv8Q295PE82Ob50cSsFqtfvzHf/zrv/7rDw8PuZ8k/jVaa/P5/GM+5mPe4R3eoe97QBL/wQBkm/9atoFxHNfrNfeTBISCF40kxGw2y8xaK/8DGcQLkpnDMHCZbV4A23t7e8eOHeN5TNPU932tNSJ4LhYvIhmQxH+hzBzHcZomLpPEv57T6dza2uJ+tnkASYAk/qPZ5n6SeH5s8/xIAmxP0zQMwziOXCZJiBedyMz5fF5LLbXYliSJ/2AAss1/B9u2uZ8kQBLPj22eH0n8R7AtiX+NNjXbxpJ4FmOel7lMPJPBdkTYBiTxH8fmXyRAtJalRClFEv9r2eZfSRL/QWzz/AjxLOJ52bYtSRL/DrYBbCEkAPEfB6Dyr5eZrbXM5N/Btm3uJwmQxH8hSRFRSuFfIzNba09+8pP/5E/+pLUmiWcSCGwDAsAAJCAABIDBEBGtNUmS+NeQBNgGIsI297NtA+KFs1vmrO9f6ZVf8aEPfWjXdRHBi8y27cy0zf8YtnnRSAIASREhSRL/erZt828lCbDNv4NtQMaYfz1JvDAAlX+l1Wr1y7/8yz/2Yz/2l3/5l8vlEpAkiReZJMB2ZnI/SYAk/jVs82/Vdd2jH/3oV3/1V/+QD/mQjY2NiAAk8YLZzswnPvGJn/mZn/m3f/u3mSmJ52SDxTMZGZB5ljApAIsrJPGvJMk2UEqxzfMXPLfkfpkJtNZe4iVe4rM/+7Nf/MVfvOs6/iW2W2uHh4e/+Iu/+Kd/+qd7e3v860niv5UkwPaZM2de7MVe7M3e7M22t7drrbxonJ7aNI7j2bNn9/b2eE4CDGAsY3GFef4EgADEA1i8KGSeizGXmRfINrC5uXnTTTfVWiMiIng+AGSbF9kwDJ/7uZ/7/d///cePH3+t13qta6+9VpIkLpPEi0ZSZtrmOUnifkIIwLYknh/b/CvZ5rLVavVbv/Vb586de+hDH/o93/M9J0+etC1JEi9AZt53331v93Zvd/bs2Xd/93d/jdd4jeuvv14Sz8kWz2SweAAjsADMM0niX0MSYFuSJNs8f+K5mftJuuuuu37t137th37oh86cOfOTP/mT119/vSReKNtPfOITP+uzPuvP/uzPaq233HJL13WtNf41IoL/VpIA4NZbb12v1y/3ci/3OZ/zOY997GN5Edje39//qq/6qh/7sR/b39+vtfKcZGwDdsoYEMkLJBAAUsgIUgAWLwqZ52InkLwwkrhsc3Pz9V//9T/jMz7jxIkTEcFzA5BtXjTjOP7+7//++73f+73Kq7zKl3/5l19zzTURwb+VbdtcZnu9Wj/+CY+/7777Tp069bCHPWxnZ0eSJEnjOPZ9n5kRwXOSxL+JbdvL5fL7vu/7vuqrvuod3/EdP//zP3+aplKKJF4Apz/6Yz72N37j177wCz//jd7oTbqukySJF848k3k2gXi+bGdmtjx3/typU6dKKZJsS5LE/WwDkiTxb2I7W/727/z2R33UR73Ga7zGl3/5l29ubvJCXbx48RM/8RP/8A//8H3e530+4AM+YGtrKyL43ykzL168+H3f933f/u3f/pqv+Vpf/MVfdOzYMV4w28Du7u4HfeCH/NVf/dVjHvuYV32VV+36wnOSkQQIBAaEwTx/AkAIEAAGwOKFsM1lMs9NXGH+Ba21Jz3pSb/1W7918uTJ7/7u737sYx/LcwOovMiGYfiFX/iFxWLxuZ/7uddccw0gif8IZ8+e/ZRP+ZQnPelJ1157bUTs7u7efvvtpZSXf/mXf5u3eZtrr732mmuueehDHwo4rRD3sy2J52GbF0ASIEnSYrF453d+51/8xV/8lV/5lc/6rM/quo4Xarla//mf//m11177Kq/6an1fQZL4FwkMgMAAiBdCUmZ+3ud/3p/8yZ/ceeed3/iN3/jKr/zKtVZJgCTb3E8S/1aSFHrVV33VU6dO3XbbbdM08S/5m7/5m7/5m795vdd7vQ/8wA/c3t6WxP9apZRTp059yId8yLmz537xF3/593//D9/0Td9YEi+ApOVy+QM/8IN/9dd/9Uqv9Epf+ZVfeezYMZSSeAAZSQghgQGweCHEFQIE5jLxQtgGbIvnJQnzghljICJWq9X3fd/3ftVXfcXnfd7nfcd3fOfGxoLnAFD517j11lvPnDlz4403SgJsA5L4N8nMaZx+67d/60d/9Eff/u3f/jVe4zXm87nt1tpTn/rUu++++x/+4R9+9Ed/9OzZs3fcccervuqrfuqnfuott9zSlQ6wLYkXICJ4AWxzv1LKbDZ75CMfeccddzzjGc94+MMfzgt1dHg4DsPx49tdV/hXEQAG8UzihXj84x/fWvue7/meJz7xiV/2ZV/2RV/0RY9+9KMjIiL4DxURi8XitV7rtX7nd34nM/mXPOUpTxnH8bVe67W2trYk8a9n27akcRxDUWtFPAeTzsy0XUqxLUkSIIn/UBKzWfdar/1av/RLv/Z3f/Ok13vd158vOl6waZx+53d+dz6bf8iHfvCZa05J4gUxzyaer8wcx3Ecxz/4gz94+tOeNqyHt3iLt3jIQx+KeJZpmkoptiXZjggewDYAyDxbiBfZxsbi7d7ubX/kR37wqU++/RlPO/+ox94UwQMABC8yScDJkyf7vuffTdLFixe//Cu+/Hu+53u+8iu/8vVf//W3trZqrbXWruse+9jHvu7rvu6HfdiHfdu3fdvP/MzP/MAP/MAjHvGIj/7ojz48PAQkSQIkSZIkSZIkSZJ4wSRJkiQJiIibbrppmqbM5EVgBAaDwfyrCAQC8cLdfffdL//yL3/s2LFXfuVX/sqv/Mqv/dqvvXjxImDbNv/RFovFMAy2+Ze01mqt11xzTUTwr2db0sHBwU//9E9/y7d8y7333ZtOnlPLdvfdd3/VV33VR3/0R//6r/+67cy0DdjmP56uuea6EnVvbz8zeaHGqY3jeP0N17/0S72UJF4IgUAgXhBJpZTf+q3f+vu/+/t3fbd3e6VXeeUv+bIvffwTHs/9Wmu7u7t//dd//Q//8A8HBweZyQtgYWFh8a+k2Wz2sIc99ODw6MlPuk3iOQEELzLbXBYRkiTxb5WZ586d+57v+Z6f/dmf3d/f39/fr6U63Vo7ODgAJEVEidL3/WKxeOmXfumP/diPvf7663/8x398HEdAkiTbtm3btm3btm3btm3btm3btm3btm3btm0bkCTJtm3+BcIFAkDmP01E3HfffU73ff+gBz3o/Pnzf/3Xf71arWzb5j+abUm8CDITWCwW/JtkZmvta77ma+64445XfdVX/eqv/uq9vT3bPEBmfv/3f/+ZM2fe673e62d/9mef/vSnRwT/mUopipJp8S8oJYZhsHM2n2fixImTf7P9/f2//uu/fpu3fZvNjc2XeImXeMVXfMVP+7RPG4aByw4PD7/lW77ly7/8y7/7u7/7q77qq4ZhcJoHkCRJkiRJkiTZtm3btm3btm3z/Eiqtd50882CaZqy2bZtngkg+C+XmU9+8pM/+ZM+eW9v78u//Mtf/MVffGtrCwB2d3c/8iM/8uLFi3/yJ39y7733IiTt7e1FRN/37/qu7/ojP/Ijly5dWq/X/CewzQtn7if+Mz3olgf90R/90TiOmIj4qI/6qK/8yq+89957AUlcJkkS/xFs86KxDdjm38T23t7e/v7+e7/3e7/4i7/4G7/xG//mb/7mwcGBbe537733Pu1pT3v913/9l3qpl3rZl33Z3/3d3x3Hkf80kgDbvGhsREgS/16Slsvlfffdt7m5qdB8Pn/d133dWurtt9/O/S5cuPBN3/RNn/EZn7G5uflVX/VVR0dHtm3b5l/JNi+ABEiSJEmSeCaA4EVmG7BtW5IkSZL4VxrH8ad+6qee/JQnHx0dfeM3fuMnfMInbG1upXP/YP/v/u7vnva0p73927/9r/3ar33pl37pwcHBU57ylE/6pE+ybfuVX/mVM/Mbv/EbW2sAIEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJEmSJCkzbfNCGBtbTmGJIon/HA968INms9k4jemMiMc85jHr9frv//7vp2kCJEniv4lt/q1qrbXWiCil9H3/6q/+6j/90z/9tKc9rbXG/cZxvPnmm7e2tmaz2Wu8xms8+clP3t/f5zJJ/CeQkCSJf5kheQAFCl50tm+//fa//du/vXjxYmZi9vb29vb2pmmSdNONN336p3/6T/3UTw3DMI4jcOrUqZ2dnWPHjn3gB3zg4x//+N/+nd/OTF4oSZIkSZIkSZIkSbxA4vkDCP41bPPvVkv9kA/+kE/91E/9h3/4h6c97Wnv937v9wd/+Ac/9mM/9sZv/MYf//EfP47j137t137QB33Qk5/85Dd90zf9+I//+E//9E/nsvl8/h7v8R4/+IM/+Jd/8ZetNf77if8cXdfdcMMNd9xxR2ZKOnHixJkzZ/7u7/5uHEdJXCZJEv+1bPNvZdv25ubm4eFhmxrQ1e6t3uqtvviLv/jg4MA2l2WmbUnAddddt729/Q//8A/81xD/IoEkSQoU/Gv9zd/8zcd+7Md+xmd8xkd91Ec9/elPny/mD3rQgw4ODrquE4oSx08c/8u//EvbmWn76OgIkDSfz7/kS77kN3/zNwHbgG3+AwlJEs8JIPgvV2o5duzY67zO6/zQD/3Qj/zIj7zjO77j2bNnp2m65ZZbFovFF37hFz7mMY8ppZw4ceI1X/M1v/u7vvv6667nfm/xFm9x6tSpb/nWb1mtVrYz07Zt2/xXEBYE//le53Ve53d+53fGcRyGobX2qZ/6qX/+539+6dIl/rvZ5t9EEhARd955Z2sNUOiN3/iN9/b2zp49O00Tl81mszvuuGO5XK7X68Vi8bIv+7Jnz55trfE/hEAJ5l/J9nq9/vmf//kv/qIv/rZv/bZ3eZd3+bIv+7K777770Y9+9C//8i/bbq0BW1tbpZS77rqr1mr7qU996uHhoe1Sy+nTp6dx2t3d5b8OQPDfQvR9f/z48VtuvuUd3v4d3vzN3/zVXvXVHvKQh3zHd3zHH/zBHwCZ+bIv+7IXLlw4efKkJC6zferUqfd+7/f+y7/8y9/5nd9Zr9eAbf7rGIDkP5ntF3/xF/+Lv/iL3//93//QD/3Q937v9/6ar/mag4ODixcv8r+ZJOC1Xuu10mkb6Pv+sz/7s3/iJ37i4OAgM4GdnZ3jx48vl8uIODg46Pv+93//9/f29qZp4n+5w8PDYRi2trbOXHPmtV7rtR72sIf90i/90sMe9rC77rrr3Llzh0eHT3jCE37wB39Q0vd///cDpZSNjY2/+Iu/kASUUl7rtV/rqU99Kv91ACr/rRQqKkfLo1//jV9/x3d8x5MnT/7O7/zOHXfccccdd+zt7R0eHl64eOH48eOFwv3e5V3e5Yd+6Ie+9Vu/9dVe7dX6vue/mBIlJBgM4gWzPU3TxYsX/+Zv/uahD33ogx70oFKKJP4lknZ2dt7qrd7qcz7nc778y7/8IQ95SET8wA/8wG/+5m8+9KEPXSwW/G/2mMc85jd+4zfe4A3e4Jd+6Zf+8i//8o3e6I1+4zd+493f/d2PHz8ObGxsnD59+vbbb//e7/3eW2+99eDgYLVaXbp06cSJE/wv1/f9wcHB1KbMnM1mL/VSL/WHf/iHp06d2t/fP3fu3O/8zu8cHBy853u+56u92qt92qd92qVLl2az2ZkzZz7t0z7t937v96ZpKqVcf/31j3/841/u5V4OiAj+0wEE/60kXbx48Qd/8Ad/7ud+7rrrrtvY2HilV3qld3u3d/uqr/qqr//6r6+1/sEf/IEkHmBrc+szPuMznvzkJ//Gb/zGOI78V5IhwZBgXgR33333p3/6p8/n88/4jM/4vd/7PV40kkop11577fXXX//IRz7y5MmTp0+ffsu3fMvf/d3ffdzjHpeZ/G/20i/90r/+679+6dKll3qpl/rwD//wW2655ZZbbrlw4UJmAqWUBz/4wffdd9+HfdiHffEXf/GXfumXPupRj/qlX/olSfwvN+tni8XiyU9+MiDpxV7sxZ7+9KeP4/gmb/Imv/iLv/j2b//2H/zBH3zNNdc85CEPOXXq1B133BER11xzzf7+fmbu7u4+5SlPycy///u/578OQPDfx/a5c+e+8Ru+cbVaTdNUa93Y2PiMz/iMV33VV33EIx7xyEc+8jVf8zW//Mu//ODgwDYASEK86qu+6tu+7dt+2Zd92VOe8pRpmmzzH0BY/MsM5kUzDMNv/dZvveu7vuvLvuzLfs7nfM43fuM33nnnnbzItra2uq4rpUgCbrjhhjNnznzLt3zL3t4e/5udPn366Ojo3LlzD33oQ2+55ZZHPvKRH/MxH/OjP/qjR0dH6/U6M1/mZV6m1nr8+PEzZ87cfPPN7/7u7/6Upzxlb2+P/81s166++Iu/+H333Xfbbbd993d/90//9E/v7e394R/+4c7OzuMe97j5fL61tSVpa2vrEz/xEz/rsz7r4ODgmmuuiQjbf/VXf5WZf/AHf7C3t9da478IQPDfZ71e/8M//MN9Z+97vdd7va7rgIiYz+e1Vkld133gB37gYrH4jd/4DR5A0nw+f9/3fd/1ev01X/M1y+VSEv8BBOI/VGttvV5ff/31i8Xi+uuvf6VXeqVf+IVf4EUgSdLGxsbBwcGwHkoUSYvF4q3f+q1vvfXWJz3pSfxvJuld3uVd/vqv/zpbSgIe9KAH3XnnnY9//OMjIiI2Njbuuuuuw8NDSbXWhzzkIddee+2FCxf430wS8PIv//J/9Ed/tFqtXumVXulVX/VV3+md3un2228vpWxsbEjislLKNddcc3h4eM8999iepknSq7/6qz/ykY98n/d5n52dnWmabNvmPx1A8F/Odmau1+unPOUp3/Vd3/UhH/IhG4uN1lprjcvGcVwul+fOndvf36+1Pv3pT5cE2M7MzJym6YYbbnjLt3zLv/u7v7v33nszk38XgbjC/AeqtW5tbT3hCU9orXVd93Zv93Y///M/v7u7my15AVprrbXW2jAMJ06cODw8PDg82L20+9M//dNf/dVf/Qd/8AeXLl36y7/8y8PDQ9v8N7HNv88rvdIr3XP3PcM42Aa2trbe+I3f+Jd+6Zf29/eBEydOZObBwUFrLTMXi8VjHvOYP/mTP+F/M0mSjh8/3nXdYrF4zGMe81Iv9VKv/dqvfe+9967X6+VyeXh42Fo7ODi4/fbbn/GMZ0zT9Hu/93t93wOZubGxERE7Ozsv/dIv/Ud/9EfL5ZL/CgDBf4dhGP7mb/7mcz7nc970Td/04Q9/+Nb21nw+P3fuHGB7HMcf/uEf/qiP+qiv+ZqveeM3fuO+77nM9nq9Xq1Wtmut7//+73/ttdf+4A/+4Gq14t9FIP51BOJfUmt99KMf/Q//8A+2SyknTpxYLBa/9mu/NowDL4Dtv/3bv/2ar/maz/7sz/7SL/3SixcvPv3pT7/11lsXi0Wt9XVe53Xe9m3f9nd+53eOjo74b2Kbfx9Jp06duvmWm20DgO3Xf/3XH8fx7rvvBubz+Ww2e/zjH3/hwoWLFy8C1193/VOf+lT+4wnEf6ETJ04cP378cY97HCDp+PHjb/zGb/zEJz7xwoULf/mXf3nfffd9zud8zsd93Md96Zd+6aMe9ajf+Z3fefjDH9513d133y1JUtd1r/Var/X1X//1tVb+KwAE/+Vs7+7uft3Xfd199913yy231Fq3t7Zf8RVf8fd///dt2+77/q3e6q0+/MM//MM+7MPe4R3e4Xd/93fvvffe22+//Z577vnDP/zD3/md3wEk3XDDDR/0QR/053/+58vlkv9A5oUSiGcSL1REnDhx4klPetL+/n5ELBaLN3uzN/uTP/mTiACAaZqAzLRtOzP/8i//8uu+7use/OAHv+mbvulbvMVbvN/7vd/v/s7vnjlz5uTJk2/2Zm/2Kq/yKu/6ru9q++LFi/wv91Iv9VJ/+qd/Oo4jIGl7e3uxWJw/fz4zgTNnznzf933f93//9991110RccONN9herVa2bfO/Vq31zJkz+/v74ziePXv23nvvveaaa57+9Ke/0Ru90V/8xV8Ar/Zqr/ZRH/VRX/iFX/jxH//xFy9e3NzcvO6662677TYAkHTttdf2ff/4xz8+M/kPIwDEcwOo/Jc7PDz81V/91dd5ndeZ9bNf+ZVfefmXf/nZfPZmb/ZmX/u1X7u3t7e1tVVrPXny5Gu8xmsAR0dHT3ziE7/sy77s3Llzkk6cOPFxH/dxtiUBL/dyL3d4ePjXf/3Xr/Zqr7ZYLPjPZoFAIF4Etk+cOPHgBz/44sWLJ0+e7LruDd7gDX7kR35kf3//5MmTQGbu7u7u7e3ddttt99xzz87OzhOe8IRP+qRPeuhDH1prtf2SL/mSH/ERH7G5sfnyL//ykjLz1KlTN9xww5Of/OSHP/zhkvhf65prrvnFX/zFV3u1V+v7Hqi1vu7rvu5v/dZvvcRLvMSxY8euv/761todd9zxxCc+8eEPf/hisbjnnnsuXLhwzTXX1Fr53+xhD3vY3t7eM57xjF/7tV/7q7/6K9vXXnvtdddd92d/9mfz+fzN3/zNSymSdnd3NzY27rjjjlOnTp09e5b7lVJe5mVe5o//+I9f/MVfvJTCf5gA8dwAKv+FMnOapj/5kz/58z//80//9E8/e/bsr/7ar547d+7aa6990IMedN111/3Jn/zJ673e6/EApZTHPOYxGxsbX/gFX5jOra2tnZ0dSQBw4sSJ13u91/var/3al3u5l5vP55L4tzCYZxEvlEC8yBaLxUu/9Ev/9m//9kMf+lBA0qMf/ejbbrvt1KlTti9cuPBlX/ZlN95440Mf+tCTJ08+4xnP+I3f+I33eZ/3qbUCrbXt7e0HP/jBy9XymI9JKlE2NjZuvvnms2fPjuM4m834X+vkyZNHR0fDMNjmsoc85CFf93Vf9+d//uev+Zqvec011wAv8RIvcd1115VSZrPZ67/+61+6dOn666/nf6fMjAjbD37wg//kT/5kc3PzMY95zMu+7MueOXPm1ltv/YEf+IFjx46dP3/+5MmTXDabzV7lVV7ll37plzY2Ns6fP3/x4sXf+I3fmKbp4sWL+/v76/Wa/0AWBovnBhD8F5J0cHDwvd/7vTfffPOxY8duvPHGBz/4wY973OOAjY2Nt3zLt/yyL/uy1WrFc3qlV3qlX/3VXz195vRNN910/PjxiJAEAKWUN3qjN9rd3T08PJTEv5HBYGTEv0RYELxoaq2Pecxj/vAP/3D34u6tt976gz/4g1tbW7/yK78yjmNr7Wd/9mcf9ahHvf/7v/9bvuVbvs7rvM47v/M7nzlzprVmG6i1SnrLt3zL3/iN3xiGQRICeI3XeI0/+qM/Ojw85H+zrute5VVe5WlPe9o0TYCknZ2dG2+88Vu/9Vt3d3cXi8X29vaLvdiLvfqrv3rf9xsbG2dOn/nFX/zFaZr430nSMAzjOG5ubv7lX/7larV67dd+7Vd+5Vd+2MMe9mIv9mLL5XJ7e/vP/uzPuF/f96/2aq/2R3/0R/P5/K/+6q9+53d+56677rr11lt/9Ed/9FVf9VXPnz9vm/8YAYIA8dwAgv9aT3rSk2677baHPOQhfd9vb2+/+Zu/+S//8i8vl8uIuOGGGyTdfffdPECt9XVe53UuXLjwF3/xF7ZtA5nZWuOy666/fhjH5XLFv5fB/IssEADiRSBpY2Pj7NmzT3ryk3Z3d1/+5V/+fd/3fYGnP/3p0zQ97nGPe+u3fuvt7W1Jtvu+v/HGG7/jO77jV3/1V//wD//wD/7gD/7wD/9wuVz+wi/8wn333bd3ae/w8HAYhmPHjt17771Pf/rTW2v8rxURj3rUo/7hH/5hvV5zWd/3r/M6r3Pp0qULFy5sbm7ecsstFy5cAICIuOnmm/7hH/7hSU96Umba5n+VzLzrrrt+5md+5lu/9Vuf/OQnl1Ke9KQn2QZs7+zsvPVbv3Vr7a677spM24Ckhz/s4aWUhz/ikbfdfsdrvfbrXHPttddee+3HfuzHvuEbvuHGxsbh4WFrjf8Y4vkDqPwXWi6Xv/d7v5eZe5f2xnHs+/4Rj3hE3/dPeMITHvzgB//Jn/zJer2+9dZbH/7wh3O/UspLvuRLvvzLv/y3fdu3vcRLvEREZOb58+drrdddd12ttdY6Tu38xYsPy4wI/nMJBALxIpBk+9ixYy//8i///d///V/7tV8LjOP4Hu/xHj/8wz/8Xu/1XpcuXer7XhIgCXjZl33Zn/qpn3rxF3/xO++48/d+//ee9KQn3XLLLRsbG1/xFV9x6tSpjY2NRz3qUX//93+/v7//Ez/xE494xCO2t7cl8b/TqVOnfvd3f/dN3uRNNjc3AeDlX/7lr7/++r/+678+derU3t7e4eEh97vmmmte8zVf89u+7du++Iu/eDab8b9HZrbWfuzHfuzRj35013U//dM//Q//8A/7+/snT5588IMfnJmZ+chHPvKXfumXLly4sLe3l5m11mEYdi/tnjx58mEPf8TTvuVbVqvVYx774jW4+eabI+IN3/ANf/iHf/i93uu9NjY2+A9gnj+Ayn+hvb29vUt7H/zBH/xbv/Vbb/Kmb3LddddtbW299Vu/9Wd/9me/7uu+7vXXX3/LLbc87nGPe/3Xf30eICLe673e66M+6qP++I//eLVa/fVf//XTnva0re3tz/iMzzh9+gyKUro777xrepmX7iP4H0bSYrF4lVd5la/4iq/IzFJKLfXEiRN33XXXn/zJnzzsYQ/7u7/7u9d4jdfgfq/6qq/68z//849+9KMXi8VNN9/00z/90x/5kR85TdMznvGMrut+9Vd/9fM///M//uM//iVf8iW/7/u+b3d3d3NzMyIk8b9NZm5vb994441Pf/rTT548WUqRdOLEiXd+53f+tm/7tqc85Sl/8Ad/8JIv+ZLr9brruoiYz+ev8Aqv8Iu/+IsXLly4/vrr+d8jM/f29lar1Su8wiscP3781V7t1f7sz/7sL/7iL776q7+6lNJa293d3drauvvuuyPiK77iK2azWUTs7e0fLo82trZ/7ud/bmrtK77yq6699rps47u88zvN5/NXf/VX/9Zv/db3eq/34j9AApBgnhtA5b/QsB4e9OAHveEbvuFqtfqFX/iFt3u7t9vY2Hj4wx/+4Ac/+Nd+7de+/du//fjx4z/1Uz/Fc4qI13md13nsYx/7pV/6pW/8xm/8Fm/xFmfPnt07OJzPF4ISxel77r7bmfzPI6nrupd4iZfY2NjY39/f3t4GSilv9mZv9v3f//2f+Imf+Au/8Auv/uqvLonLjh07duONN37TN31T3/e7u7uf+ImfeO2115ZSbrnlFuDhD3/4vffeu16vX/EVX3F7e/tv/uZvrrvuuoiwLYn/VVprl3YvXXPNNT/1Uz91ww039H1/cHBw4cKFcRz7vl+tVq/7uq/7R3/0Rw9/+MMf85jHbG1tAddff/1Lv/RLP+lJTzp9+nTf9/wvUUq54447HvnIR25ubkbE6dOn3/iN3/g1XuM1Ll68ePfdd3/CJ3zCLbfccubMmfl8fuedd956662v9Eqv9Iqv+IpbO8cWiwVRPuD93u/U6Wve5d3e7aVe4sXB+5cu/f7v/37f9VtbW+v1erFYSOLfSUYg89wAKv+FSi1/+Zd/+SZv8iZv+ZZv+Q3f8A3f9m3f9gEf8AFbW1uf/Mmf/PEf//EXL158xCMe+Vd/9deZGRE8QK31Mz/zMz/jMz4DeNjDHvbiL/7iKjWkEOM4go6fOJkoQSD+Z7G9WCxe7uVe7uzZs/P5vOu61Wp14sQJSXfeeefTnva0Jz7xiY961KMkAZLe9E3f9Pbbb3/d133d1trW1lYpZZqmiIiI06dPf/Inf/KnfdqnvfIrv/JLv/RL/8Zv/MYrv/Irnz59mv+FLly48LEf+7H7+/sR8SEf8iF931933XVnzpx5lVd5la/5mq85ceJEiXLu/Llf+IVfOHv27Eu+5EuePXv26Ojo0qVLv/M7v/Mar/Ea/O/h9N/8zd+80iu9UkTYlmS767ozZ86cOnXqfd7nfV7qpV7qoQ996P7+/q/92q+95Vu+5alTpyRNzVObWvrbv/1bv+d7f+Bbv/XbP+AD3u/hD3nQXXfd9d3f9d07x3buvvvuX/zFX3zHd3zHUkpE8J8CIPgvtLm5ef3113/3d393Zr7ne77nP/zDP3z1V3/1uXPnnvCEJxweHn7Kp3zKj/zIj0lx/vxFnlOt9WVe5mW+7uu+7u/+7u9+4Ad+oLUWIUQzzjScOH4iIoz5T2cAzItM0sbGxru+67vecccdfd/b/omf+Ilf+7Vfe7/3e79f/uVffpM3eZOP+ZiPOTg44LJSysu+7Msul8vHP/7xm1tbs9mstfY7v/u7Fy9eBCLipptueshDHvJTP/VTr/d6r7darX71V391GAb+F9rZ2bH9cR/3cZ/xGZ/x7u/+7g9/+MO/8Au/8HM+53Pe+I3f+OTJk6EATp8+/XZv93attQ/5kA/5lm/5lvvuu+/FXuzF/vRP/3R/f5//PRQ6fvz47/z27xweHGYmsFqt/uqv/mpvb6/W+shHPvIbv/EbP//zP//TPu3Tjo6OdnZ2jo6OgAhCLGbdLbfc8kmf9PFv/MZv/Amf8Ilf+mVffvLkye/6ru/6mq/+mi/90i/9oR/6oXvvvTcz+c8CEPwX2tnZeeu3fus/+qM/+v7v//7Nzc0P/MAP/OVf/uUv/uIv/rIv+7KP/uiPfshDHvqXf/1XG1tbT3rKU1ryXCTddNNNn/qpn/oDP/ADFy5cyJY2IS7tXRrH4cabri+l8J/OKCEhwWBeNF3Xzefz3/iN33C61vpe7/Ven/iJn/jSL/3Sz3jGM17u5V5ud3f3CU94gm0ui4jXfd3X/cqv+qrf+/3fu/3OO/YODr78K7/ivrNnuayU8r7v+75/8Ad/8Eu/9Etv8RZv8au/+qt33XUX/wtdunSp7/sf/MEf/KEf+qEf/MEffI3XeI2NjQ3bmfm0pz3tcY97nCTb29vbr/M6r/PhH/7hD3nIQ171VV/1Dd7gDba3t//+7/+e/z1sv8ZrvMaf/OmffP3Xf/2Tn/zk8+fPX7p06ad/+qf/8A//8M477/zjP/7jixcvvuZrvuYnf/Inv9u7vVtEbGxsZObR0dE9995rBHRd90Zv9Pof+9Ef9Td/8zef9Vmf9YxnPONoefToRz/6hhtu+NEf/dHVasW/n3l+AIL/QpIe+9jHfvM3f/MTn/jEz/3cz/2FX/iFj/3Yjz179uzm5uYtD3rwS73My9j5+Z//eT/3c78wTmNC8hz6vn/kIx/5eq/3et/xHd8xjYOc2H/9N3+zXh8uFvMiwoj/bEYJBvOvUWt9whOeYFtSKQXo+/4t3/Itf+u3fuu93uu9fuiHfqi1Zjsz9w8P5psb11x37Y/+2I+9+3u958d+wscpYufYMUlcdvPNN3/GZ3zG3/zN30h6qZd6qe///u8fx9G2bf43sH14ePgFX/AFb/Zmb3bPPffce++9X/3VX/1ar/Vatm0vl8sf/uEffvwTHm9cSgHm8/lrvdZrvf7rv/6XfdmX7e/vv83bvM13fdd3HR0d2eZ/g4g4ceLEh3zIh/z6b/z613/913/UR33UJ33SJ506deqP/uiPfvVXf/UN3/ANP+ETPuEv/uIvNjY2pmkax3EYhqPDox/6wR+8eP58YBvMrOve5E3e+Lu+49ttv+/7ve/Xfd3XnTt37gu/8AtvvfXWcRxt828XWCAcPDeA4L+QpIg4ffr0q73aq/3N3/zNH/7BHw7r4b3e672e8pSnvud7vddP/uRPvc3bve1DH/qQn//FX3jiE5+YmcY8J0lv9VZvdXh4aBtYrtd/8Ae/d+r0qY3NDf4LyGAwmH+l2Wy2vb199z132wYycxiG13qt1/qu7/quxz3ucXfcccedd965Wq3Onz//bd/+7d/xHd9x4tSp133913uN13rNd3/P9zx24sRiY4HEZaWUl3u5l/ucz/mc3/iN3/jxH//xv/iLv7h48aJt/oczVwzD8Ed/9EcnT5584zd+46/7uq87fvz4D/zAD/ziL/7i3t7e2bNnf+zHfuxv//ZvZ7MZ95NUSnmJl3iJD/iAD/ie7/me48eP33777U9+8pPHceR/j8c85jFv//Zv/1Iv9VJf8RVf8Q3f8A0f9mEf9oqv+Io/+IM/+Gd/9mc7OzullA/90A/98A//8B/7sR97ylOectvtt/30T//01tZWaw0ZGVnBNddc83Vf93Vf+IVfuL+//5Ef+ZH/8A//8GIv9mJ/8zd/01rj30Ugng+Ayn8h20Df97XWd3/3d3+VV3mVb/u2b3v6059+y4Nu+bzP/8Lrr79+a3NxeHg472ef93mf9w3f8A3XnDnN83iZl3mZhzzkIX3fS1oeHv793/7djdddv7O5BUjiP5UFAkHwryGp7/uXfdmXveP2O2666aZxHM+dO/dt3/Ztf/VXf/Uar/EaH/dxH/c7v/M7f/AHf/C2b/u2n/u5n/uu7/7uX/mVX/GWb/lWv/M7v3t0ePgSH/bi81m/Xq95Tg960IM+9mM/9tM//dOf+MQn3nfffceOHSuldF3H/0iZ2VpbLpe2ge///u9/jdd4DUk33njj53zO5/zBH/zB7/7u7/7Ij/xIKWW1Wr3xG7/xK73SK0WEbUBSKQV4xCMe8VEf9VFf8iVfsrGx8bd/+7cv+ZIvyf8eGxsbb/EWb/GJn/iJb/EWb9H3/TAML/uyL/te7/Vef/Inf/Ibv/EbH/ABH/D6r//63/iN3/iN3/iNwzBIaq211mxHGpAkCZjP56/6qq/6qq/6qn//93//1V/91YeHh6dPn26t1Vr5tzPPH0Dlv0Ob2v7+/iMe8Yiv/MqvrLVmesq89557P/MzP/NLv+SLX/qlXuo93vPdNzc2jYV4TraPHz8OAPfdd980TZ/z2Z+zsbEhif8KAvGvN5vN3vzN3/wLv/ALX+qlX+of/uEffumXfukN3+ANP/ZjP3axWETEa77ma37sx37sDTfc8Fqv9Vqv+PIv/5mf+Vnf933f+9u//Vs72zsXL1648cYbz549e82p07VWSVw2juP111//ZV/2ZT/2Yz/2JV/yJV/yJV9y3XXX8T+V0/fcc88XfdEXvczLvMzDH/7whz/84d/xHd9xww03vPZrv/aJEyfe+I3f+FVf9VW//du//fz585/wCZ/wt3/7t3/7t3978uTJruuAzLRdSgFOnz79GZ/xGd/wDd/wMz/zM+/wDu8wn8/5X8L25uZm13V//dd/feONN373d393RLzjO77ju7zLu/R9D0h6+Zd/+f39/WEYbrvttvd93/f9mI/5mO/5nu/Z2NgAFosFz+kxj3nM13/913/nd37nb/7mb77RG71R3/eAJP7VDECCeW4AwX8hSZIi4sw1Z574xCf++q//+i/90i9JKiXaOPze7/72q73qq8wXi0//jE85trO9tbkIxPOICEmSJP3Kr/zKYrE4c82ZWiv/RYQFAQHiRRYRx44du3jx4u/+7u/+wi/8wo033vgKr/AK29vbtdaI2Nra+tAP/dAv+IIv+MM//MML5y/Muu6Xf/lXXvHlXt7wQR/0QY9//OM/6ZM+6Rm33TZlJiBJ6roOOHny5Lu927u96qu+6qd/+qdf2r3E/1QKfc3XfM3bvu3bvud7vudrv/Zrf9qnfdo3fdM3ffd3f/fu7i5wdHT0tV/7tddff/3tt9/+27/92y/zMi/z9Kc//Q//8A+naTo6OvqLv/iLu+66yzaX7ezsfNzHfdyHf/iH/9iP/Rj/e0TE8ePHjx07du+9967X64/8yI/8oi/6old8xVfc2NgopZRSIqKUsr29fezYsWuuueaGG2645557vumbvunzPu/z/uiP/kgSz6mUMp/P3+3d3u3pT3/67u5ua41/I6NEIPPcAIL/DltbW3feeeev//qv7+/vA7b39/d/+Vd+5fVe//W7Um644YYXf/EXt80LZtv2+fPnp3Hiv5L5N9va2nr7t3/7r/zKr3z84x//dm/3dogHeqmXeqmv+IqvuPPOOz/ogz/oXd/1Xefz2cd/wif8yA//8C/+wi/+4A/84Nbm5nd9z3cfHOzb5jltbm6+xVu8xcWLF3/u538uMwHbtvmfRNLZs2dtc7+XeImXePSjH/0bv/Eb6/V6e3v70z/909/93d/97d/+7X/4h394tVq9yZu8yY/8yI/89m//9k/+5E/+7M/+7MbGhiRJkoCu6x71qEf97M/+bGuN/z26rtve3j537tyjH/3oBz3oQbVWwLZt25mZmev1+md+5me++qu/+qd/+qc/8AM/8Kd/+qf39/df+qVfmhdge3v7zJkz3/Vd33V4eMi/i3k+AIL/Jq/0Sq/0FV/xFe/yLu9iu7V25513bm9tXXfNGXCNCAHY5gU7PDz8g9//g9l8Vkrhv4hAEPybzGazN33TNz1z5sxLvMRLbG9tl1IA25mZmZn5qEc96nu+53s+93M/95u+6Zve+R3f6Yu/+Iv/5q/+SqYr9b3e+73//M///Du+/TvuveeebI3LJEmKiGuuueaN3/iNf+mXfung4KC1Ztu2bdv8z2BbkqSI4DJJH/ZhH/ZjP/Zjt956a5taKIb18Aqv8Ap93//DP/zDmTNnHv7wh3/t135t3/cf9VEfdebMmYjgMkkRsbGx8ZjHPOapT33qOI787/F2b/d2f/7nf37rrbdO49RaWy6Xu7u7d95559mzZ//u7/7uF37hFz7nsz8nM7/gC75gNpu967u+68u+7Mt+7Md+7Ld+67fa5vmZpunN3/zNn/jEJ7bWJPFvZDAkzw2g8l/O9t7eXmZyv9Vq9Yu/+Iuv8zqvI8SLprX2uMc97uy5sy/xEi9Ra+W/iJEhwWD+9TY3Nz/iIz7iN37jN0otkjJzmqZ77rkHGMfx8PDwnnvu+YM/+IM///M/b6194zd904033VhKtf3Kr/TKf/onf3rNtdf84A//0Pu/7/sdP3aMB6i1PvjBD5Y0TVNmllL4H+lP//RPX/3VX73WajszT544+Rmf8Rnf+I3f+Fmf+Vnb29t//Td//Zd/+ZePfexjf+/3fu9lX/Zlu6576EMf+nqv93onT57kMkmAbWBjsfFe7/VeP/ETP/GJn/iJ/O/x0i/90o985CO/67u+66M/+qP/5m/+5m/+5m8uXry4Xq9Xq9XOzs5rvuZrfsRHfsRNN97UspVSImJ7a/vzP//z3+d93ocXoOu6l3u5l/uBH/gB/lMAVP7LZeY999yzWq0iQhJw8eLFv/3bv33v935vYyFeBLYPDg5KKQ996EMjwrYk/iskSkgwGMS/Rq31xV/8xX/7t3+7tZaZf/qnf/oVX/EVj3vc44CIyMzMfPSjH/1mb/Zmb/mWb3n9DTcoQnDHHbc/4QlPfO3Xfq1XeIVX2N7eEQCSuF8p5dSpU7VW2xHB/1Tf/d3f/ZjHPObN3vTNjGutCr3sy77sk570pEt7l/7qr//qSU960mu91mt993d/96233nrx4sW7775b0u2333769GkeQBLQ9d329vZv/MZvfOzHfmytlf8lutq93du93Ud/9Ed/8id/8ju/8zt/4id+YihKKQjANmC71jpNU0QcO37swsULr/zKr8wLMAxDa+3o6Mi2bUn8RwKo/Jc7PDx8whOecPPNN0sCWmvPeMYzVqvVsWPHIgKwzb+klHLbbbcNw3DzzTfXWiXxX0CGhATzb5WZf/7nf/67v/u73/iN3/hKr/RKX/M1X3PTTTcBd95556lTp/q+t11KkWQJsI35/h/4/i/5ki/Z3t7mBZjP56UUQJIk/qcxwAd90Af96q/+6qlTp17lVV6Fy5x+7dd+7c/6rM967dd+7Z2dnQc/+MGv+qqv+qQnPen8+fO/93u/9zIv8zJ/8id/sr29/YhHPILnUUrp+77Wyv8eCj3kIQ952MMe9vCHP/wN3/ANI6KUAtjmfq21Jz7xib/7u7/75m/+5s94xjPe//3fv9bKC9D3/TiOtvlPAVD5L7e/v3/vvfe+zdu8DWD7/PnzX/mVX/myL/uys9mMyyTxL5mm6e/+7u/GcXzIQx5SSuE/QODgP1+t9SVe4iW+8iu/8qu+8qse9vCHRQSX3XTTTYBtQBLgzIu7u1/ypV/yru/yrgcHh9PUbJCFeB6ZOY5jay0ibAOS+J9DbGxsvNzLvdwbvuEbfv/3f3+t9WVf9mVtf//3f/8znvGMv/3bv73mmmumaXq913u9/f39N3uzN7vtttuuu+66Bz/4wW/xFm/x5Cc/udb6kIc8hOckSZJtSfzvsb29/chHPvKuu+4ahqHvey6TxGUXL1780R/90fl8/gEf8AHL5fLo6Oj48eO8UBcvXpQkSRL/wQCC/3L/8A//cP78+euuuw6wvVqtjo6Orr322q7rJEniRdB13e23315KefEXf/FaK/979H3/Wq/1WjfddNNDHvoQSbZ5Ae68886P/KiP/IgP/4hSyjAOxrxgq9UqM1trtiVJ4n+Yl33Zl/28z/u83/md3/nMz/zM7/7u7/6TP/mTs2fP/sM//MPHfezHfdmXfdkv/MIv/MZv/MZ7vMd73H777W//9m//xCc+8XVf93Vf//Vf/y//8i9f9VVf9cYbbxzHkefRdd3h4SH/e9gG3vmd33l3d/eeu+/hOd12223f/M3f/M7v/M7v/u7v3ve9JKDv+2EYeAEy8/Dw0Db/KQCC/3K33nqrpMViAUTE4eHh3t7ejTfeCNi2zYvA6Wc84xlnzpwBbPPvEhBcYV404t+q1vqyL/uyd91116/8yq9I4n6SJEmybfuP/uiPfvZnfuYLP+/zb7juur7WgtQcUJB4Pvb29tbrdUTwP5Kkt3iLtxjH8TVe4zVKlHd7t3f7oR/6od/5nd/50z/900t7l176pV76sY997Ld/+7f/wi/8wid90ift7OzM5/OHP/zhj370o0+ePPmzP/uzly5dkmSbB5A0juMTnvAE/veQJOn48ePHjx9/3OMf11rLTO538803f+InfuLOzk6tlctms1kppZQiiRdguVxGRGbyHw8g+C/XWtvc3JzNZoDto6OjaZrGcWyt8aITTnddN5vNJPHfQCD+TRaLxdu//dt/xVd8xdHRkW0eIDOXy+U3f/M3P+EJT3jP93jPm2++uZQiSVI6eQEy8+joSFLXdZL4H8b2MAxnzpz5gR/4gc/93M/9/T/4feArvuIrHvnIR77US73U13zN13zqp33qm7zJmzzqUY/q+15Say0iHvOYx8xms1d8xVd8szd7s+3t7cy0bds2l0VErfXv/u7v+N9ma2vrpV/6pf/8z/98GAZJ3E9SKUUSl0na3Ny88847a628YLaBzLTNfzCAyn+taZqWy2XXdREhybZQZt52222AJF400zRNbXrQgx7U9z3/McQVBvH8WRAQ/DtI6rrurd7qrX7iJ37id37nd97ojd6IB4iI3//935/P5+/0Tu+0sbEhCbANzGYzSTw/mXnx4sVpmjKztVZK4X+YruuAG2644U3e5E3+7M/+7CM/8iMXi8XLvdzLveRLvqRtSbXWUgpQSjk6Ojo6OooIoO/7rut4AElcVkpZLBZPfepT+d8mMx/1qEf9zu/8zl133fWwhz4M8XxFxHw+/4mf+IlXf/VXjwien8zc2NgYhkES//EAgv9Ctler1d13380DHDt+rO/7Jz/5yRHBi+zee++1/Uqv9Eq1VP5jmH8F8e8gaWNj40M/9EO/4zu+o7XGA5nXeI3XeLd3e7f5fC6J+9kGbNvmeRweHv7Zn/1ZKSUiIsI2/yNFxKu8yqs8/vGPn6ZJUillPp8vFov5fF5rlQRk5mq1Ojg4KKVwmSSen1LKox71KNv8b1NKefEXf/GTJ09+z/d8z3pY8wLUWh/2sIf98R//sSRegFLKuXPnzp8/PwwD//EAgv9awzBcunQpMzPTNrC9vX3ixIm//uu/vu+++7Klbdu2eaGe8pSn2H6t13qtftbz75WQXCEQL1TgAPHv03Xdq7/6q2fm05/+dNvcT6GNjY2u62zbtj1Nk+1xHA8PD20Dtm3zAMMwPPWpTy2lSJqmCbBt2zb/A0iSJMn2Ix/5yEuXLt19992tNV6A5XI5m836vgds2+b5iYgHPehBrTX+F9rc3HyjN3qjP/uzP7vrrrt4ASSdPHlyGIa//Mu/tM0L8JSnPKW1VmuVxH8wgOC/lu1pmmxzv5MnT37u537u0dHRD/7ADw7D4DQvgqOjo5d5mZe58cYbSyn8x0iUvKjEv1vf9y/7si/7+7//+zw/EcFlpRRJrbVxHDPTNs8jM8dx7LpuPp+XUmzb5n+kUsp7vud7fsu3fMt6veYFyMy77rprGAanecEi4tSpU7VW/heS9DIv8zLHjx//kR/5EV6AiDhx4sSlS5f+7u/+jhegtbaxsSFJEv/xAIL/WvP5/OTJk+v1erlcclmt9cEPfvDLvdzLff8PfP8Hf8gH33b7bbYl8UL98R//8cu+7Mv2fW+b/zrCggBA/PuUUt7jPd7jR3/0R1erlW3uJ0mSJC6TdN1110na3d1trfH82M7MxWLRdZ0k/ueRJEmSpJd7uZf70z/907/927+dponnp9Z6dHQ0DINCvFCllGEY+N9GkqQTJ0685Vu+5dOf/vRz587x/ETEK77iKz7sYQ/7hm/4hqc//enTNGWmbS6bpsl2KeWRj3zker1urfFvJwCC5wYQ/BfKzK7rXv7lX/7222//mZ/5mdVq1VqTdPz48c/5nM95+MMf/qQnPekDP/ADv+mbvum+++47ODiwzQvwq7/6q+/wDu/Q9z3/1QSA+HeLiOuuu+7VXu3VvvEbvzEzecG2Nrfe+Z3f+eu//uvvu+8+XoCdnZ3Tp0+XUmzzP9sNN9zwsi/7st/7vd+7Wq14HpKOHTt24403/s3f/A3/p9VaX+d1Xufee+/9kz/5E54fSTs7O9/6rd/6iEc84j3e4z1+5md+5p577pmmybbt9Xr9RV/0RWfPnh2G4Zprrum6jn+X4PkACP4LlVK6rnvjN37jN3iDN/iBH/iBH/uxH1sul7Yz87rrrvvmb/7mH/iBH/i2b/u2Jz3pSe///u//fd/3fZnJ87C9Wq0k9X0P2LZtm/+F+r7/sA/7sJ/+6Z9erVY8gCRJkiTZLrW8/du//enTp7/3e7/33Llzmclz6vv+pptuuuWWW/ifTZKkUsqnfMqnPPWpT73vvvsyk+exsdjY3t5+2tOednh4CNi2bZvn1Fp72tOedvLkSf53sn3q1Kk3eIM3+MEf/MGz953leUiqtd5yyy1f/uVfPo7j3t7exYsXI6K19uQnP/mLv/iLz507t7Ozc+edd5ZSWmv8mzkwWDw3gOC/lqSdnZ2P+7iP++AP/uDv+I7v+Kqv+qr9/f1aa6315MmT11133c/+7M++4zu+44d+6Ie+7Mu+LGDbNg/QWjt37tz+/v40Tbb530zS8ePHP+qjPuo7vuM7MpMXbLFYfPqnf/o7v/M7932PeS611muvvfbaa67lf4OIuO666970Td/0G7/xG4+OjgBJkrjMtkLb29s//dM//Tu/8zsHBweZyfNj+4477jg8POR/J0l937/yK7/yHXfc8WEf/mFPfvKTW2s8j1LKmTNnPuqjPuov/uIvlssl8D3f8z3f/d3f/W7v9m6f9mmf1vf9q7zKq1y6dOncuXPjOPJvEQAEiOcGEPyXi4iTJ08+9rGPBV7qpV5qmiYhAOi67g3e4A2+/uu//iu+4iu+4Ru+4b777uN5RMTtt90OrFYrp/lfLiLe9E3f9M477/zrv/7rcRx5ANuZef78+fV6DZw8efKhD33o8ePHSy2SeICNjY2XeqmXOloejeMISJLE/2xv//Zv/7d/+7d33XVXa43LJEmSJOnd3vXdXuqlXuqLv/iL3/d93/cLv/ALf/zHfzwzbdu2bRvouu6t3uqt/vRP//Ts2bP87yTpxV7sxT78wz/8/PnzH/IhH/LlX/7lf/3Xf/34xz/+nrvvmabJtm2g1vo2b/M2n/u5n/viL/7iEXHdddc95CEPecQjHnHixImIeOxjH/tKr/RKn/M5n3N4eGjbtm3+dQIAYZ4TQPDfZLFY1FrPnTu3s7NjDAC11oc//OFf93Vf9xVf8RWf+ImfePz4cZ6H7bPnzo7j+Au/8AvrYc1/KYPBYP7jdF33fu/3fp//+Z//jGc8o7XG/aZpesatz/icz/mc5XIpCZDE81NKueaaa373d3/38PAwIvgfT9J11133zu/8zp/+6Z9+33332eY5nThx4tM/7dO/+Iu/+F3f9V3vuuuul3iJlwAkSeJ+pZRHP/rRr/zKr/zLv/zLmcn/TovF4s3f/M2//du//SVf8iW///u//yM/8iM/7uM+7nM+93N4TvP5/MSJE7PZTNLrvM7rvPM7v3NERISkra2tT/qkT3ra0572F3/xF+M48m9hAIx4TgDBf5Nbbrnlnd7pnX7pl37pnnvusc39aq0nT558sRd7sRd7sRdbLBaSJPEAkra2tubz+fd93/f9/d///TRN/FdSooQEg/mPUEp52MMe9qqv+qof/dEf/f3f//133XXXxYsXn/zkJ3/f933f13zt17z/+7//5uYmIIkX7EEPetD+/v7f/M3fZKYkSZIk8T+PJEmllDd6oze6++67v/RLv/S+++4bhzEzM7O1BpRaTp069aqv8qpv9mZv9pVf+ZUPf/jDSxQukyQJkLS9vf3e7/3ev/RLv3TPPfe01jKT/20kLRaLBz3oQe///u9//fXXb29vf/Znf/YnfMInlFK4nyRJgCTbGxsb29vbkrjfLbfc8sZv/Mbf+q3feunSJf7VEiUkSp4bQPDfZLFYvPEbvzHwHd/xHcvlkgeQFBGSeAFe5mVe5hGPeMTZs2e/8Au/cG9vzzb/NWRISDCY/yARUUp5r/d6rw/7sA/78z//80/4hE/4iI/4iG/6pm86ffr0x33cxz32sY/t+14SL5jt48ePv9/7vd8v//Ivr9dr/jeQdN11133Jl3zJXXfd9dEf/dHf/wPf/2M/9mM/8zM/84xnPKO1ZlshRNd1GxsbXdcpJEkSD1BKufHGG9/2bd/2O7/zOw8PD23zv5CkiHjwgx/8Fm/xFufOnTt+/PhDHvIQXgBJPI9Sygd/8Afffffdf/AHf7BcLvlXS5RgnhtA8N9E0nXXXfehH/qhf/RHf3TnHXdmpm0ukySJFyAijh8//qmf+ql93//d3/3dr//6r4/jyH8BcZkBEIj/UKdOnXqjN3qjT//0T3/lV37lu+6664M+6IPe+I3f+Kabbuq6jn+JpMVi8ehHP/ree++9ePEi/0vUWl/5lV/5q77qqx7+8If/+I//+Dd8wzf89m//9jRNPIAkXqiu697wDd/wyU9+8m/+5m9K4t/CYDD/fSQtFovXfu3Xzszv+Z7v4XlIksQLdt11133wB3/w133d1917772tNf4tzHMDCP6bSKq1vszLvMxDHvKQz/6czz48PGyt8aKR9BIv8RIv93Ivt1gsfuiHfmi5XPJfRgnimcR/nIiIiGvOXPMar/Eako4dO9Z1nSReBJKAkydPvuzLvuwf/uEf8r9HKeXGG2/8xE/8xK//+q9/27d92zd/8zd/0IMeVGuVxItG0nw+f5/3eZ9v/dZvvfvuu/k3MWkb898oIs6cOXPq1Km/+qu/4l/DtqSIeKM3eqOu677sy77s4sWLtvkPABD895G0vb39/u///q21pzzlKfxr1Fo/6RM/6cYbb+S/lHkmgfhPoNCjHvWo1trf/d3fTdPEi0zSYrF4jdd4jd/+7d8+Ojqyzf8eW1tbt9xyy5u96Zv9yZ/8yeHhIS8y27a7rnvFV3zFN3/zN/+Wb/kW27Z5ANu2bdu2bZvnkLaxIcH8tzp16tTHfMzHrNfrixcu8iKTxGXHjx//5m/+5ic/+cl/93d/11rjRSYkJCTxnACC/1a11Mc+9rFf+7Vf+5AHP6TWyossIh75qEd+13d+19d//dcvFgv+q4n/NIvF4qM+6qN+5md+ZhgG/jUi4rrrrpvP5/fee+80TfzPNk3TOI7jOI7jOI5ja+3U6VNnz559xjOeMY7jOI7jOI7jOI7jOI7TNE3TNE3TNE3TNE3TNE3TNE3Z0jawWCze7d3e7e6777777rszk/vZ5kVgMDb/Crb5j9b3/WMf+9hSyt//w9/zryQpIm668aaP+qiPuuuuu9rUeFEJCQKC5wZQeZFJaq3VWjOzlMK/Q2ttuVyuVitgGIa+78dpPHfunCReNLYj4sTJE/P5XJIk/k1sZyYgyTYvlG07I0ISiH+HcRzvvPPOP/7jP661ZibP6d57733CE57wEz/xE/P5nBdAUmvtpptuetmXfdnFYiEJOHbs2Iu92Iv9xE/8xEd91EfxIrPNiywi+He7++67f+3Xfi0ieIDMPHny5J/8yZ884QlPAGxL4gWzXUt9/Td4/RMnTkja3t7+rM/6LEmttVIKLxrbERESNv9VMnNvb+/s2bN7e3vjOGYm99vf318sFj/1Uz9VSiml8EJJetjDHnbixImI4LJSy6u/2qufO3+u1MKLzKaUMLaReACAyr9G3/eHh4e2+fe59957v/3bv/3cuXNAm1rXd9M0hUIhXjBJ3C8zI+Lt3/7tX/EVX7Hve/6tbN933322AUm8ULN5X0qs1+vWEsS/wzAMX/d1X3fdddedOHFimiae08bGxju/8zsPw7Ber23zAqzX61/8xV/8kA/5kFd6pVcCbHdd9yqv8irf/d3fbZsX2ROe8ISNjY2I4F9iG1itVvz7/MzP/Mw4jg9+8INrrZIASbbPnDnTdR3PwzbmuUxt+sM//MMTJ0+8/uu/PhARN910E89JEv8CDcO6ZatdgPiX2AZny1IL/1aHh4ff8i3fctddd3Vd11qzDQCYqU033XTTer3+iZ/4CduAbZ4fSavVyvZXfuVXLhYL7nfs+LFjx49J4kVje1iPEQHmuQFUXmSttTNnzjz+8Y+/7777rr/+ekAS/3qZ+ed/9ucHBwcf8zEfYzszSym2AUm8yJ761Kd++7d/+0u8xEv0XY/4N7C9Wq3+4R/+oZRy5syZiOCF2t7avvbaa++777477rhje3u76zrbXCaJF5nt1trFixc/53M+Z7FYlFL417N9eHh43333/fmf//krvMIrlFIiIiIe9rCHfeZnfmbXdbwQBnHFM57xjMc//vEv9mIv1nUd/5Jjx45l5tOe9rRXfdVXLaVI4l/P9nK5fNu3fdubb75ZkiTuZxuQxHOyzfOYpmlra+upT30qIIl/k9baU5/6NDtPnTpZavBCZWYp5dKlS/fcc/7Gm66xkfg32N3dveeeez76oz/69OnT2VIh7me7lGJbEi9UZrbW3uiN3mhYD4vFgn8T29PUnv70p0/TuuvbOLZ+Vng2gMqLbLFYvO7rvu7v//7vf/3Xf/2HfdiHnTx5su96BGCDxRVKnoOweIB0rlbDbDZ/yEMewgvwEz/243fffTcP8OEf+RE8p+Vy2XWd0+PUMFdIgIwxz4+RJUlqrR0cHPzGb/zGM57xjNd5ndc5efKkJF6oKPEpn/IpH/IhH/Kpn/qpn/mZn3nTTTdFhCRACPGiMsvVcr1eSwoFz+mXfvEXP+JDP+xDPuzDPu4TPv5nfvqnP+6jP+YDP/iDPvGTP5nnZbLl4eHhxYsXASAzgYg4PDzkOTnN/YxDkc577733S7/0SyW913u9V9/3/Ete+qVfemtr6+d//udf+7Vf+5prrtnY2OBfT9I0TVwmiQeQ9K7v9M5/+id/8gu//MuPevSjfuLHf/yTPv4TPu4TPv5DPuzDeB6SxnHc3d2VxL9Ja+2ee+79yZ/8ya2tjdd8rVfvusILZruUcvz48Vtvvf2v/vxx11x7uqvFGJDEv4btvu+3trY2Nzd5AEnArbfe+gav87pv9dZv/UVf+iWv+LIvd3Bw8Ad//Ec/9IM/+PVf+3Xf94M/8Aqv+Iq1Vu7XdR1CEv96trPl3t7e05729PmivsRLPbTW4DkAVF5kXde92Zu92e/+7u/+wi/8wuMf//g3fMM3vPmmW6SQQMKBEgCDebbgOWXm3//D30/TxAv2Ez/+43/6J3/CA3z4R34Ez+P8+fO//Tu/XWuPuUIKAGyb5yUgM9P2OI6/+qu/+md/9mc33HDDx37sx0riRfAyL/Myb/M2b/PzP//zH/7hH765udn3PZCZtvnXaK1dunQpIhD/ZsZTm370R3/0N37jN2wDtoGIACRxWWYCtnkASZJ2d3e7rnubt3mbl3/5l6+18i956EMf+nZv93bf933f93Ef93Ef8iEfcubMmVqrbf6Vzp49m5k8P2/zdm/7p3/yJ7/6K7/yqEc/6td/9deAl335l+cFsH3u3Lm///u/5/mRxAtiWube3qVv+PpvePJTnvJ+7/t+j3nMoyOCF0xS3/dv+IZv+Fd/9bff/T3fe+rMiUc/9pau6wBJ/Gssl8vWGi/AQx7ykDNnzjzucY/70z/504ODA+C3f+u3/+Hv/6HU8rIv93KlFB7A9tHRUa2Vf73M3N3d/a7v+q6jo6M3f/O3PHXq1DS1vlSeDaDyrzGbzT7/8z//O7/zO3/6p3/6W7/1W6cpAUkAFhgByXMQiGcxCk3T9KZv+qa8YD/4Iz8MvNHrv8FTn/KUv/2Hf9jY3OD5uf322z/90z8dC8QVQsg2z5+RbduOiK2trZd6qZf6jM/4jJtuvIkXzWKx+KRP+qS3e7u3+4Zv+Ia777p7apMkQBL/GpIODw8BSfybSBICuq7ruq7rOsA2IEkS97MN2OZ+tgHbL/uyL/sBH/ABL/mSL1lrBWxL4gXb2Nh4l3d5l9Vq9TM/8zOf8Amf0Pc9IIl/pUuXLn3gB34gYBuQxP3e/M3f/PM+53N+7Vd/9X3f//1+73d/99Ve/dVuvPFGXoDM/Nmf/dlf+ZVfkcTzkMTz5QCmaTLe3Nx8l3d513d8p3fY3NzgXzKfz9/+7d/+T/7kL37vd//44z72Ux79YtedOnUKkMS/xuHhoW1JknhOkoBXeKVX/MWf/4Vf/IWf39raOnPmzO/+zu/83d/+7Yu/+Ev0fQ/Y5n6Hh4df/uVf3nUdz48kXrDlcvm4xz3u3Lnzj3zkoz/moz9msVgATgAFlwFU/jUkbW5ufuRHfuR7vMd7/NEf/VE2m2cJSAAM5tmCBxBk+i//6i+naeTf7dprr32f93mfvpuZZxICwOYFScB2rfXFXuzFbrjhBklRghfZfD5/8Rd/8W/8hm8cxkGSEAKQxItsuVx+wid8gm2eRyiA1ibMNE5ARME8F2NEKeVd3uVdPvRDP1QS/1omSkQELzJJp0+f/oiP+Ih3fdd3feITn/hXf/VXEWGbf6U///M/jwien8XGxlu+1Vv98A/+0Pd+93evVqu3edu329nZ4QWIiIc//OGv9mqvxvMjiecvgFLK6VOnXu3VX/26667tuo4XzWKx+OIv/sIf+eEf/amf/OXHPe5xgG1JPDfxAjkzX+IlXkISDyCJ+73My77sL/78L/zUT/zka7/O69x0883f/73fO47jm7/lW0jiOUn6wz/8Q8A2z0MSL1hELBaLd3qnd/ygD/qg48dPYJ4HQOVfKSKAkydPvtmbvRn/JrZbjn/913/Nv9v111//hm/4htvb2/wXkgSoaF7m/FuN4zgMQ2aO41hK4QEe/oiHAz/9Uz/d97Nf/7VfAx75yEe21nguYhiGzIwISbVWQBL/erYBSbwIImJjY2NjY+Pmm29+/dd/ff71bL76q79mmqblcgkAkgBJtoE3edM3/eEf/KGv/9qv29nZeY3Xes2IODo64nmM47hcLl/rtV7rIz/yI3l+JPEfTdL29ub7f8B7v/d7v8eFi+dWq8E2YJvnEJjnQwbfe++9P/IjPyJJEs/Py7zMywDjOL76a77GTTfd9F3f8R3AS77US/E8tra2vvVbvnVre4t/vVrrzs7O5uamJMAYkMSzAVT+O0iapml/f5/nZJsHyExgf3+/ZeN5HB0d8b9WRCwWi7/8y7+85pprSik8pw/8kA/+zm/79q//2q/d2tp6n/d7v0c++lFPfdpTeU6SlkfLixcvvtZrvVYphX8HSfxXMpubm1/1lV91y4Nuaa0BkgBJtgHbOzs7e3t7D37IQ77pm74JkMTzaK39zd/8zZu/+Zvz30ClltOnr4kI/vX6vi+lXLp0SRLPz0033dx13TiOL/mSL3ntddfNZrP1ev2whz3s4sWLPEBmzmazG2+68dixY/zr2eYBJPHcAGSb/1qZ+Zd/+Zdf9EVfdObMGdu2uV9m8gB33n7HOI4PesiDSynczzaXXbx48SVe4iU+7uM+bnNzE5DE/x7TNN1zzz0/9mM/xvOTraVtW1IoFOI5SQIkPfrRj371V3/1jY0NSYAk/sdrU9vbO/id3/2tL/uyL5umicsk2c5M7mc7IiQBknh+rr322m/6pm+67tobAAkAmftJ4r+Pk+clgdjf3//2b//23/md35nNZlwmiX+91Wr1aq/2ah/2oR+2sbnBfwoA2ea/3Gq1On/+/P7+Pi+UbUASz4/tBz/4wbPZTBIgif89bNsGMM8rnTwPSbYBSYAQwjYQEQAgif/ZbK6QaK2VUvh3sN1aC1VAAkDmfpL47+PkeUkgbE/TNE0TDyCJfyXbEdF3PUIS//EAZJv/crYBQBL/Dra5nyT+t7HN82Ob5+R0lMjM1pptSbYBSVzmNIAAsHgm8Uzm+ZJ5UQlzPwFgBJgHsng2AWDu1zK7rpRSIoL/OE6eg8xlknguBkD897LN85DE/zgAlf8l1uv1NE2ZaZv72eZ+kvgfT1Ippdba9z0gCbDNv6Rlu7h78elPf/rf//3fT9NkG7Aticts80wC8WwCwDx/yYtKIJ6bwTyH4NkEgLlfKbFYbLz8y738zTffOF/MI4IXWWaO49has81zcgrxAMllkgAQD2QQL5j5txAvGomIKKWUUiTxPx1A5b+DJP41Ll269FVf9VV//Md//PSnP90297PN/SQBkrjMNv/z1Fof+tCHvuEbvuH7vd/7zefzUgogiRdqGIaf//mf/6qv+qrz58+31mqtkmwDkrgsM3kmgXhRJS8qgXgmgSABSJ5NIF6wWupqtWrZXuu1XuvzPu9zb7jh+ojgX5KZ4zjedttt3/Zt3/akJz3p4OBAUkSARCAAKbBQApBcJgmAACB4bgbzHAzmBbBtm/tJ4tmCF00E11xzzeu//uu/1Vu91WKxkARI4n8oANnmf7bVavVWb/VWT3va0x7zmMe89mu/9mw24362uZ8kQBJgm/8ImZmZXBYRkiTx7zAMw8/8zM/ceeedr/7qr/7N3/zNs9mM58c292ut/c3f/M0Hf/AH2/6AD/iAF3/xF18sFoBtQBKXZSYgCcDiRSTzIrJ4DgIDyDyQxQshzp4998u/9Mu/8qu/8qAH3fz93//9J0+ejAheqGmafud3fufTPu3TLly4sL29/aAHPQiwDUBIAiThQAlAcpkkAAKA4Lklz81gXgDbtrmfJJ4teNFcvHj+7rvvnqbplV/5lb/0S7/02muvBSQBkvgfB6DyP9s4jl/zNV/z1Kc+9QM+4AM+/uM/PiIk8Z/Pdpvak578pN/8zd88PDxsrb34i7/4673e6y0WC0ASIIl/vfd93/f99E//9N/+7d/+8R//8Xd7t3fj+ZHE/Vprn/mZnzlN09d8zde86qu+aq2V/xzTNEWEJNu2JUnieVk8i8zzZfFcZEAS8Pqv/7pnvuTUD/7gD/7mb/7mW73VW/V9L4kXIDOf8pSnfMEXfMF8Pv+Gb/iG13iN15jP5/yvlZkXL178hm/4hh//8R//7u/+7o/92I/tus62JNuS+J8FIPifbZqm3/7t337EIx7xsR/7sRHBfxVJ3/ld3/n1X//129vb7/Zu7/aBH/iB995777d/+7cfHh7y72B7c3Pz8z7v83Z2dn7lV36FF8HZs2fPnTv3xm/8xi/90i9dSuE/h+3W2n333Xfu3LlpmvjPFBHv8z7vU2v9mZ/5mfV6DdjmBZim6Xd/93fvvffeD//wD3+913292WzGv5Jt27Zt27Zt27Zt27Zt2+Z+tm3zn+bkiZMf/uEf/vIv//K//Mu/fNddd9nmfy6Ayv94T37yk9/+7d8ekMR/of39/a/+6q/u+x6Ypuk93/M9P/3TP32aJkmAJP71JNVau6678cYb//qv/5oXwTiOth/96EfP53NJPD+2bUvKzIiQxL+G7c/93M+dzWav+ZqvefHixb/+67+ezWbv+R7vefrMaUk8F5kXLDNLKYjnIe4n6brrrlssFkdHR5kJSOIFWK/XT3jCE66//vpXeqVXihL86/3e7/7eF33xlyyXS56TUERkZqnlwz7sQ9/mbd56mqY//uM//uu//utbbrnlDd/gDeeLOf/RJAHHjh17qZd6qcc97nH33nvvQx7yENuAJP7HAaj8zyYJ2NnZiQj+a0WEbQAopQBbW1sXL1zc2dmJCP4dJG1tbUniRWDbNv+S1trtt9/+93//9zfffPNLvuRLllJ4kZ07e269Xn/cx33cxsYG8CZv8ia/+Zu/+T3f+z3v//7vf/z4cV4EtoFLly6VUra3tyXxgkkCxnEchgGQxAuVmYvFYjab8W9Sanfy5MlhGHhO4zA+6UlPOloezma9ncCFCxfuvvvu937v9/61X/u13/29333DN3xD/nNExOnTp0spe3t7mSmJ/6EAKv9LRAT/tZbLpaTHP/7xP/3TP/0Wb/EWD3/4wzPzj/74j2648YbZbGZbEv9WkqZp4kVgOzMl8UKYJz3pSV/4hV94/fXXP+MZz3ilV3qlj/zIj+z7nucnMzPz53/+53/zN39zHMft7e33eZ/3WS6XkiICsP3ar/Xat99++w/+4A++27u92/Hjx3nBbB/sH/zkT/3kr//6r0fEjTfe+C7v8i4v8RIvwQslqZQyTRMvAknTNNnm3+RVXuWVXvEVXx7ApG0zjeM/PO5xX/7lXz5Ow2Mf+5gP/dAPef3Xfz273XXXnddee+18Pn+d13mdH/mRH7nvvvuOHz/edR0gif8IkrgsIiJCUmaWUvgfCqDyv4EkSfzXunjxYmY+7GEP++iP/ugSpeu7Jz3pSWfOnLENSOJ/DnHhwoVP+qRPesQjHrFcLn/pl37pD/7gD177tV9bEs+jtfZnf/ZnX/M1X/OjP/qjGxsbv/mbv/l93/d99913X2uN+01tetSjHvU93/M911133Zu/+ZvPZjNegHPnzv3gD/7gy7/cyz/5yU9+r/d6rx/4gR9Yr9e8CCTxXyIiIgIAWvMTn/jE7/zO7/y1X/21M2fOfMEXfMEbvuHrb2wsatcBR0eHq/VSYnt7+5Vf+ZW///u//0M/9EP5T2AbACRJ4n8ugOCqF2B/fz8iImI+n5daDg4Obr/99ic/+cmtNUn8D2O7lLJYLE6cOPHmb/7mf/RHf3Tp0qVpmngeEfH3f//3r/3ar33ixInNzc03eZM3eehDH7q3t2ebyyT1XX/vvfd+4Ad+4N/+7d/ee++9tnkerbWzZ8/+6I/+6Ju+6Zs+6MEPuvXWW/u+v+uuu2azGf/zTFNz8ru/+7vv937v9/M/9/OPevSjvuVbv/mt3/qtNjc3a+2EgGmanvqUp4zjWEp5yIMfkpl7e3v8vwYQXPUC1Fp3d3d/4Rd+4Wu+5mv+8A//8O67777mmmue/OQnD8PA/zx7e3vf+q3fOgyDpO3t7fd+7/f+8R//8VIKz0PS0dHRYrHITKDW+vZv//anT5+WxP0U2t/fn81mL/mSL/m4xz1utVpN08Rzioj9/f1HPvKRD37wg0spFy5ciAjbOzs7vAgkSeJFIEkS/z6Pe9zj3vVd3+0jP+Ijz5w5823f/m3f/m3fdvLkqcc97vH7+/sSYOChD33oxYsXh2EEtra33u7t3u6JT3xiZvL/F0Bw1QswjuNnfMZnvNIrvdIHfMAHHB4e/uAP/uBbvuVbnj17dr1e8z/P1tbWPffcc+HChcwEJP3sz/7sL/zCL6zX62xpe71e/97v/d7v/e7vnT9/fjabAZK47NixY6/zOq/zpCc96eDggPsdO3ZssVi8xmu8xt/93d+dO3cuImzzAIeHh4973ONe/uVfvuu6nZ2dV37lV/7iL/7iu++++8SJE/yHss2/Q2vtq7/6q9/3fd/nL//qzxcb81d91Vd+4hOf8JM/9RM/9EM/8KEf+iFv9EZv/O3f/u1nz57H2traufba61erFSDpxIkTh4eHmSmJ/2i2bdsGJEnifyKA4KoXYGdnZxiGzKy17u/vP+MZz5jNZqdPn+Z/HkmPecxjWmu/8iu/Ynu1Wn3Hd3zHq73aq33rt37rbbfdNrVpf3//cz/3c8+dO/erv/ar7/AO7/CMZzxjb28vM7ksM1/plV7pu77ru3iA13/917/llluOHz/+4Ac/+I477mitAba533q9/uu//uvt7W1gsVh8/Md//Ed+5Efu7e31fc+LQJIkHsC2bZ6HJEmS+DeJiF/7tV/b29trrV28eOHbvu3bvvqrv+rLvuzLvu7rvu7s2bNnz5799m//jh/8wR8Yp2ljY6OUcvfdd2POnTv30z/90z//8z9/dHTE/18AlategOVy+e7v/u4/8iM/cuONN5ZStre3H/e4x91zzz133nnnmTNnSimS+B/j5MmTH/mRH/nDP/zDb/M2b7OxsfExH/Mx4zi+4iu+4ld91Vd96qd+6h/90R+95Eu+5Bu90Ru9/uu//tu93dt9yqd8ysWLFz/90z/dNpfddNNNGxsb0zQBgG3bv/RLv/TKr/zKx48f//Iv//Kv//qvv+6663iAzMxMSYCk+Xx++vTp06dPR4RtQBL/A7TWvvEbv3G9XktqrZVSWmu11gsXLnzRF33R8ePHP+iDPujFX/zFSymSjh8//mu/9msv/uIvfmzn2Ou93usdHBwcHR3t7OxI4v8jgOCqF+AN3/ANv+/7vu+N3/iNX/3VX/2GG274q7/6q/39/dd8zdd86lOf2lqTxP8ktdbHPvaxi8Xi7/7u7yJisVgsFotXeqVXeq3Xeq2v+qqv+sEf/MHZbDafzzc3N1/8xV78C7/wC0spv/u7v9tasy1psVh8+Id/+Ld+67ceHR0Bkmaz2V//9V///d///V//9V+fPXv2T//0T6dp4nkZ21xWa73hhhvW67UkSfwPYDsibr755kc84hEPe9jDHvnIRz784Q9/1KMe9bCHPeylXuqlvvVbv/UrvuIrXvIlX3JjYyMiJJ04ceLs2bPr9VqhM2fOvPEbv/FTn/rUaZr4fwoguOoFePmXf/nHPe5xH/qhH/pu7/Zun//5n/8Jn/AJn/7pn37q1Klz585lJv/DSDp+/Pjbv/3bf/u3f/tqtXK67/v5fP76r//6d91119Oe9rT1eo0Baldf8iVf8gu/8At/9Vd/dbVatday5TAMN9xwQ6314OCgtWa7lPL+7//+99xzzzu8wzt8+7d/+x/90R+dO3cuM23bBiJisVggrpAkKTMl8T+JpIiQFBERIUmSpNlsdubMmZMnTy4WC0mSgMVicc8991y6dCkzu647tnPs6U9/eq2V/6cAgqtegOVy+Vqv9Vpf8zVf89Vf/dXf+I3f+IZv+Ibb29sRcffdd7fWbPM/TK31UY961E033fQTP/ET62EtaZqmpz/96ZcuXdrb2/ue7/keY0lAKeXlX/7lDw4OlsulpGEY/viP//iP//iP3+Zt3uazP/uzn/a0pz3+8Y//lE/5lHvuueeOO+7Y3Nx8yEMe8rZv+7a//du/PQwDz8lp7peZBwcH8/mc/zEkSeKFkiQJyMyHP/zhL/MyL3Px4sWIwMzms9ba/v4+/08BBFe9AJcuXXrQgx706Ec/+iVf8iVvueWW2WxWa93Z2em6zmn+6wjEi+bYsWPv9V7v9Uu/9Et33nlna+2uu+763d/93Uc+4pFv8zZvc8011/zFX/wFkJnjOM5ms/d+7/f+xm/8xmmaaldvuumml3/5l7/h+hue9rSnffAHf/CFCxfe5V3e5Yu+6Iv++I//2HbXdY961KNuv/32e+65ByMJaK1dunRJIe7XWtvf348I/ncqpRw/fvzlXu7l/uAP/qC1dvbc2fPnzx8/fvzs2bP8PwUQXPUCtKldc801EcH9bD/84Q+/dOmSsST+5ymlPOhBD/q0T/u0L/3SL12v18ePH3+f93mfz/28z/2sz/qsz/7sz/7N3/zNaZxsnz179ulPf/pLvdRLnT9//mlPe1op5eEPf/h8Pu/67tGPfvQ999zzEz/xE13XLZfLRz7ykYvFAtja2nrkIx/5W7/1W+m0DZw/f/7mm2+WJEkSYPtBD3qQJF4EtnlOkiTx3yoiHvrQhx4eHgLnzp371V/91a/8yq/8/M//fP6fAgiuegFsnz59mud0/Pjxs2fPTtNkm/86wuJF03Xdgx/84Nd8jdf8u7/7u62trWPHju3s7Bw7duz666/f2Nj4y7/6S9t//Md/HIpa69u93dt9+qd/+u7ubmvNtqQ3eqM3ms1mj3rUo/7gD/7gVV/1Vd/5nd8ZuOuuu2699daHPOQhf/M3f/PUpz41M1tr58+ff/mXf3lJXGY7Il7yJV8yM3nR2OZ/htaa7cxcLpfr9Xq1Wk3TdPbs2Zd8yZd8u7d7u/39/b29Pdv8vwNQueoFEV3XSeJ+tdTrrr1OkiT+p7K9WCze9M3e9NM//dMf8pCHXHPNNVxWSnnsYx/7OZ/zOV/xFV/xBq//BhsbG8ArvMIrvPiLv/h3f/d3f8iHfEhr7fGPf/xP/uRPttZsv9VbvdXm5uZ6vf6d3/mde+6559KlS6/wCq9w7bXX/sRP/MQHf/AHA3/2Z3/2Tu/0TpK4rLX2pCc96aEPfagkXgS2+R9jmqZ77733j/7oj3Z3d8+ePXvfffdN0/SKr/iKXde9xEu8xCu+4iteuHBha2tLEv+/AARXPUBrbZqmzDw6Ojp79uw4jkBmjuNo2/i6668rpfBfw2AAEIh/je3t7Q/6oA/6hm/4hvV6bXu9Xv/qr/7q4x//+Hd6p3f69m//9lJLZiLm8/m7vuu7/vRP//TP//zPf97nfV6t9XM+53NOnz79xCc+cT6fL5fLP/7jP37FV3zFt33bt334wx/+sIc9bLFY/NRP/dTv/d7v3XfffXfffffx48e5X2vtH/7hH/q+l8SLxjb/rWxnpu2nPvWpX/3VX33q1Km3equ3etu3fdthGIZh2NjYkNT3/aMf/egf+7Ef4/8jgMpV98vMO++88/d///dXq9X+/v5Tn/pUSWfPnj1z5swNN9wwm82uveba+WJum/8aAvOvJQnouu6hD33oxYsX//zP//wVX/EVgQsXLrzTO71Ta+1HfuRHvumbvul93ud9tre3V8vV3t7e7u7u537u50bEx33cx11zzTVf9EVf9HM/93OXLl365V/+5Td6ozc6derUwcHBMAyZubW19V7v9V6/8zu/88u//Mvb29uLxQIAbA/DcPfddz/2sY/lf49hGFarVa31V37lVx772Me+2Iu9WN/3Z06fec3XeM27776767rZbNZ13Ww2m6bp7Nmz1157Lf+/AFT+f7MNYNL51Kc+9Yd/+Iff4R3eYXNz0/b58+d/8Ad/8Gd+5mde93Vf9wlPeMKdd9557bXXXnfddcvlsrXGfw0BgJH5V9rY2PiUT/mUL/3SL33kIx95/Njx137t1/7DP/zDpz3tabPZ7Od+7uee8IQnvPVbv/Vf/uVfHh0dve/7vu+3f/u3D8Pwi7/4i6/5mq/5Ez/xE6//+q9/7733vumbvumZ02dC0Xf967/+6//1X//1W77lW+7s7Pzu7/7uz/7sz544cWJ/f397e1sSl43jaJv/8WwD586d++M//uNTp05dvHhxHMc/+qM/mqZpY2Pj3Llzf/EXf/EDP/gDD3/4w2+++eabb775tV/7td/1Xd/1V37lV9793d9dkiT+vwCo/P+WmYeHh8BqtfqBH/iB133d1334wx9+7733/viP//iTnvSk3d3dU6dOvcu7vAtwdHS0v7//u7/7u+fOnVsul/yXMpgXmSSglHLs2LFHPvKRv/mbv/nWb/3WP/ZjP/bar/3ar/7qr/4u7/Iu586d+/Zv//ZP/dRPfdVXfdXP/dzP3Vhs/MVf/MUjH/nIX/mVX/n7v//7hz3sYS/7si+7ublZSglFOvtZX0o5e99ZP9qSSikf//EfP03TNE3jOHZdB/R9/4iHP8I2/+PZXq/X3//93/9Gb/RGt9xyy2q1esITnnD33Xc//OEPf/jDH/5Lv/RLtdYP/uAPPnny5HXXXfc7v/M7n//5n//+7//+3/qt3/o6r/M6119/fa2V/y8AKv8v2W6tHR0d/d7v/d6lS5e6rrv11lt/53d+5xM/8RPX6/W3f/u3/9Vf/dXbvd3b7e3t/d7v/d6Tn/zkxz72sSdPnjx16tRDHvKQcRxt81/HYP5NZrPZa77ma/7AD/zA27z127zJm7zJox71KEDSmTNnPu3TPk3Svffee++99y6Xy8c+9rEf/dEffe+99166dOnRj370bDZrrQGIK46WR7ffcftytSz75cEPfvANN9zQdV1rLSK4rOu6F3+JF7/vvvsk8T+P0wrZBlpr+/v729vbN9xww+bm5ubm5nu913vdfffd3/3d3/2Qhzzkvvvue7u3e7s3fdM3xSBOnTr1xCc+8Wu/9mtvueWW7//+7//4j/94/h8BCP7/sT1N01133fULv/ALEfEGr/8Gr/u6r/tGb/RGx48fL6X84i/+4sHBwYd8yIe88zu/8wd90Ad9zMd8zLd927f91m/91t13391ak/Rqr/Zq58+fb63xP5ukUsr1111/9913T2165CMfKUkSUGs9derUp3/6p7/hG77hd3/3d//ET/zEe7zHe4zjKOnFX/zF+763HRERIUmSpLNnzy4Wiz/6wz/6+7//+1OnTs1ms4iotUYEl0XELbfc8g//8A+8yCTxX0asVqvbbrvtGc94xjRNEXH+/PlpmgBJJ06ceKu3eivbr/Zqr/bu7/7ur/mar5kt02l7a2vrYz/2Y1er1Ud+5Ec+/elPP3v2LP+PAFT+P7FtW9LR0dF3f/d3v//7v//111/PZYvF4syZM+fPn3/iE5+4s7Nzww03SIqIl33Zlz179uzXf/3XX3/99e/6ru/6Cq/wCrXWP/zDP3z0ox9da+V/PIVaa4973ONe7uVejssyMyIkHT9+/N3f/d0PDg5qrcMwfPM3f/MbvdEb3XTTTZJs8wAHBwe/8Au/8Oqv/up/8zd/k5n/8A//8Oqv/uo8j9baE5/4RF40kvgvNKyH3/zN3/zTP/1TSa/2aq/22Mc+9mlPe9r58+dPnToFRMSjH/3o06dP33vvvW/6pm86n8+NL5y/sLGxMZvNjh8//omf+Ilf+ZVf+R7v8R6///u//47v+I62JfF/H0Dl/xPbR0dH+/v7j3/849fr9ebmJvcrpbziK77in/3Zn0l65Vd+5R/4gR94q7d6q5tvvnk2m128ePEzP/MzgR/7sR974hOfeOONNz79aU+fpon/Debz+Vu+xVt+zud8zo/88I9sbG7YBmwfHh6uVqthGIZhGMfxL//yLzPzwQ9+cCkFkMT9jo6Ofv3Xf317e/vMmTOPfvSjI+Lg4ACwzQNkZmZO05SZEcGLQBL/VZ5x2zP+4R/+4aM+6qNsf+/3fu8999yzt7d32223PeQhD+n7PjN3dnZe4iVe4pu+6Zt+7dd+7W3f5m03tzaf9KQnvf7rv/7DHvawUspNN910/fXX/8Zv/MZyuXyrt3qrWmsphf/7ACr/P9jOzN3d3Z/4iZ/4h3/4h4c97GFPe9rT/v7v//5VX/VVJXHZQx/60Gfc+owbbrih7/uP+qiP+umf/umf//mfr7WeOXPmJV/yJUspj3rUo57+9Kd/xEd8xM0335yZ/G9QSnmt136tH/rhH3ryU578Ei/xEsBqtbrj9jv+7M//7GlPe9qJEyduvvnmRz3qUa/92q/9dm/3dpIkAbYzMyJaa3/xF3/xq7/6q5/5mZ85m82e9rSnvcqrvMr+/r5tHsD2OI5/8zd/84qv+IqAbUm8UJJ4TrYBSfwn+J3f+Z0P+IAP2N7etv2BH/iBv/VbvzWO42/91m894hGPuOmmmw4PDg8OD97kjd/kZ3/2Z48dO7a3v/cN3/gNJ0+efJVXeRVJknZ2dj75kz/5i7/4i//2b//2/Pnzp0+fjghJ/B8HUPl/4+Dg4Hu/93tPnTr1GZ/xGfP5/E3e5E0+8iM/8sd//Mc3NzeBruse/OAHP/GJT3zXd33XD/mQD/noj/7o93u/95MESKq17u/v33777V3XdV136dIl2/xv0Pf9zs7Oq73aqz3jGc948Rd78XT+0i/90sMe9rC3equ32tzclMT9bHO/cRxDgdjd3f21X/u1j/u4j7vuuutWq9XFixdvv/3213u917MNSAIAScM4/MiP/MjHfMzHALYBQBL/M9iepsl2KaXv+9d7vdertf7t3/7tD/zAD3zIB3/IE574hOuvv/7MmTPv8i7v8v3f//2f9EmfdMstt3z+539+ay0UQClle3v7Yz/2Yz/t0z7tzjvv3Nra6rqO//sAgv/rbNterVZ/8zd/89SnPvX1X//1T506tbm5ecMNNzz2sY998pOfzP2uv/76e+65Z3t7+4M+6IO+7uu+7uzZs4AkLvvt3/7tO+644/M+7/Pe7/3ebzab2eZ/ib7v3+u93uvixYvG69V6GIaXeqmXms1mrbXM5AFst9YuXrz493//99M0Gf/5n//5S77kS954442ttfl8/pCHPKTWGhGSJPEAq/WwHNY3P+iWsTXE/zTL5fJP/uRP1uu17VprrfXVX+3Vb7zxRtt/9/d/97Iv+7IPuuVBs9nsFV7hFba2tv72b//2L/7iL06dOvWHf/iHQGYCkk6ePPmJn/iJ3/Ed3/F7v/d7ly5dss3/cQDB/wPDMPzlX/7lP/zDP3ze533etddeK0nSfD7/iI/4iF/5lV/hMkmllMViMQzDIx/5yJd6qZf61E/91HNnz61Wq2EYMvPixYt/8zd/8zVf8zVv+ZZvOZ/Px3HkfwlJi8Xiz/7sz/b29v7iL/9ia2srM0spESHJNvfLzPvuu+/DP/zDn/rUp6YTuPPOO1/mZV5mY2Oj1irpjd/4jV/hFV6B+0mSZGls7fDw8IYbbtjY2Ky1Wkohif8xbrzxxr7vL126tFwup2kqpSw2Fm/3dm/3qEc96vd+7/d2d3fHaWytnThx4q3e6q0+93M/93Ve53W+4iu+wvbUpogAJJVSHvrQh37u537uwcHB137t17bW+D8OoPL/wG233fZ7v/d7b/AGb3B0dHTs2DEuK1E2Fht/9md/ZhuQJGkYhsz8hV/4hZ2dndd93df92I/72NNnzrzcy77sy738y7/zO79zrbW1FhEv9VIvdXh4aBuQxP94Xde93uu93o//+I+/+7u/e0REhG2eU2aeO3fuoz/6o9/8zd/8jd/4jfu+Xx4tx3E8fvw49ytRbEvifgbg4ODgkz/tUz/4gz5IpQBg/od57GMfu7u7+x3f8R2Hh4cf8zEfc/r06VJK13Wv/MqvfHR09Mu//Mtv/dZv/Sd//Cd//w9/P5/PT5w48fu///ullN/5nd95+7d/+5tuukmSJNsRcfz48bd6q7f6rM/6rIsXL545c4b/ywAq/9dN0/SkJz3p6U9/+lOe8pRv+ZZv+cIv/MJTp05JMi61HD9+fBzHrusAYJqm2Wz2ru/6rsMwzGaz1toXfckX//0//MMzbr/9fd7nvW+64cZaK/DKr/zKy6Nla62UYlsS/7N1XffGb/zG7/zO7/wu7/Iu8/ncNs9jmqa//Mu//NiP/diXeqmXms1mwHK13N7e7vveNgAoVFUzUxIgSZCtXbx48fz5cydPnQIbxP8429vbf/RHf/RiL/Zifd9/+qd/+hd8wRecOHGitTaO42/91m9tb28PwzAMwzu8wzucOnVqZ2fnK77iKzJzNpv94R/+4du/3duXWgBJQN/34zg+9rGPve22286cOcP/ZQDB/3X7+/tnz559z/d4z8x8+tOf/n3f+33TNLXWhmGQFBGXLl1qrR0dHbXWFovFk5/85O/8zu/8rM/6rF/4hV/4i7/4i7Tf6z3f8+M+7mOvv/56LmutXXPNNT/38z+3XC7536Prund/93f/kR/5kdYaz8P2vffe+7SnPe3FX/zFa622ASFJ/Esy8y//6i9Xy9VDH/IQXmS2bfMAkiTxn+PEiRO/9mu/9sqv/Mqv+zqv+4Ef+IGf+Zmf+YxnPOMv//Iv/+Ef/uFTPuVTzpw5s1wun/SkJ/V9P03TNE2f8Amf8LIv+7Lv8R7v8Yu/+Iv7B/s8J0mz2Wy1WvF/HEDwf93R0dEdd9zx2Bd77Nu93dt927d929//w9/fddddT37yk7/qq77q9ttvP3Xq1O7u7l133fWd3/md99xzz3q9/pIv+ZI3f/M3/5zP+Zyo5Q/++I/GcfyN3/yN1WpVojjk0DAON9100xOf+MTd3V3btm3b5n+2Wuubv/mbf//3f/9f//VfS5Ikifut1+vP/dzPfed3fueNjY1aa0RI6vputVrZ5gEkSeJ+tvf29n7/d373cz7rszfnC4F4poTkf4r5fP5Kr/RKP/mTP/ld3/1dD3nIQz77sz/7V37lVzJzvV7P5/Npmt7wDd/w9V7v9T7v8z7v67/+6//6r/96tVq953u+5zu90zt94id+4jd8wzesViuMbduA7eVyOY4j/8cBBP/X3XbbbRcuXPjbv/3bf/iHf7h06VLXdT//8z8/n89f53Ve5yEPfsgNN9xw6dKlm2+++QM/8AOPHz8+jmNr7ZZbbun7/pprrhnW69ls9rVf87U7O8eMhYCNxcbW1tbe3t7Tn/70zOR/lbd7u7f7zu/8zmmceADb+/v7d955Z2byAFtbWy/xEi9Ra+WFOjw8fPrTn/4yL/3SzcmLzLZt/quE4q3f+q1/+qd/+q3e6q12dnaOHz/+wR/8wX3f/8M//MPnfM7nPOhBD7r22mvf+I3f+DM+4zNe/dVf/X3f931/93d/95GPfKSk1tonfdInzWazYRhsA0BEPPShD10ul/wfB1D5v261Wo3jWEoZhsH2S77kS/7hH/7h+77v+958881HR0cbGxuHh4d7e3s/9EM/tLu7+7SnPW25XO7t7f3Ij/zIXffe81mf+Vn/8Lh/mKax1iIFl0kAr/d6r/c3f/M3L//yL79YLPjPJRD/Eebz+Qd8wAe8x3u8x5/86Z+86qu+qiTut7Oz8zEf8zFbW1uSuF9EvMIrvEJm8oKN4/h3f/d3r/Zqr3bs+DEk/qeKEg95yENe8zVfs+/7rusk2X75l3/5V3iFV5imaZqmO++4c71eX3fddddcc82v/dqvHT9+/Nprr93b2/uVX/mVhz3sYZubm6UWwDaXXX/99X/yJ3/C/3EAwf91kh7+8Ie/2qu92iu+4iu+0iu90ju90zs99rGPveeeew4ODr7ru77rwoUL0zT9wA/8wCu90iu993u/9+Mf//jd3d2P/uiPfvjDH/6Zn/bpb/02b/O2b/t2P/KjPzqOE8/pNV7jNbqus83/HpL6vv/gD/7gb/iGb2it2eZ+s9ns9V//9ReLBc/jwoUL6/U6MyVJkiQJAGzffvvtP/iDP/g2b/u2SPyP9+Zv/uZHR0e2gcycpmm9Xt93333f933fd+999548eXK9Wj/96U8/d+7cq73aq5VSFovF273d2/3d3/0dIIn7Ser7PjP5Pw4g+P/E9vb29uu+7uv+2Z/92TOe8Yy+79///d//nnvu+f3f//3HPPoxm5ubEXH27NmP/diPfZ3XeZ1a67zvX/IlXvypT31aG0c7A4Jn29vby0z+KwiL/wgR8Wqv+mrnz5//9V//de4nCYgISTyP3/qt3/r1X//19Xptm/tJsn3p0qWv//qvv3DhwrXXXOu00/zPduONN/7yL//y/v7+NE1PecpTvumbvumLvuiLvvALv/AlXuIlXvEVX7GU8ld//Vc//dM//aQnPullX+Zlbc9ms4c+9KGv8iqvAgCSJHFZZkri/ziAyv8DkgDbQETccsst3/md3/kHf/AHb/M2b/PkJz/5937v9570pCd9y7d+y23PuG21Wn3v937vS7zES0zTVEoBQO/x7u/+a7/xG2/0hm+4mM8BwPa3f/u3X3fddRHB/zZd333sx37sl3zJl7zO67zObDYDbEviBXj913/9L/3SLz04OHj7t3/7WiuX2V6v1z/yIz/y67/+6+/1Xu+1tbVp/hc4c+bMX/zFXzz60Y8+PDw8derUB7z/B3R9B9Rap2l6+tOffs8997z1W7/1V3/1V2/vbEviASRxP9vnz58fx5H/4wAq/w/s7+8Pw9D3PVBrvfbaaz/u4z7uh37oh37+53/+lV/5lb/xG7/x7rvv/oqv+Io/+ZM/+a7v+q6Xe7mXA7qu47IS8XIv+7Iv97Iva56ptXbnHXfefvvtH/ERH7GxscF/HnM/gfgPUkp54zd+48z82q/92o/+6I8uURQCJPH8nDhx4ou+6Iv29vYyMzMjArh48eI3fdM3/c7v/M7nf/7n7+7uZqZK4X+DRzziEU996lPf9V3fdT6fSwIA26vV6o//+I9f8zVfMyIe/OAHnzp1CpAkCZDEA2TmX/7lX3Zdx/9xAJX/B+67977lctn3PZdJevjDH/7pn/7ptksptk+cOLG1tfXJn/zJL//yLy9JEs9DYFuS7a/7+q97r/d6r+uvv57/td7g9d/gV3/1V//0T//05V7u5ebzOf+Sra0tSZIA27YvXLjwVV/1VTfffPMv/MIvZGYphX8NSZL4L/cqr/IqtiNCiPtJuueee1ar1ZkzZ+69915AkiReANt33nnnIx7xCP6PAwj+r3vIQx4ym89aa4AkICJKKbXWrusiQtLR0dFdd931Oq/zOrVWSbwATrfW/uZv/ua+++571Vd91Y2NDUmSJEniP5xAAGAw/6G6vvuSL/mSn/u5n/uoj/qo3/u93xvHsbXWWrOdmTyPiJDEZZJOnjz5pV/6pY95zGO6riul3H333bIDwoQJEyYg+B/nhutv+OEf/uG9vb108gA33XTTW7/VW8/6WYnSWstMSZJ4flprT3va017iJV6C/+MAgv/rjh071nXd7bffLonn5+Dg4Ju+6ZtuuOGGzc1NXqh0/sM//MPnfd7nfdZnfdaJEyf4X67v+8/93M9913d51+/7vu/7wA/8wJ/7uZ/7+7//+3/4h38YhoF/iaRSSkREhO1v/uZvHseRfw1JkngA27b5z2T75ltu3tvb+53f+Z31es0DzOfz02dOK1S7ur+/v1qteMEy8957732xF3sx/o8DCP6vq7XWWp/2tKdlZmuN5yHpb//2bw8PDzOTF+rw8PBTPuVTXvM1X/O6664rpfBfx/wnkNT3/au/xqt/2Zd92e7u7md8xme88zu/88d//Mfv7e3xIuv7/uVf/uX/5m/+5p577rHN/2yllK7rPviDP/hXf/VXL168mJm2uUwSl83nc9t33HFHZmZLnsc4jn/5l3/5Jm/yJrVW/o8DCP6vWywWj3zkI5/85CcPwxARPI9SyqMf/ejZbJaZvGAHBwef8zmf89Zv/dYf9EEf1Pd9ZvJ/QillZ2fnoz7qo17xFV/x13/913/0R370xIkTtnmRXX/99W/5lm/53u/93pcuXQIASZL4H0nSy7/8y585c+YpT3nKNE62bXM/Sdvb26/6qq/6BV/wBU960pOMeR7DMPzGb/zGy77syzrN/3EAwf91km655Zbf+Z3fuffeewHbtnmAvu/f+I3f+N577y2l2LbN8xiG4Qd/8AfHcXz7t3/7xWJRSokI/veTJEnSwx/+8Kc85SmStra3aq2SeNFIms/n7/qu7/pyL/dyX//1Xz8MgyT+Z5vP5+/zPu/zMz/9M+fOnwMA27a5LCJe7MVe7K677nq3d3u3L/uyL1seLTOTB2it3XfffS/5ki/ZsvF/HEDwf10p5ZGPfOTGxsav/dqvZSbPo5Tyci/3cp/3eZ8niRfg537u537xF3/xkz/5k3e2dyTxf0tmXn/99Z/5mZ/5sz/zs+v1mn+liNje3v7ET/zEJz7xif/wD/9gm//xbrnlltrVv/u7vxuGwTYPEBGbm5uv8iqv8uVf/uV33nnn3v6eJB6g1rqzs3P+/PlSCv/HAQT/19ne3Nx8hVd4hb/7u78bhoHnIWk2m508efJJT3rSpUuXMpPLWmu21+v1n/3Zn337t3/7R3/0R19zzTVRwjb/h9iOiFrrYx/72F/9tV/d29trrdnmXyMiTp069bEf+7Ff9mVftl6vbfM/W9/3H/3RH/0DP/ADv/Zrv7a3t2fbaaezJbC9vf3ar/3av/SLv/T5n/f5x48f5zl1XfeGb/iGf/qnfyqJ/+MAgv/rMrPv+3d7t3e75557br31Vp4fSaWUv/zLv3z/93//3/+937/33nvvvffeO+644/u///s/9EM/9Iu/+Is/8zM/85Vf+ZVrrYAk/g+RJAk4ffr0Yx/72Hvuuae1xr9erfXhD3848Nu//dvTONnmhbLNc5Ikif8qp0+f/pzP+Zyf+7mf+7RP+7Qf/dEffcpTn3L3PXev1iug7/vXeI3X+MzP+syt7a2+73lOtdaXe7mX+/7v/37b/B8HUPm/LiIknT59+l3f5V2/7uu+7ou/+Itns9lsNgMy07bTh0eHmfkSL/ESu7u7n/4Zn761tTWO4zAMFy9eXK/XP/mTP/mIRzyi1mobkMT/RRHxLu/yLl/1VV/1ZV/2ZX3f25bEv8bm5uZHfdRHfdRHfdSjHvWoBz3oQVwmiRfANv99aq0333zzZ37mZ37GZ3zGN37jN67X61rrS7/0S3/O53zO6dOnZ7PZbDbDPC9J8/n8sY997N7e3vHjx/m/DCD4v06SpK7rXuqlX+qJT3zi+7zP+/zqr/7qk570pCc96UlPetKT/vqv//ojPvIj3vVd3/U7v/M73+d93mc2m11//fXz+by1dvPNN3/f933fox/96Ac/+MFd10ni/7SIeOhDH7q7u7s8WtoGbPOvUUp51CMf9ZCHPOTP/uzPMpMXyjb/rSSVUq699tqv+qqv+qqv+qrt7e2P/diPfad3eifbtrlCPF+11jd5kzf5u7/7O/6PA6j8PyApIjY2Nh772Mc+/OEP/5RP+ZS+72ut4zhubGz0fX/x4sVf/uVf3tnZeamXfKm3eeu3GcZhtVptbGycPHnypV7qpSJCEv8PzOfzz/iMz/i8z/+8L/zCLzx+/Dj/SpK2d7Y///M//1d/9VcBSbxQtvlvJanruuPHj7/4i7/4+73f+91+++1v8AZvsLGxIYkXKiJe/MVf/Fd+5Vde4zVeg//LACr/P0iSVEp55Vd+5T/+4z9eLpev+7qv++Iv/uLb29snTpz4qI/6qMc85jF33nnnIx/1yK2tLUlAZgInT54spfD/xkMf+tCTJ0/ee++9W1tbtVb+lWqtD33oQ9/xHd8xIvjfo+/7l37pl/6Wb/mWYRh2dnb4l0g6fvz4Pffcw/9xAJX/T1ar1bFjx77lW76ltTafz+fzObC3txcRkh7ykIdsbGzYlgQAz3jGM172ZV5WEv8PSAK2trY++qM/+sKFC7b5N4mIU6dOSeJ/D0nXXXfdDTfccHh4ePr0aV4EEXHs2DH+jwOo/D9Ta93Z2eF53HTTTUBESOIy2099ylNvuvkmSfy/IenMmTPHjx8vpfBvJYn/bba3tz/wAz9wNpvxoqm1vuu7viv/xwFU/q+zbTsiJGXmwcFBay0UiCsys+u6m2+++a677gJsc5mk1tp99933sIc9DJBkm/+dMlMSYBuQxAslJAnITJ6HJEn8b3N4eLharTKT5yEpW7Zs4zgeHh7ygkVE13VbW1sRsbGxYVsS/2cBVP6vsz0Mw7333vsP//APZ8+e/du//dtLly5lpiRA0jRNwzDceeedt99+++///u/btg201h73+MdN0zRNU62Vy17yJV9ye3s7IvhfpbW2XC6/53u+Z7Va1Vr5l2RmRAC2eQDbEfH2b//2N998M/+rrNfrL/7iL75w4UKtlechKTMzs9YqiRcsMx/+8Id/8Ad/cK21lCKJ/8sAKv+z2QYkZWZE8Dxs8zwkcT9J995771d91Vc9+MEPfu3Xfu39/f0nPvGJgCQus/3Gb/zGkm6++eYnPvGJgG0gM7uum81mT3ziEyVJ2t/f/7Vf+7VP+ZRPWSwW/LuVUngR2BaB5DT/Ets8P5n5u7/7u7fffvsrvdIrSeJ+koA//sM/+okf+7Hrr7/+oz/+45ZHyy/9oi9aLpcf/lEfee111/V9z3P627/92x/6oR/6xE/8RP6tbLfWgNlsJokXynZm2ubf5/Dw8HGPe9wXfuEXbm1t8fx8//d+77d9y7fyAJ/y6Z/2Sq/8yidOnOABDg8PP/zDP/zDP/zDbUcE/ya2bdu2jfkfDKDyP1trre/7w8ND2zwP27wIzp8/X0p5r/d6r62tLUk8wGq1euM3eIN77r7nu773e17+FV7hTd/ojW+/7bZv/OZvfrmXf7ljx4/bltSmplBE3HPPPZ/2aZ82DMNiseDfwfZqteJF0/d9KeXOO+4cxnE2n/Fvkpn7+/sv8zIv83Zv93a2uZ8k4G3e5m3uufuuP/j9P7h08eITHv+Eo6Oj93nf9/2oj/5onoft+Xz+l3/5l/z7HBwcAPP5PCJ4wWxLigjbwzDw79Ba29zcvO66644dO8bz87qv93qbm1vAk5/0pF/4+Z/f2tp6rdd6rRtvummxWPAAly5dkiRJEv8OR4dHrbX5fK4Q/3MBBP+zlVIe+chH/sVf/MXh4WFmZqbTTjvtNC+AbR7Atu1SSq21lFJKKaWUUkopm5ubn/v5nw98xZd9+U/95E/eftttr/lar/UGb/SGJ0+dKqXUWksp/azvuq6UMpvNJEni38p2a213d/cZz3jGi73Yi/EiOH7s+ImTJ37+F37+3nvvnabGv0lE2JbEC/D5X/iFG5sbX/NVX/2TP/ETD3rwgz72Ez6e/xy2bf/pn/7parV6yZd8ya7reMFs11pPnz593333nT9/PjOd5j/HK7ziK374R37EW731W/35n/95rfUbv+WbH/6IRywWC/6j2V6v13fedaekEydOSOJ/LoDK/2yllNd5ndf5mq/5mm/+5m9+j/d4j53tnX7WSwIASfxLbGemJNs8P6/zuq/7Vm/z1j/zUz/99Kc9reu6z/7cz+EFy8zW2jRNkvhXaq1N03T27Nmv+IqvOH/+/Md+7MfyItja3nybt3nbb/7mb/rSL/my9/+A97/hhusjQkLiudi8IJl58eLFU6dOAZK4n20uu+nmmz/6Yz7mCz//C4DP+bzPn8/ntnm+xNHR0T333MMLI54PA8uj5ZOf8uQv+IIv2Nraevd3f/eu63jBImLWz17uZV/uZ3/2Z7/u677u8z//80+cODGbzfjPcd+9977Xe7znvffc82Vf8RWv+mqvxn+CzDw8PPyTP/mT3/md37n55puvu+4625L4Hwqg8j9brfW93/u9//RP//S7v/u7/+AP/uC1Xuu1rr32WkkAIIkXQBKXTdN0xx13LJfLiJBkm+fxGq/5mj/zUz99eHj4lm/1VjffcgsPIIn72d7b2/uZn/mZrusk8a8k6e677/6VX/mVe++99zVe4zXe8A3ekBeBFO/0ju9469Nv/fVf//UP/7APXyxmUUICzL9MIGCaptbaR3zER/CCvcEbvdEXfv4XbG9vv9qrvxovmOC3f/s3//iP/xAMQIAAEAgAYZ6XBNj2hYsXjh8/9gmf8AlnzpwppfBCRYlXeMVXeMVXfMU/+IM/+KiP+qi3fMu3vPHGG0spkvjXuHTp0nq95gXbvXjxvd/zPW97xjM+6mM+5m3e7m15waZp+oM/+IPM5F/v8PDwb//2b3/6p3/a9vu///sfO3ZMEv9zAVT+Z5O0vb39dV/3dd/4jd/4l3/5lz/2Yz+WmZIAwDYvgCQus71arV7hFV4hFDw/R4dHX/XlXwFsbW394i/8wvt/0Ac+5jGP4fnJzPPnz3/DN3wDIIl/jcyMiK7rTp8+/YEf+IHv9E7vtLm1yYtA4viJnU/9tE96yZd68V/91V+/796zmYl5PiSJ52YhZ78ex5HnIYn7SeIySbwwEepqmQMgXAApQFIAEDwfFhDuan25l3ulD/nQD77l5htqV3kRnDhx4lM+5VO+7uu+7g/+4A++4iu+otZaSimlSOJFI2mapmuvvZYX7OM/9uOe9MQnnTx5UtLXf+3XAS/38i/34i/xEtvb2zynaZo+8RM/kcsyk3+NzAROnTr14R/+4a/+6q/edR3/owFU/seTdOLEiU/91E/d3d29ePFiZnI/2zw/koS4LDOf+KQn/vqv/3rLZpvn8XVf+zV33nnnG73xG7/qq7/aZ336Z3zGp37aj/z4j0WEJMA2D3Ddddd9xmd8xnw2R0jiRWPbtqSu67a2tk6ePDlNkyReZFtbW+/wDm/3lm/xVsvlsk2JxfMQIJ4PeWzrn//5nx/HcbVaSeL5GYaBy9brNS+A7WEYXvM1X/OjPvIjceAAAAmQxAsjkKEcP7ETEaXwIpJ07bXXfvqnf/rZs2ef/vSnX7p0aRgGSZJ4kR0eHv7UT/3UxYsXp2ni+bnjjtuBCxcufPVXfiWXvd8HvP/2zs51113HAxwcHGTmR3/0R9da+dfLzIc85CE33XTTiRMnuq6zLYn/uQBkm/9VbHM/SfxLbP/93//9t3zLt7zv+77v1taWbR7g/Lnz7/Oe71lK+cZv+ebrrr/+Iz/0w572tKe9z/u932u/zuucPnOa53Tfffd9//d//5d/+ZdvbW1J4l/JNmCb+0kCJPHCmSuMwSAsnh+J5yYw4zT+7u/+3k/91E++7Mu+7Gw2s52Z/KsJ+Ku/+osHP/jmj/jID4FCzgBAAlkCzAvhQPy3ODw8/KiP+qjDw8PDw8PM5Pmxzb+ktXbq1Knv+Z7vKaUAkvi/DEC2+d/GNgBI4kVw6dKlH/uxH/u1X/u1iGit2ebfJDO/8Au/8CEPeUjXdZL4V7LN8yOJf5lBNpgXSIjnJADAZrVa//Vf/+Vv//Zvt9YASZIk8a8REdddd91bvsVbHj9+ggeSwRJgXggXDOKZxH8Z23t7e/fdd99v/uZvjuPIC2CbF6rv+zd5kze58cYbIwKQxP9lALLNVf+FbHM/SfzvZJsHkMRV/wcByDZXXXXVVc8HQHDVVVdd9fwBBFddddVVzx9AcNVVV131/AEEV1111VXPH8A/AvYGpedQYgSlAAAAAElFTkSuQmCC"""
_TEXTOSENAS_NUMEROS_SHEET_B64 = """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GIYv+uIv+vzP//zVavX3f//3x44de/jDH761tfXgBz/4nd/5nf/sz/6M/5Gy5T/8wz+8//u//6//+q9n5jiOJ06cuO2222yv1+uu617mZV7m13/911/lVV7lD//wD//iL/7iZ37mZ574xCc+5CEPuXTp0o033rharUopfd8/7GEP+4d/+IejoyP+91sul6WUiGitRcRyuXzCE57w13/918Dv/d7v/dZv/dadd9556dKliPjZn/3ZxzzmMX3fl1JsHxwc8CIYhuHMmTM33HDD27/923ddx2Xz+fz48eOSeE6Z2VqzPU2T7WmaANv8L3T77bdv72z/xV/8xXq97roO+JVf+ZXv/4Hv/7u/+7uDg4Mf/dEf/fqv//rHPe5x3//93x8RwHK5fJmXeZmIiIi//uu/BiRxWWYC119//UMe8pBHPvKRtrms7/v1es1zmqYpM20vl8tf+qVf+r3f+73VagW01lprtm3b5r8MBP9BMnN/b/8lXuIlfuZnfuZVX/VVf+mXfun48eN33nnnK77iK77u677u273d273aq73aU5/61Guvvfaee+55rdd6rcz88R//8ac//ekbGxsRMY6j7Yc+9KHf+Z3fmZn8D2Mb8ZM/+ZPf8z3f84M/+IPf+73fW2uV9PIv//J33XVXKUXSG7zBG/zN3/zNYx/72D//8z8/fvz4arW67bbbXvqlX/rbvu3bNjY2hmGw/aqv+qpv93Zvd+bMmfvuu4///WwvFovWmm3M0dFRa+2uu+6yfdttt73iK77i6dOnp2m6++67n/SkJz360Y+WBFy8ePG6666b2gRI4gWw3fd9LfXUqVO1VklcVkqJCMRzsf3EJz7xkz7pk97szd7sXd7lXT7xEz/xN3/zN4E2Nf73aK211n7/93//NV/zNT/gAz6g1lpKueuuu86cOXPDDTecPn36O77jO97nfd7n8z7v8971Xd81M3/hF37B9nq9fp3XeR1A0rlz50oU7nfvvffa3traOnnyJA8QEbPZTBL3y8yIWB4tv/mbv/lN3/RNv+/7vu/7vu/7PuiDPuhbv/Vbj46OJNm2DdjmvwYE/0EiomW7dOmSpI/4iI/47M/+7GEY+r6/dOkSl73ES7zEn/3Zn11zzTV33nnny73cy33AB3zAJ3zCJ0gCuq77/d///c///M9/3OMe98QnPnG1WvE/T2Yul8uNjY1v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"""

SIGN_IMAGE_ALPHABET_LABELS = [
    "A", "B", "C", "D",
    "E", "F", "G", "H",
    "I", "J", "K", "L",
    "M", "N", "Ñ", "O",
    "P", "Q", "R", "S",
    "T", "U", "V", "W",
    "X", "Y", "Z",
]
SIGN_IMAGE_SUPPORTED_NUMBER_TOKENS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "100", "1000", "1000000",
}


def _escribir_archivo_base64_si_falta(ruta, contenido_b64):
    ruta = Path(ruta)
    if ruta.exists() and ruta.stat().st_size > 0:
        return ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(base64.b64decode(contenido_b64.encode("ascii")))
    return ruta


def _recortar_contenido_blanco(imagen, padding=6):
    if imagen is None or getattr(imagen, "size", 0) == 0:
        return imagen

    if len(imagen.shape) == 2:
        gris = imagen
    else:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    mascara = gris < 245
    if not mascara.any():
        return imagen

    ys, xs = mascara.nonzero()
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(imagen.shape[1], int(xs.max()) + padding + 1)
    y2 = min(imagen.shape[0], int(ys.max()) + padding + 1)
    return imagen[y1:y2, x1:x2].copy()


def _detectar_celdas_alfabeto_desde_lamina(ruta):
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return []

    _, umbral = cv2.threshold(imagen, 210, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        area = w * h
        if area >= 4000 and w >= 60 and h >= 55:
            rects.append((x, y, w, h))

    rects.sort(key=lambda item: (item[1], item[0]))
    if len(rects) < len(SIGN_IMAGE_ALPHABET_LABELS):
        return []

    return rects[:len(SIGN_IMAGE_ALPHABET_LABELS)]


def _generar_assets_alfabeto_desde_lamina(ruta_lamina, carpeta_destino):
    imagen = cv2.imread(str(ruta_lamina), cv2.IMREAD_COLOR)
    if imagen is None:
        return

    rects = _detectar_celdas_alfabeto_desde_lamina(ruta_lamina)
    if len(rects) < len(SIGN_IMAGE_ALPHABET_LABELS):
        return

    carpeta_destino.mkdir(parents=True, exist_ok=True)
    for etiqueta, (x, y, w, h) in zip(SIGN_IMAGE_ALPHABET_LABELS, rects):
        recorte = imagen[y:y + h, x:x + w].copy()
        if recorte.size == 0:
            continue
        cv2.imwrite(str(carpeta_destino / f"{etiqueta}.png"), recorte)


def _generar_assets_numeros_desde_lamina(ruta_lamina, carpeta_destino):
    imagen = cv2.imread(str(ruta_lamina), cv2.IMREAD_COLOR)
    if imagen is None:
        return

    carpeta_destino.mkdir(parents=True, exist_ok=True)
    alto_total, _ = imagen.shape[:2]

    filas = [
        (0.00, 0.19, ["0", "1", "2", "3", "4", "5", "6"]),
        (0.19, 0.41, ["7", "8", "9", "10", "11", "12", "13"]),
        (0.41, 0.61, ["14", "15", "16", "17"]),
        (0.61, 0.83, ["18", "19", "20", "21"]),
        (0.83, 1.00, ["100", "1000", "1000000"]),
    ]

    for top_frac, bottom_frac, etiquetas in filas:
        y1 = max(0, int(round(alto_total * top_frac)))
        y2 = min(alto_total, int(round(alto_total * bottom_frac)))
        fila = imagen[y1:y2, :].copy()
        if fila.size == 0:
            continue

        ancho_celda = fila.shape[1] / max(1, len(etiquetas))
        for idx, etiqueta in enumerate(etiquetas):
            x1 = max(0, int(round(idx * ancho_celda)))
            x2 = min(fila.shape[1], int(round((idx + 1) * ancho_celda)))
            celda = fila[:, x1:x2].copy()
            if celda.size == 0:
                continue

            limite_superior = max(1, int(round(celda.shape[0] * 0.76)))
            signo = celda[:limite_superior, :].copy()
            signo = _recortar_contenido_blanco(signo, padding=6)
            if signo is None or signo.size == 0:
                signo = celda
            cv2.imwrite(str(carpeta_destino / f"{etiqueta}.png"), signo)


def _asegurar_assets_laminas_texto_senas():
    global texto_senas_sheet_assets_ready
    if texto_senas_sheet_assets_ready:
        return

    try:
        SIGN_IMAGE_SHEET_DIR.mkdir(parents=True, exist_ok=True)
        SIGN_IMAGE_SHEET_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        ruta_alfabeto = _escribir_archivo_base64_si_falta(
            SIGN_IMAGE_SHEET_DIR / "alfabeto_referencia.png",
            _TEXTOSENAS_ALFABETO_SHEET_B64,
        )
        ruta_numeros = _escribir_archivo_base64_si_falta(
            SIGN_IMAGE_SHEET_DIR / "numeros_referencia.png",
            _TEXTOSENAS_NUMEROS_SHEET_B64,
        )

        recursos_clave = [
            SIGN_IMAGE_SHEET_ASSETS_DIR / "A.png",
            SIGN_IMAGE_SHEET_ASSETS_DIR / "Ñ.png",
            SIGN_IMAGE_SHEET_ASSETS_DIR / "Z.png",
            SIGN_IMAGE_SHEET_ASSETS_DIR / "0.png",
            SIGN_IMAGE_SHEET_ASSETS_DIR / "10.png",
            SIGN_IMAGE_SHEET_ASSETS_DIR / "21.png",
            SIGN_IMAGE_SHEET_ASSETS_DIR / "1000.png",
        ]
        if not all(ruta.exists() and ruta.stat().st_size > 0 for ruta in recursos_clave):
            _generar_assets_alfabeto_desde_lamina(ruta_alfabeto, SIGN_IMAGE_SHEET_ASSETS_DIR)
            _generar_assets_numeros_desde_lamina(ruta_numeros, SIGN_IMAGE_SHEET_ASSETS_DIR)
    except Exception:
        pass
    finally:
        texto_senas_sheet_assets_ready = True

texto_senas_image_queue = queue.Queue()
texto_senas_image_pending = set()
texto_senas_image_missing = set()
texto_senas_image_worker_started = False
texto_senas_photo_cache = {}


def _buscar_imagen_sena_local(label):
    """Busca una imagen exacta por etiqueta sin cambiar ninguna lógica del modelo."""
    clave = _clave_texto_sena(label)
    if not clave:
        return None

    _asegurar_assets_laminas_texto_senas()

    # Prioridad:
    # 1) imágenes personalizadas del proyecto (HOLA.png, GRACIAS.jpg, etc.)
    # 2) recortes automáticos de las láminas de abecedario/números dadas por el usuario
    # 3) caché local (descargas o copias previas)
    carpetas = (SIGN_IMAGE_PROJECT_DIR, SIGN_IMAGE_SHEET_ASSETS_DIR, SIGN_IMAGE_CACHE_DIR)
    nombres = (clave, clave.lower())

    for carpeta in carpetas:
        for nombre in nombres:
            for ext in SIGN_IMAGE_EXTENSIONS:
                ruta = carpeta / f"{nombre}{ext}"
                if ruta.is_file():
                    return ruta
    return None


def _candidatos_descarga_imagen_sena(label):
    """Devuelve URLs candidatas. Primero imágenes del proyecto; luego alfabeto LSP."""
    clave = _clave_texto_sena(label)
    if not clave:
        return []

    candidatos = []

    # Para letras estáticas vamos primero al dataset LSP: así aparecen rápido
    # aunque la carpeta personalizada del proyecto todavía esté vacía.
    if len(clave) == 1 and clave in SIGN_IMAGE_STATIC_LSP_LETTERS:
        letra = clave.lower()
        archivo = urllib.parse.quote(f"{letra} (1).jpg")
        candidatos.append((
            f"{SIGN_IMAGE_LSP_DATASET_BASE_URL}/{letra}/{archivo}",
            ".jpg",
            "expo99",
        ))
        return candidatos

    # Para palabras/frases o letras dinámicas buscamos una imagen propia:
    # HOLA.png, BUENOS_DIAS.jpg, J.gif convertido a PNG/JPG, etc.
    for ext in SIGN_IMAGE_EXTENSIONS:
        nombre = urllib.parse.quote(f"{clave}{ext}")
        candidatos.append((f"{SIGN_IMAGE_CUSTOM_BASE_URL}/{nombre}", ext, "proyecto"))

    return candidatos


def _iniciar_worker_imagenes_texto_senas():
    global texto_senas_image_worker_started
    if texto_senas_image_worker_started:
        return
    texto_senas_image_worker_started = True

    def worker():
        while True:
            clave = texto_senas_image_queue.get()
            if clave is None:
                texto_senas_image_queue.task_done()
                break

            descargada = False
            try:
                SIGN_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

                for url, ext, fuente in _candidatos_descarga_imagen_sena(clave):
                    try:
                        req = urllib.request.Request(
                            url,
                            headers={
                                "User-Agent": "ManosQueHablan-SignImages/1.0",
                                "Accept": "image/*,*/*;q=0.1",
                                "Cache-Control": "no-cache",
                            },
                        )
                        with urllib.request.urlopen(req, timeout=6) as response:
                            contenido = response.read()

                        # Evita guardar respuestas HTML/errores como si fueran imágenes.
                        if not contenido or len(contenido) < 500:
                            continue

                        destino = SIGN_IMAGE_CACHE_DIR / f"{clave}{ext}"
                        temporal = destino.with_suffix(destino.suffix + ".part")
                        temporal.write_bytes(contenido)
                        temporal.replace(destino)

                        # Confirmamos con OpenCV que realmente puede abrirse.
                        if cv2.imread(str(destino), cv2.IMREAD_COLOR) is None:
                            try:
                                destino.unlink()
                            except OSError:
                                pass
                            continue

                        # Atribución persistente para el dataset usado como respaldo.
                        if fuente == "expo99":
                            try:
                                (SIGN_IMAGE_CACHE_DIR / "FUENTE_IMAGENES.txt").write_text(
                                    "Imágenes estáticas de alfabeto de Lengua de Señas Peruana\n"
                                    "Fuente: Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet\n"
                                    "Licencia: CC BY-SA 4.0\n"
                                    "https://github.com/Expo99/Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet\n",
                                    encoding="utf-8",
                                )
                            except OSError:
                                pass

                        descargada = True
                        break
                    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
                        continue
                    except Exception:
                        continue
            finally:
                texto_senas_image_pending.discard(clave)
                if not descargada:
                    texto_senas_image_missing.add(clave)

                texto_senas_image_queue.task_done()

                if descargada:
                    def refrescar_si_corresponde():
                        if globals().get("sidebar_active") == "Comunicar con señas":
                            try:
                                convertir_texto_a_senas()
                            except tk.TclError:
                                pass
                    try:
                        root.after(0, refrescar_si_corresponde)
                    except Exception:
                        pass

    # Tres workers máximos: las fotos llegan rápido sin crear un hilo por tarjeta.
    for numero in range(3):
        threading.Thread(
            target=worker,
            daemon=True,
            name=f"SignImageDownloader-{numero + 1}",
        ).start()


def _solicitar_imagen_sena(label):
    """Encola una descarga sin bloquear Tkinter ni la cámara."""
    clave = _clave_texto_sena(label)
    if not clave:
        return

    if _buscar_imagen_sena_local(clave) is not None:
        return
    if clave in texto_senas_image_pending or clave in texto_senas_image_missing:
        return

    # Para palabras/frases siempre probamos la carpeta propia de GitHub.
    # Para letras estáticas, además existe el respaldo Expo99.
    texto_senas_image_pending.add(clave)
    texto_senas_image_queue.put(clave)
    _iniciar_worker_imagenes_texto_senas()


def _photo_texto_senas_desde_archivo(ruta, max_width=118, max_height=108):
    """Carga la imagen y mejora su calidad visual sin tocar la lógica."""
    try:
        ruta = Path(ruta)
        firma = (str(ruta), ruta.stat().st_mtime_ns, max_width, max_height, "v4")
    except OSError:
        return None

    photo = texto_senas_photo_cache.get(firma)
    if photo is not None:
        return photo

    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if imagen is None:
        return None

    try:
        imagen = _recortar_contenido_blanco(imagen, padding=6)
    except Exception:
        pass

    if imagen is None or getattr(imagen, "size", 0) == 0:
        return None

    alto, ancho = imagen.shape[:2]
    if ancho <= 0 or alto <= 0:
        return None

    # Estas láminas son pequeñas; por eso primero mejoramos la imagen
    # internamente y luego la llevamos al tamaño de la tarjeta.
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Fondo más limpio y trazo más legible.
    gris = cv2.fastNlMeansDenoising(gris, None, 8, 7, 21)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    gris = cv2.convertScaleAbs(gris, alpha=1.10, beta=4)

    # Escalado de mayor calidad.
    escala_objetivo = min(max_width / ancho, max_height / alto)
    escala_objetivo = max(0.01, float(escala_objetivo))
    escala_mejora = max(1.0, min(3.0, escala_objetivo * 1.55))

    mejora_w = max(1, int(round(ancho * escala_mejora)))
    mejora_h = max(1, int(round(alto * escala_mejora)))
    gris = cv2.resize(gris, (mejora_w, mejora_h), interpolation=cv2.INTER_LANCZOS4)

    # Enfoque suave tipo unsharp mask para que la mano se vea más definida.
    blur = cv2.GaussianBlur(gris, (0, 0), 0.9)
    gris = cv2.addWeighted(gris, 1.32, blur, -0.32, 0)

    # Blanqueamos el fondo sin destruir el dibujo.
    _, fondo = cv2.threshold(gris, 244, 255, cv2.THRESH_BINARY)
    gris[fondo == 255] = 255

    # Si aún no coincide exactamente con el tamaño de la tarjeta, ajustamos al final.
    final_w = max(1, int(round(mejora_w)))
    final_h = max(1, int(round(mejora_h)))
    if final_w > max_width or final_h > max_height:
        escala_final = min(max_width / final_w, max_height / final_h)
        final_w = max(1, int(round(final_w * escala_final)))
        final_h = max(1, int(round(final_h * escala_final)))
        gris = cv2.resize(gris, (final_w, final_h), interpolation=cv2.INTER_AREA)

    imagen_final = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    ok, buffer = cv2.imencode('.ppm', imagen_final)
    if not ok:
        return None

    try:
        photo = tk.PhotoImage(data=buffer.tobytes())
    except tk.TclError:
        return None

    texto_senas_photo_cache[firma] = photo
    return photo


def _dibujar_muestra_texto_senas(canvas, sample=None, label=None, visible=None):
    """
    Muestra una IMAGEN REAL en lugar de los landmarks/puntos 3D.

    `sample` se conserva en la firma únicamente para no romper llamadas anteriores;
    no se modifica ni se usa para cambiar la lógica de reconocimiento.
    """
    c = THEMES.get(current_theme_name, THEMES["Oscuro"])
    canvas.delete("all")
    canvas.configure(bg=c["card"], highlightbackground=c["border"])

    try:
        canvas_w = max(1, int(canvas.cget("width")))
        canvas_h = max(1, int(canvas.cget("height")))
    except Exception:
        canvas_w, canvas_h = 128, 118

    clave = _clave_texto_sena(label or visible or "")
    ruta = _buscar_imagen_sena_local(clave)

    if ruta is not None:
        photo = _photo_texto_senas_desde_archivo(
            ruta,
            max_width=max(52, canvas_w - 12),
            max_height=max(52, canvas_h - 12),
        )
        if photo is not None:
            canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center")
            canvas.image = photo
            return

    _solicitar_imagen_sena(clave)

    if clave in SIGN_IMAGE_DYNAMIC_LSP_LETTERS:
        titulo = "SEÑA DINÁMICA"
        detalle = "Requiere movimiento"
    elif clave in texto_senas_image_pending:
        titulo = "DESCARGANDO"
        detalle = "Buscando imagen..."
    else:
        titulo = "SIN IMAGEN"
        detalle = "No hay imagen exacta"

    canvas.create_text(
        canvas_w // 2, max(36, canvas_h // 2 - 14),
        text=titulo,
        fill=c["muted"],
        font=(FONT, 11, "bold"),
        justify="center",
    )
    canvas.create_text(
        canvas_w // 2, min(canvas_h - 24, canvas_h // 2 + 18),
        text=detalle,
        fill=c["muted"],
        font=(FONT, 8),
        justify="center",
    )


def _expandir_plan_visual_texto_senas(plan):
    """
    Conserva `crear_plan_texto_a_senas()` intacto.

    Solo decide la REPRESENTACIÓN:
    - si hay imagen exacta de una palabra/frase -> una tarjeta;
    - si todavía no existe -> se solicita esa imagen y, mientras tanto,
      se muestra la palabra con imágenes del alfabeto;
    - si el usuario escribió un número soportado (0-21, 100, 1000, 1000000),
      se muestra una sola imagen del número en vez de separar cada dígito.
    """
    visual = []
    i = 0

    while i < len(plan):
        item = plan[i]

        # Soporte visual para números escritos por el usuario sin tocar el plan lógico.
        visible_inicial = str(item.get("visible") or item.get("label") or "").strip()
        if item.get("inicio_palabra") and visible_inicial.isdigit():
            j = i
            partes = []
            while j < len(plan):
                actual = plan[j]
                visible_actual = str(actual.get("visible") or actual.get("label") or "").strip()
                if j > i and actual.get("inicio_palabra"):
                    break
                if len(visible_actual) != 1 or not visible_actual.isdigit():
                    break
                partes.append(visible_actual)
                j += 1

            numero = "".join(partes)
            if numero in SIGN_IMAGE_SUPPORTED_NUMBER_TOKENS and _buscar_imagen_sena_local(numero) is not None:
                visual.append({
                    "label": numero,
                    "image_label": numero,
                    "visible": numero,
                    "sample": None,
                    "tipo": "numero",
                    "inicio_palabra": item.get("inicio_palabra", False),
                })
                i = j
                continue

        item_visual = dict(item)
        clave = _clave_texto_sena(item.get("label") or item.get("visible"))

        if item.get("tipo") == "seña":
            exacta = _buscar_imagen_sena_local(clave)
            if exacta is not None:
                item_visual["image_label"] = clave
                visual.append(item_visual)
                i += 1
                continue

            _solicitar_imagen_sena(clave)

            primera = True
            agregadas = 0
            texto_visible = str(item.get("visible") or clave).strip()
            for ch in texto_visible:
                letra = _letra_sin_tilde(ch)
                if not letra.isalnum():
                    primera = True
                    continue

                visual.append({
                    "label": letra,
                    "image_label": letra,
                    "visible": letra,
                    "sample": None,
                    "tipo": "letra",
                    "inicio_palabra": primera or item.get("inicio_palabra", False),
                    "origen_palabra": clave,
                })
                primera = False
                agregadas += 1

            if agregadas == 0:
                item_visual["image_label"] = clave
                visual.append(item_visual)
        else:
            item_visual["image_label"] = clave
            visual.append(item_visual)

        i += 1

    return visual


def _limpiar_resultados_texto_senas():
    if "texto_senas_result_inner" not in globals():
        return
    for hijo in texto_senas_result_inner.winfo_children():
        hijo.destroy()


def convertir_texto_a_senas(event=None):
    texto = texto_senas_var.get().strip()
    _limpiar_resultados_texto_senas()

    c = THEMES.get(current_theme_name, THEMES["Oscuro"])

    if not texto:
        texto_senas_info_var.set(
            "Escribe una palabra o frase para traducir a imágenes."
        )
        return

    # La biblioteca y el plan siguen siendo EXACTAMENTE los mismos de antes.
    # Solo cambia la representación visual de cada resultado.
    biblioteca = obtener_biblioteca_texto_a_senas()

    if not biblioteca:
        texto_senas_info_var.set(
            "No hay modelos disponibles. Entrena o carga señas primero."
        )
        aviso = tk.Label(
            texto_senas_result_inner,
            text="Carga o entrena modelos de señas para poder mostrarlas.",
            bg=c["panel_alt"],
            fg=c["muted"],
            font=(FONT, 10),
            padx=24,
            pady=30,
        )
        aviso.pack(side="left", padx=12, pady=16)
        return

    plan = crear_plan_texto_a_senas(texto, biblioteca)
    if not plan:
        texto_senas_info_var.set("No encontré texto válido para convertir.")
        return

    # El conteo lógico se calcula SOBRE EL PLAN ORIGINAL, no sobre el fallback visual.
    faltantes = sum(1 for item in plan if item["sample"] is None)
    completos = sum(1 for item in plan if item["tipo"] == "seña")
    letras = sum(1 for item in plan if item["tipo"] == "letra")

    visual_plan = _expandir_plan_visual_texto_senas(plan)

    texto_senas_info_var.set(
        f"Mostrando {len(visual_plan)} imagen(es) para traducir."
    )

    # Evitamos crear cientos de widgets si alguien pega un texto enorme.
    max_items = 40
    for index, item in enumerate(visual_plan[:max_items]):
        margen_izq = 12 if item.get("inicio_palabra") and index > 0 else 4

        card = tk.Frame(
            texto_senas_result_inner,
            bg=c["card"],
            width=150,
            height=178,
            highlightthickness=1,
            highlightbackground=c["border"],
        )
        card.pack(side="left", padx=(margen_izq, 4), pady=7)
        card.pack_propagate(False)

        tipo_texto = "IMAGEN"

        tipo_label = tk.Label(
            card,
            text=tipo_texto,
            bg=c["card"],
            fg=c["muted"],
            font=(FONT, 6, "bold"),
        )
        tipo_label.pack(fill="x", padx=8, pady=(8, 1))

        dibujo = tk.Canvas(
            card,
            width=128,
            height=118,
            bd=0,
            highlightthickness=0,
            bg=c["card"],
        )
        dibujo.pack(padx=4, pady=(0, 0))
        _dibujar_muestra_texto_senas(
            dibujo,
            item.get("sample"),
            label=item.get("image_label") or item.get("label"),
            visible=item.get("visible"),
        )

        etiqueta = tk.Label(
            card,
            text=str(item.get("visible") or "").upper(),
            bg=c["card"],
            fg=c["text"],
            font=(FONT, 7, "bold"),
            anchor="center",
        )
        etiqueta.pack(fill="x", padx=4, pady=(0, 5))

    if len(visual_plan) > max_items:
        extra = tk.Label(
            texto_senas_result_inner,
            text=f"+ {len(visual_plan) - max_items} imágenes más",
            bg=c["panel_alt"],
            fg=c["muted"],
            font=(FONT, 9, "bold"),
            padx=14,
            pady=20,
        )
        extra.pack(side="left", padx=8, pady=10)

    texto_senas_result_inner.update_idletasks()
    texto_senas_result_canvas.configure(
        scrollregion=texto_senas_result_canvas.bbox("all")
    )
    texto_senas_result_canvas.xview_moveto(0.0)

def programar_conversion_texto_senas(event=None):
    global texto_senas_after_id

    if texto_senas_after_id is not None:
        try:
            root.after_cancel(texto_senas_after_id)
        except tk.TclError:
            pass

    texto_senas_after_id = root.after(280, convertir_texto_a_senas)


def limpiar_texto_a_senas():
    texto_senas_var.set("")
    _limpiar_resultados_texto_senas()
    texto_senas_info_var.set(
        "Escribe una palabra o frase para traducir a imágenes."
    )
    texto_senas_entry.focus_set()


def abrir_vista_texto_a_senas():
    """Cubre cámara + panel derecho sin modificar su grid."""
    camera_panel.grid(row=0, column=1, sticky="nsew", padx=6)
    side_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

    main.grid_columnconfigure(
        0, weight=0, minsize=SIDEBAR_FIXED_WIDTH, uniform=""
    )
    main.grid_columnconfigure(1, weight=6, uniform="main_content")
    main.grid_columnconfigure(
        2, weight=3, minsize=0, uniform="main_content"
    )

    root.update_idletasks()
    x = camera_panel.winfo_x()
    y = min(camera_panel.winfo_y(), side_panel.winfo_y())
    right = side_panel.winfo_x() + side_panel.winfo_width()
    bottom = max(
        camera_panel.winfo_y() + camera_panel.winfo_height(),
        side_panel.winfo_y() + side_panel.winfo_height(),
    )

    text_to_sign_panel.place(
        x=x,
        y=y,
        width=max(1, right - x),
        height=max(1, bottom - y),
    )
    text_to_sign_panel.lift()
    texto_senas_entry.focus_set()
    convertir_texto_a_senas()


texto_senas_header = register_theme(
    tk.Frame(text_to_sign_panel),
    "panel",
)
texto_senas_header.pack(fill="x", padx=22, pady=(20, 10))

texto_senas_title = register_theme(
    tk.Label(
        texto_senas_header,
        text="Texto a señas",
        font=(FONT, 18, "bold"),
        anchor="w",
    ),
    "text_panel",
)
texto_senas_title.pack(fill="x")

texto_senas_input_card = register_theme(
    tk.Frame(text_to_sign_panel, highlightthickness=1),
    "card",
)
texto_senas_input_card.pack(fill="x", padx=22, pady=(0, 12))

texto_senas_input_label = register_theme(
    tk.Label(
        texto_senas_input_card,
        text="ESCRIBE UNA PALABRA O FRASE",
        font=(FONT, 8, "bold"),
        anchor="w",
    ),
    "muted_card",
)
texto_senas_input_label.pack(fill="x", padx=14, pady=(12, 6))

texto_senas_input_row = register_theme(
    tk.Frame(texto_senas_input_card),
    "card",
)
texto_senas_input_row.pack(fill="x", padx=14, pady=(0, 12))
texto_senas_input_row.grid_columnconfigure(0, weight=1)

texto_senas_var = tk.StringVar()

texto_senas_entry = tk.Entry(
    texto_senas_input_row,
    textvariable=texto_senas_var,
    relief="flat",
    bd=0,
    highlightthickness=1,
    font=(FONT, 12),
)
texto_senas_entry.grid(
    row=0, column=0, sticky="ew", padx=(0, 8), ipady=9
)

texto_senas_convert_button = register_theme(
    tk.Button(
        texto_senas_input_row,
        text="Convertir",
        command=convertir_texto_a_senas,
        relief="flat",
        bd=0,
        padx=16,
        pady=9,
        cursor="hand2",
        font=(FONT, 9, "bold"),
    ),
    "primary_button",
)
texto_senas_convert_button.grid(row=0, column=1, padx=(0, 5))
add_button_hover(texto_senas_convert_button, "primary_button")

texto_senas_clear_button = register_theme(
    tk.Button(
        texto_senas_input_row,
        text="Limpiar",
        command=limpiar_texto_a_senas,
        relief="flat",
        bd=0,
        padx=14,
        pady=9,
        cursor="hand2",
        font=(FONT, 9, "bold"),
    ),
    "button",
)
texto_senas_clear_button.grid(row=0, column=2, padx=(5, 0))
add_button_hover(texto_senas_clear_button, "button")

texto_senas_entry.bind("<KeyRelease>", programar_conversion_texto_senas)
texto_senas_entry.bind("<Return>", convertir_texto_a_senas)

texto_senas_info_var = tk.StringVar(
    value="Escribe una palabra o frase. La conversión aparecerá aquí automáticamente."
)
texto_senas_info = register_theme(
    tk.Label(
        text_to_sign_panel,
        textvariable=texto_senas_info_var,
        font=(FONT, 8),
        anchor="w",
    ),
    "muted_panel",
)
texto_senas_info.pack(fill="x", padx=24, pady=(0, 7))

texto_senas_result_shell = register_theme(
    tk.Frame(text_to_sign_panel, highlightthickness=1),
    "panel_alt",
)
texto_senas_result_shell.pack(
    fill="both", expand=True, padx=22, pady=(0, 20)
)

texto_senas_result_canvas = tk.Canvas(
    texto_senas_result_shell,
    bd=0,
    highlightthickness=0,
)
texto_senas_result_canvas.pack(
    fill="both", expand=True, padx=10, pady=(10, 0)
)

texto_senas_scroll_x = tk.Scrollbar(
    texto_senas_result_shell,
    orient="horizontal",
    command=texto_senas_result_canvas.xview,
)
texto_senas_scroll_x.pack(fill="x", padx=10, pady=(0, 10))
texto_senas_result_canvas.configure(
    xscrollcommand=texto_senas_scroll_x.set
)

texto_senas_result_inner = tk.Frame(texto_senas_result_canvas)
texto_senas_result_window = texto_senas_result_canvas.create_window(
    (0, 0),
    window=texto_senas_result_inner,
    anchor="nw",
)

def _ajustar_scroll_texto_senas(event=None):
    try:
        texto_senas_result_canvas.configure(
            scrollregion=texto_senas_result_canvas.bbox("all")
        )
    except tk.TclError:
        pass

texto_senas_result_inner.bind("<Configure>", _ajustar_scroll_texto_senas)


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

# Las antiguas acciones rápidas Copiar / Escuchar / Limpiar se mantienen
# internamente por compatibilidad, pero ya no se muestran en la interfaz.
translate_actions = register_theme(tk.Frame(features_panel), "panel")
# translate_actions.pack(...) eliminado a petición del usuario.
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

# ---------------- FORMAR ORACIONES ----------------
# Acumula únicamente señas estables. De esta manera una misma seña mantenida
# frente a la cámara no se repite en cada frame.
sentence_words = []
# Id de la oración actual dentro del historial. Se crea al llegar a 2 palabras
# y luego se actualiza, evitando guardar HOLA / HOLA COMO / HOLA COMO ESTAS
# como tres oraciones diferentes.
sentence_history_id = None
sentence_candidate = None
sentence_candidate_since = 0.0
sentence_candidate_confirmations = 0
sentence_candidate_confidence_sum = 0.0
sentence_last_committed_sign = None
sentence_no_hand_since = None

# Cuando el modelo reconoce letras sueltas (A, B, C...), las acumulamos aquí
# para formar una palabra real. Ejemplo: H + O + L + A -> HOLA.
sentence_letter_buffer = ""
sentence_letter_last_commit_time = 0.0

# Una pausa corta permite cambiar de letra; una pausa algo mayor cierra la palabra.
# Así también se pueden escribir letras repetidas, por ejemplo L + L.
SPELLING_WORD_GAP_SECONDS = 0.90

# Lectura automática del reconocimiento.
# - Las señas que representan palabras se leen al confirmarse.
# - Las letras se juntan para formar la palabra. Opcionalmente, con
#   “Deletrear mientras detecta”, también se pronuncian una por una.
AUTO_READ_RECOGNIZED_SIGNS = True
AUTO_READ_SPELLED_WORDS = True

# Opción visual: si está activa, cada letra reconocida se pronuncia
# inmediatamente mientras se sigue formando la palabra completa.
# La palabra final se sigue leyendo al terminar, como antes.
spell_live_var = tk.BooleanVar(value=False)

sentence_speech_lock = threading.Lock()
sentence_tts_missing_notified = False
sentence_last_spoken_text = None
sentence_last_spoken_time = 0.0
AUTO_SPEECH_REPEAT_GUARD_SECONDS = 0.75

# Filtro conservador para formar frases:
# - exige una seña estable durante un poco más de tiempo;
# - exige varias confirmaciones consecutivas;
# - exige una confianza mínima antes de aceptar la palabra.
SENTENCE_STABLE_SECONDS = 0.60
SENTENCE_RELEASE_SECONDS = 0.45
SENTENCE_MIN_CONFIDENCE = 72.0
SENTENCE_MIN_CONFIRMATIONS = 5

sentence_card = register_theme(
    tk.Frame(features_panel, highlightthickness=1),
    "card",
)
sentence_card.pack(fill="x", pady=(0, 10))

sentence_title = register_theme(
    tk.Label(
        sentence_card,
        text="FORMAR ORACIONES",
        font=("DejaVu Sans", 8, "bold"),
        anchor="w",
    ),
    "muted_card",
)
sentence_title.pack(fill="x", padx=14, pady=(12, 4))

sentence_value = register_theme(
    tk.Label(
        sentence_card,
        text="Las señas detectadas se unirán aquí...",
        justify="left",
        anchor="w",
        wraplength=330,
        font=("DejaVu Sans", 13, "bold"),
    ),
    "text_card",
)
sentence_value.pack(fill="x", padx=14, pady=(3, 8))

# Opción de deletreo en vivo. Desactivada por defecto para conservar
# el comportamiento anterior: acumular letras y leer la palabra al finalizar.
spell_live_row = tk.Frame(
    sentence_card,
    bd=1,
    relief="solid",
    highlightthickness=0,
)
spell_live_row.pack(fill="x", padx=14, pady=(0, 9))

spell_live_check = tk.Checkbutton(
    spell_live_row,
    text="🔤  Deletrear mientras detecta",
    variable=spell_live_var,
    indicatoron=True,
    anchor="w",
    cursor="hand2",
    bd=0,
    highlightthickness=0,
    padx=8,
    pady=6,
    font=("DejaVu Sans", 9, "bold"),
)
spell_live_check.pack(fill="x")

def _es_sena_letra(signo):
    """Devuelve True solo para etiquetas de una letra del alfabeto."""
    signo = str(signo or "").strip()
    return len(signo) == 1 and signo.isalpha()


def _texto_oracion_actual():
    """Texto visible, incluyendo una palabra que todavía se está deletreando."""
    partes = list(sentence_words)
    if sentence_letter_buffer:
        partes.append(sentence_letter_buffer)
    return " ".join(partes).strip()


def _refrescar_texto_oracion():
    """Actualiza el texto visible del constructor de oraciones."""
    texto = _texto_oracion_actual()
    if texto:
        sentence_value.configure(text=texto)
    else:
        sentence_value.configure(text="Las señas detectadas se unirán aquí...")


def _finalizar_palabra_deletreada(hablar=True):
    """Convierte las letras acumuladas en una palabra y puede leerla al finalizar."""
    global sentence_letter_buffer
    global sentence_letter_last_commit_time

    palabra = str(sentence_letter_buffer or "").strip().upper()
    if not palabra:
        return None

    sentence_letter_buffer = ""
    sentence_letter_last_commit_time = 0.0
    sentence_words.append(palabra)

    _refrescar_texto_oracion()
    agregar_historial(palabra, "Palabra")
    _sincronizar_oracion_con_historial()

    if hablar and AUTO_READ_SPELLED_WORDS:
        _hablar_sena_en_segundo_plano(palabra)

    return palabra


def _sincronizar_oracion_con_historial():
    global sentence_history_id
    texto = " ".join(sentence_words).strip()
    if len(sentence_words) >= 2:
        if sentence_history_id is None:
            sentence_history_id = agregar_historial(texto, "Oración")
        else:
            actualizar_historial(sentence_history_id, texto, "Oración")
    elif sentence_history_id is not None:
        eliminar_historial(sentence_history_id)
        sentence_history_id = None
    if "refrescar_historial_ui" in globals():
        try:
            refrescar_historial_ui()
        except Exception:
            pass


def copiar_oracion():
    """Copia al portapapeles la oración completa formada con señas."""
    texto = _texto_oracion_actual()
    if not texto:
        set_status("Todavía no hay una oración para copiar.")
        return
    root.clipboard_clear()
    root.clipboard_append(texto)
    root.update()
    set_status("Oración copiada al portapapeles.")


def borrar_ultima_sena_oracion():
    """Borra la última letra en escritura o, si no hay letras pendientes, la última palabra."""
    global sentence_candidate
    global sentence_candidate_since
    global sentence_candidate_confirmations
    global sentence_candidate_confidence_sum
    global sentence_last_committed_sign
    global sentence_no_hand_since
    global sentence_letter_buffer
    global sentence_letter_last_commit_time

    if sentence_letter_buffer:
        borrada = sentence_letter_buffer[-1]
        sentence_letter_buffer = sentence_letter_buffer[:-1]
        if not sentence_letter_buffer:
            sentence_letter_last_commit_time = 0.0
        sentence_candidate = None
        sentence_candidate_since = 0.0
        sentence_candidate_confirmations = 0
        sentence_candidate_confidence_sum = 0.0
        sentence_last_committed_sign = None
        sentence_no_hand_since = None
        _refrescar_texto_oracion()
        set_status(f"Se borró la última letra: {borrada}.")
        return

    if not sentence_words:
        set_status("No hay señas para borrar.")
        return

    borrada = sentence_words.pop()
    sentence_candidate = None
    sentence_candidate_since = 0.0
    sentence_candidate_confirmations = 0
    sentence_candidate_confidence_sum = 0.0
    sentence_last_committed_sign = None
    sentence_no_hand_since = None
    _refrescar_texto_oracion()
    _sincronizar_oracion_con_historial()
    set_status(f"Se borró la última seña: {borrada}.")


def limpiar_oracion():
    """Vacía por completo la oración y reinicia el filtro anti-repetición."""
    global sentence_candidate
    global sentence_candidate_since
    global sentence_candidate_confirmations
    global sentence_candidate_confidence_sum
    global sentence_last_committed_sign
    global sentence_no_hand_since
    global sentence_history_id
    global sentence_letter_buffer
    global sentence_letter_last_commit_time

    # La oración completa ya se mantiene actualizada en Historial; al limpiar
    # solo cerramos esta sesión para que la siguiente frase tenga una entrada nueva.
    _sincronizar_oracion_con_historial()
    sentence_words.clear()
    sentence_letter_buffer = ""
    sentence_letter_last_commit_time = 0.0
    sentence_history_id = None
    sentence_candidate = None
    sentence_candidate_since = 0.0
    sentence_candidate_confirmations = 0
    sentence_candidate_confidence_sum = 0.0
    sentence_last_committed_sign = None
    sentence_no_hand_since = None
    _refrescar_texto_oracion()
    set_status("Oración limpiada.")


def leer_oracion_en_voz_alta():
    """Lee en voz alta la oración completa formada hasta el momento."""
    texto = _texto_oracion_actual()
    if not texto:
        set_status("Todavía no hay una oración para leer.")
        return

    hablar = globals().get("_hablar_sena_en_segundo_plano")
    if hablar is None:
        set_status("El motor de voz todavía no está disponible.", error=True)
        return

    hablar(texto)
    set_status("Leyendo la oración en voz alta...")


# Acciones del constructor de oraciones: mismo ancho y alineación.
sentence_actions = register_theme(tk.Frame(sentence_card), "card")
sentence_actions.pack(fill="x", padx=14, pady=(0, 8))
for col in range(4):
    sentence_actions.grid_columnconfigure(col, weight=1, uniform="sentence_action")

sentence_copy_button = make_translate_action(
    sentence_actions, "⧉  Copiar", copiar_oracion, 0
)
sentence_delete_button = make_translate_action(
    sentence_actions, "⌫  Borrar", borrar_ultima_sena_oracion, 1
)
sentence_clear_button = make_translate_action(
    sentence_actions, "×  Limpiar", limpiar_oracion, 2
)
sentence_read_button = make_translate_action(
    sentence_actions, "🔊  Leer", leer_oracion_en_voz_alta, 3
)



def _hablar_sena_en_segundo_plano(texto):
    """Pronuncia una seña sin bloquear cámara ni Tkinter.

    Prioriza voces más naturales cuando están disponibles:
    Windows usa una voz española instalada; Linux puede aprovechar Piper
    automáticamente y conserva motores del sistema como respaldo.
    """
    global sentence_tts_missing_notified
    global sentence_last_spoken_text
    global sentence_last_spoken_time

    if not texto:
        return

    texto = str(texto).strip()
    if not texto:
        return

    ahora_voz = time.monotonic()
    texto_guardia = texto.upper()
    if (
        sentence_last_spoken_text == texto_guardia
        and (ahora_voz - sentence_last_spoken_time) < AUTO_SPEECH_REPEAT_GUARD_SECONDS
    ):
        return
    sentence_last_spoken_text = texto_guardia
    sentence_last_spoken_time = ahora_voz

    # Une letras sueltas cuando corresponden a una palabra deletreada.
    texto_limpio = " ".join(
        texto.replace("_", " ").replace("-", " ").split()
    ).lower()

    tokens = texto_limpio.split()
    tokens_voz = []
    letras_pendientes = []

    def vaciar_letras():
        nonlocal letras_pendientes
        if not letras_pendientes:
            return
        if len(letras_pendientes) >= 2:
            tokens_voz.append("".join(letras_pendientes))
        else:
            tokens_voz.extend(letras_pendientes)
        letras_pendientes = []

    for token in tokens:
        if len(token) == 1 and token.isalpha():
            letras_pendientes.append(token)
        else:
            vaciar_letras()
            tokens_voz.append(token)

    vaciar_letras()
    texto_voz = " ".join(tokens_voz).strip()
    if not texto_voz:
        return

    def _buscar_modelo_piper_es():
        """Busca una voz Piper española en ubicaciones pequeñas y conocidas."""
        candidatos = []
        modelo_env = os.environ.get("MQH_PIPER_MODEL", "").strip()
        if modelo_env:
            candidatos.append(Path(modelo_env).expanduser())

        base_app = Path(__file__).resolve().parent
        carpetas = [
            base_app / "voces",
            base_app / "voices",
            Path.home() / ".local" / "share" / "piper",
            Path.home() / ".local" / "share" / "piper-voices",
            Path.home() / ".config" / "piper",
        ]
        for carpeta in carpetas:
            try:
                if carpeta.exists():
                    candidatos.extend(carpeta.rglob("*.onnx"))
            except Exception:
                pass

        for ruta in candidatos:
            try:
                nombre = ruta.name.lower()
                if ruta.is_file() and (
                    nombre.startswith("es_")
                    or nombre.startswith("es-")
                    or "spanish" in nombre
                ):
                    return ruta
            except Exception:
                pass
        return None

    def worker():
        global sentence_tts_missing_notified
        import shutil

        hablado = False
        with sentence_speech_lock:
            try:
                # Windows: elige una voz española instalada y ajusta el ritmo.
                if os.name == "nt":
                    powershell = shutil.which("powershell") or shutil.which("pwsh")
                    if powershell:
                        try:
                            env = os.environ.copy()
                            env["MQH_TTS_TEXT"] = texto_voz
                            script = (
                                "Add-Type -AssemblyName System.Speech; "
                                "$v=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                                "$voices=@($v.GetInstalledVoices() | ForEach-Object {$_.VoiceInfo} | "
                                "Where-Object {$_.Culture.Name -like 'es-*'}); "
                                "if($voices.Count -gt 0){$v.SelectVoice($voices[0].Name)}; "
                                "$v.Rate=-1; $v.Volume=100; "
                                "$v.Speak($env:MQH_TTS_TEXT)"
                            )
                            resultado = subprocess.run(
                                [powershell, "-NoProfile", "-Command", script],
                                env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                                timeout=20,
                            )
                            hablado = resultado.returncode == 0
                        except Exception:
                            pass

                # Linux: si Piper y una voz española están instalados, se usan
                # antes que eSpeak porque la voz es mucho más natural.
                if not hablado and os.name != "nt" and shutil.which("piper"):
                    modelo_piper = _buscar_modelo_piper_es()
                    if modelo_piper is not None:
                        try:
                            import tempfile
                            with tempfile.TemporaryDirectory(prefix="mqh_tts_") as td:
                                wav = Path(td) / "voz.wav"
                                resultado = subprocess.run(
                                    ["piper", "--model", str(modelo_piper), "--output_file", str(wav)],
                                    input=texto_voz.encode("utf-8"),
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    check=False,
                                    timeout=20,
                                )
                                if resultado.returncode == 0 and wav.exists():
                                    if shutil.which("paplay"):
                                        comando = ["paplay", str(wav)]
                                    elif shutil.which("aplay"):
                                        comando = ["aplay", "-q", str(wav)]
                                    elif shutil.which("ffplay"):
                                        comando = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav)]
                                    else:
                                        comando = None

                                    if comando:
                                        play = subprocess.run(
                                            comando,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                            check=False,
                                            timeout=20,
                                        )
                                        hablado = play.returncode == 0
                        except Exception:
                            pass

                if not hablado and shutil.which("say"):
                    try:
                        resultado = subprocess.run(
                            ["say", "-r", "175", texto_voz],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=15,
                        )
                        hablado = resultado.returncode == 0
                    except Exception:
                        pass

                # Linux/Unix: Speech Dispatcher primero; eSpeak como respaldo.
                if not hablado and shutil.which("spd-say"):
                    try:
                        resultado = subprocess.run(
                            ["spd-say", "--wait", "-l", "es", texto_voz],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            timeout=15,
                        )
                        hablado = resultado.returncode == 0
                    except Exception:
                        pass

                if not hablado:
                    for ejecutable in ("espeak-ng", "espeak"):
                        if not shutil.which(ejecutable):
                            continue
                        try:
                            resultado = subprocess.run(
                                [ejecutable, "-v", "es", "-s", "145", "-p", "48", "-g", "3", texto_voz],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                                timeout=15,
                            )
                            if resultado.returncode == 0:
                                hablado = True
                                break
                        except Exception:
                            continue

                # Último respaldo opcional: pyttsx3.
                if not hablado:
                    try:
                        import pyttsx3
                        motor = pyttsx3.init()
                        try:
                            for voz in motor.getProperty("voices") or []:
                                datos = " ".join([
                                    str(getattr(voz, "id", "")),
                                    str(getattr(voz, "name", "")),
                                    str(getattr(voz, "languages", "")),
                                ]).lower()
                                if "spanish" in datos or "es_" in datos or "es-" in datos:
                                    motor.setProperty("voice", voz.id)
                                    break
                        except Exception:
                            pass
                        motor.setProperty("rate", 165)
                        motor.setProperty("volume", 1.0)
                        motor.say(texto_voz)
                        motor.runAndWait()
                        motor.stop()
                        hablado = True
                    except Exception:
                        pass
            except Exception:
                hablado = False

        if not hablado and not sentence_tts_missing_notified:
            sentence_tts_missing_notified = True
            try:
                root.after(
                    0,
                    lambda: set_status(
                        "No se encontró un motor de voz disponible."
                        if os.name == "nt"
                        else "No se encontró un motor de voz. En Arch instala: sudo pacman -S espeak-ng",
                        error=True,
                    ),
                )
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True, name="SignSpeech").start()


def actualizar_oracion_detectada(signo=None, sin_manos=False, confidence=0.0):
    """Añade una seña solo cuando es estable, repetida y suficientemente confiable."""
    global sentence_candidate
    global sentence_candidate_since
    global sentence_candidate_confirmations
    global sentence_candidate_confidence_sum
    global sentence_last_committed_sign
    global sentence_no_hand_since
    global sentence_letter_buffer
    global sentence_letter_last_commit_time

    now = time.monotonic()

    if sin_manos:
        sentence_candidate = None
        sentence_candidate_since = 0.0
        sentence_candidate_confirmations = 0
        sentence_candidate_confidence_sum = 0.0
        if sentence_no_hand_since is None:
            sentence_no_hand_since = now
        else:
            pausa = now - sentence_no_hand_since
            if pausa >= SENTENCE_RELEASE_SECONDS:
                sentence_last_committed_sign = None
            # Una pausa mayor cierra la palabra formada por letras.
            if sentence_letter_buffer and pausa >= SPELLING_WORD_GAP_SECONDS:
                _finalizar_palabra_deletreada(hablar=True)
        return

    sentence_no_hand_since = None

    if not signo:
        sentence_candidate = None
        sentence_candidate_since = 0.0
        sentence_candidate_confirmations = 0
        sentence_candidate_confidence_sum = 0.0
        return

    signo = str(signo).strip()
    if not signo:
        return

    try:
        confidence = float(confidence or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    etiquetas_activas = {
        str(item.get("label", "")).strip()
        for item in recognition_model_samples
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    }
    if signo not in etiquetas_activas:
        sentence_candidate = None
        sentence_candidate_since = 0.0
        sentence_candidate_confirmations = 0
        sentence_candidate_confidence_sum = 0.0
        return

    if confidence < SENTENCE_MIN_CONFIDENCE:
        sentence_candidate = None
        sentence_candidate_since = 0.0
        sentence_candidate_confirmations = 0
        sentence_candidate_confidence_sum = 0.0
        return

    if signo != sentence_candidate:
        sentence_candidate = signo
        sentence_candidate_since = now
        sentence_candidate_confirmations = 1
        sentence_candidate_confidence_sum = confidence
        return

    sentence_candidate_confirmations += 1
    sentence_candidate_confidence_sum += confidence

    if now - sentence_candidate_since < SENTENCE_STABLE_SECONDS:
        return

    if sentence_candidate_confirmations < SENTENCE_MIN_CONFIRMATIONS:
        return

    promedio_confianza = (
        sentence_candidate_confidence_sum / max(1, sentence_candidate_confirmations)
    )
    if promedio_confianza < SENTENCE_MIN_CONFIDENCE:
        return

    if signo == sentence_last_committed_sign:
        return

    sentence_last_committed_sign = signo

    sentence_candidate = None
    sentence_candidate_since = 0.0
    sentence_candidate_confirmations = 0
    sentence_candidate_confidence_sum = 0.0

    if _es_sena_letra(signo):
        # Las letras se unen sin espacios.
        # Ejemplo: B + A + B + A -> BABA.
        sentence_letter_buffer += signo.upper()
        sentence_letter_last_commit_time = now
        _refrescar_texto_oracion()

        # Si el usuario activa “Deletrear mientras detecta”, pronunciamos
        # cada letra justo cuando supera el filtro de estabilidad/confianza.
        # Esto NO habla por frame y no cambia el reconocimiento de MediaPipe.
        if globals().get("spell_live_var") is not None and spell_live_var.get():
            _hablar_sena_en_segundo_plano(signo.upper())
            set_status(f"Deletreando: {sentence_letter_buffer}")
        else:
            set_status(f"Formando palabra: {sentence_letter_buffer}")
        return

    # Si llega una seña que ya es una palabra completa, primero cerramos
    # cualquier palabra que el usuario estuviera deletreando.
    if sentence_letter_buffer:
        _finalizar_palabra_deletreada(hablar=True)

    sentence_words.append(signo)
    _refrescar_texto_oracion()

    agregar_historial(signo, "Palabra")
    _sincronizar_oracion_con_historial()

    if AUTO_READ_RECOGNIZED_SIGNS:
        _hablar_sena_en_segundo_plano(signo)

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

    if "spell_live_row" in globals():
        spell_live_row.configure(
            bg=c["panel_alt"],
            highlightbackground=c["border"],
        )
        spell_live_check.configure(
            bg=c["panel_alt"],
            fg=c["text"],
            activebackground=c["panel_alt"],
            activeforeground=c["text"],
            selectcolor=c["button"],
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

# Carga primero TODOS los JSON entrenados localmente (uno por seña).
# Después GitHub se combina como fuente externa, sin reemplazar los locales.
cargar_modelos_locales_entrenados(mostrar_estado=False)

# Carga/actualiza el modelo desde GitHub en segundo plano.
# No toca process_frames(), por lo que no añade delay a MediaPipe.
sincronizar_modelo_github()

# La campanita comprueba una vez al iniciar si existe una versión nueva.
# Se hace después de arrancar la interfaz y en un hilo aparte, sin bloquear cámara.
root.after(1200, buscar_actualizaciones_app)

root.mainloop()
