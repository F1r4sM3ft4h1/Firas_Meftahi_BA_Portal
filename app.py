from flask import Flask, render_template, request, redirect, url_for, abort, jsonify, session
import sqlite3
import qrcode
import os
import base64
import json
import threading
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

DEMO_TOOLS = [
    ("Akkuschrauber", "Elektrowerkzeug", "Werkzeugwagen", 1, "akkuschrauber.jpg"),
    ("Bit Set", "Handwerkzeug", "Werkzeugwagen", 1, "bit_set.jpg"),
    ("Bohrer Set", "Handwerkzeug", "Werkzeugwagen", 1, "bohrer_set.jpg"),
    ("Bügelsäge", "Handwerkzeug", "Werkzeugwagen", 1, "buegelsaege.jpg"),
    ("Feinschraubendreher Set", "Handwerkzeug", "Werkzeugwagen", 1, "feinschraubendreher_set.jpg"),
    ("Hammer", "Handwerkzeug", "Werkzeugwagen", 1, "hammer.jpg"),
    ("Inbusschlüssel Set", "Handwerkzeug", "Werkzeugwagen", 1, "inbusschluesselset.jpg"),
    ("Lineal 15 cm", "Messmittel", "Werkzeugwagen", 1, "lineal_15cm.jpg"),
    ("Lineal 30 cm", "Messmittel", "Werkzeugwagen", 1, "lineal_30cm.jpg"),
    ("Maßband", "Messmittel", "Werkzeugwagen", 1, "massband.jpg"),
    ("Messleitungen", "Messmittel", "Werkzeugwagen", 1, "messleitungen.jpg"),
    ("Messschieber", "Messmittel", "Werkzeugwagen", 1, "messschieber.jpg"),
    ("Multimeter", "Messmittel", "Werkzeugwagen", 1, "multimeter.jpg"),
    ("Nadelfeilen Set", "Handwerkzeug", "Werkzeugwagen", 1, "nadelfeilen_set.jpg"),
    ("Ringschlüssel Set", "Handwerkzeug", "Werkzeugwagen", 1, "ringschluessel_set.jpg"),
    ("Schonhammer", "Handwerkzeug", "Werkzeugwagen", 1, "schonhammer.jpg"),
    ("Schraubendreher Set", "Handwerkzeug", "Werkzeugwagen", 1, "schraubendreher_set.jpg"),
    ("Seitenschneider", "Handwerkzeug", "Werkzeugwagen", 1, "seitenschneider.jpg"),
    ("Spitzzange", "Handwerkzeug", "Werkzeugwagen", 1, "spitzzange.jpg"),
    ("Steckschlüssel Set", "Handwerkzeug", "Werkzeugwagen", 1, "steckschluessel_set.jpg"),
    ("Wasserpumpenzange", "Handwerkzeug", "Werkzeugwagen", 1, "wasserpumpenzange.jpg"),
    ("Winkelschraubendreher", "Handwerkzeug", "Werkzeugwagen", 1, "winkelschraubendreher.jpg")
]

app = Flask(__name__)

########### Admin Login ############
app.secret_key = "ba_portal_secret_key_2026"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"
####################################


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "labtools.db")

RETURN_IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "return_images")
os.makedirs(RETURN_IMAGE_FOLDER, exist_ok=True)

# Debug-Ordner: hier werden zusätzliche Kopien der Steckschlüssel-Bilder
# mit rotem Live-Overlay und blauen Analyse-Polygonen gespeichert.
# Dieser Ordner ist nur zum Prüfen/Debuggen gedacht und kann später gelöscht werden.
RETURN_OVERLAY_DEBUG_FOLDER = os.path.join(BASE_DIR, "static", "Return_images_+_overlay")
os.makedirs(RETURN_OVERLAY_DEBUG_FOLDER, exist_ok=True)

# Abgabe-Debug: Hier werden Bilder mit den erkannten YOLO-Boxen gespeichert.
# Damit kann man nach jeder Rückgabe prüfen, was das Portal wirklich erkannt hat.
RETURN_IDENTITY_OVERLAY_FOLDER = os.path.join(BASE_DIR, "static", "return_identity_overlay")
os.makedirs(RETURN_IDENTITY_OVERLAY_FOLDER, exist_ok=True)

RETURN_COMPLETENESS_OVERLAY_FOLDER = os.path.join(BASE_DIR, "static", "return_completeness_overlay")
os.makedirs(RETURN_COMPLETENESS_OVERLAY_FOLDER, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("RETURN_IMAGE_FOLDER:", RETURN_IMAGE_FOLDER)
print("RETURN_OVERLAY_DEBUG_FOLDER:", RETURN_OVERLAY_DEBUG_FOLDER)
print("RETURN_IDENTITY_OVERLAY_FOLDER:", RETURN_IDENTITY_OVERLAY_FOLDER)
print("RETURN_COMPLETENESS_OVERLAY_FOLDER:", RETURN_COMPLETENESS_OVERLAY_FOLDER)

IDENTITY_MODEL_PATH = os.path.join(BASE_DIR, "models", "identity_best.pt")
IDENTITY_IMGSZ = 640
IDENTITY_CONF = 0.25

identity_model = YOLO(IDENTITY_MODEL_PATH)

TOOL_NAME_TO_IDENTITY_CLASS = {
    "Akkuschrauber": "akkuschrauber",
    "Bit Set": "bit_set",
    "Bohrer Set": "bohrer_set",
    "Bügelsäge": "buegelsaege",
    "Feinschraubendreher Set": "feinschraubendreher_set",
    "Hammer": "hammer",
    "Inbusschlüssel Set": "inbusschluesselset",
    "Lineal 15 cm": "lineal_15cm",
    "Lineal 30 cm": "lineal_30cm",
    "Maßband": "massband",
    "Messleitungen": "messleitungen",
    "Messschieber": "messschieber",
    "Multimeter": "multimeter",
    "Nadelfeilen Set": "nadelfeilen_set",
    "Ringschlüssel Set": "ringschluessel_set",
    "Schonhammer": "schonhammer",
    "Schraubendreher Set": "schraubendreher_set",
    "Seitenschneider": "seitenschneider",
    "Spitzzange": "spitzzange",
    "Steckschlüssel Set": "steckschluessel_set",
    "Wasserpumpenzange": "wasserpumpenzange",
    "Winkelschraubendreher": "winkelschraubendreher",
}


MULTIMETER_TOOL_NAME = "Multimeter"
MESSLEITUNGEN_TOOL_NAME = "Messleitungen"
MULTIMETER_REQUIRED_CLASSES = ["multimeter", "messleitungen"]


COMPLETENESS_REQUIRED_TOOLS = [
    "Akkuschrauber",
    "Steckschlüssel Set",
    "Nadelfeilen Set",
    "Ringschlüssel Set",
    "Schraubendreher Set"
]

COMPLETENESS_MODEL_PATHS = {
    "Akkuschrauber": os.path.join(BASE_DIR, "models", "completeness", "akkuschrauber.pt"),
    # Steckschlüssel Set wird über Slotanalyse geprüft, kein YOLO-Vollständigkeitsmodell nötig.
    "Nadelfeilen Set": os.path.join(BASE_DIR, "models", "completeness", "nadelfeilen_set.pt"),
    # Ringschlüssel Set wird jetzt über Slotanalyse geprüft, kein YOLO-Vollständigkeitsmodell nötig.
    "Schraubendreher Set": os.path.join(BASE_DIR, "models", "completeness", "schraubendreher_set.pt"),
}

COMPLETENESS_IMGSZ = 1280
COMPLETENESS_CONF = 0.25

completeness_models = {
    tool_name: YOLO(model_path)
    for tool_name, model_path in COMPLETENESS_MODEL_PATHS.items()
}

TRAINING_VOLL_FOLDER = os.path.join(BASE_DIR, "Voll_Werkzeuge")
# Der Ordner Voll_Werkzeuge wird im Portal nicht mehr automatisch erstellt.

TRAINING_VOLL_TOOLS = {
    "akkuschrauber": {
        "display_name": "Akkuschrauber",
        "folder": "Akkuschrauber",
        "overlay": "overlays/akkuschrauber_overlay.png",
        "orientation": "landscape"
    },
    "steckschluessel_set": {
        "display_name": "Steckschlüssel Set",
        "folder": "Steckschluessel_Set",
        "overlay": "overlays/steckschluessel_set_overlay.png",
        "orientation": "landscape"
    },
    "nadelfeilen_set": {
        "display_name": "Nadelfeilen Set",
        "folder": "Nadelfeilen_Set",
        "overlay": "overlays/nadelfeilen_set_overlay.png",
        "orientation": "landscape"
    },
    "ringschluessel_set": {
        "display_name": "Ringschlüssel Set",
        "folder": "Ringschluessel_Set",
        "overlay": "overlays/ringschluessel_set_overlay.png",
        "orientation": "portrait"
    },
    "schraubendreher_set": {
        "display_name": "Schraubendreher Set",
        "folder": "Schraubendreher_Set",
        "overlay": "overlays/schraubendreher_set_overlay.png",
        "orientation": "portrait"
    }
}


STECK_SLOT_ANALYSIS_JSON_PATH = os.path.join(
    BASE_DIR,
    "slot_reference",
    "steckschluessel_overlay_analysis_near.json"
)

STECK_SLOT_ANALYSIS_FALLBACK_JSON_PATH = os.path.join(
    BASE_DIR,
    "slot_reference",
    "steckschluessel_slots.json"
)

RING_SLOT_ANALYSIS_JSON_PATH = os.path.join(
    BASE_DIR,
    "slot_reference",
    "ringschluessel_overlay_analysis_near.json"
)

print("RING_SLOT_ANALYSIS_JSON_PATH:", RING_SLOT_ANALYSIS_JSON_PATH)
print("RING_JSON_EXISTIERT:", os.path.exists(RING_SLOT_ANALYSIS_JSON_PATH))

print("STECK_SLOT_ANALYSIS_JSON_PATH:", STECK_SLOT_ANALYSIS_JSON_PATH)
print("STECK_JSON_EXISTIERT:", os.path.exists(STECK_SLOT_ANALYSIS_JSON_PATH))
print("STECK_FALLBACK_JSON_EXISTIERT:", os.path.exists(STECK_SLOT_ANALYSIS_FALLBACK_JSON_PATH))

STECK_STANDARD_SLOT_MIN_METAL_RATIO = 0.08
STECK_SPECIAL_SLOT_MIN_METAL_RATIO = 0.03
STECK_SPECIAL_SLOT_MIN_VERTICAL_EDGE_RATIO = 0.12
STECK_EDGE_THRESHOLD = 45

# Ringschlüsselset Slotanalyse:
# Die blauen Analysebereiche liegen auf dem silbernen Metall der einzelnen Schlüssel.
# Ein Slot gilt als belegt, wenn genügend helle/metallische Fläche und Kanten im Polygon liegen.
RING_SLOT_MIN_METAL_RATIO = 0.10
RING_SLOT_MIN_EDGE_RATIO = 0.015
RING_EDGE_THRESHOLD = 38


def _point_to_xy(point):
    """Konvertiert verschiedene Punktformate in (x, y)."""
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])

    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])

    raise ValueError(f"Ungültiges Punktformat: {point}")


def _normalize_polygon_points(points):
    polygon = []

    for point in points:
        x, y = _point_to_xy(point)
        polygon.append([x, y])

    return polygon


def _is_polygon_points(obj):
    """Prüft, ob obj direkt eine Polygon-Punktliste ist."""
    if not isinstance(obj, list) or len(obj) < 2:
        return False

    first = obj[0]

    if isinstance(first, dict) and "x" in first and "y" in first:
        return True

    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return True

    return False


def _get_points_from_polygon_item(item):
    """Holt Punkte aus verschiedenen Objektstrukturen."""
    if _is_polygon_points(item):
        return item

    if isinstance(item, dict):
        for key in [
            "analysis_points", "overlay_points", "points", "polygon", "vertices",
            "coords", "coordinates", "pts", "data", "geometry"
        ]:
            value = item.get(key)
            if _is_polygon_points(value):
                return value

        # Falls geometry selbst wieder ein Dict mit points/polygon ist
        geom = item.get("geometry")
        if isinstance(geom, dict):
            for key in ["points", "polygon", "vertices", "coords", "coordinates"]:
                value = geom.get(key)
                if _is_polygon_points(value):
                    return value

    return None


def _item_text_blob(item):
    """Sammelt Textfelder eines JSON-Objekts für robuste Typ-/Farberkennung."""
    if not isinstance(item, dict):
        return ""

    values = []
    for key in [
        "type", "kind", "role", "mode", "color", "farbe", "name", "label",
        "category", "class", "class_name", "group", "usage", "purpose"
    ]:
        if key in item:
            values.append(str(item.get(key)).lower())

    return " ".join(values)


def _walk_json_items(obj):
    """Läuft rekursiv durch das JSON und liefert alle dict/list-Objekte."""
    yield obj

    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_json_items(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_json_items(value)


def _extract_steck_analysis_polygons(json_data):
    """
    Liest die blauen Analyse-Polygone aus der JSON-Datei.

    Deine aktuelle JSON-Struktur ist:
    - reference_image
    - set_outline
    - slots: Liste mit 31 Einträgen
      - id
      - overlay_points  = rote Live-/Slot-Umrisse
      - analysis_points = blaue Analysefläche
    """

    print("Slotanalyse: JSON Top-Level Keys:", list(json_data.keys()) if isinstance(json_data, dict) else type(json_data))

    polygons = []
    seen = set()

    def add_polygon(slot_id, points):
        if points is None:
            return

        try:
            polygon = _normalize_polygon_points(points)
        except Exception:
            return

        if len(polygon) < 3:
            return

        sig = tuple((round(x, 1), round(y, 1)) for x, y in polygon)
        if sig in seen:
            return
        seen.add(sig)

        try:
            slot_id_int = int(slot_id)
        except Exception:
            slot_id_int = len(polygons) + 1

        polygons.append({
            "slot_id": slot_id_int,
            "points": polygon
        })

    # Wichtigster Fall: deine JSON mit slots[*].analysis_points
    if isinstance(json_data, dict) and isinstance(json_data.get("slots"), list):
        print("Slotanalyse: slots-Key gefunden mit", len(json_data["slots"]), "Einträgen")

        for index, slot in enumerate(json_data["slots"], start=1):
            if not isinstance(slot, dict):
                continue

            slot_id = slot.get("id", index)
            points = slot.get("analysis_points")

            if points is not None:
                add_polygon(slot_id, points)

    # Fallback: andere mögliche direkte Keys
    if len(polygons) == 0 and isinstance(json_data, dict):
        direct_keys = [
            "analysis_polygons",
            "slot_analysis_polygons",
            "blue_polygons",
            "analysis_slots",
            "slots_analysis",
            "analysis_areas",
            "blue_analysis",
            "blue",
        ]

        for key in direct_keys:
            value = json_data.get(key)
            if isinstance(value, list):
                print(f"Slotanalyse: direkte Analyse-Key gefunden: {key} mit {len(value)} Einträgen")

                for index, item in enumerate(value, start=1):
                    if isinstance(item, dict):
                        slot_id = item.get("id") or item.get("slot_id") or index
                        points = item.get("analysis_points") or _get_points_from_polygon_item(item)
                    else:
                        slot_id = index
                        points = item

                    add_polygon(slot_id, points)

    # Letzter Fallback: rekursive Suche nach analysis_points
    if len(polygons) == 0:
        for item in _walk_json_items(json_data):
            if not isinstance(item, dict):
                continue

            if "analysis_points" in item:
                slot_id = item.get("id") or item.get("slot_id") or len(polygons) + 1
                add_polygon(slot_id, item.get("analysis_points"))

    polygons.sort(key=lambda p: p["slot_id"])

    print("Slotanalyse: Analyse-Polygone gefunden:", len(polygons))

    if len(polygons) == 0:
        raise ValueError("Keine gültigen Analyse-Polygone gefunden.")

    return polygons

def _get_json_reference_size(json_data, image):
    image_h, image_w = image.shape[:2]

    if isinstance(json_data, dict):
        ref_w = (
            json_data.get("image_width")
            or json_data.get("width")
            or json_data.get("ref_width")
            or json_data.get("reference_width")
        )
        ref_h = (
            json_data.get("image_height")
            or json_data.get("height")
            or json_data.get("ref_height")
            or json_data.get("reference_height")
        )

        if isinstance(json_data.get("image_size"), dict):
            ref_w = ref_w or json_data["image_size"].get("width")
            ref_h = ref_h or json_data["image_size"].get("height")

        if ref_w and ref_h:
            return float(ref_w), float(ref_h)

        # Deine JSON enthält reference_image. Daraus lesen wir die echte Referenzgröße.
        reference_image = json_data.get("reference_image")
        if reference_image:
            possible_reference_paths = [
                reference_image,
                os.path.join(BASE_DIR, reference_image),
                os.path.join(BASE_DIR, "slot_reference", os.path.basename(reference_image)),
            ]

            for path in possible_reference_paths:
                if path and os.path.exists(path):
                    ref_img = cv2.imread(path)
                    if ref_img is not None:
                        ref_h_img, ref_w_img = ref_img.shape[:2]
                        print("Slotanalyse: Referenzbildgröße aus reference_image:", ref_w_img, ref_h_img)
                        return float(ref_w_img), float(ref_h_img)

        # Fallback: maximale Koordinaten aus den Polygonen bestimmen.
        max_x = 0.0
        max_y = 0.0

        def collect_points(obj):
            if isinstance(obj, dict):
                for key in ["set_outline", "overlay_points", "analysis_points", "points", "polygon"]:
                    points = obj.get(key)
                    if _is_polygon_points(points):
                        for x, y in _normalize_polygon_points(points):
                            yield x, y

                for value in obj.values():
                    yield from collect_points(value)

            elif isinstance(obj, list):
                if _is_polygon_points(obj):
                    for x, y in _normalize_polygon_points(obj):
                        yield x, y
                else:
                    for value in obj:
                        yield from collect_points(value)

        for x, y in collect_points(json_data):
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        if max_x > 0 and max_y > 0:
            # kleine Sicherheitsreserve, falls äußerste Kante nicht bei voller Bildgröße liegt
            guessed_w = max(float(image_w), max_x)
            guessed_h = max(float(image_h), max_y)
            print("Slotanalyse: Referenzgröße aus Polygon-Maximum geschätzt:", guessed_w, guessed_h)
            return guessed_w, guessed_h

    # Fallback: Polygone gelten als Koordinaten im aktuellen Bild.
    print("Slotanalyse: Fallback Referenzgröße = Bildgröße")
    return float(image_w), float(image_h)

def _load_steck_analysis_config(image):
    json_path = STECK_SLOT_ANALYSIS_JSON_PATH

    if not os.path.exists(json_path):
        json_path = STECK_SLOT_ANALYSIS_FALLBACK_JSON_PATH

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Slotanalyse-JSON nicht gefunden. Erwartet: "
            f"{STECK_SLOT_ANALYSIS_JSON_PATH}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    polygons = _extract_steck_analysis_polygons(json_data)
    ref_w, ref_h = _get_json_reference_size(json_data, image)

    return polygons, ref_w, ref_h, json_path



def _extract_steck_live_overlay_polygons(json_data):
    """
    Liest die roten Live-/Ausrichtungs-Polygone aus der JSON-Datei.

    Deine aktuelle JSON:
    - set_outline = roter äußerer Umriss
    - slots[*].overlay_points = rote Slot-/Werkzeugumrisse
    """

    polygons = []
    seen = set()

    def add_polygon(points):
        if points is None:
            return

        try:
            polygon = _normalize_polygon_points(points)
        except Exception:
            return

        if len(polygon) < 3:
            return

        sig = tuple((round(x, 1), round(y, 1)) for x, y in polygon)
        if sig in seen:
            return
        seen.add(sig)

        polygons.append({
            "points": polygon
        })

    # Wichtigster Fall: set_outline und slots[*].overlay_points
    if isinstance(json_data, dict):
        if "set_outline" in json_data:
            add_polygon(json_data.get("set_outline"))

        if isinstance(json_data.get("slots"), list):
            for slot in json_data["slots"]:
                if isinstance(slot, dict):
                    add_polygon(slot.get("overlay_points"))

    # Fallback für andere JSON-Strukturen
    if len(polygons) == 0 and isinstance(json_data, dict):
        direct_keys = [
            "overlay_polygons",
            "red_polygons",
            "live_overlay_polygons",
            "live_polygons",
            "outline_polygons",
            "red_overlay",
            "live_overlay",
            "outlines",
            "red",
            "overlay",
        ]

        for key in direct_keys:
            value = json_data.get(key)
            if isinstance(value, list):
                print(f"Overlay-Debug: direkte Live-Key gefunden: {key} mit {len(value)} Einträgen")
                for item in value:
                    add_polygon(_get_points_from_polygon_item(item) if isinstance(item, dict) else item)

    print("Overlay-Debug: Rote Live-Polygone gefunden:", len(polygons))

    return polygons

def _scaled_polygon_points(polygon_points, ref_w, ref_h, image_w, image_h):
    scale_x = image_w / ref_w
    scale_y = image_h / ref_h

    return np.array([
        [int(round(x * scale_x)), int(round(y * scale_y))]
        for x, y in polygon_points
    ], dtype=np.int32)


def _draw_steck_debug_polygon(image_bgr, polygon_points, ref_w, ref_h, color, thickness=3, label=None):
    image_h, image_w = image_bgr.shape[:2]

    if len(polygon_points) < 3:
        return

    pts = _scaled_polygon_points(
        polygon_points=polygon_points,
        ref_w=ref_w,
        ref_h=ref_h,
        image_w=image_w,
        image_h=image_h
    )

    cv2.polylines(
        image_bgr,
        [pts],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    if label is not None:
        x0, y0 = pts[0]
        cv2.putText(
            image_bgr,
            str(label),
            (int(x0), max(22, int(y0) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )


def save_steckschluessel_overlay_debug_image(image_path, loan_id=None, photo_number=None):
    """
    Speichert zusätzlich zum Originalbild eine Debug-Kopie mit Overlays.

    Zielordner:
        static/Return_images_+_overlay

    Farben im Debug-Bild:
        Rot  = Live-/Ausrichtungs-Overlay
        Blau = blaue Analyse-Polygone/Slots

    Diese Funktion verändert NICHT das Originalbild in static/return_images.
    """
    try:
        if not os.path.exists(image_path):
            print("Overlay-Debug: Bild nicht gefunden:", image_path)
            return None

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print("Overlay-Debug: Bild konnte nicht gelesen werden:", image_path)
            return None

        possible_json_paths = [
            STECK_SLOT_ANALYSIS_JSON_PATH,
            STECK_SLOT_ANALYSIS_FALLBACK_JSON_PATH,
            os.path.join(BASE_DIR, "slot_reference", "steckschluessel_overlay_analysis_near.json"),
            os.path.join(BASE_DIR, "slot_reference", "steckschluessel_slots.json"),
            os.path.join("slot_reference", "steckschluessel_overlay_analysis_near.json"),
            os.path.join("slot_reference", "steckschluessel_slots.json"),
        ]

        json_path = None
        print("Overlay-Debug: gesuchte JSON-Pfade:")
        for p in possible_json_paths:
            print(" -", os.path.abspath(p), "exists=", os.path.exists(p))
            if json_path is None and p and os.path.exists(p):
                json_path = p

        if json_path is None:
            print("Overlay-Debug: Keine JSON gefunden.")
            return None

        print("Overlay-Debug: Verwende JSON:", os.path.abspath(json_path))

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        red_polygons = _extract_steck_live_overlay_polygons(json_data)
        blue_polygons = _extract_steck_analysis_polygons(json_data)
        ref_w, ref_h = _get_json_reference_size(json_data, image_bgr)

        print("Overlay-Debug: Referenzgröße:", ref_w, ref_h)
        print("Overlay-Debug: Bildgröße:", image_bgr.shape[1], image_bgr.shape[0])
        print("Overlay-Debug: Rot:", len(red_polygons), "Blau:", len(blue_polygons))

        if len(red_polygons) == 0 and len(blue_polygons) == 0:
            print("Overlay-Debug: Keine roten und keine blauen Polygone gefunden.")
            return None

        debug_image = image_bgr.copy()

        # Rot: Live-/Ausrichtungs-Overlay
        for poly in red_polygons:
            _draw_steck_debug_polygon(
                image_bgr=debug_image,
                polygon_points=poly["points"],
                ref_w=ref_w,
                ref_h=ref_h,
                color=(0, 0, 255),
                thickness=3,
                label=None
            )

        # Blau: Analyse-Polygone mit Slotnummer
        for poly in blue_polygons:
            _draw_steck_debug_polygon(
                image_bgr=debug_image,
                polygon_points=poly["points"],
                ref_w=ref_w,
                ref_h=ref_h,
                color=(255, 0, 0),
                thickness=3,
                label=poly.get("slot_id")
            )

        base_name = os.path.basename(image_path)
        name_without_ext, _ = os.path.splitext(base_name)

        if photo_number is not None:
            debug_filename = f"{name_without_ext}_overlay_debug_photo_{photo_number}.jpg"
        else:
            debug_filename = f"{name_without_ext}_overlay_debug.jpg"

        debug_path = os.path.join(RETURN_OVERLAY_DEBUG_FOLDER, debug_filename)
        os.makedirs(RETURN_OVERLAY_DEBUG_FOLDER, exist_ok=True)

        ok = cv2.imwrite(debug_path, debug_image)

        print("Overlay-Debug cv2.imwrite ok:", ok)
        print("Overlay-Debug gespeichert:", debug_path)

        if not ok:
            return None

        return debug_path

    except Exception as error:
        print("Overlay-Debug Fehler:", error)
        return None

def _analyze_steck_slot(image_bgr, polygon_points, ref_w, ref_h):
    image_h, image_w = image_bgr.shape[:2]

    scale_x = image_w / ref_w
    scale_y = image_h / ref_h

    pts = np.array([
        [int(round(x * scale_x)), int(round(y * scale_y))]
        for x, y in polygon_points
    ], dtype=np.int32)

    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    pixel_count = int(np.count_nonzero(mask))

    if pixel_count < 20:
        return {
            "metal_ratio": 0.0,
            "vertical_edge_ratio": 0.0,
            "pixel_count": pixel_count
        }

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv)

    # Silber/Metall im Steckschlüsselsatz: relativ hell, geringe bis mittlere Sättigung.
    # Schwarze leere Slots: deutlich dunkler.
    metal_mask = (
        (mask > 0)
        & (v >= 70)
        & (s <= 95)
    )

    metal_ratio = float(np.count_nonzero(metal_mask)) / float(pixel_count)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    vertical_edges = (
        (mask > 0)
        & (np.abs(sobel_x) >= STECK_EDGE_THRESHOLD)
    )

    vertical_edge_ratio = float(np.count_nonzero(vertical_edges)) / float(pixel_count)

    return {
        "metal_ratio": metal_ratio,
        "vertical_edge_ratio": vertical_edge_ratio,
        "pixel_count": pixel_count
    }


def check_steckschluessel_slot_analysis(image_path):
    """
    Prüft das Steckschlüssel Set über feste blaue Analyse-Polygone.

    Kein altes YOLO-Vollständigkeitsmodell wird hier verwendet.
    Das Bild muss bereits 4:3 Landscape und passend zum Referenzbild gespeichert sein.
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        return {
            "ok": False,
            "result_type": "incomplete",
            "message": "Bild konnte für die Slotanalyse nicht gelesen werden.",
            "slot_analysis": True
        }

    try:
        polygons, ref_w, ref_h, json_path = _load_steck_analysis_config(image_bgr)
    except Exception as error:
        return {
            "ok": False,
            "result_type": "incomplete",
            "message": f"Slotanalyse-Konfiguration fehlt oder ist ungültig: {error}",
            "slot_analysis": True
        }

    slot_results = []
    missing_slots = []
    occupied_slots = []

    for polygon in polygons:
        slot_id = polygon["slot_id"]
        metrics = _analyze_steck_slot(
            image_bgr,
            polygon["points"],
            ref_w,
            ref_h
        )

        metal_ratio = metrics["metal_ratio"]
        vertical_edge_ratio = metrics["vertical_edge_ratio"]

        if slot_id == 31:
            is_occupied = (
                metal_ratio >= STECK_SPECIAL_SLOT_MIN_METAL_RATIO
                and vertical_edge_ratio >= STECK_SPECIAL_SLOT_MIN_VERTICAL_EDGE_RATIO
            )
        else:
            is_occupied = metal_ratio >= STECK_STANDARD_SLOT_MIN_METAL_RATIO

        if is_occupied:
            occupied_slots.append(slot_id)
        else:
            missing_slots.append(slot_id)

        slot_results.append({
            "slot": slot_id,
            "occupied": bool(is_occupied),
            "metal_ratio": round(metal_ratio, 4),
            "vertical_edge_ratio": round(vertical_edge_ratio, 4),
            "pixel_count": metrics["pixel_count"]
        })

    print("RINGSCHLUESSEL SLOT RESULTS:", slot_results)
    print("RINGSCHLUESSEL FEHLENDE SLOTS:", missing_slots)

    if len(missing_slots) == 0:
        return {
            "ok": True,
            "result_type": "success",
            "detected_class": "slot_analysis_complete",
            "confidence": 1.0,
            "message": "Steckschlüssel-Set vollständig. Kein Slot ist leer.",
            "tool_name": "Steckschlüssel Set",
            "slot_unit": "slot",
            "slot_analysis": True,
            "slot_config": json_path,
            "slot_count": len(slot_results),
            "occupied_slots": occupied_slots,
            "missing_slots": missing_slots,
            "empty_slots": missing_slots,
            "slot_results": slot_results
        }

    return {
        "ok": False,
        "result_type": "incomplete",
        "detected_class": "slot_analysis_missing_slots",
        "confidence": 1.0,
        "message": de_empty_slots_message(len(missing_slots)),
        "tool_name": "Steckschlüssel Set",
        "slot_unit": "slot",
        "slot_analysis": True,
        "slot_config": json_path,
        "slot_count": len(slot_results),
        "occupied_slots": occupied_slots,
        "missing_slots": missing_slots,
        "empty_slots": missing_slots,
        "slot_results": slot_results
    }



def _load_ring_analysis_config(image):
    json_path = RING_SLOT_ANALYSIS_JSON_PATH

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Ringschlüssel-Slotanalyse-JSON nicht gefunden. Erwartet: "
            f"{RING_SLOT_ANALYSIS_JSON_PATH}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    polygons = _extract_steck_analysis_polygons(json_data)
    ref_w, ref_h = _get_json_reference_size(json_data, image)

    return polygons, ref_w, ref_h, json_path


def save_ringschluessel_overlay_debug_image(image_path, loan_id=None, photo_number=None):
    """
    Speichert für das Ringschlüsselset eine Debug-Kopie mit:
    Rot  = Live-/Ausrichtungs-Overlay
    Blau = Analyse-Polygone/Slots

    Zielordner:
        static/Return_images_+_overlay
    """
    try:
        if not os.path.exists(image_path):
            print("Ring-Overlay-Debug: Bild nicht gefunden:", image_path)
            return None

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print("Ring-Overlay-Debug: Bild konnte nicht gelesen werden:", image_path)
            return None

        if not os.path.exists(RING_SLOT_ANALYSIS_JSON_PATH):
            print("Ring-Overlay-Debug: JSON nicht gefunden:", RING_SLOT_ANALYSIS_JSON_PATH)
            return None

        print("Ring-Overlay-Debug: Verwende JSON:", os.path.abspath(RING_SLOT_ANALYSIS_JSON_PATH))

        with open(RING_SLOT_ANALYSIS_JSON_PATH, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        red_polygons = _extract_steck_live_overlay_polygons(json_data)
        blue_polygons = _extract_steck_analysis_polygons(json_data)
        ref_w, ref_h = _get_json_reference_size(json_data, image_bgr)

        print("Ring-Overlay-Debug: Referenzgröße:", ref_w, ref_h)
        print("Ring-Overlay-Debug: Bildgröße:", image_bgr.shape[1], image_bgr.shape[0])
        print("Ring-Overlay-Debug: Rot:", len(red_polygons), "Blau:", len(blue_polygons))

        debug_image = image_bgr.copy()

        for poly in red_polygons:
            _draw_steck_debug_polygon(
                image_bgr=debug_image,
                polygon_points=poly["points"],
                ref_w=ref_w,
                ref_h=ref_h,
                color=(0, 0, 255),
                thickness=3,
                label=None
            )

        for poly in blue_polygons:
            _draw_steck_debug_polygon(
                image_bgr=debug_image,
                polygon_points=poly["points"],
                ref_w=ref_w,
                ref_h=ref_h,
                color=(255, 0, 0),
                thickness=3,
                label=poly.get("slot_id")
            )

        base_name = os.path.basename(image_path)
        name_without_ext, _ = os.path.splitext(base_name)

        if photo_number is not None:
            debug_filename = f"{name_without_ext}_ringschluessel_overlay_debug_photo_{photo_number}.jpg"
        else:
            debug_filename = f"{name_without_ext}_ringschluessel_overlay_debug.jpg"

        debug_path = os.path.join(RETURN_OVERLAY_DEBUG_FOLDER, debug_filename)
        os.makedirs(RETURN_OVERLAY_DEBUG_FOLDER, exist_ok=True)

        ok = cv2.imwrite(debug_path, debug_image)

        print("Ring-Overlay-Debug cv2.imwrite ok:", ok)
        print("Ring-Overlay-Debug gespeichert:", debug_path)

        return debug_path if ok else None

    except Exception as error:
        print("Ring-Overlay-Debug Fehler:", error)
        return None


def _analyze_ring_slot(image_bgr, polygon_points, ref_w, ref_h):
    """
    Prüft, ob im blauen Analysepolygon ein Ringschlüssel liegt.

    Farbunabhängig vom Wagen:
    - Metall ist meistens hell und wenig gesättigt.
    - Zusätzlich werden Kanten berücksichtigt, damit helle glatte Hintergründe
      nicht zu leicht als Metall zählen.
    """
    image_h, image_w = image_bgr.shape[:2]

    pts = _scaled_polygon_points(
        polygon_points=polygon_points,
        ref_w=ref_w,
        ref_h=ref_h,
        image_w=image_w,
        image_h=image_h
    )

    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    pixel_count = int(np.count_nonzero(mask))

    if pixel_count < 20:
        return {
            "metal_ratio": 0.0,
            "edge_ratio": 0.0,
            "pixel_count": pixel_count,
            "mean_v": 0.0,
            "mean_s": 0.0
        }

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    h, sat, val = cv2.split(hsv)

    # Silber/Metall: hell und nicht stark farbig.
    metal_mask = (
        (mask > 0)
        & (val >= 85)
        & (sat <= 115)
    )

    metal_ratio = float(np.count_nonzero(metal_mask)) / float(pixel_count)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.abs(sobel_x) + np.abs(sobel_y)

    edge_mask = (
        (mask > 0)
        & (edge_strength >= RING_EDGE_THRESHOLD)
    )

    edge_ratio = float(np.count_nonzero(edge_mask)) / float(pixel_count)

    mean_v = float(np.mean(val[mask > 0]))
    mean_s = float(np.mean(sat[mask > 0]))

    return {
        "metal_ratio": metal_ratio,
        "edge_ratio": edge_ratio,
        "pixel_count": pixel_count,
        "mean_v": mean_v,
        "mean_s": mean_s
    }


def check_ringschluessel_slot_analysis(image_path):
    """
    Prüft das Ringschlüsselset über 15 feste blaue Analyse-Polygone.

    Kein YOLO-Vollständigkeitsmodell wird hier verwendet.
    Vollständig = alle 15 Analysebereiche sind mit Metall belegt.
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        return {
            "ok": False,
            "result_type": "incomplete",
            "message": "Bild konnte für die Ringschlüssel-Slotanalyse nicht gelesen werden.",
            "slot_analysis": True
        }

    try:
        polygons, ref_w, ref_h, json_path = _load_ring_analysis_config(image_bgr)
    except Exception as error:
        return {
            "ok": False,
            "result_type": "incomplete",
            "message": f"Ringschlüssel-Slotanalyse-Konfiguration fehlt oder ist ungültig: {error}",
            "slot_analysis": True
        }

    slot_results = []
    missing_slots = []
    occupied_slots = []

    for polygon in polygons:
        slot_id = polygon["slot_id"]
        metrics = _analyze_ring_slot(
            image_bgr=image_bgr,
            polygon_points=polygon["points"],
            ref_w=ref_w,
            ref_h=ref_h
        )

        metal_ratio = metrics["metal_ratio"]
        edge_ratio = metrics["edge_ratio"]

        # Für Ringschlüssel reicht der Metallanteil als Hauptkriterium.
        # Der Kantenwert bleibt nur zur Kontrolle im Debug, weil glatte/reflektierende
        # Schlüssel teilweise wenig Kanten liefern und sonst fälschlich als fehlend gelten.
        is_occupied = (
            metal_ratio >= RING_SLOT_MIN_METAL_RATIO
        )

        if is_occupied:
            occupied_slots.append(slot_id)
        else:
            missing_slots.append(slot_id)

        slot_results.append({
            "slot": slot_id,
            "occupied": bool(is_occupied),
            "metal_ratio": round(metal_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
            "pixel_count": metrics["pixel_count"],
            "mean_v": round(metrics["mean_v"], 2),
            "mean_s": round(metrics["mean_s"], 2)
        })

    print("RINGSCHLUESSEL SLOT RESULTS:", slot_results)
    print("RINGSCHLUESSEL FEHLENDE SLOTS:", missing_slots)

    if len(missing_slots) == 0:
        return {
            "ok": True,
            "result_type": "success",
            "detected_class": "ring_slot_analysis_complete",
            "confidence": 1.0,
            "message": "Ringschlüsselset vollständig. Kein Schlüssel fehlt.",
            "tool_name": "Ringschlüssel Set",
            "slot_unit": "schluessel",
            "slot_analysis": True,
            "slot_config": json_path,
            "slot_count": len(slot_results),
            "occupied_slots": occupied_slots,
            "missing_slots": missing_slots,
            "empty_slots": missing_slots,
            "slot_results": slot_results
        }

    return {
        "ok": False,
        "result_type": "incomplete",
        "detected_class": "ring_slot_analysis_missing_slots",
        "confidence": 1.0,
        "message": de_missing_message(len(missing_slots), "Schlüssel", "Schlüssel"),
        "tool_name": "Ringschlüssel Set",
        "slot_unit": "schluessel",
        "slot_analysis": True,
        "slot_config": json_path,
        "slot_count": len(slot_results),
        "occupied_slots": occupied_slots,
        "missing_slots": missing_slots,
        "empty_slots": missing_slots,
        "slot_results": slot_results
    }

def _extract_yolo_detections(result, model):
    """Extrahiert YOLO-Detection-Boxen als einfache Liste."""
    detections = []

    if result is None or result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class": cls_name,
            "confidence": conf,
            "xyxy": xyxy
        })

    return detections


def _draw_text_panel(image_bgr, lines):
    """Schreibt einen gut sichtbaren Textblock oben links ins Bild."""
    if image_bgr is None or not lines:
        return image_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    line_height = 34
    padding = 12

    max_width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(str(line), font, font_scale, thickness)
        max_width = max(max_width, w)

    panel_w = min(image_bgr.shape[1] - 20, max_width + 2 * padding)
    panel_h = min(image_bgr.shape[0] - 20, len(lines) * line_height + 2 * padding)

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    image_bgr = cv2.addWeighted(overlay, 0.60, image_bgr, 0.40, 0)

    y = 10 + padding + 22
    for line in lines:
        cv2.putText(
            image_bgr,
            str(line),
            (10 + padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        y += line_height

    return image_bgr


def _safe_overlay_filename(image_path, suffix):
    base_name = os.path.basename(image_path)
    name_without_ext, _ = os.path.splitext(base_name)
    return f"{name_without_ext}_{suffix}.jpg"


def save_identity_overlay_debug_image(image_path, expected_tool_name):
    """
    Speichert das Identitätsbild mit YOLO-Rahmen.
    Ordner: static/return_identity_overlay
    """
    try:
        expected_class = TOOL_NAME_TO_IDENTITY_CLASS.get(expected_tool_name, "unbekannt")

        results = identity_model.predict(
            source=image_path,
            imgsz=IDENTITY_IMGSZ,
            conf=IDENTITY_CONF,
            verbose=False
        )

        if not results:
            print("Identity-Overlay: keine YOLO-Ergebnisse")
            return None

        result = results[0]
        detections = _extract_yolo_detections(result, identity_model)

        annotated = result.plot()

        expected_detections = [
            d for d in detections
            if d["class"] == expected_class
        ]

        best_any = max(detections, key=lambda d: d["confidence"], default=None)
        best_expected = max(expected_detections, key=lambda d: d["confidence"], default=None)

        lines = [
            "IDENTITAETSPRUEFUNG",
            f"Erwartet: {expected_class}",
            f"Erwartete Treffer: {len(expected_detections)}",
        ]

        if best_expected:
            lines.append(f"Beste erwartete Box: {best_expected['confidence']:.2f}")
        elif best_any:
            lines.append(f"Beste andere Box: {best_any['class']} {best_any['confidence']:.2f}")
        else:
            lines.append("Keine Box erkannt")

        annotated = _draw_text_panel(annotated, lines)

        filename = _safe_overlay_filename(image_path, "identity_overlay")
        output_path = os.path.join(RETURN_IDENTITY_OVERLAY_FOLDER, filename)
        os.makedirs(RETURN_IDENTITY_OVERLAY_FOLDER, exist_ok=True)

        ok = cv2.imwrite(output_path, annotated)
        print("Identity-Overlay gespeichert:", output_path, "ok=", ok)

        return output_path if ok else None

    except Exception as error:
        print("Identity-Overlay Fehler:", error)
        return None


def _completeness_summary_lines(tool_name, detections):
    """Erzeugt verständliche Debug-Zeilen für Vollständigkeitsbilder."""
    classes = [d["class"] for d in detections]

    if tool_name == "Akkuschrauber":
        akkuschrauber_count = sum(1 for d in detections if d["class"] == "akkuschrauber")
        akku_count = sum(1 for d in detections if d["class"] == "akku")
        return [
            "VOLLSTAENDIGKEIT: AKKUSCHRAUBER",
            f"akkuschrauber erkannt: {akkuschrauber_count}",
            f"akku erkannt: {akku_count}",
            "Vollstaendig = akkuschrauber + akku"
        ]

    if tool_name == "Schraubendreher Set":
        griff_all = [d for d in detections if d["class"] == "griff"]
        griff_080 = [d for d in detections if d["class"] == "griff" and d["confidence"] >= 0.80]
        return [
            "VOLLSTAENDIGKEIT: SCHRAUBENDREHER SET",
            f"Griffe erkannt gesamt: {len(griff_all)}",
            f"Griffe ab 0.80: {len(griff_080)}/6",
            "Vollstaendig = mindestens 6 Griffe ab 0.80"
        ]

    if tool_name == "Nadelfeilen Set":
        voll = sum(1 for d in detections if d["class"] == "vollstaendig")
        unvoll = sum(1 for d in detections if d["class"] == "unvollstaendig")
        return [
            "VOLLSTAENDIGKEIT: NADELFEILEN SET",
            f"vollstaendig: {voll}",
            f"unvollstaendig: {unvoll}"
        ]

    if tool_name == "Ringschlüssel Set":
        voll_detections = [d for d in detections if d["class"] == "ringschluessel_set_vollstaendig"]
        unvoll_detections = [d for d in detections if d["class"] == "ringschluessel_set_unvollstaendig"]
        best_voll_conf = max([d["confidence"] for d in voll_detections], default=0.0)
        best_unvoll_conf = max([d["confidence"] for d in unvoll_detections], default=0.0)
        return [
            "VOLLSTAENDIGKEIT: RINGSCHLUESSEL SET",
            f"vollstaendig Boxen: {len(voll_detections)}",
            f"beste vollstaendig Conf: {best_voll_conf:.2f}",
            f"unvollstaendig Boxen: {len(unvoll_detections)}",
            f"beste unvollstaendig Conf: {best_unvoll_conf:.2f}",
            "Vollstaendig nur ab 0.85"
        ]

    return [
        f"VOLLSTAENDIGKEIT: {tool_name}",
        f"Boxen erkannt: {len(detections)}"
    ]


def save_completeness_overlay_debug_image(image_path, tool_name):
    """
    Speichert Vollständigkeitsbild mit YOLO-Rahmen.
    Für Steckschlüssel Set wird kein YOLO benutzt; dort übernimmt die Slotanalyse-Funktion.
    Ordner: static/return_completeness_overlay
    """
    try:
        if tool_name in ["Steckschlüssel Set", "Ringschlüssel Set"]:
            print("Completeness-Overlay:", tool_name, "nutzt Slotanalyse, kein YOLO-Overlay.")
            return None

        model = completeness_models.get(tool_name)

        if model is None:
            print("Completeness-Overlay: kein Modell für", tool_name)
            return None

        results = model.predict(
            source=image_path,
            imgsz=COMPLETENESS_IMGSZ,
            conf=COMPLETENESS_CONF,
            verbose=False
        )

        if not results:
            print("Completeness-Overlay: keine YOLO-Ergebnisse")
            return None

        result = results[0]
        detections = _extract_yolo_detections(result, model)
        annotated = result.plot()

        lines = _completeness_summary_lines(tool_name, detections)
        annotated = _draw_text_panel(annotated, lines)

        filename = _safe_overlay_filename(image_path, "completeness_overlay")
        output_path = os.path.join(RETURN_COMPLETENESS_OVERLAY_FOLDER, filename)
        os.makedirs(RETURN_COMPLETENESS_OVERLAY_FOLDER, exist_ok=True)

        ok = cv2.imwrite(output_path, annotated)
        print("Completeness-Overlay gespeichert:", output_path, "ok=", ok)
        print("Completeness-Overlay detections:", [
            {"class": d["class"], "confidence": round(d["confidence"], 3)}
            for d in detections
        ])

        return output_path if ok else None

    except Exception as error:
        print("Completeness-Overlay Fehler:", error)
        return None

def check_completeness(image_path, tool_name):
    # Steckschlüssel Set wird ab jetzt NICHT mehr mit dem alten YOLO-Vollständigkeitsmodell geprüft.
    # Stattdessen werden die festen blauen Slot-Analyse-Polygone verwendet.
    if tool_name == "Steckschlüssel Set":
        return check_steckschluessel_slot_analysis(image_path)

    if tool_name == "Ringschlüssel Set":
        return check_ringschluessel_slot_analysis(image_path)

    model = completeness_models.get(tool_name)

    if model is None:
        return {
            "ok": False,
            "result_type": "incomplete",
            "message": "Kein Vollständigkeitsmodell für dieses Werkzeug gefunden."
        }

    results = model.predict(
        source=image_path,
        imgsz=COMPLETENESS_IMGSZ,
        conf=COMPLETENESS_CONF,
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            detections.append({
                "class": cls_name,
                "confidence": conf
            })

    detected_classes = [d["class"] for d in detections]
    max_confidence = max([d["confidence"] for d in detections], default=0.0)

    # --------------------------------------------------
    # Akkuschrauber
    # Klassen: akkuschrauber, akku
    # Vollständig = akkuschrauber + akku erkannt
    # --------------------------------------------------
    if tool_name == "Akkuschrauber":
        has_akkuschrauber = "akkuschrauber" in detected_classes
        has_akku = "akku" in detected_classes

        if has_akkuschrauber and has_akku:
            return {
                "ok": True,
                "result_type": "success",
                "detected_class": "akkuschrauber + akku",
                "confidence": max_confidence,
                "message": "Akkuschrauber vollständig. Der Akku ist vorhanden."
            }

        if has_akkuschrauber and not has_akku:
            return {
                "ok": False,
                "result_type": "incomplete",
                "detected_class": "akkuschrauber",
                "confidence": max_confidence,
                "message": "Der <strong>Akku</strong> fehlt."
            }

        if has_akku and not has_akkuschrauber:
            return {
                "ok": False,
                "result_type": "incomplete",
                "detected_class": "akku",
                "confidence": max_confidence,
                "message": "Der <strong>Akkuschrauber</strong> fehlt."
            }

        return {
            "ok": False,
            "result_type": "incomplete",
            "detected_class": None,
            "confidence": 0.0,
            "message": "Akkuschrauber wurde nicht sicher erkannt."
        }

        # --------------------------------------------------
    # Steckschlüssel Set
    # Klassen: steckschluessel_set, unvoll_steckschluessel_set, leer_slot
    #
    # Wichtig:
    # leer_slot und unvoll_steckschluessel_set sind fehleranfällig.
    # Deshalb zählen sie nur mit hoher Confidence.
    # --------------------------------------------------
    if tool_name == "Steckschlüssel Set":
        MIN_LEER_SLOT_CONF = 0.90
        MIN_UNVOLL_SET_CONF = 0.90
        MIN_VOLL_SET_CONF = 0.50

        leer_slot_detections = [
            d for d in detections
            if d["class"] == "leer_slot" and d["confidence"] >= MIN_LEER_SLOT_CONF
        ]

        unvoll_set_detections = [
            d for d in detections
            if d["class"] == "unvoll_steckschluessel_set" and d["confidence"] >= MIN_UNVOLL_SET_CONF
        ]

        voll_set_detections = [
            d for d in detections
            if d["class"] == "steckschluessel_set" and d["confidence"] >= MIN_VOLL_SET_CONF
        ]

        if leer_slot_detections:
            best_conf = max(d["confidence"] for d in leer_slot_detections)

            return {
                "ok": False,
                "result_type": "incomplete",
                "detected_class": "leer_slot",
                "confidence": best_conf,
                "message": de_empty_slots_message(len(leer_slot_detections))
            }

        if unvoll_set_detections:
            best_conf = max(d["confidence"] for d in unvoll_set_detections)

            return {
                "ok": False,
                "result_type": "incomplete",
                "detected_class": "unvoll_steckschluessel_set",
                "confidence": best_conf,
                "message": "Steckschlüssel Set ist unvollständig."
            }

        if voll_set_detections:
            best_conf = max(d["confidence"] for d in voll_set_detections)

            return {
                "ok": True,
                "result_type": "success",
                "detected_class": "steckschluessel_set",
                "confidence": best_conf,
                "message": "Steckschlüssel-Set vollständig. Kein Slot ist leer."
            }

        return {
            "ok": False,
            "result_type": "incomplete",
            "detected_class": None,
            "confidence": 0.0,
            "message": "Steckschlüssel Set wurde nicht sicher erkannt. Bitte Bild erneut aufnehmen."
        }

        # --------------------------------------------------
    # Schraubendreher Set
    # Klasse: griff
    # Vollständig = 6 Griffe erkannt
    # Nur Griffe mit Confidence >= 0.80 zählen
    # --------------------------------------------------
    if tool_name == "Schraubendreher Set":
        MIN_GRIFF_CONF = 0.80

        griff_detections = [
            d for d in detections
            if d["class"] == "griff" and d["confidence"] >= MIN_GRIFF_CONF
        ]

        griff_count = len(griff_detections)
        best_conf = max([d["confidence"] for d in griff_detections], default=0.0)

        print("SCHRAUBENDREHER SET - alle Detections:", detections)
        print("SCHRAUBENDREHER SET - Griffe ab 0.80:", griff_count)

        if griff_count >= 6:
            return {
                "ok": True,
                "result_type": "success",
                "detected_class": "griff",
                "confidence": best_conf,
                "message": "Schraubendreherset vollständig. Kein Schraubendreher fehlt."
            }

        missing = 6 - griff_count

        return {
            "ok": False,
            "result_type": "incomplete",
            "detected_class": "griff",
            "confidence": best_conf,
            "message": de_missing_message(missing, "Schraubendreher", "Schraubendreher")
        }

    # --------------------------------------------------
    # Nadelfeilen Set
    # Klassen: vollstaendig, unvollstaendig
    # --------------------------------------------------
    if tool_name == "Nadelfeilen Set":
        has_unvoll = "unvollstaendig" in detected_classes
        has_voll = "vollstaendig" in detected_classes

        if has_unvoll:
            return {
                "ok": False,
                "result_type": "incomplete",
                "detected_class": "unvollstaendig",
                "confidence": max_confidence,
                "message": "Nadelfeilen Set ist unvollständig."
            }

        if has_voll:
            return {
                "ok": True,
                "result_type": "success",
                "detected_class": "vollstaendig",
                "confidence": max_confidence,
                "message": "Nadelfeilen Set ist vollständig."
            }

        return {
            "ok": False,
            "result_type": "incomplete",
            "detected_class": None,
            "confidence": 0.0,
            "message": "Nadelfeilen Set wurde nicht sicher erkannt. Bitte Bild erneut aufnehmen."
        }

    # --------------------------------------------------
    # Ringschlüssel Set
    # Klassen: ringschluessel_set_vollstaendig, ringschluessel_set_unvollstaendig
    # Abgabe-Regel:
    # - unvollständig erkannt => unvollständig
    # - vollständig erkannt, aber Confidence < 0.85 => unvollständig
    # - vollständig erkannt mit Confidence >= 0.85 => vollständig
    # --------------------------------------------------
    if tool_name == "Ringschlüssel Set":
        RING_VOLL_MIN_CONF = 0.85

        voll_detections = [
            d for d in detections
            if d["class"] == "ringschluessel_set_vollstaendig"
        ]

        unvoll_detections = [
            d for d in detections
            if d["class"] == "ringschluessel_set_unvollstaendig"
        ]

        best_voll_conf = max([d["confidence"] for d in voll_detections], default=0.0)
        best_unvoll_conf = max([d["confidence"] for d in unvoll_detections], default=0.0)

        print("RINGSCHLUESSEL SET - alle Detections:", detections)
        print("RINGSCHLUESSEL SET - best_voll_conf:", best_voll_conf)
        print("RINGSCHLUESSEL SET - best_unvoll_conf:", best_unvoll_conf)

        if len(unvoll_detections) > 0:
            return {
                "ok": False,
                "result_type": "incomplete",
                "detected_class": "ringschluessel_set_unvollstaendig",
                "confidence": best_unvoll_conf,
                "message": f"Ringschlüssel Set ist unvollständig. Unvollständig wurde mit {best_unvoll_conf:.2f} erkannt."
            }

        if best_voll_conf >= RING_VOLL_MIN_CONF:
            return {
                "ok": True,
                "result_type": "success",
                "detected_class": "ringschluessel_set_vollstaendig",
                "confidence": best_voll_conf,
                "message": f"Ringschlüssel Set ist vollständig. Vollständig wurde mit {best_voll_conf:.2f} erkannt."
            }

        return {
            "ok": False,
            "result_type": "incomplete",
            "detected_class": "ringschluessel_set_vollstaendig" if best_voll_conf > 0 else None,
            "confidence": best_voll_conf,
            "message": f"Ringschlüssel Set ist unvollständig, weil vollständig nur mit {best_voll_conf:.2f} erkannt wurde. Mindestwert ist 0.85."
        }

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------
    return {
        "ok": False,
        "result_type": "incomplete",
        "detected_class": None,
        "confidence": 0.0,
        "message": "Für dieses Werkzeug ist keine passende Vollständigkeitslogik definiert."
    }

RETURN_CHECK_RESULTS = {}

# BASE_URL = "http://127.0.0.1:5000"
BASE_URL = "https://subheader-nimbly-snowboard.ngrok-free.dev"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def de_missing_message(count, singular_text, plural_text, zero_text=None):
    """Deutsche Fehlermeldung mit korrektem Singular/Plural und fettem Objektwort."""
    try:
        count = int(count)
    except Exception:
        count = 0

    if count <= 0:
        if zero_text:
            return zero_text
        return f"Kein <strong>{singular_text}</strong> fehlt."

    if count == 1:
        return f"1 <strong>{singular_text}</strong> fehlt."

    return f"{count} <strong>{plural_text}</strong> fehlen."


def de_empty_slots_message(count):
    """Meldung für leere Slots mit Singular/Plural und fettem Wort Slot/Slots."""
    try:
        count = int(count)
    except Exception:
        count = 0

    if count <= 0:
        return "Kein <strong>Slot</strong> ist leer."

    if count == 1:
        return "1 <strong>Slot</strong> ist leer."

    return f"{count} <strong>Slots</strong> sind leer."



def _best_detection_for_class(detections, class_name):
    """Gibt die beste Detection für eine bestimmte Klasse zurück."""
    target = str(class_name).lower()

    matching = [
        d for d in detections
        if str(d.get("class", "")).lower() == target
    ]

    if not matching:
        return None

    return max(matching, key=lambda d: d.get("confidence", 0.0))


def _detect_identity_classes(image_path):
    """
    Führt die Identitäts-Erkennung aus und gibt alle erkannten Klassen zurück.
    Wird besonders für Multimeter + Messleitungen benötigt.
    """
    results = identity_model.predict(
        source=image_path,
        imgsz=IDENTITY_IMGSZ,
        conf=IDENTITY_CONF,
        verbose=False
    )

    if not results:
        return []

    return _extract_yolo_detections(results[0], identity_model)

def check_identity(image_path, expected_tool_name):
    expected_class = TOOL_NAME_TO_IDENTITY_CLASS.get(expected_tool_name)

    if expected_class is None:
        return {
            "ok": False,
            "result_type": "identity_failed",
            "expected_class": None,
            "detected_class": None,
            "confidence": 0.0,
            "message": "Keine passende Modellklasse gefunden."
        }

    detections = _detect_identity_classes(image_path)
    print("IDENTITY DETECTIONS:", detections)

    # ============================================================
    # Sonderfall Multimeter:
    # Multimeter bleibt eine einstufige Identitätsprüfung.
    # Im Rückgabebild müssen Multimeter UND Messleitungen erkannt werden.
    # Es gibt keine zusätzliche Vollständigkeitsstufe.
    # ============================================================
    if expected_tool_name == MULTIMETER_TOOL_NAME:
        multimeter_detection = _best_detection_for_class(detections, "multimeter")
        messleitungen_detection = _best_detection_for_class(detections, "messleitungen")

        has_multimeter = multimeter_detection is not None
        has_messleitungen = messleitungen_detection is not None

        best_confidence = max(
            [d["confidence"] for d in [multimeter_detection, messleitungen_detection] if d is not None],
            default=0.0
        )

        if has_multimeter and has_messleitungen:
            return {
                "ok": True,
                "result_type": "success",
                "expected_class": "multimeter + messleitungen",
                "detected_class": "multimeter + messleitungen",
                "confidence": best_confidence,
                "missing_items": [],
                "detections": detections,
                "message": "Multimeter und Messleitungen wurden erkannt."
            }

        missing_items = []

        if not has_multimeter:
            missing_items.append("Multimeter")

        if not has_messleitungen:
            missing_items.append("Messleitungen")

        if len(missing_items) == 2:
            message = "<strong>Multimeter</strong> und <strong>Messleitungen</strong> fehlen. Bitte beide Gegenstände im Bild aufnehmen."
        elif missing_items[0] == "Multimeter":
            message = "Das <strong>Multimeter</strong> fehlt. Bitte Multimeter und Messleitungen gemeinsam im Bild aufnehmen."
        else:
            message = "Die <strong>Messleitungen</strong> fehlen. Bitte Multimeter und Messleitungen gemeinsam im Bild aufnehmen."

        best_any_detection = max(detections, key=lambda d: d["confidence"], default=None)

        return {
            "ok": False,
            "result_type": "identity_failed",
            "expected_class": "multimeter + messleitungen",
            "detected_class": best_any_detection["class"] if best_any_detection else None,
            "confidence": best_any_detection["confidence"] if best_any_detection else 0.0,
            "missing_items": missing_items,
            "detections": detections,
            "message": message
        }

    # ============================================================
    # Normale Identitätsprüfung für alle anderen Werkzeuge
    # ============================================================
    best_expected_detection = None
    best_any_detection = None

    for detection in detections:
        cls_name = detection["class"]
        conf = detection["confidence"]

        if best_any_detection is None or conf > best_any_detection["confidence"]:
            best_any_detection = {
                "detected_class": cls_name,
                "confidence": conf
            }

        if cls_name == expected_class:
            if best_expected_detection is None or conf > best_expected_detection["confidence"]:
                best_expected_detection = {
                    "detected_class": cls_name,
                    "confidence": conf
                }

    if best_expected_detection is not None:
        return {
            "ok": True,
            "result_type": "success",
            "expected_class": expected_class,
            "detected_class": best_expected_detection["detected_class"],
            "confidence": best_expected_detection["confidence"],
            "detections": detections,
            "message": "Erwartetes Werkzeug wurde erkannt."
        }

    return {
        "ok": False,
        "result_type": "identity_failed",
        "expected_class": expected_class,
        "detected_class": best_any_detection["detected_class"] if best_any_detection else None,
        "confidence": best_any_detection["confidence"] if best_any_detection else 0.0,
        "detections": detections,
        "message": f"Das Werkzeug {expected_tool_name} wurde nicht erkannt."
    }

def generate_all_qrcodes():
    conn = get_db()
    cur = conn.cursor()

    tools = cur.execute("SELECT id, name FROM tools").fetchall()

    qr_folder = os.path.join(BASE_DIR, "static", "qrcodes")
    os.makedirs(qr_folder, exist_ok=True)

    for tool in tools:
        tool_id = tool["id"]
        tool_name = tool["name"]

        safe_name = tool_name.replace(" ", "_").replace("/", "_").lower()
        qr_url = f"{BASE_URL}/tool/{tool_id}"
        qr_path = os.path.join(qr_folder, f"{safe_name}_{tool_id}.png")

        if not os.path.exists(qr_path):
            img = qrcode.make(qr_url)
            img.save(qr_path)

    conn.close()


def generate_all_material_qrcodes():
    conn = get_db()
    cur = conn.cursor()

    materials = cur.execute("SELECT id, name, variant FROM materials").fetchall()

    qr_folder = os.path.join(BASE_DIR, "static", "qrcodes")
    os.makedirs(qr_folder, exist_ok=True)

    for material in materials:
        material_id = material["id"]
        material_name = material["name"]
        material_variant = material["variant"] or "standard"

        safe_name = material_name.replace(" ", "_").replace("/", "_").lower()
        safe_variant = material_variant.replace(" ", "_").replace("/", "_").lower()

        qr_url = f"{BASE_URL}/material/{material_id}"
        qr_path = os.path.join(qr_folder, f"material_{safe_name}_{safe_variant}_{material_id}.png")

        if not os.path.exists(qr_path):
            img = qrcode.make(qr_url)
            img.save(qr_path)

    conn.close()


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tools (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      category TEXT,
      location TEXT,
      total_quantity INTEGER NOT NULL CHECK (total_quantity >= 0),
      image_filename TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS loans (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      tool_id INTEGER NOT NULL,
      borrower TEXT NOT NULL,
      checkout_ts TEXT NOT NULL,
      return_ts TEXT,
      FOREIGN KEY (tool_id) REFERENCES tools(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS materials (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      variant TEXT,
      category TEXT,
      location TEXT,
      quantity INTEGER NOT NULL CHECK (quantity >= 0)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS material_transactions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      material_id INTEGER NOT NULL,
      user_name TEXT NOT NULL,
      amount INTEGER NOT NULL CHECK (amount > 0),
      action TEXT NOT NULL CHECK (action IN ('take', 'return')),
      timestamp TEXT NOT NULL,
      FOREIGN KEY (material_id) REFERENCES materials(id)
    );
    """)

    columns = [row["name"] for row in cur.execute("PRAGMA table_info(tools)").fetchall()]
    if "image_filename" not in columns:
        cur.execute("ALTER TABLE tools ADD COLUMN image_filename TEXT")

    existing = cur.execute("SELECT COUNT(*) as c FROM tools").fetchone()["c"]

    if existing == 0:
        cur.executemany("""
        INSERT INTO tools (name, category, location, total_quantity, image_filename)
        VALUES (?, ?, ?, ?, ?)
        """, DEMO_TOOLS)

    existing_materials = cur.execute("SELECT COUNT(*) as c FROM materials").fetchone()["c"]

    if existing_materials == 0:
        materials = [
            ("Schrauben", "M4", "Befestigungselement", "Schraubenschrank", 50),
            ("Schrauben", "M5", "Befestigungselement", "Schraubenschrank", 40),
            ("Schrauben", "M6", "Befestigungselement", "Schraubenschrank", 30),
            ("Muttern", "M4", "Befestigungselement", "Schraubenschrank", 60),
            ("Muttern", "M5", "Befestigungselement", "Schraubenschrank", 50),
            ("Muttern", "M6", "Befestigungselement", "Schraubenschrank", 40),
            ("Unterlegscheiben", "M4", "Befestigungselement", "Schraubenschrank", 100),
            ("Unterlegscheiben", "M5", "Befestigungselement", "Schraubenschrank", 80),
            ("Unterlegscheiben", "M6", "Befestigungselement", "Schraubenschrank", 60),
            ("Pneumatikanschlüsse", "Standard", "Pneumatik", "Pneumatikschrank", 25),
            ("Pneumatikschlauch", "6 mm", "Pneumatik", "Pneumatikschrank", 15)
        ]

        cur.executemany("""
            INSERT INTO materials (name, variant, category, location, quantity)
            VALUES (?, ?, ?, ?, ?)
        """, materials)

    conn.commit()
    conn.close()

    generate_all_qrcodes()
    #generate_all_material_qrcodes()

def admin_required():
    if not session.get("admin_logged_in"):
        return False
    return True

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("index"))

        error = "Benutzername oder Passwort ist falsch."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    if not admin_required():
        return redirect(url_for("login"))
    init_db()
    conn = get_db()
    cur = conn.cursor()

    search = request.args.get("search", "").strip()

    total_tools_count = cur.execute("""
        SELECT COUNT(*) AS c FROM tools
    """).fetchone()["c"]

    borrowed_tools_count = cur.execute("""
        SELECT COUNT(DISTINCT tool_id) AS c
        FROM loans
        WHERE return_ts IS NULL
    """).fetchone()["c"]

    total_material_positions = cur.execute("""
        SELECT COUNT(*) AS c FROM materials
    """).fetchone()["c"]

    total_material_stock = cur.execute("""
        SELECT COALESCE(SUM(quantity), 0) AS c FROM materials
    """).fetchone()["c"]

    tools = cur.execute("""
      SELECT
        t.id, t.name, t.category, t.location, t.total_quantity,
        COALESCE(o.open_count, 0) AS open_count,
        (t.total_quantity - COALESCE(o.open_count, 0)) AS available
      FROM tools t
      LEFT JOIN (
        SELECT tool_id, COUNT(*) AS open_count
        FROM loans
        WHERE return_ts IS NULL
        GROUP BY tool_id
      ) o ON o.tool_id = t.id
      WHERE
        t.name LIKE ?
        OR t.category LIKE ?
        OR t.location LIKE ?
      ORDER BY t.name ASC;
    """, (f"%{search}%", f"%{search}%", f"%{search}%")).fetchall()

    open_loans = cur.execute("""
      SELECT l.id AS loan_id, l.tool_id, l.borrower, l.checkout_ts
      FROM loans l
      WHERE l.return_ts IS NULL
      ORDER BY l.checkout_ts DESC;
    """).fetchall()

    loans_by_tool = {}
    for l in open_loans:
        loans_by_tool.setdefault(l["tool_id"], []).append(l)

    conn.close()
    return render_template(
        "index.html",
        tools=tools,
        loans_by_tool=loans_by_tool,
        search=search,
        total_tools_count=total_tools_count,
        borrowed_tools_count=borrowed_tools_count,
        total_material_positions=total_material_positions,
        total_material_stock=total_material_stock
    )


@app.route("/history")
def history():
    conn = get_db()
    cur = conn.cursor()

    history_entries = cur.execute("""
      SELECT
        l.id,
        t.name AS tool_name,
        l.borrower,
        l.checkout_ts,
        l.return_ts
      FROM loans l
      JOIN tools t ON l.tool_id = t.id
      ORDER BY l.checkout_ts DESC
    """).fetchall()

    conn.close()
    return render_template("history.html", history_entries=history_entries)

@app.route("/history/delete", methods=["POST"])
def delete_history():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM loans
        WHERE return_ts IS NOT NULL
    """)

    conn.commit()
    conn.close()

    return redirect(url_for("history"))

@app.route("/materials")
def materials():
    init_db()
    conn = get_db()
    cur = conn.cursor()

    search = request.args.get("search", "").strip()

    materials = cur.execute("""
        SELECT id, name, variant, category, location, quantity
        FROM materials
        WHERE
            name LIKE ?
            OR variant LIKE ?
            OR category LIKE ?
            OR location LIKE ?
        ORDER BY name ASC, variant ASC
    """, (f"%{search}%", f"%{search}%", f"%{search}%", f"%{search}%")).fetchall()

    conn.close()
    return render_template("materials.html", materials=materials, search=search)


@app.route("/admin/add_tool", methods=["GET", "POST"])
def admin_add_tool():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        category = request.form.get("category", "").strip()
        location = request.form.get("location", "").strip()
        total_quantity = request.form.get("total_quantity", "").strip()

        if not name:
            abort(400, "Bitte einen Werkzeugnamen eingeben.")

        if not total_quantity.isdigit():
            abort(400, "Bitte eine gültige Anzahl eingeben.")

        total_quantity = int(total_quantity)

        if total_quantity < 1:
            abort(400, "Die Anzahl muss mindestens 1 sein.")

        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO tools (name, category, location, total_quantity)
            VALUES (?, ?, ?, ?)
        """, (name, category, location, total_quantity))

        conn.commit()
        conn.close()

        return redirect(url_for("index"))

    return render_template("admin_add_tool.html")


@app.route("/admin/delete_tool/<int:tool_id>", methods=["POST"])
def delete_tool(tool_id):
    conn = get_db()
    cur = conn.cursor()

    open_loan = cur.execute("""
        SELECT COUNT(*) AS c
        FROM loans
        WHERE tool_id = ? AND return_ts IS NULL
    """, (tool_id,)).fetchone()["c"]

    if open_loan > 0:
        conn.close()
        abort(400, "Werkzeug kann nicht gelöscht werden, da noch offene Ausleihen existieren.")

    cur.execute("DELETE FROM loans WHERE tool_id = ?", (tool_id,))
    cur.execute("DELETE FROM tools WHERE id = ?", (tool_id,))

    conn.commit()
    conn.close()

    return redirect(url_for("index"))


@app.route("/tool/<int:tool_id>")
def tool_detail(tool_id):
    conn = get_db()
    cur = conn.cursor()

    tool = cur.execute("""
        SELECT
          t.id, t.name, t.category, t.location, t.total_quantity,
          COALESCE(o.open_count, 0) AS open_count,
          (t.total_quantity - COALESCE(o.open_count, 0)) AS available
        FROM tools t
        LEFT JOIN (
            SELECT tool_id, COUNT(*) AS open_count
            FROM loans
            WHERE return_ts IS NULL
            GROUP BY tool_id
        ) o ON o.tool_id = t.id
        WHERE t.id = ?
    """, (tool_id,)).fetchone()

    if tool is None:
        conn.close()
        abort(404, "Werkzeug nicht gefunden.")

    open_loans = cur.execute("""
        SELECT id AS loan_id, borrower, checkout_ts
        FROM loans
        WHERE tool_id = ? AND return_ts IS NULL
        ORDER BY checkout_ts DESC
    """, (tool_id,)).fetchall()

    loan_history = cur.execute("""
        SELECT borrower, checkout_ts, return_ts
        FROM loans
        WHERE tool_id = ?
        ORDER BY checkout_ts DESC
    """, (tool_id,)).fetchall()

    # QR-Code wird hier nicht mehr neu als tool_<id>.png erzeugt.
    # Die QR-Codes werden zentral über generate_all_qrcodes() erstellt.
    qr_url = f"{BASE_URL}/tool/{tool_id}"

    conn.close()

    return render_template(
        "tool_detail.html",
        tool=tool,
        open_loans=open_loans,
        loan_history=loan_history,
        qr_image=None,
        qr_url=qr_url
    )

@app.route("/admin/tool/<int:tool_id>")
def admin_tool_detail(tool_id):
    conn = get_db()
    cur = conn.cursor()

    tool = cur.execute("""
        SELECT
          t.id, t.name, t.category, t.location, t.total_quantity,
          COALESCE(o.open_count, 0) AS open_count,
          (t.total_quantity - COALESCE(o.open_count, 0)) AS available
        FROM tools t
        LEFT JOIN (
            SELECT tool_id, COUNT(*) AS open_count
            FROM loans
            WHERE return_ts IS NULL
            GROUP BY tool_id
        ) o ON o.tool_id = t.id
        WHERE t.id = ?
    """, (tool_id,)).fetchone()

    if tool is None:
        conn.close()
        abort(404, "Werkzeug nicht gefunden.")

    open_loans = cur.execute("""
        SELECT id AS loan_id, borrower, checkout_ts
        FROM loans
        WHERE tool_id = ? AND return_ts IS NULL
        ORDER BY checkout_ts DESC
    """, (tool_id,)).fetchall()

    loan_history = cur.execute("""
        SELECT borrower, checkout_ts, return_ts
        FROM loans
        WHERE tool_id = ?
        ORDER BY checkout_ts DESC
    """, (tool_id,)).fetchall()

    qr_url = f"{BASE_URL}/tool/{tool_id}"
    qr_image = f"qrcodes/tool_{tool_id}.png"

    conn.close()

    return render_template(
        "admin_tool_detail.html",
        tool=tool,
        open_loans=open_loans,
        loan_history=loan_history,
        qr_image=qr_image,
        qr_url=qr_url
    )

def _get_tool_by_name(cur, tool_name):
    return cur.execute(
        "SELECT id, name, total_quantity FROM tools WHERE name = ?",
        (tool_name,)
    ).fetchone()


def _get_open_count(cur, tool_id):
    return cur.execute(
        "SELECT COUNT(*) AS c FROM loans WHERE tool_id = ? AND return_ts IS NULL",
        (tool_id,)
    ).fetchone()["c"]


def _get_available_quantity(cur, tool_id, total_quantity):
    return int(total_quantity) - int(_get_open_count(cur, tool_id))


def _insert_loan(cur, tool_id, borrower, checkout_timestamp):
    cur.execute("""
      INSERT INTO loans (tool_id, borrower, checkout_ts, return_ts)
      VALUES (?, ?, ?, NULL)
    """, (tool_id, borrower, checkout_timestamp))


def _borrow_tool_with_multimeter_pair(cur, tool_id, borrower):
    """
    Normale Ausleihe, aber Sonderfall Multimeter:
    Wenn Multimeter ausgeliehen wird, werden Messleitungen automatisch
    als separate offene Ausleihe mit ausgeliehen.
    """
    tool = cur.execute(
        "SELECT id, name, total_quantity FROM tools WHERE id = ?",
        (tool_id,)
    ).fetchone()

    if tool is None:
        abort(404, "Tool nicht gefunden.")

    requested_available = _get_available_quantity(cur, tool["id"], tool["total_quantity"])

    if requested_available <= 0:
        abort(400, "Dieses Werkzeug ist aktuell nicht verfügbar.")

    checkout_timestamp = now_ts()

    if tool["name"] == MULTIMETER_TOOL_NAME:
        messleitungen_tool = _get_tool_by_name(cur, MESSLEITUNGEN_TOOL_NAME)

        if messleitungen_tool is None:
            abort(400, "Messleitungen wurden im Portal nicht gefunden.")

        messleitungen_available = _get_available_quantity(
            cur,
            messleitungen_tool["id"],
            messleitungen_tool["total_quantity"]
        )

        if messleitungen_available <= 0:
            abort(400, "Multimeter kann nicht ausgeliehen werden, weil die Messleitungen aktuell nicht verfügbar sind.")

        _insert_loan(cur, tool["id"], borrower, checkout_timestamp)
        _insert_loan(cur, messleitungen_tool["id"], borrower, checkout_timestamp)
        return tool

    _insert_loan(cur, tool["id"], borrower, checkout_timestamp)
    return tool


def _return_multimeter_partner_if_needed(cur, loan, return_timestamp):
    """
    Wenn eine erfolgreiche Multimeter-Rückgabe bestätigt wurde,
    wird die automatisch mit ausgeliehene offene Messleitungen-Ausleihe
    desselben Ausleihers ebenfalls zurückgegeben.
    """
    if loan["tool_name"] != MULTIMETER_TOOL_NAME:
        return None

    messleitungen_tool = _get_tool_by_name(cur, MESSLEITUNGEN_TOOL_NAME)

    if messleitungen_tool is None:
        return None

    partner_loan = cur.execute("""
        SELECT id
        FROM loans
        WHERE tool_id = ?
          AND borrower = ?
          AND return_ts IS NULL
          AND id != ?
        ORDER BY checkout_ts DESC, id DESC
        LIMIT 1
    """, (messleitungen_tool["id"], loan["borrower"], loan["loan_id"])).fetchone()

    if partner_loan is None:
        return None

    cur.execute("""
        UPDATE loans
        SET return_ts = ?
        WHERE id = ? AND return_ts IS NULL
    """, (return_timestamp, partner_loan["id"]))

    return partner_loan["id"]


@app.route("/borrow/<int:tool_id>", methods=["POST"])
def borrow(tool_id):
    borrower = request.form.get("borrower", "").strip()
    if not borrower:
        abort(400, "Bitte einen Namen eingeben.")

    conn = get_db()
    cur = conn.cursor()

    tool = _borrow_tool_with_multimeter_pair(cur, tool_id, borrower)

    conn.commit()
    conn.close()

    return redirect(url_for("tool_detail", tool_id=tool["id"]))


@app.route("/admin/borrow/<int:tool_id>", methods=["POST"])
def admin_borrow(tool_id):
    conn = get_db()
    cur = conn.cursor()

    tool = _borrow_tool_with_multimeter_pair(cur, tool_id, "Admin")

    conn.commit()
    conn.close()

    return redirect(url_for("index"))

@app.route("/return_prepare/<int:loan_id>")
def return_prepare(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            l.checkout_ts,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        abort(400, "Diese Ausleihe ist nicht mehr offen.")

    return render_template("return_prepare.html", loan=loan)


@app.route("/return_camera/<int:loan_id>")
def return_camera(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            l.checkout_ts,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        abort(400, "Diese Ausleihe ist nicht mehr offen.")

    return render_template("return_camera.html", loan=loan)

@app.route("/return_completeness_prepare/<int:loan_id>")
def return_completeness_prepare(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            l.checkout_ts,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        abort(400, "Diese Ausleihe ist nicht mehr offen.")

    return render_template("return_completeness_prepare.html", loan=loan)

@app.route("/return_completeness_camera/<int:loan_id>")
def return_completeness_camera(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            l.checkout_ts,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        abort(400, "Diese Ausleihe ist nicht mehr offen.")

    return render_template("return_completeness_camera.html", loan=loan)

@app.route("/save_return_photo/<int:loan_id>", methods=["POST"])
def save_return_photo(loan_id):
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Kein Bild empfangen."}), 400

    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        return jsonify({"success": False, "message": "Diese Ausleihe ist nicht mehr offen."}), 400

    image_data = data["image"]

    if "," in image_data:
        image_data = image_data.split(",")[1]

    filename = f"return_loan_{loan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(RETURN_IMAGE_FOLDER, filename)

    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_data))

    identity_result = check_identity(image_path, loan["tool_name"])

    identity_overlay_path = save_identity_overlay_debug_image(image_path, loan["tool_name"])

    if identity_result["ok"] and loan["tool_name"] in COMPLETENESS_REQUIRED_TOOLS:
        result_type = "needs_completeness"
        result_url = url_for("return_completeness_prepare", loan_id=loan_id)
    else:
        result_type = identity_result["result_type"]
        result_url = url_for("return_result", loan_id=loan_id)

    RETURN_CHECK_RESULTS[loan_id] = {
        "image_path": image_path,
        "identity_overlay_path": identity_overlay_path,
        "identity_result": identity_result,
        "result_type": result_type
    }

    return jsonify({
        "success": True,
        "message": "Bild wurde gespeichert und geprüft.",
        "image_path": image_path,
        "image_url": url_for("static", filename=f"return_images/{filename}"),
        "identity_overlay_path": identity_overlay_path,
        "identity_overlay_url": url_for("static", filename=f"return_identity_overlay/{os.path.basename(identity_overlay_path)}") if identity_overlay_path else None,
        "result_type": result_type,
        "result_url": result_url
    })


def aggregate_three_completeness_results(results):
    """
    Kombiniert die Ergebnisse von 3 Vollständigkeitsprüfungen.

    Für das Steckschlüssel Set werden drei automatisch aufgenommene Bilder
    einzeln geprüft. Danach wird eine Mehrheitsentscheidung gebildet.

    - result_type wird per Mehrheit bestimmt.
    - Falls Slotlisten vorhanden sind, gilt ein Slot als leer/fehlend,
      wenn er in mindestens 2 von 3 Bildern leer/fehlend erkannt wurde.
    """

    from collections import Counter

    if not results:
        return {
            "ok": False,
            "result_type": "incomplete",
            "message": "Keine Vollständigkeitsergebnisse vorhanden."
        }

    result_types = [
        result.get("result_type", "incomplete")
        for result in results
    ]

    majority_result_type = Counter(result_types).most_common(1)[0][0]

    base_result = None

    for result in results:
        if result.get("result_type") == majority_result_type:
            base_result = dict(result)
            break

    if base_result is None:
        base_result = dict(results[-1])

    base_result["result_type"] = majority_result_type
    base_result["used_images"] = len(results)
    base_result["aggregation"] = "single_image" if len(results) == 1 else "majority_vote_3_images"

    possible_slot_keys = [
        "empty_slots",
        "missing_slots",
        "fehlende_slots",
        "leere_slots"
    ]

    for key in possible_slot_keys:
        if any(key in result for result in results):
            slot_counter = Counter()

            for result in results:
                slots = result.get(key, [])

                if slots is None:
                    slots = []

                for slot in slots:
                    slot_counter[slot] += 1

            # Bei 3 Bildern gilt Mehrheitsentscheidung: mindestens 2 von 3.
            # Bei manuellem Einzelbild gilt der eine erkannte Zustand direkt.
            required_missing_votes = 2 if len(results) >= 3 else 1

            final_slots = sorted([
                slot for slot, count in slot_counter.items()
                if count >= required_missing_votes
            ])

            base_result[key] = final_slots

    # Spezialfall Slotanalyse: Die Slot-Mehrheit ist wichtiger als die reine
    # Mehrheit über result_type. Ein Slot gilt nur dann als fehlend, wenn er
    # in mindestens 2 von 3 Bildern fehlt.
    if any(result.get("slot_analysis") for result in results):
        final_missing_slots = base_result.get("missing_slots", []) or []
        final_empty_slots = base_result.get("empty_slots", []) or final_missing_slots

        base_result["slot_analysis"] = True
        base_result["missing_slots"] = final_missing_slots
        base_result["empty_slots"] = final_empty_slots

        all_slot_ids = sorted({
            slot_result.get("slot")
            for result in results
            for slot_result in result.get("slot_results", [])
            if slot_result.get("slot") is not None
        })

        base_result["slot_count"] = len(all_slot_ids)
        base_result["occupied_slots"] = [
            slot_id for slot_id in all_slot_ids
            if slot_id not in final_missing_slots
        ]

        tool_name = base_result.get("tool_name") or ""
        slot_unit = base_result.get("slot_unit") or "slot"

        if len(final_missing_slots) == 0:
            base_result["ok"] = True
            base_result["result_type"] = "success"

            if tool_name == "Ringschlüssel Set" or slot_unit == "schluessel":
                base_result["message"] = "Ringschlüsselset vollständig. Kein Schlüssel fehlt."
            elif tool_name == "Steckschlüssel Set" or slot_unit == "slot":
                base_result["message"] = "Steckschlüssel-Set vollständig. Kein Slot ist leer."
            else:
                base_result["message"] = "Werkzeug vollständig. Keine fehlende Position bestätigt."
        else:
            base_result["ok"] = False
            base_result["result_type"] = "incomplete"

            if tool_name == "Ringschlüssel Set" or slot_unit == "schluessel":
                base_result["message"] = de_missing_message(len(final_missing_slots), "Schlüssel", "Schlüssel")
            elif tool_name == "Steckschlüssel Set" or slot_unit == "slot":
                base_result["message"] = de_empty_slots_message(len(final_missing_slots))
            else:
                base_result["message"] = de_missing_message(len(final_missing_slots), "Position", "Positionen")

    base_result["individual_results"] = results

    if base_result.get("result_type") == "success":
        base_result["ok"] = True
        if "message" not in base_result or not base_result["message"]:
            base_result["message"] = "Werkzeug ist in der Mehrheitsentscheidung vollständig."
    else:
        base_result["ok"] = False
        if "message" not in base_result or not base_result["message"]:
            base_result["message"] = "Werkzeug ist in der Mehrheitsentscheidung unvollständig."

    return base_result


def analyze_guided_slot_images_background(loan_id, tool_name, image_paths):
    """
    Wird erst NACH dem dritten automatisch aufgenommenen Bild gestartet.
    Die ersten zwei Bilder werden nur gespeichert. Keine Slotanalyse davor.
    """
    try:
        print(f"GUIDED ANALYSE START: loan={loan_id}, tool={tool_name}, bilder={len(image_paths)}")

        overlay_debug_paths = []

        for index, path in enumerate(image_paths, start=1):
            overlay_path = None

            if tool_name == "Steckschlüssel Set":
                overlay_path = save_steckschluessel_overlay_debug_image(
                    image_path=path,
                    loan_id=loan_id,
                    photo_number=index
                )
            elif tool_name == "Ringschlüssel Set":
                overlay_path = save_ringschluessel_overlay_debug_image(
                    image_path=path,
                    loan_id=loan_id,
                    photo_number=index
                )

            if overlay_path is not None:
                overlay_debug_paths.append(overlay_path)

        individual_results = []

        for path in image_paths:
            result = check_completeness(path, tool_name)
            individual_results.append(result)

        completeness_result = aggregate_three_completeness_results(individual_results)

        RETURN_CHECK_RESULTS[loan_id] = {
            **RETURN_CHECK_RESULTS.get(loan_id, {}),
            "guided_slot_auto_images": image_paths,
            "guided_slot_overlay_debug_images": overlay_debug_paths,
            "completeness_image_path": image_paths[-1],
            "completeness_result": completeness_result,
            "result_type": completeness_result["result_type"]
        }

        print(f"GUIDED ANALYSE FERTIG: loan={loan_id}, result={completeness_result.get('result_type')}")

    except Exception as error:
        print("GUIDED ANALYSE FEHLER:", error)
        RETURN_CHECK_RESULTS[loan_id] = {
            **RETURN_CHECK_RESULTS.get(loan_id, {}),
            "result_type": "incomplete",
            "completeness_result": {
                "ok": False,
                "result_type": "incomplete",
                "message": "Die Vollständigkeitsprüfung konnte nicht durchgeführt werden. Bitte Verantwortlichen kontaktieren."
            }
        }

@app.route("/save_completeness_photo/<int:loan_id>", methods=["POST"])
def save_completeness_photo(loan_id):
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Kein Bild empfangen."}), 400

    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        return jsonify({"success": False, "message": "Diese Ausleihe ist nicht mehr offen."}), 400

    image_data = data["image"]

    if "," in image_data:
        image_data = image_data.split(",")[1]

    auto_capture = bool(data.get("auto_capture", False))
    manual_single = bool(data.get("manual_single", False))
    photo_number = int(data.get("photo_number", 1))
    photo_count = int(data.get("photo_count", 1))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"completeness_loan_{loan_id}_photo_{photo_number}_{timestamp}.jpg"
    image_path = os.path.join(RETURN_IMAGE_FOLDER, filename)

    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_data))

    overlay_debug_path = None

    print("COMPLETENESS TOOL:", loan["tool_name"])
    print("COMPLETENESS BILD GESPEICHERT:", image_path)

    # ============================================================
    # Manueller Schnellmodus für geführte Slotanalyse:
    # Der Benutzer klickt auf die rote/grüne Statusfläche.
    # Dann wird genau EIN Bild gespeichert und sofort auf der Warte-Seite
    # im Hintergrund analysiert. Die normale automatische 3-Bild-Logik
    # bleibt unverändert, wenn nicht geklickt wird.
    # ============================================================
    if loan["tool_name"] in ["Steckschlüssel Set", "Ringschlüssel Set"] and manual_single:
        RETURN_CHECK_RESULTS[loan_id] = {
            **RETURN_CHECK_RESULTS.get(loan_id, {}),
            "guided_slot_auto_images": [image_path],
            "guided_slot_overlay_debug_images": [],
            "completeness_image_path": image_path,
            "result_type": "pending",
            "completeness_result": {
                "ok": False,
                "result_type": "pending",
                "message": "Das Bild wird gerade analysiert."
            }
        }

        threading.Thread(
            target=analyze_guided_slot_images_background,
            args=(loan_id, loan["tool_name"], [image_path]),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "message": "Ein Vollständigkeitsbild wurde gespeichert. Die Analyse startet jetzt.",
            "pending": True,
            "manual_single": True,
            "saved_count": 1,
            "image_path": image_path,
            "image_url": url_for("static", filename=f"return_images/{filename}"),
            "result_type": "pending",
            "result_url": url_for("return_completeness_wait", loan_id=loan_id)
        })

    # ============================================================
    # Geführte Slotanalyse: Erst ALLE 3 Bilder speichern,
    # danach im Hintergrund gemeinsam analysieren.
    # Keine Analyse nach Bild 1 oder Bild 2.
    # ============================================================
    if loan["tool_name"] in ["Steckschlüssel Set", "Ringschlüssel Set"] and auto_capture and photo_count == 3:
        previous_data = RETURN_CHECK_RESULTS.get(loan_id, {})
        guided_image_paths = previous_data.get("guided_slot_auto_images", [])
        guided_image_paths.append(image_path)

        # pro Rückgabe nur die letzten 3 geführten Bilder verwenden
        guided_image_paths = guided_image_paths[-3:]

        RETURN_CHECK_RESULTS[loan_id] = {
            **previous_data,
            "guided_slot_auto_images": guided_image_paths,
            "guided_slot_overlay_debug_images": [],
            "completeness_image_path": image_path,
            "result_type": "pending",
            "completeness_result": {
                "ok": False,
                "result_type": "pending",
                "message": f"Bild {len(guided_image_paths)}/3 gespeichert."
            }
        }

        if len(guided_image_paths) < 3:
            return jsonify({
                "success": True,
                "message": f"Bild {len(guided_image_paths)}/3 gespeichert.",
                "pending": True,
                "saved_count": len(guided_image_paths),
                "image_path": image_path,
                "image_url": url_for("static", filename=f"return_images/{filename}"),
                "result_type": "pending",
                "result_url": None
            })

        # Nach Bild 3: sofort zur Warte-Seite, Analyse läuft im Hintergrund.
        analysis_paths = list(guided_image_paths)
        threading.Thread(
            target=analyze_guided_slot_images_background,
            args=(loan_id, loan["tool_name"], analysis_paths),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "message": "3 Vollständigkeitsbilder wurden gespeichert. Die Analyse startet jetzt.",
            "pending": True,
            "saved_count": 3,
            "image_path": image_path,
            "image_url": url_for("static", filename=f"return_images/{filename}"),
            "result_type": "pending",
            "result_url": url_for("return_completeness_wait", loan_id=loan_id)
        })

    if loan["tool_name"] == "Steckschlüssel Set":
        print("STECKSCHLÜSSEL OVERLAY-DEBUG WIRD ERSTELLT...")
        overlay_debug_path = save_steckschluessel_overlay_debug_image(
            image_path=image_path,
            loan_id=loan_id,
            photo_number=photo_number
        )
        print("OVERLAY DEBUG PATH:", overlay_debug_path)

    if loan["tool_name"] == "Ringschlüssel Set":
        print("RINGSCHLÜSSEL OVERLAY-DEBUG WIRD ERSTELLT...")
        overlay_debug_path = save_ringschluessel_overlay_debug_image(
            image_path=image_path,
            loan_id=loan_id,
            photo_number=photo_number
        )
        print("RING OVERLAY DEBUG PATH:", overlay_debug_path)

    completeness_yolo_overlay_path = save_completeness_overlay_debug_image(
        image_path=image_path,
        tool_name=loan["tool_name"]
    )

    # ============================================================
    # Sonderfall: geführte Slotanalyse mit 3 automatischen Bildern
    # Steckschlüssel Set und Ringschlüssel Set
    # ============================================================

    if loan["tool_name"] in ["Steckschlüssel Set", "Ringschlüssel Set"] and auto_capture and photo_count == 3:
        previous_data = RETURN_CHECK_RESULTS.get(loan_id, {})

        guided_image_paths = previous_data.get("guided_slot_auto_images", [])
        guided_image_paths.append(image_path)

        guided_overlay_debug_paths = previous_data.get("guided_slot_overlay_debug_images", [])

        if overlay_debug_path is not None:
            guided_overlay_debug_paths.append(overlay_debug_path)

        RETURN_CHECK_RESULTS[loan_id] = {
            **previous_data,
            "guided_slot_auto_images": guided_image_paths,
            "guided_slot_overlay_debug_images": guided_overlay_debug_paths,
            "completeness_image_path": image_path,
            "result_type": "pending",
            "completeness_result": {
                "ok": False,
                "result_type": "pending",
                "message": f"Bild {len(guided_image_paths)}/3 gespeichert."
            }
        }

        # Bild 1 oder Bild 2: nur speichern, noch nicht final auswerten
        if len(guided_image_paths) < 3:
            return jsonify({
                "success": True,
                "message": f"Bild {len(guided_image_paths)}/3 gespeichert.",
                "pending": True,
                "saved_count": len(guided_image_paths),
                "image_path": image_path,
                "image_url": url_for("static", filename=f"return_images/{filename}"),
                "overlay_debug_path": overlay_debug_path,
                "completeness_yolo_overlay_path": completeness_yolo_overlay_path,
                "result_type": "pending",
                "result_url": None
            })

        # Ab Bild 3: nur die letzten 3 Bilder gemeinsam auswerten
        guided_image_paths = guided_image_paths[-3:]
        guided_overlay_debug_paths = guided_overlay_debug_paths[-3:]

        individual_results = []

        for path in guided_image_paths:
            result = check_completeness(path, loan["tool_name"])
            individual_results.append(result)

        completeness_result = aggregate_three_completeness_results(individual_results)

        RETURN_CHECK_RESULTS[loan_id] = {
            **RETURN_CHECK_RESULTS.get(loan_id, {}),
            "guided_slot_auto_images": guided_image_paths,
            "guided_slot_overlay_debug_images": guided_overlay_debug_paths,
            "completeness_image_path": guided_image_paths[-1],
            "completeness_result": completeness_result,
            "result_type": completeness_result["result_type"]
        }

        return jsonify({
            "success": True,
            "message": "3 Vollständigkeitsbilder wurden gespeichert und gemeinsam geprüft.",
            "pending": False,
            "saved_count": 3,
            "image_path": guided_image_paths[-1],
            "image_url": url_for(
                "static",
                filename=f"return_images/{os.path.basename(guided_image_paths[-1])}"
            ),
            "overlay_debug_path": overlay_debug_path,
            "overlay_debug_paths": guided_overlay_debug_paths,
            "completeness_yolo_overlay_path": completeness_yolo_overlay_path,
            "result_type": completeness_result["result_type"],
            "result_url": url_for("return_completeness_wait", loan_id=loan_id)
        })

    # ============================================================
    # Standardfall: alle anderen Werkzeuge wie bisher
    # ============================================================

    completeness_result = check_completeness(image_path, loan["tool_name"])

    RETURN_CHECK_RESULTS[loan_id] = {
        **RETURN_CHECK_RESULTS.get(loan_id, {}),
        "completeness_image_path": image_path,
        "completeness_overlay_debug_image": overlay_debug_path,
        "completeness_yolo_overlay_path": completeness_yolo_overlay_path,
        "completeness_result": completeness_result,
        "result_type": completeness_result["result_type"]
    }

    return jsonify({
        "success": True,
        "message": "Vollständigkeitsbild wurde gespeichert und geprüft.",
        "image_path": image_path,
        "image_url": url_for("static", filename=f"return_images/{filename}"),
        "overlay_debug_path": overlay_debug_path,
        "completeness_yolo_overlay_path": completeness_yolo_overlay_path,
        "completeness_yolo_overlay_url": url_for("static", filename=f"return_completeness_overlay/{os.path.basename(completeness_yolo_overlay_path)}") if completeness_yolo_overlay_path else None,
        "result_type": completeness_result["result_type"],
        "result_url": url_for("return_result", loan_id=loan_id)
    })

@app.route("/return_completeness_wait/<int:loan_id>")
def return_completeness_wait(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ?
    """, (loan_id,)).fetchone()

    conn.close()

    if loan is None:
        abort(404, "Ausleihe nicht gefunden.")

    return render_template("return_completeness_wait.html", loan=loan)


@app.route("/return_completeness_status/<int:loan_id>")
def return_completeness_status(loan_id):
    check_data = RETURN_CHECK_RESULTS.get(loan_id)

    if check_data is None:
        return jsonify({
            "ready": False,
            "message": "Die Bilder werden gerade analysiert."
        })

    result_type = check_data.get("result_type", "pending")

    if result_type == "pending":
        return jsonify({
            "ready": False,
            "message": "Die Bilder werden gerade analysiert."
        })

    return jsonify({
        "ready": True,
        "result_type": result_type,
        "result_url": url_for("return_result", loan_id=loan_id)
    })


@app.route("/return_result/<int:loan_id>")
def return_result(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ?
    """, (loan_id,)).fetchone()

    if loan is None:
        conn.close()
        abort(404, "Ausleihe nicht gefunden.")

    check_data = RETURN_CHECK_RESULTS.get(loan_id)

    if check_data is None:
        conn.close()
        return render_template(
            "return_result.html",
            loan=loan,
            result_type="identity_failed",
            result_message="Keine gespeicherten Prüfdaten gefunden.",
            check_data=None,
            returned_partner_loan_id=None
        )

    result_type = check_data["result_type"]

    result_message = None

    if check_data.get("completeness_result"):
        result_message = check_data["completeness_result"].get("message")
    elif check_data.get("identity_result"):
        result_message = check_data["identity_result"].get("message")

    returned_partner_loan_id = None

    if result_type == "success":
        return_timestamp = now_ts()

        cur.execute("""
            UPDATE loans SET return_ts = ?
            WHERE id = ? AND return_ts IS NULL
        """, (return_timestamp, loan_id))

        returned_partner_loan_id = _return_multimeter_partner_if_needed(cur, loan, return_timestamp)

        conn.commit()

    conn.close()

    return render_template(
        "return_result.html",
        loan=loan,
        result_type=result_type,
        result_message=result_message,
        check_data=check_data,
        returned_partner_loan_id=returned_partner_loan_id
    )

@app.route("/return/<int:loan_id>", methods=["POST"])
def return_loan(loan_id):
    conn = get_db()
    cur = conn.cursor()

    loan = cur.execute("""
        SELECT
            l.id AS loan_id,
            l.borrower,
            t.id AS tool_id,
            t.name AS tool_name
        FROM loans l
        JOIN tools t ON l.tool_id = t.id
        WHERE l.id = ? AND l.return_ts IS NULL
    """, (loan_id,)).fetchone()

    if loan is None:
        conn.close()
        abort(400, "Diese Ausleihe ist nicht (mehr) offen.")

    return_timestamp = now_ts()

    cur.execute("""
      UPDATE loans SET return_ts = ? WHERE id = ? AND return_ts IS NULL
    """, (return_timestamp, loan_id))

    _return_multimeter_partner_if_needed(cur, loan, return_timestamp)

    conn.commit()
    conn.close()
    return redirect(url_for("index"))

@app.route("/materials/take/<int:material_id>", methods=["POST"])
def take_material(material_id):
    user_name = request.form.get("user_name")
    amount = int(request.form.get("amount"))

    conn = get_db()
    cur = conn.cursor()

    material = cur.execute(
        "SELECT quantity FROM materials WHERE id = ?", (material_id,)
    ).fetchone()

    if material["quantity"] < amount:
        conn.close()
        abort(400, "Nicht genug Bestand")

    cur.execute(
        "UPDATE materials SET quantity = quantity - ? WHERE id = ?",
        (amount, material_id)
    )

    cur.execute("""
        INSERT INTO material_transactions
        (material_id, user_name, amount, action, timestamp)
        VALUES (?, ?, ?, 'take', ?)
    """, (material_id, user_name, amount, now_ts()))

    conn.commit()
    conn.close()

    return redirect(url_for("materials"))


@app.route("/materials/return/<int:material_id>", methods=["POST"])
def return_material(material_id):
    user_name = request.form.get("user_name")
    amount = int(request.form.get("amount"))

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "UPDATE materials SET quantity = quantity + ? WHERE id = ?",
        (amount, material_id)
    )

    cur.execute("""
        INSERT INTO material_transactions
        (material_id, user_name, amount, action, timestamp)
        VALUES (?, ?, ?, 'return', ?)
    """, (material_id, user_name, amount, now_ts()))

    conn.commit()
    conn.close()

    return redirect(url_for("materials"))


@app.route("/materials/history")
def materials_history():
    conn = get_db()
    cur = conn.cursor()

    entries = cur.execute("""
        SELECT
            mt.id,
            m.name,
            m.variant,
            mt.user_name,
            mt.amount,
            mt.action,
            mt.timestamp
        FROM material_transactions mt
        JOIN materials m ON mt.material_id = m.id
        ORDER BY mt.timestamp DESC
    """).fetchall()

    conn.close()
    return render_template("materials_history.html", entries=entries)


@app.route("/material/<int:material_id>")
def material_detail(material_id):
    conn = get_db()
    cur = conn.cursor()

    material = cur.execute("""
        SELECT id, name, variant, category, location, quantity
        FROM materials
        WHERE id = ?
    """, (material_id,)).fetchone()

    if material is None:
        conn.close()
        abort(404, "Material nicht gefunden.")

    transactions = cur.execute("""
        SELECT user_name, amount, action, timestamp
        FROM material_transactions
        WHERE material_id = ?
        ORDER BY timestamp DESC
    """, (material_id,)).fetchall()

    material_name = material["name"]
    material_variant = material["variant"] or "standard"

    safe_name = material_name.replace(" ", "_").replace("/", "_").lower()
    safe_variant = material_variant.replace(" ", "_").replace("/", "_").lower()

    qr_url = f"{BASE_URL}/material/{material_id}"
    qr_image = f"qrcodes/material_{safe_name}_{safe_variant}_{material_id}.png"

    conn.close()

    return render_template(
        "material_detail.html",
        material=material,
        transactions=transactions,
        qr_url=qr_url,
        qr_image=qr_image
    )

# Trainingsseiten wurden deaktiviert, damit der Ordner Voll_Werkzeuge
# nicht mehr durch das Portal erstellt oder benutzt wird.
@app.route("/training_voll")
def training_voll_master():
    abort(404, "Training wurde deaktiviert.")


@app.route("/training_voll/<tool_key>")
def training_voll_camera(tool_key):
    abort(404, "Training wurde deaktiviert.")


@app.route("/save_training_voll_photo/<tool_key>", methods=["POST"])
def save_training_voll_photo(tool_key):
    return jsonify({
        "success": False,
        "message": "Training wurde deaktiviert. Der Ordner Voll_Werkzeuge wird nicht mehr erstellt."
    }), 404

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)