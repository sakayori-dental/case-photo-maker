"""
歯科症例写真 自動合成アプリ v3.6
OpenCV自動処理 → 手動微調整 → 3〜5パネル合成
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import base64
import json
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps, ExifTags

# ============================================================
# 定数
# ============================================================
APP_DIR = Path(__file__).parent
REFERENCE_DIR = APP_DIR / "reference"

OUTPUT_PRESETS = {
    "A4横向き (3508×2480)": (3508, 2480),
    "A4縦向き (2480×3508)": (2480, 3508),
    "1920×1080 (フルHD)": (1920, 1080),
    "1600×900": (1600, 900),
    "1280×720 (HD)": (1280, 720),
}
MIN_PANELS = 3
MAX_PANELS = 5
BG_COLOR = "#0f0f0f"

PHOTO_LABELS_DEFAULT = [
    "① 術前・頬側観",
    "② 術前・咬合面観",
    "③ 術中・窩洞形成後",
    "④ 術中・充填中",
    "⑤ 術後・頬側観",
]


FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴ ProN W6.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
]


# ============================================================
# ユーティリティ
# ============================================================
def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """利用可能な日本語フォントを探す"""
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    st.warning("⚠️ 日本語フォントが見つかりません。デフォルトフォントを使用します。")
    return ImageFont.load_default()


def fix_orientation(img: Image.Image) -> Image.Image:
    """EXIF Orientation タグで画像を正しい向きに補正"""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == "Orientation":
                orientation_key = key
                break
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass
    return img


# ============================================================
# OpenCV 歯検出パイプライン
# ============================================================
def _pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL Image → OpenCV BGR numpy array"""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _downscale_pil(img: Image.Image, max_dim: int = 600) -> Image.Image:
    """処理高速化のためダウンスケール"""
    scale = max_dim / max(img.size)
    if scale < 1:
        return img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.BICUBIC,
        )
    return img


def _detect_rubber_dam(img_hsv: np.ndarray) -> bool:
    """ラバーダム（青）が存在するか判定"""
    blue_mask = cv2.inRange(img_hsv, (90, 80, 40), (130, 255, 255))
    h, w = img_hsv.shape[:2]
    return cv2.countNonZero(blue_mask) > h * w * 0.05


def _crop_rubber_dam(img_hsv: np.ndarray) -> dict:
    """ラバーダム写真: 青いダムの「穴」（歯が見える領域）を検出してcrop算出"""
    h, w = img_hsv.shape[:2]
    blue_mask = cv2.inRange(img_hsv, (90, 80, 40), (130, 255, 255))

    # 青領域をモルフォロジーで閉じて穴を検出
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blue_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=8)

    # 青の外輪郭を塗りつぶし → 実際の青を引く = 穴
    contours, _ = cv2.findContours(
        blue_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    blue_filled = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(blue_filled, contours, -1, 255, -1)

    hole_mask = cv2.bitwise_and(blue_filled, cv2.bitwise_not(blue_mask))
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    ys, xs = np.where(hole_mask > 0)
    if len(xs) > 50:
        cx = float(np.mean(xs)) / w
        cy = float(np.mean(ys)) / h
        x_lo, x_hi = np.percentile(xs, [5, 95])
        y_lo, y_hi = np.percentile(ys, [5, 95])
        # 縦長パネルに合わせたcropサイズ（高さ重視）
        detected_w = float(x_hi - x_lo) / w
        detected_h = float(y_hi - y_lo) / h
        # ミラー写真と歯サイズを統一するためcrop幅をミラーと同等(0.35)に制限
        crop_w = min(max(detected_w * 1.1, 0.28), 0.35)
        crop_h = min(max(detected_h * 1.5, 0.55), 0.80)
    else:
        # フォールバック: 青領域の重心を使用
        b_ys, b_xs = np.where(blue_mask > 0)
        cx = float(np.mean(b_xs)) / w if len(b_xs) > 0 else 0.5
        cy = float(np.mean(b_ys)) / h if len(b_ys) > 0 else 0.5
        crop_w, crop_h = 0.35, 0.75

    # 穴の重心はクランプ・フロス側に引かれるため、歯冠中心に向けて補正
    cx -= 0.05
    # クランプ・フロスを上端に押し出すためYを下方向へ補正
    cy += 0.03

    return {
        "x0": max(0.0, cx - crop_w / 2),
        "y0": max(0.0, cy - crop_h / 2),
        "x1": min(1.0, cx + crop_w / 2),
        "y1": min(1.0, cy + crop_h / 2),
    }


def _crop_mirror(img_hsv: np.ndarray, img_gray: np.ndarray) -> dict:
    """ミラー写真: エッジ密度から歯列中心を推定してcrop算出"""
    h, w = img_hsv.shape[:2]

    # 画像端（リトラクター・ガーゼ）を除外
    border_x = int(w * 0.10)
    border_y = int(h * 0.10)
    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[border_y : h - border_y, border_x : w - border_x] = 255

    # エッジ検出（暗部・青領域除外）
    edges = cv2.Canny(img_gray, 40, 120)
    edges = cv2.bitwise_and(edges, center_mask)
    dark = (img_hsv[:, :, 2] < 50).astype(np.uint8) * 255
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(dark))
    # 青領域（ラバーダム残留）を除外
    blue_in_mirror = cv2.inRange(img_hsv, (90, 60, 40), (130, 255, 255))
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(blue_in_mirror))

    # 密度ヒートマップ
    ksize = max(h, w) // 6
    if ksize % 2 == 0:
        ksize += 1
    density = cv2.GaussianBlur(edges.astype(np.float32), (ksize, ksize), 0)

    if density.max() > 0:
        density_norm = density / density.max()
        x_profile = density_norm.mean(axis=0)
        y_profile = density_norm.mean(axis=1)
        x_coords = np.arange(w, dtype=np.float64)
        y_coords = np.arange(h, dtype=np.float64)
        cx = float(np.average(x_coords, weights=x_profile + 1e-10)) / w
        cy = float(np.average(y_coords, weights=y_profile + 1e-10)) / h
        # 回転後、歯は中心よりやや左寄り・下寄り（大臼歯側）になる傾向を補正
        cx -= 0.05
        cy += 0.06
    else:
        cx, cy = 0.38, 0.55

    # 縦長パネル(384×1080)に合わせた縦長crop（高さ重視で歯3-4本表示）
    default_w, default_h = 0.35, 0.70
    return {
        "x0": max(0.0, cx - default_w / 2),
        "y0": max(0.0, cy - default_h / 2),
        "x1": min(1.0, cx + default_w / 2),
        "y1": min(1.0, cy + default_h / 2),
    }


def cv_detect_crop(img_pil: Image.Image, rotation_deg: float = -90.0) -> dict:
    """
    OpenCVベースの歯検出パイプライン。
    - ラバーダム写真: 青いダムの穴を検出 → crop
    - ミラー写真: エッジ密度から歯列中心を推定 → crop

    Returns: {"rotation_cw_deg": float, "crop": {x0, y0, x1, y1}, "method": str}
    """
    img_fixed = fix_orientation(img_pil)
    if abs(rotation_deg) > 0.1:
        img_rotated = img_fixed.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)
    else:
        img_rotated = img_fixed

    img_small = _downscale_pil(img_rotated, max_dim=600)
    img_bgr = _pil_to_cv(img_small)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if _detect_rubber_dam(img_hsv):
        crop = _crop_rubber_dam(img_hsv)
        method = "rubber_dam"
    else:
        crop = _crop_mirror(img_hsv, img_gray)
        method = "mirror"

    # 写真タイプ別の自動ズームブースト
    # mirror: 歯冠4本がパネルいっぱいになる強めのズーム
    # rubber_dam: 歯+青ダム少し見える程度の控えめズーム
    auto_zoom = 2.0 if method == "mirror" else 1.6

    return {"rotation_cw_deg": rotation_deg, "crop": crop, "method": method, "zoom_boost": auto_zoom}


def cv_create_debug_image(img_pil: Image.Image, rotation_deg: float) -> Image.Image:
    """デバッグ用: 検出領域をオーバーレイした画像を返す"""
    img_fixed = fix_orientation(img_pil)
    if abs(rotation_deg) > 0.1:
        img_rotated = img_fixed.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)
    else:
        img_rotated = img_fixed

    img_small = _downscale_pil(img_rotated, max_dim=600)
    img_bgr = _pil_to_cv(img_small)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]

    result = img_bgr.copy()

    if _detect_rubber_dam(img_hsv):
        # ラバーダム: 青を水色、穴を緑でオーバーレイ
        blue_mask = cv2.inRange(img_hsv, (90, 80, 40), (130, 255, 255))
        result[blue_mask > 0] = (
            result[blue_mask > 0] * 0.5 + np.array([200, 200, 0]) * 0.5
        ).astype(np.uint8)
        crop = _crop_rubber_dam(img_hsv)
    else:
        # ミラー: エッジ密度をヒートマップ表示
        edges = cv2.Canny(img_gray, 40, 120)
        ksize = max(h, w) // 6
        if ksize % 2 == 0:
            ksize += 1
        density = cv2.GaussianBlur(edges.astype(np.float32), (ksize, ksize), 0)
        if density.max() > 0:
            density_norm = (density / density.max() * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
            result = cv2.addWeighted(result, 0.6, heatmap, 0.4, 0)
        crop = _crop_mirror(img_hsv, img_gray)

    # crop範囲を赤枠で描画
    x0 = int(crop["x0"] * w)
    y0 = int(crop["y0"] * h)
    x1 = int(crop["x1"] * w)
    y1 = int(crop["y1"] * h)
    cv2.rectangle(result, (x0, y0), (x1, y1), (0, 0, 255), 3)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


# ============================================================
# 画像処理
# ============================================================
def boost_crop(crop: dict, zoom: float) -> dict:
    """crop領域の中心を保ったまま範囲を zoom 倍に縮小する"""
    cx = (crop["x0"] + crop["x1"]) / 2
    cy = (crop["y0"] + crop["y1"]) / 2
    hw = (crop["x1"] - crop["x0"]) / 2 / zoom
    hh = (crop["y1"] - crop["y0"]) / 2 / zoom
    return {
        "x0": max(0.0, cx - hw),
        "y0": max(0.0, cy - hh),
        "x1": min(1.0, cx + hw),
        "y1": min(1.0, cy + hh),
    }


def process_single_panel(
    img: Image.Image,
    rotation_deg: float,
    crop: dict,
    panel_w: int,
    panel_h: int,
    zoom_boost: float = 1.0,
) -> Image.Image:
    """1枚を 回転 → クロップ → パネルサイズにリサイズ"""
    img = fix_orientation(img)

    if abs(rotation_deg) > 0.1:
        img = img.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)

    if zoom_boost > 1.0:
        crop = boost_crop(crop, zoom_boost)

    w, h = img.size
    x0 = max(0, min(int(crop["x0"] * w), w - 1))
    y0 = max(0, min(int(crop["y0"] * h), h - 1))
    x1 = max(x0 + 1, min(int(crop["x1"] * w), w))
    y1 = max(y0 + 1, min(int(crop["y1"] * h), h))
    img = img.crop((x0, y0, x1, y1))

    img = ImageOps.fit(img, (panel_w, panel_h), method=Image.BICUBIC)
    return img


def compose_panels(
    panels: list[Image.Image],
    output_size: tuple[int, int],
    mode: str = "clinic",
    clinic_name: str = "坂寄歯科医院",
    sns_handle: str = "@dentist_mickey",
    logo_img: Image.Image | None = None,
) -> Image.Image:
    """Nパネル合成 + フッター/オーバーレイ描画"""
    out_w, out_h = output_size
    num_panels = len(panels)
    canvas = Image.new("RGB", (out_w, out_h), BG_COLOR)

    # パネル幅を均等に配分（余りは左側のパネルに1pxずつ加算）
    base_w = out_w // num_panels
    remainder = out_w % num_panels
    x_offset = 0
    for i, panel in enumerate(panels):
        pw = base_w + (1 if i < remainder else 0)
        panel_resized = ImageOps.fit(panel, (pw, out_h), method=Image.BICUBIC)
        canvas.paste(panel_resized, (x_offset, 0))
        x_offset += pw

    canvas_rgba = canvas.convert("RGBA")

    if mode == "clinic":
        canvas_rgba = _draw_clinic_footer(canvas_rgba, out_w, out_h, clinic_name)
    elif mode == "sns":
        canvas_rgba = _draw_sns_overlay(canvas_rgba, out_w, out_h, sns_handle, logo_img)

    return canvas_rgba.convert("RGB")


def _draw_clinic_footer(
    canvas: Image.Image, w: int, h: int, clinic_name: str
) -> Image.Image:
    """モードA: 半透明フッター + 医院名"""
    footer_h = max(70, int(h * 0.083))

    overlay = Image.new("RGBA", (w, footer_h), (0, 0, 0, 128))
    canvas.paste(overlay, (0, h - footer_h), overlay)

    line = Image.new("RGBA", (w, 2), (255, 255, 255, 180))
    canvas.paste(line, (0, h - footer_h - 1), line)

    draw = ImageDraw.Draw(canvas)
    font_size = int(footer_h * 0.60)
    font = find_font(font_size)
    bbox = draw.textbbox((0, 0), clinic_name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (w - text_w) // 2
    y = h - footer_h + (footer_h - text_h) // 2 - int(font_size * 0.1)
    draw.text((x, y), clinic_name, fill=(255, 255, 255, 230), font=font)

    return canvas


def _draw_sns_overlay(
    canvas: Image.Image,
    w: int,
    h: int,
    sns_handle: str,
    logo_img: Image.Image | None,
) -> Image.Image:
    """モードB: 右下にSNSハンドル + ロゴ"""
    draw = ImageDraw.Draw(canvas)
    margin = int(w * 0.02)
    font_size = int(h * 0.04)
    font = find_font(font_size)

    logo_offset_w = 0
    if logo_img is not None:
        logo_h = int(h * 0.06)
        logo_w = int(logo_img.width * (logo_h / logo_img.height))
        logo_resized = logo_img.resize((logo_w, logo_h), Image.BICUBIC)
        if logo_resized.mode != "RGBA":
            logo_resized = logo_resized.convert("RGBA")
        logo_x = w - margin - logo_w
        logo_y = h - margin - logo_h
        canvas.paste(logo_resized, (logo_x, logo_y), logo_resized)
        logo_offset_w = logo_w + int(margin * 0.5)

    bbox = draw.textbbox((0, 0), sns_handle, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = w - margin - logo_offset_w - text_w
    text_y = h - margin - text_h
    draw.text((text_x + 2, text_y + 2), sns_handle, fill=(0, 0, 0, 160), font=font)
    draw.text((text_x, text_y), sns_handle, fill=(255, 255, 255, 230), font=font)

    return canvas


# ============================================================
# ドラッグ編集エディタ
# ============================================================
def _img_to_data_url(img: Image.Image, quality: int = 95) -> str:
    """PIL Image → base64 data URL"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def generate_editor_source(
    img_pil: Image.Image,
    rotation_deg: float,
    crop: dict,
    zoom_boost: float = 1.0,
    target_w: int = 768,
    target_h: int = 2160,
) -> Image.Image:
    """CV結果に上下左右40%マージンを追加したソース画像を生成（ドラッグ余白用）

    zoom_boost適用前のCV cropを基準に、十分な余白を確保する。
    JSエディタではこの画像の中央部分がデフォルト表示位置になる。
    """
    img = fix_orientation(img_pil)
    if abs(rotation_deg) > 0.1:
        img = img.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)

    # zoom_boost適用前のCVクロップを基準にする（広い範囲）
    w, h = img.size
    cx = (crop["x0"] + crop["x1"]) / 2
    cy = (crop["y0"] + crop["y1"]) / 2
    cw = crop["x1"] - crop["x0"]
    ch = crop["y1"] - crop["y0"]

    # CV cropの上下左右に40%マージンを追加（zoom_boost前の広い範囲が基準）
    margin = 0.4
    crop_w = cw * (1 + margin * 2)
    crop_h = ch * (1 + margin * 2)

    # アスペクト比をターゲットに合わせる
    target_ratio = target_w / target_h
    if crop_w / max(crop_h, 0.001) > target_ratio:
        crop_h = crop_w / target_ratio
    else:
        crop_w = crop_h * target_ratio

    x0 = max(0, int((cx - crop_w / 2) * w))
    y0 = max(0, int((cy - crop_h / 2) * h))
    x1 = min(w, int((cx + crop_w / 2) * w))
    y1 = min(h, int((cy + crop_h / 2) * h))

    cropped = img.crop((x0, y0, x1, y1))
    return ImageOps.fit(cropped, (target_w, target_h), method=Image.BICUBIC)


def generate_editor_source_simple(
    img_pil: Image.Image,
    rotation_deg: float,
    max_dim: int = 3000,
) -> Image.Image:
    """CV処理なしでソース画像を生成。アスペクト比を維持してリサイズ。"""
    img = fix_orientation(img_pil)
    if abs(rotation_deg) > 0.1:
        img = img.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)
    # アスペクト比を維持してリサイズ
    img.thumbnail((max_dim, max_dim), Image.BICUBIC)
    return img


def build_editor_html(
    source_data_urls: list[str],
    labels: list[str],
    zoom_boosts: list[float],
    source_sizes: list[tuple[int, int]],
    output_w: int = 3508,
    output_h: int = 2480,
    footer_text: str = "坂寄歯科医院",
    footer_mode: str = "clinic",
    sns_handle: str = "@dentist_mickey",
) -> str:
    """単一キャンバスWYSIWYGエディタのHTML/JSを生成"""
    num = len(source_data_urls)
    panel_w = output_w // num

    panels_js = json.dumps([
        {"src": url, "label": lbl, "sw": sw, "sh": sh}
        for url, lbl, (sw, sh) in zip(source_data_urls, labels, source_sizes)
    ])

    footer_text_escaped = footer_text.replace("'", "\\'").replace('"', '\\"')
    sns_handle_escaped = sns_handle.replace("'", "\\'").replace('"', '\\"')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#1a1a1a;color:#fff;font-family:-apple-system,sans-serif;
  -webkit-user-select:none;user-select:none;overflow-x:hidden}}
.wrap{{padding:4px;max-width:100vw}}
#preview{{width:100%;aspect-ratio:{output_w}/{output_h};background:#0f0f0f;
  cursor:grab;touch-action:none;border:2px solid #444;border-radius:4px;
  image-rendering:auto}}
#preview:active{{cursor:grabbing}}
.sel-info{{text-align:center;color:#aaa;font-size:12px;margin:4px 0}}
.sel-info .sel-name{{color:#ff4b4b;font-weight:bold}}
.ctrl-row{{display:flex;gap:2px;justify-content:center;flex-wrap:wrap;margin:3px 0}}
.ctrl-btn{{background:#333;color:#ccc;border:1px solid #555;border-radius:3px;
  padding:2px 6px;font-size:10px;cursor:pointer;min-height:22px;
  display:flex;align-items:center;justify-content:center}}
.ctrl-btn:hover,.ctrl-btn:active{{background:#555}}
.ctrl-btn.active{{background:#ff4b4b;color:#fff;border-color:#ff4b4b}}
.btns{{display:flex;gap:8px;justify-content:center;margin:6px 0;flex-wrap:wrap}}
.btn{{padding:8px 20px;font-size:14px;font-weight:bold;border:none;border-radius:6px;
  cursor:pointer;color:#fff;min-height:36px}}
.btn-dl{{background:#ff4b4b}}.btn-dl:hover{{background:#e03e3e}}
.btn-reset{{background:#555}}.btn-reset:hover{{background:#666}}
</style></head><body><div class="wrap">
<canvas id="preview"></canvas>
<div class="sel-info">選択中: <span class="sel-name" id="sel-name">① (クリックで写真を選択)</span>
  <span id="sel-info-detail">角度:0.0° ズーム:1.0x</span></div>
<div class="ctrl-row" id="panel-tabs"></div>
<div class="ctrl-row" id="rot-ctrl"></div>
<div class="ctrl-row" id="zoom-ctrl"></div>
<div class="ctrl-row">
  <select id="footer-mode" class="ctrl-btn" style="min-width:120px" onchange="changeFooterMode()">
    <option value="plain">シンプル（ロゴなし）</option>
    <option value="clinic">医院名フッター</option>
    <option value="sns">SNSオーバーレイ</option>
  </select>
  <input id="footer-text" type="text" class="ctrl-btn" style="min-width:120px;background:#222;color:#fff" value="{footer_text_escaped}" placeholder="医院名" oninput="draw()">
  <input id="sns-text" type="text" class="ctrl-btn" style="min-width:100px;background:#222;color:#fff;display:none" value="{sns_handle_escaped}" placeholder="@handle" oninput="draw()">
</div>
<div class="btns">
<button class="btn btn-dl" onclick="dlFull('png')">PNG ダウンロード</button>
<button class="btn btn-dl" onclick="dlFull('webp')">WebP ダウンロード</button>
<button class="btn btn-reset" onclick="resetAll()">リセット</button>
</div></div>
<canvas id="cv" style="display:none"></canvas>
<script>
const P={panels_js};
const OW={output_w},OH={output_h},PW={panel_w},N=P.length;
/* State per panel: ox,oy in output coords (px), sc=zoom, rot=degrees */
const S=P.map((p,i)=>{{
  return{{ox:PW*i+PW/2, oy:OH/2, sc:1.0, rot:0, imgEl:null,
    initOx:PW*i+PW/2, initOy:OH/2}};
}});
let sel=0; /* selected panel index */
let showGrid=true;

function getFooterMode(){{return document.getElementById('footer-mode').value}}
function getFooterText(){{return document.getElementById('footer-text').value}}
function getSnsText(){{return document.getElementById('sns-text').value}}
function changeFooterMode(){{
  const m=getFooterMode();
  document.getElementById('footer-text').style.display=m==='clinic'?'':'none';
  document.getElementById('sns-text').style.display=m==='sns'?'':'none';
  draw();
}}

function drawFooter(ctx,w,h){{
  const fm=getFooterMode();
  const ft=getFooterText();
  const sn=getSnsText();
  if(fm==='clinic'&&ft){{
    const fh=Math.round(h*0.06),fy=h-fh;
    ctx.fillStyle='rgba(0,0,0,0.6)';ctx.fillRect(0,fy,w,fh);
    ctx.strokeStyle='rgba(255,255,255,0.8)';ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(0,fy);ctx.lineTo(w,fy);ctx.stroke();
    const fs=Math.round(fh*0.55);
    ctx.font='bold '+fs+'px "Hiragino Kaku Gothic ProN","Hiragino Sans","Noto Sans CJK JP","Meiryo",sans-serif';
    ctx.fillStyle='#fff';ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(ft,w/2,fy+fh/2);
  }}else if(fm==='sns'&&sn){{
    const fs=Math.round(h*0.025);
    ctx.font=fs+'px "Hiragino Kaku Gothic ProN",sans-serif';
    ctx.fillStyle='rgba(255,255,255,0.85)';ctx.textAlign='right';ctx.textBaseline='bottom';
    ctx.fillText(sn,w-30,h-30);
  }}
}}

/* Load images */
const imgs=[];
let loadCount=0;
P.forEach((p,i)=>{{
  const im=new Image();
  im.onload=()=>{{loadCount++;if(loadCount===N)draw()}};
  im.src=p.src;
  imgs.push(im);
  S[i].imgEl=im;
}});

const cv=document.getElementById('preview');
function getCvScale(){{return cv.clientWidth/OW}}

function draw(){{
  const dpr=1;/* use 1 for performance, actual output uses full res */
  cv.width=cv.clientWidth;cv.height=cv.clientHeight;
  const sc=cv.width/OW;
  const ctx=cv.getContext('2d');
  ctx.fillStyle='#0f0f0f';ctx.fillRect(0,0,cv.width,cv.height);
  ctx.save();ctx.scale(sc,sc);

  S.forEach((s,i)=>{{
    const im=imgs[i];if(!im.complete)return;
    const pw=PW,ph=OH;
    const px=i*PW;

    ctx.save();
    ctx.beginPath();ctx.rect(px,0,pw,ph);ctx.clip();

    /* Center of this panel slot */
    const cx=px+pw/2, cy=ph/2;

    /* Image dimensions scaled to fill panel slot while maintaining aspect ratio */
    const imgR=im.naturalWidth/im.naturalHeight;
    const slotR=pw/ph;
    let drawW,drawH;
    if(imgR>slotR){{drawH=ph*s.sc;drawW=drawH*imgR}}
    else{{drawW=pw*s.sc;drawH=drawW/imgR}}

    ctx.translate(s.ox, s.oy);
    if(Math.abs(s.rot)>0.01)ctx.rotate(s.rot*Math.PI/180);
    ctx.drawImage(im, -drawW/2, -drawH/2, drawW, drawH);
    ctx.restore();
  }});

  /* Footer / overlay (dynamic) */
  drawFooter(ctx,OW,OH);

  /* Grid overlay (thirds + center) */
  if(showGrid){{
    ctx.strokeStyle='rgba(255,255,0,0.5)';ctx.lineWidth=2;
    for(let i=0;i<N;i++){{
      const px=i*PW;
      /* Vertical thirds */
      ctx.beginPath();ctx.moveTo(px+PW/3,0);ctx.lineTo(px+PW/3,OH);ctx.stroke();
      ctx.beginPath();ctx.moveTo(px+PW*2/3,0);ctx.lineTo(px+PW*2/3,OH);ctx.stroke();
      /* Horizontal thirds */
      ctx.beginPath();ctx.moveTo(px,OH/3);ctx.lineTo(px+PW,OH/3);ctx.stroke();
      ctx.beginPath();ctx.moveTo(px,OH*2/3);ctx.lineTo(px+PW,OH*2/3);ctx.stroke();
      /* Center crosshair */
      ctx.strokeStyle='rgba(0,255,255,0.45)';ctx.lineWidth=2;
      ctx.setLineDash([12,8]);
      ctx.beginPath();ctx.moveTo(px+PW/2,0);ctx.lineTo(px+PW/2,OH);ctx.stroke();
      ctx.beginPath();ctx.moveTo(px,OH/2);ctx.lineTo(px+PW,OH/2);ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle='rgba(255,255,0,0.5)';
    }}
    /* Panel dividers */
    ctx.strokeStyle='rgba(255,100,100,0.7)';ctx.lineWidth=3;
    for(let i=1;i<N;i++){{
      ctx.beginPath();ctx.moveTo(i*PW,0);ctx.lineTo(i*PW,OH);ctx.stroke();
    }}
  }}

  /* Selection highlight */
  const spx=sel*PW;
  ctx.strokeStyle='#ff4b4b';ctx.lineWidth=4;
  ctx.strokeRect(spx+2,2,PW-4,OH-4);

  ctx.restore();
}}

function updateUI(){{
  const s=S[sel];
  document.getElementById('sel-name').textContent=P[sel].label;
  document.getElementById('sel-info-detail').textContent=
    '角度:'+s.rot.toFixed(1)+'° ズーム:'+s.sc.toFixed(1)+'x';
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{{
    b.classList.toggle('active',i===sel);
  }});
  draw();
}}

/* Panel selection tabs */
(function(){{
  const tabs=document.getElementById('panel-tabs');
  P.forEach((p,i)=>{{
    const b=document.createElement('button');
    b.className='ctrl-btn tab-btn'+(i===0?' active':'');
    b.textContent=p.label;
    b.addEventListener('click',e=>{{e.preventDefault();sel=i;updateUI()}});
    tabs.appendChild(b);
  }});
}})();

/* Rotation controls */
(function(){{
  const row=document.getElementById('rot-ctrl');
  [[-90,'↺90'],[-5,'-5°'],[-1,'-1°'],[-0.1,'-0.1°'],[0.1,'+0.1°'],[1,'+1°'],[5,'+5°'],[90,'↻90'],[180,'180°']].forEach(([deg,txt])=>{{
    const b=document.createElement('button');b.className='ctrl-btn';b.textContent=txt;
    b.addEventListener('click',e=>{{e.preventDefault();S[sel].rot+=deg;updateUI()}});
    row.appendChild(b);
  }});
}})();

/* Zoom controls */
(function(){{
  const row=document.getElementById('zoom-ctrl');
  const zminus=document.createElement('button');zminus.className='ctrl-btn';zminus.textContent='🔍−';
  zminus.addEventListener('click',e=>{{e.preventDefault();S[sel].sc=Math.max(0.2,S[sel].sc-0.05);updateUI()}});
  const zplus=document.createElement('button');zplus.className='ctrl-btn';zplus.textContent='🔍+';
  zplus.addEventListener('click',e=>{{e.preventDefault();S[sel].sc=Math.min(5,S[sel].sc+0.05);updateUI()}});
  const gridBtn=document.createElement('button');gridBtn.className='ctrl-btn active';gridBtn.textContent='# グリッド';gridBtn.id='grid-btn';
  gridBtn.addEventListener('click',e=>{{e.preventDefault();showGrid=!showGrid;gridBtn.classList.toggle('active',showGrid);draw()}});
  row.appendChild(zminus);row.appendChild(zplus);row.appendChild(gridBtn);
}})();

/* --- Mouse interaction on canvas --- */
function hitTest(x,y){{
  /* Which panel slot does (x,y) in output coords fall into? */
  const idx=Math.floor(x/PW);
  return Math.max(0,Math.min(N-1,idx));
}}

let drag=false,dsx,dsy,dsox,dsoy;
cv.addEventListener('mousedown',e=>{{
  const rect=cv.getBoundingClientRect();
  const sc=getCvScale();
  const ox=(e.clientX-rect.left)/sc, oy=(e.clientY-rect.top)/sc;
  const hit=hitTest(ox,oy);
  sel=hit; updateUI();
  drag=true;dsx=e.clientX;dsy=e.clientY;
  dsox=S[sel].ox;dsoy=S[sel].oy;
  e.preventDefault();
}});
document.addEventListener('mousemove',e=>{{
  if(!drag)return;
  const sc=getCvScale();
  S[sel].ox=dsox+(e.clientX-dsx)/sc;
  S[sel].oy=dsoy+(e.clientY-dsy)/sc;
  draw();updateUI();
}});
document.addEventListener('mouseup',()=>{{drag=false}});

cv.addEventListener('wheel',e=>{{
  e.preventDefault();
  const zf=e.deltaY>0?0.97:1.03;
  const s=S[sel];
  const rect=cv.getBoundingClientRect();
  const sc=getCvScale();
  /* Zoom toward mouse position in output coords */
  const mx=(e.clientX-rect.left)/sc, my=(e.clientY-rect.top)/sc;
  const ns=Math.max(0.2,Math.min(5,s.sc*zf));
  const ratio=ns/s.sc;
  s.ox=mx-(mx-s.ox)*ratio;
  s.oy=my-(my-s.oy)*ratio;
  s.sc=ns;
  updateUI();
}},{{passive:false}});

/* --- Touch interaction --- */
let tState=null;
cv.addEventListener('touchstart',e=>{{
  e.preventDefault();
  const rect=cv.getBoundingClientRect();
  const sc=getCvScale();
  if(e.touches.length===1){{
    const t=e.touches[0];
    const ox=(t.clientX-rect.left)/sc, oy=(t.clientY-rect.top)/sc;
    sel=hitTest(ox,oy);updateUI();
    tState={{mode:'drag',sx:t.clientX,sy:t.clientY,sox:S[sel].ox,soy:S[sel].oy}};
  }}else if(e.touches.length===2){{
    const t0=e.touches[0],t1=e.touches[1];
    const dist=Math.hypot(t1.clientX-t0.clientX,t1.clientY-t0.clientY);
    tState={{mode:'pinch',dist0:dist,sc0:S[sel].sc,
      cx0:(t0.clientX+t1.clientX)/2,cy0:(t0.clientY+t1.clientY)/2,
      ox0:S[sel].ox,oy0:S[sel].oy}};
  }}
}},{{passive:false}});

cv.addEventListener('touchmove',e=>{{
  e.preventDefault();
  if(!tState)return;
  const sc=getCvScale();
  const s=S[sel];
  if(tState.mode==='drag'&&e.touches.length===1){{
    const t=e.touches[0];
    s.ox=tState.sox+(t.clientX-tState.sx)/sc;
    s.oy=tState.soy+(t.clientY-tState.sy)/sc;
    draw();updateUI();
  }}else if(e.touches.length===2){{
    const t0=e.touches[0],t1=e.touches[1];
    const dist=Math.hypot(t1.clientX-t0.clientX,t1.clientY-t0.clientY);
    if(tState.mode==='drag'){{
      tState={{mode:'pinch',dist0:dist,sc0:s.sc,
        cx0:(t0.clientX+t1.clientX)/2,cy0:(t0.clientY+t1.clientY)/2,
        ox0:s.ox,oy0:s.oy}};
      return;
    }}
    const ratio=dist/tState.dist0;
    const ns=Math.max(0.2,Math.min(5,tState.sc0*ratio));
    const r=ns/tState.sc0;
    const rect=cv.getBoundingClientRect();
    const mx=(tState.cx0-rect.left)/sc, my=(tState.cy0-rect.top)/sc;
    s.ox=mx-(mx-tState.ox0)*r;
    s.oy=my-(my-tState.oy0)*r;
    const cx=(t0.clientX+t1.clientX)/2;
    const cy=(t0.clientY+t1.clientY)/2;
    s.ox+=(cx-tState.cx0)/sc;
    s.oy+=(cy-tState.cy0)/sc;
    s.sc=ns;
    draw();updateUI();
  }}
}},{{passive:false}});
cv.addEventListener('touchend',e=>{{
  if(e.touches.length===0)tState=null;
  else if(e.touches.length===1){{
    const t=e.touches[0];
    tState={{mode:'drag',sx:t.clientX,sy:t.clientY,sox:S[sel].ox,soy:S[sel].oy}};
  }}
}});
cv.addEventListener('touchcancel',()=>{{tState=null}});

/* Resize */
let resizeT;
window.addEventListener('resize',()=>{{clearTimeout(resizeT);resizeT=setTimeout(draw,100)}});

/* Full-res render for download */
function renderFull(){{
  const c=document.getElementById('cv');c.width=OW;c.height=OH;
  const ctx=c.getContext('2d');
  ctx.fillStyle='#0f0f0f';ctx.fillRect(0,0,OW,OH);

  S.forEach((s,i)=>{{
    const im=imgs[i];if(!im.complete)return;
    const pw=PW,ph=OH;
    const px=i*PW;
    ctx.save();
    ctx.beginPath();ctx.rect(px,0,pw,ph);ctx.clip();
    const imgR=im.naturalWidth/im.naturalHeight;
    const slotR=pw/ph;
    let drawW,drawH;
    if(imgR>slotR){{drawH=ph*s.sc;drawW=drawH*imgR}}
    else{{drawW=pw*s.sc;drawH=drawW/imgR}}
    ctx.translate(s.ox, s.oy);
    if(Math.abs(s.rot)>0.01)ctx.rotate(s.rot*Math.PI/180);
    ctx.drawImage(im, -drawW/2, -drawH/2, drawW, drawH);
    ctx.restore();
  }});

  drawFooter(ctx,OW,OH);
  return c;
}}

function dlFull(fmt){{
  const c=renderFull();const a=document.createElement('a');
  if(fmt==='webp'){{a.download='case_composite.webp';a.href=c.toDataURL('image/webp',0.92)}}
  else{{a.download='case_composite.png';a.href=c.toDataURL('image/png')}}
  a.click();
  draw();/* restore preview canvas */
}}

function resetAll(){{
  S.forEach((s,i)=>{{
    s.sc=1;s.rot=0;s.ox=s.initOx;s.oy=s.initOy;
  }});
  updateUI();
}}

/* Initial draw once all images loaded or on next frame */
requestAnimationFrame(()=>{{if(loadCount===N)draw();else setTimeout(draw,500)}});
</script></body></html>"""


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(
        page_title="歯科症例写真 合成アプリ",
        page_icon="🦷",
        layout="wide",
    )

    st.title("🦷 歯科症例写真 自動合成アプリ")
    st.caption("OpenCV + AI で口腔内写真を自動で回転・クロップ・合成します")

    # ---- サイドバー ----
    with st.sidebar:
        st.header("⚙️ 設定")

        output_mode = st.radio(
            "出力モード",
            ["医院名フッター", "SNSオーバーレイ", "シンプル（ロゴなし）"],
            help="完成画像のフッターデザインを選択",
        )

        if output_mode == "医院名フッター":
            clinic_name = st.text_input("医院名", value="坂寄歯科医院")
            sns_handle = ""
            logo_file = None
        elif output_mode == "SNSオーバーレイ":
            clinic_name = ""
            sns_handle = st.text_input("SNSハンドル", value="@dentist_mickey")
            logo_file = st.file_uploader(
                "ロゴ画像（PNG, 透過推奨）",
                type=["png"],
                help="右下に表示するロゴアイコン",
            )
        else:
            clinic_name = ""
            sns_handle = ""
            logo_file = None

        st.divider()

        size_label = st.selectbox("出力サイズ", list(OUTPUT_PRESETS.keys()))
        output_size = OUTPUT_PRESETS[size_label]

        st.divider()
        st.subheader("🔄 回転設定")
        default_rotation = st.selectbox(
            "デフォルト回転方向",
            [-90.0, 90.0, -85.0, 85.0, 0.0],
            index=0,
            format_func=lambda x: f"{x:+.0f}° ({'反時計' if x < 0 else '時計'}回り)" if x != 0 else "回転なし",
            help="上顎ミラー写真は通常-90°、下顎は+90°",
        )

        st.divider()
        st.caption("v3.6 — 3〜5枚対応")

    # ---- 写真アップロード ----
    st.subheader("📷 写真をアップロード（3〜5枚・ドラッグ＆ドロップ可）")
    uploaded_files = st.file_uploader(
        "症例写真を選択またはドラッグ＆ドロップ",
        type=["jpg", "jpeg", "png", "tiff"],
        accept_multiple_files=True,
        key="photos",
    )

    if len(uploaded_files) == 0:
        st.info("3〜5枚の写真をアップロードしてください（複数選択・ドラッグ＆ドロップ対応）")
        return

    num_panels = len(uploaded_files)
    if num_panels < MIN_PANELS or num_panels > MAX_PANELS:
        st.warning(f"現在 {num_panels} 枚です。{MIN_PANELS}〜{MAX_PANELS} 枚アップロードしてください。")
        return

    labels = PHOTO_LABELS_DEFAULT[:num_panels] if num_panels <= len(PHOTO_LABELS_DEFAULT) else [f"写真{i+1}" for i in range(num_panels)]
    st.success(f"✅ {num_panels}枚アップロード済み")

    # プレビュー
    cols = st.columns(num_panels)
    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        with col:
            st.image(f, caption=labels[i], use_container_width=True)

    images = []
    for f in uploaded_files:
        f.seek(0)
        img = Image.open(f)
        images.append(img.copy())

    # ---- エディタ用ソース画像を生成（アップロード直後に表示） ----
    # ファイル数が変わったらリセット
    if "editor_sources" in st.session_state and len(st.session_state["editor_sources"]) != num_panels:
        st.session_state.pop("editor_sources", None)
        st.session_state.pop("editor_zooms", None)
        st.session_state.pop("editor_source_sizes", None)
        st.session_state.pop("cv_result", None)
    if "editor_sources" not in st.session_state:
        sources = []
        source_sizes = []
        for img in images:
            src = generate_editor_source_simple(img, rotation_deg=default_rotation)
            sources.append(_img_to_data_url(src))
            source_sizes.append(src.size)
        st.session_state["editor_sources"] = sources
        st.session_state["editor_source_sizes"] = source_sizes
        st.session_state["editor_zooms"] = [1.0] * num_panels

    # ---- ドラッグ編集エディタ（常に表示） ----
    st.subheader("🎨 各写真を調整 → ダウンロード")
    st.caption("ドラッグ=移動 ／ ホイール=ズーム ／ 下のボタンで回転・ズーム調整")

    mode = "clinic" if output_mode == "医院名フッター" else ("sns" if output_mode == "SNSオーバーレイ" else "plain")
    out_w, out_h = output_size

    # エディタHTMLをキャッシュ（出力モード変更時に再生成しない）
    editor_cache_key = f"v5_{num_panels}_{out_w}_{out_h}"
    if st.session_state.get("_editor_cache_key") != editor_cache_key:
        # 全モードの情報を渡す（JS側で切り替え可能）
        html = build_editor_html(
            source_data_urls=st.session_state["editor_sources"],
            labels=labels,
            zoom_boosts=st.session_state.get("editor_zooms", [2.0] * num_panels),
            source_sizes=st.session_state.get("editor_source_sizes", [(800, 600)] * num_panels),
            output_w=out_w,
            output_h=out_h,
        )
        st.session_state["_editor_html"] = html
        st.session_state["_editor_cache_key"] = editor_cache_key

    # キャンバス高さ(ブラウザ幅ベース、約800px幅想定) + 操作パネル分
    canvas_display_h = int(800 * out_h / out_w)
    editor_height = min(900, canvas_display_h + 180)
    components.html(st.session_state["_editor_html"], height=editor_height, scrolling=True)

    # ---- CV自動編集（オプション） ----
    with st.expander("📐 CV自動編集（オプション：自動トリミング）", expanded=False):
        st.caption("OpenCVで歯列を検出し自動トリミングします。手動調整で十分な場合は不要です。")
        cv_clicked = st.button(
            "CV自動編集を実行",
            use_container_width=True,
        )

        if cv_clicked:
            st.session_state.pop("editor_sources", None)
            st.session_state.pop("editor_zooms", None)
            st.session_state.pop("editor_source_sizes", None)
            progress = st.progress(0, text="OpenCV処理中...")

            result_panels = []
            for i, img in enumerate(images):
                progress.progress(
                    int((i / num_panels) * 60),
                    text=f"写真 {i+1}/{num_panels} を解析中...",
                )
                cv_result = cv_detect_crop(img, rotation_deg=default_rotation)
                result_panels.append({
                    "index": i + 1,
                    "rotation_cw_deg": cv_result["rotation_cw_deg"],
                    "crop": cv_result["crop"],
                    "method": cv_result.get("method", "mirror"),
                    "zoom_boost": cv_result.get("zoom_boost", 2.0),
                })

            result = {"panels": result_panels}
            st.session_state["cv_result"] = result

            progress.progress(60, text="エディタソース生成中...")

            sources = []
            zooms = []
            source_sizes = []
            for i, panel_data in enumerate(result_panels):
                zb = panel_data.get("zoom_boost", 2.0)
                src = generate_editor_source(
                    images[i],
                    panel_data["rotation_cw_deg"],
                    panel_data["crop"],
                    zoom_boost=zb,
                )
                sources.append(_img_to_data_url(src))
                zooms.append(zb)
                source_sizes.append(src.size)
            st.session_state["editor_sources"] = sources
            st.session_state["editor_zooms"] = zooms
            st.session_state["editor_source_sizes"] = source_sizes

            progress.progress(100, text="完了！ページをリロードしてエディタに反映します。")
            st.rerun()

    # ---- CVデバッグ ----
    if "cv_result" in st.session_state:
        with st.expander("🔬 CV歯検出デバッグ", expanded=False):
            debug_cols = st.columns(num_panels)
            for i in range(min(num_panels, len(images))):
                cv_res = st.session_state["cv_result"]
                if i < len(cv_res["panels"]):
                    debug_img = cv_create_debug_image(images[i], cv_res["panels"][i]["rotation_cw_deg"])
                    with debug_cols[i]:
                        st.image(debug_img, caption=labels[i], use_container_width=True)

        with st.expander("🐛 分析結果（デバッグ用）", expanded=False):
            st.json(st.session_state["cv_result"])


if __name__ == "__main__":
    main()
