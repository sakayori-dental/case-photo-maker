"""
歯科症例写真 自動合成アプリ v3.2
OpenCV自動処理 → 手動微調整 → 5パネル合成
"""
from __future__ import annotations

import streamlit as st
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
    "1920×1080 (フルHD)": (1920, 1080),
    "1600×900": (1600, 900),
    "1280×720 (HD)": (1280, 720),
}
NUM_PANELS = 5
BG_COLOR = "#0f0f0f"

PHOTO_LABELS = [
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
    """5パネル合成 + フッター/オーバーレイ描画"""
    out_w, out_h = output_size
    canvas = Image.new("RGB", (out_w, out_h), BG_COLOR)

    # パネル幅を均等に配分（余りは左側のパネルに1pxずつ加算）
    base_w = out_w // NUM_PANELS
    remainder = out_w % NUM_PANELS
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
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(
        page_title="歯科症例写真 合成アプリ",
        page_icon="🦷",
        layout="wide",
    )

    st.title("🦷 歯科症例写真 自動合成アプリ")
    st.caption("OpenCV + AI で5枚の口腔内写真を自動で回転・クロップ・合成します")

    # ---- サイドバー ----
    with st.sidebar:
        st.header("⚙️ 設定")

        output_mode = st.radio(
            "出力モード",
            ["医院名フッター", "SNSオーバーレイ"],
            help="完成画像のフッターデザインを選択",
        )

        if output_mode == "医院名フッター":
            clinic_name = st.text_input("医院名", value="坂寄歯科医院")
            sns_handle = ""
            logo_file = None
        else:
            clinic_name = ""
            sns_handle = st.text_input("SNSハンドル", value="@dentist_mickey")
            logo_file = st.file_uploader(
                "ロゴ画像（PNG, 透過推奨）",
                type=["png"],
                help="右下に表示するロゴアイコン",
            )

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
        st.caption("v3.2 — CV自動 + 手動微調整")

    # ---- 写真アップロード ----
    st.subheader("📷 写真をアップロード（5枚まとめてドラッグ＆ドロップ可）")
    uploaded_files = st.file_uploader(
        "5枚の症例写真を選択またはドラッグ＆ドロップ",
        type=["jpg", "jpeg", "png", "tiff"],
        accept_multiple_files=True,
        key="photos",
    )

    if len(uploaded_files) == 0:
        st.info("5枚の写真をアップロードしてください（複数選択・ドラッグ＆ドロップ対応）")
        return

    if len(uploaded_files) != NUM_PANELS:
        st.warning(f"現在 {len(uploaded_files)} 枚です。{NUM_PANELS} 枚ちょうどアップロードしてください。")
        return

    st.success("✅ 5枚すべてアップロード済み")

    # プレビュー
    cols = st.columns(NUM_PANELS)
    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        with col:
            st.image(f, caption=PHOTO_LABELS[i], use_container_width=True)

    images = []
    for f in uploaded_files:
        f.seek(0)
        img = Image.open(f)
        images.append(img.copy())

    # ---- 処理ボタン ----
    cv_clicked = st.button(
        "📐 CV自動編集",
        type="primary",
        use_container_width=True,
    )

    # ---- CV処理 ----
    if cv_clicked:
        progress = st.progress(0, text="OpenCV処理中...")

        result_panels = []
        for i, img in enumerate(images):
            progress.progress(
                int((i / NUM_PANELS) * 60),
                text=f"写真 {i+1}/5 を解析中...",
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

        progress.progress(60, text="画像合成中...")

        out_w, out_h = output_size
        panel_w = out_w // NUM_PANELS

        processed_panels = []
        for i, panel_data in enumerate(result_panels):
            panel = process_single_panel(
                images[i],
                panel_data["rotation_cw_deg"],
                panel_data["crop"],
                panel_w,
                out_h,
                zoom_boost=panel_data.get("zoom_boost", 2.0),
            )
            processed_panels.append(panel)
            progress.progress(
                60 + int((i + 1) / NUM_PANELS * 25),
                text=f"写真 {i+1}/5 処理完了",
            )

        st.session_state["processed_panels"] = processed_panels

        # デバッグ画像を保存
        debug_images = []
        for i, img in enumerate(images):
            debug_img = cv_create_debug_image(img, result_panels[i]["rotation_cw_deg"])
            debug_images.append(debug_img)
        st.session_state["cv_debug_images"] = debug_images

        logo_img = None
        if logo_file is not None:
            logo_img = Image.open(logo_file)

        mode = "clinic" if output_mode == "医院名フッター" else "sns"
        composite = compose_panels(
            processed_panels,
            output_size,
            mode=mode,
            clinic_name=clinic_name,
            sns_handle=sns_handle,
            logo_img=logo_img,
        )
        st.session_state["composite"] = composite
        progress.progress(100, text="完成！")

    # ---- 結果表示 ----
    if "composite" in st.session_state:
        st.subheader("🎨 合成結果")
        st.image(st.session_state["composite"], use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            buf_png = io.BytesIO()
            st.session_state["composite"].save(buf_png, format="PNG")
            st.download_button(
                "📥 PNG ダウンロード",
                data=buf_png.getvalue(),
                file_name="case_composite.png",
                mime="image/png",
                use_container_width=True,
            )
        with col_dl2:
            buf_webp = io.BytesIO()
            st.session_state["composite"].save(buf_webp, format="WEBP", quality=90)
            st.download_button(
                "📥 WebP ダウンロード",
                data=buf_webp.getvalue(),
                file_name="case_composite.webp",
                mime="image/webp",
                use_container_width=True,
            )

        # ---- 手動微調整（リアルタイムプレビュー） ----
        with st.expander("🔧 手動微調整", expanded=True):
            st.caption("スライダーを動かすと即座にプレビューが更新されます")

            if "cv_result" in st.session_state:
                cv_result = st.session_state["cv_result"]
                out_w, out_h = output_size
                panel_w = out_w // NUM_PANELS

                # 参考画像を表示
                ref_path = Path(__file__).parent / "reference" / "1.png"
                if ref_path.exists():
                    st.markdown("**📌 完成形（参考）**")
                    st.image(str(ref_path), use_container_width=True)
                    st.divider()

                adjusted_panels = []
                adjusted_panels_data = []
                for i, panel_data in enumerate(cv_result["panels"]):
                    st.markdown(f"**{PHOTO_LABELS[i]}**")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        rot = st.number_input(
                            "回転 (°)",
                            value=float(panel_data["rotation_cw_deg"]),
                            min_value=-180.0,
                            max_value=180.0,
                            step=0.5,
                            key=f"rot_{i}",
                        )
                    with c2:
                        cx = st.slider(
                            "X中心",
                            0.0, 1.0,
                            value=(panel_data["crop"]["x0"] + panel_data["crop"]["x1"]) / 2,
                            step=0.01,
                            key=f"cx_{i}",
                        )
                    with c3:
                        cy = st.slider(
                            "Y中心",
                            0.0, 1.0,
                            value=(panel_data["crop"]["y0"] + panel_data["crop"]["y1"]) / 2,
                            step=0.01,
                            key=f"cy_{i}",
                        )
                    with c4:
                        zoom = st.slider(
                            "ズーム",
                            0.5, 2.0,
                            value=1.0,
                            step=0.05,
                            key=f"zoom_{i}",
                        )

                    orig_w = panel_data["crop"]["x1"] - panel_data["crop"]["x0"]
                    orig_h = panel_data["crop"]["y1"] - panel_data["crop"]["y0"]
                    new_w = orig_w / zoom
                    new_h = orig_h / zoom
                    adj_crop = {
                        "x0": max(0.0, cx - new_w / 2),
                        "y0": max(0.0, cy - new_h / 2),
                        "x1": min(1.0, cx + new_w / 2),
                        "y1": min(1.0, cy + new_h / 2),
                    }
                    adjusted_panels_data.append({
                        "rotation_cw_deg": rot,
                        "crop": adj_crop,
                    })

                    # リアルタイムパネルプレビュー
                    panel_img = process_single_panel(
                        images[i], rot, adj_crop, panel_w, out_h,
                    )
                    adjusted_panels.append(panel_img)
                    st.image(panel_img, caption=f"プレビュー: {PHOTO_LABELS[i]}", width=120)
                    st.divider()

                # リアルタイム合成プレビュー
                st.markdown("**🖼️ 合成プレビュー**")
                logo_img = None
                if logo_file is not None:
                    logo_file.seek(0)
                    logo_img = Image.open(logo_file)
                mode = "clinic" if output_mode == "医院名フッター" else "sns"
                live_composite = compose_panels(
                    adjusted_panels,
                    output_size,
                    mode=mode,
                    clinic_name=clinic_name,
                    sns_handle=sns_handle,
                    logo_img=logo_img,
                )
                st.image(live_composite, use_container_width=True)

                # 適用ボタン: session_stateに保存してDLボタン等に反映
                if st.button("✅ この結果を確定", use_container_width=True):
                    st.session_state["composite"] = live_composite
                    st.session_state["processed_panels"] = adjusted_panels

                    # 比較ログ作成
                    comparison_rows = []
                    for i, (orig, adj) in enumerate(zip(cv_result["panels"], adjusted_panels_data)):
                        orig_cx = (orig["crop"]["x0"] + orig["crop"]["x1"]) / 2
                        orig_cy = (orig["crop"]["y0"] + orig["crop"]["y1"]) / 2
                        adj_cx = (adj["crop"]["x0"] + adj["crop"]["x1"]) / 2
                        adj_cy = (adj["crop"]["y0"] + adj["crop"]["y1"]) / 2
                        orig_cw = orig["crop"]["x1"] - orig["crop"]["x0"]
                        adj_cw = adj["crop"]["x1"] - adj["crop"]["x0"]
                        zoom_val = orig_cw / adj_cw if adj_cw > 0 else 1.0
                        comparison_rows.append({
                            "写真": PHOTO_LABELS[i],
                            "回転(CV)": f"{orig['rotation_cw_deg']:.1f}",
                            "回転(手動)": f"{adj['rotation_cw_deg']:.1f}",
                            "X中心(CV)": f"{orig_cx:.2f}",
                            "X中心(手動)": f"{adj_cx:.2f}",
                            "Y中心(CV)": f"{orig_cy:.2f}",
                            "Y中心(手動)": f"{adj_cy:.2f}",
                            "ズーム": f"{zoom_val:.2f}",
                        })
                    comparison_json = json.dumps({
                        "cv_original": cv_result["panels"],
                        "manual_adjusted": adjusted_panels_data,
                        "comparison": comparison_rows,
                    }, ensure_ascii=False, indent=2)
                    st.session_state["adjustment_log"] = {
                        "comparison": comparison_rows,
                        "json": comparison_json,
                    }
                    st.rerun()

        # ---- 調整ログ ----
        if "adjustment_log" in st.session_state:
            with st.expander("📊 最新の調整ログ", expanded=False):
                st.table(st.session_state["adjustment_log"]["comparison"])
                st.download_button(
                    "📋 比較ログをJSONでコピー",
                    data=st.session_state["adjustment_log"]["json"],
                    file_name="adjustment_log.json",
                    mime="application/json",
                    use_container_width=True,
                    key="dl_adj_log",
                )
                st.code(st.session_state["adjustment_log"]["json"], language="json")

        # ---- CVデバッグ ----
        if "cv_debug_images" in st.session_state:
            with st.expander("🔬 CV歯検出デバッグ（緑=歯マスク, 赤枠=crop範囲）", expanded=False):
                debug_cols = st.columns(NUM_PANELS)
                for i, (col, dbg) in enumerate(zip(debug_cols, st.session_state["cv_debug_images"])):
                    with col:
                        st.image(dbg, caption=PHOTO_LABELS[i], use_container_width=True)

        # ---- デバッグ ----
        with st.expander("🐛 分析結果（デバッグ用）", expanded=False):
            if "cv_result" in st.session_state:
                st.json(st.session_state["cv_result"])


if __name__ == "__main__":
    main()
