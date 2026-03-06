"""
歯科症例写真 自動合成アプリ v2.0
Claude Vision AI で5枚の口腔内写真を自動分析 → 回転・クロップ・ズーム統一 → 5パネル合成
"""

import streamlit as st
import anthropic
import base64
import json
import io
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps, ExifTags

# ============================================================
# 定数
# ============================================================
APP_DIR = Path(__file__).parent
PROMPT_FILE = APP_DIR / "prompt.txt"

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

PHOTO_DESCRIPTIONS = [
    "術前の頬側観（ミラー使用、口腔内写真）",
    "術前の咬合面観（ミラー使用、口腔内写真）",
    "術中の窩洞形成後（ラバーダム装着、咬合面観）",
    "術中の充填中（ラバーダム装着、咬合面観）",
    "術後の頬側観（ミラー使用、口腔内写真）",
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
def load_prompt() -> str:
    """prompt.txt からプロンプトテンプレートを読み込み"""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8")
    st.error(f"prompt.txt が見つかりません: {PROMPT_FILE}")
    return ""


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


def image_to_base64(img_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """画像バイトを Claude API 用 base64 辞書に変換"""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(img_bytes).decode("utf-8"),
        },
    }


def resize_for_api(img: Image.Image, max_dim: int = 1568) -> bytes:
    """API送信用にリサイズ（トークン節約）"""
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def extract_json_from_response(text: str) -> dict | None:
    """Claude 応答から JSON を抽出"""
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


# ============================================================
# AI 分析
# ============================================================
def analyze_photos_with_ai(
    client: anthropic.Anthropic,
    images: list[Image.Image],
    descriptions: list[str],
) -> tuple[dict | None, str]:
    """Claude Vision API で5枚を一括分析"""
    # プロンプト読み込み & 写真説明を埋め込み
    prompt_template = load_prompt()
    if not prompt_template:
        return None, "prompt.txt の読み込みに失敗"

    desc_text = "\n".join(f"写真{i+1}: {d}" for i, d in enumerate(descriptions))
    prompt = prompt_template.replace("{photo_descriptions}", desc_text)

    # メッセージ構築
    content = []
    for i, img in enumerate(images):
        img_bytes = resize_for_api(img)
        content.append({"type": "text", "text": f"--- 写真{i+1}: {descriptions[i]} ---"})
        content.append(image_to_base64(img_bytes))
    content.append({"type": "text", "text": prompt})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        response_text = response.content[0].text
        return extract_json_from_response(response_text), response_text
    except Exception as e:
        st.error(f"API エラー: {e}")
        return None, str(e)


# ============================================================
# 画像処理
# ============================================================
def process_single_panel(
    img: Image.Image,
    rotation_deg: float,
    crop: dict,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    """1枚を 回転 → クロップ → パネルサイズにリサイズ"""
    img = fix_orientation(img)

    if abs(rotation_deg) > 0.1:
        img = img.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)

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

    panel_w = out_w // NUM_PANELS
    for i, panel in enumerate(panels):
        panel_resized = ImageOps.fit(panel, (panel_w, out_h), method=Image.BICUBIC)
        canvas.paste(panel_resized, (i * panel_w, 0))

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
    font_size = int(footer_h * 0.55)
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
    st.caption("Claude Vision AI が5枚の口腔内写真を自動で回転・クロップ・合成します")

    # ---- サイドバー ----
    with st.sidebar:
        st.header("⚙️ 設定")

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Claude APIキーを入力してください",
        )

        st.divider()

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
        st.caption("v2.0 — Claude Vision AI powered")

    # ---- 写真アップロード ----
    st.subheader("📷 写真をアップロード")
    cols = st.columns(NUM_PANELS)
    uploaded_files = []

    for i, col in enumerate(cols):
        with col:
            f = st.file_uploader(
                PHOTO_LABELS[i],
                type=["jpg", "jpeg", "png", "tiff"],
                key=f"photo_{i}",
            )
            uploaded_files.append(f)
            if f is not None:
                st.image(f, use_container_width=True)

    all_uploaded = all(f is not None for f in uploaded_files)

    if not all_uploaded:
        remaining = sum(1 for f in uploaded_files if f is None)
        st.info(f"あと {remaining} 枚の写真をアップロードしてください")
        return

    st.success("✅ 5枚すべてアップロード済み")

    images = []
    for f in uploaded_files:
        f.seek(0)
        img = Image.open(f)
        images.append(img.copy())

    # ---- AI分析 & 合成 ----
    if st.button("🤖 AI自動編集 & 合成を開始", type="primary", use_container_width=True):
        if not api_key:
            st.error("サイドバーでAPIキーを入力してください")
            return

        client = anthropic.Anthropic(api_key=api_key)

        with st.spinner("Claude Vision AI が写真を分析中..."):
            progress = st.progress(0, text="AI分析中...")
            result, raw_response = analyze_photos_with_ai(
                client, images, PHOTO_DESCRIPTIONS
            )
            progress.progress(50, text="分析完了、画像処理中...")

        if result is None:
            st.error("AI分析に失敗しました。APIキーと画像を確認してください。")
            with st.expander("API応答（デバッグ用）"):
                st.code(raw_response)
            return

        st.session_state["ai_result"] = result
        st.session_state["raw_response"] = raw_response

        out_w, out_h = output_size
        panel_w = out_w // NUM_PANELS

        processed_panels = []
        for i, panel_data in enumerate(result["panels"]):
            panel = process_single_panel(
                images[i],
                panel_data["rotation_cw_deg"],
                panel_data["crop"],
                panel_w,
                out_h,
            )
            processed_panels.append(panel)
            progress.progress(
                50 + int((i + 1) / NUM_PANELS * 40),
                text=f"写真 {i+1}/5 処理完了",
            )

        st.session_state["processed_panels"] = processed_panels

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

        # ---- 手動微調整 ----
        with st.expander("🔧 手動微調整", expanded=False):
            st.caption("AI分析結果をスライダーで調整 → 「再合成」で反映")

            if "ai_result" in st.session_state:
                ai_result = st.session_state["ai_result"]

                adjusted_panels_data = []
                for i, panel_data in enumerate(ai_result["panels"]):
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
                    adjusted_panels_data.append({
                        "rotation_cw_deg": rot,
                        "crop": {
                            "x0": max(0.0, cx - new_w / 2),
                            "y0": max(0.0, cy - new_h / 2),
                            "x1": min(1.0, cx + new_w / 2),
                            "y1": min(1.0, cy + new_h / 2),
                        },
                    })

                if st.button("🔄 微調整を適用して再合成", use_container_width=True):
                    out_w, out_h = output_size
                    panel_w = out_w // NUM_PANELS

                    new_panels = []
                    for i, adj in enumerate(adjusted_panels_data):
                        panel = process_single_panel(
                            images[i],
                            adj["rotation_cw_deg"],
                            adj["crop"],
                            panel_w,
                            out_h,
                        )
                        new_panels.append(panel)

                    logo_img = None
                    if logo_file is not None:
                        logo_file.seek(0)
                        logo_img = Image.open(logo_file)

                    mode = "clinic" if output_mode == "医院名フッター" else "sns"
                    composite = compose_panels(
                        new_panels,
                        output_size,
                        mode=mode,
                        clinic_name=clinic_name,
                        sns_handle=sns_handle,
                        logo_img=logo_img,
                    )
                    st.session_state["composite"] = composite
                    st.rerun()

        # ---- デバッグ ----
        with st.expander("🐛 AI分析結果（デバッグ用）", expanded=False):
            if "ai_result" in st.session_state:
                st.json(st.session_state["ai_result"])
            if "raw_response" in st.session_state:
                st.code(st.session_state["raw_response"], language="json")


if __name__ == "__main__":
    main()
