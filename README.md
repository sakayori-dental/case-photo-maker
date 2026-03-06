# 🦷 歯科症例写真 自動合成アプリ

5枚の歯科口腔内写真をアップロードすると、Claude Vision AI が自動で回転・クロップ・ズーム統一を行い、SNS投稿用の5パネル合成画像（PNG / WebP）を出力する Streamlit アプリです。

## 完成イメージ

| 医院名フッター版 | SNSオーバーレイ版 |
|:-:|:-:|
| ![mode_a](docs/sample_clinic.png) | ![mode_b](docs/sample_sns.png) |

> サンプル画像は `reference/` フォルダに配置してください（Git管理外）

## セットアップ

```bash
git clone https://github.com/<your-account>/case-photo-maker.git
cd case-photo-maker
pip install -r requirements.txt
```

## 使い方

```bash
streamlit run app.py
```

1. サイドバーで Anthropic API Key を入力
2. 出力モード（医院名フッター / SNSオーバーレイ）を選択
3. 5枚の症例写真をアップロード
4. 「AI自動編集 & 合成を開始」をクリック
5. 必要に応じて手動微調整 → PNG / WebP でダウンロード

## 必要なもの

- Python 3.10+
- [Anthropic API Key](https://console.anthropic.com/)
- macOS の場合はヒラギノフォントが自動検出されます

## 5枚の写真構成

| # | 内容 | 撮影状況 |
|---|------|----------|
| ① | 術前・頬側観 | ミラー使用 |
| ② | 術前・咬合面観 | ミラー使用 |
| ③ | 術中・窩洞形成後 | ラバーダム装着 |
| ④ | 術中・充填中 | ラバーダム装着 |
| ⑤ | 術後・頬側観 | ミラー使用 |

## ローカル参考画像（Git管理外）

テスト用に `reference/` フォルダを作成し、以下を配置してください：

```
reference/
├── 1.png              # 完成形サンプル（医院名フッター版）
├── 2.png              # 完成形サンプル（SNSオーバーレイ版）
├── DSC_0044.JPG       # 元写真①〜⑤
├── ...
└── cropped/           # 手動クロップ済み中間ファイル（精度検証用）
    ├── panel_1.png
    └── ...
```

## ライセンス

Private — 坂寄歯科医院 内部利用
