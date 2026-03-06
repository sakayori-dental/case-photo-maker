# 歯科症例写真 自動合成アプリ — Claude Code 設計書

## プロジェクト概要
5枚の歯科口腔内写真 → Claude Vision AI自動分析 → 5パネル合成画像（PNG + WebP）出力の Streamlit アプリ。

## 起動
```bash
cd ~/書類/AI_projects/case_photo_maker
streamlit run app.py
```

## ファイル構成
```
├── CLAUDE.md           # この設計書
├── README.md           # GitHub用README
├── app.py              # メインアプリ
├── prompt.txt          # Claude Vision APIに送るプロンプト（1ブロック）
├── requirements.txt    # 依存ライブラリ
├── .gitignore          # Git除外設定
└── reference/          # 参考画像（.gitignore対象、ローカルのみ）
    ├── 1.png           # 完成形: 医院名フッター版
    ├── 2.png           # 完成形: SNSオーバーレイ版
    ├── DSC_0044.JPG〜DSC_0050.JPG  # 元写真5枚
    └── cropped/        # 手動クロップ済み中間ファイル
```

## 完成形の仕様（reference/1.png, 2.png を必ず目視確認）

### レイアウト
- 最終出力: 1920×1080px（16:9）
- 5枚の縦長パネルを隙間なし横並び
- 各パネル: 384×1080px（アスペクト比約 1:2.81）
- 背景色: `#0f0f0f`

### 2つの出力モード
- **モードA（医院名フッター）**: 半透明黒帯 + 白ライン + 中央テキスト
- **モードB（SNSオーバーレイ）**: 右下にハンドル + ロゴPNG（フッター帯なし）

## 元写真 → 完成形の変換（最重要）

### 元写真の特徴
- 一眼レフ横長（約3:2）、歯列が横方向に並んでいる
- 口腔内全体（口唇・舌・歯肉・ミラー等）が写っている

### AIが行う処理
1. **回転**: 歯列主軸が垂直になるよう回転（通常80〜100°）
2. **クロップ**: 臼歯3-4本中心に、歯と歯肉のみ残す大幅クロップ
3. **ズーム統一**: 全5枚で歯1本あたりの見かけサイズを揃える

### AIプロンプト
`prompt.txt` に完全なプロンプトを格納。app.py から読み込んで使用。
プロンプト修正時は prompt.txt だけ編集すればOK。

## 画像処理パイプライン
```
1. Image.open() → EXIF orientation 自動補正
2. img.rotate(-rotation_cw_deg, expand=True, resample=BICUBIC)
3. 回転後画像に crop 座標を適用
4. ImageOps.fit() で 384×1080 にリサイズ
5. 5パネルを横並びで canvas に paste
6. フッター or オーバーレイ描画
```

## UI構成（Streamlit）
- サイドバー: APIキー / 出力モード / 医院名 or SNSハンドル / ロゴ / 出力サイズ
- メイン: 5枚アップローダー → プレビュー → AI処理ボタン → 結果表示 → 手動微調整 → DLボタン

## フォント
macOS ヒラギノ → Linux Noto Sans CJK → Windows メイリオ の順で自動検出。
見つからない場合は Pillow デフォルト + 警告。

## デバッグ
- reference/ の5枚をアップロードしてテスト
- 「AI分析結果」expander でJSON確認
- 完成形（1.png）と出力を並べて目視比較
- prompt.txt の修正→再実行で精度チューニング

## 今後の拡張候補
- [ ] バッチ処理（複数症例一括）
- [ ] Google Drive連携保存
- [ ] テンプレート保存
- [ ] 上顎/下顎・前歯/臼歯の自動判定
