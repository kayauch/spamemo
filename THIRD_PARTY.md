# 使用技術・出典・帰属

スパメモは、チーム開発のアプリと、公開されたモデル・ライブラリを組み合わせて制作したものです。本資料は出典と役割の整理であり、これらの成果をこのリポジトリの作者が単独で開発したという主張ではありません。

## 元となるチーム開発

元のプロジェクトは [TsukubaDevGuild / hackathon-2026-autumn-team-b](https://github.com/TsukubaDevGuild/hackathon-2026-autumn-team-b) です。この公開版は、その開発内容をポートフォリオとして整理したスナップショットです。チームの履歴を、個人が新たに制作した履歴として置き換える意図はありません。

ホーム・部屋一覧のUIには、チームの `yuu_v2` ブランチ由来の構造とスタイルが含まれます。[web/yuu-ui/README.md](web/yuu-ui/README.md) に、元コミット、元ファイル名、取り込み範囲の記録を残しています。

- `web/yuu-ui/spatial-ui.css`、`workspace.css`: 同ブランチから取り込んだスタイル。
- `web/ui-integration.css`: 現行アプリの操作・レイアウトへの調整。
- 同ブランチの独立したアプリ・サーバー・固定部屋モデルは、このUI統合には含めていません。

## 主要なAIモデル

モデルの重みはこのリポジトリに同梱していません。コードと重みでは利用条件が異なる場合があるため、取得するモデルの公式モデルカード・ライセンスを確認してください。

| モデル | このアプリでの用途 | 公式出典 |
|---|---|---|
| VGGT-1B | 画像からカメラ・深度を推定し、点群を作る基礎情報を得る | [Meta公式実装](https://github.com/facebookresearch/vggt)、[モデルカード](https://huggingface.co/facebook/VGGT-1B) |
| SpatialLM1.1-Qwen-0.5B | 点群から部屋の壁・窓・扉などを推定する | [Manycore Research公式実装](https://github.com/manycore-research/SpatialLM)、[モデルカード](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B) |
| Grounding DINO Tiny | 写真内の家具をテキスト指定に基づいて検出する | [IDEA Research公式実装](https://github.com/IDEA-Research/GroundingDINO)、[モデルカード](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| SAM 2.1 Hiera Tiny | 検出枠などを手がかりに物体領域を分割する | [Meta公式実装](https://github.com/facebookresearch/sam2)、[モデルカード](https://huggingface.co/facebook/sam2.1-hiera-tiny) |
| multilingual-e5-small | 質問と物の記録をテキスト埋め込みで比較する | [作者のモデルカード](https://huggingface.co/intfloat/multilingual-e5-small) |

VGGT-1Bの公式モデルカードは、確認時点で重みに `CC BY-NC 4.0` を表示しています。アプリのソース公開と、モデル重みを用いたサービスの利用条件は別です。[公式モデルカード](https://huggingface.co/facebook/VGGT-1B)

モデルの論文上の性能を、本アプリで測定した性能として掲載していません。モデルへの入力準備、複数の推定結果の統合、固定メッシュへの写真投影、記録と検索の連携はアプリケーション側の処理です。

## 同梱したブラウザ向けライブラリ

| 同梱ファイル | 上流 | ライセンス・注記 |
|---|---|---|
| `web/vendor/three.core.js`、`three.module.js`、`OrbitControls.js`、`PLYLoader.js` | [Three.js](https://github.com/mrdoob/three.js/tree/r180) | MIT。ローカルのThree.js本体は `REVISION = '180'`。r180の上流ライセンス全文を [three.LICENSE](web/vendor/three.LICENSE) に同梱。既存のファイル内表示も保持。 |
| `web/vendor/gaussian-splats-3d.module.js` | [Mark Kellogg / GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | MIT。既存の [gaussian-splats-3d.LICENSE](web/vendor/gaussian-splats-3d.LICENSE) を保持。比較・実験用ビューアーで使用し、通常の固定メッシュ表示とは別経路。 |

Three.jsライセンスの取得元: [r180/LICENSE](https://raw.githubusercontent.com/mrdoob/three.js/r180/LICENSE)。ライセンス確認日: 2026-09-28。

## 主な実行時依存

依存パッケージは [requirements.txt](requirements.txt) とモデル用のセットアップスクリプトに記載しています。下表は用途の概要です。ライブラリごとのライセンスは各配布元に従います。

| 技術 | 用途 | 公式資料 |
|---|---|---|
| Python / FastAPI / Uvicorn | API、非同期ジョブの起動、保存・検索処理 | [Python](https://docs.python.org/3/)、[FastAPI](https://fastapi.tiangolo.com/)、[Uvicorn](https://www.uvicorn.org/) |
| PyTorch / Transformers / Hugging Face Hub | モデルの読み込みと推論 | [PyTorch](https://pytorch.org/)、[Transformers](https://huggingface.co/docs/transformers/)、[Hub](https://huggingface.co/docs/huggingface_hub/) |
| NumPy / SciPy | 配列計算、幾何計算、最適化 | [NumPy](https://numpy.org/)、[SciPy](https://scipy.org/) |
| OpenCV / Pillow | フレーム・画像処理 | [OpenCV](https://opencv.org/)、[Pillow](https://python-pillow.github.io/) |
| Open3D / Trimesh | 点群・メッシュ処理 | [Open3D](https://www.open3d.org/)、[Trimesh](https://trimesh.org/) |
| SQLite | 物・収納・部屋などの記録 | [SQLite](https://sqlite.org/about.html) |

## 比較・研究用の経路

Depth Anything 3、VGGT-OMEGA、COLMAP、2D/3D Gaussian Splatting、音声認識、LLM接続などを試すコードも残しています。これらは通常の部屋生成・場所回答の全工程で常に使われるものではありません。必要な外部実装や重みは同梱しておらず、各セットアップスクリプトの参照先で取得します。

## 公開データと権利表示の範囲

実撮影した部屋の動画・写真、登録DB、再構成結果、モデル重み、開発環境固有の設定は公開版から除いています。

この資料は上流コード・モデルのライセンスを変更せず、チーム由来のアプリ全体に新しいライセンスを一括付与するものでもありません。
