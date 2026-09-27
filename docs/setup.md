# 起動・検証方法

## 1. 合成デモでアプリを試す

Python 3.11、Gitを用意し、リポジトリを取得してルートで操作します。Windows PowerShellの手順は[README](../README.md#手元で試す)に記載しています。

Linuxでは次のように起動できます。依存ライブラリの対応OS・wheelが必要です。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
export SEMANTIC_SEARCH_ENABLED=0
export RECONSTRUCTION_DEVICE=cpu
python scripts/seed_local_demo.py
python -m uvicorn spatial_memory.api:app --host 127.0.0.1 --port 8001
```

ブラウザで `http://127.0.0.1:8001/?scene=00000000-0000-4000-8000-000000000001` を開きます。APIの動作確認は `http://127.0.0.1:8001/health`、API仕様は `/docs` から行えます。

デモはコードで生成した点群・仮画像・6件の物と収納を使用します。「充電ケーブルどこ？」「父の眼鏡どこ？」を試せます。モデルのダウンロードや実部屋の画像は不要です。`seed_local_demo.py` の再実行は同じデモの登録内容を作り直します。追加した登録の削除を避けるには `--no-reset` を指定してください。ただし、既定の6件と生成モデルは再作成されます。

`pip install -e .` だけでは依存関係を導入しません。`requirements.txt` とPyTorchも必要です。初期デモでもAPIがPyTorchをimportします。

## 2. 意味検索を使う

`SEMANTIC_SEARCH_ENABLED=1` に変更し、サーバーを再起動します。既定モデルは `intfloat/multilingual-e5-small` です。初回利用時にはインターネット接続とモデル取得が必要です。

意味検索のスコアは「正しい確率」ではありません。閾値・候補間の差で曖昧さを判定します。モデルが利用できない場合はキーワード検索へフォールバックします。

`/ask` の通常回答は保存済みの物・収納・段・左右などから定型文を組み立てます。任意のOpenAI互換LLM連携コードも含みますが、基本の検索デモにLLMサーバーやAPIキーは不要です。

## 3. 本物の動画から部屋を生成する

こちらは開発向けの追加構成です。合成デモの起動だけでは実写のフル生成は動きません。

| 要素 | 準備 |
| --- | --- |
| GPU | 利用するGPU・CUDAに対応したPyTorch環境 |
| 動画・音声 | ffmpegをPATHに設定 |
| VGGT | `scripts/setup_reconstruction_backends.py` で外部ソース・モデルを取得 |
| 家具認識 | `scripts/setup_image_models.py` でGrounding DINO / SAM 2を取得 |
| SpatialLM | `scripts/download_spatiallm.py` でモデルを取得し、SpatialLMの別環境を構築 |

取得前に[第三者技術の利用条件](../THIRD_PARTY.md)を確認してください。重み・外部モデルのソースはリポジトリに同梱していません。

現状の `scripts/run_spatiallm.py` はWindowsからWSL Ubuntuを呼び、`third_party/spatiallm-env` などを利用する構成です。`setup_spatiallm_wsl.sh` も既存の実験環境を前提にしています。SpatialLM側とアプリ側ではtransformersの要求バージョンが異なるため、別環境が必要です。これらは汎用の一括インストーラーとしての検証を終えていません。

`Dockerfile` / `compose.yaml` はGPU開発向けです。SpatialLM用WSL環境の構築まで代行するものではありません。まず合成デモでAPI・UIを確認してください。

スマートフォンのブラウザで撮影する場合はHTTPSが必要です。初期手順は自分のPCのlocalhostにのみ公開します。外部に公開する場合は、認証・アクセス制御・保存画像の扱いを別途設定してください。

## 4. テスト

モデル・実写データを使わないPythonテストと、Node.js標準テストランナーのUIロジックテストを含みます。

```powershell
$env:SEMANTIC_SEARCH_ENABLED = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/*.test.mjs
```

Pythonテストで使う一時画像・点群・DBはテスト内で生成します。`scripts/browser_*` などの一部の手動・ブラウザ検証には、自分で用意する動画・音声やブラウザ環境が必要です。旧環境の実写fixtureは公開版から除いています。

公開版の検証結果は[検証記録](validation.md)に記載します。新規環境での依存インストール、Linux、GPUを使ったフル生成を今回すべて再検証したわけではありません。

## 保存場所

通常は `data/` にDBや生成結果が保存されます。`SPATIAL_DB`、`SCENE_ROOT` で変更できます。モデルは `checkpoints/`、外部ソース・環境は `third_party/` を使います。これらと実写データはGitの対象外です。
