# Install DriveVLA

## 1. Python仮想環境のセットアップ

```shell
# venv環境を作成（Python 3.10が必要）
python3 -m venv .venv

# 仮想環境をアクティベート
source .venv/bin/activate

# pipをアップグレード（PEP 660サポートを有効化）
pip install --upgrade pip

# PyTorch 2.1.2をインストール（CUDA 11.8版）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# pip install -e ".[train]"
```

**注意**:

- Python 3.10以上が必要です
- CUDA対応GPUが必要です（CUDA 11.8以降を推奨）
- この手順では`python3 -m venv`を使用します（conda不要）

## 2. transformersとLLaVA依存関係のインストール

```shell
# pyproject.tomlで指定されたバージョンのtransformersとaccelerateをインストール
pip install "tokenizers~=0.15.2" "accelerate==0.29.3" "git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4"

# flash-attnをインストール
pip install flash-attn==2.5.7 --no-build-isolation
```

## 3. mmcv-fullをプリビルドwheelからインストール

PyTorch 2.1.2とCUDA 11.8用のプリビルドwheelを使用します（ソースからのビルドよりも高速で信頼性が高い）：

```shell
# mmcv-fullをインストール
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# NumPyを1.x系にダウングレード（互換性のため）
pip install "numpy<2"

# mmcv._extが利用可能か確認
python -c "import mmcv; from mmcv import _ext; print('mmcv._ext available!')"
```

**注意**:

- CUDA 12.3を使用している場合でも、cu118版のwheelをインストールしてください（PyTorch 2.1.2がcu118でビルドされているため）
- 異なるCUDAバージョンの場合は、<https://download.openmmlab.com/mmcv/dist/> から対応するwheelを探してください

## 4. Install mmdet and mmseg

```shell
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0 motmetrics==1.4.0 casadi==3.6.0
```

## 5. mmdet3dをソースからインストール

```shell
# プロジェクトのルートディレクトリにいることを確認
# cd /path/to/OpenDriveVLA

# 依存関係をインストール
pip install scipy==1.10.1 scikit-image==0.19.3 fsspec

# wheelとsetuptoolsをインストール（必要な場合）
pip install wheel setuptools

# mmdet3dをインストール（--no-build-isolationオプションが必要）
cd third_party/mmdetection3d_1_0_0rc6
pip install --no-build-isolation .
cd ../..  # プロジェクトルートに戻る

# NumPyバージョンの修正（互換性のため）
pip install "numpy<2"
```

**注意**:

- mmdet3dのインストールには`--no-build-isolation`オプションが必要です。これにより、現在の仮想環境のパッケージ（特にtorch）を正しく参照できます。
- インストール後、NumPy 2.xがインストールされている場合は1.x系にダウングレードする必要があります（互換性のため）。

## 6. CARLA Leaderboard 2.1のセットアップ（評価用）

Leaderboard評価を実行する場合、以下の追加セットアップが必要です：

### (1) CARLA追加マップのインストール

Town12などの追加マップをCARLAにコピーします：

```shell
# AdditionalMaps_0.9.16が存在する場合
cp -r ~/AdditionalMaps_0.9.16/CarlaUE4/Content/Carla/Maps/Town12 ~/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/
```

### (2) Leaderboard 2.1のPython 3.10互換性修正

Python 3.10では`getchildren()`メソッドが削除されているため、Leaderboardコードを修正します：

```shell
sed -i 's/scenario\.getchildren()/list(scenario)/g' ~/CARLA_0.9.16/leaderboard/leaderboard/utils/route_parser.py
```

## 7. トラブルシューティング

### エラー: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

OpenCVやmatplotlibを使用する際に発生する可能性があります:

```shell
sudo apt-get update && sudo apt-get install libgl1
```

### エラー: `libgfortran.so.5: cannot open shared object file: No such file or directory`

scipyやnumpyの数値計算ライブラリで発生する可能性があります:

```shell
sudo apt-get install libgfortran5
```

### 仮想環境の再アクティベート

新しいターミナルセッションを開始した場合、仮想環境を再度アクティベートする必要があります:

```shell
cd /path/to/OpenDriveVLA
source .venv/bin/activate
```

### インストールの確認

すべてのパッケージが正しくインストールされたか確認:

```shell
python -c "import torch; import mmcv; import mmdet; import mmdet3d; print('All packages imported successfully!')"
```
