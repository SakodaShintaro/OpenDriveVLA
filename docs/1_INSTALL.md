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

## 2. mmcv-fullをソースからインストール

### (1) GCC: gcc>=5が必要です

システムのgccバージョンを確認:

```shell
gcc --version
```

gccが古い場合やインストールされていない場合:

```shell
# Ubuntuの場合
sudo apt-get update
sudo apt-get install build-essential

# 特定のバージョンのgccを使用する場合（オプション）
# export PATH=YOUR_GCC_PATH/bin:$PATH
# 例: export PATH=/usr/bin:$PATH
```

### (2) CUDA: MMCV演算子をGPU用にコンパイルするためCUDA_HOMEを設定

```shell
# CUDAのインストールパスを確認
ls -d /usr/local/cuda*

# CUDA_HOMEを設定
export CUDA_HOME=/usr/local/cuda-12.3  # 実際のCUDAバージョンに合わせて調整
# 例: export CUDA_HOME=/usr/local/cuda-11.8
```

### (3) mmcv-fullをソースからインストール

```shell
# プロジェクトのルートディレクトリに移動
cd /path/to/OpenDriveVLA  # 実際のパスに置き換え

# 依存関係をインストール
pip install -r third_party/mmcv_1_7_2/requirements/optional.txt

# mmcv-fullをビルドしてインストール
export CUDA_HOME=/usr/local/cuda-12.3  # 上で確認したCUDAパスを設定
cd third_party/mmcv_1_7_2
MMCV_WITH_OPS=1 pip install .
cd ../..  # プロジェクトルートに戻る
```

## 3. Install mmdet and mmseg

```shell
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0 motmetrics==1.4.0 casadi==3.6.0
```

## 4. mmdet3dをソースからインストール

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

## 5. トラブルシューティング

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
