# MediaPipe 0.10.14 离线备份

本目录保存官方 **Windows x64 / CPython 3.11** 安装包，以及本项目实际使用的四个 TFLite 模型和两个计算图。安装包包含完整 MediaPipe Python 包、Windows 原生扩展及其包内资源。

| 路径 | 内容 |
| --- | --- |
| `wheels/mediapipe-0.10.14-cp311-cp311-win_amd64.whl` | 未修改的官方 Windows 安装包 |
| `models/pose_detection.tflite` | 人体检测 |
| `models/pose_landmark_full.tflite` | 人体姿态关键点 |
| `models/face_detection_short_range.tflite` | 人脸检测 |
| `models/face_landmark.tflite` | 脸部关键点 |
| `graphs/*.binarypb` | Pose / FaceMesh 的 CPU 计算图 |
| `LICENSE`、`METADATA`、`WHEEL` | 从安装包原样保存的许可及元数据 |
| `manifest.json` | 下载来源、所有备份文件大小和 SHA-256 |
| `expected_resources.json` | 原本地 MediaPipe 0.10.14 资源指纹 |

文件由 `Vendor MediaPipe offline assets` 工作流从官方 PyPI 获取。下载包先对照 PyPI 的 SHA-256，再核对六个资源与本地已验证版本是否完全一致；不安装或执行下载包中的 Python 代码。

## 在 Windows 使用

完成项目其他依赖准备后，可直接使用仓库安装包：

```powershell
uv pip install --python .venv\Scripts\python.exe third_party/mediapipe/wheels/mediapipe-0.10.14-cp311-cp311-win_amd64.whl
```

已有其他依赖时，仅离线安装此包：

```powershell
uv pip install --python .venv\Scripts\python.exe --no-index --no-deps third_party/mediapipe/wheels/mediapipe-0.10.14-cp311-cp311-win_amd64.whl
```

完整 wheel 已包含模型和计算图，正常安装后会从 MediaPipe 包内加载；单独的 `models/` 与 `graphs/` 是备份副本。仓库还保留其他平台的标准依赖安装方式，Windows wheel 不适用于 Linux/macOS 或其他 Python 版本。整套离线环境仍需要准备 PyTorch、NumPy、OpenCV 等依赖包。

## 校验和更新

```powershell
python scripts/vendor_mediapipe.py --verify
```

此校验只使用 Python 标准库，不需要导入 MediaPipe 或联网。重新获取官方包可运行 `--download`，或手动执行仓库中的相应工作流。现有资源未修改；许可与第三方声明完整保留于 `LICENSE` 和官方 wheel 内。SMPL-X 权重仍按单独的获取说明在本地准备。
