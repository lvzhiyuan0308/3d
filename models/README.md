# 外部模型权重

本仓库不分发 SMPL-X 权重。其附带许可包含 **No Distribution** 条款；按仓库拥有者本次选择，权重继续保留本地。

请从 [SMPL-X 官方项目](https://smpl-x.is.tue.mpg.de/)自行获取符合使用条件的模型，并在 `config.local.toml` 的 `paths.smplx_model` 中配置本地路径。

`required_assets.json` 只记录本次实验所用 `SMPLX_FEMALE.npz` 的文件名、大小和 SHA-256，用于核对版本，不包含模型参数。代码要求 UV、脸部静态标志点和 55 关节等数组，运行 `doctor` 可检查这些字段。

`.gitignore` 继续排除 `.npz`、`.pkl`、`.pth` 等权重与拟合缓存。输入图像和预览图已单独加入版本管理。
