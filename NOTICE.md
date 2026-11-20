# 第三方与数据说明

本项目调用 SMPL-X Python 包、PyTorch、NumPy、SciPy、Numba、OpenCV、MediaPipe、trimesh、scikit-image 与 Pillow。各依赖遵循各自的许可证和分发条款。

SMPL-X 模型文件需要用户单独取得；本仓库不附带这些权重，也不授予其额外使用权限。参考：[SMPL-X](https://smpl-x.is.tue.mpg.de/)、[SMPL-X Python 实现](https://github.com/vchoutas/smplx)。

原始照片、合成参考图、拟合检查点、含模板信息的中间网格和生成模型默认排除在 Git 提交外。使用者应自行确认输入和输出的使用范围。

`reference_numpy.py` 从更早的本地单图原型中提取数值前向部分，用于与 Torch 实现对照，不包含完整旧项目、外部权重或照片。

本仓库当前未声明统一的开源许可证；私有提交不自动授权公开分发第三方素材或模型权重。
