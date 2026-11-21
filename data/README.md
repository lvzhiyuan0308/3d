# 输入图片

本目录包含本次人物重建使用的原照片及合成参考图。路径、原素材文件名、字节数与 SHA-256 记录于 `manifest.json`。

| 文件 | 用途 |
| --- | --- |
| `original.jpg` | 原始人物照片，提供单视角脸部外观 |
| `turnaround.png` | 四视图板，默认配置裁切 front / left / right / back |
| `references/detailed_character_board.png` | 综合细节参考板 |
| `references/turnaround_head_clothing_pbr_board.png` | 三视图、头部、服装和材质参考板 |
| `references/clean_layout_board.png` | 整理后的参考板 |
| `references/pbr_appearance_reference.png` | 材质外观参考 |
| `references/a_pose_t_pose_reference.png` | A / T 姿态参考 |
| `references/clothing_and_shoes_details.png` | 服装与鞋履细节参考 |
| `references/head_details.png` | 头部多角度参考 |

除原照片外，其余图片为生成参考，不是实际拍摄或相机标定的多视角测量，也不是精确材质扫描。额外细节板作为建模参考保存，当前自动拟合主要使用 `turnaround.png`。

实际网格的预览位于 `docs/images/v2/`。这些是之前 V2 的输出，不代表补丁修改后重新拟合出的新人物效果。

更换人物时请配置自己的输入路径和新的 `work_dir`，避免复用当前缓存。原始素材及人物肖像的权利不因代码仓库公开而被额外授予。
