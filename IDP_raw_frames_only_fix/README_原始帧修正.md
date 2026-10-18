# 原始帧 pipeline 修正

基于当前 Blink-Detection 分支 dfc0a75，撤销已放弃的 Lanczos 放大方案。
本包仅处理预处理配置与缓存，不修复已发现的时间轴错位。
当前还不要重跑完整 pipeline。

## 安装

优先使用补丁，避免再次替换整个文件夹：
在项目根目录执行（把路径换成解压后的位置）：
git apply --check /路径/raw_frames_only.patch
git apply /路径/raw_frames_only.patch

或者仅逐个合并覆盖包内 Skripte/ 的文件，不要删除或替换整个旧目录。
两种方式二选一。

## 修改

- config.py 的 PREPROCESS_MODE = "none"。
- 移除共享预处理器的放大实现，原始 frame 对象、尺寸、像素保持不变。
- 移除 MediaPipe/PTGaze 推理及校准中的 2 倍尺寸分支。
- 推理、校准、blink 和主 pipeline 拒绝非 none 的预处理参数。
- 缓存元数据读取保留 preprocess_mode，缺失信息标为 unknown。
- 复制图像相关缓存时要求来源及目标明确为 none，阻止显式选择绕过检查。
- 与图像处理无关的 EyeLink 导入仍可复用。
- 旧预处理实验入口改为原始帧契约检查。
- blink_extra/ 和核心 detector 不变。

## 检查

在项目根目录执行：
PYTHONPATH=Skripte python -m unittest discover -s Skripte/tests -v

13 项测试已在开发环境通过，补丁在干净的本地基线副本中适用并通过语法检查。
没有执行真实视频推理。

## 已有 beo7 Run

本次 Run 确认使用了 lanczos_2x，不能通过改 JSON 标签变成原始帧结果。
请保留已有 Run、extra 和旧 test60，用于后续时间轴核对。
旧、新日志共 29098 帧，valid mask 仅 40 帧不同（99.8625% 一致）。
这说明数秒级图形错位不能仅归因于放大方案。
时间轴修复和真实结果验收仍未完成。
