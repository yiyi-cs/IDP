# IDP pipeline 整合代码包（2026-10-18）

基于 yiyi-cs/IDP 的 Blink-Detection 分支，提交 fb767fb。
当前 GitHub 连接只有读取权限，因此没有修改远程分支。

## 安装（二选一）

1. 覆盖方式：先备份本地代码，把本包的 Skripte/ 合并到仓库的 Skripte/，覆盖同名文件。
   本包仅包含新增/修改的文件，不是整个仓库，不要删除原来的其他文件。
2. Git 补丁方式：在该版本仓库根目录执行：
   git apply --check /补丁所在路径/integration.patch
   git apply /补丁所在路径/integration.patch

两种方式不要重复执行。如果本地代码已继续修改，优先先检查补丁能否适用。

## 本次内容

- 第 10 步自动运行 blink 导出、严格 frame mask 合并、fixation、EyeLink sanity check 和两份 PDF。
- 规则中的 10 个文件统一输出到当前 Run/extra/。
- EyeLink 和 Webcam 共用原 detector，独立估计 P75。
- 区间比较使用同步 gaze 时间。PTS 仅用于定位眼部截图。
- 保留第 11 步比较逻辑及缓存开关。
- 修复 master_cli.py 退出后重复启动的问题。
- blink_extra/、核心 detector 和算法阈值没有改动。

## 验证与下一步

已通过 8 项边界测试及合成单 Run 烟雾测试，生成并检查两份 PDF 的版式。
合成测试替代了视频推理、benchmark 导出与截图，不能代替真实 VP 验收。
当前环境没有实际视频/ASC 数据及 MediaPipe/OpenCV，尚未跑完真实 VP。

请先使用新 Run 跑 1 个 VP，核对结果后再批量处理 11 个 VP。
详细命令见 Skripte/tests/INTEGRATION_ACCEPTANCE.md。

在 Skripte/ 下启动完整 pipeline：
python master_cli.py --data-root /Volumes/Empra10/Ergebnisse60 --video-root /Volumes/Empra9/Videos_60

选择单个 VP、60 Hz、FullCalib、mode 2。首次验收关闭缓存。
