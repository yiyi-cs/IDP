# 音频同步修复与验证边界

基于远端 Blink-Detection 提交 c045a2c（2026-10-18）。

## 修改内容

- 880 Hz 不再把 Hilbert 包络的最大值直接当成音调中心。对已检测候选，使用 820–940 Hz 带通、100 Hz 包络平滑、局部峰值 20% 的起始边沿，再加 100 ms。已知音调长度为 200 ms。边界截断或缺乏持续信号的候选被拒绝。440/1760 Hz 的检测取时保持原逻辑。
- JSON 预测不再冒充观测到的音频锚点。440 Hz 弱信号回退结果不再被第二次普通搜索覆盖。修正录制开头被裁剪搜索窗口的局部预测时间。
- 分别保存实际 440、880、1760 Hz 偏移统计。两组 trial 偏移中位数相差超过 20 ms 时，不再混合，优先使用已验证较稳定的 440 Hz 家族，并记录被排除的家族。20 ms 是当前 QC 策略阈值，不是绝对同步误差保证。
- 标准差写入 `anchor_std_ms`，不再冒充 clock drift。未拟合回归时 `r_squared`、`drift_ms` 导出为 null。保留旧 confidence 分数用于下游兼容，同时明确它不代表绝对同步验证。
- `sync_info.quality` 和 debug_4 结果保留未验证的采集音画延迟、实验日志字段语义、未拟合漂移等信息。两个 PDF 增加绝对音画延迟尚未独立验证的备注。
- 拒绝旧音频同步版本的缓存。模型窗口和下游缓存还必须与本次 phases 完全一致，防止新同步结果混入旧窗口或时间戳。EyeLink 原始数据缓存不受影响。
- 没有增加 Lanczos，没有添加 175 ms 或125 ms 的 gaze 补偿，没有修改 blink/fixation 核心算法。

## 已验证的范围

1. 合成 200 ms 音调：不同载波相位、幅度和噪声下，880 Hz 中心误差小于 12 ms。静音、缺失匹配、边界裁剪、频率家族冲突及缓存污染均有回归测试。
2. 用户提供的 beo7 视频 190–220 s 音频，覆盖 trial 1–3。调用实际生产 FrequencyDetector 后，440 Hz 观测保持原值，880 Hz 新中心分别为 195.176009、205.087279、214.932608 s。
3. 这三个 trial 的 440 Hz 偏移中位数为 11560438.686927 ms，修正后 880 Hz 为 11560439.637537 ms，两者相差约 0.95 ms。合并诊断偏移为 11560439.543167 ms，相较旧值增加约 49.095 ms。
4. 只验证了 50 个 trial 标记中的 6 个。该值不是整段录制的新正式偏移，也不能说明剩余 gaze 时间差的来源。38 项已有及新增测试全部通过，详见包内验证日志。两种 PDF 用合成数据生成并检查布局。

## 安装

ZIP 中 `Skripte/` 与仓库同名目录合并覆盖。无额外 Python 依赖，继续使用原环境中的 NumPy、SciPy 和 FFmpeg。

若 ZIP 位于 Downloads，可以运行：

```bash
unzip -o ~/Downloads/IDP_audio_sync_fix_c045a2c.zip 'Skripte/*' -d /Users/yiyi_mac/IDP_Code
```

## 下一步：只复核整段音频，不跑模型

在 `/Users/yiyi_mac/IDP_Code/Skripte` 运行：

```bash
RUN="/Volumes/Empra10/Ergebnisse60/beo7/Analyse/Run_60hz_FullCalib_mode2_20261019_005328"
LOG="$(find /Volumes/Empra10/Ergebnisse60/beo7 -type f -name experiment_sync_log.json -print -quit)"
python -m debug.audit_audio_sync \
  --phases "$RUN/phases_detected.json" \
  --log "$LOG" \
  --video /Volumes/Empra9/Videos_60/beo7.mp4 \
  --output-dir "$RUN/extra/audio_sync_audit_v1"
```

`LOG` 必须找到原实验日志。命令只提取音轨、重新定位已观测到的 880 Hz 锚点，并保留原来观测到的440 Hz 中心。它不会加载模型、修改现有 gaze/blink/fixation 数据或自动应用新的同步偏移，也不会自动生成一份声称已完成校准的新 Run。输出目录须为新目录，重复执行时可改为 `audio_sync_audit_v2`。

输出 `audio_anchor_audit.csv` 和 `audio_sync_quality.json`。先核对是否覆盖全部 50 个标记、是否仍有频率家族冲突，再决定现有结果的重新同步与下游重算。现有 gaze 的检测窗口是否覆盖新边界、校准是否仍适用也需要检查，不能仅改一列时间就宣布全部结果已修复。

## 同步限制如何写入报告

原硕士报告第17–18、23–24、57、59页明确描述既有同步限制，尤其部分60 Hz录制的音频标记受损。当前修复解决可复现的音频取点和状态记录错误。绝对采集音画延迟、日志时间戳的物理含义及剩余 gaze 相对偏移仍未经独立验证。互相关最佳移动量是诊断量，不作为自动校正值。缺少采集源代码时，无法承诺完全恢复所有采集时序。

## 本地测试

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

`test_audio_sync.py` 以 AST 提取实际生产函数，避免为了音频逻辑测试加载旧模块的可选 CV/Librosa 环境。完整视频、实际模型初始化和所有 VP 的端到端运行尚未在本环境复测。
