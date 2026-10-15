# IDP Eye Tracking --- Fixation Detection Handover

**日期：2026-10-12**\
**GitHub：** `yiyi-cs/IDP`，检查分支 `Blink-Detection`。\
**用途：** 新对话快速接手 Fixation 代码、结果核查、Report
文字与可视化。\
**证据范围：** 已重新阅读 GitHub 中
`fixation_detector.py`、`run_fixation_detection.py`、`run_final_validation.py`、`analysis_pipeline.py`。本次**无法直接读取用户
Mac 外置硬盘**，因此历史数值属于先前运行记录，最终应从本地 CSV/JSON
复核。

## 1. 当前状态

  -----------------------------------------------------------------------
  项目                                状态
  ----------------------------------- -----------------------------------
  X-only adaptive I-DT 算法           已实现，GitHub 已确认

  11 VP 独立 PTGaze 验证              历史运行记录：11/11
                                      成功；本次未复核本地文件

  主 Pipeline Step 10                 已接入代码，GitHub 已确认

  FullCalib 端到端成功                尚未确认

  Report                              可以先写，不必等集成验证

  MediaPipe 11 VP 最终 Fixation       尚不能由 PTGaze 独立验证推断
  metrics                             
  -----------------------------------------------------------------------

**重要：独立 final validation 使用 PTGaze calibrated gaze；集成 runner
支持 MediaPipe 和 PTGaze 两种方法。两者结果不能混为一谈。**

## 2. Report Outline（已经商定）

1.  Introduction
2.  System Overview
3.  Methodology
    -   3.1 Blink and Invalid-Eye-Frame Detection
    -   3.2 Fixation Detection
4.  Evaluation and Results
    -   4.1 Experimental Setup and Evaluation Metrics
    -   4.2 Blink Detection Results
    -   4.3 Fixation Detection Results
5.  Discussion and Conclusion

目标正文约 12--15 页。Methodology 只写算法和公式，Evaluation/Results
写评价定义、实验数据和图，避免重复。

## 3. 目录与数据

``` text
本地代码：/Users/yiyi_mac/IDP_Code/
60Hz 视频：/Volumes/Empra9/Videos_60/<VP>.mp4
25Hz 视频：/Volumes/Empra9/Videos_25/<VP>.mp4
60Hz 结果：/Volumes/Empra10/Ergebnisse60/<VP>/
```

11 VP：

``` text
beo7 bjs4 egf5 fbn6 fgt6 jkl7 kdn8 kro3 mhe9 oem4 ogt7
```

beo7 原始正式 gaze run 示例：

``` text
/Volumes/Empra10/Ergebnisse60/beo7/Analyse/Run_60hz_FullCalib_mode2_20260119_160844/
```

Blink 独立导出：

``` text
/Volumes/Empra10/Ergebnisse60/<VP>/test60/
  blink_filter_frame_log.csv
  blink_filter_regions.csv
```

## 4. 核心代码（GitHub 已核实）

### `Skripte/fixation/fixation_detector.py`

核心数据结构：

``` python
@dataclass
class FixationConfig:
    smoothing_window_ms: float = 100.0
    dispersion_window_ms: float = 100.0
    min_fixation_duration_ms: float = 100.0
    adaptive_percentile: float = 75.0
    max_gap_ms: float = 50.0

@dataclass
class FixationResult:
    fixations: pd.DataFrame
    threshold_deg: float
    noise_distribution: np.ndarray
    config: FixationConfig
```

主要函数：

``` text
centered_time_median
get_valid_mask
build_valid_segments
estimate_adaptive_threshold
detect_segment_fixations
detect_fixations
```

**技术流程：**

``` text
Calibrated gaze X + timestamp_ms_synced + valid_eye_frame
          ↓
validity mask（还检查 detected / plausibility_check，如存在）
          ↓
按 trial 分组；invalid 样本排除
          ↓
>50 ms invalid gap 切断 segment（短 gap 可以跨越）
          ↓
100 ms centered time-median smoothing
          ↓
已知 fixation phase 的 100 ms X-dispersion 噪声分布
          ↓
participant/method-specific P75 threshold
          ↓
X-only I-DT，最短 100 ms，逐步延长事件
          ↓
Fixation events
```

核心公式： \[
D_x(W)=`\max`{=tex}*{i`\in `{=tex}W}x_i-`\min`{=tex}*{i`\in `{=tex}W}x_i
\] \[ T\_{`\mathrm{adaptive}`{=tex}}=P\_{75}({D_x(W):
W`\text{ in known fixation phases}`{=tex}}) \]

不是 X/Y 联合 I-DT，也没有实现完整 saccade classifier。

### `Skripte/fixation/run_final_validation.py`

**独立 11-VP final validation：只针对 PTGaze。**

逐 VP： 1. 搜索原始 `Run_60hz_FullCalib_mode2_*`（排除名称包含
`BlinkFiltered`、`PTS` 的衍生 Run）。 2. 加载
`debug_5_ptgaze_calibrated.csv` 和 EyeLink `debug_3_fixations.csv`。 3.
加载 `<VP>/test60/blink_filter_frame_log.csv`，按
`frame == frame_number` merge `valid_eye_frame`。 4. 调用相同的
`detect_fixations()` 和冻结参数。 5. 将事件和 metadata 写回所选原始
Run。 6. 与 EyeLink native fixation event 做 trial 内时间区间比较。 7.
汇总 11 VP CSV/JSON。

**注意：代码选择排序后的第一个符合条件的 Run，不一定是最新
Run。必须检查汇总 CSV 的 `run` 列。**

运行：

``` bash
python Skripte/fixation/run_final_validation.py \
  /Volumes/Empra10/Ergebnisse60
```

代码规定的汇总输出：

``` text
/Volumes/Empra10/Ergebnisse60/final_fixation_validation.csv
/Volumes/Empra10/Ergebnisse60/final_fixation_validation_summary.json
```

每 VP 原始 Run 输出：

``` text
debug_6_ptgaze_fixations.csv
debug_6_fixation_metadata.json
```

这些是**代码目标路径**，本次尚未直接确认文件是否存在。

### `Skripte/fixation/run_fixation_detection.py`

**正式集成 runner，支持 MediaPipe / PTGaze。**

输入：

``` text
debug_5_pupil_data_calibrated.csv
debug_5_ptgaze_calibrated.csv
```

需要：

``` text
timestamp_ms_synced
phase_type
gaze_deg_x_calib
valid_eye_frame
trial_number 或 trial_assignment
```

通过 `PIPELINE_OUTPUT_BASE_DIR` 获取 Run 文件夹，`PIPELINE_METHODS` 选择
`mediapipe`、`ptgaze` 或 `both`。不依赖独立 Blink CSV。

预期输出：

``` text
debug_6_mediapipe_fixations.csv
debug_6_mediapipe_fixation_metadata.json
debug_6_ptgaze_fixations.csv
debug_6_ptgaze_fixation_metadata.json
```

手动调用示例：

``` bash
python Skripte/fixation/run_fixation_detection.py \
  /Volumes/Empra10/Ergebnisse60/beo7/Analyse/<RUN_FOLDER>
```

注意旧的 20260119 Run 可能缺 `valid_eye_frame`，不能假设可直接作为集成
runner 输入；建议使用已具备新字段的 Run 或复制测试目录。

### `Skripte/utils/analysis_pipeline.py`

已经存在 Step 10/11：

``` text
debug_5 calibration
      ↓
[10/11] fixation_detection
      ↓
[11/11] debug_6_extended_comparison
```

`_run_fixation_detection()` 调用
`Skripte/fixation/run_fixation_detection.py`，超时 300 s，失败时主
Pipeline 报错。

**尚未确认一次真实 FullCalib end-to-end 运行成功**，也未确认 cache 对新
Step 的处理及 debug_6 是否消费新 fixation events。

其他相关代码：

``` text
Skripte/master_cli.py
Skripte/utils/vp_data_manager.py
Skripte/debug/debug_1_video_analysis.py
Skripte/debug/debug_4_synchronization.py
Skripte/debug/debug_5_partial_calibration.py
Skripte/shared/shared_blink_detection.py
Skripte/shared/shared_pupil_detection.py
blink_extra/export_blink_filter_logs.py
blink_extra/generate_blink_diagnostics.py
blink_extra/fixation_diagnostics.py
```

## 5. Evaluation 定义（代码已核实）

`run_final_validation.py` 对**每个 EyeLink native fixation**，在同一
trial 内寻找 IoU 最大的检测事件；若无重叠则 best IoU = 0。

\[ IoU(A,B)=`\frac{|A\cap B|}{|A\cup B|}`{=tex} \]

`eyelink_overlap_pct` = 有至少一个非零 IoU 匹配的 EyeLink events
比例。**它不是 accuracy，也不是连续时间覆盖率。**

`mean_best_iou` = 所有 EyeLink fixation 的 best IoU 平均（包括 0）。

还输出：

``` text
median_best_iou
iou_ge_0_3_pct
iou_ge_0_5_pct
iou_ge_0_7_pct
```

匹配不强制 one-to-one；同一个预测事件可能是多个 EyeLink events
的最佳匹配。总体指标是对每 VP 指标取非加权均值。

## 6. 历史已有结果（需 CSV/JSON 复核）

先前记录显示 final 11-VP validation：

``` text
Successful: 11/11
Mean EyeLink event overlap: 94.06%
Mean best IoU: 0.391
Mean detected fixations / VP: 676.2
Mean adaptive threshold: 1.945°
```

这些来自历史进度记录，**不是本次直接读取结果文件所得**；报告定稿前需复核。尤其不能把
94.06% 称为 fixation classification accuracy。

早期参数探索（**不是 final P75 结果**）：

``` text
beo7 P80 T=1.86°, mean best IoU=0.426, IoU>=0.5 36.3%
bjs4 P80 T=0.93°, mean best IoU=0.424, IoU>=0.5 37.0%
kdn8 P80 T=1.01°, mean best IoU=0.394, IoU>=0.5 32.1%
kro3 P80 T=2.30°, mean best IoU=0.391, IoU>=0.5 28.8%
mhe9 P90 T=1.48°, mean best IoU=0.425, IoU>=0.5 33.6%
oem4 P75 T=2.00°, mean best IoU=0.329, IoU>=0.5 27.0%
```

beo7 早期 X-only 固定阈值 sweep：

``` text
4°: 797 events, mean duration 226.9 ms, median 181.8 ms
5°: 682 events, mean duration 306.1 ms, median 250.0 ms
6°: 562 events, mean duration 395.6 ms, median 295.5 ms
EyeLink native: mean duration 296.8 ms, median 247.5 ms
```

这些用于解释调参思路，不可混入 frozen final validation table。

## 7. 时间同步 / Blink 注意事项

beo7 nominal 60 Hz 视频为 VFR，平均约 44 FPS（29,099 frames / 661.345
s）。Blink export 使用实际 frame PTS；原 gaze pipeline 的
`timestamp_ms_synced` 则已经基于原有 audio/video synchronization。

**错误实验：** `new timestamp_ms_synced = PTS + old offset` 曾造成
gaze/EyeLink 错位。不要再次直接替换。

正确做法：

``` text
Blink mask: frame_number
Gaze: frame
→ 按 frame merge
→ gaze timestamp_ms_synced 保持原样
```

正式集成 runner 应从 `debug_5` 直接消费 `valid_eye_frame`，不要求独立
Blink CSV；独立 final validation 因旧 Run 不含新字段才做 frame merge。

## 8. Report 优先工作

先整理**真实结果**，不需要重跑昂贵视频处理：

``` bash
ls -lh /Volumes/Empra10/Ergebnisse60/final_fixation_validation*
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
root=Path('/Volumes/Empra10/Ergebnisse60')
df=pd.read_csv(root/'final_fixation_validation.csv')
print('VPs:',df.vp.nunique(),'rows:',len(df))
print(df.to_string(index=False))
print(json.dumps(json.loads((root/'final_fixation_validation_summary.json').read_text()),indent=2))
PY
```

报告可视化建议： 1. Overall architecture diagram。 2. 11 VP
表格：threshold、EyeLink/detected event counts、duration、overlap、best
IoU、IoU≥0.5。 3. Per-VP mean best IoU bar chart。 4. Per-VP EyeLink
event overlap chart。 5. 选取实际数据中好/差的 representative trial
fixation timeline。 6. 可选：阈值与事件时长/IoU 的关系。

解释结果时突出：高 overlap 但中等/较低 best IoU 意味着许多 EyeLink
events 至少存在时间重叠，但事件边界/分割仍不一致。

## 9. 后续工程任务

1.  先核实 final CSV/JSON，完成 report 正文与图。
2.  检查 Step 10 真实输入是否包含 `valid_eye_frame`，尤其 PTGaze
    分支的字段传播。
3.  在**复制的测试 Run** 上独立调用 `run_fixation_detection.py`。
4.  核实 MediaPipe/PTGaze 输出 CSV 和 metadata、trial
    数、threshold、有效样本。
5.  最后验证 FullCalib Step 10 → Step 11 和 cache；不必重跑 11 VP。
6.  `blink_extra/fixation_diagnostics.py`
    是历史辅助文件，移动/删除前应核对依赖。

## 10. 给新对话的开场提示

> 请以这份 handover 为准接手 LMU IDP Eye Tracking 的 Fixation
> 工作，不要从零实现算法。当前 X-only adaptive I-DT 已完成，11 VP 的独立
> PTGaze final validation 在历史运行中已成功；正式 Pipeline Step 10
> 代码已写好但端到端成功尚未确认。当前优先完成 12--15 页 Report 的
> Fixation Methodology、Evaluation/Results 和图表。先核查
> `/Volumes/Empra10/Ergebnisse60/final_fixation_validation.csv` 及
> summary JSON 的真实数值，再选取代表性 trial 绘图。不要擅自修改 gaze
> synchronization，不要把 PTGaze 的独立结果写成 MediaPipe 的结果。
