# IDP Eye-Tracking Pipeline

MediaPipe、PTGaze 与 EyeLink 的眼动分析流水线，集成 blink 检测日志、X-only I-DT fixation 检测、EyeLink sanity check 及逐 trial 的诊断 PDF。

本文面向 `Blink-Detection` 分支，可放在仓库根目录。命令以 macOS 和外接硬盘布局为例。

> **当前重要限制：一次启动只能传入一个 `--video-root`。当 60 Hz 与 25 Hz 视频分别放在两个目录时，必须分两次启动，不能在交互菜单中选择同时分析两者。**

## 目录

- [当前版本与验收状态](#当前版本与验收状态)
- [相较 main 的变化](#相较-main-的变化)
- [安装与依赖](#安装与依赖)
- [输入数据与目录](#输入数据与目录)
- [快速运行](#快速运行)
- [流水线执行顺序](#流水线执行顺序)
- [算法与数据约定](#算法与数据约定)
- [输出文件](#输出文件)
- [新增文件与职责](#新增文件与职责)
- [已有文件的主要改动](#已有文件的主要改动)
- [单独执行与排错命令](#单独执行与排错命令)
- [测试与复现](#测试与复现)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [原项目文档与外部说明](#原项目文档与外部说明)

## 当前版本与验收状态

本说明按以下固定提交对照编写，后续分支更新时应同步修改本文。

| 项目 | 版本 |
| --- | --- |
| 原版基线 `main` | [`dd0e5b2`](https://github.com/yiyi-cs/IDP/tree/dd0e5b2f0ad7bd6924b4252cccac4af5f792537c) |
| 当前集成分支 `Blink-Detection` | [`a81ee3f`](https://github.com/yiyi-cs/IDP/tree/a81ee3fa4d8829b4cd9c26229a1f899bdc282749) |
| 说明更新时间 | 2026-10-19，Europe/Berlin |

截至本说明编写时：

- 远程 `Skripte` 下 65 个 Python／Markdown 文件与已检查的修复版本一致。
- 31 项回归测试通过，包含两个实际校准脚本入口的成功及失败路径。
- 合成 Step 10 集成测试完成，生产 fixation 核心、EyeLink 采样、统计和两份 PDF 均执行；视频模型推理、ASC 导出和 ROI 图像使用测试夹具。
- FFV1 与 H.264 变帧率视频测试通过，包含 B 帧及非零起始 PTS；窗口取帧与完整顺序解码逐像素一致。
- 用户本机 Apple M2 上的 beo7 解码、模型初始化及单帧调用预检已通过；PTGaze 日志显示 `Device: cpu`。
- **新的 beo7 全流程重跑正在进行，完整输出与科学结果尚未验收。上述检查不等于全量 VP 或最新 25 Hz 分支结果已通过验收。**

## 相较 main 的变化

原版已包含 gaze 检测、校准、EyeLink 导入及同步等组件。本分支将 blink 的完整日志导出、fixation 检测与诊断产物接入主流程。

| 方面 | 本分支的行为 |
| --- | --- |
| 主入口 | `master_cli.py` 增加 `--data-root`、`--video-root`，脚本目录按自身位置确定 |
| Blink 集成 | 自动扫描所选完整视频，保存逐帧 EAR、最终有效性、原因和合并区间 |
| Fixation 集成 | 选定 gaze 方法使用共享 X-only I-DT 核心，自动估计各自 P75 阈值 |
| EyeLink 对照 | 导出 EBLINK benchmark、保留 native fixation，并增加同一 I-DT 核心的 EyeLink sanity check |
| 可视化 | 自动生成 blink 诊断 PDF 和四行 fixation timeline PDF |
| 结果组织 | 新增中间数据和两份 PDF 统一写入各自 Run 的 `extra/`；兼容原有根目录输出 |
| 帧与时间 | 用真实视频 PTS 选择帧并建立 gaze 时间，随后沿用同步 offset 得到 `timestamp_ms_synced` |
| 图像输入 | 生产预处理仅允许 `none`；不引入已放弃的 Lanczos 放大方案 |
| 缓存 | 检查图像处理、时钟版本及该步骤相关配置，拒绝不兼容或来源不明的旧数据 |
| 失败处理 | 启动前检查解码及模型；校准失败返回非零；缺同步数据不再退回未同步输入 |

本分支没有把 `blink_extra/` 作为生产依赖。该目录保存历史实验和参考脚本，其中旧路径、参数与处理方案不代表当前生产配置。

## 安装与依赖

### 已能运行 main 的环境：额外需要什么

**新增 blink／fixation 主流程没有引入一套新的第三方 Python 包。** 它复用原版源码已经使用的 NumPy、pandas、Matplotlib、SciPy、scikit-learn、OpenCV、MediaPipe、PyTorch、PTGaze 等依赖。

需要额外确认的是系统命令 **`ffmpeg` 和 `ffprobe`**。当前代码直接调用这两个可执行程序，用于 PTS 提取、准确解码和 ROI 截图；仅安装 Python 的 `ffmpeg-python` 或 `imageio-ffmpeg` 不能替代这里的 PATH 检查。

macOS 已安装 Homebrew 时：

```bash
brew install ffmpeg
```

Homebrew 的 FFmpeg 包提供 `ffmpeg` 与 `ffprobe`。安装说明见 [Homebrew FFmpeg](https://formulae.brew.sh/formula/ffmpeg)。若已经能找到二者，无需重复安装。

```bash
command -v ffmpeg
command -v ffprobe
ffmpeg -version
ffprobe -version
```

保留当前已验证可用的 Python 环境，先运行下文预检。当前没有随仓库提供完整的版本锁定环境，不建议为了这次集成直接批量升级已有包。

### 原版环境依赖清单

下表用于核对环境，并非声明所有包都是本分支新增，也不是经过全新环境安装验收的 lockfile。

| Python 安装包 | 主要用途 |
| --- | --- |
| `numpy`、`pandas` | 数值处理、时间序列与 CSV |
| `scipy`、`librosa` | 音频标记检测和同步相关计算 |
| `matplotlib`、`seaborn` | PDF、诊断图和原有比较图 |
| `scikit-learn` | 校准回归模型；Python 导入名是 `sklearn` |
| `opencv-python` | 视频读取、图像处理；Python 导入名是 `cv2` |
| `mediapipe` | 面部关键点、虹膜和 EAR 相关处理 |
| `torch`、`ptgaze` | PTGaze gaze 模型；PTGaze 环境还需其自身依赖，例如 torchvision |
| `omegaconf`、`PyYAML`、`tqdm` | PTGaze 配置、YAML 和进度显示 |

若已有工作环境仅缺某个包，可按上表补装，例如：

```bash
python -m pip install matplotlib scipy scikit-learn
```

上面的例子只适用于确实缺少这些包的情况。MediaPipe、PTGaze、PyTorch 和 NumPy 的兼容性应以现有工作环境及预检为准，本文不虚构一组尚未验证的固定版本。旧 main 中还存在 `hmmlearn` 相关代码；它不是新增 X-only I-DT 核心的依赖。

当前生产代码依赖 `mediapipe.solutions.face_mesh.FaceMesh` 接口。安装一个名为 MediaPipe 的包并不自动证明接口与当前项目兼容，预检会实际初始化并调用相关组件。

### 模型与可选工具

- PTGaze 使用 `eth-xgaze` 模型；已有 main 的权重与相机参数继续使用，不需为 fixation 另下载模型。
- 缺失权重时，PTGaze 配置生成器会调用 PTGaze 的下载函数；首次运行可能需要网络和可写缓存目录。权重不由本 README 或修复包提供。
- 新增 PDF 由 Matplotlib 的 `PdfPages` 输出，生产流程不需要额外安装 LaTeX、ReportLab 或独立 PDF 阅读 Python 包。
- 仅在使用历史的带图片 Excel 复核脚本时，额外安装以下可选包；它们不是主 pipeline 的要求：

```bash
python -m pip install openpyxl pillow
```

`openpyxl` 写入工作簿，Pillow 支持嵌入图片，见 [openpyxl 安装说明](https://openpyxl.readthedocs.io/en/stable/tutorial.html#installation)。Excel 复核脚本目前位于参考目录，不会随主流程自动执行。

## 输入数据与目录

### 代码位置

命令中的 `/Users/yiyi_mac/IDP_Code` 是本机示例路径，请替换为自己的项目根目录。主要代码按用途分布在以下位置。

| 路径 | 内容 |
| --- | --- |
| `Skripte/master_cli.py` | 交互式主入口 |
| `Skripte/config.py` | 实验、屏幕、模型及质量配置 |
| `Skripte/utils/` | 数据发现、Run 管理、缓存及编排 |
| `Skripte/debug/` | 原有分步入口和独立研究脚本 |
| `Skripte/calibration/` | 两种 gaze 方法的离线校准 |
| `Skripte/shared/` | 检测、坐标、原图输入和统一帧时钟 |
| `Skripte/blink/` | 正式 blink 导出、统计、诊断及 Step 10 编排 |
| `Skripte/fixation/` | 正式 fixation 核心、适配、EyeLink sanity 和 timeline |
| `Skripte/tests/` | 回归、合成集成测试及验收说明 |
| `Skripte/statistic/`、`R-Skripte/` | 原有统计准备与 R 分析 |
| `blink_extra/` | 历史实验、复核工具及交接文档，非生产入口 |

### 数据位置

`--data-root` 指向包含各 VP 文件夹的目录。示例：

```text
/Volumes/Empra10/Ergebnisse60/beo7/
/Volumes/Empra10/Ergebnisse60/bjs4/
```

每个 VP 目录通常包含：

| 文件／目录 | 要求与用途 |
| --- | --- |
| `<VP><记录编号>.asc`，如 `beo71339.asc` | 完整集成模式需要，提供 EyeLink gaze、消息和事件；EDF 需先按原项目流程转换为 ASC |
| `experiment_sync_log.json` 或带同前缀的 JSON | 同步和实验结构输入；代码保留缺失时的 fallback，但新数据优先保留完整原始日志 |
| `Calibration/calibration_timing*beg*.json` | 开始阶段校准时间点 |
| `Calibration/calibration_timing*mid*.json` | 中间阶段校准时间点，视所选校准方案使用 |
| `Calibration/calibration_timing*end*.json` | 结束阶段校准时间点，视所选校准方案使用 |
| `Analyse/` | 自动生成和发现 Run 的位置 |

视频可独立放置在：

```text
/Volumes/Empra9/Videos_60/beo7.mp4
/Volumes/Empra9/Videos_25/beo7.mp4
```

程序也会优先尝试 VP 目录中的 `<VP>_60.mp4`、`<VP>_25.mp4` 等命名；25 Hz 兼容 `_24.mp4` 的旧命名。外部目录的发现主要依赖文件名包含 VP code，**不会仅凭菜单选择就可靠鉴别真实录制频率**，因此一个目录应只放当前分析条件的视频，避免同一 VP 有多个模糊匹配。

`60hz`／`25hz` 是录制条件标签，实际视频可能为变帧率；时间计算使用真实 PTS，不把名义频率作为逐帧时钟。

## 快速运行

### 1. 使用当前分支

首次获取仓库可以运行：

```bash
git clone --branch Blink-Detection https://github.com/yiyi-cs/IDP.git IDP_Code
cd IDP_Code/Skripte
```

已有项目时直接进入自己的 `Skripte` 目录，并确认使用本分支。下面的启动命令不需要手动修改 `config.py` 中用于 pipeline 覆盖的路径占位文本。

### 2. 正式运行前做模型预检

```bash
cd /Users/yiyi_mac/IDP_Code/Skripte
python -m shared.pipeline_preflight \
  --video /Volumes/Empra9/Videos_60/beo7.mp4 \
  --methods both
```

预检会检查系统命令和 Python 依赖、解码真实视频的一帧、初始化相关检测器并调用单帧接口。成功结束时出现：

```text
[OK] Preflight: raw frame, decoder, requested models and PDF imports
```

主 pipeline 开始正式阶段前也会自动执行该预检。它证明当前环境能完成基本调用，不证明整个视频的人脸检测率、校准质量或结果一致性已经验收。

### 3. 运行 60 Hz

在 `Skripte` 目录执行：

```bash
python master_cli.py \
  --data-root /Volumes/Empra10/Ergebnisse60 \
  --video-root /Volumes/Empra9/Videos_60
```

首次验证建议按下表选择。具体菜单编号以当前界面为准。

| 菜单 | 选择 |
| --- | --- |
| 主菜单 | `Analysemodus (VP-Daten analysieren)` |
| 视频 | `Nur 60 Hz` |
| 校准 | `FullCalib`，若实验使用其他方案则按实验设计选择 |
| PKL 来源 | 新生成校准，首次验收不混用旧结果 |
| 分析模式 | **模式 2：Webcam + EyeLink 比较模式** |
| 方法 | `both`，或明确选择需要的方法 |
| VP | 首先单独 `beo7` |
| 缓存 | 首次验收选择重新计算，新建 Run |

**两个“模式”菜单不要混淆：** 主菜单的 `[2] Statistikmodus` 是准备统计数据，不会代替完整 VP 检测；进入主菜单的分析入口后，后续询问的 `analysis_mode=2` 才是当前集成要求的 EyeLink 比较模式。当前完整 blink／EyeLink／fixation 集成不支持 `analysis_mode=1`，会在耗时阶段前拒绝。

### 4. 运行 25 Hz：另起一次

完成或退出上一轮后，用 25 Hz 对应的目录重新启动。例如实际文件位于下列位置时：

```bash
python master_cli.py \
  --data-root /Volumes/Empra9/Ergebnisse25 \
  --video-root /Volumes/Empra9/Videos_25
```

这次只选 `Nur 25 Hz`。`Ergebnisse25` 的磁盘位置是示例，应按实际挂载位置替换。

即使菜单仍显示 `Beide (separate Runs + Vergleich)`，在目前两个独立视频目录的布局下也不要选择它：`--video-root` 只有一个值，代码会把同一外部目录用于两种频率的查找，可能将同一个视频错误地用于两个条件。**当前没有独立的 `--video-root-60` 与 `--video-root-25` 参数。**

## 流水线执行顺序

同一个 VP／视频条件／校准方案各自生成 Run。以下为比较模式的主要步骤；PTGaze 步骤按方法选择执行，校准等步骤还可能使用通过校验的缓存。

| 顺序 | 入口／阶段 | 作用 |
| --- | --- | --- |
| 启动前 | `shared/pipeline_preflight.py` | 路径、依赖、视频解码及模型单帧调用检查 |
| 1 | `debug/debug_0_phase_detection.py` | 获取校准／实验阶段、音频标记和同步信息 |
| 2 | `debug/debug_1_video_analysis.py` | MediaPipe 视频检测与逐帧 gaze 原始数据 |
| 3 | `debug/debug_1_ptgaze.py` | PTGaze 原始 gaze 数据 |
| 4 | `calibration/offline_calibration.py` | 生成 MediaPipe 校准 PKL |
| 5 | `calibration/offline_calibration_ptgaze.py` | 生成 PTGaze 校准 PKL |
| 6 | `debug/debug_3_eyetracker_load.py` | ASC 导入、EyeLink gaze 和 native fixation |
| 7 | `debug/debug_4_synchronization.py` | 生成同步 gaze 时间并重新分配 trial／phase |
| 8 | `debug/debug_5_partial_calibration.py` | 应用 MediaPipe 校准 |
| 9 | `debug/debug_5_ptgaze.py` | 应用 PTGaze 校准 |
| 10 | `blink/run_pipeline_extras.py` | 完整 blink 扫描、benchmark、fixation、EyeLink sanity 和两份 PDF |
| 11 | `debug/debug_6_extended_comparison.py` | 保留原有 gaze 方法比较输出 |

Step 10 内部先检查校准 CSV 与真实视频时钟，然后依次导出 blink 日志、解析 EyeLink benchmark、严格验证 mask 对齐、调用共享 fixation 核心、完成 EyeLink sanity check、生成两份 PDF。

选择 `both` 时，两种 webcam gaze 都生成 fixation 结果；两份主要 PDF 以 PTGaze 为首选方法。仅选 MediaPipe 时，fixation timeline 和 sanity 对照使用 MediaPipe。不是每种方法各输出两份不同命名 PDF。

MediaPipe 的质量／blink 信息也是现有上游链路的一部分；选择 PTGaze 不表示整个流程完全不使用 MediaPipe。

## 算法与数据约定

### 图像与时钟

- 正式图像预处理只接受 `PREPROCESS_MODE = "none"`；进入检测接口的帧不做 Lanczos 放大。模型内部正常的归一化属于模型自身流程。
- 校准与实验窗口按 `ffprobe` 的真实帧时间表选择，通过 FFmpeg 准确解码；不依赖变帧率视频的名义帧号 seek 来假定帧身份。
- gaze 的 `timestamp_ms` 来自实际视频 PTS；同步阶段应用既有同步 offset，形成 `timestamp_ms_synced`。
- fixation、区间比较和诊断图均使用同步时钟；ROI 图像定位使用视频 PTS。
- Run 记录 `timebase_version = "frame_pts_v1"`。60 Hz 名义条件不会让算法忽略实际不等距采样。

### Blink 与 EyeLink benchmark

- 逐帧日志保存原始／平滑 EAR、阈值、闭眼状态、最终 `valid_eye_frame` 和原因；完整扫描保持时间状态的连续性。
- `blink_filter_regions.csv` 合并连续无效区间，可能包括 blink、suspected 和其他 invalid；不要把所有无效区间直接解释为真实生理眨眼。
- EyeLink benchmark 使用 ASC 中的 EBLINK 事件，不按 `>40 ms` 额外过滤。
- CSV 指标采用 100 ms 时间容差及一对一贪心事件匹配。ROI 示例的 `BOTH` 分类采用“EyeLink 事件中点位于 webcam 区间内”的展示规则，**与统计匹配的判据不同**，图中也注明了这一点。

### Fixation

正式核心为 `fixation/fixation_detector.py`，配置如下。

| 参数 | 当前值 |
| --- | --- |
| 信号 | 水平方向 X，单位为视角度数 |
| 平滑 | 基于时间的居中中位数，100 ms |
| 噪声／离散度窗口 | 100 ms |
| 自适应阈值 | 已知 fixation phase 的 X 离散度分布 P75 |
| 最小 fixation 时长 | 100 ms |
| 最大允许无效间隔 | 50 ms；更长间隔拆段 |

P75 按各方法当前有效信号独立估计。EyeLink sanity 也调用同一个核心并独立估计阈值，不直接复用 PTGaze 的数值阈值。正式流程不执行研究脚本中的 2°—6° sweep 或 P75／P80／P90 多方案选择。

gaze 与最终 blink mask 通过 `frame == frame_number` 一对一连接。缺帧、重复帧、无效布尔值或时钟不匹配会报错，不默认为有效。未得到校准 gaze 的帧保留为显式无效样本；不插值补造 gaze，以便核心正确处理较长缺口。

EyeLink sanity 在同一准备后的帧时间网格上，以不超过 2 ms 的距离选择最近 EyeLink 样本；缺失或无效数据保留为无效。它与完整高频 EyeLink native fixation 的对照有用途区别。

当前配置的显示器参数为 1920×1080 px、53.2×29.9 cm、观看距离 70 cm。图像坐标到视角的转换使用项目配置。换实验设备时应按真实实验参数更新，不能用调屏幕参数或同步 offset 的方式追求旧图外观一致。

### 缓存与失败处理

图像相关缓存需要明确的 `preprocess_mode: none` 和 `timebase_version: frame_pts_v1`，还会检查相关 FPS、校准模式等字段；同步／比较缓存额外区分所选方法。旧的 EyeLink 导入和音频阶段按各自条件管理，不因图像模式改变而一概作废。

自选校准 PKL 需要源目录的 `config_used.json` 提供兼容来源信息。缺少来源时应重新生成，不要手动把旧元数据改成 `none` 或 `frame_pts_v1`。显式指定的 PTGaze PKL 优先于 Run 内其他较新的文件。

校准失败、全无效校准输入、缺同步 CSV、ROI 解码失败等会明确失败。存在某个 CSV 或 PDF 不代表整个 Run 成功，完整性标记见下节。

## 输出文件

Run 路径形如：

```text
<data-root>/<VP>/Analyse/Run_<video-condition>_<calibration>_mode2_<timestamp>/
```

### 保留在 Run 根目录的主要输出

| 文件 | 内容 |
| --- | --- |
| `config_used.json` | Run 配置、方法、图像处理与时钟来源 |
| `phases_detected.json` | 校准／trial／phase 边界与同步信息 |
| `debug_1_pupil_data.csv`、`debug_1_ptgaze_data.csv` | 各方法原始检测结果，按方法执行情况存在 |
| `calibration_*.pkl`、`calibration_ptgaze_*.pkl` | 两种校准模型，按实际生成／选择情况存在 |
| `debug_3_eyetracker_data.csv`、`debug_3_fixations.csv` | EyeLink gaze 和 native fixation |
| `debug_4_pupil_data_synced.csv`、`debug_4_ptgaze_gaze_synced.csv` | 同步后的 gaze |
| `debug_4_sync_info.json` | 实际使用的同步 offset 等信息 |
| `debug_5_pupil_data_calibrated.csv`、`debug_5_ptgaze_calibrated.csv` | 校准并同步的 gaze |
| `debug_5_calibration_info.json`、`debug_5_ptgaze_calibration_info.json` | 校准应用状态和质量信息 |
| `debug_6_mediapipe_fixations.csv`、`debug_6_ptgaze_fixations.csv` | 正式 fixation 事件，按所选方法生成 |
| `debug_6_mediapipe_fixation_metadata.json`、`debug_6_ptgaze_fixation_metadata.json` | fixation 参数、阈值及汇总信息 |
| `debug_6_three_way_comparison.csv` 等 | 原有 gaze 比较产物 |
| `run_success.json` | 整条主流程成功结束的标记 |
| `error_log.json` | 主流程捕获失败时的步骤、错误和 traceback；不保证被强制终止时也能生成 |

原有 QC 图、overview、metadata 等仍可能出现在根目录，上表列出主要文件而非原项目全部附件。

### `extra/` 中固定要求的 10 项产物

| 文件 | 内容 |
| --- | --- |
| `blink_filter_frame_log.csv` | 完整视频逐帧 blink／眼部质量与时间日志 |
| `blink_filter_regions.csv` | 连续无效帧合并后的区间及原因 |
| `eyelink_blink_benchmark.csv` | 按 trial 整理的 EyeLink EBLINK 区间 |
| `eyelink_trial_markers.csv` | EyeLink trial／fixation 标记 |
| `blink_diagnostic_metrics.csv` | 各 trial 的 TP、FP、FN、precision、recall、F1 等 |
| `blink_event_matches.csv` | 逐个预测区间的匹配结果 |
| `blink_diagnostic_summary.csv` | 当前 VP／Run 的 blink 汇总 |
| `eyelink_fixations_sanity.csv` | 同一 I-DT 核心作用于采样 EyeLink 的 fixation 结果 |
| `blink_diagnostics.pdf` | 按非练习 trial 分页：gaze 与 blink 区间、EAR、状态及代表 ROI |
| `fixation_four_row_timelines.pdf` | 按非练习 trial 分页：gaze＋blink、EyeLink native、EyeLink I-DT、webcam I-DT |

默认实验通常对应 24 个非练习 trial；PDF 页数取决于本 Run 的实际 phase 定义，不写死为固定页数。

### `extra/` 的其他中间文件与完成标记

| 文件 | 内容 |
| --- | --- |
| `frame_timestamps_<hash>.csv` | 特定视频的帧编号与真实 PTS 表 |
| `frame_timestamps_<hash>.json` | 对应视频路径、大小、修改时间及时钟版本；一个 Run 可能涉及多个视频，各自一组 |
| `mediapipe_fixation_input.csv`、`ptgaze_fixation_input.csv` | 真正送入核心的完整帧网格、mask 和校准 gaze，按方法生成 |
| `eyelink_sanity_samples.csv` | EyeLink sanity 的实际采样输入 |
| `eyelink_sanity_metadata.json` | 采样容差、参考方法、I-DT 阈值与配置 |
| `integration_complete.json` | Step 10 全部要求的产物生成并检查后写入的完成标记 |

**`integration_complete.json` 只标记 Step 10 完成；整个 Run 的成功还要看根目录 `run_success.json`。** 如果最后的原有比较阶段失败，Step 10 的 PDF 仍可能存在。

历史 `test60/`／`test25/` 的四个基本 CSV 在正式集成中对应这里的前四项，不再要求手动先在 VP 外围运行一套导出。新 Run 不会自动生成可选 Excel 复核表或历史的独立 inspection PDF。

## 新增文件与职责

以下文件均相对上文固定的 `main` 提交为新增。正式代码路径与参考目录同名文件要区分。

### 生产 blink 模块

| 新文件 | 职责 |
| --- | --- |
| `Skripte/blink/__init__.py` | 声明正式 blink 包 |
| `Skripte/blink/export_blink_filter_logs.py` | 扫描完整视频，导出逐帧日志和区间 |
| `Skripte/blink/extract_eyelink_blink_benchmark.py` | 从 ASC 消息与 EBLINK 提取 benchmark 和 trial 标记 |
| `Skripte/blink/run_data.py` | 严格解析布尔 mask、按帧连接、核对 gaze 时钟及空事件表 schema |
| `Skripte/blink/diagnostic_statistics.py` | 同步时钟上的预测区间、事件匹配结果和汇总 CSV |
| `Skripte/blink/diagnostic_support.py` | 事件匹配、EAR／ROI、真实 PTS 截图及绘图辅助；正式 renderer 使用的模块，不作为独立主入口 |
| `Skripte/blink/generate_blink_diagnostics.py` | 生成每个 Run 的 blink PDF 与统计 |
| `Skripte/blink/run_pipeline_extras.py` | 主 pipeline 的 Step 10 入口，串联新流程并检查完整性 |
| `Skripte/blink/repair_run_timestamps.py` | 为旧 Run 新建 blink 时钟诊断副本；不修复无法确认的 gaze 帧身份，不重新生成可验收的 gaze／fixation |

### 生产 fixation 与共享模块

| 新文件 | 职责 |
| --- | --- |
| `Skripte/fixation/__init__.py` | 声明 fixation 包 |
| `Skripte/fixation/fixation_detector.py` | 正式共享 X-only、时间中位数、P75 I-DT 核心 |
| `Skripte/fixation/run_fixation_detection.py` | Run 数据适配，调用核心并保存事件与 metadata |
| `Skripte/fixation/eyelink_sanity.py` | 屏幕像素转视角、EyeLink 采样及同核心 sanity check |
| `Skripte/fixation/generate_fixation_timeline_pdf.py` | 绘制已保存结果的四行 PDF，不另实现 detector |
| `Skripte/shared/frame_timing.py` | PTS 表、准确窗口解码、同步时钟、trial 分配及补齐显式无效网格 |
| `Skripte/shared/shared_image_preprocessing.py` | 生产原始帧输入契约，仅允许 `none` |
| `Skripte/shared/pipeline_preflight.py` | 启动前依赖、解码和模型调用检查 |

### 研究／验证脚本：不自动进入正式流程

| 新文件 | 用途 |
| --- | --- |
| `Skripte/fixation/run_final_validation.py` | 历史布局上的批量 fixation 验证，输出全体结果与摘要；不是 Step 10，不应无检查地套用到最新 Run |
| `Skripte/debug/debug_6_fixation_detection.py` | 固定角度阈值的 X-only I-DT sweep |
| `Skripte/debug/debug_6_fixation_comparison.py` | fixation 与 native EyeLink 的独立比较 |
| `Skripte/debug/debug_6_eyelink_idt_validation.py` | 高质量 EyeLink 上的 X-only I-DT 验证 |
| `Skripte/debug/debug_6_adaptive_threshold_analysis.py` | 已知 fixation phase 的噪声／离散度分析 |
| `Skripte/debug/debug_6_adaptive_threshold_batch.py` | P75／P80／P90 的批量实验比较 |
| `Skripte/debug/debug_6_adaptive_failure_diagnostic.py` | 检查无可用噪声窗口的 VP：phase、mask、时间间距等 |
| `Skripte/debug/debug_6_ptgaze_smoothing_experiment.py` | raw、50 ms、100 ms 平滑及阈值 sweep 实验 |
| `Skripte/test_preprocessing.py` | 原图输入和旧模式拒绝的轻量检查 |
| `Skripte/commands.text` | 历史命令便笺，包含当前仓库未提供的脚本名，不作为本文运行指令的替代 |

### 测试与验收说明

| 新文件 | 覆盖范围 |
| --- | --- |
| `Skripte/tests/test_pipeline_integration.py` | mask 连接、EyeLink 采样、共享 detector 和独立阈值等 |
| `Skripte/tests/test_raw_frame_contract.py` | 原始帧、旧模式拒绝和缓存来源隔离 |
| `Skripte/tests/test_frame_timing.py` | 变帧率取帧、时钟、trial 标签、显式缺口和旧缓存拒绝 |
| `Skripte/tests/test_pipeline_entrypoints.py` | 真实校准 CLI、失败状态、PKL 优先级、图表几何和启动前失败路径 |
| `Skripte/tests/smoke_pipeline_extras.py` | 合成 Step 10 集成测试，推理／ASC／ROI 使用夹具 |
| `Skripte/tests/INTEGRATION_ACCEPTANCE.md` | 更详细的契约、历史问题、验证范围和手工验收说明 |

### `blink_extra/` 中相对 main 新增的参考文件

这些文件保留既有研究记录，不由主 pipeline 自动调用。

| 文件（均在 `blink_extra/` 下） | 历史用途 |
| --- | --- |
| `export_blink_filter_logs.py` | 独立 blink 导出及片段实验版本 |
| `extract_eyelink_blink_benchmark.py` | 独立 ASC benchmark 导出版本 |
| `run_all_blink_filters.sh` | 按旧 `test60` 等目录批处理的 shell 脚本 |
| `compare_blink_preprocessing.py` | 历史预处理结果比较；不代表重新启用 Lanczos |
| `plot_blink_visualization.py` | 多 VP 的 gaze／blink 对照绘图 |
| `generate_blink_diagnostics.py`、`generate_blink_diagnostics_v2.py` | 早期逐 trial blink 诊断 PDF |
| `generate_eyelink_inspection_pdf.py` | EyeLink 事件的独立 ROI inspection PDF |
| `generate_webcam_blink_review_excel.py` | 带截图、可人工标注的 Excel 复核表 |
| `fixation_diagnostics.py` | 在 native EyeLink fixation 区间中检查 PTGaze |
| `generate_fixation_timeline_pdf.py` | 历史四行 fixation 绘图版本 |
| `IDP_Blink_Handoff_v1.1.md`、`IDP_Blink_Handoff_v1.1.docx` | Blink 阶段交接文档 |
| `IDP_EyeTracking_Blink_Handover.md` | Eye-tracking／blink 历史交接 |
| `IDP_Fixation_Handover_2026-10-12.md` | Fixation 阶段交接 |
| `IDP_integration_layout_plan.md` | 原集成布局规划；与最新实现冲突时，以当前源码及本文为准 |

生成的 `.pyc`、缓存和系统元数据不属于功能文件清单。

## 已有文件的主要改动

本节列出有使用意义的改动，不把换行、包初始化文件等文本差异当作独立功能。

| 文件／范围 | 改动 |
| --- | --- |
| `Skripte/master_cli.py` | 数据／视频根目录参数，按项目实际位置定位脚本 |
| `Skripte/utils/analysis_pipeline.py` | 自动调用 Step 10，预检、来源检查、校准输出校验和失败传播 |
| `Skripte/utils/vp_data_manager.py` | 外部视频发现、Run 配置及缓存兼容字段 |
| `Skripte/config.py` | 原图模式和当前实验／质量设置，路径由 pipeline 环境变量覆盖 |
| 两个 `debug_1` 视频入口 | 使用共享 PTS 与准确窗口，保留真实帧身份和时间 |
| 两个 `offline_calibration` | 校准视频提取使用同一帧时钟和原始输入 |
| `debug_4_synchronization.py` | 拒绝旧帧时钟，输出同步时间并按同步边界重新分配 phase／trial |
| 两个 `debug_5` 校准应用入口 | 失败退出、有效输入检查；PTGaze 指定 PKL 优先、比较模式禁止未同步回退 |
| `shared_pupil_detection.py` 等共享检测代码 | 与 blink／眼部质量日志的字段衔接；新增流程复用共享检测实现 |

相较 `main` 的其余改动可用 Git 完整查看：

```bash
git diff --stat dd0e5b2f0ad7bd6924b4252cccac4af5f792537c a81ee3fa4d8829b4cd9c26229a1f899bdc282749
```

## 单独执行与排错命令

除非标注为仓库根目录，下列命令都在 `Skripte` 目录执行。`/path/to/Run` 是占位符，需要替换成真实的新 Run 路径。对同一个 Run 不要同时启动主流程和单独重建命令。

### 重建已经完成上游的 Run 的 Step 10

仅用于本分支新时钟、原始帧且已经完成同步与校准的 Run；不是将旧 CSV 自动转成新数据的工具。

```bash
python -m blink.run_pipeline_extras \
  --run-dir /path/to/Run \
  --video /Volumes/Empra9/Videos_60/beo7.mp4 \
  --asc /Volumes/Empra10/Ergebnisse60/beo7/beo71339.asc \
  --vp beo7 \
  --methods both \
  --preprocess-mode none
```

该命令会重新进行完整 blink 推理，并覆盖该 Run 对应的导出、fixation、统计和 PDF。调试时先保留要比较的 Run。

### 仅重画已有 fixation timeline

前提是该 Run 已有最新 mask、正式 fixation、EyeLink sanity 等结果。只绘图，不重新运行 detector。

```bash
python -m fixation.generate_fixation_timeline_pdf \
  --run-dir /path/to/Run \
  --vp beo7 \
  --method ptgaze
```

`blink/generate_blink_diagnostics.py` 当前主要作为 Step 10 调用的模块使用，不提供同样的独立 argparse 主入口。

### 旧 Run 的 blink 时钟诊断副本

旧结果出现明显错位时，可不重新推理，先生成独立诊断副本：

```bash
python -m blink.repair_run_timestamps \
  --run-dir /path/to/legacy_Run \
  --output-dir /path/to/new_blink_clock_diagnostic \
  --vp beo7
```

输出目录必须尚不存在，原 Run 不会被覆盖。默认省略 ROI；需要真实截图时再加 `--include-rois --video /path/to/video.mp4`。

尽管文件名带 `repair`，它目前**只重建 blink 时钟诊断与统计**，旧 gaze 保持原值并标记为未经确认的背景。不会把有歧义的旧帧号重标成可信的新 gaze，也不会据此重新生成 fixation。副本标记为 `diagnostic_only`，不能作为正式 Run 或生产图像缓存使用。副本还会保存 `timebase_audit.csv` 和 `blink_clock_diagnostic_complete.json`。

### 查看入口帮助

```bash
python master_cli.py --help
python -m shared.pipeline_preflight --help
python -m blink.run_pipeline_extras --help
python -m fixation.generate_fixation_timeline_pdf --help
python -m blink.repair_run_timestamps --help
```

## 测试与复现

以下命令在**仓库根目录**执行。

```bash
PYTHONPATH=Skripte python -m unittest discover -s Skripte/tests -v
python -m compileall -q Skripte
```

固定提交 `a81ee3f` 的已执行结果为 31 项通过。VFR 测试需要 FFmpeg／FFprobe；H.264 测试还需要该 FFmpeg 构建具有 `libx264` 编码器，否则相应测试会跳过，应检查日志中的 `skipped`，不能只看最后的 `OK`。

合成集成测试写入独立的新测试目录：

```bash
python Skripte/tests/smoke_pipeline_extras.py \
  --output-dir /tmp/idp-smoke/Run_synthetic
```

该目录会生成两份合成 PDF 及中间文件。它验证流程衔接，不评估真实 VP 的 blink／fixation 准确率。

用于数值／解码回归的环境为 Python 3.12.14、FFmpeg 6.1.1；这不是完整模型运行环境的版本承诺。本机模型环境已经通过单帧预检，但当前还未把全部实际包版本作为 lockfile 纳入仓库。

建议在实际运行环境记录以下信息，用于后续复现：

```bash
git rev-parse HEAD
python --version
python -m pip freeze > requirements-local.txt
ffmpeg -version > ffmpeg-version.txt
```

同时保留 Run 的 `config_used.json`、校准 PKL、原始输入文件及完整日志。个人路径和原始 VP 数据不应随意加入公共仓库。

## 常见问题

| 现象 | 说明／处理 |
| --- | --- |
| `Head-Pose-Extraktion: ✗` | 当前 `ENABLE_HEAD_POSE_EXTRACTION=False`，表示 MediaPipe 额外的 head-pose 提取关闭；不是初始化失败，也不等于 PTGaze 自身 head pose 被关闭 |
| PTGaze 显示 `Device: cpu` | 当前选择逻辑为 CUDA 可用则 CUDA，否则 CPU；Apple M2 当前不自动选择 MPS，CPU 运行可能较慢 |
| TensorFlow Lite／MediaPipe 提示和 `np.bool` FutureWarning | 当前本机记录中这些提示没有阻止预检完成；仍应以进程退出状态、最终 OK 和后续实际输出为准，不笼统忽略所有 warning |
| 找不到 `ffmpeg` 或 `ffprobe` | 检查系统安装与 PATH；Python 包不能替代所需的可执行命令 |
| MediaPipe 没有 `solutions` 或 FaceMesh 初始化失败 | 当前安装版本与项目调用接口不兼容；使用原有已验证环境或核对版本，不能仅靠包名判断兼容 |
| 缓存提示 raw-frame／presentation-time 不兼容 | 重新计算图像相关步骤；不要改旧 metadata 来绕过校验 |
| Custom calibration 缺来源信息 | PKL 所在源目录需要兼容的 `config_used.json`；首次验收直接重新生成校准 |
| 时钟 mismatch、缺 mask 或重复帧报错 | 核对视频、Run 和 frame log 是否来自同一次正确运行；不要用自动平移或插值掩盖不一致 |
| `no valid fixation-phase windows` | 检查已知 fixation phase、有效 gaze、blink mask 和时间覆盖，未能估计 P75 时不能随意填固定阈值当作正式结果 |
| 看到 PDF 但流程仍失败 | 检查 `integration_complete.json`、`run_success.json`、`error_log.json` 和终端报错；文件可能是部分阶段的产物 |
| Run 耗时很长 | 视频解码、两路检测、校准与 Step 10 完整 blink 扫描均有成本；目前没有整合成只做一次所有模型推理的单遍流程 |
| main 的旧图与新图不一致 | 先核对图像处理、真实帧身份和时钟，再检查有效性与校准；不以任意调 offset 或阈值强行复现旧图 |

主流程各步骤有独立超时，例如 PTGaze 视频分析 2400 s、各离线校准 3600 s、Step 10 为 7200 s；超过相应阶段的上限会失败。总运行时间不是其中某一个阶段的超时时间。

## 已知限制

1. **单视频目录：** 当前 `--video-root` 只有一个，外部分离目录中的 60 Hz 与 25 Hz 必须分别启动，菜单中的双频率选项尚未适配这套目录参数。
2. **完整集成要求 EyeLink 比较模式：** 需要 ASC 与同步输入，`analysis_mode=1` 不支持这套完整 Step 10。
3. **真实验收仍在进行：** 单帧预检和合成测试不等于新的 beo7／全部 VP／25 Hz 最新流程已验收。
4. **环境尚未锁版本：** 新机器的完整安装、模型兼容性和运行速度需要分别确认。
5. **CPU 与重复扫描：** 当前没有 Apple MPS 自动选择，也没有将上游 gaze 与 Step 10 blink 合并为单次扫描；不承诺明显缩短到某个固定时长。
6. **旧 Run 不能仅靠改时间戳变成可信新 Run：** VFR 帧号身份可能已经错误，需要重新准确解码与推理。
7. **预检不能提前发现所有数据问题：** 长视频中断、后续 trial 无可用窗口、校准质量及跨 VP 的个体差异仍需实际运行核查。
8. **研究工具保留历史假设：** `blink_extra/` 和独立 debug 验证脚本可能使用旧目录和匹配标准，不属于生产验收的替代入口。
9. **统计入口延续原项目：** 现有统计准备与 R 流程仍保留；新增 fixation CSV 不表示已自动纳入所有原有 R 模型。

## 原项目文档与外部说明

- [原项目 Readme.pdf](Readme.pdf)、[Skriptbeschreibung.pdf](Skriptbeschreibung.pdf)：原项目入口文档，具体当前分支命令以本文及源码为准。
- [MA_Köthnig.pdf](MA_K%C3%B6thnig.pdf)：原项目论文资料。
- [本分支验收说明](Skripte/tests/INTEGRATION_ACCEPTANCE.md)。
- [固定版本源码](https://github.com/yiyi-cs/IDP/tree/a81ee3fa4d8829b4cd9c26229a1f899bdc282749) 与 [main 对比基线](https://github.com/yiyi-cs/IDP/tree/dd0e5b2f0ad7bd6924b4252cccac4af5f792537c)。
- [FFprobe 官方文档](https://ffmpeg.org/ffprobe.html)、[Homebrew FFmpeg](https://formulae.brew.sh/formula/ffmpeg)、[openpyxl 安装与图片支持](https://openpyxl.readthedocs.io/en/stable/tutorial.html#installation)。

更新算法、图像模式、时间映射或输出结构时，请同步更新 README、回归测试与验收记录。提交问题时附上 commit、运行命令、方法／频率／校准选择、环境版本、失败步骤和必要日志；避免直接公开原始 VP 视频或其他个人数据。

截至上述对照版本未发现独立的仓库 `LICENSE` 文件；不要将本文理解为新增许可声明。项目许可由维护者确认，第三方库与模型保留各自许可。
