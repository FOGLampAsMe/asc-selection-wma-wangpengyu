# UnifoLM-WMA-0 大题提交材料

姓名：王鹏宇  
学号：240809010506  
专业和年级：24届计算机科学与技术  
题目：Embodied World Model / UnifoLM-WMA-0

## 正式实验环境

- GPU：NVIDIA GeForce RTX 4090 D
- AMP：FP16 autocast
- 运行方式：同一台服务器、同一份权重、同一组输入和 `seed=123`
- 采样：50 步 DDIM，官方 `n_iter`，完整输出
- 输出：512×320，8 fps；不减少采样步数、交互轮数或输出帧数
- 评价：官方 PSNR 脚本，另检查进程返回码和视频元数据

模型权重、数据集和生成视频没有提交到仓库，按复现命令放置在服务器的 `/root/autodl-tmp` 下。

## 正式 20 Case 对照

AMP Baseline 总时间为 `25357.059 s`，优化后总时间为 `4588.333 s`，总加速比为 `5.526×`，时间下降 `81.905%`。优化后 20/20 个 Case 有效，最低 PSNR 为 `25.9339 dB`。

逐 Case 的完整数据见 [`results/formal_20_case.csv`](results/formal_20_case.csv)。加速比按同一个场景和 Case 的 AMP Baseline 时间除以优化后时间计算，范围为 `4.709×–6.608×`。

## Baseline 与优化命令口径

两组实验均使用同一套官方场景和 Case、同一权重、`seed=123`、50 步 DDIM、官方 `n_iter`、512×320 输出和官方 PSNR 评价。外层用 `time.perf_counter()` 记录端到端时间，用 `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1` 保存 GPU 监控。

## 优化方法

实测日志显示采样循环中存在跨步不变的条件编码、注意力 K/V 和 mask 构造、残差时间投影，以及 DM 分支中重复的特征计算。提交的优化围绕这些重复部分展开：

1. 缓存 image conditioning、固定文本条件、空间注意力 K/V、mask 和残差 time projection；
2. 持久化 FP16 Conv/Linear 权重，避免运行过程中反复转换和初始化；
3. 在 DM 分支使用受控特征复用，`feature_interval=3`、`feature_initial=8`、`feature_final=8`、`feature_extrapolate=0`；
4. WM 分支保持逐步计算，`feature_wm_interval=1`，以避免改变动作条件路径。

这些改动没有减少 DDIM 步数、交互轮数、输出帧数、分辨率或评价方式。主要实现和补丁位于 `patches/`，正式结果文件位于 `results/formal_20_case.csv`。

## 复现入口

```bash
git clone https://github.com/FOGLampAsMe/asc-selection-wma-wangpengyu.git
cd asc-selection-wma-wangpengyu

# 按报告中的环境说明准备官方权重和 20 个 Case 数据，
# 先运行统一 FP16 AMP Baseline，再运行带缓存和受控特征复用的版本。
# 运行结束后用官方评分脚本计算 PSNR，并核对 results/formal_20_case.csv。
```

仓库保留运行补丁、旧记录和结果说明，旧的单 Case 或其他 GPU 记录只作为历史材料，不参与本次正式结论。
