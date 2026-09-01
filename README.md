# UnifoLM-WMA-0 大题提交材料

> 本仓库对应最终选拔材料的重复提交第二版。

姓名：王鹏宇  
学号：240809010506  
专业和年级：24届计算机科学与技术  
题目：Embodied World Model / UnifoLM-WMA-0  
筛选场景：五个场景 Case1-4，共 20 个 Case；最终选用 `unitree_z1_stackbox / case1`  
资源：AutoDL NVIDIA GeForce RTX 4080，Python 3.10.18，PyTorch 2.3.1+cu121，CUDA 12.1

## 完成情况

已在同一 AutoDL RTX 4080 实例完成 20 Case 筛选、WMA Baseline、日志计时、官方 PSNR 计算，以及一次与瓶颈对应的有效优化。最高分 Case 输出视频为 16 帧，程序返回码为 0，PSNR 为 31.5206 dB。全量分数见 `results/all20_scores.csv`，同 Case 优化结果见 `results/autodl_optimization_results.csv`。

同一进程固定 seed 的冷启动基线为 149.5939 s / 31.5206 dB；模型和数据集常驻复用为 83.6751 s / 31.5235 dB，速度提升 1.7878 倍，时间下降 44.07%。另有同 Case 完整 `n_iter=11`、176 帧复核，PSNR 为 22.8061 dB。

当前目录是轻量复现仓库，不包含模型权重、数据集和生成视频；这些文件按报告中的 AutoDL 命令下载到 `/root/autodl-tmp`。

## 固定参数

```text
seed=123, height=320, width=512, video_length=16,
ddim_steps=50, n_iter=1, frame_stride=4, exe_steps=16,
unconditional_guidance_scale=1.0, guidance_rescale=0.7, perframe_ae
```

## Baseline 命令

```bash
/content/micromamba/envs/wma/bin/python \
  scripts/evaluation/world_model_interaction.py \
  --seed 123 --ckpt_path ckpts/unifolm_wma_dual.ckpt \
  --config configs/inference/world_model_interaction.yaml \
  --savedir results/all20_unitree_z1_stackbox_c1 \
  --bs 1 --height 320 --width 512 --unconditional_guidance_scale 1.0 \
  --ddim_steps 50 --ddim_eta 1.0 \
  --prompt_dir /root/autodl-tmp/ASC26-Embodied-World-Model-Optimization/unitree_z1_stackbox/case1/world_model_interaction_prompts \
  --dataset unitree_z1_stackbox --video_length 16 --frame_stride 4 \
  --n_action_steps 16 --exe_steps 16 --n_iter 1 \
  --timestep_spacing uniform_trailing --guidance_rescale 0.7 --perframe_ae
```

## 分析工具

外层使用 `time.perf_counter()` 记录子进程时间，使用 `nvidia-smi --query-gpu ... -l 1` 每秒保存 GPU 利用率、显存和功耗，并使用官方 `psnr_score_for_challenge.py` 计算 PSNR。详见 `results/` 和 `patches/`。

## 优化修改

`patches/resident_model_reuse.patch` 展示了将模型与数据集初始化移入模块级缓存，并在后续请求复用的修改。此修改只针对实测的冷启动瓶颈，不改变模型权重、输入、采样参数或评价脚本。

第四次作业只要求一次单点优化，本仓库记录了模型和数据集常驻复用的完整对照结果。
