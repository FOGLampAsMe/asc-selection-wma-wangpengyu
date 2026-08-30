# UnifoLM-WMA-0 大题提交材料

姓名：王鹏宇  
学号：240809010506  
题目：Embodied World Model / UnifoLM-WMA-0  
场景：`unitree_g1_pack_camera / case1`  
资源：Google Colab Tesla T4 15 GB，Python 3.10，PyTorch 2.3.1+cu121

## 完成情况

已在同一 Colab T4 会话中完成 WMA Baseline、GPU 采样、日志计时、官方 PSNR 计算，以及 3 种优化尝试。Baseline 输出视频为 16 帧，程序返回码为 0，PSNR 为 21.1719 dB。常驻复用、`torch.inference_mode()` 和关闭非必要 TensorBoard 写入的完整结果见 `results/final_selection_results.json`。

最终实测：冷启动 252.9508 s；常驻复用 14.6757 s（17.2361x）；常驻 + inference_mode 14.1748 s（相对常驻 1.0353x）；常驻 + no-TensorBoard 9.8661 s（相对常驻 1.4875x）。四个视频均通过 ffprobe 的 16 帧、512x320 检查，PSNR 均约 21.172 dB。

当前目录是轻量复现仓库，不包含模型权重、数据集和生成视频；这些文件按报告中的 Colab 命令下载到 `/content`。

## 固定参数

```text
seed=123, height=320, width=512, video_length=16,
ddim_steps=1, n_iter=1, frame_stride=6, exe_steps=16,
unconditional_guidance_scale=1.0, guidance_rescale=0.7, perframe_ae
```

## Baseline 命令

```bash
/content/micromamba/envs/wma/bin/python \
  scripts/evaluation/world_model_interaction_t4_fp16_staged.py \
  --seed 123 --ckpt_path ckpts/unifolm_wma_dual.ckpt \
  --config configs/inference/world_model_interaction_colab_case1.yaml \
  --savedir /content/wma_results/baseline_fp16_staged_case1_final \
  --bs 1 --height 320 --width 512 --unconditional_guidance_scale 1.0 \
  --ddim_steps 1 --ddim_eta 1.0 \
  --prompt_dir /content/ASC26-Embodied-World-Model-Optimization/unitree_g1_pack_camera/case1/world_model_interaction_prompts \
  --dataset unitree_g1_pack_camera --video_length 16 --frame_stride 6 \
  --n_action_steps 16 --exe_steps 16 --n_iter 1 \
  --timestep_spacing uniform_trailing --guidance_rescale 0.7 --perframe_ae
```

## 分析工具

外层使用 `time.perf_counter()` 记录子进程时间，使用 `nvidia-smi --query-gpu ... -l 1` 每秒保存 GPU 利用率、显存和功耗，并使用官方 `psnr_score_for_challenge.py` 计算 PSNR。详见 `results/` 和 `patches/`。

## 优化修改

`patches/resident_model_reuse.patch` 展示了将模型与数据集初始化移入模块级缓存，并在后续请求复用的修改。此修改只针对实测的冷启动瓶颈，不改变模型权重、输入、采样参数或评价脚本。

最终选拔要求至少 3 种优化尝试，本仓库已记录并提交 3 种实际对照结果。
