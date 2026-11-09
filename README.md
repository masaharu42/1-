# Euler+GB 组织与成分：服务器训练项目

本项目从零训练 VAE，比较普通 VAE 与成分监督 VAE，并训练以 VAE 粗图为条件的图像空间扩散模型。输入为 256×256 的原始 Euler+GB 窗口。最终目标是图像→成分；性能搜索和强化学习留待性能标签核实及前阶段验证后实现。本版本为可审核的首轮 pilot，另含探索性性能代理；尚不包含完整的性能驱动潜空间搜索或 RL。

折 19 的具体运行顺序、九元素回归比较及中文结果路径见 [折19服务器操作与结果说明](折19服务器操作与结果说明.md)。

Ni、Fe、Cr 优先加权的新实验和独立结果目录见 [主元素加权服务器操作说明](主元素加权_服务器操作说明.md)。

现有折 19 权重的训练集核查，以及折 18 统一尺度对照，见 [训练集核查与折18尺度对照](训练集核查与折18尺度对照_服务器操作.md)。

## 数据与划分

代码仓库公开时，研究图片、成分标签、模型权重和运行结果应单独传输，不提交到 Git。将独立数据包解压到仓库根目录后，结构为：

```text
data/manifest.json             # 图像、成分标签与图像 SHA-256
 data/split.json                # 合金级 train/val/test
 data/images/01.bmp ... 22.bmp  # 原始 Euler+GB
 data/compositions_source.csv   # 原成分表副本
```

数据包包含 ID01–22；ID22 的标签为标准范围中点，不是已确认的实测值。**默认示例划分**：测试 ID19、验证 ID14、其余 20 个用于训练（包括 ID22）。这只是待审核的单折示例，不是全面验证；相同窗口不能跨分组。更改划分前请备份原 split，并重新从零训练该折全部模型。不要用测试结果选择参数。对各元素中的缺失/常零成分，模型输出不能视为有识别能力。

可在提供源图片和成分 CSV 时重新准备（会生成 split 和数据副本）：

```bash
python prepare_data.py --images-dir /path/to/Euler_GB --compositions /path/to/compositions.csv --output data --val-ids 14 --test-ids 19
```

已有数据目录默认拒绝覆盖；请用不同输出目录或确认后使用 `--overwrite`。

## 服务器环境

Linux、Python 3.10–3.12 和与服务器驱动兼容的 NVIDIA PyTorch；在 GPU 环境中按服务器 CUDA/驱动情况安装 PyTorch 后执行 `python -m pip install -r requirements.txt`。运行前检查：

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -m unittest discover -s tests -v
```

标准 RTX 4090 通常为 24GB，实际设备和可用显存以 `nvidia-smi` 为准；不要仅按显卡名称推断。下面 batch 是待实测的起始值，不保证在任何 GPU 上无溢出。如果显存不足，减少合金数或每合金窗口数，并记录变更。

## A/B：从零训练两种 VAE

A 为普通 VAE；B 为加入合金级成分监督的 VAE。为公平比较，两个命令使用相同划分、步数、seed 和增强；每次成分损失按合金而不是窗口计权。增强包含训练窗口的水平/垂直像素翻转、90°倍数旋转及随机模糊/遮挡恢复；这些操作尚未证明晶体取向或性能方向上的物理等变性。务必再用关闭 `--train-flip --train-rotate --corruption-prob` 的版本作对照；验证/测试始终不增强。

先短程 100 步检查无误，再决定是否延长；这里的 5000 步和参数只是 pilot，不是最优配置：

```bash
python -u train_vae.py --manifest data/manifest.json --split data/split.json --output runs/vae_A \
  --patch-size 256 --alloys-per-batch 2 --windows-per-alloy 4 --eval-windows 16 \
  --steps 5000 --lr 0.0002 --composition-weight 0 --precision fp16 --device cuda \
  --train-flip --train-rotate --corruption-prob 0.2

python -u train_vae.py --manifest data/manifest.json --split data/split.json --output runs/vae_B \
  --patch-size 256 --alloys-per-batch 2 --windows-per-alloy 4 --eval-windows 16 \
  --steps 5000 --lr 0.0002 --composition-weight 0.0001 --precision fp16 --device cuda \
  --train-flip --train-rotate --corruption-prob 0.2
```

`--composition-weight` 是待审核的试验权重，不应无依据认为有效；成分损失归一化后可能大于图像重建损失，务必看日志里的 `comp` 以及乘权后的量级，在训练折内对照。默认 L1 + Sobel 0.2 + KL 1e-6，非完整 NC 论文实现。默认同一训练窗口既可能发生模糊或遮挡，目标是该窗口的干净图；不是输出其原始 CTF 取向。

训练打印 step、loss、L1、Sobel、KL、成分 loss、MAE、时间及峰值显存，保存 `metrics.csv`、`run_config.json` 与检查点；测试合金不参与训练/验证。若中断，在同一目录和相同关键参数下恢复，`--steps` 是总目标步数：

```bash
# 在原命令末尾增加（使用存在的检查点，保持其余参数不变）
--resume runs/vae_B/checkpoint_0000500.pt
```

恢复模型、优化器、AMP 和随机状态并保留原日志，但 DataLoader 重新启动，后续窗口顺序不保证逐步完全一致。仅加载来源可信的 PyTorch 检查点。

## 冻结 VAE 的成分基线与测试图

先完成 A，再冻结 A 的编码器训练成分头；然后与 B 比较：

```bash
python -u fit_frozen_head.py --vae-checkpoint runs/vae_A/checkpoint_0005000.pt --output runs/head_A --device cuda --steps 1000 --eval-windows 16 --train-flip --train-rotate --corruption-prob 0.2
python evaluate_vae.py --checkpoint runs/head_A/frozen_head.pt --partition val --output runs/eval_A_val --device cuda
python evaluate_vae.py --checkpoint runs/vae_B/checkpoint_0005000.pt --partition val --output runs/eval_B_val --device cuda
python plot_metrics.py runs/vae_B/metrics.csv --output runs/vae_B/plots
```

先用验证集合决定检查点与配置。预先固定最终方案后，把上述 `--partition val` 换成 `--partition test` 并使用新输出路径；每个元素生成一张记录值—预测值图和 CSV，并生成首个固定窗口的原图、重建图及差异对照。普通 VAE 无成分头时仅输出重建图。成分恒定的训练列在 CSV 中由 active_in_training 标记；即使 softmax 对该列产生非零数值，也不能理解为模型识别了该元素。单折仅一个测试 ID，图上只有一个点；完整结论应通过多个严格合金留出折汇总，不能把该单点当成整体精度。训练期被冻结的模型和成分头使用同一折对应的训练、验证、测试列表；检查点与数据清单哈希不一致会中止。

## 图像空间扩散细化（实验性）

使用折 19 已训练 VAE 的服务器步骤和独立结果目录见 [折19 VAE条件扩散操作说明](折19_VAE条件扩散_服务器操作.md)。

本阶段冻结已选 VAE，以它对真实窗口产生的粗重建图作为条件，训练 RGB 图像空间 DDPM 预测噪声；没有直接输入已知合金成分。它是参考 NC 思路的基础实现，当前使用噪声 MSE，尚未实现论文中的完整 SSIM/边缘训练损失。

```bash
python -u train_diffusion.py --manifest data/manifest.json --split data/split.json \
  --vae-checkpoint runs/vae_B/checkpoint_0005000.pt --output runs/diff_B \
  --device cuda --amp fp16 --patch-size 256 --batch-size 1 --windows-per-alloy 2 \
  --steps 10000 --lr 0.0001 --timesteps 1000 --val-every 250 --save-every 1000
```

`--batch-size` 是合金数量，每个合金有 `--windows-per-alloy` 个窗口，实际图像批量为两者相乘。可从上述保守配置先跑 100 步，确认峰值显存后调整；32GB 及以上仍需实测。实时打印 step、train/val 噪声 MSE、时间与显存，`metrics.csv` 和 `best.pt/last.pt` 保存到运行目录，恢复用原参数加 `--resume runs/diff_B/last.pt`。VAE 与扩散训练划分、VAE 权重 SHA、数据哈希需一致。

生成示例（仅粗图条件细化）：

```bash
python sample.py --vae-checkpoint runs/vae_B/checkpoint_0005000.pt \
  --diffusion-checkpoint runs/diff_B/best.pt --reference data/images/19.bmp \
  --output runs/refine_demo --sample-steps 50 --count 4 --device cuda
```

输入参考图时为**条件重建/细化演示**，不是未知目标的自由生成。脚本也支持 `--latent` 输入外部单窗口潜变量，但项目尚未训练能按照目标性能产生潜变量的模型，不能把它称为完整性能驱动设计。不能拿测试图进行训练/调参；最终一次报告其图像重建可作为测试，不证明真实制造。

## 训练输出与下一阶段

- 新训练任务独立输出，不覆盖旧实验；`.gitignore` 排除数据、权重和 runs。
- 断点续训需原始 run 目录和可信检查点；使用终端 `python -u ...` 实时输出，也可 `2>&1 | tee train.log` 记录终端信息。
- 本地 CPU 单步与恢复检查仅检查代码链路；服务器 CUDA 版本、显存、吞吐及训练质量尚需实际检查。
- 已有探索性性能代理和候选筛选脚本，完整性能驱动优化搜索与强化学习尚未实现。峰值应力需核对试验和曲线定义；曲线终点应变不可直接称断后延伸率。

## 后续：探索性性能代理与候选筛选

压缩包中的曲线是与 Ncorr 帧对齐的记录，未证明覆盖完整试验机最大载荷；一些曲线早期段存在回退或错位问题、部分峰值出现在最后一个保留帧。首轮建议只把整理后表格的 `ncorr_observed_peak_mpa_median` 视作“**已观测曲线峰值应力的合金级中位数**”探索目标，绝不能自动认定为已确认的抗拉强度；不使用屈服强度、断后伸长率，直到原始试验记录核查。候选标签表包含 `n_peak_at_last`、`peak_status` 等标记，需人工审核。**性能训练脚本不自动排除或修正标记条目**；若做敏感性分析，可显式用 `--exclude-train-ids 1,10,15,20` 从性能代理拟合中排除这些训练 ID（必须确实属于该折训练集），并与全部纳入的结果比较。此开关不移除这些 ID 的 VAE 训练图，也不会自动排除其他标记试次；若要严格排除这些图像，需调整划分并重训 VAE。验证/测试中的疑问样本须单独标注，不能用排除参数掩盖。

另行传输私有力学数据包并解压后，下面是预先固定的单折探索：

```bash
python extract_performance_features.py --checkpoint runs/vae_B/checkpoint_0005000.pt \
  --manifest data/manifest.json --split data/split.json \
  --windows 16 --batch-windows 4 --device cuda --output runs/perf/features.json

python fit_performance.py --features runs/perf/features.json \
  --targets 'data/mechanics/分析输出/合金级候选标签_含质量标记.csv' \
  --target-column ncorr_observed_peak_mpa_median --split data/split.json \
  --alpha 10 --output runs/perf/peak_val
```

模型比较训练均值、图像潜变量、记录成分、图像+记录成分；后两者在测试时使用已知的**记录成分**，属于“已知成分”对照，并不代表仅凭未知生成图片就能直接使用。默认只输出验证 ID 的图和表；确定方法之后才用新输出路径显式增加 `--evaluate-test`。默认一折只有一个验证/测试合金，不能作为总体精度，必须逐合金严格留出并汇总；性能输入标签及模型结果不得视为实测最优。

`rank_candidates.py` 对外部提供的候选预测表做简单双目标 Pareto 筛选，产生一张散点图和候选表；它不是 NSGA-II、强化学习、成分搜索或材料验证。使用前需要独立准备具有 `candidate_id` 及两列模型预测的 CSV，例如：

```bash
python rank_candidates.py --candidates /path/to/candidate_predictions.csv \
  --objective-x predicted_peak --objective-y predicted_plasticity --output runs/candidate_review
```

如果塑性指标没有明确确认，不应把曲线终点命名为断后延伸率或作为可靠塑性目标。真正的目标驱动候选生成、NSGA-II/贝叶斯搜索、奖励设计和强化学习未实现，必须在性能标签和评分模型可信后继续研发。
