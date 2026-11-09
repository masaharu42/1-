# 折 19：VAE 条件图像扩散准备

本阶段先检查扩散细化能否在现有组织图的 VAE 粗重建基础上生成细节。采用现有折 19 的 Ni/Fe/Cr 加权 VAE，从其原始图片得到粗重建图，冻结 VAE 后训练图像空间条件 DDPM。这个 VAE 作为已训练的可用起点，不表示其成分预测已达目标精度。参考 NC 正文 Methods 中 DDPM 以 VAE 图像为条件细化；本项目目前训练损失仅为噪声 MSE，并非论文的完整像素、SSIM 和边缘损失实现。

流程为真实图 → VAE 编码/解码 → 粗图；训练扩散时对真实图加噪并以粗图为条件预测噪声。采样输入真实图时，是重建与细化试验。若以后从潜空间设计新组织，还需要另行建立可靠的潜变量采样或目标性能搜索方式。

## 1. 更新代码与核查

在服务器已有项目根目录执行。使用上次私有数据和已训练权重，无需重新训练 VAE。请查看已有文件，若代码有本地修改则先检查，不要覆盖。

```bash
git status --short
git pull --ff-only origin main
cat data/split.json
nvidia-smi
python -m unittest discover -s tests -v
```

确认 test 为 19、val 为 14，并确认 `runs/折19_三主元素加权/成分监督VAE/checkpoint_0005000.pt` 存在。脚本会核验 VAE 权重记录中的数据清单和合金划分；训练 ID19 没有参加，ID14 用于验证。ID14、ID19 均已参与前面的研究分析，因此新结果用于探索，不视作完全独立盲测。

## 2. 先做 100 步 GPU 试跑

训练的 batch-size 表示合金数，windows-per-alloy 表示每合金随机窗口数，下面每次实际为 2 张 256×256 窗口。训练在新目录运行，实时输出噪声 MSE、时间和峰值显存。

```bash
python -u train_diffusion.py \
  --manifest data/manifest.json --split data/split.json \
  --vae-checkpoint 'runs/折19_三主元素加权/成分监督VAE/checkpoint_0005000.pt' \
  --output 'runs/折19_VAE条件扩散/试跑100步' \
  --device cuda --amp fp16 --patch-size 256 --batch-size 1 --windows-per-alloy 2 \
  --steps 100 --lr 0.0001 --timesteps 1000 --val-every 50 --save-every 100 --eval-windows 8
```

试跑检查 CUDA、显存和是否有非有限 loss。100 步结果不用于判断生成质量。混合精度及显存依实际 GPU 测试，遇到显存不足可降低 `--windows-per-alloy` 并记录。

## 3. 正式训练

使用单独目录从零训练扩散模型，10000 步是首轮起点，需结合 ID14 的验证和图像结果判断。

```bash
python -u train_diffusion.py \
  --manifest data/manifest.json --split data/split.json \
  --vae-checkpoint 'runs/折19_三主元素加权/成分监督VAE/checkpoint_0005000.pt' \
  --output 'runs/折19_VAE条件扩散/正式训练' \
  --device cuda --amp fp16 --patch-size 256 --batch-size 1 --windows-per-alloy 2 \
  --steps 10000 --lr 0.0001 --timesteps 1000 --val-every 250 --save-every 1000 --eval-windows 8

python plot_metrics.py 'runs/折19_VAE条件扩散/正式训练/metrics.csv' \
  --output 'runs/折19_VAE条件扩散/正式训练/训练曲线'
```

断点续训时，在相同的正式命令末尾加入 `--resume 'runs/折19_VAE条件扩散/正式训练/last.pt'`，`--steps` 为总目标步数，并保持其他配置和 VAE 权重一致。

## 4. 先查看 ID14 的条件细化图

```bash
python sample.py \
  --vae-checkpoint 'runs/折19_三主元素加权/成分监督VAE/checkpoint_0005000.pt' \
  --diffusion-checkpoint 'runs/折19_VAE条件扩散/正式训练/best.pt' \
  --reference data/images/14.bmp \
  --output 'runs/折19_VAE条件扩散/验证_ID14_图像细化' \
  --sample-steps 50 --count 4 --device cuda
```

`reference_crop.png`、`coarse.png`、`refined_000.png` 等分别是原始中心窗口、VAE 粗图和扩散细化图。采样中心窗口与 VAE 验证程序的首个固定窗口不一定是同一位置，比较前须按坐标对应。固定模型与方法后，可以把参考文件换为 `data/images/19.bmp`、输出换成 `runs/折19_VAE条件扩散/探索_ID19_图像细化`。输出目录需为空。

## 5. 结果和传回

`正式训练/metrics.csv` 保存训练和验证的噪声 MSE；`训练曲线/` 为逐项曲线，`best.pt`、`last.pt` 为权重。预测噪声损失不能单独证明组织结构或晶界真实性。图像细化的成分稳定性尚未形成完整评估程序，本阶段先检查生成图和重建图，并记录参数。

```bash
zip -r '折19_VAE条件扩散_结果供分析.zip' 'runs/折19_VAE条件扩散' \
  -x '*.pt' '*.pth' '*.ckpt' '*.safetensors'
unzip -l '折19_VAE条件扩散_结果供分析.zip'
```

压缩包包含裁切的研究组织图及其重建，按私有研究数据保管。扩散与 VAE 权重留在服务器单独备份。
