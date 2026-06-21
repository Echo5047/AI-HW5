# 人工智能导论第 5 次作业：生成模型

本次编程作业包含两部分：MNIST 图像扩散模型，以及 Stable Diffusion 风格的文生图扩散模型。第一部分帮助大家理解扩散模型的训练、采样和条件生成；第二部分关注现代文生图模型的推理流程，并通过 classifier-free guidance（CFG）实验观察文本条件强度对生成结果的影响。

本仓库只保留本次作业需要的代码和脚本。

## 环境依赖

请在 Python 环境中安装以下依赖：

```bash
pip install torch torchvision numpy opencv-python einops tqdm scipy matplotlib pillow huggingface_hub transformers safetensors packaging requests typing_extensions
```

文生图部分不需要额外安装 `diffusers` 包。仓库中已经包含本作业需要的本地 `diffusers/` 推理代码，用于加载 UNet、VAE 和 scheduler。

## 项目结构

```text
unified_model/
├── models/                 # MNIST diffusion / conditional diffusion 模型
├── samplers/               # MNIST diffusion 采样器
├── train.py                # MNIST 扩散模型训练入口
├── inference.py            # MNIST 扩散模型推理入口
└── visualize_denoising.py  # 去噪过程可视化
diffusers/                  # 文生图所需的本地推理代码
text_to_image.py            # 文生图推理脚本
text_to_image_utils.py      # 文生图模型下载、加载与图像保存工具
cal_fid.py                  # 条件扩散模型 FID 计算脚本
```

## 编程任务

### 1. 走进 MNIST 扩散模型

请阅读 `unified_model/models/diffusion_model.py` 与 `unified_model/samplers/diffusion_samplers.py`，理解 DDPM 前向加噪和反向去噪的基本流程，并根据代码中的提示补全相应实现。

### 2. 训练、推理与可视化 MNIST 扩散模型

训练 MNIST 扩散模型：

```bash
python -m unified_model.train --model diffusion --batch_size 512 --n_epochs 100 --output_dir out/diffusion --sampler ddpm --n_steps 1000
```

生成样本：

```bash
python -m unified_model.inference --model diffusion --model_path out/diffusion/diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 100 --output_dir out/samples
```

可视化去噪过程：

```bash
python -m unified_model.visualize_denoising --model diffusion --model_path out/diffusion/diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 5 --n_steps_to_show 10 --output_dir out/visualization
```

请在报告中展示生成图像网格和去噪过程可视化结果，并简要说明你观察到的扩散去噪过程。

### 3. 为 MNIST 扩散模型加入类别条件

请阅读 `unified_model/models/unet.py`，理解时间步信息和类别条件如何进入 U-Net。完成条件注入后，训练条件扩散模型：

```bash
python -m unified_model.train --model conditional_diffusion --batch_size 512 --n_epochs 100 --output_dir out/cond_diffusion --sampler ddpm --n_steps 1000 --num_classes 10 --label_emb_dim 32
```

生成条件扩散模型样本：

```bash
python -m unified_model.inference --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 100 --output_dir out/samples
```

可视化条件扩散模型的去噪过程：

```bash
python -m unified_model.visualize_denoising --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --n_samples 5 --n_steps_to_show 10 --output_dir out/visualization
```

按类别生成图像并计算 FID：

```bash
python -m unified_model.inference --model conditional_diffusion --model_path out/cond_diffusion/conditional_diffusion/epoch_100/model.pth --sampler ddpm --n_steps 1000 --save_by_class --n_samples_per_class 100 --output_dir out/samples
python cal_fid.py --cond_diffusion
```

请在报告中展示条件生成结果、条件去噪过程可视化图像，并报告每个类别的 FID 分数。

### 4. 文生图扩散模型与 CFG

请阅读 `text_to_image.py`、`text_to_image_utils.py` 以及本地 `diffusers/` 中与推理相关的代码，理解一个 Stable Diffusion 风格文生图模型如何从随机 latent 逐步生成图像。你需要关注以下组件在推理中的作用：

- tokenizer 与 text encoder：将 prompt 转换为文本条件表示；
- UNet：根据当前 noisy latent、时间步和文本条件预测去噪方向；
- scheduler：根据模型预测更新 latent；
- VAE：将最终 latent 解码成 RGB 图像；
- CFG：调节文本条件对生成结果的影响强度。

完成代码中的 CFG 相关部分后，使用不同的 `guidance_scale` 生成图像。建议至少实验四组取值，例如 `1.0, 3.0, 7.5, 12.0`。请在报告中比较不同 CFG 设置对以下方面的影响：

- prompt 匹配程度；
- 图像多样性；
- 颜色饱和度与过锐化现象；
- 局部伪影或不自然细节。

模型文件会在第一次运行 `text_to_image.py` 时自动下载。默认下载位置为 `~/.cache/t2i_model/small-stable-diffusion-v0`；如果需要使用其他位置，可以通过 `--model_dir` 指定。

下载时脚本会优先检查 `https://huggingface.co` 是否可访问；如果官方 endpoint 不可达，会自动切换到 `https://hf-mirror.com/`。

运行文生图模型：

```bash
python text_to_image.py \
  --model_dir ~/.cache/t2i_model/small-stable-diffusion-v0 \
  --prompt "an apple, 4k" \
  --steps 15 \
  --guidance_scale 7.5
```

注意：`OFA-Sys/small-stable-diffusion-v0` 是 Stable Diffusion 风格模型，默认推理分辨率对应 `512x512` 图像（latent 为 `64x64`）。本作业代码已经固定使用 `512x512`。请不要改用很低的分辨率，否则 VAE 和 UNet 会偏离模型熟悉的数据分布，生成结果可能发黑、颜色异常，或者与 prompt 明显无关。

CFG 实验示例：

```bash
for cfg in 1.0 3.0 7.5 12.0; do
  python text_to_image.py \
    --model_dir ~/.cache/t2i_model/small-stable-diffusion-v0 \
    --prompt "an apple, 4k" \
    --steps 15 \
    --guidance_scale "$cfg"
done
```

输出文件名会根据 prompt 自动生成，例如 `an_apple_4k.png`。

## 提交内容

请提交代码、报告 PDF、生成图像与必要 checkpoint。请不要提交数据集缓存、模型下载缓存、`*.so`、`*.pyd`、临时输出目录或无关环境文件。
