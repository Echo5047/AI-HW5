import os
import re
import urllib.error
import urllib.request

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel


DEFAULT_MODEL_DIR = "~/.cache/t2i_model/small-stable-diffusion-v0"
DEFAULT_REPO_ID = "OFA-Sys/small-stable-diffusion-v0"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
MIRROR_HF_ENDPOINT = "https://hf-mirror.com"
IMAGE_SIZE = 512

REQUIRED_MODEL_PATHS = [
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
]

ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "unet/*",
    "vae/*",
]


def model_files_exist(model_dir):
    model_dir = os.path.expanduser(model_dir)
    return all(os.path.exists(os.path.join(model_dir, path)) for path in REQUIRED_MODEL_PATHS)


def endpoint_reachable(endpoint, timeout=5):
    request = urllib.request.Request(endpoint, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except (OSError, urllib.error.URLError):
        return False


def choose_download_endpoint(primary_endpoint, mirror_endpoint):
    if endpoint_reachable(primary_endpoint):
        print(f"Using Hugging Face endpoint: {primary_endpoint}")
        return primary_endpoint
    print(f"{primary_endpoint} is not reachable; falling back to {mirror_endpoint}")
    return mirror_endpoint


def snapshot_download_with_fallback(repo_id, model_dir, primary_endpoint, mirror_endpoint):
    endpoint = choose_download_endpoint(primary_endpoint, mirror_endpoint)
    try:
        return snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            endpoint=endpoint,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception:
        if endpoint == mirror_endpoint:
            raise
        print(f"Download from {primary_endpoint} failed; retrying with {mirror_endpoint}")
        return snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            endpoint=mirror_endpoint,
            allow_patterns=ALLOW_PATTERNS,
        )


def ensure_model_downloaded(
    model_dir,
    repo_id=DEFAULT_REPO_ID,
    local_files_only=False,
    hf_endpoint=DEFAULT_HF_ENDPOINT,
    hf_mirror_endpoint=MIRROR_HF_ENDPOINT,
):
    model_dir = os.path.expanduser(model_dir)
    if model_files_exist(model_dir):
        return model_dir
    if local_files_only:
        missing = [path for path in REQUIRED_MODEL_PATHS if not os.path.exists(os.path.join(model_dir, path))]
        raise FileNotFoundError(
            f"Model files are missing from {model_dir}: {missing}. "
            "Run without --local_files_only to download them."
        )
    print(f"Downloading {repo_id} to {model_dir} ...")
    return snapshot_download_with_fallback(repo_id, model_dir, hf_endpoint, hf_mirror_endpoint)


def choose_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prompt_to_filename(prompt):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt).strip("_")
    return f"{name or 'sample'}.png"


def load_models(model_dir, device, dtype):
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(model_dir, "text_encoder"))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(model_dir, "unet"))
    vae = AutoencoderKL.from_pretrained(os.path.join(model_dir, "vae"))
    scheduler = DPMSolverMultistepScheduler.from_pretrained(os.path.join(model_dir, "scheduler"))

    text_encoder.to(device=device, dtype=dtype).eval()
    unet.to(device=device, dtype=dtype).eval()
    vae.to(device=device, dtype=dtype).eval()
    return tokenizer, text_encoder, unet, vae, scheduler


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt, negative_prompt, device, dtype):
    text = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_emb = text_encoder(text.input_ids.to(device))[0].to(dtype=dtype)
    uncond_emb = text_encoder(uncond.input_ids.to(device))[0].to(dtype=dtype)
    return uncond_emb, text_emb


def latents_to_pil(vae, latents):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = (image * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(image)
