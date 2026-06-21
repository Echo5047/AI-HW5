import argparse
import os

import torch

from text_to_image_utils import (
    DEFAULT_MODEL_DIR,
    IMAGE_SIZE,
    choose_device,
    encode_prompt,
    ensure_model_downloaded,
    latents_to_pil,
    load_models,
    prompt_to_filename,
)


def parse_args():
    parser = argparse.ArgumentParser(description="HW5 minimal Stable Diffusion inference")
    parser.add_argument("--prompt", type=str, default="an apple, 4k")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


@torch.no_grad()
def generate_image(args):
    device = choose_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer, text_encoder, unet, vae, scheduler = load_models(args.model_dir, device, dtype)
    uncond_emb, text_emb = encode_prompt(
        tokenizer, text_encoder, args.prompt, args.negative_prompt, device, dtype
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_shape = (1, unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)

    scheduler.set_timesteps(args.steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    for timestep in scheduler.timesteps:
        latent_model_input = scheduler.scale_model_input(latents, timestep)
        t = timestep.expand(latent_model_input.shape[0])

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states=uncond_emb,
        ).sample
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_emb,
        ).sample

        #########################################################
        #                  TODO: CFG Implementation             #
        #########################################################
        # Experiment with different guidance_scale values and
        # describe how they affect prompt alignment, diversity,
        # saturation, and artifacts in your report.
        #########################################################
        # Your code here.
        #########################################################
        #                     End of TODO                       #
        #########################################################

        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents_to_pil(vae, latents)


def main():
    args = parse_args()
    args.model_dir = ensure_model_downloaded(model_dir=args.model_dir)

    image = generate_image(args)
    output_path = prompt_to_filename(args.prompt)
    image.save(output_path)
    print(f"Saved {IMAGE_SIZE}x{IMAGE_SIZE} image to {output_path}")


if __name__ == "__main__":
    main()
