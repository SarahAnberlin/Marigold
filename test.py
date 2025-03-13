import os

from diffusers import MarigoldDepthPipeline, DDPMScheduler
from PIL import Image
import torchvision
from diffusers.utils.torch_utils import randn_tensor
import torch
from torchvision.utils import save_image

device = 'cuda'

ppl = MarigoldDepthPipeline.from_pretrained('prs-eth/marigold-depth-lcm-v1-0')

vae = ppl.vae
scheduler = ppl.scheduler
text_encoder = ppl.text_encoder
tokenizer = ppl.tokenizer
unet = ppl.unet
vae = vae.to(device)
scheduler = scheduler
text_encoder = text_encoder.to(device)
unet = unet.to(device)
vae.eval()
text_encoder.eval()
unet.eval()

image = Image.open('plain.jpg').convert('RGB')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((512, 512)),
])

image = transform(image).unsqueeze(0).to(device)

prompt = ''
text_inputs = tokenizer(
    prompt,
    padding="do_not_pad",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids.to(device)
save_root = 'output'
os.makedirs(save_root, exist_ok=True)
with torch.no_grad():
    empty_text_embedding = text_encoder(text_input_ids)[0]
    print(f"Shape of empty_text_embedding: {empty_text_embedding.shape}")

    generator = torch.Generator(device=device)
    latent = vae.encode(image)['latent_dist'].mode()
    print(f"Shape of latent: {latent.shape}")
    # print(f"latent keys: {latent.keys()}")
    latent = latent * vae.config.scaling_factor

    pred_latent = randn_tensor(
        latent.shape,
        generator=generator,
        device=latent.device,
        dtype=latent.dtype,
    )
    rgb_noise_scheduler = DDPMScheduler.from_pretrained('google/ddpm-ema-celebahq-256', device=device)
    rgb_scheduler_timestep = 1000
    rgb_noise_scheduler.set_timesteps(rgb_scheduler_timestep, device=device)
    timesteps = torch.tensor([i for i in range(rgb_scheduler_timestep)]).to(device)

    for rgb_t in timesteps:
        noise = torch.randn(
            latent.shape,
            device=device,
            generator=generator,
        )  # [B, 4, h, w]
        noisy_latent = rgb_noise_scheduler.add_noise(
            latent, noise, rgb_t,
        )  # [B, 4, h, w]

        batch_image_latent = noisy_latent
        batch_pred_latent = pred_latent
        effective_batch_size = batch_image_latent.shape[0]
        text = empty_text_embedding[:effective_batch_size]  # [B,2,1024]

        time_step_to_denoise = 1000
        scheduler.set_timesteps(time_step_to_denoise, device=device)
        for depth_denoise_t in scheduler.timesteps:
            batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
            print(f"Shape of batch_latent: {batch_latent.shape}")
            noise = unet(batch_latent, depth_denoise_t, encoder_hidden_states=text, return_dict=False)[0]  # [B,4,h,w]
            print(f"Shape of noise: {noise.shape}")
            batch_pred_latent = scheduler.step(
                noise, depth_denoise_t, batch_pred_latent, generator=generator
            ).prev_sample  # [B,4,h,w]
            print(f"Shape of batch_pred_latent: {batch_pred_latent.shape}")

        depth = vae.decode(batch_pred_latent / vae.config.scaling_factor, return_dict=False)[0]
        rgb = vae.decode(noisy_latent / vae.config.scaling_factor, return_dict=False)[0]
        depth = depth.mean(dim=1, keepdim=True)
        depth = torch.clip(depth, -1, 1)
        depth = (depth + 1) / 2

        save_image(depth, f"{save_root}/depth_{rgb_t}.png")
        save_image(rgb, f"{save_root}/rgb_{rgb_t}.png")
