from diffusers import MarigoldDepthPipeline
from PIL import Image
import torchvision
from diffusers.utils.torch_utils import randn_tensor
import torch

ppl = MarigoldDepthPipeline.from_pretrained('prs-eth/marigold-depth-lcm-v1-0')

vae = ppl.vae
scheduler = ppl.scheduler
text_encoder = ppl.text_encoder
tokenizer = ppl.tokenizer
unet = ppl.unet

device = 'cuda'

image = Image.open('data/depth/0000000000.png').convert('RGB')

transform = torchvision.transforms.Compose([

    torchvision.transforms.ToTensor(),
])
prompt = ''
text_inputs = tokenizer(
    prompt,
    padding="do_not_pad",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_input_ids = text_inputs.input_ids.to(device)
empty_text_embedding = text_encoder(text_input_ids)[0]
print(f"Shape of empty_text_embedding: {empty_text_embedding.shape}")

generator = torch.Generator(device=device)
latent = vae.encode(image)
latent = latent * vae.config.scaling_factor

pred_latent = randn_tensor(
    latent.shape,
    generator=generator,
    device=latent.device,
    dtype=latent.dtype,
)

batch_image_latent = latent
batch_pred_latent = pred_latent
effective_batch_size = batch_image_latent.shape[0]
text = empty_text_embedding[:effective_batch_size]  # [B,2,1024]

time_step_to_denoise = 50
scheduler.set_timesteps(time_step_to_denoise, device=device)
for t in range(time_step_to_denoise):
    batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
    noise = unet(batch_latent, t, encoder_hidden_states=text, return_dict=False)[0]  # [B,4,h,w]
    batch_pred_latent = scheduler.step(
        noise, t, batch_pred_latent, generator=generator
    ).prev_sample  # [B,4,h,w]

depth = vae.decode(batch_pred_latent / vae.config.scaling_factor, return_dict=False)[0]
depth = depth.mean(dim=1, keepdim=True)
depth = torch.clip(depth, -1, 1)
depth = (depth + 1) / 2
