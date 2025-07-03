# Lulu
# Install dependencies (if not already installed)
!pip install diffusers transformers accelerate safetensors --quiet

# Import required libraries
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# Set the prompt
prompt = "A futuristic samurai standing on a neon-lit rooftop during a rainstorm, cinematic lighting, high detail, 4K resolution, Blade Runner vibes."

# Generate the image
image = pipe(prompt, guidance_scale=7.5).images[0]

# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()

# Optionally save the image
image.save("futuristic_samurai.png")
