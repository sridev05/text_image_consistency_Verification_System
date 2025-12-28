import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("data/images/test.jpg").convert("RGB")

# Text to compare
text = "a man with a beard and a white shirt"

# Prepare inputs
inputs = processor(
    text=[text],
    images=image,
    return_tensors="pt",
    padding=True
)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Similarity score calculation
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds

# Normalize
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

similarity = (image_embeds @ text_embeds.T).item()

print("Similarity score:", similarity)
