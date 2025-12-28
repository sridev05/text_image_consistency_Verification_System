from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load image
image = Image.open("data/images/test.jpg").convert("RGB")

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Prepare input
inputs = processor(image, return_tensors="pt")

# Generate caption
with torch.no_grad():
    output = model.generate(**inputs)

# Decode caption
caption = processor.decode(output[0], skip_special_tokens=True)

print("Generated Caption:")
print(caption)
