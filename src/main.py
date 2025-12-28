from PIL import Image

# Path to test image
image_path = "data/images/test.jpg"


# Load image
img = Image.open(image_path)

# Sample text
text = "A man riding a motorcycle on a road"

print("Image loaded successfully")
print("Image size:", img.size)
print("Text:", text)
