import os
from torchvision import datasets

# Output directory
output_dir = "./test_fashion_images"
os.makedirs(output_dir, exist_ok=True)

# Load test dataset (returns PIL images by default)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True)

# Save first 10 images directly
for i in range(10):
    image, label = test_dataset[i]  # image is already a PIL.Image.Image
    image.save(os.path.join(output_dir, f"test_img_{i}_label_{label}.png"))

print(f"✅ Saved 10 test images to {output_dir}")
