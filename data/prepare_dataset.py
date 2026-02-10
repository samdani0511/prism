from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset(
    "poloclub/diffusiondb",
    "large_random_1k",
    split="train",
    trust_remote_code=True   # 🔥 THIS LINE FIXES IT
)

os.makedirs("data/images", exist_ok=True)

records = []

for idx, sample in enumerate(dataset):
    image = sample["image"]
    prompt = sample["prompt"]

    image_path = f"data/images/{idx}.png"
    image.save(image_path)

    records.append({
        "id": idx,
        "prompt": prompt,
        "image_path": image_path,
        "category": "diffusiondb"
    })

pd.DataFrame(records).to_csv("data/dataset.csv", index=False)

print("✅ DiffusionDB dataset prepared successfully")
