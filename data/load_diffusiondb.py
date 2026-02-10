from datasets import load_dataset
import os

dataset = load_dataset("poloclub/diffusiondb", "large_random_1k")

print(dataset)
print(dataset["train"][0].keys())
