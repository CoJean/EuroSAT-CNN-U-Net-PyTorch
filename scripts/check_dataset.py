from pathlib import Path
from torchvision.datasets import ImageFolder

DATA_DIR = Path("./EuroSAT")

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}\n"
        "Download and extract EuroSAT RGB, then place the 10 class folders inside ./EuroSAT."
    )

dataset = ImageFolder(str(DATA_DIR))
print(f"Dataset directory: {DATA_DIR.resolve()}")
print(f"Number of images: {len(dataset):,}")
print(f"Number of classes: {len(dataset.classes)}")
for class_name, count in sorted(
    ((name, sum(1 for _, label in dataset.samples if dataset.classes[label] == name))
     for name in dataset.classes),
    key=lambda item: item[0],
):
    print(f"- {class_name}: {count:,} images")
