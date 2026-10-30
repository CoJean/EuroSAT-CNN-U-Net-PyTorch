import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


# =========================================================
# CONFIG
# =========================================================
SEED = 42
DATA_DIR = "./EuroSAT"   # folderul cu clasele EuroSAT
IMAGE_SIZE = 64          # EuroSAT RGB este 64x64
BATCH_SIZE = 64
NUM_EPOCHS_CNN = 15
NUM_EPOCHS_UNET = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FILE = "logs.txt"
IMAGES_DIR = "images"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# =========================================================
# SETUP OUTPUTS
# =========================================================
os.makedirs(IMAGES_DIR, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== TRAINING LOG ===\n")


def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


# =========================================================
# HELPERS
# =========================================================
def denormalize(img_tensor):
    mean = torch.tensor(
        [0.5, 0.5, 0.5], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def show_batch(images, labels, class_names, n=6, prefix="batch"):
    n = min(n, len(images))

    for i in range(n):
        img = denormalize(images[i].cpu()).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(class_names[labels[i].item()])
        plt.axis("off")

        save_path = os.path.join(IMAGES_DIR, f"{prefix}_{i}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        log(f"Saved sample batch image: {save_path}")


def plot_history(train_losses, val_losses, val_scores, score_name, title_prefix):
    epochs = range(1, len(train_losses) + 1)
    safe_title = safe_filename(title_prefix)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = os.path.join(IMAGES_DIR, f"{safe_title}_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    log(f"Saved plot: {loss_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_scores, label=score_name)
    plt.xlabel("Epoch")
    plt.ylabel(score_name)
    plt.title(f"{title_prefix} - {score_name}")
    plt.legend()
    plt.tight_layout()

    score_path = os.path.join(IMAGES_DIR, f"{safe_title}_score.png")
    plt.savefig(score_path, bbox_inches="tight")
    plt.close()
    log(f"Saved plot: {score_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    save_path = os.path.join(IMAGES_DIR, safe_filename(title) + ".png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    log(f"Saved confusion matrix: {save_path}")

# =========================================================
# 1) CNN FOR CLASSIFICATION
# =========================================================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_cnn_classification():
    log("\n================ CNN CLASSIFICATION ================\n")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)

    log(f"Total images: {len(dataset)}")
    log(f"Detected classes: {dataset.classes}")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    images, labels = next(iter(train_loader))
    show_batch(images, labels, dataset.classes, n=6, prefix="cnn_batch")

    model = CNNClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    for epoch in range(NUM_EPOCHS_CNN):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        log(
            f"[CNN] Epoch {epoch+1}/{NUM_EPOCHS_CNN} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_eurosat_cnn.pth")
            log("Saved best CNN model to best_eurosat_cnn.pth")

    log(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    plot_history(
        history_train_loss,
        history_val_loss,
        history_val_acc,
        score_name="Validation Accuracy (%)",
        title_prefix="EuroSAT CNN"
    )

    model.load_state_dict(torch.load(
        "best_eurosat_cnn.pth",
        map_location=DEVICE,
        weights_only=True
    ))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = 100.0 * correct / total
    log(f"Test Accuracy: {test_acc:.2f}%")

    # ================= CONFUSION MATRIX =================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes,
                yticklabels=dataset.classes)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - CNN")

    cm_path = os.path.join(IMAGES_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    log(f"Saved confusion matrix: {cm_path}")

# ================= METRICS =================
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    log(f"Accuracy: {accuracy:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall: {recall:.4f}")
    log(f"F1-score: {f1:.4f}")

    return model, dataset.classes, test_loader


# =========================================================
# 2) U-NET FOR WEAKLY-SUPERVISED "SEGMENTATION"
#    IMPORTANT: pseudo-masks, not real segmentation labels
# =========================================================
class EuroSATPseudoSegmentationDataset(Dataset):
    """
    Creează pseudo-măști:
    fiecare pixel primește clasa imaginii.
    Asta NU este segmentare semantică reală, ci un demo
    pentru cerința U-Net pe un dataset de clasificare.
    """

    def __init__(self, root_dir):
        self.base_dataset = datasets.ImageFolder(
            root_dir,
            transform=transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
        )
        self.num_classes = len(self.base_dataset.classes)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        mask = torch.full((IMAGE_SIZE, IMAGE_SIZE),
                          fill_value=label, dtype=torch.long)
        return image, mask, label


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=10):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        bn = self.bottleneck(p3)

        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        out = self.final(u1)
        return out


def pixel_accuracy(logits, masks):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == masks).sum().item()
    total = masks.numel()
    return correct / total


def visualize_unet_predictions(model, loader, class_names, num_examples=4):
    model.eval()
    shown = 0

    with torch.no_grad():
        for images, masks, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu()

            for i in range(images.size(0)):
                if shown >= num_examples:
                    return

                img = denormalize(images[i].cpu()).permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                gt_mask = masks[i].cpu().numpy()
                pred_mask = preds[i].numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title(
                    f"Image\nLabel: {class_names[labels[i].item()]}")
                axes[0].axis("off")

                axes[1].imshow(gt_mask, cmap="tab20")
                axes[1].set_title("Pseudo Ground Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(pred_mask, cmap="tab20")
                axes[2].set_title("U-Net Prediction")
                axes[2].axis("off")

                plt.tight_layout()
                save_path = os.path.join(
                    IMAGES_DIR, f"unet_result_{shown}.png")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

                log(f"Saved U-Net visualization: {save_path}")
                shown += 1


def train_unet_demo():
    log("\n================ U-NET SEGMENTATION DEMO ================\n")

    dataset = EuroSATPseudoSegmentationDataset(DATA_DIR)
    num_classes = dataset.num_classes

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = UNet(in_channels=3, out_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_pixel_acc = 0.0
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    for epoch in range(NUM_EPOCHS_UNET):
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                val_acc_sum += pixel_accuracy(logits, masks)

        val_loss /= len(val_loader)
        val_pixel_acc = 100.0 * (val_acc_sum / len(val_loader))

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_pixel_acc)

        log(
            f"[U-Net] Epoch {epoch+1}/{NUM_EPOCHS_UNET} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Pixel Acc: {val_pixel_acc:.2f}%"
        )

        if val_pixel_acc > best_val_pixel_acc:
            best_val_pixel_acc = val_pixel_acc
            torch.save(model.state_dict(), "best_eurosat_unet_demo.pth")
            log("Saved best U-Net demo model to best_eurosat_unet_demo.pth")

    log(f"\nBest validation pixel accuracy: {best_val_pixel_acc:.2f}%")

    plot_history(
        history_train_loss,
        history_val_loss,
        history_val_acc,
        score_name="Validation Pixel Accuracy (%)",
        title_prefix="EuroSAT U-Net Demo"
    )

    model.load_state_dict(torch.load(
        "best_eurosat_unet_demo.pth",
        map_location=DEVICE,
        weights_only=True
    ))
    model.eval()

    test_acc_sum = 0.0
    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            test_acc_sum += pixel_accuracy(logits, masks)

    test_pixel_acc = 100.0 * (test_acc_sum / len(test_loader))
    log(f"Test Pixel Accuracy: {test_pixel_acc:.2f}%")

    visualize_unet_predictions(
        model, test_loader, dataset.base_dataset.classes, num_examples=4
    )

    return model


# =========================================================
# OPTIONAL: CNN INFERENCE ON SINGLE IMAGE
# =========================================================
def predict_single_image(model, class_names, image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"Predicted class: {class_names[pred]}")
    plt.axis("off")

    save_path = os.path.join(IMAGES_DIR, "single_prediction.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    log(f"Saved single image prediction visualization: {save_path}")
    log(f"Predicted class for {image_path}: {class_names[pred]}")

    return class_names[pred]


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    log(f"Using device: {DEVICE}")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Nu găsesc folderul datasetului: {DATA_DIR}\n"
            f"Pune imaginile EuroSAT într-un folder local, de ex. ./EuroSAT"
        )

    cnn_model, class_names, test_loader = train_cnn_classification()
    unet_model = train_unet_demo()

    # exemplu de inferență pe o imagine:
    # sample_img = "./EuroSAT/Forest/Forest_1.jpg"
    # pred = predict_single_image(cnn_model, class_names, sample_img)
    # log(f"Predicție: {pred}")
