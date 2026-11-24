from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from app.model import UNet

ROOT = Path(__file__).resolve().parent
SIZE, BATCH, EPOCHS, LR = 256, 4, 20, 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FloodDataset(Dataset):
    def __init__(self, split):
        self.images = ROOT / "data" / split / "images"
        self.masks = ROOT / "data" / split / "masks"
        self.paths = sorted([p for p in self.images.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"}])
        if not self.paths:
            raise RuntimeError(f"No images found in {self.images}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        mask_path = self.masks / (p.stem + ".png")
        if not mask_path.exists():
            mask_path = self.masks / p.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing matching mask for {p.name}")
        x = Image.open(p).convert("RGB").resize((SIZE,SIZE), Image.Resampling.BILINEAR)
        y = Image.open(mask_path).convert("L").resize((SIZE,SIZE), Image.Resampling.NEAREST)
        xa = np.asarray(x, dtype=np.float32) / 255.0
        ya = (np.asarray(y, dtype=np.float32) >= 128).astype(np.float32)
        return torch.from_numpy(xa).permute(2,0,1), torch.from_numpy(ya).unsqueeze(0)

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1,2,3)
    score = (2*(probs*targets).sum(dims)+eps)/(probs.sum(dims)+targets.sum(dims)+eps)
    return 1-score.mean()

def main():
    train_ds = FloodDataset("train")
    val_dir = ROOT / "data" / "val" / "images"
    has_val = val_dir.exists() and any(val_dir.iterdir())
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(FloodDataset("val"), batch_size=BATCH, shuffle=False, num_workers=0) if has_val else None
    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    best = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x,y in train_loader:
            x,y = x.to(DEVICE),y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = bce(logits,y) + dice_loss(logits,y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        score = total / max(1,len(train_loader))
        if val_loader:
            model.eval(); val_total = 0.0
            with torch.no_grad():
                for x,y in val_loader:
                    x,y = x.to(DEVICE),y.to(DEVICE)
                    val_total += (bce(model(x),y)+dice_loss(model(x),y)).item()
            score = val_total / max(1,len(val_loader))
        print(f"Epoch {epoch+1:02d}/{EPOCHS} - loss: {total/max(1,len(train_loader)):.4f} - validation: {score:.4f}")
        if score < best:
            best = score
            torch.save(model.state_dict(), ROOT / "models" / "best_unet.pth")
    print("Training complete. Saved models/best_unet.pth")

if __name__ == "__main__":
    main()
