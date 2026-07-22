
import os
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    def __init__(self, model, classifier, train_loader, val_loader,
                 optimizer, scheduler, device, epochs, checkpoint_dir):
        self.model = model.to(device)
        self.classifier = classifier.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.scaler = GradScaler()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_accuracy = 0.0
        self.start_epoch = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def save_checkpoint(self, epoch, best=False):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
            "best_accuracy": self.best_accuracy,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, "latest_model.pt"))
        if best:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pt"))

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.classifier.load_state_dict(ckpt["classifier"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt["scheduler"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.history = ckpt.get("history", self.history)
        self.best_accuracy = ckpt.get("best_accuracy", 0.0)
        self.start_epoch = ckpt["epoch"] + 1

    def train_epoch(self):
        self.model.train()
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(self.train_loader, desc="Training"):
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                embeddings = self.model(features)
                logits, loss = self.classifier(embeddings, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(self.val_loader, desc="Validation"):
            features = features.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(features)
            logits, loss = self.classifier(embeddings, labels)

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.val_loader), 100.0 * correct / total

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            if self.scheduler:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            best = val_acc > self.best_accuracy
            if best:
                self.best_accuracy = val_acc

            self.save_checkpoint(epoch, best)

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        return self.history

