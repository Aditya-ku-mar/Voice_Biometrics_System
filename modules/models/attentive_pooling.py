import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from modules.metrics.metrics import (
    verification_accuracy,
    compute_eer,
    compute_min_dcf
)


class Trainer:

    def __init__(
        self,
        model,
        classifier,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs,
        checkpoint_dir="checkpoints",
        grad_clip=5.0,
        mixed_precision=True
    ):

        self.model = model.to(device)

        self.classifier = classifier.to(device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.epochs = epochs

        self.grad_clip = grad_clip

        self.mixed_precision = mixed_precision

        self.scaler = GradScaler(enabled=mixed_precision)

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_accuracy = 0.0

        self.start_epoch = 1

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "eer": [],
            "mindcf": []
        }
    def save_checkpoint(self, epoch, is_best=False):

        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "history": self.history,
        }

        latest_path = os.path.join(
            self.checkpoint_dir,
            "latest_model.pt"
        )

        torch.save(state, latest_path)

        if is_best:

            best_path = os.path.join(
                self.checkpoint_dir,
                "best_model.pt"
            )

            torch.save(state, best_path)
    def load_checkpoint(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.classifier.load_state_dict(
            checkpoint["classifier"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        self.best_accuracy = checkpoint["best_accuracy"]

        self.history = checkpoint["history"]

        self.start_epoch = checkpoint["epoch"] + 1

        print("=" * 60)
        print("Checkpoint Loaded Successfully")
        print(f"Resume Epoch : {self.start_epoch}")
        print("=" * 60)
        
    def train_epoch(self, epoch):
        self.model.train()
        self.classifier.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch}/{self.epochs}]"
        )

        for features, labels in progress_bar:

            features = features.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):

                embeddings = self.model(features)

                logits, loss = self.classifier(embeddings,labels)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            predictions = torch.argmax(logits,dim=1)

            running_correct += (predictions == labels).sum().item()

            running_total += labels.size(0)

            running_loss += (loss.item() * labels.size(0))

        avg_loss = running_loss / running_total

        avg_acc = (100.0 *running_correct / running_total)

        progress_bar.set_postfix(loss=f"{avg_loss:.4f}",acc=f"{avg_acc:.2f}%")

    epoch_loss = running_loss / running_total

    epoch_acc = (100.0 *running_correct /running_total)

    self.history["train_loss"].append(epoch_loss)

    self.history["train_acc"].append(epoch_acc)

    return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):

        self.model.eval()
        self.classifier.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_embeddings = []
    all_labels = []

    progress_bar = tqdm(
        self.val_loader,
        desc="Validation"
    )

    for features, labels in progress_bar:

        features = features.to(self.device)
        labels = labels.to(self.device)

        with autocast(enabled=self.mixed_precision):

            embeddings = self.model(features)

            logits, loss = self.classifier( embeddings,labels)

        predictions = torch.argmax(logits,dim=1 )

        running_correct += ( predictions == labels).sum().item()

        running_total += labels.size(0)

        running_loss += (loss.item() * labels.size(0))
        

        all_embeddings.append(embeddings.detach().cpu())

        all_labels.append(labels.detach().cpu())

        avg_loss = running_loss / running_total

        avg_acc = (100.0 * running_correct /running_total)

        progress_bar.set_postfix(loss=f"{avg_loss:.4f}",acc=f"{avg_acc:.2f}%")

    epoch_loss = running_loss / running_total

    epoch_acc = (100.0 * running_correct /running_total)

        embeddings = torch.cat( all_embeddings,dim=0)

        labels = torch.cat(all_labels,dim=0)

        accuracy = verification_accuracy(embeddings,labels)

        eer = compute_eer(embeddings,labels)

        mindcf = compute_min_dcf(embeddings,labels)

        self.history["val_loss"].append(epoch_loss)

        self.history["val_acc"].append(epoch_acc)

        self.history["eer"].append(err)

        self.history["mindcf"].append(mindcf)

        return (
            epoch_loss,
            epoch_acc,
            eer,
            mindcf
        )
    def fit(self):
        print("=" * 70)
        print("Starting Training...")
        print("=" * 70)

        for epoch in range(self.start_epoch, self.epochs + 1):

            train_loss, train_acc = self.train_epoch(epoch)(
                val_loss,
                val_acc,
                eer,
                mindcf
                ) = self.validate()
           
            if self.scheduler is not None:
                self.scheduler.step()

            is_best = False

            if val_acc > self.best_accuracy:

                self.best_accuracy = val_acc

                is_best = True

            self.save_checkpoint(epoch,is_best=is_best)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print()
            print("-" * 70)
            print(f"Epoch {epoch}/{self.epochs}")
            print(f"Learning Rate : {current_lr:.8f}")
            print(f"Train Loss    : {train_loss:.4f}")
            print(f"Train Acc     : {train_acc:.2f}%")
            print(f"Valid Loss    : {val_loss:.4f}")
            print(f"Valid Acc     : {val_acc:.2f}%")
            print(f"EER           : {eer:.4f}")
            print(f"minDCF        : {mindcf:.4f}")
            print(f"Best Acc      : {self.best_accuracy:.2f}%")
            print("-" * 70)
            print()
            print("=" * 70)
            print("Training Finished Successfully")
            print("=" * 70)

            return self.history