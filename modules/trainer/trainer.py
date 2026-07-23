import os
import torch
from torch.amp import autocast
from tqdm import tqdm

from modules.metrics.metrics import evaluate

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
        grad_clip=3.0,
        mixed_precision=False,
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

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        # Track the lowest EER instead of classification accuracy
        self.best_eer = float('inf')
        self.start_epoch = 1

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_eer": [],
            "val_acc": [],
            "lr": []
        }

    # Save Checkpoint
    def save_checkpoint(
        self,
        epoch,
        is_best=False
    ):

        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler":
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
            "best_eer": self.best_eer,
            "history": self.history
        }

        latest_path = os.path.join(
            self.checkpoint_dir,
            "latest_model.pt"
        )

        torch.save(
            checkpoint,
            latest_path
        )

        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                "best_model.pt"
            )

            torch.save(
                checkpoint,
                best_path
            )

    # Load Checkpoint
    def load_checkpoint(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False
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

        if (
            self.scheduler is not None
            and checkpoint["scheduler"] is not None
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        self.best_eer = checkpoint.get(
            "best_eer",
            float('inf')
        )

        self.history = checkpoint.get(
            "history",
            self.history
        )

        self.start_epoch = checkpoint["epoch"] + 1

        print("=" * 70)
        print("Checkpoint Loaded Successfully")
        print(f"Resume Training From Epoch {self.start_epoch}")
        print("=" * 70)

    # Train One Epoch
    def train_epoch(
        self,
        epoch
    ):

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

            features = features.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                device_type="cuda",
                enabled=self.mixed_precision
            ):

                embeddings = self.model(
                    features
                )

                logits, loss = self.classifier(
                    embeddings,
                    labels
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.classifier.parameters()),
                self.grad_clip
            )

            self.optimizer.step()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            batch_size = labels.size(0)

            running_correct += (
                predictions == labels
            ).sum().item()

            running_total += batch_size

            running_loss += (
                loss.item() * batch_size
            )

            avg_loss = (
                running_loss /
                running_total
            )

            avg_acc = (
                100.0 *
                running_correct /
                running_total
            )

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.2f}%",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
            )

        epoch_loss = (
            running_loss /
            running_total
        )

        epoch_acc = (
            100.0 *
            running_correct /
            running_total
        )

        self.history["train_loss"].append(
            epoch_loss
        )

        self.history["train_acc"].append(
            epoch_acc
        )

        return (
            epoch_loss,
            epoch_acc
        )
        
    # Validation
    @torch.no_grad()
    def validate(self):
        
        self.model.eval()

        all_embeddings = []
        all_labels = []

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation"
        )

        for features, labels in progress_bar:

            features = features.to(
                self.device,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=self.mixed_precision
            ):

                embeddings = self.model(features)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = evaluate(all_embeddings, all_labels)
        
        epoch_eer = metrics["EER"] * 100.0
        epoch_acc = metrics["Accuracy"] * 100.0

        self.history["val_eer"].append(epoch_eer)
        self.history["val_acc"].append(epoch_acc)

        return (
            epoch_eer,
            epoch_acc
        )
        
    # Training Loop
    def fit(self):

        print("=" * 70)
        print("Starting ECAPA-TDNN Training...")
        print("=" * 70)

        for epoch in range(
            self.start_epoch,
            self.epochs + 1
        ):

            # Training
            train_loss, train_acc = self.train_epoch(
                epoch
            )

            # Validation
            val_eer, val_acc = self.validate()

            # Learning Rate Scheduler
            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # EER 
                    self.scheduler.step(
                        val_eer
                    )

                else:

                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["lr"].append(
                current_lr
            )

            # Save Best Model
            is_best = False

            #lowest EER
            if val_eer < self.best_eer:

                self.best_eer = val_eer
                is_best = True

            self.save_checkpoint(
                epoch,
                is_best=is_best
            )

            # Epoch Summary
            print()
            print("-" * 70)

            print(f"Epoch {epoch}/{self.epochs}")
            print(f"Learning Rate       : {current_lr:.8f}")
            print(f"Train Loss          : {train_loss:.4f}")
            print(f"Train Accuracy      : {train_acc:.2f}%")
            print(f"Validation EER      : {val_eer:.4f}%")
            print(f"Validation Ver. Acc : {val_acc:.2f}%")
            print(f"Best Validation EER : {self.best_eer:.4f}%")
            
            print("-" * 70)
            print()

        print("=" * 70)
        print("Training Finished Successfully")
        print("=" * 70)

        return self.history