import os

from tqdm import tqdm

import torch
from torch.amp import autocast


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

        self.best_accuracy = 0.0
        self.start_epoch = 1

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

    ############################################################
    # Save Checkpoint
    ############################################################

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

            "best_accuracy": self.best_accuracy,

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

    ############################################################
    # Load Checkpoint
    ############################################################

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

        self.best_accuracy = checkpoint.get(
            "best_accuracy",
            0.0
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

    ############################################################
    # Train One Epoch
    ############################################################

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

    ############################################################
    # Validation
    ############################################################

    @torch.no_grad()
    def validate(self):

        self.model.eval()
        self.classifier.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation"
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
                acc=f"{avg_acc:.2f}%"
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

        self.history["val_loss"].append(
            epoch_loss
        )

        self.history["val_acc"].append(
            epoch_acc
        )

        return (
            epoch_loss,
            epoch_acc
        )
    ############################################################
    # Training Loop
    ############################################################

    def fit(self):

        print("=" * 70)
        print("Starting ECAPA-TDNN Training...")
        print("=" * 70)

        for epoch in range(
            self.start_epoch,
            self.epochs + 1
        ):

            # ---------------------------------------
            # Training
            # ---------------------------------------

            train_loss, train_acc = self.train_epoch(
                epoch
            )

            # ---------------------------------------
            # Validation
            # ---------------------------------------

            val_loss, val_acc = self.validate()

            # ---------------------------------------
            # Learning Rate Scheduler
            # ---------------------------------------

            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):

                    self.scheduler.step(
                        val_loss
                    )

                else:

                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["lr"].append(
                current_lr
            )

            # ---------------------------------------
            # Save Best Model
            # ---------------------------------------

            is_best = False

            if val_acc > self.best_accuracy:

                self.best_accuracy = val_acc
                is_best = True

            self.save_checkpoint(
                epoch,
                is_best=is_best
            )

            # ---------------------------------------
            # Epoch Summary
            # ---------------------------------------

            print()
            print("-" * 70)

            print(
                f"Epoch {epoch}/{self.epochs}"
            )

            print(
                f"Learning Rate       : {current_lr:.8f}"
            )

            print(
                f"Train Loss          : {train_loss:.4f}"
            )

            print(
                f"Train Accuracy      : {train_acc:.2f}%"
            )

            print(
                f"Validation Loss     : {val_loss:.4f}"
            )

            print(
                f"Validation Accuracy : {val_acc:.2f}%"
            )

            print(
                f"Best Validation Acc : {self.best_accuracy:.2f}%"
            )

            print("-" * 70)
            print()

        print("=" * 70)
        print("Training Finished Successfully")
        print("=" * 70)

        return self.history