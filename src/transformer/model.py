"""PhoBERT model builder with weighted CrossEntropyLoss."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


def build_model(
    model_name: str,
    num_labels: int,
    class_weights: torch.Tensor | None = None,
) -> AutoModelForSequenceClassification:
    """Load PhoBERT for sequence classification (full fine-tuning)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    return model


class WeightedLossTrainer:
    """Manual training loop with weighted CE, AdamW, warmup, clipping."""

    def __init__(
        self,
        model,
        device: torch.device,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        epochs: int = 3,
        class_weights: torch.Tensor | None = None,
        gradient_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.gradient_clip = gradient_clip

        # AdamW with correct param groups (no decay for bias / LayerNorm)
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        self.warmup_ratio = warmup_ratio
        self._scheduler = None

        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

    def setup_scheduler(self, total_steps: int):
        from transformers import get_linear_schedule_with_warmup

        warmup_steps = int(total_steps * self.warmup_ratio)
        self._scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self, dataloader, log_every: int = 20, log_path: str = None) -> dict:
        import time

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        def _log(msg):
            line = msg + "\n"
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            else:
                import sys
                sys.stdout.write(line)
                sys.stdout.flush()

        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = self.loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            self.optimizer.step()
            if self._scheduler:
                self._scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (i + 1) % log_every == 0 or (i + 1) == len(dataloader):
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(dataloader) - i - 1)
                _log(
                    f"  batch {i+1}/{len(dataloader)}  "
                    f"loss={loss.item():.4f}  avg={total_loss/n_batches:.4f}  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
                )

        return {"train_loss": round(total_loss / max(n_batches, 1), 6)}

    @torch.no_grad()
    def predict(self, dataloader) -> dict:
        """Return predictions, true labels, ids, texts."""
        self.model.eval()
        all_preds, all_labels, all_ids, all_texts = [], [], [], []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_ids.extend(batch["id"])
            all_texts.extend(batch["text"])

        return {
            "predictions": all_preds,
            "labels": all_labels,
            "ids": all_ids,
            "texts": all_texts,
        }
