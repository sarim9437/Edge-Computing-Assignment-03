"""
finetune.py
-----------
Generic fine-tuning pipeline shared by QAT and both pruning methods.
SGD + momentum with cosine LR decay. Optional per-epoch hook (QAT uses it to
freeze observers / BN statistics) and optional custom evaluation function
(QAT uses it to evaluate the real INT8 model after every epoch).
"""
import time

import torch

from common import evaluate, train_one_epoch


def fine_tune(model, train_loader, test_loader, device, epochs, lr, tag="FT",
              use_amp=None, epoch_hook=None, eval_fn=None, weight_decay=5e-4):
    if use_amp is None:
        use_amp = device.type == "cuda"
    if eval_fn is None:
        eval_fn = lambda m: evaluate(m, test_loader, device)
    if epochs <= 0:
        return eval_fn(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    acc = 0.0
    for epoch in range(1, epochs + 1):
        if epoch_hook is not None:
            epoch_hook(epoch)
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, use_amp)
        acc = eval_fn(model)
        print(f"  [{tag}] epoch {epoch}/{epochs} | loss {loss:.4f} | train {train_acc:6.2f}% "
              f"| test {acc:6.2f}% | {time.time() - t0:6.1f}s")
    return acc
