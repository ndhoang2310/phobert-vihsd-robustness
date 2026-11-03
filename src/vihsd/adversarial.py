"""FGM weight-embedding variant with isolated, unscaled microbatch gradients."""
from contextlib import contextmanager
import torch


class FGM:
    def __init__(self, model, epsilon):
        self.embedding = model.get_input_embeddings()
        self.weight = self.embedding.weight
        self.epsilon = float(epsilon)
        if self.epsilon < 0:
            raise ValueError('epsilon must be nonnegative')

    def direction(self, loss, scaler):
        # autograd.grad does not read or write accumulated parameter .grad buffers.
        gradient, = torch.autograd.grad(scaler.scale(loss), self.weight, retain_graph=True)
        gradient = gradient.detach().float() / scaler.get_scale()
        if self.embedding.padding_idx is not None:
            gradient[self.embedding.padding_idx].zero_()
        norm = torch.linalg.vector_norm(gradient)
        if not torch.isfinite(norm) or norm.item() == 0:
            return None
        return gradient.mul_(self.epsilon / norm)

    @contextmanager
    def attack(self, direction):
        backup = self.weight.detach().clone()
        try:
            with torch.no_grad():
                self.weight.add_(direction.to(self.weight.dtype))
            yield
        finally:
            with torch.no_grad():
                self.weight.copy_(backup)


def backward_with_fgm(model, batch, loss, scaler, divisor, epsilon, device_type='cuda', dtype=torch.float16):
    if epsilon == 0:
        scaler.scale(loss / divisor).backward()
        return float(loss.detach()), False
    fgm = FGM(model, epsilon)
    direction = fgm.direction(loss, scaler)
    if direction is None:
        scaler.scale(loss / divisor).backward()
        return float(loss.detach()), True
    scaler.scale(loss / (2 * divisor)).backward()
    clean_value = float(loss.detach())
    with fgm.attack(direction):
        with torch.autocast(device_type, dtype=dtype, enabled=device_type == 'cuda'):
            adversarial_loss = model(**batch).loss
        if not torch.isfinite(adversarial_loss):
            raise RuntimeError('Nonfinite adversarial loss; embedding restored by finally')
        scaler.scale(adversarial_loss / (2 * divisor)).backward()
        adversarial_value = float(adversarial_loss.detach())
    return (clean_value + adversarial_value) / 2, False
