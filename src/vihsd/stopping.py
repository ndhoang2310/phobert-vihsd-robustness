"""Validation-only stopping state, serializable in resumable checkpoints."""
from dataclasses import dataclass, asdict
import math


@dataclass
class EarlyStopping:
    patience: int | None = None
    min_delta: float = 0.0
    best_score: float = -1.0
    bad_epochs: int = 0
    epochs_observed: int = 0

    def __post_init__(self):
        if self.patience is not None and (isinstance(self.patience,bool) or not isinstance(self.patience,int) or self.patience < 1):
            raise ValueError('early_stopping_patience must be a positive integer or None')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('early_stopping_min_delta must be finite and nonnegative')

    @property
    def should_stop(self):
        return self.patience is not None and self.bad_epochs >= self.patience

    def observe(self, score):
        if not math.isfinite(score):raise ValueError('Validation score must be finite')
        self.epochs_observed += 1
        if score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.should_stop

    def state_dict(self):return asdict(self)

    def restore(self, state):
        if state['patience'] != self.patience or state['min_delta'] != self.min_delta:
            raise ValueError('Resume early-stopping policy differs from checkpoint')
        self.best_score=float(state['best_score'])
        self.bad_epochs=int(state['bad_epochs'])
        self.epochs_observed=int(state['epochs_observed'])
        if not math.isfinite(self.best_score) or self.bad_epochs<0 or self.epochs_observed<self.bad_epochs:
            raise ValueError('Invalid early-stopping checkpoint state')
