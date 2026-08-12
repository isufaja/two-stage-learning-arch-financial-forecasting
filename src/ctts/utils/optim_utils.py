# optim_utils.py
import io
import torch
import contextlib
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ OPTIMIZERS                                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ Lookahead Helper ━━━━━━━━━━┓
class Lookahead(Optimizer):
    """
    A simple Lookahead wrapper for any torch optimizer.
    Paper: https://arxiv.org/abs/1907.08610
    """
    def __init__(self, base_optimizer, k=6, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.base_optimizer = base_optimizer
        defaults = base_optimizer.defaults
        super().__init__(base_optimizer.param_groups, defaults)

        self.k = k
        self.alpha = alpha
        self.step_counter = 0

        # store slow weights
        self.slow_weights = [
            [p.data.clone().detach() for p in group['params']]
            for group in self.base_optimizer.param_groups
        ]

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            # every k steps, update slow weights & sync back
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    slow = self.slow_weights[group_idx][p_idx]
                    # slow += alpha * (fast - slow)
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
            self.step_counter = 0
        return loss

    def zero_grad(self, *, set_to_none: bool = False):
        """
        Match the torch.optim.Optimizer signature so 
        epoch_loop can do optimizer.zero_grad(set_to_none=True).
        """
        # delegate to the wrapped optimizer
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

# ┏━━━━━━━━━━ Optimizers Choice ━━━━━━━━━━┓
def get_optimizer(model_parameters, cfg):
    """
    Factory: returns one of
      - Adagrad
      - RMSprop
      - Adam
      - AdamW
      - AdaBelief
      - Ranger (AdamW + Lookahead)
    based on cfg["optimizer"].
    cfg must contain:
      optimizer:      str
      lr:             float
      weight_decay:   float
    Optionally for:
      - betas (Adam/AdamW/AdaBelief): list of two floats
      - alpha (RMSprop): float
      - eps, weight_decouple, rectify (AdaBelief)
      - lookahead_alpha, lookahead_k (Ranger)
    """
    name = cfg["optimizer"].lower()
    learning_rate   = float(cfg["lr"])
    weight_decay   = float(cfg.get("weight_decay", 0.0))

    if name == "adagrad":
        return optim.Adagrad(model_parameters, 
                             lr = learning_rate, 
                             weight_decay = weight_decay)

    if name == "rmsprop":
        alpha = float(cfg.get("alpha", 0.99))
        return optim.RMSprop(model_parameters, 
                             lr = learning_rate, 
                             alpha = alpha, 
                             weight_decay = weight_decay)

    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return optim.Adam(model_parameters, 
                          lr = learning_rate, 
                          betas = betas, 
                          weight_decay = weight_decay)

    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return optim.AdamW(model_parameters, 
                           lr = learning_rate, 
                           betas = betas, 
                           weight_decay = weight_decay)

    if name == "adabelief":
        from adabelief_pytorch import AdaBelief
        buf = io.StringIO() # To redirect the messages from this Optimizer
        with contextlib.redirect_stdout(buf):
            opt = AdaBelief(model_parameters,
                            lr               = learning_rate,
                            betas            = tuple(cfg.get("betas", (0.9, 0.999))),
                            eps              = float(cfg.get("eps", 1e-16)),
                            weight_decouple  = cfg.get("weight_decouple", True),
                            rectify          = cfg.get("rectify", False),
                            weight_decay     = weight_decay,
                            print_change_log = False)
        return opt

    if name == "ranger":
        # base = AdamW
        base_opt = optim.AdamW(model_parameters,
                               lr           = learning_rate,
                               betas        = tuple(cfg.get("betas", (0.9, 0.999))),
                               weight_decay = weight_decay)
        return Lookahead(
            base_opt,
            k     = int(cfg.get("lookahead_k", 6)),
            alpha = float(cfg.get("lookahead_alpha", 0.5))
        )

    raise ValueError(f"Unknown optimizer '{name}'")


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ LEARNING RATE SCHEDULER                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━ Learning Rate Scheduler ━━━━━━━━━━┓
def make_scheduler(optim, sche_cfg, max_epochs):
    name      = sche_cfg.get("sch_name", sche_cfg.get("sch"))
    warm_ep   = sche_cfg.get("warmup_epochs", sche_cfg.get("warmup_ep"))
    pat_plate = sche_cfg.get("plateau_patience", sche_cfg.get("plat_pat"))
    factor    = sche_cfg.get("plateau_factor", sche_cfg.get("factor"))

    # Power-specific params
    power_s = sche_cfg.get("power_s", 1)   # denominator scale
    power_c = sche_cfg.get("power_c", 1)   # exponent

    # Exponential-specific params
    exp_s      = sche_cfg.get("exp_s", 1)         # epochs per factor drop
    exp_factor = sche_cfg.get("exp_factor", 0.1)   # base drop per exp_s

    if name == "none":
        # No scheduling: constant learning rate
        return None

    # 1) Linear Warm-up:
    #    η(t) = η_max * (start_factor + (end_factor - start_factor) * t / warmup_ep)
    sched_warm = LinearLR(optim, start_factor=1e-4, end_factor=1.0, total_iters=warm_ep)

    # 2) Cosine Annealing:
    #    η(t) = η_min + 0.5*(η_max - η_min)*(1 + cos(pi * t / (T_max)))
    #    η_max is the original learning rate; n_min the final (~ 0 by default)
    sched_cos  = CosineAnnealingLR(optim, T_max=max_epochs - warm_ep)

    # 3) Reduce on Plateau:
    #    LR <- LR * factor when validation loss hasn't improved for pat_plate epochs
    sched_plat = ReduceLROnPlateau(optim, mode="min", patience=pat_plate, factor=factor, min_lr=1e-7)

    if name == "linear":
        # Warm-up only
        return sched_warm

    if name == "cosine":
        # Warm-up → Cosine decay
        return SequentialLR(optim, schedulers=[sched_warm, sched_cos], milestones=[warm_ep])

    if name == "plateau":
        # Performance-based drop only
        return sched_plat

    if name == "power":
        # η(t) = η0 / (1 + t/s)^c
        return LambdaLR(
            optim,
            lr_lambda=lambda t: 1.0 / ((1.0 + t / power_s) ** power_c)
        )

    if name == "exponential":
        # η(t) = η0 * (exp_factor)^(t/exp_s)
        # implement via ExponentialLR with gamma = exp_factor^(1/exp_s)
        gamma = exp_factor ** (1.0 / exp_s)
        return ExponentialLR(optim, gamma=gamma)

    if name == "linear_plateau":
        # Warm-up then performance-based drop
        return {
            "warm":      sched_warm,
            "plateau":   sched_plat,
            "warmup_epochs": warm_ep,
        }

    if name == "cosine_plateau":
        # Warm-up → Cosine decay → then performance-based drop after cos_end
        cos_end = max_epochs - pat_plate
        sched_cos = CosineAnnealingLR(optim, T_max=max(1, cos_end - warm_ep))
        return {
            "warm":      sched_warm,
            "cos":       sched_cos,
            "plateau":   sched_plat,
            "warmup_epochs": warm_ep,
            "cos_end":   cos_end,
        }

    raise ValueError(f"Unknown scheduler '{name}'")

# ┏━━━━━━━━━━ Updater of Learning Rate ━━━━━━━━━━┓
def step_scheduler(sched, epoch: int, val_loss: float):
    """
    Advance the learning-rate scheduler by one epoch.

    Supports:
      - None: no scheduling.
      - dict: composite {warm, cos, plateau}.
      - ReduceLROnPlateau: performance-based drops.
      - Other schedulers (LinearLR, CosineAnnealingLR).
    
    Args:
      sched     : Scheduler object or dict from make_scheduler.
      epoch     : Current epoch index (0-based).
      val_loss  : Latest validation loss (for plateau steps).
    """
    # 1) No scheduler chosen → nothing to do
    if sched is None:
        return

    # 2) Composite scheduler: warm-up, optional cosine, then plateau
    if isinstance(sched, dict):
        # Warm-up phase: linearly ramp LR
        if epoch < sched["warmup_epochs"]:
            sched["warm"].step()
        # Cosine decay phase: half-cosine drop
        elif "cos" in sched and epoch < sched.get("cos_end", 0):
            sched["cos"].step()
        # Performance drop: reduce on plateau of val_loss
        sched["plateau"].step(val_loss)
        return

    # 3) Single-objective ReduceLROnPlateau → step with metric
    if isinstance(sched, ReduceLROnPlateau):
        sched.step(val_loss)
    else:
        # 4) Other schedulers (e.g. LinearLR, CosineAnnealingLR) → step without metric
        sched.step()
