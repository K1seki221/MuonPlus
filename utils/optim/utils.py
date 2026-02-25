import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from typing import Tuple
from transformers import get_cosine_schedule_with_warmup
from .momo import Momo
from .momo_adam import MomoAdam


def get_optimizer_factory(name: str):
    """
    Returns the optimizer class corresponding to the given name.
    """
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD
    elif name == 'adam':
        return torch.optim.Adam
    elif name == 'adamw':
        return torch.optim.AdamW
    elif name == 'momo':
        return Momo
    elif name == 'momo-adam':
        return MomoAdam
    elif name == 'muon-plus':
        from .muon_plus import MuonPlus
        return MuonPlus
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def get_optimizer(opt_config: dict, lr = 1e-3) -> Tuple[torch.optim.Optimizer, dict]:
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).  
    For all hyperparameters which are not specified, we use PyTorch default.
    """
    
    name = opt_config['name']
    
    if opt_config.get('lr') is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")
    
    if name == 'sgd':
        opt_obj = torch.optim.SGD
        
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0)
                  }
        
    elif name == 'sgd-m':
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get('dampening') == 'momentum':
            dampening = opt_config.get('momentum', 0.9)
        else:
            dampening = opt_config.get('dampening', 0)
            
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': False,
                  'dampening': dampening
                  }

    elif name == 'sgd-nesterov':
        opt_obj = torch.optim.SGD
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'momentum': opt_config.get('momentum', 0.9),
                  'nesterov': True,
                  'dampening': opt_config.get('dampening', 0)
                  }
               
    elif name == 'adam':
        opt_obj = torch.optim.Adam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'fused': True
                  }
    
    elif name == 'adamw':
        opt_obj = torch.optim.AdamW
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'fused': True
                  }
    
    elif name == 'momo':
        opt_obj = Momo
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': False
                  }
    
    elif name == 'momo-adam':
        opt_obj = MomoAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': False
                  }
        
    elif name == 'momo-star':
        opt_obj = Momo
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'beta': opt_config.get('beta', 0.9),
                  'lb': opt_config.get('lb', 0.),
                  'bias_correction': opt_config.get('bias_correction', False),
                  'use_fstar': True
                  }
        
    elif name == 'momo-adam-star':
        opt_obj = MomoAdam
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'betas': opt_config.get('betas', (0.9, 0.999)),
                  'eps': opt_config.get('eps', 1e-8),
                  'lb': opt_config.get('lb', 0.),
                  'divide': opt_config.get('divide', True),
                  'use_fstar': True
                  }

    elif name == 'muon-plus':
        from .muon_plus import MuonPlus
        opt_obj = MuonPlus
        hyperp = {'lr': lr,
                  'weight_decay': opt_config.get('weight_decay', 0),
                  'adamw_betas': opt_config.get('betas', (0.95, 0.95)),
                  'momentum': opt_config.get('momentum', 0.95),
                  'nesterov': opt_config.get('nesterov', True),
                  'split_heads': opt_config.get('split_heads', False),
                  'nheads': opt_config.get('nheads', 12),
                  'ns_steps': opt_config.get('ns_steps', 5),
                  'rms_scaling': opt_config.get('rms_scaling', True),
                  'nuclear_scaling': opt_config.get('nuclear_scaling', False),
                  'polar_method': opt_config.get('polar_method', 'Keller'),
                  'polar_args': opt_config.get('polar_args', {}),
                  # post-polar normalization
                  'norm_mode': opt_config.get('norm_mode', 'none'),
                  'norm_eps': opt_config.get('norm_eps', 1e-8),
                  }

    else:
        raise KeyError(f"Unknown optimizer name {name}.")
        
    return opt_obj, hyperp
def get_scheduler(config, opt: torch.optim.Optimizer, total_iterations = None) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.

    config 可以是：
    - dict：来自 optimizer 的配置（run.py 的 opt_config）
        期望字段：
            - 'lr_schedule'（优先）
            - 或 'name'（保持向后兼容）
            - 可选 'warm_up_fraction'
    - str：直接给 scheduler 名字（run_hydra.py）
    """

    # 统一成 dict + name
    if isinstance(config, str):
        # 例如 "constant-linear"
        name = config
        cfg = {"name": name}
    else:
        # 例如 opt_config = {'name': 'adamw', 'lr_schedule': 'constant-linear', ...}
        cfg = config
        # 优先使用 lr_schedule，其次兼容老的 name 字段
        name = cfg.get("lr_schedule", cfg.get("name", "constant"))

    # 如果没给 warm_up_fraction，就默认 0.0
    warm_up_fraction = cfg.get("warm_up_fraction", 0.0)

    # --------- 以下是原来的逻辑，改成用 name / warm_up_fraction ---------
    if name == 'constant':
        lr_fun = lambda epoch: 1  # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)

    elif name == 'linear':
        lr_fun = lambda epoch: 1 / (epoch + 1)  # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)

    elif name == 'sqrt':
        lr_fun = lambda epoch: (epoch + 1) ** (-1 / 2)  # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)

    elif 'exponential' in name:
        # use sth like 'exponential_60_0.5': decay by factor 0.5 every 60 epochs
        step_size = int(name.split('_')[1])
        gamma = float(name.split('_')[2])
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

    elif 'warm-up-cosine' in name:
        assert total_iterations is not None, "total_iterations must be provided for warm-up schedulers"
        num_warmup_steps = int(warm_up_fraction * total_iterations)
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_iterations,
        )

    elif 'constant-linear' in name:  # New scheduler
        assert total_iterations is not None, "total_iterations must be provided for constant-linear scheduler"
        num_warmup_steps = int(warm_up_fraction * total_iterations)

        def get_lr(step: int):
            if step < num_warmup_steps:
                return 1.0  # Constant learning rate during warm-up
            else:
                # Linearly decay after warm-up
                return max(0.1, 1.0 - (step - num_warmup_steps) / (total_iterations - num_warmup_steps))

        scheduler = LambdaLR(opt, lr_lambda=get_lr)

    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")

    return scheduler

