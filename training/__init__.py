"""
Training module for traffic signal RL agents
"""

# Lazy imports to avoid loading heavy dependencies at package import time
def __getattr__(name):
    if name == 'train_stage':
        from training.train_single import train_stage
        return train_stage
    elif name == 'evaluate_final_model':
        from training.train_single import evaluate_final_model
        return evaluate_final_model
    elif name == 'train_single_main':
        from training.train_single import main as train_single_main
        return train_single_main
    elif name == 'train_multi_main':
        from training.train_multi import main as train_multi_main
        return train_multi_main
    raise AttributeError(f"module 'training' has no attribute {name!r}")

__all__ = ['train_stage', 'evaluate_final_model', 'train_single_main', 'train_multi_main']
