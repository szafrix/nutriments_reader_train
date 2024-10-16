MODEL_CONFIG = {
    "checkpoint": "microsoft/trocr-large-printed",
}

DATALOADER_CONFIG = {
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "dataloader_drop_last": True,
    "max_length": 64,
}

TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "use_cpu": False,
}

EVAL_CONFIG = {
    "eval_steps": 50,
    "eval_strategy": "steps",
}

CALLBACK_CONFIG = {
    "output_dir": "./results",
    "save_strategy": "no",
}
