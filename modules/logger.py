import wandb

from lightning.pytorch.loggers import WandbLogger

def get_logger(args):
    wandb.login(key=args.wandb_api_key)

    non_relevant_keys = ["wandb_api_key", "wandb_project", "checkpoint_dir"]
    config = {key: value for key, value in vars(args).items() if key not in non_relevant_keys}

    run_name = ""
    for key, value in config.items():
        run_name += f"{key}={value};"

    return WandbLogger(
        name=run_name,
        project=args.wandb_project,
        save_dir=args.checkpoint_dir,
        log_model="all",
        config=config,
    )