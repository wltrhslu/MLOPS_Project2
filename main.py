from dotenv import load_dotenv
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from modules.parser import parse_args
from modules.data_module import GLUEDataModule
from modules.transformer import GLUETransformer 
from modules.logger import get_logger
from os import getenv

if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = getenv("WANDB_API_KEY")

    # Before running the script, make sure that the WANDB_API_KEY is set in the environment variables.
    assert wandb_api_key is not None, "Wandb API key is required."
    assert wandb_api_key != "YOUR_API_KEY_GOES_HERE", "Change the WANDB_API_KEY value from YOUR_API_KEY_GOES_HERE to your actual API key."

    args = parse_args()
    args.wandb_api_key = wandb_api_key
    args.epochs = 3

    seed_everything(42)

    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name="mrpc",
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.val_batch_size,
    )
    dm.setup("fit")

    if args.warmup_steps is None:
        args.warmup_steps = int((len(dm.dataset["train"]) // args.train_batch_size) * args.epochs * 0.1)    

    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.val_batch_size,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    logger = get_logger(args)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[lr_monitor]
    )

    trainer.fit(model, datamodule=dm)