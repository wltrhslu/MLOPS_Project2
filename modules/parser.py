import os

from argparse import ArgumentParser

def parse_args(args=None):
    defaults = {}
    changed_args = {}

    parser = ArgumentParser()

    parser.add_argument("--wandb_project", type=str, help="Wandb project to be used.", default="Project2")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint directory to be used.", default="models")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path to be used.", default="distilbert-base-uncased")
    parser.add_argument("--train_batch_size", type=int, help="Training batch size to be used.")
    parser.add_argument("--val_batch_size", type=int, help="Validation batch size to be used.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate to be used.")
    parser.add_argument("--learning_rate_modifier", type=float, help="Learning rate modifier to be used.")
    parser.add_argument("--warmup_steps", type=int, help="Warmup steps to be used.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay to be used.")
    parser.add_argument("--beta1", type=float, help="Beta1 to be used.")
    parser.add_argument("--beta2", type=float, help="Beta2 to be used.")

    args = parser.parse_args(args)
    
    if not os.path.isdir(os.path.join(os.getcwd(), args.checkpoint_dir)):
        print(f"Creating checkpoint directory at {os.path.join(os.getcwd(), args.checkpoint_dir)}")
        os.mkdir(args.checkpoint_dir)

    if args.train_batch_size is None:
        args.train_batch_size = 32
        defaults["train_batch_size"] = args.train_batch_size
    else:
        changed_args["train_batch_size"] = args.train_batch_size
    
    if args.val_batch_size is None:
        args.val_batch_size = 32
        defaults["val_batch_size"] = args.val_batch_size
    else:
        changed_args["val_batch_size"] = args.val_batch_size
    
    if args.learning_rate is None:
        args.learning_rate = 2e-5
        defaults["learning_rate"] = args.learning_rate
    else:
        changed_args["learning_rate"] = args.learning_rate

    if args.learning_rate_modifier is None:
        args.learning_rate_modifier = 1.0
        defaults["learning_rate_modifier"] = args.learning_rate_modifier
    else:
        changed_args["learning_rate_modifier"] = args.learning_rate_modifier
    
    if args.weight_decay is None:
        args.weight_decay = 0.0
        defaults["weight_decay"] = args.weight_decay
    else:
        changed_args["weight_decay"] = args.weight_decay
    
    if args.beta1 is None:
        args.beta1 = 0.9
        defaults["beta1"] = args.beta1
    else:
        changed_args["beta1"] = args.beta1
    
    if args.beta2 is None:
        args.beta2 = 0.999
        defaults["beta2"] = args.beta2
    else:
        changed_args["beta2"] = args.beta2

    if len(defaults) > 0:
        print("Using the following defaults values:")
        for k, v in defaults.items():
            print(f"\t-{k}: {v}")

    if len(changed_args) > 0:
        print("Using the following changed values:")
        for k, v in changed_args.items():
            print(f"\t-{k}: {v}")

    return args