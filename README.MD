# MLOPS Project 2 Containerization

This repo contains the code to train [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on [MRPC](https://huggingface.co/datasets/glue/viewer/mrpc/train) for paraphrase detection.

## 1. Pre-requisites

There are a few pre-requisites to run this project:

1. [Docker](https://www.docker.com/) needs to be installed on your system.
2. [Wandb](https://wandb.ai/) is used to log the training metrics. You need to have a wandb account and the API key to log the metrics. You can get the API key from your wandb account settings.

### 1.1 Environment Variables

You need to set your wandb API key as an environment variable. There are two ways of doing this:

1. Rename the .env.example file to .env and set the value of the WANDB_API_KEY variable to your wandb API key.
2. Set the variable when running the docker container. Check step 4.3 for more details.

## 3. Parameters

There are several parameters that can be passed to the training script. To change the parameters, edit [train.sh](./train.sh) **before** building the docker image.

1. `--wand-project`: Project name where wandb is goint to save the logs to. Default is `Project2`.
2. `--checkpoint_dir`: Where the models should be saved to. Default is `./models`.
3. `--model_name_or_path`: Which model to train. Default is `distilbert-base-uncased`.
4. `--train_batch_size`: Training batch size to be used. Default is 32.
5. `--val_batch_size`: Validation batch size to be used. Default is 32.
6. `--learning_rate`: Learning rate to be used. Default is 2e-5.
7. `--learning_rate_modifier`: Learning rate modifier to be used. Default is 1.0.
8. `--warmup_steps`: Warmup steps to be used. Default is to calculate the warmup steps based on the number of training steps.
9. `--weight_decay`: Weight decay to be used. Default is 0.0.
10. `--beta1`: Beta1 to be used for AdamW. Default is 0.9.
11. `--beta2`: Beta2 to be used for AdamW. Default is 0.999.

The hyperparameters of the best run from project 1 are pre-configured in [train.sh](./train.sh).

:bangbang: **The training batch size of 128 is too large for Codespaces. If you are running this on Codespaces, change the training batch size to 64 in [train.sh](./train.sh).** :bangbang:

## 4. Steps to run the training

### 4.1 Clone the repository

```bash
git clone https://github.com/wltrhslu/MLOPS_Project2
cd MLOPS_Project2
```

### 4.2 Build the docker image

If you want to change the hyperparameters, remember to edit [train.sh](./train.sh) before building the docker image.

```bash
docker build -t mlops_project2 .
```

### 4.3 Run the docker container

Either run the container with the environment variable set in the .env file:

```bash
docker run -it --env-file .env mlops_project2
```

Or set the environment variable when running the container:

```bash
docker run -it -e WANDB_API_KEY=YOUR_API_KEY_GOES_HERE mlops_project2
```
