# ZO-LLM with Federated Learning

This repository contains the implementation of ZO-Bench (Zero-Order optimization for Large Language Models) in a Federated Learning (FL) setting. The codebase supports the following experiments:

1. Finetuning:
   - Federated finetuning of large language models using zero-order optimization techniques.
   - Comparison with traditional gradient-based finetuning methods.

2. Compression:
   - Model compression experiments in a federated setting using ZO optimization.
   - Quantization and pruning techniques for reducing model size while maintaining performance.

3. Supported Models:
   - OPT-6.7B
   - LLAMA-3B

4. Evaluation:
   - Performance metrics for both centralized and federated learning scenarios.
   - Comparison of ZO-based methods with traditional optimization techniques.

The experiments are designed to showcase the effectiveness of ZO-LLM in a federated learning environment, addressing privacy concerns and communication efficiency in distributed settings.

# Environment
```sh
conda env create -f environment.yml
conda activate fedllm
```

# Running Experiments with Weights & Biases (wandb) Sweeps

This section describes how to run experiments using wandb sweeps, which allow for efficient hyperparameter tuning and experiment tracking.

## Setup

1. Ensure you have wandb installed and configured in your environment.
2. Prepare your sweep configuration files (*.yml) in the appropriate directory.

## Running Sweeps

Use the following bash script to run all wandb sweeps in a specified directory:

```sh
path="/home/jshaoaj/project/ZO-LLM/zo-bench/sweeps/ours_opt-6.7b-zo-fl"
# rm mypiip if it exists
[ -p mypipe ] && rm mypipe
# Create a named pipe
mkfifo mypipe

for file in $(find "$path" -name "*.yml"); do
    # Run the wandb sweep command and redirect its output to the named pipe
    wandb sweep "$file" &> mypipe &

    # Read the output from the named pipe
    output=$(cat mypipe)

    # Extract the command from the output
    command=$(echo "$output" | grep "Run sweep agent with:" | awk -F': ' '{print $3}')

    # Run command
    eval $command

done

# Remove the named pipe
rm mypipe
```