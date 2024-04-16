#!/bin/bash
set -e
set -x


path="/home/SHAO-Jiaqi757/project/FLZOLLM/ZO-LLM/zo-bench/sweeps/classification_llama-7b-ft"
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