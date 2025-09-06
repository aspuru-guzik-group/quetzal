#!/bin/bash

# Read job commands line-by-line, skipping comments and empty lines
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    echo "Submitting: $line"
    echo

    sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --job-name=train
#SBATCH --output=slurm/%j.out

source ~/.bashrc
conda activate quetzal

wandb offline
$line
EOF

    sleep 2

done < jobs
