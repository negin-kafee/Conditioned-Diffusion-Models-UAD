#!/bin/bash
#SBATCH --job-name=convert_h5_fast
#SBATCH --output=slurm/logs/convert_h5_fast_%j.log
#SBATCH --error=slurm/logs/convert_h5_fast_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Activate environment
# Activate your conda/venv environment
# source /path/to/your/env/bin/activate

# Set environment variables
export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}

# Set working directory
cd "$(dirname "$0")/.."  # Change to repo root

# Run conversion
python scripts/convert_h5_fast_to_nifti.py

echo "Conversion complete!"
