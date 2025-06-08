#!/bin/bash
#SBATCH --job-name=Pr         # Job name
#SBATCH --output=slurm-%j.out               # Standard output log
#SBATCH --error=slurm-%j.err                # Error log
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=64                         # Total 64 tasks (for parallel execution)
#SBATCH --cpus-per-task=1                   # CPUs per task
#SBATCH --partition=long                    # Partition name (long-running jobs)
#SBATCH --qos=long                          # Quality of service (QOS) for long jobs
#SBATCH --mail-type=ALL                     # Send an email for all job events (begin, end, fail)
#SBATCH --mail-user=your@email.com          # Set your email address for notifications

# --- Environment setup ---
echo "Starting job on $HOSTNAME"
export JOB=PYTHON_TRAIN
export DIR=$SLURM_SUBMIT_DIR                 # Use the directory from which the SLURM script was submitted
cd $DIR                                       # Change to the directory where the script is located

# Load necessary modules and activate the conda environment
module purge                                  # Clear any loaded modules
module load compilers/python3.11             # Load the Python module (adjust if necessary)
source /apps/compilers/anaconda3-11/etc/profile.d/conda.sh  # Source the conda setup script
conda activate /apps/compilers/anaconda3-2022/envs/cnet  # Activate the 'cnet' conda environment
pip install chgnet

# Check if the environment is activated
echo "Activated environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Ensure necessary Python packages are installed (if not already installed)
echo "Installing required Python packages..."
pip install --upgrade pymatgen phonopy  # Make sure packages are up to date

# --- Run the Python script ---
echo "Running train.py with 64 cores..."

# Run the train.py script and capture output and errors
python train.py > train.log 2>&1
if [ $? -ne 0 ]; then  # Check if the previous command was successful
    echo "train.py failed. Check train.log for details."
    exit 1
fi

# Run the phonopy-related script
echo "Running chgnet_phonopy_run.py script..."
python chgnet_phonopy_run.py > phonon.out 2>&1
if [ $? -ne 0 ]; then  # Check if the previous command was successful
    echo "chgnet_phonopy_run.py failed. Check phonon.out for details."
    exit 1
fi

# Job completed successfully
echo "Job completed successfully!"
