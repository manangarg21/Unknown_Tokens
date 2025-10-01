#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH -w gnode042
#SBATCH -o corpus_download.out  # Save output log to scratch
#SBATCH -e corpus_download.err  # Save error log to scratch
#SBATCH --time=2-00:00:00

# --- Define Paths ---
SCRATCH_DIR="/ssd_scratch/aanvik"
# Path to the new Python script in your home directory
PYTHON_DOWNLOAD_SCRIPT_PATH="./download_corpora.py" 
# Path where the final, combined corpus will be saved
OUTPUT_CORPUS_PATH="$SCRATCH_DIR/indiccorp_hindi_corpus.txt"
# Path to your requirements file
REQUIREMENTS_PATH="./requirements.txt"


# --- URLs for IndicCorpV2 Hindi Data ---
URL1="https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-1.txt"
URL2="https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-2.txt"
URL3="https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-3.txt"


# --- Conda Environment Setup ---
echo "--- Initializing Conda and activating environment: anlp_unk_tok ---"
source /home2/aanvik_bhatnagar/miniconda3/etc/profile.d/conda.sh
conda activate anlp_unk_tok

# --- Install Necessary Packages ---
# Only need requests and tqdm for this script, but installing all is fine.
echo "--- Installing necessary packages from $REQUIREMENTS_PATH ---"
pip install -r "$REQUIREMENTS_PATH"

# --- Run the Download Script ---
echo "--- Starting the direct download script for IndicCorpV2 ---"
python3 "$PYTHON_DOWNLOAD_SCRIPT_PATH" \
    --output_file "$OUTPUT_CORPUS_PATH" \
    --urls "$URL1" "$URL2" "$URL3"

echo "--- Script finished. The combined corpus is at $OUTPUT_CORPUS_PATH ---"
