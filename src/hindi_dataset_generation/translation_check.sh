#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH -w gnode042
#SBATCH -o hindi_augmentation.out  # Save output log to scratch
#SBATCH -e hindi_augmentation.err  # Save error log to scratch
#SBATCH --time=1-00:00:00

# --- User Configuration ---
# IMPORTANT: PASTE YOUR HUGGING FACE ACCESS TOKEN HERE (Needed for translation model)
HF_TOKEN="hf_xxxxxx" 

# --- Define Paths ---
SCRATCH_DIR="/ssd_scratch/aanvik"
HF_CACHE_PATH="$SCRATCH_DIR/huggingface_cache"

# --- Paths in your HOME directory ---
HOME_PROJECT_DIR="." 
PYTHON_SCRIPT_PATH="$HOME_PROJECT_DIR/hindi_corp_gen.py" 
INPUT_DATA_PATH="$HOME_PROJECT_DIR/data/english/isarcasm2022.csv"
REQUIREMENTS_PATH="$HOME_PROJECT_DIR/requirements_try.txt"

# --- Paths in your SCRATCH directory ---
COMBINED_CORPUS_PATH="$SCRATCH_DIR/indiccorp_hindi_corpus.txt"
# Directory to save the final augmented datasets
AUGMENTATION_OUTPUT_DIR="$HOME_PROJECT_DIR/data/hindi_dataset_generation/augmentation_output"


# --- Set Hugging Face Cache Directory ---
echo "--- Setting Hugging Face cache to: $HF_CACHE_PATH ---"
export HF_HOME="$HF_CACHE_PATH"

# --- Conda Environment Setup ---
echo "--- Initializing Conda and activating environment: anlp_unk_tok ---"
source /home2/aanvik_bhatnagar/miniconda3/etc/profile.d/conda.sh
conda activate anlp_unk_tok

# --- Hugging Face Authentication ---
echo "--- Logging in to Hugging Face Hub ---"
huggingface-cli login --token $HF_TOKEN

# --- Verification ---
echo "--- Verifying PyTorch Installation ---"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# # --- Run Main Application ---
# echo "--- Starting the data augmentation script ---"
# # This command runs the main.py script with all the required paths as arguments.
# # By default, it will perform the translation step.
# python "$PYTHON_SCRIPT_PATH" \
#     --input_file "$INPUT_DATA_PATH" \
#     --corpus_file "$COMBINED_CORPUS_PATH" \
#     --output_dir "$AUGMENTATION_OUTPUT_DIR"

# # --- Example of running WITHOUT translation ---
# # If you have already translated the file once and want to re-run the similarity search,
# # you would comment out the command above and uncomment the one below.

# # TRANSLATED_INPUT_PATH="$AUGMENTATION_OUTPUT_DIR/output_sar_h_prime.csv"
# echo "--- Starting the data augmentation script (SKIPPING TRANSLATION) ---"
# python3 "$PYTHON_SCRIPT_PATH" \
#     --input_file "$TRANSLATED_INPUT_PATH" \
#     --skip_translation \
#     --corpus_file "$COMBINED_CORPUS_PATH" \
#     --output_dir "$AUGMENTATION_OUTPUT_DIR"

ISARCASM_INPUT_PATH="$HOME_PROJECT_DIR/data/english/isarcasm2022.csv"
HUFFPOST_INPUT_PATH="$HOME_PROJECT_DIR/data/english/Sarcasm_Headlines_Dataset_v2.json"

# --- === Run Pipeline for iSarcasm Dataset === ---
echo "--- Starting the data augmentation script for iSarcasm dataset ---"
python3 "$PYTHON_SCRIPT_PATH" \
    --input_file "$ISARCASM_INPUT_PATH" \
    --dataset_name "isarcasm" \
    --corpus_file "$COMBINED_CORPUS_PATH" \
    --output_dir "$AUGMENTATION_OUTPUT_DIR" \
    --similarity_threshold 0.60

# --- === Run Pipeline for Huffington Post Dataset === ---
echo "--- Starting the data augmentation script for Huffington Post dataset ---"
python3 "$PYTHON_SCRIPT_PATH" \
    --input_file "$HUFFPOST_INPUT_PATH" \
    --dataset_name "huffpost" \
    --corpus_file "$COMBINED_CORPUS_PATH" \
    --output_dir "$AUGMENTATION_OUTPUT_DIR" \
    --similarity_threshold 0.60


# --- To run WITHOUT translation in the future ---
# After the first successful run, translated files will be created.
# You can then comment out the two blocks above and uncomment the two blocks below to save time.

# ISARCASM_TRANSLATED_PATH="$AUGMENTATION_OUTPUT_DIR/output_isarcasm_translated.csv"
# echo "--- Starting iSarcasm augmentation (SKIPPING TRANSLATION) ---"
# python3 "$PYTHON_SCRIPT_PATH" \
#     --input_file "$ISARCASM_TRANSLATED_PATH" \
#     --dataset_name "isarcasm" \
#     --skip_translation \
#     --corpus_file "$COMBINED_CORPUS_PATH" \
#     --output_dir "$AUGMENTATION_OUTPUT_DIR" \
#     --similarity_threshold 0.75

# HUFFPOST_TRANSLATED_PATH="$AUGMENTATION_OUTPUT_DIR/output_huffpost_translated.csv"
# echo "--- Starting Huffington Post augmentation (SKIPPING TRANSLATION) ---"
# python3 "$PYTHON_SCRIPT_PATH" \
#     --input_file "$HUFFPOST_TRANSLATED_PATH" \
#     --dataset_name "huffpost" \
#     --skip_translation \
#     --corpus_file "$COMBINED_CORPUS_PATH" \
#     --output_dir "$AUGMENTATION_OUTPUT_DIR" \
#     --similarity_threshold 0.75

echo "--- Script finished. ---"

