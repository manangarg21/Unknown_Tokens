# Hindi Sarcasm Data Augmentation

This repository contains a pipeline for generating a potential **Hindi sarcasm dataset**. The pipeline translates existing English sarcasm datasets into Hindi and then finds semantically similar sentences from a large, pre-downloaded Hindi monolingual corpus.

The workflow is designed to run on an HPC cluster with a SLURM scheduler and uses a temporary scratch space to handle large files and avoid home-directory disk quota issues.

---

## Project structure

Place files in your home directory in a structure similar to the following:

```
sarcasm_project/
│
├── data/
│   └── english/
│       ├── isarcasm2022.csv
│       └── Sarcasm_Headlines_Dataset_v2.json
│
├── hindi_corp_gen.py          # The main Python augmentation script
├── download_corpora.py       # Python script for direct URL downloads
├── corpus_download.sh        # SLURM script to download the Hindi corpus (run once)
├── translation_check.sh      # SLURM script to run the main pipeline
└── requirements.txt          # Python package requirements
```

---

## Setup and workflow

Follow these steps to configure and run the pipeline.

### Step 1 — One-time environment setup

You only need to do this once.

1. **Log in to the cluster.** Access your HPC cluster's head node.
2. **Create directories in scratch.** Start an interactive session and create the project folder and a Hugging Face cache folder in the scratch space:

```bash
sinteractive -w gnode042 --gres=gpu:1
mkdir -p /ssd_scratch/aanvik/huggingface_cache
exit
```

3. **Get a Hugging Face token.**

   * Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   * Create a token with at least **read** permissions
   * Copy the token — you will paste it into the SLURM scripts (`HF_TOKEN` variable)

---

### Step 2 — Download the Hindi monolingual corpus (run once)

This step downloads the large IndicCorpV2 dataset (or other Hindi corpora) and concatenates it into a single text file on scratch.

1. Edit **`corpus_download.sh`** and paste your Hugging Face token into the `HF_TOKEN` variable.
2. Submit the job:

```bash
sbatch corpus_download.sh
```

This will produce a file such as:

```
/ssd_scratch/aanvik/indiccorp_hindi_corpus.txt
```

---

### Step 3 — Run the main augmentation pipeline

This runs translation (optional) and the similarity-search augmentation.

1. Edit **`translation_check.sh`** and paste your Hugging Face token into the `HF_TOKEN` variable.
2. Review the configuration options described below.
3. Submit the job:

```bash
sbatch translation_check.sh
```

---

## Configuration and customization

You can tweak the behavior by editing `translation_check.sh` before submitting it.

### Choosing which datasets to process

`translation_check.sh` contains separate blocks for each dataset (for example: iSarcasm and Huffington Post). To run only one dataset, comment out the other block by prefixing its lines with `#`.

**Example — Run only the iSarcasm dataset:**

```bash
# --- Run pipeline for iSarcasm dataset ---
echo "--- Starting the data augmentation script for iSarcasm dataset ---"
python "$PYTHON_SCRIPT_PATH" \
    --input_file "$ISARCASM_INPUT_PATH" \
    --dataset_name "isarcasm" \
    # ... other arguments ...

# --- Run pipeline for Huffington Post dataset ---
# echo "--- Starting the data augmentation script for Huffington Post dataset ---"
# python "$PYTHON_SCRIPT_PATH" \
#     --input_file "$HUFFPOST_INPUT_PATH" \
#     --dataset_name "huffpost" \
#     # ... other arguments ...
```

### Skipping translation for faster re-runs

Translation is slow; once you have the translated files, you can skip translation for subsequent runs by using the `--skip_translation` flag and passing the previously produced translated CSVs.

**Example — Run without translation (both datasets):**

```bash
# --- To run WITHOUT translation in the future ---
ISARCASM_TRANSLATED_PATH="$AUGMENTATION_OUTPUT_DIR/output_isarcasm_translated.csv"
echo "--- Starting iSarcasm augmentation (SKIPPING TRANSLATION) ---"
python "$PYTHON_SCRIPT_PATH" \
    --input_file "$ISARCASM_TRANSLATED_PATH" \
    --dataset_name "isarcasm" \
    --skip_translation \
    # ... other arguments ...

HUFFPOST_TRANSLATED_PATH="$AUGMENTATION_OUTPUT_DIR/output_huffpost_translated.csv"
echo "--- Starting Huffington Post augmentation (SKIPPING TRANSLATION) ---"
python "$PYTHON_SCRIPT_PATH" \
    --input_file "$HUFFPOST_TRANSLATED_PATH" \
    --dataset_name "huffpost" \
    --skip_translation \
    # ... other arguments ...
```

### Adjusting similarity and corpus parameters

* **Similarity threshold**: `--similarity_threshold 0.75` — lower (e.g., `0.70`) to get more matches, raise for stricter matches.
* **Corpus offset**: `--corpus_offset 0` — set to the number of lines to skip in the big corpus file to process later chunks (e.g., `--corpus_offset 1000000` to skip the first million lines).
* **Corpus size per run**: To change how many sentences are processed per run, edit the `MAX_CORPUS_SIZE` variable in the top of `hindi_corp_gen.py`.

---

## Output files

All outputs are written to the directory set by `AUGMENTATION_OUTPUT_DIR` in the SLURM script (for example, `/ssd_scratch/aanvik/augmentation_output`). The most relevant files are:

* `output_{dataset_name}_translated.csv` — intermediate file containing original English text and its Hindi translation.
* `output_{dataset_name}_potential_sarcasm.csv` — main output: contains the original English text, rephrased English (if applicable), the translated Hindi query, the matched Hindi sentence from the corpus, and the similarity score.
* `output_{dataset_name}_verification_sample.csv` — a smaller random sample useful for manual quality checks and annotation.

---
