"""
This script implements a data augmentation pipeline to create a potential
Hindi sarcasm dataset from an English source provided via a command-line argument.

The pipeline now supports two input formats:
1. iSarcasm CSV format ('tweet', 'sarcastic', optional 'rephrase' columns).
2. Huffington Post JSON Lines format ('headline', 'is_sarcastic' keys).

It follows these steps:
1.  Load an English sarcasm dataset, adapting to the file type and format.
2.  (Optional) Translate the sarcastic English text to Hindi.
3.  Load and pre-filter a large monolingual Hindi corpus from a local file.
4.  Generate sentence embeddings for both datasets.
5.  Use FAISS to perform a semantic similarity search.
6.  Create a new potential sarcasm dataset.
7.  Save the results with a dynamic name based on the source dataset.

To run (example for iSarcasm dataset):
python main.py --input_file path/to/isarcasm.csv --dataset_name isarcasm --corpus_file path/to/hindi_corpus.txt --output_dir path/to/save/results

To run (example for Huffington Post dataset, skipping translation):
python main.py --input_file path/to/huffpost_translated.csv --dataset_name huffpost --skip_translation --corpus_file path/to/hindi_corpus.txt --output_dir path/to/save/results
"""
import pandas as pd
import torch
import faiss
import numpy as np
import argparse
import traceback
import os
import json
from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# --- 1. Configuration ---
# Model and dataset settings
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
EMBEDDING_MODEL = "sentence-transformers/LaBSE"

# Corpus processing settings
MAX_CORPUS_SIZE = 1_000_000

# Search and filtering parameters
NEAREST_NEIGHBORS_K = 10
SAMPLE_FOR_VERIFICATION = 200


def load_sarcasm_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads a sarcasm dataset, adapting logic based on the file extension (.csv or .jsonl).
    """
    print(f"--- Step 1: Loading Sarcasm Dataset from {filepath} ---")
    
    try:
        if filepath.endswith('.csv'):
            # Logic for iSarcasm2022 CSV format
            df = pd.read_csv(filepath)
            if 'tweet' not in df.columns or 'sarcastic' not in df.columns:
                raise ValueError("CSV file must contain 'tweet' and 'sarcastic' columns.")
            
            sarcastic_df = df[df['sarcastic'] == 1].copy()
            # **CHANGE**: Always use the sarcastic 'tweet' as the text to process.
            # Keep both 'tweet' and 'rephrase' columns for the final output.
            sarcastic_df['text'] = sarcastic_df['tweet']

            # Ensure 'rephrase' column exists for schema consistency
            if 'rephrase' not in sarcastic_df.columns:
                sarcastic_df['rephrase'] = None
            
            print(f"✅ Loaded {len(df)} rows from CSV, found {len(sarcastic_df)} sarcastic entries.\n")
            # The 'text' column will be used for translation, and 'rephrase' will be carried through.
            return sarcastic_df[['text', 'rephrase']]

        elif filepath.endswith('.json') or filepath.endswith('.json'):
            # Logic for Huffington Post JSON Lines format
            records = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    records.append(json.loads(line))
            
            df = pd.DataFrame(records)
            if 'headline' not in df.columns or 'is_sarcastic' not in df.columns:
                raise ValueError("JSON file must contain 'headline' and 'is_sarcastic' keys.")

            sarcastic_df = df[df['is_sarcastic'] == 1].copy()
            sarcastic_df.rename(columns={'headline': 'text'}, inplace=True)
            # Add a null rephrase column for schema consistency with the CSV path
            sarcastic_df['rephrase'] = None

            print(f"✅ Loaded {len(df)} rows from JSON, found {len(sarcastic_df)} sarcastic entries.\n")
            return sarcastic_df[['text', 'rephrase']]
            
        else:
            raise ValueError(f"Unsupported file format for: {filepath}. Please use .csv or .jsonl.")

    except FileNotFoundError:
        print(f"❌ ERROR: The file was not found at {filepath}")
        raise
    except Exception as e:
        print(f"❌ ERROR: An error occurred while reading the input file: {e}")
        raise


def translate_english_to_hindi(df: pd.DataFrame, output_dir: str, dataset_name: str) -> pd.DataFrame:
    """Translates a DataFrame with English text to Hindi."""
    print("--- Step 2: Translating English Sarcasm Dataset to Hindi ---")
    
    translator = pipeline(
        'translation',
        model=TRANSLATION_MODEL,
        src_lang='eng_Latn',
        tgt_lang='hin_Deva',
        device=0 if torch.cuda.is_available() else -1
    )
    
    english_texts = df['text'].tolist()
    batches = np.array_split(english_texts, (len(english_texts) // 32) + 1)
    batches = [b for b in batches if b.size > 0]
    translated_items = [translator(batch.tolist(), batch_size=32) for batch in tqdm(batches, desc="Translating")]
    translated_items = [item for sublist in translated_items for item in sublist]

    sar_h_prime_df = pd.DataFrame({
        'original_english': english_texts,
        'translated_hindi': [item['translation_text'] for item in translated_items]
    })
    
    # Carry over the rephrase column. The row order is preserved.
    sar_h_prime_df['original_rephrase'] = df['rephrase'].values

    os.makedirs(output_dir, exist_ok=True)
    translated_file = os.path.join(output_dir, f"output_{dataset_name}_translated.csv")
    sar_h_prime_df.to_csv(translated_file, index=False)
    print(f"✅ Translation complete. Saved to {translated_file}\n")
    return sar_h_prime_df


def get_monolingual_hindi_corpus(corpus_filepath: str, offset: int) -> Dataset:
    """Loads and pre-filters a subset of the Hindi corpus, starting from a given offset."""
    print(f"--- Step 3: Loading and Filtering Monolingual Hindi Corpus from {corpus_filepath} ---")
    try:
        hindi_texts = []
        lines_read = 0
        
        def is_valid_sentence(line: str) -> bool:
            return True
            # return len(line.split()) >= 5 and line.strip().endswith('।')

        with open(corpus_filepath, 'r', encoding='utf-8') as f:
            if offset > 0:
                print(f"⏩ Skipping first {offset} lines of the corpus file...")
                for _ in tqdm(range(offset), desc="Applying offset"):
                    next(f, None)
            
            pbar = tqdm(f, desc="Filtering corpus file", total=MAX_CORPUS_SIZE, unit=" lines")
            for line in pbar:
                lines_read += 1
                if len(hindi_texts) >= MAX_CORPUS_SIZE:
                    pbar.set_description(f"Reached {MAX_CORPUS_SIZE} valid sentences. Stopping.")
                    break
                
                clean_line = line.strip()
                if is_valid_sentence(clean_line):
                    hindi_texts.append(clean_line)
                    if len(hindi_texts) % 10000 == 0:
                         pbar.set_postfix_str(f"Kept {len(hindi_texts)}")

        if not hindi_texts:
            raise ValueError("Corpus file is empty or no valid sentences were found after filtering.")

        mono_h_dataset = Dataset.from_dict({'text': hindi_texts})
        print(f"✅ Read {lines_read} lines after offset, kept {len(mono_h_dataset)} valid sentences.\n")
        return mono_h_dataset
    except FileNotFoundError:
        print(f"❌ ERROR: The corpus file was not found at {corpus_filepath}")
        raise
    except Exception as e:
        print(f"❌ ERROR: An error occurred while reading the corpus file: {e}")
        raise


def find_semantically_similar(query_sentences: list, corpus_dataset: Dataset):
    """Encodes sentences and uses FAISS to find similar ones."""
    print("--- Step 4: Performing Semantic Similarity Search ---")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print("Generating embeddings for translated sarcastic sentences (queries)...")
    query_embeddings = model.encode(
        query_sentences, show_progress_bar=True, convert_to_numpy=True, device=device
    )

    print("Generating embeddings for monolingual corpus (search space)...")
    corpus_sentences = corpus_dataset['text']
    corpus_embeddings = model.encode(
        corpus_sentences, show_progress_bar=True, batch_size=128, convert_to_numpy=True, device=device
    )

    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    
    faiss.normalize_L2(corpus_embeddings)
    faiss.normalize_L2(query_embeddings)
    
    index.add(corpus_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    distances, indices = index.search(query_embeddings, NEAREST_NEIGHBORS_K)
    print("✅ Search complete.\n")
    return distances, indices


def create_and_save_datasets(distances, indices, sar_h_prime_df: pd.DataFrame, corpus_dataset, output_dir: str, similarity_threshold: float, dataset_name: str):
    """Filters search results and creates the final datasets with detailed diagnostic info."""
    print("--- Step 5: Creating Final Datasets ---")
    
    corpus_sentences = corpus_dataset['text']
    query_sentences = sar_h_prime_df['translated_hindi'].tolist()
    found_matches = []
    
    for i in tqdm(range(len(query_sentences)), desc="Filtering results"):
        query_sentence = query_sentences[i]
        for j in range(NEAREST_NEIGHBORS_K):
            neighbor_index = indices[i][j]
            similarity_score = distances[i][j]
            
            if (similarity_score >= similarity_threshold):
                matched_sentence = corpus_sentences[neighbor_index]
                if matched_sentence != query_sentence:
                    original_english = sar_h_prime_df.iloc[i]['original_english']
                    original_rephrase = sar_h_prime_df.iloc[i].get('original_rephrase', None)
                    found_matches.append({
                        'original_english': original_english,
                        'original_rephrase': original_rephrase,
                        'query_sentence_hindi': query_sentence,
                        'matched_sentence_hindi': matched_sentence,
                        'similarity_score': similarity_score
                    })
    
    if not found_matches:
        print(f"⚠️ No sentences found in the corpus above the current similarity threshold of {similarity_threshold}.\n")
        return
    
    matches_df = pd.DataFrame(found_matches)
    matches_df = matches_df.sort_values(by='similarity_score', ascending=False)
    matches_df = matches_df.drop_duplicates(subset=['matched_sentence_hindi'])

    print(f"Found {len(matches_df)} unique potential sarcastic sentences.")

    os.makedirs(output_dir, exist_ok=True)
    potential_sarcasm_file = os.path.join(output_dir, f"output_{dataset_name}_potential_sarcasm.csv")
    verification_sample_file = os.path.join(output_dir, f"output_{dataset_name}_verification_sample.csv")
    
    matches_df.to_csv(potential_sarcasm_file, index=False)
    
    sample_size = min(SAMPLE_FOR_VERIFICATION, len(matches_df))
    df_for_review = matches_df.sample(n=sample_size, random_state=42)
    df_for_review.to_csv(verification_sample_file, index=False)
    
    print(f"✅ Saved potential sarcasm dataset to {potential_sarcasm_file}")
    print(f"✅ Saved {len(df_for_review)} samples for review to {verification_sample_file}\n")
    

def main():
    """Main function to parse arguments and run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Hindi Sarcasm Data Augmentation Pipeline.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV or JSONL dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="A short name for the dataset (e.g., 'isarcasm', 'huffpost') to be used in output filenames."
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        required=True,
        help="Path to the pre-downloaded combined monolingual Hindi corpus text file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory in scratch space to save output files."
    )
    parser.add_argument(
        "--skip_translation",
        action="store_true",
        help="If set, skips the translation step. Assumes --input_file is already translated."
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.75,
        help="The minimum similarity score to consider a match."
    )
    parser.add_argument(
        "--corpus_offset",
        type=int,
        default=0,
        help="Line number to start reading from in the corpus file. Useful for chunking."
    )
    args = parser.parse_args()

    try:
        if args.skip_translation:
            print("--- Step 1 & 2: Skipping translation as per --skip_translation flag ---")
            print(f"Loading pre-translated data from: {args.input_file}")
            try:
                sar_h_prime_df = pd.read_csv(args.input_file)
                if 'translated_hindi' not in sar_h_prime_df.columns or 'original_english' not in sar_h_prime_df.columns or 'original_rephrase' not in sar_h_prime_df.columns:
                    raise ValueError("Input file for --skip_translation must contain 'translated_hindi', 'original_english', and 'original_rephrase' columns.")
                print(f"✅ Loaded {len(sar_h_prime_df)} pre-translated sentences.\n")
            except FileNotFoundError:
                print(f"❌ ERROR: The pre-translated file was not found at {args.input_file}")
                print("Please run the script without --skip_translation first to generate this file.")
                raise
            except Exception as e:
                print(f"❌ ERROR: An error occurred while reading the pre-translated CSV: {e}")
                raise
        else:
            # Step 1: Load English Dataset (handles both CSV and JSONL)
            sar_e_df = load_sarcasm_dataset(args.input_file)
            
            # Step 2: Translate (now saves to the directory specified in --output_dir)
            sar_h_prime_df = translate_english_to_hindi(sar_e_df, args.output_dir, args.dataset_name)
        
        # Step 3: Load local Hindi Corpus, applying the offset
        mono_h_dataset = get_monolingual_hindi_corpus(args.corpus_file, args.corpus_offset)
        
        # Step 4: Find Similar Sentences
        query_sentences = sar_h_prime_df['translated_hindi'].tolist()
        distances, indices = find_semantically_similar(query_sentences, mono_h_dataset)

        # Step 5: Create and Save Datasets, passing the full dataframe for context
        create_and_save_datasets(distances, indices, sar_h_prime_df, mono_h_dataset, args.output_dir, args.similarity_threshold, args.dataset_name)
        
        print("--- Pipeline Finished Successfully! ---")
    except Exception:
        print("\n--- Pipeline Failed: An unexpected error occurred ---")
        traceback.print_exc()

if __name__ == "__main__":
    main()