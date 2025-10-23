"""
This script downloads large text files from a list of specified URLs and
combines them into a single output file. It streams the downloads to handle
large files efficiently without consuming excessive memory.

Usage:
python download_from_urls.py --output_file /path/to/save/corpus.txt --urls URL1 URL2 URL3 ...
"""
import requests
import argparse
from tqdm import tqdm

def download_and_combine(urls: list, output_file: str):
    """
    Downloads content from a list of URLs and appends it to a single text file.
    
    Args:
        urls (list): A list of URL strings to download from.
        output_file (str): The path to the file where content will be saved.
    """
    print(f"--- Starting download process ---")
    print(f"Output will be saved to: {output_file}")

    # Open the output file in append mode with utf-8 encoding
    with open(output_file, 'wb') as f_out:
        for i, url in enumerate(urls):
            print(f"\nDownloading URL {i+1}/{len(urls)}: {url}")
            try:
                # Use stream=True to avoid loading the whole file into memory
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                    
                    # Get total file size for the progress bar
                    total_size_in_bytes = int(r.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kibibyte
                    
                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")
                    
                    for chunk in r.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f_out.write(chunk)
                    
                    progress_bar.close()
                    
                    # Add a newline between files to ensure separation
                    f_out.write(b'\n')

            except requests.exceptions.RequestException as e:
                print(f"ERROR: Failed to download {url}. Reason: {e}")
                continue # Skip to the next URL

    print("\n--- Download and combination complete! ---")

def main():
    parser = argparse.ArgumentParser(description="Download and combine text files from URLs.")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The path to the final combined output text file."
    )
    parser.add_argument(
        "--urls",
        nargs='+',  # This allows specifying one or more URLs
        required=True,
        help="A space-separated list of URLs to download."
    )
    args = parser.parse_args()
    
    download_and_combine(args.urls, args.output_file)

if __name__ == "__main__":
    main()
