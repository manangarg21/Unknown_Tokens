#!/usr/bin/env bash
set -euo pipefail

# Downloads public datasets referenced in README and organizes them under data/

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p data/english data/hinglish data/hindi

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Resolve Kaggle command: prefer `kaggle`, fallback to `python3 -m kaggle`
resolve_kaggle_cmd() {
  if have_cmd kaggle; then
    echo "kaggle"
  elif have_cmd python3 && python3 - <<'PY' 2>/dev/null >/dev/null
import importlib
import sys
sys.exit(0 if importlib.util.find_spec('kaggle') else 1)
PY
  then
    echo "python3 -m kaggle"
  else
    echo ""  # not available
  fi
}

KAGGLE_CMD=$(resolve_kaggle_cmd)

echo "==> Checking tools"
if [ -z "$KAGGLE_CMD" ]; then
  echo "Kaggle CLI not found. Install via: pip install kaggle (or activate your venv)." >&2
else
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "kaggle.json not found at ~/.kaggle/kaggle.json. Configure Kaggle API before continuing." >&2
  fi
fi
have_cmd unzip || { echo "unzip not found. Please install unzip." >&2; }
have_cmd git || { echo "git not found. Please install git." >&2; }

echo "==> English datasets"
if [ -n "$KAGGLE_CMD" ]; then
  # News headlines sarcasm (Rishabh Misra)
  echo "Downloading: rmisra/news-headlines-dataset-for-sarcasm-detection"
  $KAGGLE_CMD datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection -p data/english || true
  if ls data/english/*.zip >/dev/null 2>&1; then
    for z in data/english/*.zip; do unzip -o "$z" -d data/english && rm -f "$z"; done
  fi
  # HackArena (placed under Hinglish below)
else
  echo "Skipping Kaggle downloads (CLI not available)." >&2
fi

echo "==> Hinglish datasets"
if [ -n "$KAGGLE_CMD" ]; then
  echo "Downloading: nikhilmaram/hackarena-multilingual-sarcasm-detection"
  $KAGGLE_CMD datasets download -d nikhilmaram/hackarena-multilingual-sarcasm-detection -p data/hinglish || true
  if ls data/hinglish/*.zip >/dev/null 2>&1; then
    for z in data/hinglish/*.zip; do unzip -o "$z" -d data/hinglish && rm -f "$z"; done
  fi
fi

echo "Cloning Hinglish GitHub datasets"
download_github_zip() {
  local repo_url=$1
  local target_dir=$2
  local default_branch=main
  local tmp_zip=$(mktemp -t repo.XXXXXX.zip)
  # Try main then master
  if curl -fL "$repo_url/archive/refs/heads/$default_branch.zip" -o "$tmp_zip"; then
    :
  elif curl -fL "$repo_url/archive/refs/heads/master.zip" -o "$tmp_zip"; then
    default_branch=master
  else
    echo "Failed to download zip for $repo_url" >&2
    rm -f "$tmp_zip"
    return 1
  fi
  mkdir -p "$target_dir"
  unzip -o "$tmp_zip" -d "$(dirname "$target_dir")"
  rm -f "$tmp_zip"
  # Move extracted folder to target_dir
  local extracted_dir
  extracted_dir=$(find "$(dirname "$target_dir")" -maxdepth 1 -type d -name "$(basename "$repo_url")-*" | head -n 1 || true)
  if [ -n "$extracted_dir" ]; then
    rm -rf "$target_dir"
    mv "$extracted_dir" "$target_dir"
  fi
}

if [ "${HINGLISH_GITHUB:-0}" = "1" ]; then
  repo1_url="https://github.com/nikhilmaram/Sarcasm-Detection-Code-Mixed-Dataset"
  repo1_dir="data/hinglish/Sarcasm-Detection-Code-Mixed-Dataset"
  repo2_url="https://github.com/nikhilmaram/Sarcasm-Detection-in-Hindi-English-Code-Mixed-Data"
  repo2_dir="data/hinglish/Sarcasm-Detection-in-Hindi-English-Code-Mixed-Data"

  if have_cmd git; then
    [ -d "$repo1_dir" ] || git clone "$repo1_url.git" "$repo1_dir" || download_github_zip "$repo1_url" "$repo1_dir" || true
    [ -d "$repo2_dir" ] || git clone "$repo2_url.git" "$repo2_dir" || download_github_zip "$repo2_url" "$repo2_dir" || true
  else
    download_github_zip "$repo1_url" "$repo1_dir" || true
    download_github_zip "$repo2_url" "$repo2_dir" || true
  fi
else
  echo "Skipping Hinglish GitHub clones (set HINGLISH_GITHUB=1 to enable)."
fi

echo "==> Manual downloads required"
cat <<EOF
The following require manual download/acceptance:
- English iSarcasm dataset (see: paperswithcode.com/dataset/isarcasm and authors' link)
- SemEval-2018 Task 3 Irony in English tweets (task page/CodaLab): aclanthology.org/S18-1005/
- Hindi corpora for mining:
  * IndicCorpv2 (Hindi subset): ai4bharat.iitm.ac.in/indiccorp
  * IITB English–Hindi Parallel Text: cfilt.iitb.ac.in/iitb_parallel/
  * OSCAR Hindi subset: oscar-corpus.com

After downloading, place raw text under:
  corpora/hindi/*.txt   # one or more files
EOF

echo "==> Done"


