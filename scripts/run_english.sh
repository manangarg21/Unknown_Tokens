#!/bin/bash
#SBATCH -J "ANLP"
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem-per-cpu=3000
#SBATCH -o english.txt
#SBATCH --time="4-00:00:00"

python3 -m venv /tmp/ab/newvenv
source /tmp/ab/newvenv/bin/activate
pip install -r requirements.txt
echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"


echo start
python -m src.train_english --config configs/english_roberta_rcnn_semeval.yaml
echo end

echo "Time at exit: $(date)"
