#!/bin/bash
#SBATCH -J "ANLP"
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem-per-cpu=3000
#SBATCH -o moe.out
#SBATCH --time="4-00:00:00"
#SBATCH -w gnode082
python3 -m venv /tmp/ab/newvenv
source /tmp/ab/newvenv/bin/activate
pip install -r requirements.txt
echo "Time at entrypoint: $(date)"
echo "Working directory: ${PWD}"


echo start
python3 -m src.train_moe \
  --hinglish_config configs/hinglish_mbert_gru.yaml \
  --english_config configs/english_roberta_rcnn.yaml \
  --hinglish_weights outputs/hinglish_mbert_gru/best.pt \
  --english_weights outputs/english_roberta_rcnn/best.pt \
  --train_file_eng data/english/train.csv \
  --train_file_hin data/hinglish/train.csv \
  --val_file_eng data/english/val.csv \
  --val_file_hin data/hinglish/val.csv \
  --output_dir /tmp/ab/outputs/moe \
  --epochs 3 --batch_size 16 --freeze_backbones
echo done

echo "Time at exit: $(date)"
