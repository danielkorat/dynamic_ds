python run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir output \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 1