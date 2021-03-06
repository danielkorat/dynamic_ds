# Dynamic Data Streaming Methods

## Install + Train
 
```
git clone https://github.com/danielkorat/learning-ds.git
cd learning-ds
python3.6 -m pip install -U pip virtualenv
python3.6 -m virtualenv .env
source .env/bin/activate

pip install gdown
gdown --id 1jWot63TJY8WHk5_bQPgIz-EPV0a6BDVX
tar -xf aol_char_embed.tar.gz

gdown --id 1RclH6mFvbGKm5aB4MQT9HVgQO94GLNmd
tar -xf aol_query_counts.tar.gz

pip install -r requirements.txt

python3 run_aol_model.py \
    --train 1-day_len60/aol_0000_len60.npz \
            1-day_len60/aol_0001_len60.npz \
            1-day_len60/aol_0002_len60.npz \
            1-day_len60/aol_0003_len60.npz \
            1-day_len60/aol_0004_len60.npz \
    --valid 1-day_len60/aol_0005_len60.npz \
    --test  1-day_len60/aol_0006_len60.npz \
    --save exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra  --embed_size 64 --rnn_hidden 256 --hiddens 32 --batch_size 128 --n_epoch 2000 --lr 0.0001 --word_max_len 60 --regress_actual --eval_n 10
```
