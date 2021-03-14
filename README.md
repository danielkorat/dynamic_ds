# Dynamic Data Streaming Methods

## Install
 
```
git clone https://github.com/danielkorat/learning-ds.git
cd learning-ds
python3.6 -m pip install -U pip virtualenv
python3.6 -m virtualenv .env
source .env/bin/activate

```

## TODO add notebook link

## recreating graphs
to recreate graphs from original paper, do the following:
1. download from here the predictions https://drive.google.com/file/d/1PlmYUYEWHKJWOOyR1GrBuV3mcbSBdpiV/view and extract
2. download the saved ```param_results``` here: https://drive.google.com/open?id=1n2jDVhvKPwtFevyej42hGRieOeqIYoSJ
download

3. to plot the loss vs space used for count sketch (on ip flows), run:
```
python3 plotting\plot_loss_vs_space.py --algo "Count-Sketch" --count_min ../param_results/count_sketch/csketch_ip_1329.npz --learned ../param_results/cutoff_count_sketch_param/csketch_ip_1329_ru64_test.npz --perfect ../param_results/cutoff_count_sketch_param_perfect/csketch_ip_1329_pcut_test.npz --lookup_table ../param_results/lookup_table_count_sketch/csketch_ip_1329_test.npz --model_names "Learned Count-Sketch (NNet)" --title "IP - @ 20 th test minute - model ru64" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200
``` 

4. to plot the loss vs space used for count min (on ip flows), run:
```
python3 plotting\plot_loss_vs_space.py --algo "Count-Min" --count_min ../param_results/count_min/cmin_ip_1329.npz --learned ../param_results/cutoff_count_min_param/cmin_ip_1329_ru64_test.npz --perfect ../param_results/cutoff_count_min_param_perfect/cmin_ip_1329_pcut_test.npz --lookup_table ../param_results/lookup_table_count_min/cmin_ip_1329_test.npz --model_names "Learned Count-Min (NNet)" --title "IP - @ 20 th test minute - model ru64" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200
``` 

## run pipeline
1. run 
   ```python3 ./nlp/data.py```
   to create a json with true counts of the data.
2. run 
   ```python3 count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --count_sketch --save cmin_connel --n_workers 30 --data_name conll --data conll2003.json```
   to evaluate count_min and count_sketch on this data (not learned, original algorithms)
3. run
   ```python3 ./nlp/model.py```
   to create a simple model for frequency predictions + saving test/train/validation true 
   and prediction results.
   
4. run  
   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --lookup_data ./nlp/true_conll2003_train.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll```
   to evaluate learned count min on conll data with lookup data.
   
5. run  
   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll```
   to evaluate learned count min on conll data without lookup data.

6. run 
   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --lookup_data ./nlp/true_conll2003_train.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll --count_sketch```
   to evaluate learned count sketch on conll data with lookup data.
   
7. run 
   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll```
   to evaluate learned count sketch on conll data without lookup data.
   
8. run 
```python3 plot_loss_vs_space.py --algo "Count-sketch" --count_min ../param_results/count_min/cmin_connel.npz --learned ../param_results/cutoff_count_sketch_param/cmin_ip_conll_test.npz --perfect ../param_results/cutoff_count_sketch_param_perfect/cmin_ip_conll_test.npz --lookup_table ../param_results/lookup_table_count_sketch/cmin_ip_conll_test.npz --model_names "Learned Count-sketch (NNet)" --title "conll" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200```
   to plot count sketch loss (learned vs perfect vs original)

9. run
```--algo "Count-Min" --count_min ../param_results/count_min/cmin_connel.npz --learned ../param_results/cutoff_count_min_param/cmin_ip_conll_test.npz --perfect ../param_results/cutoff_count_min_param_perfect/cmin_ip_conll_test.npz --lookup_table ../param_results/lookup_table_count_min/cmin_ip_conll_test.npz --model_names "Learned Count-Min (NNet)" --title "conll" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200```
   to plot count min loss (learned vs perfect vs original)
   