# Learning-Based Bigrams Frequency Estimation Algorithms

## Requirements
**[Python 3.6](https://www.python.org/ftp/python/3.6.12/Python-3.6.12.tgz)**

Developed on Ubuntu 16.04.3 LTS

## Install
 
```
git clone https://github.com/danielkorat/learning-ds.git
cd learning-ds
pip install -r requirements
```

## Notebook

See the included [notebook](https://github.com/danielkorat/learning-ds/blob/main/notebook.ipynb) for a full execution example.

## Run Pipeline

1. Evaluate Count-Min and Count-Sketch on this data (not learned, original algorithms): 

   ```python3 count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --count_sketch --save cmin_connel --n_workers 30 --data_name conll --data conll2003.json```

2. Generate bigrams dataset and train a simple model for frequency predictions + saving test/train/validation true and prediction results:

   ```python3 ./nlp/model.py```
   
3. Evaluate learned Count-Min on conll data with lookup data:

   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --lookup_data ./nlp/true_conll2003_train.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll```
   
4. Evaluate learned Count-Min on conll data without lookup data:

   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll```

5. Evaluate learned Count-Sketch on conll data with lookup data:

   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --lookup_data ./nlp/true_conll2003_train.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll --count_sketch```
   
6. Evaluate learned Count-Sketch on conll data without lookup data:

   ```python3 cutoff_count_min_param.py --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 3 4 --n_hashes 1 2 3 4 --save cmin_ip_conll --n_workers 30 --test_data ./nlp/true_conll2003_test.npz --valid_data ./nlp/true_conll2003_valid.npz --test_result ./nlp/pred_conll2003.npz --valid_result ./nlp/pred_conll2003.npz --data_name conll```
   
7. Plot Count-Sketch loss (learned vs perfect vs original):

```python3 plot_loss_vs_space.py --algo "Count-sketch" --count_min ../param_results/count_min/cmin_connel.npz --learned ../param_results/cutoff_count_sketch_param/cmin_ip_conll_test.npz --perfect ../param_results/cutoff_count_sketch_param_perfect/cmin_ip_conll_test.npz --lookup_table ../param_results/lookup_table_count_sketch/cmin_ip_conll_test.npz --model_names "Learned Count-sketch (NNet)" --title "conll" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200```

8. Plot Count-Min loss (learned vs perfect vs original):

```--algo "Count-Min" --count_min ../param_results/count_min/cmin_connel.npz --learned ../param_results/cutoff_count_min_param/cmin_ip_conll_test.npz --perfect ../param_results/cutoff_count_min_param_perfect/cmin_ip_conll_test.npz --lookup_table ../param_results/lookup_table_count_min/cmin_ip_conll_test.npz --model_names "Learned Count-Min (NNet)" --title "conll" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200```
   

## Recreating Graphs

To recreate graphs from original paper, do the following:

1. Download the predictions [here](https://drive.google.com/file/d/1PlmYUYEWHKJWOOyR1GrBuV3mcbSBdpiV/view) and extract.

2. Download the saved ```param_results``` [here](https://drive.google.com/open?id=1n2jDVhvKPwtFevyej42hGRieOeqIYoSJ) and extract.

3. To plot the loss vs space used for count sketch (on ip flows), run:

```
python3 plotting\plot_loss_vs_space.py --algo "Count-Sketch" --count_min ../param_results/count_sketch/csketch_ip_1329.npz --learned ../param_results/cutoff_count_sketch_param/csketch_ip_1329_ru64_test.npz --perfect ../param_results/cutoff_count_sketch_param_perfect/csketch_ip_1329_pcut_test.npz --lookup_table ../param_results/lookup_table_count_sketch/csketch_ip_1329_test.npz --model_names "Learned Count-Sketch (NNet)" --title "IP - @ 20 th test minute - model ru64" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200
``` 

4. To plot the loss vs space used for count min (on ip flows), run:

```
python3 plotting\plot_loss_vs_space.py --algo "Count-Min" --count_min ../param_results/count_min/cmin_ip_1329.npz --learned ../param_results/cutoff_count_min_param/cmin_ip_1329_ru64_test.npz --perfect ../param_results/cutoff_count_min_param_perfect/cmin_ip_1329_pcut_test.npz --lookup_table ../param_results/lookup_table_count_min/cmin_ip_1329_test.npz --model_names "Learned Count-Min (NNet)" --title "IP - @ 20 th test minute - model ru64" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200
``` 
