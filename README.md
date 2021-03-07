# Dynamic Data Streaming Methods

## Install
 
```
git clone https://github.com/danielkorat/learning-ds.git
cd learning-ds
python3.6 -m pip install -U pip virtualenv
python3.6 -m virtualenv .env
source .env/bin/activate

```
## recreating graphs
to recreate graphs from original paper, do the following:
1. download from here the predictions https://drive.google.com/file/d/1PlmYUYEWHKJWOOyR1GrBuV3mcbSBdpiV/view and extract
2. download the saved ```param_results``` here: https://drive.google.com/open?id=1n2jDVhvKPwtFevyej42hGRieOeqIYoSJ
download

3. to plot the loss vs space used for count sketch, run:
```
python3 plotting\plot_loss_vs_space.py --algo "Count-Sketch" --count_min ../param_results/count_sketch/csketch_ip_1329.npz --learned ../param_results/cutoff_count_sketch_param/csketch_ip_1329_ru64_test.npz --perfect ../param_results/cutoff_count_sketch_param_perfect/csketch_ip_1329_pcut_test.npz --lookup_table ../param_results/lookup_table_count_sketch/csketch_ip_1329_test.npz --model_names "Learned Count-Sketch (NNet)" --title "IP - @ 20 th test minute - model ru64" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200
``` 

4. to plot the loss vs space used for count min, run:
```
python3 --algo "Count-Min" --count_min ../param_results/count_min/cmin_ip_1329.npz --learned ../param_results/cutoff_count_min_param/cmin_ip_1329_ru64_test.npz --perfect ../param_results/cutoff_count_min_param_perfect/cmin_ip_1329_pcut_test.npz --lookup_table ../param_results/lookup_table_count_min/cmin_ip_1329_test.npz --model_names "Learned Count-Min (NNet)" --title "IP - @ 20 th test minute - model ru64" --model_sizes 0.0031 --lookup_size 0.0035 --x_lim 0 2 --y_lim 0 200
``` 
