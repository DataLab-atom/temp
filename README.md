## Dependencies

Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).
Typically, you might need to run the following commands:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric == 2.0.2
pip install torch-scatter  == 2.0.9
pip install torch-sparse == 0.6.15 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install networkx                    
pip install matplotlib  
pip install dgl-cu101
```

## Experiments

### For dir datasets 
```
python main_dir.py --dataset mnist --bias 0.8 --model 'CausalGCN'  --train_model swl
python main_dir.py --dataset mnist --bias 0.85 --model 'CausalGCN'  --train_model mgda --mgda_with_double_co True
```
### For TU datasets

```
python main_real.py --model CausalGAT --dataset MUTAG --train_model swl
python main_real.py --model CausalGAT --dataset MUTAG --train_model mgda --mgda_with_double_co True
```

## Data download
dir datasets please look at Folder ``dir_data_geter``
TU datasets can be downloaded when you run ``main_real.py``.

