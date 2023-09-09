from datasets import get_dir_dataset
import torch
from train import train_baseline_dir
from train_causal import train_causal_dir
import opts
import warnings
warnings.filterwarnings('ignore')

def main():
    args = opts.parse_args()
    args.hidden = 32
    args.drop_out = 0.5
    train_dataset, val_dataset,test_dataset = get_dir_dataset(args)
    model_func = opts.get_model(args)
    if args.model in ["GIN","GCN", "GAT"]:
        train_baseline_dir(train_dataset, val_dataset, test_dataset,model_func=model_func, args=args)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        train_causal_dir(train_dataset, val_dataset, test_dataset,model_func=model_func, args=args)
    else:
        assert False
if __name__ == '__main__':
    main()