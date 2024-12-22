import torch
import numpy as np

import os
import random
import datetime
import argparse
import logging

def set_logger(log_path, log_name='BiT4REC', mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def parse_args():
    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="ml-100k", type=str)       ##ml-100k, ml-1m, ml-20m, ml-25m
    parser.add_argument("--llm_path", default="./bert-base-uncased", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default='GRU4Rec', type=str)       ##TCTRec, SASRec, GRU4Rec, BSARec
    parser.add_argument("--train_name", default=get_local_time(), type=str)
    parser.add_argument("--num_items", default=10, type=int)
    parser.add_argument("--num_users", default=10, type=int)
    parser.add_argument("--num_neg_sample", default=100, type=int)

    # train args
    parser.add_argument("--lr", default=0.003, type=float, help="learning rate of adam")
    parser.add_argument("--weight_decay", default=0, type=int)
    parser.add_argument("--batch_size", default=128, type=int, help="number of batch_size")
    parser.add_argument("--num_epochs", default=64, type=int, help="number of epochs")
    parser.add_argument("--patience", default=5, type=int, help="how long to wait after last time validation loss improved")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers of dataloader")
    parser.add_argument("--seed", default=42, type=int)

    # model args
    parser.add_argument("--model_type", default='TCTRec', type=str)         ##TCTRec, SASRec, GRU4Rec, BSARec
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--hidden_size", default=64, type=int, help="embedding dimension")
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--attn_dropout_rate", default=0.5, type=float)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--e_layers", default=4, type=int)
    parser.add_argument("--alpha", default=0.8, type=int)
    #parser.add_argument("--hidden_act", default="gelu", type=str)          # gelu relu
    #parser.add_argument("--num_attention_heads", default=2, type=int)
    #parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    

    args, _ = parser.parse_known_args()

    if args.model_type.lower() == 'bit4rec':
        parser.add_argument("--llm", default=False, action='store_true')

    elif args.model_type.lower() == 'sasrec':
        parser.add_argument("--llm", default=False, action='store_false')

    elif args.model_type.lower() == 'caser':
        parser.add_argument("--nh", default=8, type=int)
        parser.add_argument("--nv", default=4, type=int)
        parser.add_argument("--reg_weight", default=1e-4, type=float)

    elif args.model_type.lower() == 'duorec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)

    elif args.model_type.lower() == 'fearec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--global_ratio", default=0.6, type=float)
        parser.add_argument("--fredom_type", default='us_x', type=str)
        parser.add_argument("--fredom", default='True', type=str) # use eval function to use as boolean

    elif args.model_type.lower() == 'gru4rec':
        parser.add_argument("--gru_hidden_size", default=64, type=int, help="hidden size of GRU")
        parser.add_argument("--initializer_range", default=0.02, type=float)


    return parser.parse_args()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, logger, patience=5, verbose=False, delta=0.001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)