import random
import utils
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_processing import *
import warnings

DATASET_DICT = {
    'ml-100k': ml_100k,
    'ml-1m': ml_1m,
    'ml-20m': ml_20m,
    'ml-25m': ml_25m
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
length = 50            ####################

class UserItemDataset(Dataset):
    def __init__(self, args, data, model):
        self.data = data
        self.model = model
        self.num_users = args.num_users
        self.max_seq_length = args.max_seq_length
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, index):
        user_data = self.data.users_data.iloc[index]

        user_id = user_data['user_id']

        user_discrete_data = self.data.get_user_embedding(user_id)

        item_slice = {
            'train': slice(0, -2),
            'validate': slice(-self.max_seq_length - 2, -1),    #(L + vali)
            'test': slice(-self.max_seq_length - 1, None)       #(L + test)
        }.get(self.model)
        
        item_ids, item_discrete_data, ratings, timestamps = self.data.get_user_item_data(user_id)
        neg_item_ids, neg_item_discrete_data = self.data.get_neg_user_item_data(user_id, self.model)
        neg_item_ids = torch.tensor(neg_item_ids)
        neg_item_discrete_data = torch.tensor(neg_item_discrete_data)

        user_id = torch.tensor(user_id)
        return (user_id, user_discrete_data,
                item_ids[item_slice], item_discrete_data[item_slice],
                ratings[item_slice], timestamps[item_slice],
                neg_item_ids, neg_item_discrete_data)
    
def pad_sequence(seq, maxlen):
    if len(seq) > maxlen:
        return seq[-maxlen:]
    elif isinstance(seq, list) and all(isinstance(sublist, list) for sublist in seq):
        pad_length = maxlen - len(seq)
        padded_seq = [[0] * len(seq[0])] * pad_length + seq
        return padded_seq
    elif isinstance(seq, list):  
        pad_length = maxlen - len(seq)
        return [0] * pad_length + seq
    else:
        if seq.dim() == 1:
            zeros_tensor = torch.zeros(maxlen - len(seq), dtype=seq.dtype)
            return torch.cat((zeros_tensor, seq))
        elif seq.dim() == 2:
            zeros_tensor = torch.zeros((maxlen - seq.size(0), seq.size(1)), dtype=seq.dtype)
            return torch.cat((zeros_tensor, seq), dim=0)
        else:
            raise ValueError("Unsupported tensor dimension.")

def custom_collate_fn(batch):
    warnings.filterwarnings("ignore", category=UserWarning)
    new_batch = []
    for user_ids, user_discrete_data, item_ids_batch, item_discrete_data_batch, ratings, timestamps, \
            neg_item_ids_batch, neg_item_discrete_data_batch in batch:
        if len(item_ids_batch) > length:
            i = - length - 1

            current_ids = item_ids_batch[i:]
            current_item_discrete_data = item_discrete_data_batch[i:]

            seq_item_ids = current_ids[:-1]
            seq_item_discrete_data = current_item_discrete_data[:-1]

            pos_item_ids = current_ids[1:]
            pos_item_discrete_data = current_item_discrete_data[1:]

            neg_item_ids = neg_item_ids_batch[i+1:]
            neg_item_discrete_data = neg_item_discrete_data_batch[i+1:]
        
            current_ratings = ratings[i:][:-1]
            current_timestamps = timestamps[i:][:-1]
            new_batch.append((
                user_ids,
                user_discrete_data,
                seq_item_ids, seq_item_discrete_data,
                pos_item_ids, pos_item_discrete_data,
                neg_item_ids, neg_item_discrete_data,
                current_ratings, current_timestamps
            ))
        else:
            seq_item_ids = pad_sequence(item_ids_batch[:-1], length)
            seq_item_discrete_data = pad_sequence(item_discrete_data_batch[:-1], length)

            pos_item_ids = pad_sequence(item_ids_batch[1:], length)
            pos_item_discrete_data = pad_sequence(item_discrete_data_batch[1:], length)
            
            neg_item_ids = pad_sequence(neg_item_ids_batch[1:], length)
            neg_item_discrete_data = pad_sequence(neg_item_discrete_data_batch[1:], length)

            current_ratings = pad_sequence(ratings[:-1], length)
            current_timestamps = pad_sequence(timestamps[:-1], length)
            new_batch.append((
                    user_ids,
                    user_discrete_data,
                    seq_item_ids, seq_item_discrete_data,
                    pos_item_ids, pos_item_discrete_data,
                    neg_item_ids, neg_item_discrete_data,
                    current_ratings, current_timestamps
                ))
            
    user_ids, user_discrete_data, seq_item_ids, seq_item_discrete_data, pos_item_ids, pos_item_discrete_data, neg_item_ids, neg_item_discrete_data, current_ratings, current_timestamps = zip(*new_batch)
    
    user_ids = torch.stack(user_ids)

    if any(data is None  for data in user_discrete_data):
        user_discrete_data = None
    else:
        user_discrete_data = torch.stack([torch.tensor(data) for data in user_discrete_data])
        
    seq_item_ids = torch.stack([torch.tensor(i, dtype=torch.float32) for i in seq_item_ids]).to(torch.long)

    seq_item_discrete_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in seq_item_discrete_data])
    
    pos_item_ids = torch.stack([torch.tensor(i, dtype=torch.float32) for i in pos_item_ids]).to(torch.long)

    pos_item_discrete_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in pos_item_discrete_data])

    neg_item_ids = torch.stack([torch.tensor(i, dtype=torch.float32) for i in neg_item_ids]).to(torch.long)

    neg_item_discrete_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in neg_item_discrete_data])

    current_ratings = torch.stack([torch.tensor(i, dtype=torch.float32) for i in current_ratings])

    current_timestamps = torch.stack([torch.tensor(i, dtype=torch.float32) for i in current_timestamps])

    return user_ids, user_discrete_data, seq_item_ids, seq_item_discrete_data, \
            pos_item_ids, pos_item_discrete_data, neg_item_ids, neg_item_discrete_data, \
            current_ratings, current_timestamps

def test_custom_collate_fn(batch):
    warnings.filterwarnings("ignore", category=UserWarning)

    new_batch = []
    for user_ids, user_discrete_data, item_ids_batch, item_discrete_data_batch, ratings, timestamps, \
            neg_item_ids_batch, neg_item_discrete_data_batch in batch:

        seq_item_ids = pad_sequence(item_ids_batch[:-1], length)
        seq_item_discrete_data = pad_sequence(item_discrete_data_batch[:-1], length)

        pos_item_ids = item_ids_batch[-1]
        pos_item_discrete_data = item_discrete_data_batch[-1]
        
        neg_item_ids = neg_item_ids_batch
        neg_item_discrete_data = neg_item_discrete_data_batch

        current_ratings = pad_sequence(ratings[:-1], length)
        current_timestamps = pad_sequence(timestamps[:-1], length)

        new_batch.append((
                user_ids,
                user_discrete_data,
                seq_item_ids, seq_item_discrete_data,
                pos_item_ids, pos_item_discrete_data,
                neg_item_ids, neg_item_discrete_data,
                current_ratings, current_timestamps
            ))
    user_ids, user_discrete_data, seq_item_ids, seq_item_discrete_data, pos_item_ids, pos_item_discrete_data, neg_item_ids, neg_item_discrete_data, current_ratings, current_timestamps = zip(*new_batch)
    
    user_ids = torch.stack(user_ids)

    if any(data is None  for data in user_discrete_data):
        user_discrete_data = None
    else:
        user_discrete_data = torch.stack([torch.tensor(data) for data in user_discrete_data])

    seq_item_ids = torch.stack([torch.tensor(i, dtype=torch.float32) for i in seq_item_ids]).to(torch.long)

    seq_item_discrete_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in seq_item_discrete_data])

    pos_item_ids = torch.stack([torch.tensor(i, dtype=torch.float32) for i in pos_item_ids]).to(torch.long)

    pos_item_discrete_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in pos_item_discrete_data])

    neg_item_ids = torch.stack([torch.tensor(i, dtype=torch.float32) for i in neg_item_ids]).to(torch.long)

    neg_item_discrete_data = torch.stack([torch.tensor(i, dtype=torch.float32) for i in neg_item_discrete_data])

    current_ratings = torch.stack([torch.tensor(i, dtype=torch.float32) for i in current_ratings])

    current_timestamps = torch.stack([torch.tensor(i, dtype=torch.float32) for i in current_timestamps])

    test_neg_item_ids_batch = torch.cat((pos_item_ids.unsqueeze(1), neg_item_ids), dim=1)
    test_neg_item_discrete_data_batch = torch.cat((pos_item_discrete_data.unsqueeze(1), neg_item_discrete_data), dim=1)


    return user_ids, user_discrete_data, seq_item_ids, seq_item_discrete_data, \
            test_neg_item_ids_batch, test_neg_item_discrete_data_batch, \
            current_ratings, current_timestamps

def get_dataset(args, logger):
    data_class = DATASET_DICT.get(args.data_name.lower())
    if data_class is None:
        raise ValueError(f"Unknown dataset name: {args.data_name}")
    logger.info(f'read {data_class}')
    data = data_class(args, logger)
    return data

def get_dataloader(args, data):
    #import multiprocessing
    #multiprocessing.set_start_method('spawn', force=True)

    train_dataset = UserItemDataset(args, data, 'train')
    valid_dataset = UserItemDataset(args, data, 'validate')
    test_dataset = UserItemDataset(args, data, 'test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers = 12, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = test_custom_collate_fn, num_workers = 12, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_custom_collate_fn, num_workers = 12, pin_memory=True)
    return train_dataloader, valid_dataloader, test_dataloader