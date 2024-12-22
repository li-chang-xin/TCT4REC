import numpy as np
import torch
import time
from ML_dataloader import pad_sequence

class Train():
    def __init__(self, model, data, train_dataloader, valid_dataloader, test_dataloader, args, logger):
        self.model = model.to(args.device)
        self.data = data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        self.logger = logger
    
    def load(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info("training end, saved at {checkpoint_path}")

    def evaluate(self, epoch, mode='validate'):
        self.logger.info(f'-----------------{mode}-----------------')
        
        # Metrics initialization
        nDCG5, HR5, nDCG10, HR10, nDCG20, HR20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        test_user = 0.0
        if mode == 'validate':
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (user_ids, user_discrete_data, seq_item_ids, seq_item_discrete_data, \
                test_neg_item_ids_batch, test_neg_item_discrete_data_batch,\
                train_ratings_batch, train_timestamps_batch) in enumerate(dataloader):

                test_user += user_ids.size(0)

                n = self.args.num_neg_sample + 1
                m = 20
                results = []
                for i in range(len(user_ids)):
                    test_neg_item_ids = torch.cat((seq_item_ids[i][1:].unsqueeze(0).repeat(n, 1), test_neg_item_ids_batch[i].unsqueeze(1)), dim=1)

                    
                    test_neg_item_discrete_data = torch.cat((seq_item_discrete_data[i][1:].unsqueeze(0).repeat(n, 1, 1), test_neg_item_discrete_data_batch[i].unsqueeze(1)), dim=1)
                    result = []
                    for j in range(0, n, m):

                        points = self.model.predict(user_ids[i].repeat(m).to(self.args.device),
                                                    user_discrete_data[i].repeat(m, 1).to(self.args.device) if user_discrete_data is not None else None,
                                                    seq_item_ids[i].repeat(m, 1).to(self.args.device),
                                                    seq_item_discrete_data[i].repeat(m, 1, 1).to(self.args.device),
                                                    test_neg_item_ids[j:j+m].to(self.args.device),
                                                    test_neg_item_discrete_data[j:j+m].to(self.args.device),
                                                    train_ratings_batch[i].repeat(m, 1).to(self.args.device),
                                                    train_timestamps_batch[i].repeat(m, 1).to(self.args.device))
                        result += points
                    results.append(torch.tensor(result).argsort(descending=True).argsort()[0] + 1)
                # Calculate ranking metrics
                for inx, rank in enumerate(results):

                    if rank <= 5:
                        nDCG5 += 1 / np.log2(rank + 2)
                        HR5 += 1

                    if rank <= 10:
                        nDCG10 += 1 / np.log2(rank + 2)
                        HR10 += 1

                    if rank <= 20:
                        nDCG20 += 1 / np.log2(rank + 2)
                        HR20 += 1

                if batch_idx % 10 == 0:
                    print('.', end='', flush=True)

        # Log results
        self.logger.info(f'Epoch {epoch+1}, total_user:{test_user} nDCG@5: {nDCG5/test_user}, HR5: {HR5/test_user}, nDCG@10: {nDCG10/test_user}, HR10: {HR10/test_user}, nDCG@20: {nDCG20/test_user}, HR20: {HR20/test_user}')

        return nDCG5/test_user, HR5/test_user, nDCG10/test_user, HR10/test_user, nDCG20/test_user, HR20/test_user
  
    def train(self, epoch, optimizer, scheduler):
        #torch.multiprocessing.set_start_method('spawn', force=True)
        #torch.multiprocessing.set_start_method('spawn', force=True)
        self.model.train()
        start_time = time.time()
        epoch_loss = 0
        num_batch = 0
        for batch_idx, (user_ids, user_discrete_data, seq_item_ids, seq_item_discrete_data, \
            pos_item_ids, pos_item_discrete_data, neg_item_ids, neg_item_discrete_data, \
            current_ratings, current_timestamps) in enumerate(self.train_dataloader):

            optimizer.zero_grad()
            total_loss = self.model(user_ids.to(self.args.device),
                    user_discrete_data.to(self.args.device) if user_discrete_data is not None else None,
                    seq_item_ids.to(self.args.device), seq_item_discrete_data.to(self.args.device),
                    pos_item_ids.to(self.args.device), pos_item_discrete_data.to(self.args.device),
                    neg_item_ids.to(self.args.device), neg_item_discrete_data.to(self.args.device),
                    current_ratings.to(self.args.device), current_timestamps.to(self.args.device)
                )
    
            total_loss.backward()
            optimizer.step()
            batch_loss = total_loss.item()
            
            epoch_loss += batch_loss
            num_batch += 1
            if batch_idx % 32 == 0:
                end_time = time.time() - start_time
                self.logger.info(f'\rEpoch: {epoch+1}/{self.args.num_epochs} Batch: {batch_idx+1}/{len(self.train_dataloader)} Time: {end_time:.1f}s Loss: {batch_loss:.2f}   Average_Loss: {epoch_loss / num_batch:.2f}')
            torch.cuda.empty_cache()
                    
        scheduler.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        self.logger.info(f'\nEpoch {epoch+1}/{self.args.num_epochs},  Loss: {epoch_loss / num_batch}, Total time: {hours}h {minutes}m {seconds}s')
        