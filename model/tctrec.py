import torch
import torch.nn as nn
from model.Bidimensional_Transformer import Bidimensional_Transformer

class TCTRec(nn.Module):
    def __init__(self, args):
        super(TCTRec, self).__init__()
        self.DEVICE = args.device
        self.alpha = args.alpha
        self.max_seq_length = args.max_seq_length
        self.dim_user_discrete_data = args.dim_user_discrete_data
        self.static_size = len(self.dim_user_discrete_data) + 1
        self.d_model = args.hidden_size
        self.num_heads = args.n_heads
        self.dropout_rate = args.attn_dropout_rate
        self.d_ff = args.hidden_size
        self.num_layers = args.e_layers
        self.num_items = args.num_items + 1

        self.BiTransformer = Bidimensional_Transformer(self.num_items, self.max_seq_length, self.static_size, self.d_model, self.num_heads, self.d_ff, self.num_layers, self.dropout_rate).to(self.DEVICE)
        self.item_bias = nn.Parameter(torch.zeros(self.num_items))
    
    def predict(self, user_ids, user_discrete_data, input_seq, seq_item_discrete_data,\
                test_items, test_item_discrete_data,  ratings, timestamps):
        if user_discrete_data is not None:
            user_features = torch.cat((user_ids.unsqueeze(-1), user_discrete_data), dim = -1)
        else:
            user_features = user_ids.unsqueeze(-1)
        
        variety = torch.cat([seq_item_discrete_data, timestamps.unsqueeze(-1), ratings.unsqueeze(-1)], dim = -1)

        transformer_output, invert_transformer_output = self.BiTransformer(input_seq, variety, user_features)
        
        test_item_features = self.BiTransformer.encoder.embedding(test_items)
        #test_item_features = self.BiTransformer.encoder.positional_encoding(test_item_features)################################
        test_logits = self.item_bias[test_items].squeeze() + self.alpha * torch.sum(test_item_features * transformer_output, dim=-1) + (1- self.alpha) * torch.sum(test_item_discrete_data * invert_transformer_output, dim=-1)################################
        
        return test_logits[:, -1]


    def forward(self, user_ids, user_discrete_data, input_seq, seq_item_discrete_data,\
                pos_seq, pos_item_discrete_data,\
                    neg_seq, neg_item_discrete_data,\
                     ratings, timestamps):
        
        if user_discrete_data is not None:
            user_features = torch.cat((user_ids.unsqueeze(-1), user_discrete_data), dim = -1)
        else:
            user_features = user_ids.unsqueeze(-1)
        
        variety = torch.cat([seq_item_discrete_data, timestamps.unsqueeze(-1), ratings.unsqueeze(-1)], dim = -1)

        mask = (input_seq != 0).unsqueeze(-1).to(self.DEVICE)
        transformer_output, invert_transformer_output = self.BiTransformer(input_seq, variety, user_features, mask)

        pos_item_features = self.BiTransformer.encoder.embedding(pos_seq)
        #pos_item_features = self.BiTransformer.encoder.positional_encoding(pos_item_features)########################
        neg_item_features = self.BiTransformer.encoder.embedding(neg_seq)
        #neg_item_features = self.BiTransformer.encoder.positional_encoding(neg_item_features)#################################
        
        pos_logits = self.item_bias[pos_seq].squeeze() + self.alpha * torch.sum(pos_item_features * transformer_output, dim=-1) + (1 - self.alpha) * torch.sum(pos_item_discrete_data * invert_transformer_output, dim=-1)################################

        neg_logits = self.item_bias[neg_seq].squeeze() + self.alpha * torch.sum(neg_item_features * transformer_output, dim=-1) + (1- self.alpha) * torch.sum(neg_item_discrete_data * invert_transformer_output, dim=-1)################################

        mask = mask.squeeze(-1)

        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * mask
        total_loss = torch.sum(pos_loss + neg_loss) / (torch.sum(mask) + 1e-24)

        return total_loss
