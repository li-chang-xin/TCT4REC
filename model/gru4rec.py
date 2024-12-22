import torch
from torch import nn
from torch.nn.init import xavier_uniform_

class GRU4Rec(nn.Module):

    def __init__(self, args):
        super(GRU4Rec, self).__init__()

        self.args = args
        self.num_items = args.num_items + 1
        self.embedding_size = args.hidden_size
        self.hidden_size = args.gru_hidden_size
        self.num_layers = args.num_hidden_layers
        self.dropout_prob = args.dropout_rate
        
        self.item_embeddings = nn.Embedding(self.num_items, self.hidden_size, padding_idx=0)

        # Define GRU layers
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, user_ids, user_discrete_data, input_seq, seq_item_discrete_data,\
                pos_seq, pos_item_discrete_data,\
                    neg_seq, neg_item_discrete_data,\
                     ratings, timestamps):
        # Get item embeddings for the input sequence
        item_seq_emb = self.item_embeddings(input_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)

        gru_output, _ = self.gru_layers(item_seq_emb)

        seq_out = self.dense(gru_output) 

        pos_emb = self.item_embeddings(pos_seq)  
        neg_emb = self.item_embeddings(neg_seq)  

        pos_logits = torch.sum(pos_emb * seq_out, -1) 
        neg_logits = torch.sum(neg_emb * seq_out, -1)

        gamma = 1e-10
        loss = -torch.log(gamma + torch.sigmoid(pos_logits - neg_logits)).mean()

        return loss

    def predict(self, user_ids, user_discrete_data, input_seq, seq_item_discrete_data,\
                test_items, test_item_discrete_data,  ratings, timestamps):

        item_seq_emb = self.item_embeddings(input_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)

        # Pass through GRU layers
        gru_output, _ = self.gru_layers(item_seq_emb)

        gru_output = self.dense(gru_output) 
        test_emb = self.item_embeddings(test_items)  

        test_logits = torch.sum(test_emb * gru_output, -1)

        return test_logits[:, -1]
