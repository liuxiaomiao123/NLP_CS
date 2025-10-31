from torch.nn import Module, Embedding, LSTM, Dropout, Linear, Sigmoid
from torch import zeros

class MovieLSTM(Module):
    def __init__(
            self, vocab_size: int, dim_hidden: int, dim_embed: int,
            num_hidden_layers: int, dropout_perc: float,
            batch_first: bool, device
        ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.dim_hidden = dim_hidden
        self.dim_embed = dim_embed
        self.dropout_perc = dropout_perc
        self.vocab_size = vocab_size
        self.batch_first = batch_first
        self.device = device
        self.embedding_lookup = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.dim_embed
        )
        self.drop = Dropout(p=self.dropout_perc)
        self.lstm = LSTM(
            input_size=self.dim_embed, hidden_size=self.dim_hidden, 
            num_layers=self.num_hidden_layers, batch_first=self.batch_first,
            dropout=self.dropout_perc
        )        
        self.linear_mapping = Linear(in_features=self.dim_hidden, out_features=1)
        self.activation = Sigmoid()
        

    def forward(self, inputs, hidden_states):
        embeddings = self.embedding_lookup(inputs)
        output, hidden_states = self.lstm(embeddings, hidden_states)
        output = output.contiguous().view(-1, self.dim_hidden)
        activations = self.activation(
            self.linear_mapping(
                self.drop(
                    output
                )
            )
        ).view(inputs.size(0), -1)[:, -1]
        return activations, hidden_states
    
    
    def initialize_hidden_states(self, b_size: int):
        hidden = zeros((self.num_hidden_layers, b_size, self.dim_hidden)).to(self.device)
        cell = zeros((self.num_hidden_layers, b_size, self.dim_hidden)).to(self.device)
        return hidden, cell   
