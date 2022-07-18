
from torch import nn


class linear_combiner(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 dropout):

        self.__dict__.update(locals())
        super().__init__()
        self.norm= nn.ModuleList([])
        self.linear = nn.ModuleList([])

        assert hidden_dim >= num_layers**2

        for i in range(num_layers):
            layer_in_dim = input_dim if i ==0 else layer_out_dim
            layer_out_dim = hidden_dim // (2**i)
            self.linear.append(nn.Linear(layer_in_dim, layer_out_dim))
            self.norm.append(nn.LayerNorm(layer_out_dim))




        self.out_layer = nn.Linear(layer_out_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, x):
        for layer, norm in zip(self.linear, self.norm):
            x = self.activation(self.dropout(norm(layer(x))))

        return self.out_layer(x)




