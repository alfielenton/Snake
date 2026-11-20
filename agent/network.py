import torch
from torch import nn

class Network(nn.Module):

    def __init__(self, grid_size, num_foods, num_actions, embed_size = 512, num_encoder_layers = 8, num_decoder_layers = 8):

        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_foods = num_foods
        self.grid_size = grid_size
        self.num_actions = num_actions

        self.d_model = embed_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.body_embedder = nn.Linear(2, self.d_model)
        self.body_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = 8, batch_first = True)
        self.body_encoder = nn.TransformerEncoder(self.body_encoder_layer, num_layers = self.num_encoder_layers)

        self.velocity_embedder = nn.Linear(2, self.d_model)
        self.food_embedder = nn.Linear(2, self.d_model)

        self.body_food_decoder_layer = nn.TransformerDecoderLayer(d_model = self.d_model, nhead = 8, batch_first = True)
        self.body_food_decoder = nn.TransformerDecoder(self.body_food_decoder_layer, num_layers = self.num_decoder_layers)

        self.q_value_predictor = nn.Linear(self.d_model, self.num_actions)
                                               
    def get_padding_mask(self, body_batch):

        seq_sizes = [len(batch) for batch in body_batch]
        max_seq_size = max(seq_sizes)

        mask = torch.zeros((len(body_batch), max_seq_size)) + 1
        for i, size in enumerate(seq_sizes):
            mask[i, size:] = 0 

        return mask.to(torch.bool).to(self.device)

    def get_position_encoder(self, seq_size):

        pos = torch.arange(seq_size)
        i = torch.arange(self.d_model)
        pe = torch.zeros((seq_size, self.d_model))

        pe[:, ::2] = torch.sin(pos[:, None] / 10000 ** (i[::2] / self.d_model))
        pe[:, 1::2] = torch.cos(pos[:, None] / 10000 ** (i[1::2] / self.d_model))
        return pe.to(self.device)

    def forward(self, body, vel, foods):

        body_mask = self.get_padding_mask(body)
        max_seq_size = body_mask.size(1)
        body_embed = []
        for body_inp in body:
            size = len(body_inp)
            body_inp = torch.tensor(body_inp).to(self.device) / self.grid_size

            body_emb = self.body_embedder(body_inp)
            
            padding = torch.zeros((max_seq_size - size, self.d_model)).to(self.device)
            b = torch.cat((body_emb, padding), dim=0)
            body_embed.append(b)

        pe = self.get_position_encoder(max_seq_size)

        vel_inp = torch.tensor(vel, dtype=torch.float32, device=self.device)
        food_inp = torch.tensor(foods, device=self.device) / self.grid_size

        body_embed = torch.stack(body_embed) + pe
        vel_embed = self.velocity_embedder(vel_inp)
        food_embed = self.food_embedder(food_inp)

        body_enc = self.body_encoder(body_embed, src_key_padding_mask = body_mask)
        vel_food_embed = torch.cat((vel_embed[:, None, :], food_embed), dim=1)

        dec_out = self.body_food_decoder(vel_food_embed, body_enc)
        dec_out = dec_out[:, -1, :]

        predicts = self.q_value_predictor(dec_out)
        return predicts
