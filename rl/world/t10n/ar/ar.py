import torch
from torch import nn

from ..util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    N_HEX_ACTIONS,
)


def generate_causal_mask(N: int) -> torch.Tensor:
    """Create an (N, N) mask where mask[i, j] = True if j > i."""
    return torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)


class ARTransitionModel(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()

        self.d_model = 128

        self.in_proj_global_enc = nn.LazyLinear(self.d_model)
        self.in_proj_player_enc = nn.LazyLinear(self.d_model)
        self.in_proj_hex_enc = nn.LazyLinear(self.d_model)

        # 168 = 165 hexes + 2 players + 1 global
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, 168, self.d_model))
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, 168, self.d_model))

        # encoder & decoder stacks
        self.transformer_enc = nn.TransformerEncoder(
            num_layers=3,
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=4,
                dropout=0.3,
                batch_first=True
            )
        )

        # start‐of‐sequence token for decoder
        self.sos_global_dec = nn.Parameter(torch.zeros(1, 1, STATE_SIZE_GLOBAL))
        self.sos_player_dec = nn.Parameter(torch.zeros(1, 1, STATE_SIZE_ONE_PLAYER))
        self.sos_hex_dec = nn.Parameter(torch.zeros(1, 1, STATE_SIZE_ONE_HEX))

        self.in_proj_global_dec = nn.LazyLinear(self.d_model)
        self.in_proj_player_dec = nn.LazyLinear(self.d_model)
        self.in_proj_hex_dec = nn.LazyLinear(self.d_model)

        self.transformer_dec = nn.TransformerDecoder(
            num_layers=3,
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=4,
                dropout=0.3,
                batch_first=True
            )
        )

        self.out_proj_global_dec = nn.LazyLinear(STATE_SIZE_GLOBAL)
        self.out_proj_player_dec = nn.LazyLinear(STATE_SIZE_ONE_PLAYER)
        self.out_proj_hex_dec = nn.LazyLinear(STATE_SIZE_ONE_HEX)

    def forward(self, obs, next_obs) -> torch.Tensor:
        B = x.size(0)

        split_sizes = [STATE_SIZE_GLOBAL, 2*STATE_SIZE_ONE_PLAYER, 165*STATE_SIZE_ONE_HEX]

        global_in, player_in, hex_in = obs.split(split_sizes, dim=1)
        global_in = global_in.view(B, 1, STATE_SIZE_GLOBAL)
        player_in = player_in.view(B, 2, STATE_SIZE_ONE_PLAYER)
        hex_in = hex_in.view(B, 165, STATE_SIZE_ONE_HEX)

        global_emb = self.in_proj_global_enc(global_in)
        player_emb = self.in_proj_player_enc(player_in)
        hex_emb = self.in_proj_hex_enc(hex_in)
        # => (B, {1,2,165}, d_model)

        # Join all tokens: 1xGLOBAL + 2xPLAYER + 165xHEX
        all_enc = torch.cat([global_emb, player_emb, hex_emb], dim=1) + self.pos_embed_enc
        memory = self.transformer_enc(all_enc)
        # => (B, 168, d_model)

        # Prepare decoder inputs with teacher-forcing
        next_global, next_player, next_hex = next_obs.split(split_sizes, dim=1)

        # Expand sos to batch dim
        sos_global = self.sos_global_dec.expand(B, 1, STATE_SIZE_GLOBAL)
        sos_player = self.sos_player_dec.expand(B, 1, STATE_SIZE_ONE_PLAYER)
        sos_hex = self.sos_hex_dec.expand(B, 1, STATE_SIZE_ONE_HEX)

        # Shift sequences right & prepend sos token
        next_global_in = torch.cat([sos_global, next_global[:, :-1, :]], dim=1)
        next_player_in = torch.cat([sos_player, next_player[:, :-1, :]], dim=1)
        next_hex_in = torch.cat([sos_hex, next_hex[:, :-1, :]], dim=1)

        global_dec = self.in_proj_global_dec(global_in)
        player_dec = self.in_proj_player_dec(player_in)
        hex_dec = self.in_proj_hex_dec(hex_in)
        # => (B, {1,2,165}, d_model)

        # embed and combine
        g_dec = self.global_proj_dec(g_in)          # (B, d_model)
        tiles_dec = self.tile_proj_dec(tiles_in)    # (B, N, d_model)
        h_dec = torch.cat([g_dec.unsqueeze(1), tiles_dec], dim=1)  # (B, N+1, d_model)
        h_dec = h_dec + self.pos_embed_dec          # (1, N+1, d_model)

        sos_b = self.sos.expand(B, -1, -1)           # (B, 1, feature_dim)
        y_in = torch.cat([sos_b, y[:, :-1, :]], dim=1)# (B, N, feature_dim)

        # 3) Embed decoder inputs
        h_dec = self.input_proj_dec(y_in)            # (B, N, d_model)
        h_dec = h_dec + self.pos_embed_dec           # add positional encodings

        # 4) Build causal mask
        tgt_mask = generate_causal_mask(self.N).to(x.device)  # (N, N)

        # 5) Decode with cross‐attention to encoder memory
        dec_out = self.decoder(
            tgt=h_dec,                                # (B, N, d_model)
            memory=memory,                           # (B, N, d_model)
            tgt_mask=tgt_mask                        # enforce autoregression
        )                                           # → (B, N, d_model)

        # 6) Project back to feature space
        y_pred = self.output_proj(dec_out)           # (B, N, feature_dim)
        return y_pred

# Example usage:
if __name__ == "__main__":
    B, N, F = 4, 165, 8     # batch size, num tiles, features per tile
    model = GridTransformerAR(
        feature_dim=F,
        d_model=64,
        nhead=8,
        num_enc_layers=3,
        num_dec_layers=3,
        dropout=0.1,
        N=N
    )
    x = torch.randn(B, N, F)    # current state
    y = torch.randn(B, N, F)    # true next state (for teacher‐forcing)
    y_pred = model(x, y)        # (B, N, F)
    print("Prediction shape:", y_pred.shape)
