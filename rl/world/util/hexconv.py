import torch
import torch.nn as nn


class HexConv(nn.Module):
    """
    1. Adds 1 hex of padding to the input

                  0 0 0 0
       1 2 3     0 1 2 3 0
        4 5 6 =>  0 4 5 6 0
       7 8 9     0 7 8 9 0
                  0 0 0 0

    2. Simulates a Conv2d with kernel_size=2, padding=1

     For the above example (grid of 9 total hexes), this would result in:

     1 => [0, 0, 0, 1, 2, 0, 4]
     2 => [0, 0, 1, 2, 3, 4, 5]
     3 => [0, 0, 2, 3, 0, 5, 6]
     4 => [1, 2, 0, 4, 5, 7, 8]
     ...
     9 => [5, 6, 8, 9, 0, 0, 0]

    Input: (B, ...) reshapeable to (B, Y, X, E)
    Output: (B, 165, out_channels)
    """
    def __init__(self, out_channels):
        super().__init__()

        padded_offsets0 = torch.tensor([-17, -16, -1, 0, 1, 17, 18])
        padded_offsets1 = torch.tensor([-18, -17, -1, 0, 1, 16, 17])
        padded_convinds = torch.zeros(11, 15, 7, dtype=int)

        for y in range(1, 12):
            for x in range(1, 16):
                padded_hexind = y * 17 + x
                padded_offsets = padded_offsets0 if y % 2 == 0 else padded_offsets1
                padded_convinds[y-1, x-1] = padded_offsets + padded_hexind

        self.register_buffer("padded_convinds", padded_convinds.flatten())
        self.fc = nn.LazyLinear(out_features=out_channels)

    def forward(self, x):
        b, _, hexdim = x.shape
        x = x.view(b, 11, 15, -1)
        padded_x = x.new_zeros((b, 13, 17, hexdim))  # +2 hexes in X and Y coords
        padded_x[:, 1:12, 1:16, :] = x
        padded_x = padded_x.view(b, -1, hexdim)
        fc_input = padded_x[:, self.padded_convinds, :].view(b, 165, -1)
        return self.fc(fc_input)


class HexConvResLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.act = nn.LeakyReLU()
        self.body = nn.Sequential(
            HexConv(channels),
            self.act,
            HexConv(channels),
        )

    def forward(self, x):
        return self.act(self.body(x).add(x))


class HexConvResBlock(nn.Module):
    def __init__(self, channels, depth=1):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(depth):
            self.layers.append(HexConvResLayer(channels))

    def forward(self, x):
        assert x.is_contiguous
        return self.layers(x)
