import torch
import torch.optim as optim
import random
import time
import glob
import numpy as np
from collections import deque
from statistics import mean


DETERMINISTIC = True
TRAIN_DECODER_ONLY = True
ENCODER_PARAMETERS_LOAD_FILE = None
# ENCODER_PARAMETERS_LOAD_FILE = "autoencoder-pretrained-encoder-params.pth"


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Define the autoencoder architecture
class VcmiAutoencoder(torch.nn.Module):
    def __init__(self):
        super(VcmiAutoencoder, self).__init__()

        # Input dim:
        # (B, Z, Y, X) = (B, 1, 11, 225)
        # B=batch size
        # Z=n_channels

        self.encoder = torch.nn.Sequential(
            # => (B, 1, 11, 225)
            torch.nn.Conv2d(1, 32, kernel_size=(1, 15), stride=(1, 15)),
            # => (B, 32, 11, 15)
            torch.nn.ReLU(),
            torch.nn.Flatten(),  # by default this flattens dims 1..-1 (ie. keeps B)
            # => (B, 5280)
            torch.nn.Linear(5280, 512),  # 5280 = 32*15*11
            # => (B, 512)
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            # => (B, 512)
            torch.nn.Linear(512, 5280),
            # => (B, 5280)
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (32, 11, 15)),
            # => (B, 32, 11, 15)
            torch.nn.ConvTranspose2d(32, 1, kernel_size=(1, 15), stride=(1, 15)),
            # => (B, 1, 11, 225)
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return d


class Dataset:
    def __init__(self, files):
        self.it_files = iter(files)

    def __iter__(self):
        return self

    def __next__(self):
        return torch.as_tensor(np.load(next(self.it_files)))


class DataProvider:
    def __init__(self, train_test_ratio):
        assert isinstance(train_test_ratio, float)
        assert train_test_ratio > 0 and train_test_ratio < 1

        files = glob.glob("data/observations/*.npy")
        if not DETERMINISTIC:
            random.shuffle(files)
        n_train_files = int(train_test_ratio * len(files))
        self.train_files = files[:n_train_files]
        self.test_files = files[n_train_files:]

    def train_data(self):
        return Dataset(self.train_files)

    def test_data(self):
        return Dataset(self.test_files)


# Initialize the VcmiAutoencoder
model = VcmiAutoencoder()

# if DETERMINISTIC:
#     model.load_state_dict(torch.load("autoencoder-untrained.pth"))

mse = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dp = DataProvider(0.95)

# Load pretrained features extractor as the encoder and updates for it
if ENCODER_PARAMETERS_LOAD_FILE:
    print("loading from...")
    model.encoder.load_state_dict(torch.load(ENCODER_PARAMETERS_LOAD_FILE))

if TRAIN_DECODER_ONLY:
    for param in model.encoder.parameters():
        param.requires_grad = False


def test():
    step = 0
    loss_sum = 0

    with torch.no_grad():
        for batch in dp.test_data():
            # d = batch.flatten(0, 1)
            for data in batch:
                d = data
                # for data0 in data:
                #     d = data0.unsqueeze(0)
                recon = model(d)
                loss_sum += mse(recon, d).item()
                step += 1
                # if step % 100 == 0:
                print("\r[Test] [%d] Loss: %.4f %5s" % (step, loss_sum / step, ""), end="", flush=True)

    print("")  # newline
    return step


def train():
    losses = deque(maxlen=100)
    step = 0

    for batch in dp.train_data():
        # d = batch.flatten(0, 1)
        for data in batch:
            d = data
            # for data0 in data:
            #     d = data0.unsqueeze(0)
            optimizer.zero_grad()
            recon = model(d)
            loss = mse(recon, d)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print("\r[Train] [%d] Loss: %.4f %5s" % (step, mean(losses), ""), end="", flush=True)

    print("")  # newline
    return step

    # Save the model
    # torch.save(model.state_dict(), 'conv_autoencoder.pth')

#
# feature_dims=512
#
# With default file data (batch_size=24)
#   [Test] Loss: 0.1897
#   [Train] Loss: 0.0063
#   [Test] Loss: 0.0045
#   16354ms
# n_epochs=10
#   [Test] Loss: 0.1897
#   [Train] Loss: 0.0012
#   [Test] Loss: 0.0015
#   125235ms
#
# With file flattened as 1 batch (batch_size=85*24):
#   [Test] Loss: 0.1891
#   [Train] Loss: 0.1348
#   [Test] Loss: 0.0690
#   5003ms
# n_epochs=30
#   [Test] Loss: 0.1891
#   [Train] Loss: 0.0182
#   [Test] Loss: 0.0159
#   100231ms
#
# With unflattened data (batch_size=1)
# [Test] Loss: 0.1891
# [Train] Loss: 0.0026
# [Test] Loss: 0.0033
# 169069ms
#
# output_dim=1024 Loss: 0.0003
# output_dim=512 Loss: 0.0011
# output_dim=128 Loss: 0.0023
# output_dim=32  Loss: 0.0080
#
# .....................................................
# ...........................
# training DECODER ONLY with randomly initialized encoder.........
# output+dim=512 [Train] [38500] Loss: 0.0014
# ... ie. decoder adjusts eventually, regardless if encoder is random or not
# ..... loading a pre-trained encoder simply leads faster to the same result
# ie. [Train] [26200] Loss: 0.0008  (was below 1e-4 at ~10k)


train_epochs = 10
step = 0
test()
for epoch in range(train_epochs):
    print("Steps: %d" % step)
    print("Epoch %d/%d:" % (epoch + 1, train_epochs))
    step += train()
    test()


if input("\nSave model? [y/n]: ") == "y":
    t = time.time()
    dest = "autoencoder-%d-model.pt" % t
    torch.save(model, dest)
    print("Saved %s" % dest)

    dest = "autoencoder-%d-encoder-params.pth" % t
    torch.save(model.encoder.state_dict(), dest)
    print("Saved %s" % dest)

print("\nFinished.")
