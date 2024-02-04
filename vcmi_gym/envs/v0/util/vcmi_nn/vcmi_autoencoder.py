import torch
import torch.optim as optim
import glob
import sys
import numpy as np
from datetime import datetime
from collections import deque
from statistics import mean

TRAIN_DECODER_ONLY = False

BATCH_SIZE = 32

# *** non-deterministic ***
MODEL_LOAD_FILE = None
# RNG = np.random.default_rng()

# *** deterministic ***
# MODEL_LOAD_FILE = "data/autoencoder/untrained-l3-bn-a-do-f384-model.pt"
RNG = np.random.default_rng(seed=42)  # make deterministic

ENCODER_PARAMETERS_LOAD_FILE = None
# ENCODER_PARAMETERS_LOAD_FILE = "autoencoder-pretrained-encoder-params.pth"

DATASET_FILEMASK = "data/observations/*.npy"
# DATASET_FILEMASK = "data/observations/1706*.npy"


# Define the autoencoder architecture
class VcmiAutoencoder(torch.nn.Module):
    def __init__(self):
        super(VcmiAutoencoder, self).__init__()

        # Input dim:
        # (B, Z, Y, X) = (B, 1, 11, 225)
        # B=batch size
        # Z=n_channels

        self.encoder = torch.nn.Sequential(

            # <3>
            # => (B, 1, 11, 225)
            torch.nn.Conv2d(1, 32, kernel_size=(1, 15), stride=(1, 15)),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.25, inplace=True),
            # => (B, 32, 11, 15)
            torch.nn.Flatten(),  # by default this flattens dims 1..-1 (ie. keeps B)
            # => (B, 5280)
            # </3>

            # # <1>
            # # => (B, 1, 11, 225)
            # torch.nn.Conv2d(1, 32, kernel_size=(1, 15), stride=(1, 15)),
            # torch.nn.BatchNorm2d(num_features=32),
            # torch.nn.LeakyReLU(),
            # # torch.nn.Dropout2d(p=0.25, inplace=True),
            # # # => (B, 32, 11, 15)
            # torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            # torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.LeakyReLU(),
            # # torch.nn.Dropout2d(p=0.25, inplace=True),
            # # => (B, 64, 5, 7)
            # torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            # torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.LeakyReLU(),
            # # torch.nn.Dropout2d(p=0.25, inplace=True),
            # # => (B, 64, 2, 3)
            # torch.nn.Flatten(),  # by default this flattens dims 1..-1 (ie. keeps B)
            # # => (B, 384)
            # # torch.nn.Linear(384, 1024),  # 5280 = 32*15*11
            # # torch.nn.LeakyReLU()
            # # # => (B, 1024)
            # # </1>
        )

        self.id = "l1-bn-a-do-f5280"

        self.decoder = torch.nn.Sequential(
            # <3>
            # => (B, 5280)
            torch.nn.Unflatten(1, (32, 11, 15)),
            # => (B, 32, 11, 15)
            torch.nn.ConvTranspose2d(32, 1, kernel_size=(1, 15), stride=(1, 15)),
            # => (B, 1, 11, 225)
            # </3>

            # # <1>
            # # => (B, 384)
            # torch.nn.Unflatten(1, (64, 2, 3)),
            # # => (B, 64, 3, 2)
            # torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),
            # torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.LeakyReLU(),
            # # torch.nn.Dropout2d(p=0.25, inplace=True),
            # # => (B, 64, 5, 7)
            # torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            # torch.nn.BatchNorm2d(num_features=32),
            # torch.nn.LeakyReLU(),
            # # torch.nn.Dropout2d(p=0.25, inplace=True),
            # # => (B, 64, 11, 15)
            # torch.nn.ConvTranspose2d(32, 1, kernel_size=(1, 15), stride=(1, 15)),
            # # torch.nn.BatchNorm2d(num_features=1),
            # # torch.nn.LeakyReLU(),
            # # => (B, 1, 11, 225)
            # # </1>

            torch.nn.Sigmoid()
        )

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return d


class Dataset:
    def __init__(self, files, batch_size):
        self.it_files = iter(files)
        self.it_curfile = iter([])
        self.batch_size = int(batch_size)
        self.stopped = False
        self.returned = False
        self.batch = []

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while len(self.batch) < self.batch_size:
                # print(".", end="", flush=True)
                self.batch.append(next(self.it_curfile))
                self.stopped = False
            res = torch.as_tensor(np.array(self.batch))
            self.batch = []
            # print("returning batch: %s" % str(res.shape))
            return res
        except StopIteration:
            if self.stopped:
                raise

            try:
                f = next(self.it_files)
                self.it_curfile = iter(np.vstack(np.load(f)))
                # print("New it_curfile: %s" % f)
                return self.__next__()
            except StopIteration:
                self.stopped = True
                res = torch.as_tensor(np.array(self.batch))
                self.batch = []
                # print("Last return: %s" % str(res.shape))
                return res


class DataProvider:
    def __init__(self, train_test_ratio):
        assert isinstance(train_test_ratio, float)
        assert train_test_ratio > 0 and train_test_ratio < 1

        files = glob.glob(DATASET_FILEMASK)
        files.sort()
        n_train_files = int(train_test_ratio * len(files))

        indexes = RNG.choice(np.arange(0, len(files)), size=len(files), replace=False)

        self.train_files = [files[i] for i in indexes[:n_train_files]]
        self.test_files = [files[i] for i in indexes[n_train_files:]]

        assert len(self.train_files) + len(self.test_files) == len(files)

    def train_data(self, batch_size):
        return Dataset(self.train_files, batch_size=batch_size)

    def test_data(self, batch_size):
        return Dataset(self.test_files, batch_size=batch_size)


# Initialize the VcmiAutoencoder

if MODEL_LOAD_FILE:
    assert not ENCODER_PARAMETERS_LOAD_FILE
    model = torch.load(MODEL_LOAD_FILE)
    print("loading model from %s..." % MODEL_LOAD_FILE)
else:
    model = VcmiAutoencoder()

mse = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
dp = DataProvider(0.99)

# Load pretrained features extractor as the encoder and updates for it
if ENCODER_PARAMETERS_LOAD_FILE:
    print("loading encoder params from %s..." % ENCODER_PARAMETERS_LOAD_FILE)
    model.encoder.load_state_dict(torch.load(ENCODER_PARAMETERS_LOAD_FILE))

if TRAIN_DECODER_ONLY:
    for param in model.encoder.parameters():
        param.requires_grad = False


def test():
    step = 0
    loss_sum = 0

    with torch.no_grad():
        for test_batch in dp.test_data(1000):
            recon = model(test_batch)
            loss_sum += mse(recon, test_batch).item()
            step += 1
            # if step % 100 == 0:
            print("\r[Test] [%d] Loss: %.4f %5s" % (step, loss_sum / step, ""), end="", flush=True)

    print("")  # newline
    return step


def train():
    losses = deque(maxlen=100)
    step = 0

    for train_batch in dp.train_data(batch_size=BATCH_SIZE):
        recon = model(train_batch)
        loss = mse(recon, train_batch)
        optimizer.zero_grad()
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


ts = datetime.now().strftime("%Y%m%d_%H%M%S")
base = "data/autoencoder/%s-%s" % (ts, model.id)

with open("%s.txt" % base, "w") as dest:
    print("Run: %s" % dest.name)
    dest.write("Batch size: %d\n\n%s\n\n%s\n" % (BATCH_SIZE, model.encoder, model.decoder))

train_epochs = 1
step = 0
save = True
test()

try:
    for epoch in range(train_epochs):
        print("[Train] Step: %d" % step)
        print("[Train] Epoch %d/%d:" % (epoch + 1, train_epochs))
        step += train()
        test()
finally:
    dest = "%s-model.pt" % base
    torch.save(model, dest)
    print("Saved %s" % dest)

    dest = "%s-params-encoder.pth" % base
    torch.save(model.encoder.state_dict(), dest)
    print("Saved %s" % dest)

    print("\nFinished.")
