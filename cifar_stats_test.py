
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torchinfo import summary
from linear_st import LinearST
import numpy
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

FORCE_CPU = True
LOAD_MODEL = True
TRAIN = False
EXAMINE_QUANTILES = True
LOAD_QUANTILES_FROM_FILE = True

MODEL_NAME = "cifar10_trained"

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.convs = nn.Sequential(
            # 32 x 32
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 15 x 15
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 6 x 6
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 2 x 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten = nn.Flatten()

        self.linear1 = LinearST(in_features=128, out_features=300, sim_queue_length=10, quantile_count=10)

        self.linear2 = LinearST(in_features=300, out_features=10, sim_queue_length=10, quantile_count=10)

        self.threshold = nn.Sigmoid()

    def forward(self, x):

        x = self.convs(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.threshold(x)
        x = self.linear2(x)
        x = self.threshold(x)
        return x


def main():

    device = "cpu"
    if not FORCE_CPU and torch.cuda.is_available():
        device = "cuda"

    # Create the data loader
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False)

    classe_names = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Create the model
    model = Classifier()

    # summary(model, input_size=(2, 3, 32, 32))

    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_NAME, map_location=device))

    if TRAIN:
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

        for epoch in range(4):

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            torch.save(model.state_dict(), MODEL_NAME)

            # Also save the quantiles for analysis
            numpy.save("linear1_quantiles.npy", model.linear1.quantiles.detach().numpy())

    if EXAMINE_QUANTILES:
        # Examine the quantiles of the linear layers
        quantiles = model.linear1.quantiles

        if LOAD_QUANTILES_FROM_FILE:
            quantiles = numpy.load("linear1_quantiles.npy").T

            density_estimates = numpy.ones(quantiles[:-1, :].shape) / (quantiles[1:, :] - quantiles[:-1, :])
            density_estimates = density_estimates / numpy.sum(density_estimates, axis=0, keepdims=True)

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
            axes[0].imshow(quantiles)
            axes[1].imshow(density_estimates)
            plt.savefig("plotted_density_estimates.png")
            plt.show()

    print("Done")

if __name__ == "__main__":
    main()