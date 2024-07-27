# %%
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn

import argparse

import torch.optim as optim
from vit import ViT
from primary_net import PrimaryNetwork
from cnn import swrn


# seed
torch.manual_seed(0)

########### Data Loader ###############

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4
)

testset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=4
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

#############################

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--net", default="hypernet_vit", type=str, help="network to train (primary/vit)"
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--n_epochs", default=2000, type=int, help="number of epochs to train"
)
args = parser.parse_args()

print(args)

############
best_accuracy = 0.0

if args.net == "primary":
    net = PrimaryNetwork()
    if args.resume:
        ckpt = torch.load("./hypernetworks_cifar_paper.pth")
        net.load_state_dict(ckpt["net"])
        best_accuracy = ckpt["acc"]
elif args.net == "vit":
    net = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
    )
elif args.net == "hypernet_vit":
    depth = 48
    net = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=depth,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        # hypernet_hidden_dim=512,
    )
    print("depth: ", depth)
elif args.net == "cnn":
    net = swrn(28, 10, 6)

net.cuda()

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
M = 1024**2
print(f"Total Trainable Params: {total_params / M:.2f}M")

learning_rate = 1e-4
weight_decay = 0.0005
milestones = [168000, 336000, 400000, 450000, 550000, 600000]
max_iter = 1000000

optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#    optimizer=optimizer, milestones=milestones, gamma=0.5
# )
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

criterion = nn.CrossEntropyLoss()

total_iter = 0
epochs = 0
print_freq = 50
# while total_iter < max_iter:
while epochs < args.n_epochs:

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        running_loss += loss.item()
        if i % print_freq == (print_freq - 1):
            print(
                "[Epoch %d, Total Iterations %6d] Loss: %.4f"
                % (epochs + 1, total_iter + 1, running_loss / print_freq)
            )
            running_loss = 0.0

        total_iter += 1

    epochs += 1

    correct = 0.0
    total = 0.0
    for tdata in testloader:
        timages, tlabels = tdata
        toutputs = net(Variable(timages.cuda()))
        _, predicted = torch.max(toutputs.cpu().data, 1)
        total += tlabels.size(0)
        correct += (predicted == tlabels).sum()

    accuracy = (100.0 * correct) / total
    print("After epoch %d, accuracy: %.4f %%" % (epochs, accuracy))

    if accuracy > best_accuracy:
        print("Saving model...")
        state = {"net": net.state_dict(), "acc": accuracy}
        torch.save(state, "./hypernetworks.pth")
        best_accuracy = accuracy

print("Finished Training")

# %%
