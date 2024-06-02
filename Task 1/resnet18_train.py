import torch
import torch.optim as optim
from model import ResNet, ResNetConfig, BasicBlock
from data import get_data_loaders
from utils import count_parameters, LRFinder, plot_lr_finder

data_dir = "CUB_200_2011/images/"
batch_size = 64
pretrained = True
output_dim = 200
epochs = 20

(train_loader, val_loader, _), _, _, _, _ = get_data_loaders(data_dir, batch_size)

model = ResNet(ResNetConfig(BasicBlock, [2,2,2,2], [64, 128, 256, 512]), output_dim)

if pretrained:
    pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.load_state_dict(pretrained_model.state_dict())

print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters(), lr=1e-7)
end_lr = 10
num_iter = 100
lrf = LRFinder(model, optimizer, torch.nn.CrossEntropyLoss(), device='cuda')
lrs, losses = lrf.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)

print("Finding best learning rate...")
plot_lr_finder(lrs, losses)