import torch
from model import ResNet, ResNetConfig, BasicBlock
from data import get_data_loaders
from utils import evaluate

model_path = "./best-model.pt"
data_dir = "CUB_200_2011/images/"
batch_size = 64
output_dim = 200

(_, _, test_loader), _, _, _, _ = get_data_loaders(data_dir, batch_size)

model = ResNet(ResNetConfig(BasicBlock, [2,2,2,2], [64, 128, 256, 512]), output_dim)
model.load_state_dict(torch.load(model_path))

loss, accuracy = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device='cuda')
print(f'Test Loss: {loss:.3f} | Test Acc: {accuracy*100:6.2f}%')