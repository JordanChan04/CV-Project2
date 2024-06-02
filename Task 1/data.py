from torch.utils.data import DataLoader, sampler, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
    all_data = datasets.ImageFolder(data_dir, transform=transform)
    train_data_len = int(len(all_data)*0.75)
    valid_data_len = int((len(all_data) - train_data_len)/2)
    test_data_len = int(len(all_data) - train_data_len - valid_data_len)
    train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return ((train_loader, val_loader, test_loader),train_data, val_data, test_data, all_data.classes)