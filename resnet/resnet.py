import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
import pathlib
import matplotlib.pyplot as plt

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define options
class Options(object):
    def __init__(self, **kwargs):
        super(Options, self).__init__()
        self.__dict__.update(kwargs)

resnet_config = Options(
    # the original paper uses SGD; we use Adam instead
    # training HPs
    num_epochs=10,
    minibatch_size=256,
    default_learning_rate=0.05,
    results_dir=pathlib.Path('results'),
)

class ResNetBlock(nn.Module):
    '''
    Building block in the spirit of original ResNet paper (Fig. 5 left), 
    i.e., every layer has the same number of output channels.
    '''
    def __init__(self, num_in_channel, num_out_channel, stride=1):
        super(ResNetBlock, self).__init__()
        
        # in the original paper, batch normalization is applied
        # after convolution and before activation
        self.conv1 = nn.Conv2d(num_in_channel, num_out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_out_channel)
        
        self.conv2 = nn.Conv2d(num_out_channel, num_out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_out_channel)
        
        # identifhy needs a shortcut connection while matching dimensions
        self.identity_layer = nn.Sequential()
        # use a 1x1 convolution to match dimensions
        if num_in_channel != num_out_channel or stride != 1:
            self.identity_layer = nn.Sequential(
                nn.Conv2d(num_in_channel, num_out_channel, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_out_channel)
            )
        self.relu = nn.ReLU() # no training parameter 
    
    def forward(self, x):
        identify = self.identity_layer(x)
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))
        x += identify  # add the identity connection
        x = self.relu(x)
        return x
        
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        '''
        Fashion-MNIST has 10 classes labelled 0-9
        input images are of size 28x28x1 (grayscale)
        '''
        super(ResNet, self).__init__()
        
        # layers
        self.layers = []
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        self.res_block1 = ResNetBlock(16, 
                                      32, stride=1)
        self.res_block2 = ResNetBlock(32, 
                                      64, stride=2)
        self.res_block3 = ResNetBlock(64, 
                                      64, stride=2
                                      )
        # classifier
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)  # softmax for multi-class classification
        )
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.average_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
def create_data_loader(batch_size=256, num_workers=2):
    '''
    data loader for Fashion-MNIST dataset
    '''
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Train dataset size: {len(train_dataset)}, "
          f"Test dataset size: {len(test_dataset)}")
    return train_loader, test_loader

def evaluate_accuracy(model, data_loader):
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():   
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
    accuracy = 100 * correct_test / total_test
    return accuracy

def train_restnet_with_lr(model, train_loader, test_loader,
                          learning_rate=0.01):
    loss_fc = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), 
                     lr=learning_rate)
    
    
    # train loop
    for epoch in range(resnet_config.num_epochs):
        model.train()
        # running loss and train accuracy
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fc(output, target)
            loss.backward()
            optimizer.step()
            
            # logging
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = 100 * correct_train / total_train
                
        # epoch test
        epoch_test_accuracy = evaluate_accuracy(model, test_loader)
        print(f'Epoch: {epoch+1}/{resnet_config.num_epochs}, '
              f'Loss: {epoch_loss:.4f}, '
              f'Train Accuracy: {epoch_train_accuracy:.2f}%, '
              f'Test Accuracy: {epoch_test_accuracy:.2f}%')
        
    # final test accuracy
    final_test_accuracy = evaluate_accuracy(model, test_loader)
    return final_test_accuracy

def visualize_classification_results(model, data_loader):
    # get some test images
    images, class_label = next(iter(data_loader))
    images, class_label = images.to(device), class_label.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # visualize the results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f'True: {class_label[i].item()}, Pred: {predicted[i].item()}')
        # color the title for visualization
        if class_label[i] == predicted[i]:
            ax.title.set_color('green')
        else:
            ax.title.set_color('red')
        ax.axis('off')
    plt.tight_layout()

    # save the figure for sanity check
    fig_path = resnet_config.results_dir / 'classification_results.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)

def main():
    train_loader, test_loader = create_data_loader(batch_size=resnet_config.minibatch_size)
    model = ResNet().to(device)
    
    accuracy = train_restnet_with_lr(model, train_loader, test_loader,
                          learning_rate=resnet_config.default_learning_rate)
    print(f'Final Test Accuracy: {accuracy:.2f}%')
    
    # visualize the training results
    visualize_classification_results(model, test_loader)
    
    # save the model
    torch.save(model.state_dict(), 'resnet_final.pth')

if __name__ == '__main__':
    main()