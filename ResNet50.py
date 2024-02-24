import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt


# cat_folder = '/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/Hw2_N16127024_洪途慰_1/inference_dataset/Cat'
# dog_folder = '/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/Hw2_N16127024_洪途慰_1/inference_dataset/Dog'

# cat_files = os.listdir(cat_folder)
# dog_files = os.listdir(dog_folder)

# cat_files = [file for file in cat_files if file.endswith((".jpg", ".png", ".jpeg"))]
# dog_files = [file for file in dog_files if file.endswith((".jpg", ".png", ".jpeg"))]

# random_cat = random.choice(cat_files)
# random_dog = random.choice(dog_files)

# cat = plt.imread(os.path.join(cat_folder, random_cat))
# dog = plt.imread(os.path.join(dog_folder, random_dog))


# fig, ax = plt.subplots(1,2,figsize = (5,5))
# ax = ax.flatten()
# ax[0].imshow(cat)
# ax[0].set_axis_off()
# ax[0].set_title('Cat', fontsize=12, y=1.05, ha="center")
# ax[1].imshow(dog)
# ax[1].set_axis_off()
# ax[1].set_title("Dog", fontsize=12, y=1.05, ha="center")

# plt.tight_layout()
# plt.show()

batchsize = 32
learning_rate = 0.001
epochs = 50

traindir = './Dataset_Cvdl_Hw2_Q5/training_dataset'
valdir = './Dataset_Cvdl_Hw2_Q5/validation_dataset'

# traindir = '/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/Datasets/Dataset_CvDl_Hw2/Dataset_Cvdl_Hw2_Q5/training_dataset'
# valdir = '/Users/davidhernandez/Desktop/NCKU/Computer_Vision/HW2/Datasets/Dataset_CvDl_Hw2/Dataset_Cvdl_Hw2_Q5/validation_dataset'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],),
])

train_transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],),
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],),
])

train = datasets.ImageFolder(traindir, transform=train_transform, target_transform=lambda x: 1 if x == 0 else 0)
train2 = datasets.ImageFolder(traindir, transform=train_transform2, target_transform=lambda x: 1 if x == 0 else 0)
valid = datasets.ImageFolder(valdir, transform=test_transform, target_transform=lambda x: 1 if x == 0 else 0)

trainloader = DataLoader(train, batch_size=batchsize, shuffle=True)
trainloader2 = DataLoader(train2, batch_size=batchsize, shuffle=True)
validloader = DataLoader(valid, batch_size=batchsize, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights=None)
#add a new final layer
model.fc = nn.Sequential(
  nn.Linear(2048, 1),
  nn.Sigmoid()
)
model = model.to(device)

#loss
criterion = nn.BCELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(net, trainloader, validloader, criterion, optimizer, num_epochs, plot):
    net.train()

    train_losses = []  # To store training losses
    valid_losses = []  # To store validation losses
    train_accuracies = []  # To store training accuracies
    valid_accuracies = []  # To store validation accuracies

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in trainloader:
          inputs, labels = inputs.to(device), labels.to(device)
          labels = labels.float().unsqueeze(1)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

          predicted = (outputs > 0.5).int()
          total_train += labels.size(0)
          correct_train += (predicted == labels).sum().item()

        average_train_loss = running_loss / len(trainloader)
        train_losses.append(average_train_loss)
        training_accuracy = 100 * correct_train / total_train
        train_accuracies.append(training_accuracy)

        # Validation loop
        net.eval()  # Set the network to evaluation mode
        correct_valid = 0
        total_valid = 0
        valid_loss = 0.0

        with torch.no_grad():
          for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

            predicted = (outputs > 0.5).int()
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

        average_val_loss = valid_loss / len(validloader)
        valid_losses.append(average_val_loss)
        validation_accuracy = 100 * correct_valid / total_valid
        valid_accuracies.append(validation_accuracy)

        print(f'Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {training_accuracy:.2f}%, Validation Loss: {valid_losses[-1]:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')

    print('Finished Training')

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(f'plot{plot}')

    torch.save(model.state_dict(), f'Cv_Dl_ResNet_v{plot}.pth')

train(model, trainloader, validloader, criterion, optimizer, epochs, 1)


train(model, trainloader2, validloader, criterion, optimizer, epochs, 2)