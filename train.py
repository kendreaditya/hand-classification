# %%
import torch as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
# %%
# Preprocess Data
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder("./dataset/", preprocess) 
print(dataset)

# %%
# DataLoader
trainloader = nn.torch.utils.data.DataLoader(dataset, batch_size=5,
                                          shuffle=True, num_workers=2)

# %%
# Get Model
model = nn.torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# %%
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# %%
# Training
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')