import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import RandomRotation, RandomHorizontalFlip
from prettytable import PrettyTable

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Loss functions
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_nll = torch.nn.NLLLoss()
    
    # Saving parameters
    best_train_loss_ce = 1e9
    best_train_loss_nll = 1e9
    
    # Loss lists
    train_losses_ce = []
    test_losses_ce = []
    train_losses_nll = []
    test_losses_nll = []
    
    # Epoch Loop
    for epoch in range(1, 100):
        # Start timer
        t = time.time_ns()
        
        # Train the model
        model.train()
        train_loss_ce = 0
        train_loss_nll = 0
        
        # Batch Loop
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            
            # Apply random rotation and horizontal flip transforms
            rot_transform = RandomRotation(degrees=(-10, 10))
            flip_transform = RandomHorizontalFlip(p=0.5)
            images = flip_transform(rot_transform(images))
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute the losses
            loss_ce = criterion_ce(outputs, labels)
            loss_nll = criterion_nll(F.log_softmax(outputs, dim=1), labels)
            
            # Backward pass
            loss_ce.backward(retain_graph=True)
            loss_nll.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Accumulate the losses
            train_loss_ce = train_loss_ce + loss_ce.item()
            train_loss_nll = train_loss_nll + loss_nll.item()
        
        # Test the model
        model.eval()
        test_loss_ce = 0
        test_loss_nll = 0
        correct_ce = 0
        correct_nll = 0
        total = 0
        
        # Batch Loop
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute the losses
            loss_ce = criterion_ce(outputs, labels)
            loss_nll = criterion_nll(F.log_softmax(outputs, dim=1), labels)
            
            # Accumulate the losses
            test_loss_ce = test_loss_ce + loss_ce.item()
            test_loss_nll = test_loss_nll + loss_nll.item()
            
            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted_ce = torch.max(outputs, 1)
            _, predicted_nll = torch.max(F.log_softmax(outputs, dim=1), 1)
            
            total += labels.size(0)
            
            # Accumulate the number of correct classifications
            correct_ce += (predicted_ce == labels).sum().item()
            correct_nll += (predicted_nll == labels).sum().item()
        
        # Print the epoch statistics
        print(f'Epoch: {epoch}, '
              f'Train Loss CE: {train_loss_ce / len(trainloader):.4f}, '
              f'Test Loss CE: {test_loss_ce / len(testloader):.4f}, '
              f'Test Accuracy CE: {correct_ce / total:.4f}, '
              f'Train Loss NLL: {train_loss_nll / len(trainloader):.4f}, '
              f'Test Loss NLL: {test_loss_nll / len(testloader):.4f}, '
              f'Test Accuracy NLL: {correct_nll / total:.4f}, '
              f'Time: {(time.time_ns() - t) / 1e9:.2f}s')
        
        # Update loss lists
        train_losses_ce.append(train_loss_ce / len(trainloader))
        test_losses_ce.append(test_loss_ce / len(testloader))
        train_losses_nll.append(train_loss_nll / len(trainloader))
        test_losses_nll.append(test_loss_nll / len(testloader))
        
        # Update the best models
        if train_loss_ce < best_train_loss_ce:
            best_train_loss_ce = train_loss_ce
            torch.save(model.state_dict(), 'best_model_ce.pth')
        if train_loss_nll < best_train_loss_nll:
            best_train_loss_nll = train_loss_nll
            torch.save(model.state_dict(), 'best_model_nll.pth')
        
        # Save the model
        torch.save(model.state_dict(), 'current_model.pth')
        
        # Create the loss plots
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_ce, label='Train Loss CE')
        plt.plot(test_losses_ce, label='Test Loss CE')
        plt.plot(train_losses_nll, label='Train Loss NLL')
        plt.plot(test_losses_nll, label='Test Loss NLL')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('task5_loss_plot.png')
        
        # Tabulate the results
        table = PrettyTable()
        table.field_names = ['Loss Function', 'Train Loss', 'Test Loss', 'Test Accuracy']
        table.add_row(['CrossEntropyLoss', f'{train_losses_ce[-1]:.4f}', f'{test_losses_ce[-1]:.4f}', f'{correct_ce / total:.4f}'])
        table.add_row(['NLLLoss', f'{train_losses_nll[-1]:.4f}', f'{test_losses_nll[-1]:.4f}', f'{correct_nll / total:.4f}'])
        print(table)