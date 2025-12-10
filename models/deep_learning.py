import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
import warnings

warnings.filterwarnings('ignore')

class TransformDataset(Dataset):
    '''
    Function to load images and labels from a .npz file and apply preprocessing transforms
    '''
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.X = data['X']
        self.y = data['y']
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = torch.tensor(self.X[idx], dtype=torch.float32)  # (h, w, c)
        img = img.permute(2, 0, 1)  # convert to (c, h, w)

        if self.transform:
            img = self.transform(img)

        return img, int(self.y[idx])

def train_one_epoch(model, loader, optimizer, criterion, device):
    '''
    Function to train the model for one epoch
    '''
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    '''
    Function to compute evaluation metrics
    '''
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return accuracy, precision, recall, f1

def build_model(num_classes, device):
    '''
    Function to create efficientnet model
    '''
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False) # Replace first conv layer to allow grayscale images
    model.classifier[1] = nn.Linear(1280, num_classes) # Replace classifier output layer for target classes
    return model.to(device)

def create_dataloaders(train_path, val_path, test_path):
    '''
    Create PyTorch dataloaders for train, validation, and test sets
    '''
    # Transforms to resize and normalize
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Transform data
    train_ds = TransformDataset(train_path, transform=train_tf)
    val_ds = TransformDataset(val_path, transform=test_tf)
    test_ds = TransformDataset(test_path, transform=test_tf)

    # Create dataloader
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device):
    '''
    Function to implement training loop
    '''
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val = 0
    wait = 5
    wait_count = 0
    num_epochs = 30

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy, precision, recall, f1 = evaluate(model, val_loader, device)

        print(f'Epoch {epoch+1}/{num_epochs} | Loss {loss:.4f} | Val Acc {accuracy:.4f}')

        scheduler.step()

        if accuracy > best_val: # Save model if improved
            best_val = accuracy
            wait_count = 0
            torch.save(model.state_dict(), 'best_effnet.pth')
        else:
            wait_count += 1

        if wait_count >= wait: # Early stopping
            print('Stopped early')
            break

def upload_to_hf():
    '''
    Upload the best model to huggingface
    '''
    load_dotenv()
    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    repo_id = 'moosejuice13/cnn_doodle_id_effnet'

    api = HfApi()
    api.create_repo(repo_id, repo_type='model', exist_ok=True)

    api.upload_file(
        path_or_fileobj='best_effnet.pth',
        path_in_repo='cnn_doodle_id_effnet.pth',
        repo_id=repo_id,
    )
    
    print(f'Model pushed to https://huggingface.co/{repo_id}')

def main():
    train_path = './data/processed/train.npz'
    val_path = './data/processed/val.npz'
    test_path = './data/processed/test.npz'

    num_classes = len(np.load(train_path)['class_names'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = create_dataloaders(train_path, val_path, test_path) # Load dataloaders
    model = build_model(num_classes, device) # Create model
    train_model(model, train_loader, val_loader, device) # Train model

    model.load_state_dict(torch.load('best_effnet.pth')) # Load best checkpoint
    model.to(device)

    accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
    print('Accuracy:', round(accuracy, 4))
    print('Precision:', round(precision, 4))
    print('Recall:', round(recall, 4))
    print('F1:', round(f1, 4))

    upload_to_hf()

if __name__ == '__main__':
    main()