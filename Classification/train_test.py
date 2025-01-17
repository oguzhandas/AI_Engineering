#### Initializing and Training ####
import torch
from ImportData import load_data
from model import prunedmodel
from sklearn.metrics import classification_report

## Set Data Path and Image Dimensions to be Resized
train_path = 'Your path to training images'
test_path = 'Your path to testing images'
val_path = 'Your path to validation images'
img_size=128
 


train_data, test_data, val_data=load_data(img_size,train_path,test_path,val_path)
model,device,loss,optimizer=prunedmodel(num_classes=6) #Six defect types exist in the dataset

epoch=30 # Set Epoch


## Training Loop
for epoch in range(epoch):  
    model.train()
    #Training Parameters Initialization
    r_loss = 0.0
    corr = 0
    tot = 0
    for images, labels in train_data:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        loss_model = loss(outputs, labels)
        loss_model.backward()
        optimizer.step()
        r_loss += loss_model.item()
        tot += labels.size(0)
        corr += (predicted == labels).sum().item()
    train_loss=r_loss/len(train_data) 
    train_acc=100 * corr / tot
    print(f"Epoch [{epoch+1}]")
    print(f"Training Accuracy: {train_acc:.2f}%, Training Loss: {train_loss:.4f}")
    
    ## Validation Stage
    model.eval()
    with torch.no_grad():
        corr = 0
        tot = 0
        r_loss = 0.0
        for images, labels in val_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_model = loss(outputs, labels)
            r_loss += loss_model.item()
            _, predicted = torch.max(outputs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()
    val_loss = r_loss / len(val_data)
    val_acc = 100 * corr / tot
    print(f"Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}")
## Testing Stage
def evaluate_model(model, test_loader):
    model.eval() 
    #Testing Parameters Initialization
    corr = 0
    tot = 0
    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()

    test_acc = 100 * corr / tot
    print(f'Test Accuracy: {test_acc:.2f}%')
    return test_acc
test_accuracy = evaluate_model(model, test_data)

## Testing Performance Metrics
def performance_metrics(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=test_data.dataset.classes))

performance_metrics(model, test_data)    