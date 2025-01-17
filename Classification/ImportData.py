import torch
from torchvision import datasets
from torchvision.transforms import v2

def load_data(img_size,train_path,test_path,val_path):
    train_data_proc = datasets.ImageFolder(root=train_path, 
                                      transform=v2.Compose([
                                          v2.Resize((img_size, img_size)),
                                          v2.RandomVerticalFlip(),  
                                          v2.RandomHorizontalFlip(),  
                                          v2.RandomRotation(5),
                                          v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                          v2.ToTensor(),
                                          v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ]))

    train_data = torch.utils.data.DataLoader(train_data_proc, batch_size=32, shuffle=True)

    val_data_proc = datasets.ImageFolder(root=val_path, 
                                    transform=v2.Compose([
                                        v2.Resize((img_size, img_size)),
                                        v2.ToTensor(),
                                        v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                    ]))

    val_data = torch.utils.data.DataLoader(val_data_proc, batch_size=32, shuffle=False)

    test_data_proc = datasets.ImageFolder(root=test_path, 
                                     transform=v2.Compose([
                                         v2.Resize((img_size, img_size)),
                                         v2.ToTensor(),
                                         v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ]))

    test_data = torch.utils.data.DataLoader(test_data_proc, batch_size=32, shuffle=False)

    return train_data,test_data,val_data




