import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from utils import get_train_input_args
from workspace_utils import active_session


def main():
    """
    Example Execution Command 
    
        python train.py flowers --save_dir checkpoints --gpu --arch vgg19 --epochs 2
    """
    args = get_train_input_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    image_data_sets = get_data_sets(args.datafile)
    data_loaders = get_data_loaders(image_data_sets)
    
    model,optimizer = get_model(args.arch,args.learning_rate)
    model = train_model(model,optimizer,data_loaders,device,args.epochs)
    
    test_result = test_model(model,data_loaders,device)
    class_to_idx =  image_data_sets['train'].class_to_idx
    if 0.7 <= test_result:
        create_checkpoint(model,args.save_dir,args.arch,class_to_idx,optimizer,args.epochs)
    else:
        raise Exception("Model(%s) Accuracy(%s) is under 0.7 and not good enough to make prediction. Please train it again!!!!"%(args.arc, test_result))

def get_data_sets(data_dir):
    """
   
    Retrieves the image data set under the given data directory. 
    It is expected to have 3 different data set directories named as train,valid and test
    
    Parameters:
        data_dir: The (full) path to the folder of images that are to be
                   classified 
    
    Returns:
        image_dataset : Dataset ImageFolder
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test' : test_dir}
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])])}
    
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    return image_datasets
        
def get_data_loaders(image_datasets):
    """
    Retrieves the DataLoader object corresponding image dataset
    Parameters:
        image_datasets: ImageDataSet Dictionary for different stages such as train valid test
    
    Returns:
        data_loaders: dictionary of torch.utils.data.dataloader.DataLoader object
    """
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    
    return data_loaders

def get_model(model_name = 'vgg19',learning_rate=0.001):
    """
    Creates model according to model name, 
    Because of the need of specific output layer a custom made classifers or fc's created with models
    Parameters:
        model_name: could be vgg19, densenet121 and resnet101
        
        learning_rate: learning rate for optimizer
    
    Returns:
        model: created pre trained model
        
        optimizer: optimizer
        
    """
    t_models = {
              'vgg19':models.vgg19(pretrained=True),
              'densenet121': models.densenet121(pretrained=True),
              'resnet101':models.resnet101(pretrained=True)}
    
    model = t_models.get(model_name,'vgg19')
    classifier = None 
    optimizer = None
    if model_name == 'vgg19':
        classifier = nn.Sequential(nn.Linear(25088,4096),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(4096,102),
                          nn.LogSoftmax(dim=1))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
        
    elif model_name == 'densenet121':
        classifier = nn.Sequential(nn.Linear(1024,1000),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(1000,102),
                          nn.LogSoftmax(dim=1))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
       
    elif model_name == 'resnet101':
        classifier = nn.Sequential(nn.Linear(2048,1000),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(1000,102),
                          nn.LogSoftmax(dim=1))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(),lr=learning_rate)
   
   
    
    return model,optimizer

def train_model(model,optimizer,data_loader,device,epochs):
    """
    Trains the given pre-trained model the accuracy should be at least 0.7 or more
    Parameters:
        model: pre-trained model
        
        optimizer: optimizer
        
        data_loader: dataloader dictionary which has train and validation data
        
        device: device indicates the running mode it could be cuda or cpu
        
        epochs: one cycle through the full training dataset default is 2
        
    Returns:
        model: returns the trained model
        
    """
   
    print("Model training has been started")
    model.to(device)
    criterion = nn.NLLLoss()
    
    steps = 0
    train_loss=0
    print_interval =5
    len_validation = len(data_loader['valid'])
      
    with active_session():  
        for epoch in range(epochs):
            for inputs,labels in data_loader['train']:
                steps += 1
                inputs,labels = inputs.to(device),labels.to(device)
        
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps,labels)
                loss.backward()
                optimizer.step()
        
                train_loss += loss.item()
        
                if steps % print_interval == 0:
                    valid_loss = 0
                    accuracy =0
                    model.eval()
                    
                    with torch.no_grad():
                        for inputs,labels in data_loader['valid']:
                    
                            inputs, labels = inputs.to(device), labels.to(device)
                            log_valid_ps = model.forward(inputs)
                            batch_loss = criterion(log_valid_ps,labels)
                    
                            valid_loss += batch_loss.item()
                    
                            ps = torch.exp(log_valid_ps)
                            top_p,top_class = ps.topk(1,dim=1)
                    
                            equals = top_class == labels.view(*top_class.shape)
                            if device == 'cuda':
                                accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
                            else:
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Steps: {steps} Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {train_loss/print_interval:.3f}.. "
                          f"Validation loss: {valid_loss/len_validation:.3f}.. "
                          f"Validation accuracy: {accuracy/len_validation:.3f}")
            
                    train_loss = 0
                    model.train()     
    
    
    return model;

def test_model(model,data_loader,device):
    """
    Testing the trained model with given test data set
    Parameters:
        model: trained model
        
        data_loader: data loader dictionary which contains the test image dataset
        
        device: running mode it could be cuda or cpu
        
    Returns:
        
        accuracy: the accuracy of test data set prediction
    
    """
    
    print("Model test has been started")
    model.to(device)
    model.eval()
    accuracy=0
    with torch.no_grad():
        for inputs,labels in data_loader['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            log_test_ps = model.forward(inputs)
                    
            ps = torch.exp(log_test_ps)
            top_p,top_class = ps.topk(1,dim=1)
                    
            equals = top_class == labels.view(*top_class.shape)
            if device == 'cuda':
                accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
            else:
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_accuracy = accuracy/float(len(data_loader['test']))
    print(f"Test accuracy: {test_accuracy:.3f}")   
    return test_accuracy
   
def create_checkpoint(model,path,model_name,class_to_idx,optimizer,epochs):
    """
    Create a checkpoint object and save it to the given path. 
    This function gets the path and model_name then generates the file path and name
    Parameters:
        model: trained model
        
        path: save path for checkpoint
        
        model_name: it is used as checkpoint name 
        
        class_to_idx: image class to indexes
        
        optimizer: optimizer
        
        epoch: cycle count for tranining.
    
    """
    model.cpu()
    model.class_to_idx =  class_to_idx

    checkpoint = {'arch'model_name:,
                  'state_dict':model.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'epochs':epochs,
                  'optimizer_state_dict':optimizer.state_dict
                   }
    if model_name == 'resnet101':
        checkpoint['fc'] = model.fc
    else:
        checkpoint['classifier'] = model.classifier
    file = str()
    if path != None:
        file = path + '/' + model_name + '_checkpoint.pth'
    else:
        file = model_name + '_checkpoint.pth'
    torch.save(checkpoint,file)
    print("Model(%s) has been saved into Path(%s)"%(model_name,file))


if __name__ == "__main__":
    main()
