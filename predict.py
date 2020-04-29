import torch
from torchvision import  models
from PIL import Image
import numpy as np
import json
from utils import get_predict_input_args


def main():
    """
    Example Execution Command
        python predict.py flowers/test/102/image_08012.jpg checkpoints/vgg19_checkpoint.pth --top_k 2 --category_names cat_to_name.json --gpu
    """
    args = get_predict_input_args()
    
  
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    checkpoint = load_checkpoint(args.checkpoint_path)
    model = load_model(checkpoint)
    props,classes = predict(args.image_path,model,args.top_k) 
   
    flower_names = classToNames(model.class_to_idx,classes,load_cat_to_name_arr(args.category_names))
    
    for i in range(args.top_k):
        
        if len(flower_names) == 0:
            print("Image_Class: %s \t Probability: %f \n"%(classes[i],props[i]))
        else:
            print("Image_Class_Name: %s \t Probability: %f \n"%(flower_names[i],props[i]))
            

def load_cat_to_name_arr(file):
    if file == None:
        return dict()
            
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    return checkpoint   
    
def load_model(checkpoint):
   
    t_models = {
              'vgg19':models.vgg19(pretrained=True),
              'densenet121': models.densenet121(pretrained=True),
              'resnet101':models.resnet101(pretrained=True)}
  
    model = t_models.get(checkpoint['arch'],'vgg19')
    
    if checkpoint['arch'] == 'vgg19' or checkpoint['arch'] == 'densenet121':
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
    else:
        model.fc = checkpoint['fc']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
   
    
    return model

def load_epochs(checkpoint):
    return checkpoint['epochs']

def load_optimizer_state(checkpoint):
    return checkpoint['optimizer_state_dict']

def process_image(image_path):
    MAX_SIZE = 256
    ratio = 0
    
    img = Image.open(image_path)
    
    width, height = img.size
 
    if width > height:
        ratio = MAX_SIZE/float(height)
        height = MAX_SIZE
        width = int((float(width)*float(ratio)))
    else:
        ratio = MAX_SIZE/float(width)
        width = MAX_SIZE
        height = int((float(height)*float(ratio)))
    
    
    img = img.resize((width,height))
       
    left_margin = (img.width-224)/2
    up_margin = (img.height-224)/2
    right_margin = left_margin + 224
    bottom_margin = up_margin + 224
    
    # left, up, right, bottom
    img = img.crop((left_margin,up_margin,right_margin,bottom_margin))
      
    np_img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    np_img = (np_img - mean)/std
    np_img = np_img.transpose((2, 0, 1))
    
    return torch.from_numpy(np_img).type(torch.FloatTensor) 

def predict(image_path, model, topk):
   
    
    model.eval()
    np_img = process_image(image_path)
    np_img.unsqueeze_(0)

    output=None
    with torch.no_grad():
        output = model.forward(np_img)
    
    ps = torch.exp(output)
    top_k_prob,top_k_class = ps.topk(topk,dim=1)
  
    return top_k_prob[0].numpy(), top_k_class[0].numpy()

def classToNames(class_to_idx,top_class,cat_to_name):
    
    if len(cat_to_name) == 0:
        return dict()
    
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_class]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    return top_flowers



if __name__ == "__main__":
    main()
