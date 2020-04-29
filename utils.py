import argparse
import os

def get_train_input_args():


    parser = argparse.ArgumentParser(prog='Image Classifier Trainer',description='Get Input Arguments for train')
    parser.add_argument('datafile',type=readable_dir,help='The data directory for images')
    parser.add_argument('--save_dir',type=readable_dir,help='Save Directory of Trained Model Checkpoint')
    parser.add_argument('--arch',type=str,help='Tourch pre-trained models as --arch with default is vgg19',choices=['vgg19','densenet121','resnet101'],default='vgg19',action='store')
    parser.add_argument('--learning_rate',type=float,help='Learning rate of the network default is 0.001',default=0.001,action='store')
    parser.add_argument('--epochs',type=int,help='Epoch of the network default is 2',default=2,action='store')
    parser.add_argument('--gpu',help='GPU Mode',action='store_true')
    

    return parser.parse_args()

def get_predict_input_args():
   
    parser = argparse.ArgumentParser(prog='Image Classifier Trainer',description='Get Input Arguments for train')
    parser.add_argument('image_path',type=readable_file,help='The image file path for prediction')
    parser.add_argument('checkpoint_path',type=readable_file,help='The checkpoint file path')
    parser.add_argument('--top_k',type=int,default=1,help='Return top K most likely classes')
    parser.add_argument('--category_names',type=readable_file,help='Use a mapping of categories to real names')
    parser.add_argument('--gpu',help='GPU Mode',action='store_true')
    
    return parser.parse_args()

def readable_dir(prospective_dir):
    
    if not os.path.isdir(prospective_dir):
        raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
    if os.access(prospective_dir, os.R_OK):
        return prospective_dir
    else:
        raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

def readable_file(prosprective_file):
    if not os.path.isfile(prosprective_file):
        raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid file".format(prosprective_file))
    if os.access(prosprective_file, os.R_OK):
        return prosprective_file
    else:
        raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable file".format(prosprective_file))
    