
import os
import argparse
from tkinter.tix import REAL
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.nn import functional as F
import torch.nn as nn
# from knn_cuda import KNN
import pickle
from sklearn import metrics
import yaml
from dataloader import FaceDataset
from torch.utils.data import DataLoader
from model_zoo import (SelfBlendedModel, 
                       MAT, 
                       RosslerModel, 
                       ForgeryNet, 
                       CADDM, 
                       resnet50,
                       calculate_roc_ex, 
                       evaluate_new)
from utils import (IsotropicResizeTorch, PadIfNeeded,ToIntImage)
from sklearn.metrics import roc_curve, auc
from loops import evaluation, ict_evaluate

import sys
sys.path.append("./Capsule-Forensics-v2/")
import model_big as CapSule

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process to test a model with a folder of images')
    VALID_MODELS = ['selfblended', 'mat', 'ict', 'rossler', 'forgerynet', 'capsule', 'caddm', 'ccvit', 'add']
    parser.add_argument('--model-name', default=None, choices=VALID_MODELS, type=str, help='the model name to test')
    parser.add_argument('--test-folder',  type=str, default=None, metavar='S',  nargs='+', help='The folder of images to test the model')
    parser.add_argument('--batch-size', default=32,  type=int, help='batch size to test the model')
    parser.add_argument('--penul-ft', action='store_true', help='Return penultimate ft for plotting')
    parser.add_argument('--sampling-rate', default=1,  type=int, help='sampling frequecy each video, 1 mean all frames tested')
    parser.add_argument('--data-type', default='created',  type=str,choices=['created', 'collected', 'cdfv2', 'dfdc'],
                         help='dataset to test')
    parser.add_argument('--output-dir', default="/prediction_results/pretrained_models/", type=str, help='output directory to save the results'	)
    args = parser.parse_args()
    return args




def main(args):
    print("Test model: ", args.model_name.upper())
    softmax = True
    if args.model_name == 'selfblended':
        model = SelfBlendedModel().cuda()

        img_size = 380
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False
    elif args.model_name == 'rossler':
        model = RosslerModel(modelchoice='xception').cuda()
        

        img_size = 299
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5] * 3, [0.5] * 3)
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False
    elif args.model_name == 'mat':
        model = MAT().cuda()
        
        img_size = 380
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False
        
    elif args.model_name == 'ccvit':
        sys.path.append("Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit")
        from cross_efficient_vit import CrossEfficientViT
        with open("Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit/configs/architecture.yaml", 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
        model = CrossEfficientViT(config=config)
        model.load_state_dict(
            torch.load("/pretrained_checkpoints/ccvit/cross_efficient_vit.pth"),
            strict=True)
        model.eval()
        model = model.cuda()
        
        img_size = 224
        transformer = transforms.Compose([
                            IsotropicResizeTorch(img_size),
                            PadIfNeeded(img_size, img_size, fill=(0, 0, 0)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.,0.,0.],std=[1.0,1.0,1.0]),
                            ])
        fake_class = 1
        softmax = False
        model.eval()
        use_bgr = True
        
    elif args.model_name == 'ict':
        model = ICT().cuda()
        
        img_size = 112
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        fake_class = 1
        model.eval()
        use_bgr = False
        
    elif args.model_name == 'caddm':
        model = CADDM(2, backbone='resnet34').cuda()
        pretrained_model = '/pretrained_checkpoints/iil/resnet34.pkl'
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['network'])
        img_size = 224
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor()
                        ])
        fake_class = 1
        model.eval()
        use_bgr = True
    elif args.model_name == 'add':
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.cuda()
        pretrained_model = '/pretrained_checkpoints/add/deepfakes_c23resnet50_kd_valacc_img128_kd21_freq_swd_best.pth'
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        img_size = 128
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        fake_class = 1
        model.eval()
        use_bgr = False
        
    elif args.model_name == 'capsule':
        vgg_ext = CapSule.VggExtractor().cuda()
        capnet = CapSule.CapsuleNet(2, 0).cuda()
        capnet.load_state_dict(torch.load(os.path.join('Capsule-Forensics-v2/checkpoints/binary_faceforensicspp','capsule_21.pt')))
        capnet.eval()
        model = [vgg_ext, capnet]
        img_size = 300

        transformer =  transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        fake_class = 1
        use_bgr = False

    elif  args.model_name =='forgerynet':
        model = ForgeryNet(num_classes=2).cuda()

        weight_path = "/pretrained_checkpoints/forgerynet/ckpt_iter.pth.tar"
        weight = torch.load(weight_path)['state_dict']
        updated_weight = dict()
        for k in weight.keys(): updated_weight[k.replace('module.', '')] = weight[k]
        model.load_state_dict(updated_weight, strict=True)
        img_size = 299
        transformer = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5] * 3, [0.5] * 3)
                            ])
        fake_class = 1
        model.eval()
        use_bgr = False


    valid_dataloader = DataLoader(FaceDataset(args.test_folder, transform=transformer, use_bgr=use_bgr, sampling_rate=args.sampling_rate),
                            batch_size=args.batch_size, shuffle=False)
    if not args.penul_ft:
        if args.model_name != 'ict':
            ACC, ACC_best, AUC, Precision, Recall, F1 = evaluation(model, valid_dataloader, fake_class, args.model_name, 
                                             softmax=softmax, 
                                             model_name=args.model_name,
                                             output_dir=args.output_dir, 
                                             data_name=args.data_name)
        else:
            ACC, ACC_best, AUC,  Precision, Recall, F1  = ict_evaluate(model, valid_dataloader,
                                             model_name=args.model_name,
                                             output_dir=args.output_dir, 
                                             data_name=args.data_name)
        return (ACC, ACC_best, AUC, Precision, Recall, F1)
    

    else:
        if args.model_name != 'ict':
            ACC, ACC_best, AUC,  Precision, Recall, F1, ft_list, lb_list = evaluation(model, valid_dataloader, fake_class, args.model_name, 
                                                              softmax=softmax, penul_ft=True,                                                               
                                                                model_name=args.model_name,
                                                                output_dir=args.output_dir, 
                                                                data_name=args.data_name )
        else:
            ACC, ACC_best, AUC, Precision, Recall, F1, ft_list, lb_list = ict_evaluate(model, valid_dataloader,                                                                
                                             model_name=args.model_name,
                                             output_dir=args.output_dir, 
                                             data_name=args.data_name)
            
        return (ACC, ACC_best, AUC, Precision, Recall, F1, ft_list, lb_list)
    


def dfdcp_test_folders(args):
    """
    Test in the wild dataset
    - Dataset : DFDCP
    - ACC
    - ACC @best
    - AUC
    """
    REAL_DIR = "/DFDC//Real/"
    FAKE_DIR = "/DFDC/Fake/"
    
    print("TEST UPON SETTING")
    out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
    })
    args.data_name = "DFDC-OG-method_B-10sec"
    os.makedirs(f"{args.output_dir}/{args.data_name}", exist_ok=True)
    args.test_folder = [REAL_DIR, FAKE_DIR]
    print(args.test_folder)
    ACC, ACC_best, AUC, Precision, Recall, F1 = main(args)
    out_results = out_results.append({"Dataset":args.data_name, 
                                            "Acc": np.round(ACC,2), 
                                            "Acc_best": np.round(ACC_best,2), 
                                            "AUC": np.round(AUC,2),
                                            "Precision": np.round(Precision,2),
                                            "Recall": np.round(Recall,2),
                                            "F1": np.round(F1,2)}, ignore_index=True)

    out_results.to_csv(f"{args.output_dir}/{args.data_name}/{args.model_name}.csv", index=False)
    
        
if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    if args.data_type =='created': # Test the Stablized dataset
        if args.test_folder is not None:
            main(args)
        else:
            dfdcp_test_folders(args)
    elif args.data_type =='DeepFaceLab': # Test the updated dataset
        dfdcp_test_folders(args)
    elif args.data_type =='cdfv2': # Test the updated dataset
        cdf_test_folders(args)
    elif 'collected': # Test the collected dataset
        itw_test_folders(args)