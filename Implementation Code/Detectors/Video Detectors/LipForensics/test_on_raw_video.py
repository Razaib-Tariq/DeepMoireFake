import argparse
from collections import defaultdict

import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm

from data.transforms import NormalizeVideo, ToTensorVideo
from data.dataset_clips import ForensicsClips, CelebDFClips, DFDCClips
from data.samplers import ConsecutiveClipSampler
from models.spatiotemporal_net import get_model
from utils import get_files_from_split

import numpy as np
import bisect
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc


def eval_state(probs, labels, thr):
    probs = probs.reshape(-1)
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def find_best_threshold(scores, labels):
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        best_acc = 0
        best_thresh = None
        for i, thresh in enumerate(thresholds):
            # compute accuracy for this threshold
            pred_labels = [1 if s >= thresh else 0 for s in scores]
            acc = sum([1 if pred_labels[j] == labels[j] else 0 for j in range(len(labels))]) / len(labels)

            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        return best_thresh, roc_auc


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    parser.add_argument("--frames_per_clip", default=25, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--weights_forgery_path",
        help="Path to pretrained weights for forgery detection",
        type=str,
        default="/pretrained_checkpoints/lipforensics/lipforensics_ff.pth"
    )
    parser.add_argument('--data-type', default='created',  type=str,choices=['created', 'collected'],
                         help='dataset to test')
    parser.add_argument('--output-dir', default="/media/data1/razaib/Moire/prediction_results/pretrained_models", type=str, help='output directory to save the results')
    parser.add_argument("--model-name", type=str, default="LipForensics", choices=['LipForensics'],help="The name of the model to test")
    args = parser.parse_args()
    return args


class CelebDFClips(Dataset):
    """Dataset class for Celeb-DF-v2"""
    def __init__(
            self,
            real='real',
            fake='DeepFaceLab',
            frames_per_clip=25,
            grayscale=False,
            transform=None,
            
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []
        self.real_name = real
        self.fake_name = fake

        ds_types = [self.real_name, self.fake_name]

        if 'mouth' in os.listdir(self.real_name):
            for ds_type in ds_types:
                video_paths = os.path.join(ds_type, 'mouth')
                videos = sorted(os.listdir(video_paths))

                self.videos_per_type[ds_type] = len(videos)
                for video in videos:
                    path = os.path.join(video_paths, video)
                    num_frames = len(os.listdir(path))
                    num_clips = num_frames // frames_per_clip
                    self.clips_per_video.append(num_clips)
                    self.paths.append(path)
        else:
            print("in the wild type")
            for ds_type in ds_types:
                videos = [os.path.join(ds_type, u,'mouth') \
                          for u in sorted(os.listdir(ds_type)) \
                          if os.path.isdir(os.path.join(ds_type, u, 'mouth'))]
                self.videos_per_type[ds_type] = len(videos)
                for path in videos:
                    num_frames = len(os.listdir(path))
                    num_clips = num_frames // frames_per_clip
                    self.clips_per_video.append(num_clips)
                    self.paths.append(path)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()
        print("Number of clips: ",self.cumulative_sizes[-1])
    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.paths[video_idx]
        frames = sorted(os.listdir(path))

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip

        sample = []
        for idx in range(start_idx, end_idx, 1):
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            sample.append(img)

        sample = np.stack(sample)

        return sample, video_idx

    def __getitem__(self, idx):
        sample, video_idx = self.get_clip(idx)
        label = 0 if video_idx < self.videos_per_type[self.real_name] else 1
        label = torch.tensor(label, dtype=torch.float32)
        sample = torch.from_numpy(sample).unsqueeze(-1)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, video_idx


def compute_video_level_auc(video_to_logits, video_to_labels):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])
    lb = output_labels.cpu().numpy()
    pred = output_batch.cpu().numpy()
    fpr, tpr, _ = metrics.roc_curve(lb, pred)
    return metrics.auc(fpr, tpr), lb, pred
 
def validate_video_level(model, loader, args):
    """ "
    Evaluate model using video-level AUC score.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance
    loader : torch.utils.data.DataLoader
        Loader for forgery data
    args
        Options for evaluation
    """
    model.eval()

    video_to_logits = defaultdict(list)
    video_to_labels = {}
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels, video_indices = data
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Forward
            logits = torch.sigmoid(
                model(
                    images, lengths=[args.frames_per_clip] * images.shape[0]
                    )
                )

            # Get maps from video ids to list of logits (representing outputs for clips) as well as to label
            for i in range(len(video_indices)):
                video_id = video_indices[i].item()
                video_to_logits[video_id].append(logits[i])
                video_to_labels[video_id] = labels[i]

    auc_video , lb, pred= compute_video_level_auc(video_to_logits, video_to_labels)
    return pred, lb

def main(args):
    model = get_model(weights_forgery_path=args.weights_forgery_path)
    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    dataset = CelebDFClips(args.folder[0], 
                           args.folder[1], 
                           args.frames_per_clip, 
                           args.grayscale, 
                           transform)
    # Get sampler that splits video into non-overlapping clips
    sampler = ConsecutiveClipSampler(dataset.clips_per_video)

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    predictions, labels = validate_video_level(model, loader, args)

    predictions = np.array(predictions)
    labels = np.array(labels)
    np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_preds.npy", predictions)
    np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_labels.npy", labels)
    print("At threshold = 0.5")
    best_thresh = 0.5

    TN, FN, FP, TP = eval_state(probs=predictions, labels=labels, thr=best_thresh)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
        TPR = 0
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
        TPR = TP / float(TP + FN)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
        TPR = TP / float(TP + FN)
        
    
    HTER = (FAR + FRR) / 2.0
    ACC = (TN + TP) / (TN + FN + FP + TP)
    print(f"HTER: {HTER*100:.2f}")
    print(f"FAR: {FAR*100:.2f}")
    print(f"TPR: {TPR*100:.2f}")

    best_thresh, AUC = find_best_threshold(predictions, labels)
    print(f"At best threshold = {best_thresh:.4f}")
    TN, FN, FP, TP = eval_state(probs=predictions, labels=labels, thr=best_thresh)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
        TPR = 0
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
        TPR = TP / float(TP + FN)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
        TPR = TP / float(TP + FN)

    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    HTER = (FAR + FRR) / 2.0
    ACC_best = (TN + TP) / (TN + FN + FP + TP)
    print(f"HTER: {HTER*100:.2f}")
    print(f"FAR: {FAR*100:.2f}")
    print(f"TPR: {TPR*100:.2f}")
    print(f"ACC: {ACC_best*100:.2f}")
    print(f"AUC: {AUC*100:.2f}")
    print(f"Precision: {precision*100:.2f}")
    print(f"Recall: {recall*100:.2f}")
    print(f"F1 Score: {f1_score*100:.2f}")
    
    return (ACC*100, ACC_best*100, AUC*100, precision*100, recall*100, f1_score*100)

def test_folders(args):
    """
    Test all the generated data and export them into csv file with 4 columns
    - Dataset
    - ACC
    - ACC @best
    - AUC
    """
    print("TEST UPON SETTING")
    out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": [], "Precision": [], "Recall": [], "F1": []
    })
    data_name = "DFD-Fake"
    args.folder = ["/DFD/DFD-Real",
                f"/DFD/{data_name}/"]
    args.data_name = data_name
    os.makedirs(f"{args.output_dir}/{args.data_name}", exist_ok=True)
    ACC, ACC_best, AUC, Precision, Recall, F1 = main(args)
    out_results = out_results.append({"Dataset":data_name, 
                                        "Acc": np.round(ACC,2), 
                                        "Acc_best": np.round(ACC_best,2), 
                                        "AUC": np.round(AUC,2),
                                        "Precision": np.round(Precision,2),
                                        "Recall": np.round(Recall,2),
                                        "F1": np.round(F1,2)}, ignore_index=True)
    # include average results
    out_results = out_results.append({"Dataset":"Avg", 
                                        "Acc": f"{out_results.Acc.mean():.2f} ({out_results.Acc.std():.2f})", 
                                        "Acc_best": f"{out_results.Acc_best.mean():.2f} ({out_results.Acc_best.std():.2f})", 
                                        "AUC": f"{out_results.AUC.mean():.2f} ({out_results.AUC.std():.2f})", 
                                        "Precision": f"{out_results.Precision.mean():.2f} ({out_results.Precision.std():.2f})",
                                        "Recall": f"{out_results.Recall.mean():.2f} ({out_results.Recall.std():.2f})",
                                        "F1": f"{out_results.F1.mean():.2f} ({out_results.F1.std():.2f})"},
                                        ignore_index=True)
    
    out_results.to_csv(f"{args.output_dir}/{args.data_name}/{args.model_name}.csv", index=False)
    

if __name__ == "__main__":
    args = parse_args()
    if args.data_type =='created': 
        test_folders(args)
    else: # Test the collected dataset
        itw_test_folders(args)
