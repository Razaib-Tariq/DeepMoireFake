from utils.plugin_loader import PluginLoader
from config import config as cfg
import torch
import os
import pandas as pd
import glob
import numpy as np
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)

def eval_state(probs, labels, thr):
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

def make_prediction(args):
    cfg.init_with_yaml()
    cfg.update_with_yaml("ftcn_tt.yaml")
    cfg.freeze()

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.cuda()
    classifier.eval()
    classifier.load("/checkpoints/ftcn/ftcn_tt.pth")

    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    input_file = args.video
    # os.makedirs(args.out_dir, exist_ok=True)
    # basename = os.path.splitext(os.path.basename(input_file))[0] + ".avi"
    # out_file = os.path.join(args.out_dir, basename)

    max_frame = 768
    cache_file = f"{input_file}_{str(max_frame)}_{args.model_name}.pth"
    # os.remove(cache_file)
    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(input_file, max_size=max_frame, cvt=True)
        print("detection result loaded from cache")
        if len(frames) != len(detect_res):
            print("detecting")
            detect_res, all_lm68, frames = detect_all(input_file, return_frames=True, max_size=max_frame)
            torch.save((detect_res, all_lm68), cache_file)
            print("detect finished")
    else:
        print("detecting")
        detect_res, all_lm68, frames = detect_all(
            input_file, return_frames=True, max_size=max_frame
        )
        torch.save((detect_res, all_lm68), cache_file)
        print("detect finished")

    print("number of frames",len(frames))

    
    shape = frames[0].shape[:2]

    all_detect_res = []

    assert len(all_lm68) == len(detect_res)

    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)

    detect_res = all_detect_res

    print("split into super clips")

    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    print("full_tracks", len(tracks))

    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        print(start, end)
        # if end - start < 10: continue
        assert len(detect_res[start:end]) == len(track)

        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box,lm5,lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]

            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left

            new_box = (box.reshape(2, 2) - top_left).reshape(-1)

            info = (new_box, new_lm5, new_lm68, big_box)


            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]

            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx

            frame_boxes[frame_idx] = np.rint(box).astype(int)

    print("sampling clips from super clips", super_clips)

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))

        if super_clip_size < 10: continue
        if super_clip_size < clip_size: # padding
            post_module = inner_index[1:-1][::-1] + inner_index

            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            assert len(post_module) == pad_length

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            assert len(pre_module) == pad_length

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)

        frame_range = [
            inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)
    
    preds = []
    frame_res = {}

    for clip in tqdm(clips_for_video, desc="testing"):
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
        landmarks, images = crop_align_func(landmarks, images)
        images = torch.as_tensor(images, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images = images.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images)

        pred = float(output["final_output"])
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)
        preds.append(pred)
    return np.mean(preds)

def main(args):
    if isinstance(args.folder, list):
        print(args.folder)
        video_path = []
        for p in args.folder:
            print(p.split('/')[-2].upper())
            video_path += glob.glob(p + '/**/*.mp4', recursive=True)
        labels = [1 if 'real' not in p else 0 for p in video_path]
        print("Test lb:", sum(labels), len(video_path))
    else:
        print(args.folder.split('/')[-2].upper())
        video_path = glob.glob(args.folder + '/**/*.mp4', recursive=True)
        labels = [1] * len(video_path)
    print("Number of videos:", len(video_path))
    predictions = []
    for v in tqdm(video_path):
        args.video = v
        prob = make_prediction(args)
        predictions.append(prob)


    predictions = np.array(predictions)
    labels = np.array(labels)
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
    np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_preds.npy", predictions)
    np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_labels.npy", labels)
    
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
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
    })
    data_name = "Deepfakes"
    args.folder = ["/FF++/deepfakes/real", #real of the fake folder
                    f"/FF++/deepfakes/{data_name}/"]
    args.data_name = data_name
    os.makedirs(f"{args.output_dir}/{args.data_name}", exist_ok=True)
    ACC, ACC_best, AUC, Precision, Recall, F1 = main(args)
        # Create DataFrame for new row
    new_row = pd.DataFrame({"Dataset": data_name, 
                                "Acc": np.round(ACC, 2), 
                                "Acc_best": np.round(ACC_best, 2), 
                                "AUC": np.round(AUC, 2),
                                "Precision": np.round(Precision, 2),
                                "Recall": np.round(Recall, 2),
                                "F1": np.round(F1, 2)}, index=[0])

        # Concatenate new row to out_results DataFrame
    out_results = pd.concat([out_results, new_row], ignore_index=True)

    # Create DataFrame for average row
    avg_row = pd.DataFrame({"Dataset": "Avg", 
                            "Acc": f"{out_results['Acc'].mean():.2f} ({out_results['Acc'].std():.2f})", 
                            "Acc_best": f"{out_results['Acc_best'].mean():.2f} ({out_results['Acc_best'].std():.2f})", 
                            "AUC": f"{out_results['AUC'].mean():.2f} ({out_results['AUC'].std():.2f})", 
                            "Precision": f"{out_results['Precision'].mean():.2f} ({out_results['Precision'].std():.2f})",
                            "Recall": f"{out_results['Recall'].mean():.2f} ({out_results['Recall'].std():.2f})",
                            "F1": f"{out_results['F1'].mean():.2f} ({out_results['F1'].std():.2f})"}, index=[0])

    # Concatenate average row to out_results DataFrame
    out_results = pd.concat([out_results, avg_row], ignore_index=True)

    # Save results to CSV
    out_results.to_csv(f"{args.output_dir}/{args.data_name}/{args.model_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, metavar='S',  nargs='+', help='The folder of videos to test the model')
    parser.add_argument('--output-dir', default="/prediction_results/pretrained_models", type=str, help='output directory to save the results'	)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="FTCN", choices=['FTCN'])
    args = parser.parse_args()
	
    if args.folder is not None:
        main(args)
    else:
        test_folders(args)
 
    