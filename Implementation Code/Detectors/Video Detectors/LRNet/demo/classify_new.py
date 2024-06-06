import numpy as np
import tensorflow.keras as K
import os
from tqdm import tqdm
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import numpy as np
import pandas as pd

import argparse


block_size = 60
DROPOUT_RATE = 0.5
RNN_UNIT = 64
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
device = "CPU" if len(gpus) == 0 else "GPU"
print("Using device: {}".format(device))



def find_best_threshold(scores, labels):
    labels = np.array(labels)
    scores = np.array(scores)

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


def eval_state(probs, labels, thr):
    labels = np.array(labels)
    probs = np.array(probs)

    predict = probs >= thr
    labels = np.array(labels)
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def evaluate(prob_list, label_list):
    assert len(prob_list) == len(label_list)
    print("At threshold = 0.5")
    best_thresh = 0.5
    TN, FN, FP, TP = eval_state(probs=prob_list, labels=label_list, thr=best_thresh)
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
        
    ACC = (TN + TP) / (TN + FN + FP + TP)
    HTER = (FAR + FRR) / 2.0
    print(f"HTER: {HTER*100:.2f}")
    print(f"FAR: {FAR*100:.2f}")
    print(f"TPR: {TPR*100:.2f}")
    print(f"ACC: {ACC*100:.2f}")


    np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_retinaface_preds.npy", prob_list)
    np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_retinaface_labels.npy", label_list)
    #np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_blazeface_preds.npy", prob_list)
    #np.save(f"{args.output_dir}/{args.data_name}/{args.model_name}_blazeface_labels.npy", label_list)

    print("Max values", len(prob_list), np.max(prob_list), np.min(prob_list), np.max(label_list), np.min(label_list))

    
    best_thresh, AUC = find_best_threshold(scores=prob_list, labels=label_list)
    print(f"At best threshold = {best_thresh:.4f}")
    TN, FN, FP, TP = eval_state(probs=prob_list, labels=label_list, thr=best_thresh)
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

def get_data_for_test(path, fake, block):  # fake:manipulated=1 original=0
    files = [u for u in os.listdir(path) if (u.endswith(".txt") and (not u.endswith("landmark_logs.txt")))]
    x = []
    x_diff = []
    y = []

    video_y = []
    count_y = {}
    sample_to_video = []

    print("Loading data and embedding...")
    for file in tqdm(files):
        vectors = np.loadtxt(path + file)
        video_y.append(fake)

        for i in range(0, vectors.shape[0] - block, block):
            vec = vectors[i:i + block, :]
            x.append(vec)
            vec_next = vectors[i + 1:i + block, :]
            vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
            vec_diff = (vec_next - vec)[:block - 1, :]
            x_diff.append(vec_diff)

            y.append(fake)

            # Dict for counting number of samples in video
            if file not in count_y:
                count_y[file] = 1
            else:
                count_y[file] += 1

            # Recording each samples belonging
            sample_to_video.append(file)
    return np.array(x), np.array(x_diff), np.array(y), np.array(video_y), np.array(sample_to_video), count_y


def merge_video_prediction(mix_prediction, s2v, vc):
    prediction_video = []
    pre_count = {}
    for p, v_label in zip(mix_prediction, s2v):
        p_bi = 0
        if p >= 0.5:
            p_bi = 1
        if v_label in pre_count:
            pre_count[v_label] += p_bi
        else:
            pre_count[v_label] = p_bi
    for key in pre_count.keys():
        prediction_video.append(pre_count[key] / vc[key])
    return prediction_video


def main(args):
    landmark_path = args.landmark_path
    assert os.path.exists(landmark_path), f"{landmark_path}: Landmark path does not exist. Please extract the landmarks firstly."
    test_samples, test_samples_diff, _, _, test_sv, test_vc = get_data_for_test(landmark_path, 1, block_size)

    model = K.Sequential([
        layers.InputLayer(input_shape=(block_size, 136)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.GRU(RNN_UNIT)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(2, activation='softmax')
    ])
    model_diff = K.Sequential([
        layers.InputLayer(input_shape=(block_size - 1, 136)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.GRU(RNN_UNIT)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(2, activation='softmax')
    ])

    lossFunction = K.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = K.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss=lossFunction,
                  metrics=['accuracy'])
    model_diff.compile(optimizer=optimizer,
                  loss=lossFunction,
                  metrics=['accuracy'])

    print("Loading models and predicting...")

    #----Using Deeperforensics 1.0 Parameters----#
    # model.load_weights('./model_weights/deeper/g1.h5')
    # model_diff.load_weights('./model_weights/deeper/g2.h5')

    # ----Using FF++ Parameters----#
    model.load_weights('./model_weights/ff/g1.h5')
    model_diff.load_weights('./model_weights/ff/g2.h5')

    prediction = model.predict(test_samples)
    prediction_diff = model_diff.predict(test_samples_diff)
    mix_predict = []
    for i in range(len(prediction)):
        mix = prediction[i][1] + prediction_diff[i][1]
        mix_predict.append(mix/2)

    prediction_video = merge_video_prediction(mix_predict, test_sv, test_vc)

    """
    Show the results
    """
    print("\n\n", "#----Prediction Results----#")
    video_names = []
    for key in test_vc.keys():
        video_names.append(key)
    for i, pd in enumerate(prediction_video):
        if pd >= 0.5:
            label = "Fake"
        else:
            label = "Real"
        print("{}-Prediction label: {}; Scores:{}".format(video_names[i], label, pd))
    print("#------------End------------#")
    return prediction_video
def test_folders(args):
    if args.landmark_path is not None:
        main(args)
    else:
        print("Runing upon setting")
        out_results = pd.DataFrame({
        "Dataset":[], "Acc": [], "Acc_best": [], "AUC": []
         })
        
        real_lm_path =  f'/DFD/landmarks_new/DFD-Original/landmarks/'
        args.landmark_path = real_lm_path
        real_score = main(args)
        data_name = "DFD-Fake"
        args.data_name = data_name
        os.makedirs(f"{args.output_dir}/{args.data_name}", exist_ok=True)
        fake_lm_paths = f'/DFD/landmarks_new/{data_name}/landmarks/' 
        args.landmark_path = fake_lm_paths
        fake_score = main(args)
        label_list = [0]*len(real_score)+[1]*len(fake_score)
        # print(real_score, "\n", fake_score)
        prob_list = np.concatenate([real_score,fake_score])

        ACC, ACC_best, AUC, Precision, Recall, F1 = evaluate(label_list=label_list, prob_list=prob_list)
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
    
    out_results.to_csv(f"{args.output_dir}/{args.data_name}/{args.model_name}_retinaface.csv", index=False)
    #out_results.to_csv(f"{args.output_dir}/{args.data_name}/{args.model_name}_blazeface.csv", index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify  videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-l', '--landmark_path', type=str, default=None,
                        help="Landmark path to nake prediction(folder)"
                        )
    parser.add_argument('--data-type', default='created',  type=str,choices=['created', 'collected'],
                         help='dataset to test')
    parser.add_argument("--model-name", type=str, default="LRNet", choices=['LRNet'],  help="The name of the model to test")
    parser.add_argument('--output-dir', default="/prediction_results/pretrained_models", type=str, help='output directory to save the results')
    args = parser.parse_args()
    if args.data_type =='created': 
        test_folders(args)
    else: # Test the collected dataset
        itw_test_folders(args)
    