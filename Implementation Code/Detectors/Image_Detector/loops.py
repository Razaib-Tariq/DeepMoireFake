
import os
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import cuml 
from torch.nn import functional as F
import torch
from torch.autograd import Variable



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



def ict_evaluate(model, loader,  nrof_folds = 5, **kwargs):
    model_name = kwargs.get('model_name', 'None')
    output_dir = kwargs.get('output_dir', 'None')
    data_name = kwargs.get('data_name', 'None')
    
    model.eval()
    query = model.query
    idx = 0
    
    embeddings1 = []
    embeddings2 = []
    q_embeddings1 = []
    q_embeddings2 = []
    r_embeddings1 = []
    r_embeddings2 = []
    issame = []
    count = 0
    tot = min(500, len(loader))
    tri_num = 1
    # knn = KNN(k=tri_num, transpose_mode=True) # OLD CODE
    knn_inner_model = cuml.neighbors.NearestNeighbors(n_neighbors=tri_num)        
    knn_inner_model.fit(query['inner'][0])
    knn_outer_model = cuml.neighbors.NearestNeighbors(n_neighbors=tri_num)
    knn_outer_model.fit(query['outer'][0])
    print("Fitted KNN model", query['inner'].shape, query['outer'].shape)
    
    all_videoIDs = []
    all_labels = []
    with torch.no_grad():
        for (imgs, labels, videoID) in iter(loader):
            #print (imgs.size())
            
            imgs = imgs.cuda()
            labels = labels.cuda()
            all_videoIDs = np.append(all_videoIDs, videoID.numpy())
            all_labels =  np.append(all_labels, labels.cpu().numpy())

            inner_emb, outer_emb = model(imgs)
            embeddings1.extend(inner_emb.cpu().numpy())
            embeddings2.extend(outer_emb.cpu().numpy())

            # _, idx = knn(query['inner'], inner_emb.unsqueeze(0)) # OLD CODE
            _, idx = knn_inner_model.kneighbors(inner_emb)
            idx = torch.tensor(idx).unsqueeze(0)


            tars = query['outer'][0][idx[0,:,0]]
            for i in range(1, tri_num):
                tars += query['outer'][0][idx[0,:,i]]
            tars = tars / tri_num
            q_embeddings2.extend(tars.cpu().numpy())
            tars = query['inner'][0][idx[0,:,0]]
            for i in range(1, tri_num):
                tars += query['inner'][0][idx[0,:,i]]
            tars = tars / tri_num
            r_embeddings1.extend(tars.cpu().numpy())

            # _, idx = knn(query['outer'], outer_emb.unsqueeze(0)) # OLD CODE
            _, idx = knn_outer_model.kneighbors(outer_emb)
            idx = torch.tensor(idx).unsqueeze(0)
            tars = query['inner'][0][idx[0,:,0]]
            for i in range(1, tri_num):
                tars += query['inner'][0][idx[0,:,i]]
            tars = tars / tri_num
            q_embeddings1.extend(tars.cpu().numpy())
            tars = query['outer'][0][idx[0,:,0]]
            for i in range(1, tri_num):
                tars += query['outer'][0][idx[0,:,i]]
            tars = tars / tri_num
            r_embeddings2.extend(tars.cpu().numpy())

            temp = [True if labels[i] else False for i in range(len(labels))]
            issame.extend(temp)
            count += 1
            if count % 100 == 0:
                print(f"{str(count)}/{str(tot)}")
            if count == tot:
                break
        
    embeddings1 = np.asarray(embeddings1)
    embeddings2 = np.asarray(embeddings2)

    q_embeddings1 = np.asarray(q_embeddings1)
    q_embeddings2 = np.asarray(q_embeddings2)
    r_embeddings1 = np.asarray(r_embeddings1)
    r_embeddings2 = np.asarray(r_embeddings2)
    issame = np.asarray(issame)
    

    thres = 0.5
    thres2 = 0.5
    #print ('THRES:', thres, thres2)
    
    # tpr, fpr, accuracy, best_thresholds = evaluate_new(embeddings1, embeddings2, issame, nrof_folds)
    # auc = metrics.auc(fpr, tpr)
    
    # print ("Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'ICT'))

    
    dist1 = np.sum(np.square(np.subtract(embeddings1, embeddings2)), 1)
    dist2 = np.sum(np.square(np.subtract(embeddings2, q_embeddings2)), 1)
    dist3 = np.sum(np.square(np.subtract(embeddings1, q_embeddings1)), 1)

    tau = 0.5
    dis_exp2 = np.sum(np.square(np.subtract(embeddings1, r_embeddings1)), 1)
    #print ('DIS2', dis_exp2.mean())
    dis_exp2 = 0.75/(1+np.exp((dis_exp2 - thres)/tau))
    

    dis_exp3 = np.sum(np.square(np.subtract(embeddings2, r_embeddings2)), 1)
    #print ('DIS3', dis_exp3.mean())
    dis_exp3 = 0.75/(1+np.exp((dis_exp3 - thres2)/tau))
    
    dis_exp1 = 2 - dis_exp2 - dis_exp3
    #print (dis_exp1.mean(), dis_exp2.mean(), dis_exp3.mean())

    all_dist = dist1*dis_exp1 + dist2*dis_exp2 + dist3*dis_exp3
    # print("Test shape of all dist matrix", all_dist.shape)
    # thresholds = np.arange(0, 10, 0.01)
    # tpr, fpr, accuracy, best_thresholds = calculate_roc_ex(thresholds, all_dist, issame)
    # auc = metrics.auc(fpr, tpr)

    # print ("Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'ICT_Ref'))

    # tpr, fpr, accuracy, best_thresholds = evaluate_new(embeddings1, q_embeddings1, issame, nrof_folds)
    # auc = metrics.auc(fpr, tpr)
    # print ("Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'inner and query inner'))

    # tpr, fpr, accuracy, best_thresholds = evaluate_new(embeddings2, q_embeddings2, issame, nrof_folds)
    # auc = metrics.auc(fpr, tpr)
    # print ("Evaluating {3} Acc:{0:.4f} Best_thres:{1:.4f} AUC:{2:.4f}".format( accuracy.mean(), best_thresholds.mean(), auc, 'outer and query outer'))

    prob_dict = {}
    label_dict = {}
    for i, vid in enumerate(all_videoIDs):
        if(vid in prob_dict.keys()):
                prob_dict[vid].append(all_dist[i])
                label_dict[vid].append(all_labels[i])
        else:
            prob_dict[vid] = []
            label_dict[vid] = []
            prob_dict[vid].append(all_dist[i])
            label_dict[vid].append(all_labels[i])

    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key]) # Aberage real prob
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

    np.save(f"{output_dir}/{data_name}/{model_name}_preds.npy", prob_list)
    np.save(f"{output_dir}/{data_name}/{model_name}_labels.npy", label_list)
    print("sum label list: ", sum(label_list))
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
        
    
    HTER = (FAR + FRR) / 2.0
    ACC = (TN + TP) / (TN + FN + FP + TP)
    print(f"HTER: {HTER*100:.2f}")
    print(f"FAR: {FAR*100:.2f}")
    print(f"TPR: {TPR*100:.2f}")
    print(f"ACC: {ACC*100:.2f}")

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

def evaluation(model, dataloader, fake_class, method, softmax=True, penul_ft=False, **kwargs):
    model_name = kwargs.get('model_name', 'None')
    output_dir = kwargs.get('output_dir', 'None')
    data_name = kwargs.get('data_name', 'None')

    prob_dict = {}
    label_dict = {}
    if penul_ft:
        ft_dict = {}
        all_frame_ft = []
        all_frame_lb = []
    print("Start eval ...")
    with torch.no_grad():
        for  (input, label, videoID) in tqdm(dataloader):
            input = Variable(input.float()).cuda()
            # print("Test", torch.min(input), torch.max(input))
            if method == 'capsule':
                input_x = model[0](Variable(input))
                classes, class_ = model[1](input_x, random=False)
                cls_out = class_.data.cpu()
            else:
                if not penul_ft:
                    cls_out = model(input)
                else:
                    cls_out, ft = model(input, return_ft=True)
                    ft = ft.cpu().data.numpy()
                    # print("Test shape feature MAT:", ft.shape)
                    all_frame_ft.append(ft)
                    all_frame_lb.append(label.cpu().data.numpy())


            if softmax:
                prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, fake_class] # Get fake prob
            else:
                prob = torch.sigmoid(cls_out).cpu().data.numpy()

            videoID = videoID.cpu().data.numpy()
            label = label.cpu().data.numpy()


            for i in range(len(prob)):
                if(videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    if penul_ft:
                        ft_dict[videoID[i]].append(label[i])
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])

                    if penul_ft:
                        ft_dict[videoID[i]] = []
                        ft_dict[videoID[i]].append(ft[i])

    prob_list = []
    label_list = []
    if penul_ft:
        ft_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key]) # Aberage real prob
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        if penul_ft:
            avg_single_video_ft = np.asarray(ft_dict[key]).sum(axis=0) / len(ft_dict[key])
            ft_list = np.append(ft_list, avg_single_video_ft)
        
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


    np.save(f"{output_dir}/{data_name}/{model_name}_preds.npy", prob_list)
    np.save(f"{output_dir}/{data_name}/{model_name}_labels.npy", label_list)
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

    if not penul_ft:
        return (ACC*100, ACC_best*100, AUC*100, precision*100, recall*100, f1_score*100)
    else:
        return (ACC*100, ACC_best*100, AUC*100, precision*100, recall*100, f1_score*100, np.concatenate(all_frame_ft,axis=0), np.concatenate(all_frame_lb,axis=0))

    
    