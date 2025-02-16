import json
import os
import os.path as osp
import pickle
import numpy as np
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    
    # path
    args.add_argument("--path-label", type=str, required=True)
    args.add_argument("--path-answer", type=str, required=True)
    args.add_argument("--path-save", type=str, required=True)
    return args

def eval_and_save_pope():
    args = parse_args()

    label_list = [json.loads(q)["answer"] for q in open(args.path_label, "r")]
    answer_list = [json.loads(q)["answer"] for q in open(args.path_answer, "r")]

    for answer in answer_list:
        text = answer["text"][0]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            answer["text"] = "no"
        else:
            answer["text"] = "yes"

    for i in range(len(label_list)):
        try:
            if label_list[i] == "no":
                label_list[i] = 0
            else:
                label_list[i] = 1
        except:
            print(f"Error in qid {i}")
            continue

    pred_list = []
    for answer in answer_list:
        try:
            if answer["text"] == "no":
                pred_list.append(0)
            else:
                pred_list.append(1)
        except:
            print(f"Error in qid {i}")
            continue

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for q, (pred, label) in enumerate(zip(pred_list, label_list)):
        qid = label_list[q]["question_id"]
        try:
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1
        except:
            print(f"Error in qid {qid}")
            continue

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))
    print("%.3f, %.3f, %.3f, %.3f, %.3f" % (f1, acc, precision, recall, yes_ratio))

    results = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
    }

    # save pickle
    save_to_path = osp.join('transcripts', 'pope', args.path_save)
    os.makedirs(save_to_path, exist_ok=True)
    with open(osp.join(save_to_path, 'pred.pkl'), "wb") as f:
        pickle.dump(results, f)

def pope_result(answer, label):
    # Only keep the first sentence
    if answer.find('.') != -1:
        answer = answer.split('.')[0]

    answer = answer.replace(',', '')
    words = answer.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        answer = 'no'
    else:
        answer = 'yes'

    if answer == 'no':
        answer = 0
    else:
        answer = 1

    if label == 'no':
        label = 0
    else:
        label = 1

    if answer == label:
        return 1
    else:
        return 0

if __name__ == "__main__":
    eval_and_save_pope()