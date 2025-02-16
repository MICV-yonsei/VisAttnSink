import json
import os
import os.path as osp
import pickle
import numpy as np
import argparse
from pprint import pprint
import sys

def parse_args():
    parser = argparse.ArgumentParser()

    # num_layers and heads
    parser.add_argument("--num-layers", nargs='+', type=int, default=[])
    parser.add_argument("--num-heads", nargs='+', type=int, default=[])
    
    # path
    parser.add_argument("--path-label", type=str, required=True)
    parser.add_argument("--path-answer", type=str, required=True)
    parser.add_argument("--path-save", type=str, required=True)

    return parser.parse_args()


def eval_and_save_pope(args, answer_list, label_list, save_name=None):

    # save pickle
    args.path_save = os.path.splitext(args.path_save)[0]
    save_to_path = osp.join(args.path_save, save_name)
    os.makedirs(save_to_path, exist_ok=True)
    # if osp.exists(osp.join(save_to_path, "pred.pkl")):
    #     print("File already exists. Exiting...")
    #     return

    for answer in answer_list:
        text = answer["text"]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            answer["text"] = "no"
        else:
            answer["text"] = "yes"

    label_pack = []
    for i in range(len(label_list)):
        try:
            if label_list[i]['gt-label'] == "no":
                label_pack.append(0)
            else:
                label_pack.append(1)
        except:
            print(f"Error in qid {i}")
            continue

    pred_pack = []
    for answer in answer_list:
        try:
            if answer["text"] == "no":
                pred_pack.append(0)
            else:
                pred_pack.append(1)
        except:
            print(f"Error in qid {i}")
            continue

    pos = 1
    neg = 0
    yes_ratio = pred_pack.count(1) / len(pred_pack)

    TP, TN, FP, FN = 0, 0, 0, 0
    for q, (pred, label) in enumerate(zip(pred_pack, label_pack)):
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

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    print("TP\tFP\tTN\tFN\t")
    # print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))
    print(f"{'':<10} {'Predicted Pos':<20} {'Predicted Neg':<20}")
    print("-" * 50)
    print(f"{'Actual Pos':<20} {TP:<20} {FN:<20}")
    print(f"{'Actual Neg':<20} {FP:<20} {TN:<20}")
    print("-" * 50)    
    print(f"{'Accuracy:':<10} {acc:.4f}")
    print(f"{'Precision:':<10} {precision:.4f}")
    print(f"{'Recall:':<10} {recall:.4f}")
    print(f"{'F1 score:':<10} {f1:.4f}")
    print(f"{'Yes ratio:':<10} {yes_ratio:.4f}")
    print("-" * 50)    
    # print("%.3f, %.3f, %.3f, %.3f, %.3f" % (f1, acc, precision, recall, yes_ratio))

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
    with open(osp.join(save_to_path, "pred.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    max_key_length = max(len(key) for key in args_dict)
    print(f"{'Argument':<{max_key_length}}  Value")
    print("-" * (max_key_length + 8))
    for key, value in args_dict.items():
        print(f"{key:<{max_key_length}}  {value}")
    print("-" * (max_key_length + 8))

    layers = args.num_layers
    heads = args.num_heads
    label_list = [json.loads(q) for q in open(args.path_label, "r")][:300]
    for l in layers:
        for h in heads:
            path_answer = osp.join(args.path_answer, f'{l}-{h}.jsonl')
            if osp.exists(path_answer) is False:
                print(f"File does not exist: {path_answer}")
                continue
            answer_list = [json.loads(q) for q in open(path_answer, "r")]
            if len(answer_list) != len(label_list):
                print("Error: The number of answers does not match the number of labels")
                continue
            eval_and_save_pope(args, answer_list, label_list, save_name=f"{l}-{h}")
