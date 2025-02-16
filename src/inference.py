import argparse
import gc
import json
import math
import os
import os.path as osp
from pprint import pprint
import pickle
import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__))))

import time
from types import SimpleNamespace

import torch
from PIL import Image
from tqdm import tqdm
import yaml

from src.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                             DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from src.utils import disable_torch_init

from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):

    with open(args.exp_config, "r") as file:
        config_dict = yaml.safe_load(file)
    cfgs = SimpleNamespace(**config_dict)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    cfgs.device = device
    
    print("\n\n\n")
    print(f"Using device: {torch.cuda.get_device_name()}-{args.device}")
    pprint(vars(cfgs))
    print("\n\n\n")

    disable_torch_init()
    path_model = os.path.expanduser(cfgs.path_model)
    name_model = get_model_name_from_path(path_model)
    cfgs.name_model = name_model
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        path_model, 
        args.model_base, 
        name_model, 
        attn_implementation="eager", 
        device_map=device
    )

    # Activate StashEngine
    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)
    
    # Activate logic
    if getattr(cfgs, "logic", 0) == 1:
        LogicEngine.activate(tau=cfgs.tau, rho=cfgs.rho, summ=cfgs.summ, p=cfgs.p, except_last_layer=cfgs.except_last_layer)

    # Load questions and open answers file
    question_file_path = osp.join(cfgs.path_question_dir, f"{cfgs.name_category}-questions.jsonl" if cfgs.name_category != "" else "questions.jsonl")
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file_path), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answer_file_ver = f"[{cfgs.name_daset}-{cfgs.name_category}]{cfgs.name_exp}-{str(int(time.time()))}"
    answers_file = osp.join("E_answers", cfgs.name_model, f"{answer_file_ver}.jsonl")
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(osp.dirname(answers_file), exist_ok=True)

    file_mode = "w"
    with open(answers_file, file_mode) as ans_file:
        setattr(model, "tokenizer", tokenizer)
        for line in tqdm(questions):
            try:    
                qid = int(line.get("qid", None) or line.get("question_id", None))
                gt_label = line.get("label", None) or line.get("answer", None) or line.get("gt-label", None)

                image_file = line["image"]
                _, img_ext = osp.splitext(image_file)
                if img_ext is None or img_ext == "":
                    image_file = f"{image_file}.jpg"

                qs = line.get("text") or line.get("question") or line.get("prompt")
                assert qs is not None
                cur_prompt = qs
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[cfgs.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # Begin inference
                input_ids = tokenizer_image_token(
                    prompt, 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    conv=conv, 
                    return_tensors="pt"
                ).unsqueeze(0).to(device=device)
                image = Image.open(os.path.join(cfgs.path_image_dir, image_file)).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0]
                with torch.inference_mode():
                    with torch.no_grad():
                        setattr(model, "tokenizer", tokenizer)
                        outputs = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().to(device),
                            image_sizes=[image.size],
                            return_dict_in_generate=True,
                            output_attentions=True,
                            output_hidden_states=True,
                            do_sample=False,
                            max_new_tokens=cfgs.max_new_tokens,
                            use_cache=True,
                        )
                                        
                generated_texts = tokenizer.batch_decode(outputs.sequences)[0]

                ans_file.write(json.dumps(
                    {"question_id": qid, 
                     "prompt": cur_prompt, 
                     "label": gt_label, 
                     "response": generated_texts, 
                     "image": image_file, 
                     "model_id": name_model}
                ) + "\n")
                ans_file.flush()
                
                del outputs
                StashEngine.clear()
                LogicEngine.clear()
            except Exception as e:
                raise e

def parse_ranges(range_string):
    st = [int(num) for num in range_string.split("-")][:-1]
    ed = [int(num) for num in range_string.split("-")][1:]
    return [[st, ed] for st, ed in zip(st, ed)]


if __name__ == "__main__":

    if sys.argv[-1] in ["debug", "--debug"]:
        import debugpy
        debugpy.listen(("localhost", 7739))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        sys.argv.pop()
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=int, default=None)

    # logic
    parser.add_argument("--logic", type=str, default=None)
    parser.add_argument("--dim_prospector", action="store_true")
    parser.add_argument("--head_fork", action="store_true")
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--sink_rule", type=str, default="ours")
    parser.add_argument("--head_rule", type=str, default="ours")
    
    # exp config
    parser.add_argument("--exp_config", type=str, default=None)
    args = parser.parse_args()
    
    eval_model(args)
