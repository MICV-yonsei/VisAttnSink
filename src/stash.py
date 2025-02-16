import os
import os.path as osp
import pickle
from typing import List
from collections import defaultdict
import torch
from typing import Optional

class DefaultConfig:
    model_config = {
        "llm_name": "llama-v2-7b",
        "config": None,
        "num_hidden_layers": None,
        "num_attention_heads": None,
    }
    
    state = {
        "logic_flag": {},
        "save_to_path": {},
        "vis_sink_token_idx": defaultdict(list),
        "vis_non_sink_token_idx": defaultdict(list),
    }
    
    metadata = {
        "vis_len": 0,
        "qid": None,
        "output_token_count": -1,
        "current_decoder_layer": 0,
    }
    
    segments = {
        "id_pieces": {
            "system": [], "role_0": [], "image": [],
            "inst_q": [], "role_1": []
        },
        "text_pieces": {
            "system": [], "role_0": [], "image": [],
            "inst_q": [], "role_1": []
        },
        "begin_pos": {
            "system": float("-inf"), "role_0": float("-inf"),
            "image": float("-inf"), "inst_q": float("-inf"),
            "role_1": float("-inf")
        }
    }

class StashEngine:
    default_config = DefaultConfig()
    model_config = default_config.model_config
    state = default_config.state
    metadata = default_config.metadata
    segments = default_config.segments
    
    @classmethod
    def activate(cls):
        cls.set_flag(True)
        print(f"{cls.__name__} flag set to {cls._flag()}")

    @classmethod
    def export_model_config(cls, config):
        cls.model_config['config'] = config
        cls.model_config["num_hidden_layers"] = config.num_hidden_layers
        cls.model_config["num_attention_heads"] = config.num_attention_heads

    @classmethod
    def set_flag(cls, flag_value=True):
        cls.state["logic_flag"][cls.__name__] = flag_value

    @classmethod
    def _flag(cls, name=None):
        if name is not None:
            return cls.state["logic_flag"].get(name, False)
        return cls.state["logic_flag"].get(cls.__name__, False)

    @classmethod
    def set_save_to_path(cls, save_to_path):
        cls.state["save_to_path"][cls.__name__] = save_to_path

    @classmethod
    def get_save_to_path(cls):
        return cls.state["save_to_path"].get(cls.__name__, None)

    @classmethod
    def run_logic(cls):
        raise NotImplementedError("Running basic logic in LogicEngine.")

    @classmethod
    def clear(cls):
        _default_config = DefaultConfig()
        cls.model_config = _default_config.model_config
        cls.state = _default_config.state
        cls.metadata = _default_config.metadata
        cls.segments = _default_config.segments

    @classmethod
    def save_to_pickle(cls, path, data):
        if osp.exists(path):
            print(f"File already exists at {path}.")
            return
            
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
    @classmethod 
    def detach_to_cpu(cls, tensor_or_dict):
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.detach().cpu()
        elif isinstance(tensor_or_dict, dict):
            return {k: cls.detach_to_cpu(v) for k,v in tensor_or_dict.items()}
        return tensor_or_dict


class ValueMonitor(StashEngine):
    @classmethod
    def remember_layer(cls, l):
        cls.__base__.metadata["current_decoder_layer"] = l

    @classmethod
    def count_output_token(cls):
        cls.__base__.metadata["output_token_count"] += 1

    @classmethod
    def remember_seg_token_order(cls, n):
        cls.__base__.seg_token_order.append(n)

    @classmethod
    def remember_qid(cls, qid):
        cls.__base__.metadata["qid"] = qid

    @classmethod
    def get_output_token_count(cls):
        return cls.__base__.metadata["output_token_count"]

class MetadataStation(StashEngine):
    @classmethod
    def set_image_path(cls, image_path):
        cls.__base__.state["image_path"] = image_path

    @classmethod
    def set_qid(cls, qid):
        cls.__base__.state["qid"] = qid

    @classmethod
    def get_metadata(cls):
        """
        Returns the metadata of the class.
        """
        return {
            "prompt": cls.state["prompt"],
            "gt_label": cls.state["gt_label"],
            "answer": cls.state["answer"],
            "answer_ids": cls.state["answer_ids"],
            "id_pieces": cls.segments["id_pieces"],
            "text_pieces": cls.segments["text_pieces"],
            "begin_pos": cls.segments["begin_pos"],
            "vis_len": cls.metadata["vis_len"],
            "image_path": cls.state["image_path"],
        }

    @classmethod
    def set_correct(cls, correct):
        cls.__base__.state["correct"] = correct

    @classmethod
    def set_prompt(cls, prompt):
        cls.__base__.state["prompt"] = prompt

    @classmethod
    def set_answer(cls, answer_ids, answer):
        if isinstance(answer_ids, torch.Tensor):
            answer_ids = answer_ids.cpu().clone()

        cls.__base__.state["answer_ids"] = answer_ids
        cls.__base__.state["answer"] = answer

    @classmethod
    def set_gt_label(cls, gt_label):
        cls.__base__.state["gt_label"] = gt_label

    @classmethod
    def set_vis_len(cls, vis_len):
        cls.__base__.metadata["vis_len"] = vis_len

    @classmethod
    def set_begin_pos(cls, key, idx):
        cls.__base__.segments["begin_pos"][key] = idx

    @classmethod
    def set_id_pieces(cls, key, ids: List[int]):
        assert isinstance(ids, list)
        cls.__base__.segments["id_pieces"][key] = ids

    @classmethod
    def set_text_pieces(cls, key, texts: List[str]):
        assert isinstance(texts, list)
        cls.__base__.segments["text_pieces"][key] = texts

    @classmethod
    def save_metadata(cls, save_to_path, **kwargs):
        if osp.exists(save_to_path):
            print(f"File already exists at {save_to_path}.")
            return
        metadata = cls.get_metadata()
        metadata.update(kwargs)
        with open(save_to_path, "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def segment_prompt(
        cls, 
        tokenizer, 
        query, 
        roles,
        IMG_CONTEXT_STRING = "<image>",
        IMAGE_TOKEN_INDEX = None,
        is_llava=False,
        return_pt=False
    ):
        class TokenizeSegment:
            def __init__(self, tokenizer, is_llava=False):
                self.tokenizer = tokenizer
                self.is_llava = is_llava
                self.first_29871_found = False

            def __call__(self, text: str, add_special_tokens: bool = False, 
                        image_token_idx_not_added_in_tokenizer: Optional[int] = None) -> List[int]:
                if image_token_idx_not_added_in_tokenizer is not None:
                    return [image_token_idx_not_added_in_tokenizer]
                
                ret = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt")[0].tolist()
                
                if self.is_llava:
                    # Filter out duplicate 29871 (whitespace) tokens, keeping only the first occurrence
                    filtered_ret = []
                    for token in ret:
                        if token != 29871:
                            filtered_ret.append(token)
                        elif not self.first_29871_found:
                            filtered_ret.append(token)
                            self.first_29871_found = True
                    ret = filtered_ret
                return ret

        segments = {}
        
        role_0, role_1 = roles
        if role_0 == "" and role_1 == "":
            raise NotImplementedError("role_0 and role_1 are empty")

        else:
            # Split into system and user parts
            parts = query.split(role_0, 1)
            if len(parts) == 2:
                segments["system"] = parts[0].strip('')
                user_and_assistant = role_0 + parts[1]
            else:
                segments["system"] = query.strip('')
                user_and_assistant = ""

            # Split user and assistant parts
            parts = user_and_assistant.split(role_1, 1) 
            if len(parts) == 2:
                user_text = parts[0]
                segments["role_1"] = (role_1 + parts[1]).strip('')
            else:
                user_text = user_and_assistant
                segments["role_1"] = ""

            # Handle image context if present
            if IMG_CONTEXT_STRING in user_text:
                before_img, after_img = user_text.split(IMG_CONTEXT_STRING, 1)
                segments["role_0"] = before_img.strip('')
                segments["image"] = IMG_CONTEXT_STRING
                segments["inst_q"] = after_img.strip('')
            else:
                segments["role_0"] = user_text.strip('')
                segments["image"] = None
                segments["inst_q"] = ""

            # TODO : image position is matter
            tokenize_segment = TokenizeSegment(tokenizer, is_llava)
            tokenized_segments = {
                "system": tokenize_segment(segments["system"], False),
                "role_0": tokenize_segment(segments["role_0"], False),
                "image": tokenize_segment(segments["image"], False, image_token_idx_not_added_in_tokenizer=IMAGE_TOKEN_INDEX),
                "inst_q": tokenize_segment(segments["inst_q"], False),
                "role_1": tokenize_segment(segments["role_1"], False),
            }

            merge_tokenized_segments = []
            for l in list(tokenized_segments.values()):
                merge_tokenized_segments.extend(l)

            # set metadata; id_pieces 
            for key, value in tokenized_segments.items():
                cls.set_id_pieces(key=key, ids=value)
            
            token_id_range = {
                "system": (tokenized_segments["system"][0], tokenized_segments["role_0"][0]),
                "role_0": (tokenized_segments["role_0"][0], tokenized_segments["image"][0]),
                "image": (tokenized_segments["image"][0], tokenized_segments["inst_q"][0]),
                "inst_q": (tokenized_segments["inst_q"][0], tokenized_segments["role_1"][0]),
                "role_1": (tokenized_segments["role_1"][0], -1),
            }

        # set metadata; begin_pos and text_pieces
        begin_pos = 0
        for key, token_ids in tokenized_segments.items():
            cls.set_begin_pos(key=key, idx=begin_pos)
            if key == "image":
                end_pos = begin_pos + cls.metadata["vis_len"]
                if IMAGE_TOKEN_INDEX is not None:
                    cls.set_text_pieces(key=key, texts=["<image>"])
                else:
                    cls.set_text_pieces(key=key, texts=tokenizer.batch_decode(token_ids))
            else:
                end_pos = begin_pos + len(token_ids)
                cls.set_text_pieces(key=key, texts=tokenizer.batch_decode(token_ids))
            begin_pos = end_pos

        assert len(merge_tokenized_segments) == sum(map(len, tokenized_segments.values())), "token_ids are not matched in MetadataStation.prompt_segments and input_ids"
        
        if return_pt:
            merge_tokenized_segments = torch.tensor(merge_tokenized_segments).unsqueeze(0)
        
        return token_id_range, merge_tokenized_segments