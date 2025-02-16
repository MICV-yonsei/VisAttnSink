import torch
from collections import defaultdict
from .constants import DIM_SINK, MODEL_LLM
import copy
from src.stash import MetadataStation, ValueMonitor


class DefaultVars:
    # Model Config
    llm_name = "llama-v2-7b"
    logic_flag = True
    
    # DimProspector
    indices = defaultdict(list)
    dim_sink = DIM_SINK[llm_name]
    
    # HeadFork
    forked_head = {}
    forked_head_per_token = defaultdict(dict)
    

class LogicEngine:
    defaultVars = DefaultVars()
    logic_flag = defaultVars.logic_flag
    llm_name = defaultVars.llm_name
    indices = defaultVars.indices
    forked_head = defaultVars.forked_head
    forked_head_per_token = defaultVars.forked_head_per_token
    dim_sink = defaultVars.dim_sink
    
    @classmethod
    def activate(
        cls, 
        tau=20, 
        rho=0.5, 
        summ=0.2, 
        p=0.6, 
        except_last_layer=True, 
        layer='all'
    ):
        cls.set_flag(True)
        cls.tau = tau
        cls.rho = rho
        cls.summ = summ
        cls.p = p
        cls.except_last_layer = except_last_layer
        cls.set_sink_select_layers(layer)

    @classmethod
    def set_sink_select_layers(cls, layer='all'):
        if layer == 'all':
            cls.sink_select_layers = [i for i in range(MetadataStation.model_config["num_hidden_layers"])][2:]
        else:
            assert isinstance(layer, int)
            cls.sink_select_layers = layer

    @classmethod
    def set_llm_name(cls, model_name):
        for body, llm in MODEL_LLM.items():
            if model_name in body:
                cls.llm_name = llm
                break

    @classmethod
    def set_flag(cls, flag_value=True):
        if cls.logic_flag is True:
            return
        else:
            cls.logic_flag = flag_value

    @classmethod
    def _flag(cls, name=None):
        if cls.logic_flag is True:
            return True
        else:
            return False

    @classmethod
    def run_logic(cls):
        raise NotImplementedError("Running basic logic in LogicEngine.")

    @classmethod
    def clear(cls):
        _defaultVars = DefaultVars()
        cls.llm_name = _defaultVars.llm_name
        cls.dim_sink = _defaultVars.dim_sink
        cls.logic_flag = True if cls.logic_flag is True else cls.logic_flag
        cls.indices = _defaultVars.indices
        cls.forked_head = _defaultVars.forked_head
        cls.forked_head_per_token = _defaultVars.forked_head_per_token
        
class DimProspector(LogicEngine):
    
    @classmethod
    def fix_dim(cls, llm_name="llama-7b"):
        print(f"{cls.__name__} fixed dims: {cls.dim_sink[llm_name]}")
    
    @classmethod
    def rmsnorm(cls, hidden_states, eps=1e-6):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + eps)

    @classmethod
    def run_logic(cls, hs, layer):
        rms_norm_hs = torch.abs(cls.rmsnorm(hs)) # [bsz, tok, dim]
        rms_values = torch.stack([rms_norm_hs[:, :, idx] for idx in cls.__base__.dim_sink], dim=-1) # [bsz, tok, 2]
        max_rms_values = torch.max(rms_values, dim=-1)[0] # [bsz, tok]
        indices = torch.nonzero(max_rms_values > cls.tau)[:, 1] # [batch_axis, token_axis] -> [token_axis]
        cls.__base__.indices[layer]=indices # []

class HeadFork(LogicEngine):

    @classmethod
    def run_logic(cls, attn, layer_idx):
        layer = layer_idx
        if isinstance(cls.sink_select_layers, list):
            sink_inds = cls.indices[layer] # [1,...]
        else:
            sink_inds = cls.indices[cls.sink_select_layers]

        im, pa = MetadataStation.segments['begin_pos']['image'], MetadataStation.metadata['vis_len']
        vis_sink_inds = [i.unsqueeze(0) for i in sink_inds if im <= i < im+pa] # [1, 2, 3, ... ]

        if len(vis_sink_inds) > 0:
            vis_sink_inds = torch.cat(vis_sink_inds, dim=0) # shape: torch.Size([n])
            image_attn = attn[:, :, :, im:im+pa]

            portion = torch.sum(image_attn[:, :, :, vis_sink_inds-im], dim=-1) / torch.sum(image_attn + 1e-6, dim=-1) # [bsz, head, query]
            summation = torch.sum(image_attn, dim=-1)  # [bsz, head, query]

            # Condition 1. Portion <= rho
            portion_condition = portion <= cls.rho

            # Condition 2. Summation >= summ
            summation_condition = summation >= cls.summ

            candidate_coords = torch.nonzero( portion_condition & summation_condition )
            cls.__base__.forked_head[layer] = candidate_coords.clone()
        else:
            cls.__base__.forked_head[layer] = []
        
        cls.__base__.forked_head_per_token[ValueMonitor.get_output_token_count()][layer] = cls.__base__.forked_head[layer]
        
        return

class VARProcessor(LogicEngine):
    except_last_layer = False

    @classmethod
    def config_last_layer(cls, flag):
        cls.except_last_layer = flag

    @classmethod
    def set_selected_token(cls, selected_tokens):
        cls.selected_tokens = selected_tokens

    @classmethod
    def check_target_layer(cls):
        return cls.TARGET_LAYERS

    @classmethod
    def attn_redist(cls, attention_map, layer_idx):
        """
        attention map: # torch.Size([1, 32, 632(1), 576])
        """
        p = cls.p
        
        if cls.except_last_layer and cls.current_decoder_layer == cls.model_config.num_hidden_layers - 1:
            return attention_map    
    
        im, pa = MetadataStation.segments["begin_pos"]["image"], MetadataStation.metadata["vis_len"]
        coord = HeadFork.forked_head[layer_idx]
        indices = cls.__base__.indices[layer_idx]
        if len(coord) > 0: # if catched a valid head for query tokens
            model_head_num = MetadataStation.model_config["num_attention_heads"]
            for h in range(model_head_num):
                query_coord = coord[coord[:, 1]==h][:,2] # size: [bsz, H, Q] -> [H, Q]
                query_coord = query_coord[im+pa<=query_coord] if ValueMonitor.get_output_token_count() < 0 else query_coord # TODO
                bsz_coord = coord[coord[:, 1] == h][:,0][:len(query_coord)]
                head_coord = coord[coord[:, 1]==h][:,1][:len(query_coord)]

                if not query_coord.shape[0] or not head_coord.shape[0]:
                    continue

                # Attention map selection & split sink token indices
                selected_attn_map = attention_map[bsz_coord, head_coord, query_coord, :].clone()
                indices = indices.to(selected_attn_map.device)
                vis_indices = indices[(im<=indices) & (indices<im+pa)]
                text_indices = indices[~torch.isin(indices, vis_indices)]

                # Copy only the attention map for manipulation.
                copied_attention_map = copy.deepcopy(selected_attn_map.detach())  # [Q, K]

                # Decrease the portion of the selected_attn_map corresponding to the sink token by p.
                selected_attn_map[:, text_indices] *= p
                selected_attn_map[:, vis_indices] *= p 

                # Calculate the attention weight of some sink tokens that can be distributed. (1-p)
                weight_budget_vis = copied_attention_map[:, vis_indices].sum(dim=1) * (1 - p)
                weight_budget_text = copied_attention_map[:, text_indices].sum(dim=1) * (1 - p)

                # Set all attention weights corresponding to the sink token to 0. (to get the ratio of non-sink token values)
                copied_attention_map[:, vis_indices] *= 0  

                # Find the weight ratio of the un-sink tokens to the vision tokens.
                ratios_vis = copied_attention_map[:, im:im+pa] / copied_attention_map[:, im:im+pa].sum(dim=1, keepdim=True).to(selected_attn_map.dtype)                

                # Combines the vision and text budget and allocates it to the vision token.
                selected_attn_map[:, im:im+pa] += (weight_budget_vis + weight_budget_text).view(-1,1) * ratios_vis
                attention_map[bsz_coord, head_coord, query_coord, :] = selected_attn_map
        return attention_map