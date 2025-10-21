# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

from typing import Dict

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter


def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        for param_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            stacked_name = name.replace(shard_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[stacked_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # lm_head is not used in vllm as it is tied with embed_token.
            # To prevent errors, skip loading lm_head.weight.
            if "lm_head.weight" in name:
                continue
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def llama_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            continue
        # With tie_word_embeddings, we can skip lm_head.weight
        # The weight might appear unnecessarily in the files if the model is
        # processed with quantization, LoRA, fine-tuning, etc.
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight)


def qwen2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def qwen2vl_dtensor_weight_loader(actor_weights: Dict[str, torch.Tensor], vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (vllm_substr, hf_substr, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    vllm_params = dict(vllm_model.named_parameters(remove_duplicate=False))
    for actor_name, actor_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in actor_name:
            continue

        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_name:
            continue

        for vllm_substr, hf_substr, shard_id in stacked_params_mapping:
            if hf_substr not in actor_name:
                continue

            if "visual" in actor_name:
                continue

            vllm_name = "language_model." + actor_name.replace(hf_substr, vllm_substr)
            if actor_name.endswith(".bias") and actor_name not in vllm_params:
                continue  # skip loading extra bias for GPTQ models

            local_actor_weight = redistribute_dtensor(param_name=actor_name, loaded_weights=actor_weight)
            vllm_param = vllm_params[vllm_name]
            weight_loader = vllm_param.weight_loader
            weight_loader(vllm_param, local_actor_weight.to(dtype=vllm_param.dtype), shard_id)
            break
        else:
            if actor_name.endswith(".bias") and actor_name not in vllm_params:
                continue  # skip loading extra bias for GPTQ models

            if "visual" in actor_name:
                vllm_name = actor_name
            else:
                vllm_name = "language_model." + actor_name

            # Check if parameter exists in vllm model before loading
            if vllm_name not in vllm_params:
                # For visual parameters, the structure might differ between Qwen2/2.5/3-VL
                # Skip if not found - this is expected for vision encoder parameters
                # which may have different structures or be handled separately
                if "visual" in actor_name:
                    # Silently skip visual params that don't match
                    # (vllm may have its own vision encoder structure)
                    continue
                else:
                    raise KeyError(f"Parameter {vllm_name} not found in vllm model. Available params: {list(vllm_params.keys())[:10]}...")

            vllm_param = vllm_params[vllm_name]
            local_actor_weight = redistribute_dtensor(param_name=actor_name, loaded_weights=actor_weight)
            weight_loader = getattr(vllm_param, "weight_loader", default_weight_loader)
            weight_loader(vllm_param, local_actor_weight.to(dtype=vllm_param.dtype))


def qwen3vl_dtensor_weight_loader(actor_weights: Dict[str, torch.Tensor], vllm_model: nn.Module) -> nn.Module:
    """
    Weight loader for Qwen3-VL models.

    vLLM parameter mapping (from qwen3_vl.py:1104-1109):
    - HF: model.language_model.* → vLLM: language_model.model.*
    - HF: model.visual.* → vLLM: visual.*
    - HF: lm_head.* → vLLM: language_model.lm_head.*
    """
    stacked_params_mapping = [
        # (vllm_substr, hf_substr, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    vllm_params = dict(vllm_model.named_parameters(remove_duplicate=False))

    for actor_name, actor_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in actor_name:
            continue

        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_name:
            continue

        # Apply vLLM's weight mapping rules
        # Reference: vllm/model_executor/models/qwen3_vl.py WeightsMapper
        if actor_name.startswith("model.visual."):
            # model.visual.* -> visual.*
            base_name = actor_name.replace("model.visual.", "visual.")
        elif actor_name.startswith("model.language_model."):
            # model.language_model.* -> language_model.model.*
            base_name = actor_name.replace("model.language_model.", "language_model.model.")
        elif actor_name.startswith("lm_head."):
            # lm_head.* -> language_model.lm_head.*
            base_name = actor_name.replace("lm_head.", "language_model.lm_head.")
        else:
            base_name = actor_name

        # Handle stacked parameters (qkv_proj, gate_up_proj)
        for vllm_substr, hf_substr, shard_id in stacked_params_mapping:
            if hf_substr not in base_name:
                continue

            if "visual" in base_name:
                continue  # visual params don't use stacked projection

            vllm_name = base_name.replace(hf_substr, vllm_substr)
            if base_name.endswith(".bias") and vllm_name not in vllm_params:
                continue  # skip loading extra bias for GPTQ models

            if vllm_name not in vllm_params:
                continue  # parameter not in vllm model

            local_actor_weight = redistribute_dtensor(param_name=actor_name, loaded_weights=actor_weight)
            vllm_param = vllm_params[vllm_name]
            weight_loader = vllm_param.weight_loader
            weight_loader(vllm_param, local_actor_weight.to(dtype=vllm_param.dtype), shard_id)
            break
        else:
            # Direct parameter mapping (non-stacked)
            if base_name.endswith(".bias") and base_name not in vllm_params:
                continue  # skip loading extra bias for GPTQ models

            vllm_name = base_name

            # Check if parameter exists in vllm model before loading
            if vllm_name not in vllm_params:
                # For visual parameters, the structure might differ
                # Skip if not found - this is expected for some vision encoder parameters
                if "visual" in base_name:
                    # Silently skip visual params that don't match
                    continue
                else:
                    raise KeyError(f"Parameter {vllm_name} not found in vllm model. Available params: {list(vllm_params.keys())[:10]}...")

            vllm_param = vllm_params[vllm_name]
            local_actor_weight = redistribute_dtensor(param_name=actor_name, loaded_weights=actor_weight)
            weight_loader = getattr(vllm_param, "weight_loader", default_weight_loader)
            weight_loader(vllm_param, local_actor_weight.to(dtype=vllm_param.dtype))


def deepseekv2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=vllm_model.config.n_routed_experts,
    )

    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, vllm_model):
                continue

            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, vllm_model):
                    continue

                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    local_loaded_weight.to(dtype=param.dtype),
                    weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, vllm_model):
                    continue

                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def redistribute_dtensor(param_name: str, loaded_weights: DTensor, parallelize_plan: Dict = None):
    param_name = _process_parameter_names(name=param_name)
    if parallelize_plan is not None:
        assert param_name in parallelize_plan.keys(), (
            f"param name: {param_name} not in parallelize_plan :{parallelize_plan.keys()}"
        )
        placement = parallelize_plan[param_name]
        local_loaded_weights = loaded_weights.redistribute(
            device_mesh=loaded_weights.device_mesh, placements=placement
        ).to_local()
    else:
        local_loaded_weights = loaded_weights.full_tensor()

    return local_loaded_weights


def _process_parameter_names(name):
    # Remove '.weight' if it exists at the end of the string
    if name.endswith(".weight"):
        name = name[:-7]

    # Remove 'model.layers.x.' or 'model.' prefix
    if "model.layers" in name:
        parts = name.split(".")
        # Reconstruct the string without 'model.layers.x.'
        name = ".".join(parts[3:])  # parts[0] is 'model', parts[1] is 'layers', parts[2] is 'x'
    elif name.startswith("model."):
        name = name[6:]  # Remove 'model.'

    return name


__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__ = {
    "LlamaForCausalLM": llama_dtensor_weight_loader,
    "LLaMAForCausalLM": llama_dtensor_weight_loader,
    "MistralForCausalLM": llama_dtensor_weight_loader,  # mistral is the same as llama in vLLM
    "InternLMForCausalLM": llama_dtensor_weight_loader,
    "Phi3ForCausalLM": llama_dtensor_weight_loader,
    "GemmaForCausalLM": gemma_dtensor_weight_loader,
    "Gemma2ForCausalLM": gemma_dtensor_weight_loader,
    "Qwen2ForCausalLM": qwen2_dtensor_weight_loader,
    "DeepseekV2ForCausalLM": deepseekv2_dtensor_weight_loader,
    "Qwen2VLForConditionalGeneration": qwen2vl_dtensor_weight_loader,
    "Qwen2_5_VLForConditionalGeneration": qwen2vl_dtensor_weight_loader,
    "Qwen3VLForConditionalGeneration": qwen3vl_dtensor_weight_loader,  # Qwen3-VL has different parameter structure
}


# the actor model is .state_dict()
# Load dtensor weights
def load_dtensor_weights(actor_weights: Dict, vllm_model: nn.Module):
    # Handle CUDAGraphWrapper - unwrap to get the actual model
    actual_model = vllm_model
    model_class_name = vllm_model.__class__.__name__

    if model_class_name == "CUDAGraphWrapper":
        # CUDAGraphWrapper stores the actual model in 'runnable' attribute
        # It also provides an unwrap() method
        if hasattr(vllm_model, "unwrap"):
            actual_model = vllm_model.unwrap()
        elif hasattr(vllm_model, "runnable"):
            actual_model = vllm_model.runnable
        else:
            raise ValueError(
                f"CUDAGraphWrapper detected but cannot find the wrapped model. "
                f"Expected 'runnable' attribute or 'unwrap()' method."
            )
        model_class_name = actual_model.__class__.__name__

    weight_loader = _get_model_weight_loader(model_class_name)
    weight_loader(actor_weights, actual_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(
        f"Model architectures {arch} are not supported for now. "
        f"Supported architectures: {__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__.keys()}"
    )


# NOTE(sgm): we use per-parameter weight loader in each vllm sub
def update_dtensor_weight_loader():
    pass
