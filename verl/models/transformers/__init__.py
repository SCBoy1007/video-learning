# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

from . import qwen2_5_vl
from . import qwen3_vl


def get_rope_index_for_model(model_type: str):
    """
    Get the appropriate rope index function for the specified model type.

    Args:
        model_type: Model identifier ('qwen2.5vl', 'qwen3vl', etc.)

    Returns:
        The rope index generation function for the model
    """
    if model_type == "qwen3vl":
        return qwen3_vl.get_rope_index
    elif model_type == "qwen2.5vl":
        return qwen2_5_vl.get_rope_index
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: qwen2.5vl, qwen3vl")


__all__ = ["qwen2_5_vl", "qwen3_vl", "get_rope_index_for_model"]
