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

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        if key not in ["pixel_values", "image_grid_thw"]:
            tensors[key] = torch.stack(value, dim=0)

    return {**tensors, **non_tensors}


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key="prompt",
        max_prompt_length=1024,
        truncation="error",
        system_prompt=None,
        max_pixels=None,
        min_pixels=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # Support multiple datasets separated by comma
        if ',' in data_path:
            dataset_paths = [path.strip() for path in data_path.split(',')]
            datasets = []
            print(f"Loading {len(dataset_paths)} datasets...")
            for i, path in enumerate(dataset_paths):
                dataset = self._load_single_dataset(path)
                datasets.append(dataset)
                print(f"  [{i+1}/{len(dataset_paths)}] {path}: {len(dataset)} samples")

            self.dataset = concatenate_datasets(datasets)
            print(f"Total samples: {len(self.dataset)}")
        else:
            # Single dataset (original logic)
            self.dataset = self._load_single_dataset(data_path)

        # Set user prompt after loading dataset
        self.user_prompt = "<image>\n" \
            "Task: {Question}\n\n" \
            "Instructions:\n" \
            "1. This is a brain MRI scan. Look for abnormal regions that appear different from normal brain tissue.\n" \
            "2. Brain tumors typically appear as areas with altered intensity (brighter or darker regions) or irregular shapes.\n" \
            "3. Locate the tumor region and determine its 2D bounding box [x_min, y_min, x_max, y_max] and center point [x, y].\n" \
            "4. Output your analysis in <think></think> tags, then provide the final answer in <answer></answer> tags.\n\n" \
            "Output format example:\n" \
            "<think>Analysis of the image shows...</think>\n" \
            "<answer>{Answer}</answer>"

    def _load_single_dataset(self, data_path: str):
        """
        Load a single dataset, handling both DatasetDict and direct Dataset formats
        """
        import os

        try:
            # Try loading as DatasetDict first (original logic)
            dataset_dict = load_from_disk(data_path)
            if hasattr(dataset_dict, 'keys') and 'train' in dataset_dict:
                return dataset_dict['train']
            else:
                # If it's already a Dataset, return it directly
                return dataset_dict
        except Exception as e1:
            # If that fails, try loading path/train as a Dataset
            train_path = os.path.join(data_path, 'train')
            try:
                return load_from_disk(train_path)
            except Exception as e2:
                raise ValueError(
                    f"Failed to load dataset from {data_path}. "
                    f"Tried both DatasetDict format (error: {e1}) "
                    f"and Dataset format at {train_path} (error: {e2})"
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataset[index]
        
        ################ Old Version ################
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": self.user_prompt.format(Question=row_dict["problem"].lower().strip("."),
        #                                                         Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")},
        # ]
        ################ Old Version ################
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(
                Question=row_dict["problem"].lower().strip("."),
                Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [120,155]}]"
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["images"] = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict.update(image_inputs)

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt:
                    prompt = prompt.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)
        else:
            raw_prompt = prompt

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )  # (3, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict
