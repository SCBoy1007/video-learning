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
from verl.models.transformers import get_rope_index_for_model


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
        # Don't stack vision inputs and mrope_position_deltas (may have variable shapes)
        if key not in ["pixel_values", "image_grid_thw", "mrope_position_deltas"]:
            tensors[key] = torch.stack(value, dim=0)

    # Special handling for mrope_position_deltas (may be None for text-only, or variable shape)
    if "mrope_position_deltas" in tensors:
        # Stack if all are tensors (vision inputs)
        tensors["mrope_position_deltas"] = torch.stack(tensors["mrope_position_deltas"], dim=0)

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
        model_type="qwen2.5vl",  # "qwen2.5vl" or "qwen3vl"
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.model_type = model_type

        # Get the appropriate rope_index function
        self.get_rope_index = get_rope_index_for_model(model_type)

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
        # NOTE: Multi-image (16 slices) - Find the largest tumor across all slices
        # Keep same coordinate system as before: normalized (0-1000) with threshold-based conversion in reward
        self.user_prompt = "<image>" * 16 + "\n" \
            "Task: {Question}\n\n" \
            "Instructions:\n" \
            "1. You are viewing 16 MRI slices sampled uniformly from a 3D brain scan (from shallow to deep).\n" \
            "2. Each slice may contain brain tumor with varying sizes. Your task is to identify the slice with the LARGEST tumor.\n" \
            "3. Brain tumors typically appear as areas with altered intensity (brighter or darker regions) or irregular shapes.\n" \
            "4. Locate the largest tumor region and determine its 2D bounding box [x_min, y_min, x_max, y_max] and center point [x, y].\n" \
            "5. Use normalized coordinates in range [0, 1000] for bbox_2d and point_2d.\n" \
            "6. Output your analysis in <think></think> tags, then provide the final answer in <answer></answer> tags.\n\n" \
            "Output format example:\n" \
            "<think>Analyzing all 16 slices... Slice 8 shows the largest tumor region...</think>\n" \
            "<answer>{Answer}</answer>"

    def _load_single_dataset(self, data_path: str):
        """
        Load a single dataset, handling both DatasetDict and direct Dataset formats
        Also handles datasets with missing _split field in state.json
        """
        import os
        from datasets import Dataset

        try:
            # Try loading as DatasetDict first (original logic)
            dataset_dict = load_from_disk(data_path)
            if hasattr(dataset_dict, 'keys') and 'train' in dataset_dict:
                return dataset_dict['train']
            else:
                # If it's already a Dataset, return it directly
                return dataset_dict
        except KeyError as e:
            if "'_split'" in str(e):
                # Handle missing _split field - manually construct Dataset
                train_path = os.path.join(data_path, 'train')
                if os.path.exists(train_path):
                    try:
                        # Load using Dataset.from_file for arrow files
                        arrow_files = [f for f in os.listdir(train_path) if f.endswith('.arrow')]
                        if arrow_files:
                            arrow_path = os.path.join(train_path, arrow_files[0])
                            dataset = Dataset.from_file(arrow_path)
                            return dataset
                    except Exception as e3:
                        pass
                # If manual loading failed, try default path
                try:
                    return load_from_disk(train_path)
                except Exception:
                    pass
            raise ValueError(
                f"Failed to load dataset from {data_path}. "
                f"Missing '_split' field in state.json. Error: {e}"
            )
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
        
        # Preprocess images first to get actual dimensions
        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]

        if "images" in row_dict:
            # Process images to get final dimensions
            processed_images = [
                process_image(image, self.max_pixels, self.min_pixels) for image in row_dict["images"]
            ]
            row_dict["images"] = processed_images

            # Get image dimensions for Qwen3-VL (critical for correct bbox coordinates!)
            img_width, img_height = processed_images[0].size

            # Store image size in user_prompt for the model to understand coordinate space
            # Note: We use string-based messages to be compatible with manual image processing below
            size_hint = f"Image size: {img_width}x{img_height} pixels. "
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": size_hint + self.user_prompt.format(
                    Question=row_dict["problem"].lower().strip("."),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [120,155]}]"
                )},
            ]
        else:
            # Text-only messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(
                    Question=row_dict["problem"].lower().strip("."),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [120,155]}]"
                )},
            ]

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if "images" in row_dict:  # expand image token
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
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
            # ==================== IMPORTANT: Preserve mrope_position_deltas ====================
            # Qwen3-VL requires mrope_position_deltas (rope_deltas) for correct mRoPE encoding
            # This field is critical for temporal position encoding in video/multi-image inputs
            # ===================================================================================
            position_ids, mrope_position_deltas = self.get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            row_dict["mrope_position_deltas"] = mrope_position_deltas
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seqlen,)
            row_dict["mrope_position_deltas"] = None  # No mRoPE for text-only inputs

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        return row_dict
