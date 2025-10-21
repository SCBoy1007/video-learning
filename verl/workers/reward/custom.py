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


import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import (
    math_compute_score,
    r1v_compute_score,
    seg_compute_score,
    seg_strict_compute_score,
    vision_reasoner_compute_score,
    vision_reasoner_smooth_existence_compute_score
)


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        if compute_score == "math":
            self.compute_score = math_compute_score
            self.supports_details = False
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
            self.supports_details = False
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
            self.supports_details = False
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
            self.supports_details = False
        elif compute_score == "vision_reasoner":
            self.compute_score = vision_reasoner_compute_score
            self.supports_details = True
        elif compute_score == "vision_reasoner_smooth_existence":
            self.compute_score = vision_reasoner_smooth_existence_compute_score
            self.supports_details = True
            self.uses_has_tumor = True  # This reward function requires has_tumor field
        else:
            raise NotImplementedError()

        # Track if this reward function needs has_tumor field
        self.uses_has_tumor = compute_score == "vision_reasoner_smooth_existence"

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0

        # Initialize metric accumulators for detailed tracking
        if self.supports_details:
            if self.uses_has_tumor:
                # Smooth existence detection metrics (8 metrics)
                metric_accumulators = {
                    'format': [],
                    'existence': [],
                    'localization': [],
                    'non_repeat': [],
                    'pred_area': [],
                    'pred_iou': [],
                    'pred_point_dist': [],
                    'gt_has_tumor': []
                }
            else:
                # Original localization metrics (7 metrics)
                metric_accumulators = {
                    'thinking_tag': [],
                    'json_parseable': [],
                    'bbox_format': [],
                    'point_format': [],
                    'bbox_iou': [],
                    'point_distance': [],
                    'non_repeat': []
                }

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # FIXME: Qwen3-VL-Thinking adds "<think>\n" via add_generation_prompt
            # So the model's response doesn't include the opening <think> tag
            # We need to prepend it for reward matching
            if not response_str.startswith('<think>'):
                response_str = '<think>\n' + response_str

            # ground_truth = data_item.non_tensor_batch["answer"]
            ground_truth = data_item.non_tensor_batch["solution"]

            # Get score with optional details
            if self.uses_has_tumor:
                # For smooth existence detection, we need has_tumor field
                has_tumor = data_item.non_tensor_batch.get("has_tumor", True)
                if self.supports_details:
                    score, details = self.compute_score(response_str, ground_truth, has_tumor, return_details=True)
                    # Accumulate sub-metrics (different for smooth existence)
                    for key in metric_accumulators:
                        if key in details:
                            metric_accumulators[key].append(details[key])
                else:
                    score = self.compute_score(response_str, ground_truth, has_tumor)
            else:
                # Original behavior for other reward functions
                if self.supports_details:
                    score, details = self.compute_score(response_str, ground_truth, return_details=True)
                    # Accumulate sub-metrics
                    for key in metric_accumulators:
                        metric_accumulators[key].append(details[key])
                else:
                    score = self.compute_score(response_str, ground_truth)

            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                if self.uses_has_tumor:
                    print(f"[ground_truth] has_tumor={has_tumor}, solution={ground_truth}")
                else:
                    print("[ground_truth]", ground_truth)
                print(f"[score] {score:.3f}")

                # Print sub-metric breakdown if available
                if self.supports_details:
                    if self.uses_has_tumor:
                        # Smooth existence detection metrics
                        print(f"  └─ Format: {details['format']:.1f}/3.0")
                        print(f"     Existence: {details['existence']:.1f}/3.0 (area={details['pred_area']:.0f})")
                        print(f"     Localization: {details['localization']:.3f}/1.0 (IoU={details['pred_iou']:.3f})")
                        print(f"     Non-repeat: {details['non_repeat']:.1f}/1.0")
                    else:
                        # Original localization metrics
                        print(f"  └─ Format: think={details['thinking_tag']:.1f} json={details['json_parseable']:.1f} "
                              f"bbox_fmt={details['bbox_format']:.1f} point_fmt={details['point_format']:.1f}")
                        print(f"     Accuracy: IoU={details['bbox_iou']:.3f} point_dist={details['point_distance']:.1f} "
                              f"non_repeat={details['non_repeat']:.1f}")

        # Store aggregated metrics in data for later logging
        if self.supports_details:
            if not hasattr(data, 'reward_metrics'):
                data.reward_metrics = {}
            for key, values in metric_accumulators.items():
                data.reward_metrics[f'reward/{key}/mean'] = sum(values) / len(values) if values else 0.0
                data.reward_metrics[f'reward/{key}/max'] = max(values) if values else 0.0
                data.reward_metrics[f'reward/{key}/min'] = min(values) if values else 0.0

        return reward_tensor
