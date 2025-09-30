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
from verl.utils.reward_score import math_compute_score, r1v_compute_score, seg_compute_score, seg_strict_compute_score, vision_reasoner_compute_score, brain_tumor_3d_compute_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_name = compute_score
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict":
            self.compute_score = seg_strict_compute_score
        elif compute_score == "vision_reasoner":
            self.compute_score = vision_reasoner_compute_score
        elif compute_score == "brain_tumor_3d":
            self.compute_score = brain_tumor_3d_compute_score
        else:
            raise NotImplementedError()

        # Track if this compute_score supports detailed metrics
        self.supports_details = compute_score in ["brain_tumor_3d"]

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_detail = 0

        # Initialize metric accumulators for detailed tracking
        if self.supports_details:
            metric_accumulators = {
                'thinking_format': [],
                'video_keyword': [],
                'format': [],
                'iou': [],
                'peak_slice': [],
                'tumor_ratio': [],
                'completeness': [],
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

            # ground_truth = data_item.non_tensor_batch["answer"]
            ground_truth = data_item.non_tensor_batch["solution"]
            # print(ground_truth,response_str)

            # Get score with optional details
            if self.supports_details:
                score, details = self.compute_score(response_str, ground_truth, return_details=True)
                # Accumulate sub-metrics
                for key in metric_accumulators:
                    metric_accumulators[key].append(details[key])
            else:
                score = self.compute_score(response_str, ground_truth)

            reward_tensor[i, valid_response_length - 1] = score

            # Print first sample's full details per batch
            if already_print_detail < self.num_examine:
                already_print_detail += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print(f"[score] {score:.3f}")

                # Print sub-metric breakdown for brain_tumor_3d
                if self.supports_details:
                    # Import weights dynamically to avoid hardcoding
                    try:
                        from verl.utils.reward_score.brain_tumor_3d import get_reward_weights
                        weights = get_reward_weights()
                    except:
                        # Fallback to default if import fails
                        weights = {
                            'thinking_format': 0.5, 'video_keyword': 0.5, 'format': 1.0, 'iou': 3.0,
                            'peak_slice': 1.5, 'tumor_ratio': 1.5, 'completeness': 0.5, 'non_repeat': 0.5
                        }

                    print(f"  └─ thinking_format: {details['thinking_format']:.2f}/{weights['thinking_format']:.1f} | "
                          f"video_keyword: {details['video_keyword']:.2f}/{weights['video_keyword']:.1f} | "
                          f"format: {details['format']:.2f}/{weights['format']:.1f} | "
                          f"iou: {details['iou']:.2f}/{weights['iou']:.1f}")
                    print(f"  └─ peak_slice: {details['peak_slice']:.2f}/{weights['peak_slice']:.1f} | "
                          f"tumor_ratio: {details['tumor_ratio']:.2f}/{weights['tumor_ratio']:.1f} | "
                          f"completeness: {details['completeness']:.2f}/{weights['completeness']:.1f} | "
                          f"non_repeat: {details['non_repeat']:.2f}/{weights['non_repeat']:.1f}")
            else:
                # Print only score for remaining samples
                print(f"[score {i+1}] {score:.3f}")

        # Store aggregated metrics in data for later logging
        if self.supports_details:
            if not hasattr(data, 'reward_metrics'):
                data.reward_metrics = {}
            for key, values in metric_accumulators.items():
                data.reward_metrics[f'reward/{key}/mean'] = sum(values) / len(values) if values else 0.0
                data.reward_metrics[f'reward/{key}/max'] = max(values) if values else 0.0
                data.reward_metrics[f'reward/{key}/min'] = min(values) if values else 0.0

        return reward_tensor
