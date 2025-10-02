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
"""
A Ray logger will receive logging info from different processes.
"""

import numbers
from typing import Dict


def concat_dict_to_str(dict: Dict, step):
    """
    Tiered logging strategy:
    - Steps 1-10: Print all metrics (full detail for debugging)
    - Steps 11-50: Print core training metrics only
    - Steps 51+: Print core metrics every 10 steps (reduce noise)
    """
    # Core training metrics (always important)
    core_metrics = {
        'actor/kl_loss', 'actor/kl_coef', 'actor/pg_loss', 'actor/pg_clipfrac',
        'actor/ppo_kl_approx', 'actor/ppo_kl_true', 'actor/grad_norm', 'actor/lr',
        'critic/score/mean', 'critic/rewards/mean', 'critic/advantages/mean'
    }
    # Reward breakdown (important for diagnosing reward hacking)
    reward_metrics = [k for k in dict.keys() if k.startswith('reward/')]
    core_metrics.update(reward_metrics)

    # Determine what to print based on step
    if step <= 10:
        # Full detail for first 10 steps
        keys_to_print = dict.keys()
        prefix = f"step {step} [FULL]:"
    elif step <= 50:
        # Core metrics only for steps 11-50
        keys_to_print = [k for k in dict.keys() if k in core_metrics]
        prefix = f"step {step} [CORE]:"
    else:
        # Steps 51+: print every 10 steps
        if step % 10 != 0:
            return None  # Skip this step
        keys_to_print = [k for k in dict.keys() if k in core_metrics]
        prefix = f"step {step} [CORE]:"

    output = [prefix]
    for k in keys_to_print:
        v = dict[k]
        if isinstance(v, numbers.Number):
            # Use more decimal places for learning rate and KL to show small values
            if "lr" in k.lower() or "kl" in k.lower():
                output.append(f"{k}:{v:.6f}")
            else:
                output.append(f"{k}:{v:.3f}")

    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        self.print_to_console = print_to_console
        if print_to_console:
            print("Using LocalLogger is deprecated. The constructor API will change.")

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            log_str = concat_dict_to_str(data, step=step)
            if log_str:  # Only print if not None (skip filtered steps)
                print(log_str, flush=True)
