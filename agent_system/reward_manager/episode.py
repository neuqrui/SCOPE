# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from collections import defaultdict
import math

from verl import DataProto
import torch
import numpy as np

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length=False,
        success_length_bonus_coef=0.0,
        success_length_max_steps=15,
        success_length_pairwise_tau=0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        # Extra reward for successful episodes only. Shorter successes score higher.
        # Failed episodes keep their original reward.
        self.success_length_bonus_coef = float(success_length_bonus_coef)
        self.success_length_max_steps = float(success_length_max_steps)
        # Soft pairwise efficiency. tau=0 leaves the episode reward unchanged.
        # Among successes in one GRPO group, b_i is the mean sigmoid of log length ratios.
        self.success_length_pairwise_tau = float(success_length_pairwise_tau)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        pairwise_bias = self._pairwise_efficiency(data)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards

            if self.success_length_pairwise_tau > 0 and float(episode_rewards) > 0:
                traj_uid = data_item.non_tensor_batch["traj_uid"]
                score = 10.0 + 5.0 * pairwise_bias.get(traj_uid, 0.5)
            elif self.success_length_bonus_coef > 0 and float(episode_rewards) > 0:
                steps = float(episode_lengths)
                efficiency = max(0.0, (self.success_length_max_steps - steps) / self.success_length_max_steps)
                score = float(episode_rewards) + self.success_length_bonus_coef * efficiency
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor

    def _pairwise_efficiency(self, data: DataProto) -> dict:
        """b_i for each successful trajectory. Failures are omitted.

        p_ij = sigmoid(log(T_j / T_i) / tau). A shorter trajectory scores above 0.5.
        A group with one success gets the neutral value 0.5.
        """
        tau = self.success_length_pairwise_tau
        if tau <= 0:
            return {}
        required = ("uid", "traj_uid", "episode_rewards", "episode_lengths")
        if any(key not in data.non_tensor_batch for key in required):
            return {}

        uids = data.non_tensor_batch["uid"]
        traj_uids = data.non_tensor_batch["traj_uid"]
        rewards = data.non_tensor_batch["episode_rewards"]
        lengths = data.non_tensor_batch["episode_lengths"]
        episodes = {}
        for row, traj_uid in enumerate(traj_uids):
            if traj_uid in episodes:
                continue
            episodes[traj_uid] = (uids[row], float(rewards[row]), float(lengths[row]))

        grouped = defaultdict(list)
        for traj_uid, (uid, reward, length) in episodes.items():
            if reward > 0 and length > 0:
                grouped[uid].append((traj_uid, length))

        bias = {}
        for members in grouped.values():
            if len(members) == 1:
                bias[members[0][0]] = 0.5
                continue
            for index, (traj_uid, length) in enumerate(members):
                total = 0.0
                for other_index, (_, other_length) in enumerate(members):
                    if other_index == index:
                        continue
                    total += _sigmoid(math.log(other_length / length) / tau)
                bias[traj_uid] = total / (len(members) - 1)
        return bias


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
