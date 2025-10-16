"""
Qwen3-VL position encoding support for VeRL framework.
Adapted from Qwen3-VL official implementation.
"""
from typing import Optional, Tuple
import torch


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the position ids for Qwen3-VL with timestamp-based video encoding.

    Key difference from Qwen2.5-VL: Qwen3-VL uses timestamps rather than absolute
    time position IDs for video temporal encoding.

    Args:
        processor: Qwen3-VL processor
        input_ids: Input token IDs
        image_grid_thw: Image grid (temporal, height, width)
        video_grid_thw: Video grid (temporal, height, width)
        second_per_grid_ts: Time intervals for video frames (not used in Qwen3-VL)
        attention_mask: Attention mask

    Returns:
        position_ids: 3D position IDs (3, batch, seq_len)
        mrope_position_deltas: Position deltas for mRoPE
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    # Qwen3-VL uses timestamps to separate videos, so video_grid_thw needs special handling
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1  # Set temporal dimension to 1 (using timestamps)

    mrope_position_deltas = []

    # ==================== FIX: Support both 1D and 2D input_ids ====================
    # RLHFDataset returns 1D tensors (seqlen,) from tokenize_and_postprocess_data
    # We need to expand to 2D (batch, seqlen) for compatibility, then squeeze back
    # =============================================================================
    input_ids_was_1d = False
    if input_ids is not None and input_ids.dim() == 1:
        input_ids_was_1d = True
        input_ids = input_ids.unsqueeze(0)  # (seqlen,) -> (1, seqlen)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)  # (seqlen,) -> (1, seqlen)

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids_single in enumerate(total_input_ids):
            input_ids_single = input_ids_single[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids_single == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_single[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_single.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1
                # (we use timestamps to encode temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        # Restore original dimension if input was 1D
        if input_ids_was_1d:
            position_ids = position_ids.squeeze(1)  # (3, 1, seqlen) -> (3, seqlen)
            mrope_position_deltas = mrope_position_deltas.squeeze(0)  # (1, 1) -> (1,)

        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        # Restore original dimension if input was 1D
        if input_ids_was_1d:
            position_ids = position_ids.squeeze(1)  # (3, 1, seqlen) -> (3, seqlen)
            mrope_position_deltas = mrope_position_deltas.squeeze(0)  # (1, 1) -> (1,)

        return position_ids, mrope_position_deltas
