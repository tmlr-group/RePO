from dataclasses import dataclass

from configs import GRPOConfig


@dataclass
class RandomMaskGRPOConfig(GRPOConfig):
    thinking_mask_keep_prob: float = 0.0
    keep_final_answer: bool = True

