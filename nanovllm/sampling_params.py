from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    priority: int = 1  # 请求优先级: 1-5, 1最低, 5最高

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert 1 <= self.priority <= 5, "priority must be between 1 and 5"
