import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        
        # Call model_runner.run() which returns (token_ids, dropped_seq_ids)
        result = self.model_runner.call("run", seqs, is_prefill)
        if isinstance(result, tuple):
            token_ids, dropped_seq_ids = result
            # Handle layer-level dropped sequences
            if dropped_seq_ids:
                self.scheduler.drop_sequences(dropped_seq_ids)
        else:
            token_ids = result
            dropped_seq_ids = []
        
        # Filter out dropped sequences before postprocessing
        active_seqs = [seq for seq in seqs if seq.status != SequenceStatus.DROPPED]
        active_token_ids = []
        seq_idx = 0
        for seq in seqs:
            if seq.status != SequenceStatus.DROPPED:
                active_token_ids.append(token_ids[seq_idx])
            seq_idx += 1
        
        self.scheduler.postprocess(active_seqs, active_token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()
    
    def enable_drop_mechanism(self, probability: float = 0.3, strategy: str = "priority"):
        """Enable the request drop mechanism
        
        Args:
            probability: Base drop probability (default: 0.3)
            strategy: Drop strategy. Options: "priority", "size", "age", "hybrid"
        """
        self.scheduler.enable_drop_mechanism(probability, strategy)
    
    def set_congestion_thresholds(self, gpu_memory_threshold: float = 0.9,
                                   queue_length_threshold: int = 100,
                                   request_latency_threshold: float = 5.0):
        """Set congestion detection thresholds"""
        self.scheduler.set_congestion_thresholds(gpu_memory_threshold,
                                                  queue_length_threshold,
                                                  request_latency_threshold)
    
    def disable_drop_mechanism(self):
        """Disable the request drop mechanism"""
        self.scheduler.disable_drop_mechanism()
    
    def set_pd_drop_multipliers(self, prefill_multiplier: float = 2.0, decode_multiplier: float = 0.3):
        """Set drop probability multipliers for Prefill and Decode stages
        
        Args:
            prefill_multiplier: Multiplier for prefill stage drop probability (>1 means more aggressive)
            decode_multiplier: Multiplier for decode stage drop probability (<1 means more conservative)
        """
        self.scheduler.set_pd_drop_multipliers(prefill_multiplier, decode_multiplier)
    
    def enable_layer_drop(self, probability: float = 0.1):
        """Enable layer-level request drop mechanism
        
        Args:
            probability: Base drop probability per layer (default: 0.1)
        """
        self.scheduler.enable_layer_drop(probability)
        # 传递配置给model_runner
        if hasattr(self.model_runner, 'layer_drop_enabled'):
            self.model_runner.layer_drop_enabled = True
            self.model_runner.layer_drop_probability = probability
        else:
            # 如果没有属性，动态添加
            setattr(self.model_runner, 'layer_drop_enabled', True)
            setattr(self.model_runner, 'layer_drop_probability', probability)
    
    def get_dropped_sequences(self) -> list[int]:
        """Get list of dropped sequence IDs"""
        return self.scheduler.get_dropped_sequences()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prompt_map = {}  # Track prompts by seq_id
        prefill_throughput = decode_throughput = 0.
        # Track seq_id -> prompt mapping
        seq_counter = 0
        for prompt, sp in zip(prompts, sampling_params):
            seq = Sequence(self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt, sp)
            prompt_map[seq.seq_id] = prompt
            self.scheduler.add(seq)
            seq_counter += 1
        
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        
        # Build output with prompts
        result = []
        for seq_id in sorted(outputs.keys()):
            token_ids = outputs[seq_id]
            prompt = prompt_map.get(seq_id, "")
            # Decode prompt if it's token IDs
            if isinstance(prompt, list):
                prompt = self.tokenizer.decode(prompt)
            result.append({
                "prompt": prompt,
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids
            })
        return result
