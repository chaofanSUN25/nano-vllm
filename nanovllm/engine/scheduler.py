from collections import deque
import random
import time

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        
        # Request drop mechanism
        self.drop_enabled = False
        self.drop_probability = 0.3  # Base drop probability
        self.congestion_detected = False
        self.dropped_sequences = []
        self.step_counter = 0
        
        # 拥塞阈值配置
        self.gpu_memory_threshold = 0.9  # GPU内存使用率超过此值触发拥塞
        self.queue_length_threshold = 100  # 队列长度超过此值触发拥塞
        self.request_latency_threshold = 5.0  # 请求延迟超过此秒数触发拥塞
        
        # 丢弃策略配置
        self.drop_strategy = "priority"  # priority, size, age, hybrid
        
        # Layer-level drop策略
        self.layer_drop_enabled = False
        self.layer_drop_probability = 0.1  # base drop probability per layer
        
        # 统计信息
        self.request_start_times = {}  # seq_id -> start_time
        self.total_requests = 0
        self.dropped_count = 0
        self.dropped_layer_count = 0  # layer-level drop统计

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self.request_start_times[seq.seq_id] = time.time()
        self.total_requests += 1

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0
        
        # 检测真实拥塞信号
        self.detect_congestion()
        
        if self.congestion_detected:
            print(f"[Congestion Detected] Step {self.step_counter}: System overloaded - "
                  f"GPU={self.get_gpu_memory_usage():.1%}, Queue={len(self.waiting) + len(self.running)}, "
                  f"Latency={self.get_average_latency():.2f}s")

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            
            # Check if we should drop this request
            if self.drop_enabled and self.congestion_detected and self._should_drop_request(seq):
                self._drop_request(seq)
                print(f"[Request Drop] Dropped waiting request {seq.seq_id} (priority={seq.priority}, "
                      f"tokens={seq.num_tokens}, age={self._get_request_age(seq):.1f}s)")
                continue
                
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Check if we should drop this running request
            if self.drop_enabled and self.congestion_detected and self._should_drop_request(seq):
                self._drop_request(seq)
                print(f"[Request Drop] Dropped running request {seq.seq_id} (priority={seq.priority}, "
                      f"tokens={seq.num_tokens}, age={self._get_request_age(seq):.1f}s)")
                continue
                
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        # Add scheduled sequences back to running queue for next iteration
        self.running.extend(scheduled_seqs)
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def detect_congestion(self):
        """检测真实拥塞信号"""
        self.step_counter += 1
        
        if not self.drop_enabled:
            self.congestion_detected = False
            return
        
        # 检查GPU内存使用率
        gpu_memory_usage = self.get_gpu_memory_usage()
        
        # 检查队列长度
        queue_length = len(self.waiting) + len(self.running)
        
        # 检查平均请求延迟
        avg_latency = self.get_average_latency()
        
        # 任何一个指标超过阈值即认为拥塞
        self.congestion_detected = (
            gpu_memory_usage > self.gpu_memory_threshold or
            queue_length > self.queue_length_threshold or
            avg_latency > self.request_latency_threshold
        )
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用率"""
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return (total - free) / total
            return 0.0
        except Exception:
            return 0.0
    
    def get_average_latency(self):
        """获取等待队列中请求的平均延迟"""
        if not self.waiting:
            return 0.0
        current_time = time.time()
        total_latency = sum(current_time - seq.created_at for seq in self.waiting)
        return total_latency / len(self.waiting)
    
    def enable_layer_drop(self, probability: float = 0.1):
        """Enable layer-level request drop mechanism
        
        Args:
            probability: Base drop probability per layer (default: 0.1)
        """
        self.layer_drop_enabled = True
        self.layer_drop_probability = probability
        print(f"[Layer Drop] Mechanism enabled with probability: {probability}")
    
    def disable_layer_drop(self):
        """Disable layer-level request drop mechanism"""
        self.layer_drop_enabled = False
        print("[Layer Drop] Mechanism disabled")
    
    def _get_request_age(self, seq: Sequence):
        """获取请求的等待时间"""
        return time.time() - seq.created_at
    
    def _should_drop_request(self, seq: Sequence) -> bool:
        """基于智能策略决定是否丢弃请求"""
        base_prob = self.drop_probability
        
        if self.drop_strategy == "priority":
            # 优先级丢弃：低优先级请求更容易被丢弃
            # priority=1: 概率翻倍, priority=5: 概率减半
            priority_factor = (6 - seq.priority) / 3  # 1.67 to 0.33
            adjusted_prob = base_prob * priority_factor
            
        elif self.drop_strategy == "size":
            # 基于请求大小：大请求更容易被丢弃
            size_factor = min(seq.num_tokens / 512, 2.0)  # 最多2倍
            adjusted_prob = base_prob * size_factor
            
        elif self.drop_strategy == "age":
            # 基于等待时间：新请求更容易被丢弃
            age = self._get_request_age(seq)
            age_factor = max(1.0 - age / 10.0, 0.1)  # 随时间递减
            adjusted_prob = base_prob * age_factor
            
        elif self.drop_strategy == "hybrid":
            # 混合策略：综合考虑优先级、大小和年龄
            priority_factor = (6 - seq.priority) / 3
            size_factor = min(seq.num_tokens / 512, 1.5)
            age = self._get_request_age(seq)
            age_factor = max(1.0 - age / 15.0, 0.2)
            adjusted_prob = base_prob * priority_factor * size_factor * age_factor
            
        else:
            adjusted_prob = base_prob
        
        return random.random() < adjusted_prob
    
    def _drop_request(self, seq: Sequence):
        """Drop a request and clean up resources"""
        # 防止重复添加到dropped_sequences
        if seq.status == SequenceStatus.DROPPED:
            return
        
        seq.status = SequenceStatus.DROPPED
        self.dropped_sequences.append(seq.seq_id)
        self.dropped_count += 1
        
        # Clean up resources
        if seq.block_table:
            self.block_manager.deallocate(seq)
    
    def enable_drop_mechanism(self, probability: float = 0.3, strategy: str = "priority"):
        """Enable the request drop mechanism with configurable strategy
        
        Args:
            probability: Base drop probability (default: 0.3)
            strategy: Drop strategy. Options: "priority", "size", "age", "hybrid"
        """
        self.drop_enabled = True
        self.drop_probability = probability
        self.drop_strategy = strategy
        print(f"[Request Drop] Mechanism enabled with probability: {probability}, strategy: {strategy}")
    
    def set_congestion_thresholds(self, gpu_memory_threshold: float = 0.9,
                                   queue_length_threshold: int = 100,
                                   request_latency_threshold: float = 5.0):
        """设置拥塞检测阈值"""
        self.gpu_memory_threshold = gpu_memory_threshold
        self.queue_length_threshold = queue_length_threshold
        self.request_latency_threshold = request_latency_threshold
        print(f"[Request Drop] Thresholds updated: GPU={gpu_memory_threshold}, "
              f"Queue={queue_length_threshold}, Latency={request_latency_threshold}s")
    
    def disable_drop_mechanism(self):
        """Disable the request drop mechanism"""
        self.drop_enabled = False
        self.congestion_detected = False
        print(f"[Request Drop] Mechanism disabled")
    
    def get_dropped_sequences(self) -> list[int]:
        """Get list of dropped sequence IDs"""
        return self.dropped_sequences
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # Skip dropped sequences
            if seq.status == SequenceStatus.DROPPED:
                continue
        
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        
    def drop_sequences(self, seq_ids: list[int]):
        """Drop sequences by their IDs (for layer-level drop)
        
        Args:
            seq_ids: List of sequence IDs to drop
        """
        for seq_id in seq_ids:
            # Find and drop the sequence
            for seq in list(self.running):
                if seq.seq_id == seq_id:
                    seq.status = SequenceStatus.DROPPED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    self.dropped_sequences.append(seq_id)
                    break