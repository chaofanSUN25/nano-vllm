"""
NanoLLM Request Drop Mechanism - MVP Demo
===========================================

这个 Demo 展示了如何在 NanoLLM 框架中实现 request drop 机制：
1. 模拟拥塞信号检测
2. 在 batch 组队后、层级推理前丢弃部分请求
3. 直观展示被 drop 的请求跳过了后续所有计算流程

核心改动点：
- sequence.py: 添加 SequenceStatus.DROPPED 状态
- scheduler.py: 在 schedule() 中实现 request drop 逻辑
- llm_engine.py: 添加 enable/disable_drop_mechanism() 方法
"""
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
import random


class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    DROPPED = auto()  # 新增：被丢弃的状态


@dataclass
class MockSamplingParams:
    """模拟采样参数"""
    temperature: float = 0.7
    max_tokens: int = 50
    ignore_eos: bool = False


class MockSequence:
    """模拟序列，用于演示 request drop 机制"""
    counter = 0
    
    def __init__(self, prompt: str, sampling_params: MockSamplingParams = None):
        self.seq_id = MockSequence.counter
        MockSequence.counter += 1
        self.prompt = prompt
        self.sampling_params = sampling_params or MockSamplingParams()
        self.status = SequenceStatus.WAITING
        self.token_ids = list(range(10))  # 模拟 token IDs
        self.num_tokens = len(self.token_ids)
        self.num_scheduled_tokens = 0
        self.was_dropped = False
        self.layers_processed = 0  # 记录处理的层数
        
    def __repr__(self):
        return f"Seq#{self.seq_id}(status={self.status.name}, dropped={self.was_dropped}, layers={self.layers_processed})"


class RequestDropScheduler:
    """
    模拟调度器，支持 request drop 机制
    
    核心逻辑：
    1. schedule() 组batch
    2. 检测拥塞信号
    3. 决定是否丢弃部分请求
    """
    
    def __init__(self, max_num_seqs: int = 4):
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[MockSequence] = deque()
        self.running: deque[MockSequence] = deque()
        
        # Request Drop 机制配置
        self.drop_enabled = False
        self.drop_probability = 0.3  # 30% 概率丢弃
        self.congestion_detected = False
        self.step_counter = 0
        self.dropped_count = 0
        
    def add_request(self, prompt: str):
        """添加请求"""
        seq = MockSequence(prompt)
        self.waiting.append(seq)
        print(f"  [Add] Added request: {seq}")
        return seq
    
    def enable_drop_mechanism(self, probability: float = 0.3):
        """启用 request drop 机制"""
        self.drop_enabled = True
        self.drop_probability = probability
        print(f"\n{'='*60}")
        print(f"[Request Drop] 机制已启用，丢弃概率: {probability}")
        print(f"{'='*60}\n")
    
    def disable_drop_mechanism(self):
        """禁用 request drop 机制"""
        self.drop_enabled = False
        print(f"\n[Request Drop] 机制已禁用")
    
    def _detect_congestion(self) -> bool:
        """
        模拟拥塞检测
        在实际系统中，这里会接收来自流调度的过载信号
        """
        self.step_counter += 1
        
        # 模拟：每3步检测一次拥塞
        if self.step_counter % 3 == 0:
            # 模拟 40% 概率检测到拥塞
            is_congested = random.random() < 0.4
            if is_congested:
                print(f"\n{'='*60}")
                print(f"[拥塞信号] Step {self.step_counter}: 检测到系统拥塞/过载!")
                print(f"{'='*60}")
            return is_congested
        return False
    
    def _should_drop_request(self, seq: MockSequence) -> bool:
        """根据概率决定是否丢弃请求"""
        if not self.drop_enabled:
            return False
        return random.random() < self.drop_probability
    
    def _drop_request(self, seq: MockSequence, queue: deque, is_waiting: bool):
        """执行请求丢弃"""
        seq.status = SequenceStatus.DROPPED
        seq.was_dropped = True
        self.dropped_count += 1
        
        # 从队列中移除
        if seq in queue:
            queue.remove(seq)
        
        print(f"  [Request Drop] 丢弃请求 {seq.seq_id} (prompt: '{seq.prompt[:20]}...')")
    
    def schedule(self) -> list[MockSequence]:
        """
        调度核心：
        1. 组batch
        2. 检测拥塞
        3. 丢弃部分请求
        
        返回: 待处理的序列列表
        """
        scheduled_seqs = []
        
        # 检测拥塞
        self.congestion_detected = self._detect_congestion()
        
        # ====== 关键修改点：在这里实现 request drop ======
        # 从 waiting 队列中组batch
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 如果启用 drop 机制且检测到拥塞，尝试丢弃请求
            if self.drop_enabled and self.congestion_detected:
                if self._should_drop_request(seq):
                    self._drop_request(seq, self.waiting, is_waiting=True)
                    continue  # 跳过后续处理
            
            # 正常调度
            seq.status = SequenceStatus.RUNNING
            seq.num_scheduled_tokens = min(10, seq.num_tokens)
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            print(f"  [Schedule] 调度请求 {seq.seq_id} (prompt: '{seq.prompt[:20]}...')")
        
        return scheduled_seqs
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "dropped_total": self.dropped_count,
            "drop_enabled": self.drop_enabled,
            "congestion_detected": self.congestion_detected,
        }


class MockModelRunner:
    """
    模拟模型运行器
    展示被 drop 的请求如何跳过后续所有层计算
    """
    
    def __init__(self, num_layers: int = 4):
        self.num_layers = num_layers
        
    def run_model(self, seqs: list[MockSequence], is_prefill: bool = True):
        """
        模拟模型推理流程
        
        对于被 drop 的请求，应该直接跳过，不执行任何层计算
        """
        results = []
        
        for seq in seqs:
            print(f"\n  Processing Seq#{seq.seq_id}:")
            
            # ====== 关键：检查请求是否被丢弃 ======
            if seq.was_dropped:
                print(f"    [SKIPPED] 请求已被丢弃，跳过所有 {self.num_layers} 层计算!")
                results.append({"seq_id": seq.seq_id, "dropped": True, "layers": 0})
                continue
            
            # 正常请求：逐层处理
            seq.layers_processed = 0
            for layer_idx in range(self.num_layers):
                # 模拟每层的计算
                self._compute_layer(seq, layer_idx)
                seq.layers_processed += 1
            
            print(f"    [COMPLETED] 完成 {seq.layers_processed} 层计算")
            results.append({"seq_id": seq.seq_id, "dropped": False, "layers": seq.layers_processed})
        
        return results
    
    def _compute_layer(self, seq: MockSequence, layer_idx: int):
        """模拟单层计算"""
        # 模拟计算延迟
        time.sleep(0.01)
        print(f"    Layer {layer_idx}: computing...")


def run_demo_without_drop():
    """Demo 1: 不启用 drop 机制的正常流程"""
    print("\n" + "="*60)
    print("Demo 1: 正常流程（不启用 Request Drop）")
    print("="*60)
    
    scheduler = RequestDropScheduler(max_num_seqs=4)
    model_runner = MockModelRunner(num_layers=4)
    
    # 添加多个请求
    prompts = [
        "Write a Python function to calculate fibonacci",
        "Explain quantum computing in simple terms",
        "What is the capital of France?",
        "List all prime numbers under 100",
    ]
    
    for prompt in prompts:
        scheduler.add_request(prompt)
    
    print("\n--- Step 1: 首次调度 (Prefill) ---")
    batch = scheduler.schedule()
    results = model_runner.run_model(batch)
    
    print(f"\n--- 统计信息 ---")
    stats = scheduler.get_stats()
    print(f"  Waiting: {stats['waiting']}, Running: {stats['running']}")
    print(f"  本轮Dropped: {sum(1 for r in results if r['dropped'])}")
    print(f"  本轮完成层数: {[r['layers'] for r in results]}")


def run_demo_with_drop():
    """Demo 2: 启用 drop 机制的流程"""
    print("\n" + "="*60)
    print("Demo 2: 启用 Request Drop 机制")
    print("="*60)
    
    scheduler = RequestDropScheduler(max_num_seqs=4)
    model_runner = MockModelRunner(num_layers=4)
    
    # 启用 drop 机制
    scheduler.enable_drop_mechanism(probability=0.5)  # 50% 丢弃概率
    
    # 添加多个请求
    prompts = [
        "Write a Python function to calculate fibonacci",
        "Explain quantum computing in simple terms",
        "What is the capital of France?",
        "List all prime numbers under 100",
    ]
    
    for prompt in prompts:
        scheduler.add_request(prompt)
    
    print("\n--- Step 1: 首次调度 (Prefill) ---")
    batch = scheduler.schedule()
    results = model_runner.run_model(batch)
    
    print(f"\n--- 统计信息 ---")
    stats = scheduler.get_stats()
    print(f"  Waiting: {stats['waiting']}, Running: {stats['running']}")
    print(f"  本轮Dropped: {sum(1 for r in results if r['dropped'])}")
    print(f"  本轮完成层数: {[r['layers'] for r in results]}")
    print(f"  累计丢弃数: {stats['dropped_total']}")
    
    # 模拟继续调度（decode阶段）
    if scheduler.waiting:
        print("\n--- Step 2: 第二次调度 (Decode) ---")
        batch = scheduler.schedule()
        if batch:
            results = model_runner.run_model(batch)
            print(f"\n--- 统计信息 ---")
            stats = scheduler.get_stats()
            print(f"  本轮Dropped: {sum(1 for r in results if r['dropped'])}")
            print(f"  本轮完成层数: {[r['layers'] for r in results]}")


def run_demo_congestion_signal():
    """Demo 3: 模拟流调度拥塞信号触发 drop"""
    print("\n" + "="*60)
    print("Demo 3: 模拟拥塞信号触发 Request Drop")
    print("="*60)
    
    scheduler = RequestDropScheduler(max_num_seqs=4)
    model_runner = MockModelRunner(num_layers=4)
    
    # 启用 drop 机制
    scheduler.enable_drop_mechanism(probability=0.4)
    
    # 模拟连续调度，触发拥塞检测
    prompts = [
        "Request A: Heavy computation task",
        "Request B: Another heavy task",
        "Request C: Yet another task",
        "Request D: More work to do",
    ]
    
    for prompt in prompts:
        scheduler.add_request(prompt)
    
    print("\n--- 连续调度演示（模拟拥塞检测） ---")
    
    for step in range(1, 6):
        print(f"\n=== Step {step} ===")
        batch = scheduler.schedule()
        
        if not batch:
            print("  [Batch Empty] 所有请求已处理完毕")
            break
            
        results = model_runner.run_model(batch)
        
        dropped_this_step = sum(1 for r in results if r['dropped'])
        completed_layers = [r['layers'] for r in results]
        
        print(f"\n  [Result] Dropped: {dropped_this_step}, Layers: {completed_layers}")
        
        if scheduler.waiting:
            print(f"  [Queue] {len(scheduler.waiting)} requests remaining")
    
    print(f"\n--- 最终统计 ---")
    stats = scheduler.get_stats()
    print(f"  总丢弃请求数: {stats['dropped_total']}")


def main():
    """主函数：运行所有 Demo"""
    print("\n" + "#"*60)
    print("# NanoLLM Request Drop Mechanism - MVP Demo")
    print("# 框架: NanoLLM | 目标: 流调度过载时的 Request Drop")
    print("#"*60)
    
    print("""
    核心概念：
    1. 在 batch 组队之后、每轮层级推理之前，监听调度信号
    2. 当检测到拥塞/过载信号时，对 batch 内指定请求做 request drop
    3. 被 drop 的请求直接跳过后续所有层计算，实现计算截断
    
    适用场景：
    - 流调度系统过载时，快速降低负载
    - 优先级调度：丢弃低优先级请求
    - 拥塞控制：防止系统崩溃
    """)
    
    # 运行各个 Demo
    run_demo_without_drop()
    run_demo_with_drop()
    run_demo_congestion_signal()
    
    print("\n" + "#"*60)
    print("# Demo 完成!")
    print("#"*60)
    print("""
    总结：
    - 当 drop_enabled=True 且 congestion_detected=True 时
    - 调度器会以 drop_probability 的概率丢弃请求
    - 被丢弃的请求在 model_runner.run_model() 中直接跳过
    - 不执行任何层计算，实现计算截断
    
    后续合并到正式项目时：
    1. 将 SequenceStatus.DROPPED 添加到 sequence.py
    2. 将 request drop 逻辑集成到 scheduler.py 的 schedule() 方法
    3. 在 llm_engine.py 添加 enable_drop_mechanism() 接口
    4. 实际系统中，congestion_detected 应接收流调度的真实信号
    """)


if __name__ == "__main__":
    main()