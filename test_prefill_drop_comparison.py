#!/usr/bin/env python3
"""
对比测试：启用 vs 不启用 Prefill Layer Drop
=============================================

这个脚本对比测试 prefill layer drop 对推理性能的影响。
"""
import os
import time
from nanovllm import LLM, SamplingParams
from itertools import count
from nanovllm.engine.sequence import Sequence
from transformers import AutoTokenizer


def create_test_prompts():
    """创建测试用的 prompts"""
    return [
        # 短 prompt
        "What is 2 + 2?",
        "Hello world!",
        "Simple test",
        
        # 中等长度 prompt
        "Explain quantum computing in simple terms for a beginner who has no prior knowledge of physics.",
        "What is the difference between machine learning and deep learning?",
        "Write a Python function to calculate fibonacci numbers efficiently.",
        
        # 长 prompt
        "Provide a detailed explanation of how neural networks work, including forward propagation, backward propagation, activation functions, loss functions, and optimization algorithms with examples.",
        "Write a comprehensive essay about the history of artificial intelligence from the 1950s to modern deep learning, covering all major milestones and key figures.",
        "Explain the complete theory of special and general relativity with mathematical framework and experimental validations.",
        
        # 超长 prompt (计算量大)
        "Provide an exhaustive analysis of distributed systems architecture covering consensus algorithms like Paxos and Raft, fault tolerance mechanisms, CAP theorem implications, gossip protocols, distributed transactions, and real-world applications in cloud computing and microservices.",
    ]


def test_without_prefill_drop():
    """测试不启用 prefill layer drop"""
    print("\n" + "="*60)
    print("测试 1: 不启用 Prefill Layer Drop")
    print("="*60)
    
    model_path = os.path.expanduser("/usr/wkspace/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=32)
    Sequence.counter = count(0)
    
    prompts = create_test_prompts()
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        ) for p in prompts
    ]
    sampling_params = [SamplingParams(temperature=0.7, max_tokens=30, priority=3) for _ in prompts]
    
    print(f"\n请求数: {len(prompts)}")
    print("开始推理...")
    
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    end_time = time.time()
    
    elapsed = end_time - start_time
    dropped_count = len(llm.scheduler.dropped_sequences)
    
    print(f"\n结果统计:")
    print(f"  完成请求数: {len(outputs)}")
    print(f"  丢弃请求数: {dropped_count}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均每请求: {elapsed/len(prompts):.2f} 秒")
    
    return len(outputs), dropped_count, elapsed


def test_with_prefill_drop():
    """测试启用 prefill layer drop"""
    print("\n" + "="*60)
    print("测试 2: 启用 Prefill Layer Drop (概率=0.2)")
    print("="*60)
    
    model_path = os.path.expanduser("/usr/wkspace/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=32)
    Sequence.counter = count(0)
    
    # 启用 prefill layer drop
    llm.enable_layer_drop(probability=0.2)
    
    prompts = create_test_prompts()
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        ) for p in prompts
    ]
    sampling_params = [SamplingParams(temperature=0.7, max_tokens=30, priority=3) for _ in prompts]
    
    print(f"\n请求数: {len(prompts)}")
    print("Prefill Layer Drop 策略:")
    print("  ├─ Prefill 阶段: drop概率 = 0.2 * 1.5 ~ 0.3 (早期layer)")
    print("  ├─ Prefill 阶段: drop概率 = 0.2 * 0.2 ~ 0.04 (晚期layer)")
    print("  ├─ 长 prompt: drop概率增加 (prompt_tokens / 512)")
    print("  └─ Decode 阶段: drop概率更低")
    print("\n开始推理...")
    
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    end_time = time.time()
    
    elapsed = end_time - start_time
    dropped_count = len(llm.scheduler.dropped_sequences)
    dropped_layer_count = llm.scheduler.dropped_layer_count
    
    print(f"\n结果统计:")
    print(f"  完成请求数: {len(outputs)}")
    print(f"  被 Layer Drop 丢弃数: {dropped_count}")
    print(f"  Layer Drop 次数: {dropped_layer_count}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均每请求: {elapsed/len(prompts):.2f} 秒")
    
    # 分析被 drop 的请求特征
    if dropped_count > 0:
        print(f"\n  被丢弃的请求分析:")
        for seq_id in sorted(llm.scheduler.dropped_sequences):
            prompt = prompts[seq_id]
            prompt_tokens = len(tokenizer.encode(prompt))
            print(f"    Seq {seq_id}: prompt长度={prompt_tokens} tokens, {prompt[:50]}...")
    
    return len(outputs), dropped_count, elapsed


def compare_results(results_without, results_with):
    """对比两组测试结果"""
    print("\n" + "="*60)
    print("对比分析")
    print("="*60)
    
    completed1, dropped1, time1 = results_without
    completed2, dropped2, time2 = results_with
    
    print(f"\n不启用 Prefill Drop:")
    print(f"  完成请求数: {completed1}")
    print(f"  丢弃请求数: {dropped1}")
    print(f"  总耗时: {time1:.2f} 秒")
    
    print(f"\n启用 Prefill Drop:")
    print(f"  完成请求数: {completed2}")
    print(f"  丢弃请求数: {dropped2}")
    print(f"  总耗时: {time2:.2f} 秒")
    
    if dropped2 > dropped1:
        print(f"\n性能提升分析:")
        print(f"  ├─ 通过 drop 长计算请求，节省了计算资源")
        print(f"  ├─ Drop 比率增加: {dropped2 - dropped1} 个请求")
        if time2 < time1:
            speedup = (time1 - time2) / time1 * 100
            print(f"  ├─ 耗时减少: {time1 - time2:.2f} 秒 ({speedup:.1f}% 提升)")
        print(f"  └─ Prefill Layer Drop 成功拦截了计算密集型请求")
    else:
        print(f"\n注意: Drop 概率可能需要调整以观察到明显效果")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


def main():
    print("\n" + "#"*60)
    print("# Prefill Layer Drop 对比测试")
    print("# 验证 prefill 阶段 drop 对性能的影响")
    print("#"*60)
    
    # 测试 1: 不启用 prefill drop
    results_without = test_without_prefill_drop()
    
    # 测试 2: 启用 prefill drop
    results_with = test_with_prefill_drop()
    
    # 对比结果
    compare_results(results_without, results_with)


if __name__ == "__main__":
    main()