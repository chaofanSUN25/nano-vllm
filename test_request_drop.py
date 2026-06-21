#!/usr/bin/env python3
"""
测试 Request Drop 机制 - 高级版本
====================================

这个脚本演示了增强版的 request drop 功能：
1. 真实拥塞信号检测（GPU内存、队列长度、请求延迟）
2. 智能丢弃策略（优先级、请求大小、等待时间、混合策略）
3. 不同优先级的请求
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def test_request_drop():
    print("="*60)
    print("测试 Request Drop 机制 - 高级版本")
    print("="*60)
    
    # 模型路径
    model_path = os.path.expanduser("/usr/wkspace/Qwen3-0.6B")
    
    # 初始化 tokenizer 和 LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=4)
    
    # 启用 request drop 机制（使用混合策略）
    print("\n[Step 1] 启用 Request Drop 机制")
    llm.enable_drop_mechanism(probability=0.5, strategy="hybrid")
    
    # 设置更敏感的拥塞阈值（便于测试）
    llm.set_congestion_thresholds(
        gpu_memory_threshold=0.8,
        queue_length_threshold=5,
        request_latency_threshold=2.0
    )
    
    # 创建不同优先级的测试请求
    print("\n[Step 2] 添加多个不同优先级的推理请求")
    prompts = [
        ("Write a Python function to calculate fibonacci numbers with detailed comments", 1),  # 低优先级
        ("Explain quantum computing in simple terms for beginners", 3),  # 中优先级
        ("What is the capital of France?", 5),  # 高优先级 - 短请求
        ("List all prime numbers under 100 and explain the Sieve of Eratosthenes algorithm", 2),  # 低优先级 - 长请求
    ]
    
    # 转换为 chat 格式
    formatted_prompts = []
    sampling_params_list = []
    for prompt, priority in prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted_prompt)
        sampling_params_list.append(SamplingParams(
            temperature=0.6, 
            max_tokens=10, 
            priority=priority
        ))
        print(f"  请求 (优先级{priority}): {prompt[:50]}...")
    
    # 开始推理
    print("\n[Step 3] 开始推理（会检测真实拥塞信号）")
    outputs = llm.generate(formatted_prompts, sampling_params_list)
    
    # 检查被丢弃的请求
    dropped_seqs = llm.get_dropped_sequences()
    print(f"\n[Step 4] 检查结果")
    print(f"  总请求数: {len(prompts)}")
    print(f"  完成的请求数: {len(outputs)}")
    print(f"  被丢弃的请求ID: {dropped_seqs}")
    
    # 输出结果
    print("\n[Step 5] 输出结果")
    for i, (prompt, output) in enumerate(zip(formatted_prompts, outputs)):
        print(f"\n请求 {i+1}:")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Completion: {output['text'][:100]}...")
    
    # 测试其他策略
    print("\n" + "="*60)
    print("测试不同的丢弃策略")
    print("="*60)
    
    strategies = ["priority", "size", "age", "hybrid"]
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        llm.enable_drop_mechanism(probability=0.3, strategy=strategy)
        
        # 简单测试
        test_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ]
        try:
            outputs = llm.generate(test_prompts, SamplingParams(max_tokens=5))
            print(f"  策略 {strategy} 工作正常")
        except Exception as e:
            print(f"  策略 {strategy} 出错: {e}")
    
    # 禁用 drop 机制
    print("\n[Step 6] 禁用 Request Drop 机制")
    llm.disable_drop_mechanism()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

if __name__ == "__main__":
    test_request_drop()