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
    # 使用4张卡，max_num_seqs=32来增加并发
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=4, max_num_seqs=32)
    
    # 启用 request drop 机制（使用混合策略）
    print("\n[Step 1] 启用 Request Drop 机制")
    llm.enable_drop_mechanism(probability=0.3, strategy="hybrid")
    
    # 设置拥塞阈值
    llm.set_congestion_thresholds(
        gpu_memory_threshold=0.85,
        queue_length_threshold=50,
        request_latency_threshold=3.0
    )
    
    # 创建大量不同优先级的测试请求（设计为跑满4张卡）
    print("\n[Step 2] 添加大量不同优先级的推理请求")
    prompts_with_priority = [
        # 高优先级请求 (priority=5)
        ("What is the capital of France?", 5),
        ("What is 2 + 2?", 5),
        ("Hello!", 5),
        
        # 中高优先级请求 (priority=4)
        ("Explain quantum computing in simple terms", 4),
        ("What is machine learning?", 4),
        
        # 中优先级请求 (priority=3)
        ("Write a Python function to calculate fibonacci numbers", 3),
        ("Explain what is an LLM", 3),
        ("What are the benefits of exercise?", 3),
        
        # 中低优先级请求 (priority=2)
        ("List all prime numbers under 100 and explain the Sieve of Eratosthenes algorithm", 2),
        ("Write a detailed explanation of blockchain technology", 2),
        ("Explain the theory of relativity", 2),
        ("What is the history of artificial intelligence?", 2),
        
        # 低优先级请求 (priority=1)
        ("Write a comprehensive essay on the evolution of computer programming languages from the 1950s to present day", 1),
        ("Provide a detailed analysis of distributed systems architecture, including CAP theorem implications", 1),
        ("Explain in detail how neural networks work, including backpropagation and gradient descent", 1),
        ("Write a thorough survey of quantum computing algorithms including Shor's and Grover's algorithms", 1),
    ]
    
    # 转换为 chat 格式
    formatted_prompts = []
    sampling_params_list = []
    for prompt, priority in prompts_with_priority:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted_prompt)
        sampling_params_list.append(SamplingParams(
            temperature=0.7, 
            max_tokens=50, 
            priority=priority
        ))
    
    print(f"  总请求数: {len(formatted_prompts)}")
    print(f"  优先级分布: 高(5)={sum(1 for _,p in prompts_with_priority if p==5)}, "
          f"中高(4)={sum(1 for _,p in prompts_with_priority if p==4)}, "
          f"中(3)={sum(1 for _,p in prompts_with_priority if p==3)}, "
          f"中低(2)={sum(1 for _,p in prompts_with_priority if p==2)}, "
          f"低(1)={sum(1 for _,p in prompts_with_priority if p==1)}")
    
    # 开始推理
    print("\n[Step 3] 开始推理（检测真实拥塞信号）")
    outputs = llm.generate(formatted_prompts, sampling_params_list)
    
    # 检查被丢弃的请求
    dropped_seqs = llm.get_dropped_sequences()
    print(f"\n[Step 4] 检查结果")
    print(f"  总请求数: {len(prompts_with_priority)}")
    print(f"  完成的请求数: {len(outputs)}")
    print(f"  被丢弃的请求数: {len(dropped_seqs)}")
    print(f"  被丢弃的请求ID: {dropped_seqs}")
    
    # 输出结果
    print("\n[Step 5] 输出结果（前10个）")
    for i, (prompt, output) in enumerate(zip(formatted_prompts[:10], outputs[:10])):
        priority = prompts_with_priority[i][1]
        print(f"\n请求 {i+1} (优先级{priority}):")
        print(f"  Prompt: {prompt[25:75]}...")
        print(f"  Completion: {output['text'][:100]}...")
    
    # 禁用 drop 机制
    print("\n[Step 6] 禁用 Request Drop 机制")
    llm.disable_drop_mechanism()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

if __name__ == "__main__":
    test_request_drop()