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
from itertools import count
from nanovllm.engine.sequence import Sequence
from transformers import AutoTokenizer


def test_request_drop():
    print("="*60)
    print("测试 Layer-Level 请求丢弃机制")
    print("="*60)
    
    # 模型路径
    model_path = os.path.expanduser("/usr/wkspace/Qwen3-0.6B")
    
    # 初始化 tokenizer 和 LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=32)

    # 在 LLM 初始化完成后再重置 Sequence.counter（warmup 已消耗了一些 seq_id）
    Sequence.counter = count(0)
    
    # 启用 Layer-Level Drop 机制（仅在 prefill 阶段生效）
    print("\n[Step 1] 启用 Prefill Layer-Level 请求丢弃机制")
    llm.enable_layer_drop(probability=0.15)  # 提高概率以便观察效果
    
    # 禁用传统的 request drop 机制（只保留 prefill layer drop）
    # llm.enable_drop_mechanism(
    #     probability=0.3, 
    #     strategy="hybrid",
    # )
    
    print("\n[Layer-Level Drop 策略说明]")
    print("  ├─ 在每个Decoder Layer之后进行drop决策")
    print("  ├─ 早期Layer: drop概率高（资源投入少）")
    print("  ├─ 晚期Layer: drop概率低（资源投入多）")
    print("  ├─ 低优先级请求: 更容易被drop")
    print("  └─ 已生成token多的请求: 更不容易被drop")
    
    # 创建混合负载请求
    print("\n[Step 2] 添加混合负载请求")
    prompts_with_priority = [
        # 高优先级请求
        ("What is the capital of France?", 5),
        ("What is 2 + 2?", 5),
        ("Hello!", 5),
        
        # 中优先级请求
        ("Explain quantum computing in simple terms", 4),
        ("What is machine learning?", 4),
        ("Write a Python function to calculate fibonacci numbers", 3),
        
        # 低优先级请求（长prompt，更容易被drop）
        ("List all prime numbers under 100 and explain Sieve algorithm", 2),
        ("Write a detailed explanation of blockchain technology", 2),
        ("Explain the theory of relativity", 2),
        ("Write a comprehensive essay on programming languages evolution", 1),
        ("Provide a detailed analysis of distributed systems architecture", 1),
        ("Explain neural networks including backpropagation", 1),
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
          f"低(2)={sum(1 for _,p in prompts_with_priority if p==2)}, "
          f"最低(1)={sum(1 for _,p in prompts_with_priority if p==1)}")
    
    # 开始推理
    print("\n[Step 3] 开始推理（Layer-Level Drop生效）")
    outputs = llm.generate(formatted_prompts, sampling_params_list)
    
    # 检查结果
    dropped_seqs = llm.scheduler.dropped_sequences
    print(f"\n[Step 4] Layer-Level Drop 结果分析")
    print(f"  总请求数: {len(prompts_with_priority)}")
    print(f"  完成的请求数: {len(outputs)}")
    print(f"  被丢弃的请求数: {len(dropped_seqs)}")
    
    # 打印被 drop 的 seq 的详细信息
    if dropped_seqs:
        dropped_seq_objects = llm.scheduler.dropped_sequence_objects
        print(f"\n  被丢弃的请求详情:")
        print(f"  {'-'*120}")
        print(f"  {'Seq ID':<8} {'Layer':<12} {'Phase':<10} {'Progress':<10} {'Drop Prob':<10} {'Priority':<10} {'Prompt Tokens':<15} {'Prompt':<40}")
        print(f"  {'-'*120}")
        
        for seq_id in sorted(dropped_seqs):
            # 获取被 drop 的序列对象
            seq = dropped_seq_objects.get(seq_id)
            
            # 找到对应的原始请求
            prompt = "Unknown"
            priority = 0
            prompt_tokens = 0
            for i, (p, pri) in enumerate(prompts_with_priority):
                if i == seq_id:
                    prompt = p
                    priority = pri
                    prompt_tokens = len(tokenizer.encode(p))
                    break
            
            # 获取 drop 详细信息
            if seq:
                drop_layer = seq.drop_layer if seq.drop_layer is not None else "-"
                drop_total = seq.drop_total_layers if seq.drop_total_layers is not None else "-"
                drop_phase = seq.drop_phase if seq.drop_phase else "-"
                drop_progress = f"{seq.drop_progress:.1%}" if seq.drop_progress is not None else "-"
                drop_prob = f"{seq.drop_probability:.3f}" if seq.drop_probability is not None else "-"
                layer_info = f"{drop_layer}/{drop_total}"
            else:
                layer_info = "N/A"
                drop_phase = "N/A"
                drop_progress = "N/A"
                drop_prob = "N/A"
            
            print(f"  {seq_id:<8} {layer_info:<12} {drop_phase:<10} {drop_progress:<10} {drop_prob:<10} {priority:<10} {prompt_tokens:<15} {prompt[:38]}...")
    
    # 输出结果
    print("\n[Step 5] 输出结果（前6个）")
    for i, output in enumerate(outputs[:6]):
        print(f"\n请求 {i+1}:")
        print(f"  Prompt: {output['prompt'][:50]}...")
        print(f"  Completion: {output['text'][:80]}...")
    
    print("\n" + "="*60)
    print("Layer-Level Drop 测试完成!")
    print("="*60)

if __name__ == "__main__":
    test_request_drop()