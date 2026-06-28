#!/usr/bin/env python3
"""
测试 Prefill 阶段的 Layer-Level Drop 机制
==========================================

专门测试 prefill 阶段的 layer-level drop 功能：
1. 在 prefill 计算过程中进行 drop 决策
2. 不同长度 prompt 的 drop 行为
3. 不同优先级请求的 drop 行为
"""
import os
from nanovllm import LLM, SamplingParams
from itertools import count
from nanovllm.engine.sequence import Sequence
from transformers import AutoTokenizer


def test_prefill_layer_drop():
    print("="*60)
    print("测试 Prefill 阶段的 Layer-Level Drop 机制")
    print("="*60)
    
    # 模型路径
    model_path = os.path.expanduser("/usr/wkspace/Qwen3-0.6B")
    
    # 初始化 tokenizer 和 LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=32)

    # 在 LLM 初始化完成后再重置 Sequence.counter
    Sequence.counter = count(0)
    
    # 启用 Prefill Layer-Level Drop 机制（高概率以便观察）
    print("\n[Step 1] 启用 Prefill Layer-Level Drop 机制")
    llm.enable_layer_drop(probability=0.15)  # 15%基础概率，在prefill阶段会更高
    
    print("\n[Prefill Layer-Level Drop 策略说明]")
    print("  ├─ 在 Prefill 阶段的每个 Decoder Layer 后进行 drop 决策")
    print("  ├─ 早期 Layer: drop概率 = base_prob * 1.5")
    print("  ├─ 晚期 Layer: drop概率 = base_prob * 0.2")
    print("  ├─ 长 prompt 请求: 更容易被 drop (考虑 prompt_tokens / 512)")
    print("  ├─ 低优先级请求: 更容易被 drop")
    print("  └─ Prefill 阶段 drop 更激进，因为计算量大")
    
    # 创建混合负载请求（不同长度和优先级）
    print("\n[Step 2] 添加混合负载请求（不同长度和优先级）")
    prompts_with_priority = [
        # 高优先级 + 短 prompt (最不容易被drop)
        ("What is 2 + 2?", 5),
        ("Hello!", 5),
        
        # 高优先级 + 长 prompt (较容易 drop，但优先级保护)
        ("Please provide a detailed explanation of how neural networks work, including the concepts of forward propagation, backward propagation, activation functions, loss functions, and optimization algorithms.", 5),
        
        # 中优先级 + 短 prompt
        ("What is the capital of France?", 4),
        ("Explain quantum computing.", 4),
        
        # 中优先级 + 长 prompt
        ("List all prime numbers under 100 and explain the Sieve of Eratosthenes algorithm step by step with examples and time complexity analysis.", 4),
        
        # 低优先级 + 短 prompt
        ("Hi there.", 2),
        ("Test.", 2),
        
        # 低优先级 + 长 prompt (最容易被 drop)
        ("Write a comprehensive essay about the history of artificial intelligence, from its inception in the 1950s to modern deep learning, covering all major milestones, key figures, breakthrough algorithms, and their impact on society.", 1),
        ("Provide a detailed analysis of distributed systems architecture, including consensus algorithms, fault tolerance, CAP theorem, and real-world applications.", 1),
        ("Explain the complete theory of special and general relativity, including the mathematical framework, experimental validations, and practical applications in GPS and astronomy.", 1),
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
            max_tokens=30,  # 短输出以加快测试
            priority=priority
        ))
    
    print(f"  总请求数: {len(formatted_prompts)}")
    print(f"\n  按优先级和长度分类:")
    for i, (prompt, priority) in enumerate(prompts_with_priority):
        prompt_tokens = len(tokenizer.encode(prompt))
        print(f"    请求 {i+1}: 优先级={priority}, prompt长度={prompt_tokens} tokens")
    
    # 开始推理
    print("\n[Step 3] 开始推理（Prefill Layer-Level Drop 生效）")
    print("  注意观察每个请求在不同 layer 的 drop 情况\n")
    
    outputs = llm.generate(formatted_prompts, sampling_params_list)
    
    # 统计结果
    dropped_seqs = llm.scheduler.dropped_sequences
    dropped_layer_count = llm.scheduler.dropped_layer_count
    
    print(f"\n[Step 4] Prefill Layer-Level Drop 结果分析")
    print(f"  总请求数: {len(prompts_with_priority)}")
    print(f"  完成的请求数: {len(outputs)}")
    print(f"  被 Layer Drop 丢弃的请求数: {len(dropped_seqs)}")
    print(f"  Layer Drop 总次数: {dropped_layer_count}")
    
    if dropped_seqs:
        print(f"\n  被丢弃的请求详情:")
        for seq_id in sorted(dropped_seqs):
            # 找到对应的原始请求
            for i, (prompt, priority) in enumerate(prompts_with_priority):
                if i == seq_id:
                    prompt_tokens = len(tokenizer.encode(prompt))
                    print(f"    Seq {seq_id}: 优先级={priority}, prompt长度={prompt_tokens} tokens")
                    print(f"             Prompt: {prompt[:60]}...")
                    break
    
    # 分析完成请求的特征
    print(f"\n  完成的请求特征:")
    for i, output in enumerate(outputs[:5]):
        prompt = prompts_with_priority[i][0]
        priority = prompts_with_priority[i][1]
        prompt_tokens = len(tokenizer.encode(prompt))
        print(f"    请求 {i+1}: 优先级={priority}, prompt长度={prompt_tokens} tokens")
        print(f"             Completion: {output['text'][:60]}...")
    
    # 统计 drop 率与优先级和长度的关系
    print(f"\n[Step 5] Drop 率分析")
    priority_drop_stats = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, (prompt, priority) in enumerate(prompts_with_priority):
        if i in dropped_seqs:
            priority_drop_stats[priority].append('dropped')
        else:
            priority_drop_stats[priority].append('completed')
    
    for priority in sorted(priority_drop_stats.keys(), reverse=True):
        stats = priority_drop_stats[priority]
        if stats:
            dropped = stats.count('dropped')
            total = len(stats)
            drop_rate = dropped / total * 100 if total > 0 else 0
            print(f"  优先级 {priority}: {dropped}/{total} 被 drop ({drop_rate:.1f}%)")
    
    print("\n" + "="*60)
    print("Prefill Layer-Level Drop 测试完成!")
    print("="*60)
    print("\n预期行为:")
    print("  1. 低优先级 + 长 prompt 的请求更容易被 drop")
    print("  2. 高优先级 + 短 prompt 的请求最不容易被 drop")
    print("  3. Prefill 阶段 drop 概率比 Decode 阶段更高")
    print("  4. 早期 layer drop 概率比晚期 layer 更高")


if __name__ == "__main__":
    test_prefill_layer_drop()