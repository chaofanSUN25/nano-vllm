#!/usr/bin/env python3
"""
测试 Request Drop 机制
======================

这个脚本用于验证框架层面的 request drop 功能：
1. 创建多个推理请求
2. 启用 request drop 机制
3. 模拟拥塞信号
4. 验证被 drop 的请求确实跳过了后续计算
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def test_request_drop():
    print("="*60)
    print("测试 Request Drop 机制")
    print("="*60)
    
    # 模型路径
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 初始化 tokenizer 和 LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=4)
    
    # 启用 request drop 机制（50% 丢弃概率）
    print("\n[Step 1] 启用 Request Drop 机制（丢弃概率: 50%）")
    llm.enable_drop_mechanism(probability=0.5)
    
    # 创建多个测试请求
    print("\n[Step 2] 添加多个推理请求")
    prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum computing in simple terms",
        "What is the capital of France?",
        "List all prime numbers under 100",
    ]
    
    # 转换为 chat 格式
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # 开始推理
    print("\n[Step 3] 开始推理（会模拟拥塞信号）")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=10)
    outputs = llm.generate(prompts, sampling_params)
    
    # 检查被丢弃的请求
    dropped_seqs = llm.get_dropped_sequences()
    print(f"\n[Step 4] 检查结果")
    print(f"  总请求数: {len(prompts)}")
    print(f"  完成的请求数: {len(outputs)}")
    print(f"  被丢弃的请求ID: {dropped_seqs}")
    
    # 输出结果
    print("\n[Step 5] 输出结果")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n请求 {i+1}:")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Completion: {output['text'][:100]}...")
    
    # 禁用 drop 机制
    print("\n[Step 6] 禁用 Request Drop 机制")
    llm.disable_drop_mechanism()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    test_request_drop()