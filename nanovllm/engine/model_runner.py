import pickle
import torch
import random
import socket
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        port = self._get_available_port()
        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype or torch.float32)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # Layer-level drop mechanism (can be enabled by enable_layer_drop)
        self.layer_drop_enabled = False
        self.layer_drop_probability = 0.1
        
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
    def _get_available_port(self):
        """Find an available port for distributed communication.
        
        Uses the configured port if specified, otherwise tries to find an available port.
        Start from 2333 and increment if port is occupied.
        """
        port = getattr(self.config, 'distributed_port', 2333)
        
        # Try the configured port first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                # Port is occupied, try next ones
                pass
        
        # Try to find an available port
        for test_port in range(port + 1, port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', test_port))
                    print(f"Port {port} is in use, using available port {test_port}")
                    return test_port
                except OSError:
                    continue
        
        # If still can't find, raise error
        raise RuntimeError(f"Cannot find available port in range {port}-{port+99}")

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        dtype = hf_config.torch_dtype or torch.float32
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置layer-level drop回调（prefill阶段也支持）
        def layer_drop_callback(layer_idx, total_layers):
            return self._layer_level_drop(layer_idx, total_layers, seqs)
        
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    num_seqs=len(seqs), layer_drop_callback=layer_drop_callback)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # Ensure block_table is populated before accessing it
            if not seq.block_table:
                raise RuntimeError(f"Sequence {seq.seq_id} has empty block_table during decode phase. This indicates prefill was not completed.")
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 不设置 layer_drop_callback，关闭 decode 阶段的 drop 机制
        # Drop 只在 prefill 阶段生效
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, 
                    block_tables=block_tables, num_seqs=len(seqs))
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        if not seqs:
            return torch.tensor([], dtype=torch.float32, device="cuda")
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, device="cuda")

    def _layer_level_drop(self, layer_idx: int, total_layers: int, seqs: list[Sequence]) -> list[int]:
        """Layer-level请求丢弃机制
        
        在每层之后检查是否需要丢弃某些请求，实现细粒度的资源管理。
        Prefill阶段使用更激进的策略（计算量大），Decode阶段更保守。
        
        Args:
            layer_idx: 当前layer索引（从0开始）
            total_layers: 总layer数量
            seqs: 当前batch中的sequence列表
            
        Returns:
            需要被drop的sequence索引列表
        """
        if not hasattr(self, 'layer_drop_enabled') or not self.layer_drop_enabled:
            return []
        
        dropped_indices = []
        context = get_context()
        
        for i, seq in enumerate(seqs):
            # 检查是否已经被标记为drop
            if seq.status == SequenceStatus.DROPPED:
                dropped_indices.append(i)
                continue
            
            # Layer-level drop策略：
            # 1. 基于剩余层数计算优先级
            # 2. 越接近输出层，越不应该被drop（已经投入了很多计算）
            progress = (layer_idx + 1) / total_layers  # 已经完成的layer比例
            
            # Prefill阶段使用更激进的drop策略
            if context.is_prefill:
                # Prefill计算量大，早期层drop概率更高
                # 第0层：drop概率 = base_prob * 2.0
                # 最后一层：drop概率 = base_prob * 0.3
                drop_prob = self.layer_drop_probability * (2.0 - 1.7 * progress)
                
                # 优化：考虑prompt长度，但不对短prompt过度惩罚
                # 使用指数函数，让短prompt也有合理的drop概率
                # prompt=32 tokens → 0.37, prompt=256 → 0.78, prompt=512+ → 1.0
                if seq.num_prompt_tokens <= 32:
                    prompt_factor = 0.4  # 最短prompt也有40%的基础概率
                elif seq.num_prompt_tokens <= 128:
                    prompt_factor = 0.6  # 短prompt
                elif seq.num_prompt_tokens <= 256:
                    prompt_factor = 0.8  # 中等prompt
                elif seq.num_prompt_tokens <= 512:
                    prompt_factor = 1.0  # 标准prompt
                else:
                    prompt_factor = min(seq.num_prompt_tokens / 512, 2.0)  # 超长prompt额外惩罚
                drop_prob *= prompt_factor
            else:
                # Decode阶段更保守
                # 第0层：drop概率 = base_prob
                # 最后一层：drop概率 = base_prob * 0.1
                drop_prob = self.layer_drop_probability * (1 - 0.9 * progress)
            
            # 考虑sequence优先级
            priority_factor = (6 - seq.priority) / 3  # priority=1时为1.67, priority=5时为0.33
            adjusted_prob = drop_prob * priority_factor
            
            # Prefill阶段不考虑age_factor（还没有生成token）
            # Decode阶段考虑请求年龄
            if not context.is_prefill:
                age_factor = min(seq.num_completion_tokens / 10, 1.0)
                adjusted_prob *= (1 - age_factor * 0.5)
            
            # 调试日志：打印每一层的drop概率计算
            if self.layer_drop_probability >= 0.05:  # 概率较高时打印详细日志
                phase = "Prefill" if context.is_prefill else "Decode"
                print(f"[Layer Drop Debug] Seq {seq.seq_id} layer {layer_idx}/{total_layers} ({phase}): "
                      f"base={self.layer_drop_probability:.3f}, adjusted={adjusted_prob:.3f}, "
                      f"priority={seq.priority}, prompt={seq.num_prompt_tokens}t")
            
            if random.random() < adjusted_prob:
                dropped_indices.append(i)
                seq.status = SequenceStatus.DROPPED
                phase = "Prefill" if context.is_prefill else "Decode"
                print(f"[Layer Drop] Seq {seq.seq_id} dropped at layer {layer_idx}/{total_layers} ({phase}), "
                      f"progress={progress:.1%}, prob={adjusted_prob:.3f}, "
                      f"priority={seq.priority}, prompt={seq.num_prompt_tokens}t")
        
        return dropped_indices

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> tuple:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
                
        # 获取被drop的序列信息
        dropped_seq_ids = []
        if self.rank == 0:
            for seq in seqs:
                if seq.status == SequenceStatus.DROPPED:
                    dropped_seq_ids.append(seq.seq_id)
        
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids, dropped_seq_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(     # run_model 的时候也是复用，复写
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,  
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
