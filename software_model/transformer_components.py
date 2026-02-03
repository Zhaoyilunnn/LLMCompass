"""Modular transformer building blocks to ease extensibility."""

from __future__ import annotations

from dataclasses import dataclass

from software_model.operators import Operator, Reshape, Concat, Transpose
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.utils import Tensor, DataType
from hardware_model.system import System


class BaseTPAttentionStage(Operator):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device_count: int,
        data_type: DataType,
        *,
        use_allreduce: bool,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        self.use_allreduce = use_allreduce


class TPInitStageMHA(BaseTPAttentionStage):
    """Tensor-parallel self-attention stage for initialization workloads."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device_count: int,
        data_type: DataType,
        *,
        use_allreduce: bool = True,
    ):
        super().__init__(
            d_model, n_heads, device_count, data_type, use_allreduce=use_allreduce
        )

        d = d_model

        # weights (per device)
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)

        # operators
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

    def __call__(self, X: Tensor) -> Tensor:
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        Q = self.Q_proj(X, self.Wq)
        K = self.K_proj(X, self.Wk)
        V = self.V_proj(X, self.Wv)
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])

        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])
        K_T = self.K_transpose(K, [0, 2, 3, 1])
        V_T = self.V_transpose(V, [0, 2, 1, 3])

        A = self.Q_mul_K(Q_T, K_T)
        A_prob = self.A_softmax(A)
        H = self.A_mul_V(A_prob, V_T)
        H = self.H_transpose(H, [0, 2, 1, 3])
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        H0 = self.H_matmul0(H, self.W0)
        H0 = self.layer_norm0(H0)
        if dev_cnt > 1 and self.use_allreduce:
            H0 = self.allreduce_mha(H0)
        return H0

    # === performance ===
    def roofline_model(self, system: System) -> float:
        device = system.device
        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        allreduce_latency = 0.0
        if self.device_count > 1 and self.use_allreduce:
            latency_val = self.allreduce_mha.simulate(system.interconnect)
            allreduce_latency = float(latency_val) if latency_val is not None else 0.0

        total = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + softmax_latency
            + layernorm_latency
            + allreduce_latency
        )
        self.roofline_latency = total
        return total

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        include_fixed_io_latency: bool = False,
        fixed_io_write_coeff: float = 1.0,
    ) -> float:
        device = system.device

        for op in [
            self.Q_proj,
            self.K_proj,
            self.V_proj,
            self.Q_mul_K,
            self.A_mul_V,
            self.H_matmul0,
        ]:
            if hasattr(op, "include_fixed_io_latency"):
                op.include_fixed_io_latency = include_fixed_io_latency
            if hasattr(op, "fixed_io_write_coeff"):
                op.fixed_io_write_coeff = fixed_io_write_coeff

        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        allreduce_latency = 0.0
        if self.device_count > 1 and self.use_allreduce:
            latency_val = self.allreduce_mha.simulate(system.interconnect)
            allreduce_latency = float(latency_val) if latency_val is not None else 0.0

        total = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + softmax_latency
            + layernorm_latency
            + allreduce_latency
        )
        self.latency = total
        return total

    def run_on_gpu(self) -> float:
        qkv_latency = self.Q_proj.run_on_gpu() * 3
        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()
        h_matmul0_latency = self.H_matmul0.run_on_gpu()
        softmax_latency = self.A_softmax.run_on_gpu()
        layernorm_latency = self.layer_norm0.run_on_gpu()

        total = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + softmax_latency
            + layernorm_latency
        )
        self.latency_on_gpu = total
        return total


class TPAutoregStageMHA(BaseTPAttentionStage):
    """Self-attention stage with KV cache support for generation workloads.

    Optionally models grouped-query attention (GQA) via ``num_kv_heads``. When
    ``num_kv_heads`` is provided and smaller than ``n_heads``, the compute
    structure stays the same but the key/value cache memory requirement can be
    scaled accordingly.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        device_count: int,
        data_type: DataType,
        *,
        use_allreduce: bool = True,
        num_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model, n_heads, device_count, data_type, use_allreduce=use_allreduce
        )

        if num_kv_heads is not None:
            if num_kv_heads <= 0 or num_kv_heads > n_heads:
                raise ValueError("num_kv_heads must be in (0, num_heads]")
            if n_heads % num_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")
            if num_kv_heads % device_count != 0:
                raise ValueError("num_kv_heads must be divisible by device_count")
            self.num_kv_heads = num_kv_heads
        else:
            # Default: standard MHA (KV heads == query heads).
            self.num_kv_heads = n_heads

        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        b, _, d = x.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # Cache tensors are shaped for the full per-head view used in the
        # compute graph. We track the logical KV cache size separately via
        # ``num_kv_heads`` in ``memory_requirement``.
        K_cache = Tensor([b, h // dev_cnt, d_h, seq_len], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, seq_len, d_h], self.data_type)

        q = self.Q_proj(x, self.Wq)
        k = self.K_proj(x, self.Wk)
        v = self.V_proj(x, self.Wv)
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])
        k_T = self.K_transpose(k, [0, 2, 3, 1])
        v_T = self.V_transpose(v, [0, 2, 1, 3])

        K_T = self.K_concat(K_cache, k_T, 3)
        V_T = self.V_concat(V_cache, v_T, 2)
        a = self.Q_mul_K(q_T, K_T)
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)
        h0 = self.H_transpose(h0, [0, 2, 1, 3])
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        h0 = self.H_matmul0(h0, self.W0)
        h0 = self.layer_norm0(h0)
        if dev_cnt > 1 and self.use_allreduce:
            h0 = self.allreduce_mha(h0)

        # Logical KV cache memory: scale to the configured number of KV heads.
        kv_heads_per_device = self.num_kv_heads // dev_cnt
        logical_K_elems = b * kv_heads_per_device * d_h * seq_len
        logical_V_elems = b * kv_heads_per_device * seq_len * d_h

        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + logical_K_elems * self.data_type.word_size
            + logical_V_elems * self.data_type.word_size
        )
        return h0

    def roofline_model(self, system: System) -> float:
        device = system.device
        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        allreduce_latency = 0.0
        if self.device_count > 1 and self.use_allreduce:
            latency_val = self.allreduce_mha.simulate(system.interconnect)
            allreduce_latency = float(latency_val) if latency_val is not None else 0.0
        total = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + softmax_latency
            + layernorm_latency
            + allreduce_latency
        )
        self.roofline_latency = total
        return total

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        include_fixed_io_latency: bool = False,
        fixed_io_write_coeff: float = 1.0,
    ) -> float:
        device = system.device
        for op in [
            self.Q_proj,
            self.K_proj,
            self.V_proj,
            self.Q_mul_K,
            self.A_mul_V,
            self.H_matmul0,
        ]:
            if hasattr(op, "include_fixed_io_latency"):
                op.include_fixed_io_latency = include_fixed_io_latency
            if hasattr(op, "fixed_io_write_coeff"):
                op.fixed_io_write_coeff = fixed_io_write_coeff

        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        allreduce_latency = 0.0
        if self.device_count > 1 and self.use_allreduce:
            latency_val = self.allreduce_mha.simulate(system.interconnect)
            allreduce_latency = float(latency_val) if latency_val is not None else 0.0
        total = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + softmax_latency
            + layernorm_latency
            + allreduce_latency
        )
        self.latency = total
        return total

    def run_on_gpu(self) -> float:
        qkv_latency = self.Q_proj.run_on_gpu() * 3
        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()
        h_matmul0_latency = self.H_matmul0.run_on_gpu()
        softmax_latency = self.A_softmax.run_on_gpu()
        layernorm_latency = self.layer_norm0.run_on_gpu()
        total = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + softmax_latency
            + layernorm_latency
        )
        self.latency_on_gpu = total
        return total


class TPFeedForwardStage(Operator):
    """Feed-forward stage shared by init/autoregressive workloads."""

    def __init__(
        self,
        d_model: int,
        device_count: int,
        data_type: DataType,
        *,
        ffn_multiplier: int = 4,
        use_allreduce: bool = True,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        if (ffn_multiplier * d_model) % device_count != 0:
            raise ValueError(
                "ffn_multiplier * d_model must be divisible by device_count"
            )

        self.d_model = d_model
        self.device_count = device_count
        self.ffn_multiplier = ffn_multiplier
        self.use_allreduce = use_allreduce
        self.hidden_per_device = (ffn_multiplier * d_model) // device_count

        self.W1 = Tensor([d_model, self.hidden_per_device], data_type)
        self.W2 = Tensor([self.hidden_per_device, d_model], data_type)

        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        assert d == self.d_model
        H1 = self.H_matmul1(x, self.W1)
        assert H1.shape == [b, s, self.hidden_per_device]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)
        H2 = self.layer_norm1(H2)
        if self.device_count > 1 and self.use_allreduce:
            H2 = self.allreduce_ffn(H2)
        return H2

    def roofline_model(self, system: System) -> float:
        device = system.device
        matmul1 = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        matmul2 = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        gelu = self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        layernorm = (
            self.layer_norm1.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        allreduce = 0.0
        if self.device_count > 1 and self.use_allreduce:
            latency_val = self.allreduce_ffn.simulate(system.interconnect)
            allreduce = float(latency_val) if latency_val is not None else 0.0
        total = matmul1 + gelu + matmul2 + layernorm + allreduce
        self.roofline_latency = total
        return total

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        include_fixed_io_latency: bool = False,
        fixed_io_write_coeff: float = 1.0,
    ) -> float:
        device = system.device
        for op in [self.H_matmul1, self.H_matmul2]:
            if hasattr(op, "include_fixed_io_latency"):
                op.include_fixed_io_latency = include_fixed_io_latency
            if hasattr(op, "fixed_io_write_coeff"):
                op.fixed_io_write_coeff = fixed_io_write_coeff

        matmul1 = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        matmul2 = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        gelu = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )
        layernorm = (
            self.layer_norm1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )
        allreduce = 0.0
        if self.device_count > 1 and self.use_allreduce:
            latency_val = self.allreduce_ffn.simulate(system.interconnect)
            allreduce = float(latency_val) if latency_val is not None else 0.0
        total = matmul1 + gelu + matmul2 + layernorm + allreduce
        self.latency = total
        return total

    def run_on_gpu(self) -> float:
        matmul1 = self.H_matmul1.run_on_gpu()
        matmul2 = self.H_matmul2.run_on_gpu()
        gelu = self.H_gelu.run_on_gpu()
        layernorm = self.layer_norm1.run_on_gpu()
        total = matmul1 + gelu + matmul2 + layernorm
        self.latency_on_gpu = total
        return total


class ModularTransformerBlockInitTP(Operator):
    """Composable transformer block for initialization workloads."""

    def __init__(
        self,
        attention_stage: TPInitStageMHA,
        feedforward_stage: TPFeedForwardStage,
    ):
        super().__init__(0, 0, 0, 0, attention_stage.data_type)
        self.attention_stage = attention_stage
        self.feedforward_stage = feedforward_stage
        self.data_type = attention_stage.data_type

    def __call__(self, X: Tensor) -> Tensor:
        return self.feedforward_stage(self.attention_stage(X))

    def roofline_model(self, system: System) -> float:
        attn = self.attention_stage.roofline_model(system)
        ffn = self.feedforward_stage.roofline_model(system)
        self.roofline_latency = attn + ffn
        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        include_fixed_io_latency: bool = False,
        fixed_io_write_coeff: float = 1.0,
    ) -> float:
        attn = self.attention_stage.compile_and_simulate(
            system,
            compile_mode,
            include_fixed_io_latency=include_fixed_io_latency,
            fixed_io_write_coeff=fixed_io_write_coeff,
        )
        ffn = self.feedforward_stage.compile_and_simulate(
            system,
            compile_mode,
            include_fixed_io_latency=include_fixed_io_latency,
            fixed_io_write_coeff=fixed_io_write_coeff,
        )
        self.latency = attn + ffn
        return self.latency

    def run_on_gpu(self) -> float:
        attn = self.attention_stage.run_on_gpu()
        ffn = self.feedforward_stage.run_on_gpu()
        self.latency_on_gpu = attn + ffn
        return self.latency_on_gpu


class ModularTransformerBlockAutoTP(Operator):
    """Composable transformer block for autoregressive workloads."""

    def __init__(
        self,TPAutoregStageMHA
        attention_stage: TPAutoregAttentionStage,
        feedforward_stage: TPFeedForwardStage,
    ):
        super().__init__(0, 0, 0, 0, attention_stage.data_type)
        self.attention_stage = attention_stage
        self.feedforward_stage = feedforward_stage
        self.data_type = attention_stage.data_type

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        return self.feedforward_stage(self.attention_stage(x, seq_len))

    def roofline_model(self, system: System) -> float:
        attn = self.attention_stage.roofline_model(system)
        ffn = self.feedforward_stage.roofline_model(system)
        self.roofline_latency = attn + ffn
        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        include_fixed_io_latency: bool = False,
        fixed_io_write_coeff: float = 1.0,
    ) -> float:
        attn = self.attention_stage.compile_and_simulate(
            system,
            compile_mode,
            include_fixed_io_latency=include_fixed_io_latency,
            fixed_io_write_coeff=fixed_io_write_coeff,
        )
        ffn = self.feedforward_stage.compile_and_simulate(
            system,
            compile_mode,
            include_fixed_io_latency=include_fixed_io_latency,
            fixed_io_write_coeff=fixed_io_write_coeff,
        )
        self.latency = attn + ffn
        return self.latency

    def run_on_gpu(self) -> float:
        attn = self.attention_stage.run_on_gpu()
        ffn = self.feedforward_stage.run_on_gpu()
        self.latency_on_gpu = attn + ffn
        return self.latency_on_gpu


# === Builder helpers ===


def build_tp_init_block(
    *,
    d_model: int,
    n_heads: int,
    device_count: int,
    data_type: DataType,
    ffn_multiplier: int = 4,
    use_attention_allreduce: bool = True,
    use_ffn_allreduce: bool = True,
) -> ModularTransformerBlockInitTP:
    attention = TPInitStageMHA(
        d_model,
        n_heads,
        device_count,
        data_type,
        use_allreduce=use_attention_allreduce,
    )
    feedforward = TPFeedForwardStage(
        d_model,
        device_count,
        data_type,
        ffn_multiplier=ffn_multiplier,
        use_allreduce=use_ffn_allreduce,
    )
    return ModularTransformerBlockInitTP(attention, feedforward)


def build_tp_autoreg_block(
    *,
    d_model: int,
    n_heads: int,
    device_count: int,
    data_type: DataType,
    ffn_multiplier: int = 4,
    use_attention_allreduce: bool = True,
    use_ffn_allreduce: bool = True,
) -> ModularTransformerBlockAutoTP:
    attention = TPAutoregAttentionStage(
        d_model,
        n_heads,
        device_count,
        data_type,
        use_allreduce=use_attention_allreduce,
    )
    feedforward = TPFeedForwardStage(
        d_model,
        device_count,
        data_type,
        ffn_multiplier=ffn_multiplier,
        use_allreduce=use_ffn_allreduce,
    )
    return ModularTransformerBlockAutoTP(attention, feedforward)
