"""RoLAConfig — HF config for the RoLA causal LM. Mirrors fla.models.gla.GLAConfig
(same backbone/MLP/norm/fuse fields) but the sequence mixer is RoLA, parameterized by
a `rola_instance` name + the routed-state geometry (num_states, d_qk, d_v, n_heads)."""
from transformers.configuration_utils import PretrainedConfig


class RoLAConfig(PretrainedConfig):

    model_type = 'rola'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # --- RoLA mixer (the only architecture-specific axis) ---
        rola_instance: str = 'rola-rla-asym',   # see rola.ROLA_INSTANCES
        num_states: int = 16,                    # routed states (nc); == rola num_chunks
        d_qk: int = 16,
        d_v: int = 16,
        num_heads: int = 8,
        # --- backbone (mirrors GLAConfig) ---
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,   # FLA convention (matches GLAConfig)
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.rola_instance = rola_instance
        self.num_states = num_states
        self.d_qk = d_qk
        self.d_v = d_v
        self.num_heads = num_heads

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True.")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
