"""RoLA as an FLA-style HF model. Importing this registers RoLA with the HF AutoClasses,
so flame (training) and lm-evaluation-harness (eval) work off-the-shelf."""
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from rola_hf.configuration_rola import RoLAConfig
from rola_hf.modeling_rola import RoLAForCausalLM, RoLAModel

AutoConfig.register(RoLAConfig.model_type, RoLAConfig, exist_ok=True)
AutoModel.register(RoLAConfig, RoLAModel, exist_ok=True)
AutoModelForCausalLM.register(RoLAConfig, RoLAForCausalLM, exist_ok=True)

__all__ = ["RoLAConfig", "RoLAModel", "RoLAForCausalLM"]
