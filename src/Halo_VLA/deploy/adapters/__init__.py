"""Halo-VLA deployment adapters (registered in the ``halo_vla`` scope of ``ADAPTERS``).

==============================  ============  ============================================
adapter                         system        graph
==============================  ============  ============================================
HaloPerceptionAdapter           single        observation + prompt -> action conditioning
HaloFlowActionStepAdapter       single        one flow-matching velocity evaluation
HaloFlowPolicyAdapter           single        perception + unrolled Euler sampler
DualSystem2EncoderAdapter       dual          System 2 -> latent tokens
DualSystem1ObserverAdapter      dual          latent + frame + state -> condition tokens
DualSystem1ActionStepAdapter    dual          one System 1 sampler step
DualSystem1PolicyAdapter        dual          System 1 condition + unrolled sampler
DualSystemPolicyAdapter         dual          System 2 + System 1 in one call
HaloLanguageModelAdapter        either        pretrained LLM -> HF checkpoint / text graph
==============================  ============  ============================================
"""

from Halo_VLA.deploy.adapters.dual_system1_action_step_adapter import DualSystem1ActionStepAdapter
from Halo_VLA.deploy.adapters.dual_system1_observer_adapter import DualSystem1ObserverAdapter
from Halo_VLA.deploy.adapters.dual_system1_policy_adapter import DualSystem1PolicyAdapter
from Halo_VLA.deploy.adapters.dual_system2_encoder_adapter import DualSystem2EncoderAdapter
from Halo_VLA.deploy.adapters.dual_system_policy_adapter import DualSystemPolicyAdapter
from Halo_VLA.deploy.adapters.halo_adapter_base import HaloAdapterBase
from Halo_VLA.deploy.adapters.halo_flow_action_step_adapter import HaloFlowActionStepAdapter
from Halo_VLA.deploy.adapters.halo_flow_policy_adapter import HaloFlowPolicyAdapter
from Halo_VLA.deploy.adapters.halo_language_model_adapter import HaloLanguageModelAdapter
from Halo_VLA.deploy.adapters.halo_perception_adapter import HaloPerceptionAdapter
from Halo_VLA.deploy.adapters.observation_prompt_adapter import ObservationPromptAdapter

__all__ = [
    "DualSystem1ActionStepAdapter",
    "DualSystem1ObserverAdapter",
    "DualSystem1PolicyAdapter",
    "DualSystem2EncoderAdapter",
    "DualSystemPolicyAdapter",
    "HaloAdapterBase",
    "HaloFlowActionStepAdapter",
    "HaloFlowPolicyAdapter",
    "HaloLanguageModelAdapter",
    "HaloPerceptionAdapter",
    "ObservationPromptAdapter",
]
