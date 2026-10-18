"""Latent extraction: what System 2 passes to System 1 (the key design choice of a dual system).

Every extractor maps System 2's encoding (see :meth:`System2.forward`) to ``tokens (B, K, D)``
and ``mask (B, K)`` (1 = real token). ``D`` is ``emb_dim`` except for ``middle_layer`` (the
decoder's raw width). Extractors register in :data:`LATENT_EXTRACTORS` under the
``DualSystemConfig.latent`` value they implement.
"""

from .act_token_latent import ActTokenLatent, _act_slice  # noqa: F401
from .bridge_projector_registry import BRIDGE_PROJECTORS  # noqa: F401
from .common import _text_slice, build_latent_extractor  # noqa: F401
from .language_latent import LanguageLatent  # noqa: F401
from .last_hidden_latent import LastHiddenLatent  # noqa: F401
from .latent_bridge import LatentBridge  # noqa: F401
from .latent_extractor import LatentExtractor  # noqa: F401
from .latent_registry import LATENT_EXTRACTORS  # noqa: F401
from .middle_layer_latent import MiddleLayerLatent  # noqa: F401
from .pooled_latent import PooledLatent  # noqa: F401

__all__ = [
    "BRIDGE_PROJECTORS",
    "LATENT_EXTRACTORS",
    "ActTokenLatent",
    "LanguageLatent",
    "LastHiddenLatent",
    "LatentBridge",
    "LatentExtractor",
    "MiddleLayerLatent",
    "PooledLatent",
    "build_latent_extractor",
]
