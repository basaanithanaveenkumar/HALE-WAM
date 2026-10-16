"""Deployment of Halo-VLA policies to NVIDIA, Apple, AMD and Qualcomm targets.

Built on HaleBlocks' ``hale_core.deploy``. Importing this package (``custom_imports:
[Halo_VLA.deploy]`` in a deploy config) puts Halo-VLA's flat import roots on ``sys.path`` and
registers the adapters in the ``halo_vla`` scope. See ``docs/deploy.md``.
"""

from Halo_VLA.deploy.import_roots import ImportRoots

ImportRoots().ensure()

from Halo_VLA.deploy import adapters  # noqa: E402,F401 - registers the adapters
from Halo_VLA.deploy.model_factory import HaloModelFactory  # noqa: E402
from Halo_VLA.deploy.prompt_io_signature import PromptIOSignature  # noqa: E402
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS  # noqa: E402

__all__ = ["HALO_VLA_ADAPTERS", "HaloModelFactory", "ImportRoots", "PromptIOSignature", "adapters"]
