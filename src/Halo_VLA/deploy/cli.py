"""``hale-wam-deploy``: the HaleBlocks deployment CLI with Halo-VLA's adapters registered."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from hale_core.deploy.cli import DeployCLI


class HaleWamDeployCLI(DeployCLI):
    """``hale-deploy`` with ``Halo_VLA.deploy`` imported first (so ``list`` shows the VLA adapters
    and configs may omit ``custom_imports``)."""

    def __init__(self) -> None:
        import Halo_VLA.deploy  # noqa: F401 - registers the halo_vla adapters

        super().__init__(prog="hale-wam-deploy")


def main(argv: Sequence[str] | None = None) -> int:
    return HaleWamDeployCLI().main(argv)


if __name__ == "__main__":
    sys.exit(main())
