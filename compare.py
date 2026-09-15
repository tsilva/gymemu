"""Compatibility entry point for gymemu compare."""

import sys

from gymemu.commands import compare as _command

if __name__ == "__main__":
    _command.main()
else:
    sys.modules[__name__] = _command
