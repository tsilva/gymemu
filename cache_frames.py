"""Compatibility entry point for gymemu cache-frames."""

import sys

from gymemu.commands import cache_frames as _command

if __name__ == "__main__":
    _command.main()
else:
    sys.modules[__name__] = _command
