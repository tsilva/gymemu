"""Compatibility entry point for gymemu train."""

import sys

from gymemu.commands import train as _command

if __name__ == "__main__":
    _command.main()
else:
    sys.modules[__name__] = _command
