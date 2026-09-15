"""Compatibility entry point for gymemu save-start-state."""

import sys

from gymemu.commands import save_start_state as _command

if __name__ == "__main__":
    _command.main()
else:
    sys.modules[__name__] = _command
