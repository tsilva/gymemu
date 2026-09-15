"""Compatibility entry point for gymemu upload-checkpoints."""

import sys

from gymemu.commands import upload_checkpoints as _command

if __name__ == "__main__":
    _command.main()
else:
    sys.modules[__name__] = _command
