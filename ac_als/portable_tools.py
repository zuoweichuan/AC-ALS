"""Explicit external-tool execution on native POSIX or Windows/WSL."""
import os


def tool_command(*args,distro='Ubuntu'):
    if not args:raise ValueError('Missing external tool')
    executable=os.environ.get('AC_ALS_ABC','/usr/local/bin/abc')
    values=[executable if str(a)=='/usr/local/bin/abc' else str(a) for a in args]
    if os.name=='nt':
        return ['wsl','-d',os.environ.get('AC_ALS_WSL_DISTRO',distro),'--exec',*values]
    return values
