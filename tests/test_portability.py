import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ac_als'))
from portable_assets import asset_root, contained, load_assets
from portable_tools import tool_command


class PortabilityTests(unittest.TestCase):
    def test_frozen_child_uses_original_root(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            os.environ['NETLIST_FRONTIER_ROOT'] = tmp
            self.assertEqual(asset_root(), (Path(tmp) / 'benchmark_assets').resolve())

    def test_changed_assets_and_path_escape_are_rejected(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'toy.txt').write_bytes(b'original')
            manifest = {'schema': 1, 'files': {'toy.txt': hashlib.sha256(b'original').hexdigest()}}
            (folder / 'bundle.json').write_text(json.dumps(manifest))
            with patch.dict(os.environ, {'AC_ALS_ASSETS': tmp}):
                self.assertEqual(load_assets()[1], manifest)
                (folder / 'toy.txt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    load_assets()
            with self.assertRaisesRegex(ValueError, 'escapes'):
                contained(folder, '../outside')

    def test_configured_tool_on_windows_and_posix(self):
        env = {'AC_ALS_ABC': '/opt/abc', 'AC_ALS_WSL_DISTRO': 'Research'}
        with patch.dict(os.environ, env), patch('portable_tools.os.name', 'nt'):
            self.assertEqual(tool_command('/usr/local/bin/abc', '-c', 'strash'),
                             ['wsl', '-d', 'Research', '--exec', '/opt/abc', '-c', 'strash'])
        with patch.dict(os.environ, env), patch('portable_tools.os.name', 'posix'):
            self.assertEqual(tool_command('/usr/local/bin/abc', '-c', 'strash'),
                             ['/opt/abc', '-c', 'strash'])


if __name__ == '__main__':
    unittest.main()
