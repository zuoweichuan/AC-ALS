"""Optional exact ABC backend. No approximate search is delegated to ABC.

Every accepted mapping has a combinational equivalence proof against its own
input, plus a raw-Liberty area check. All generated files live in the run folder.
"""
from portable_tools import tool_command
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from logic import evaluate
from netlist import Circuit, Gate, ZERO, ONE
from cutmap import variable


def write_blif(circuit, path):
    """Expand Liberty functions, including both outputs of a physical FA."""
    ports = {n: 'pi_%d' % i for i, n in enumerate(circuit.inputs)}
    names = dict(ports)
    names.update({ZERO: 'const_zero', ONE: 'const_one'})
    for n in circuit.producer:
        if n not in names:
            names[n] = 'net_%d' % len(names)
    def wire(n):
        return names[circuit.resolve(n)]
    lines = ['.model numeric_exact', '.inputs ' + ' '.join(ports.values()),
             '.outputs ' + ' '.join('po_%d' % i for i in range(len(circuit.outputs))),
             '.names const_zero', '.names const_one', '1']
    for g in circuit.gates:
        cell = circuit.lib.cells[g.cell]
        k = len(cell.inputs)
        if k > 10:
            raise ValueError('Exact Liberty expansion restricted to <=10 cell inputs')
        mask = (1 << (1 << k)) - 1
        values = {'0': 0, '1': mask,
                  **{p: variable(k, i) for i, p in enumerate(cell.inputs)}}
        for pin, fun in cell.functions.items():
            if pin not in g.pins:
                continue
            lines.append('.names ' + ' '.join([wire(g.pins[p]) for p in cell.inputs] + [names[g.pins[pin]]]))
            truth = evaluate(fun, values, mask)
            for row in range(1 << k):
                if (truth >> row) & 1:
                    lines.append((''.join(str((row >> i) & 1) for i in range(k)) + ' ' if k else '') + '1')
    for i, n in enumerate(circuit.outputs):
        lines += ['.names %s po_%d' % (wire(n), i), '1 1']
    lines.append('.end')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def restore_ports(path, source):
    mapped = Circuit.read(path, source.lib)
    rename = {'pi_%d' % i: n for i, n in enumerate(source.inputs)}
    rename.update({'po_%d' % i: n for i, n in enumerate(source.outputs)})
    if set(mapped.inputs) != {'pi_%d' % i for i in range(len(source.inputs))}:
        raise ValueError('ABC changed input interface')
    if set(mapped.outputs) != {'po_%d' % i for i in range(len(source.outputs))}:
        raise ValueError('ABC changed output interface')
    # Internal mapped names must never collide with the original interface.
    used = set(source.inputs) | set(source.outputs) | set(source.aliases) | set(source.producer)
    for n in sorted(set(mapped.producer) | set(mapped.aliases) | set(mapped.aliases.values())):
        if n not in rename and n not in (ZERO, ONE):
            name = '__mapped_%d' % len(rename)
            while name in used:
                name += '_'
            rename[n] = name
            used.add(name)
    def r(n):
        return rename.get(n, n)
    return Circuit(source.lib, source.inputs, source.outputs,
                   [Gate(g.name, g.cell, {p: r(n) for p, n in g.pins.items()}) for g in mapped.gates],
                   {r(n): r(v) for n, v in mapped.aliases.items()}, source.name)


class ExactBackend:
    # Standard ABC exact rewriting/resubstitution, explicitly expanded so the
    # installed ABC does not depend on an author's abc.rc aliases.
    COMPRESS = ('balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; '
                'refactor -l; resub -K 8 -l; balance -l; resub -K 8 -N 2 -l; '
                'rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; '
                'balance -l; resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; '
                'rewrite -z -l; balance -l')
    SCRIPTS = (
        'strash; ' + COMPRESS + '; dch; amap',
        'strash; ' + COMPRESS + '; map -a',
    )

    def __init__(self, folder, library, *, distro='Ubuntu', executable='/usr/local/bin/abc', timeout=8, keep_sop=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.library = Path(library).resolve()
        self.distro, self.executable, self.timeout = distro, executable, timeout
        self.rows = []
        self.keep_sop=keep_sop
        self.working=None

    @staticmethod
    def linux(path):
        path = Path(path).resolve()
        if not path.drive:
            return path.as_posix()
        return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]

    @staticmethod
    def quote(path):
        value = str(path)
        if any(c in value for c in ('"', '\n', '\r', ';')):
            raise ValueError('Unsupported ABC path')
        return '"' + value + '"'

    def optimize(self, source, remaining, variants=2):
        best = source
        self.working=None
        for script in self.SCRIPTS[:variants]:
            if remaining() < .5:
                break
            folder = self.folder / ('map_%04d' % len(self.rows))
            folder.mkdir()
            write_blif(source, folder / 'input.blif')
            q = lambda name: self.quote(self.linux(folder / name))
            if self.keep_sop:
                script=script.replace(self.COMPRESS,self.COMPRESS+'; logic; sop; write_blif '+q('simplified.blif')+'; strash')
            proof=('cec %s %s; '%(q('input.blif'),q('simplified.blif'))) if self.keep_sop else ''
            command = ('read_lib -w %s; read_blif %s; %s; write_verilog %s; write_blif %s; cec %s %s' %
                       (self.quote(self.linux(self.library)), q('input.blif'), script,
                        q('mapped.v'), q('mapped.blif'), q('input.blif'), q('mapped.blif')))
            if proof:
                command+='; '+proof
            row = dict(script=script, source_units=source.units, accepted=False,
                       source_blif_sha256=hashlib.sha256((folder/'input.blif').read_bytes()).hexdigest())
            started = time.perf_counter()
            try:
                # No shell interpolation; ABC commands/paths are validated above.
                proc = subprocess.run(tool_command(self.executable, '-c', command, distro=self.distro),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      timeout=min(self.timeout, max(.1, remaining())))
                log = proc.stdout.decode('utf-8', errors='replace').replace('\x00', '')
                (folder/'abc.log').write_text(log, encoding='utf-8')
                proofs=len(re.findall(r'Networks are equivalent',log))
                equivalent = proofs>=(2 if self.keep_sop else 1)
                row.update(returncode=proc.returncode, cec_equivalent=equivalent)
                if proc.returncode or not equivalent:
                    raise ValueError('Exact mapping rejected: no completed CEC proof; see abc.log')
                trial = restore_ports(folder/'mapped.v', source)
                row['mapped_units'] = trial.units
                if self.keep_sop:
                    from numeric_logic_view import from_sop
                    working=from_sop(folder/'simplified.blif',source)
                    row.update(working_cec_equivalent=True,working_units=working.units,working_gates=len(working.gates))
                    if self.working is None or working.units<self.working.units:
                        self.working=working
                if trial.units < best.units:
                    best = trial
                    row['accepted'] = True
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                row['error'] = str(exc)
            row['seconds'] = time.perf_counter() - started
            self.rows.append(row)
            (self.folder/'mapping.json').write_text(json.dumps(self.rows, indent=2)+'\n', encoding='utf-8')
        return best
