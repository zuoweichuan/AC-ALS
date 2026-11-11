"""Independent combinational Liberty reader and <=2-divisor cell mapper."""
from dataclasses import dataclass
from decimal import Decimal
from functools import reduce
from math import gcd
from pathlib import Path
import itertools
import re


def blocks(text, kind):
    pattern = re.compile(r'\b' + kind + r'\s*\(\s*([^()]+?)\s*\)\s*\{')
    pos = 0
    while (m := pattern.search(text, pos)) is not None:
        start = m.end(); depth = 1; i = start; quoted = False
        while depth and i < len(text):
            c = text[i]
            if c == '"' and text[i-1] != '\\': quoted = not quoted
            if not quoted:
                depth += (c == '{') - (c == '}')
            i += 1
        if depth: raise ValueError('Unbalanced Liberty block')
        yield m.group(1).strip('"'), text[start:i-1]
        pos = i


def expression(source):
    """Parse Liberty syntax, never execute library text as Python."""
    tokens = re.findall(r'[A-Za-z_][\w.$]*|[01]|[!~&*+|^()\x27]', source)
    if re.sub(r'\s+', '', source) != ''.join(tokens):
        raise ValueError('Unsupported Boolean syntax: ' + source)
    i = 0
    def atom():
        nonlocal i
        t = tokens[i]; i += 1
        if t in ('!', '~'): n = ('not', atom())
        elif t == '(':
            n = parse(0)
            if tokens[i] != ')': raise ValueError(source)
            i += 1
        elif re.fullmatch(r'[\w.$]+', t): n = ('var', t)
        else: raise ValueError(source)
        while i < len(tokens) and tokens[i] == "'":
            n = ('not', n); i += 1
        return n
    priorities = {'|':1, '+':1, '^':2, '&':3, '*':3}
    def parse(p):
        nonlocal i
        n = atom()
        while i < len(tokens) and priorities.get(tokens[i], -1) >= p:
            op = tokens[i]; i += 1
            n = ({'+':'or','|':'or','^':'xor','&':'and','*':'and'}[op], n,
                 parse(priorities[op]+1))
        return n
    result = parse(0)
    if i != len(tokens): raise ValueError(source)
    return result


def evaluate(tree, values, mask):
    op = tree[0]
    if op == 'var': return values[tree[1]]
    if op == 'not': return evaluate(tree[1], values, mask) ^ mask
    a, b = (evaluate(x, values, mask) for x in tree[1:])
    if op == 'and': return a & b
    if op == 'or': return a | b
    if op == 'xor': return a ^ b
    raise ValueError(op)


@dataclass(frozen=True)
class Cell:
    name: str
    area_decimal: Decimal
    inputs: tuple
    functions: dict
    tables: dict
    units: int = 0


class Library:
    def __init__(self, path):
        text = re.sub(r'/\*.*?\*/', '', Path(path).read_text(encoding='utf-8'), flags=re.S)
        rows = []
        for name, body in blocks(text, 'cell'):
            if re.search(r'\b(ff|latch)\s*\(', body): continue
            area = re.search(r'\barea\s*:\s*([\d.eE+-]+)', body)
            if area is None: continue
            inputs = []; funcs = {}
            for pin, pbody in blocks(body, 'pin'):
                direction = re.search(r'\bdirection\s*:\s*(\w+)', pbody)
                if direction is None: continue
                if direction[1] == 'input': inputs.append(pin)
                if direction[1] == 'output':
                    fun = re.search(r'\bfunction\s*:\s*"([^"]+)"', pbody)
                    if fun is not None and not re.search(r'\bthree_state\s*:', pbody):
                        funcs[pin] = expression(fun[1])
            if funcs: rows.append((name, Decimal(area[1]), tuple(inputs), funcs))
        scale = 10 ** max(-r[1].as_tuple().exponent for r in rows)
        base = reduce(gcd, (int(r[1]*scale) for r in rows if r[1] > 0))
        self.unit = Decimal(base) / scale
        self.cells = {}
        for name, area, ins, funcs in rows:
            tables = {}
            for pin, tree in funcs.items():
                tables[pin] = sum(evaluate(tree, {**{'0':0,'1':1}, **dict(zip(ins,bits))}, 1) << j
                    for j,bits in enumerate(itertools.product((0,1), repeat=len(ins))))
            self.cells[name] = Cell(name,area,ins,funcs,tables,int(area/self.unit))
        if not self.cells: raise ValueError('No combinational cells')
        self.mapper = Mapper(self)

    def area(self, units): return float(self.unit * units)


@dataclass(frozen=True)
class Plan:
    # A leaf is ('leaf', index), constants index=2/3 for 0/1.
    name: str
    children: tuple = ()
    leaf: int = -1
    units: int = 0


class Mapper:
    """Precompute the fixed 16 Boolean functions, not a circuit LAC enumeration.

    Tree-cost relaxation gives a small real-cell implementation, without a claim
    of globally optimal technology mapping. Materialization shares equal trees.
    """
    def __init__(self, lib):
        self.lib = lib
        self.plans = {10:Plan('leaf',leaf=0), 12:Plan('leaf',leaf=1),
                      0:Plan('leaf',leaf=-2), 15:Plan('leaf',leaf=-1)}
        cells = {}
        for c in lib.cells.values():
            if len(c.inputs) not in (1,2) or len(c.functions) != 1: continue
            pin, tree = next(iter(c.functions.items()))
            truth = evaluate(tree, {**{'0':0,'1':15}, **dict(zip(c.inputs,(10,12)))},15)
            key = (len(c.inputs),truth)
            if key not in cells or c.units < cells[key].units: cells[key] = c
        while True:
            changed = False; known = tuple(self.plans.items())
            for c in cells.values():
                tree = next(iter(c.functions.values()))
                for args in itertools.product(known, repeat=len(c.inputs)):
                    truth = evaluate(tree, {**{'0':0,'1':15}, **dict(zip(c.inputs,(a[0] for a in args)))},15)
                    cost = c.units + sum(a[1].units for a in args)
                    if truth not in self.plans or cost < self.plans[truth].units:
                        self.plans[truth] = Plan(c.name,tuple(a[1] for a in args), units=cost)
                        changed = True
            if not changed: break
        if len(self.plans) != 16: raise ValueError('Library cannot implement all 2-input functions')

    def complete(self, care, values):
        choices = [(p.units,truth,p) for truth,p in self.plans.items() if (truth & care) == values]
        return min(choices,key=lambda row:(row[0],row[1]))[1:]
