"""Flattened combinational named-pin Verilog. Physical instances count once."""
from dataclasses import dataclass
from pathlib import Path
import re
from logic import evaluate

ZERO, ONE = "1'b0", "1'b1"


@dataclass
class Gate:
    name: str
    cell: str
    pins: dict


def canonical(s):
    s = s.strip()
    if s in ("1'b0", "1'h0", '0'): return ZERO
    if s in ("1'b1", "1'h1", '1'): return ONE
    if not re.fullmatch(r'(?:[A-Za-z_][\w$]*(?:\[\d+\])?|\\\S+)', s):
        raise ValueError('Unsupported signal/expression: '+s)
    return s


class Circuit:
    def __init__(self, lib, inputs, outputs, gates, aliases=None, name='resub'):
        self.lib=lib; self.inputs=tuple(inputs); self.outputs=tuple(outputs)
        self.gates=list(gates); self.aliases=dict(aliases or {}); self.name=name
        self.rebuild()

    @classmethod
    def read(cls, path, lib):
        text=Path(path).read_text(encoding='utf-8')
        text=re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
        statements=text.split(';'); ins=[]; outs=[]; aliases={}; gates=[]; name='resub'
        for stmt in statements:
            stmt=stmt.strip()
            if not stmt or stmt=='endmodule': continue
            if stmt.startswith('module '):
                name=re.match(r'module\s+(\S+)\s*\(',stmt)[1]; continue
            decl=re.match(r'(input|output|wire)\s+(.*)',stmt,re.S)
            if decl:
                kind, ports=decl.groups()
                if kind=='wire': continue
                bus=re.match(r'\[(\d+)\s*:\s*(\d+)\]\s*(.*)',ports,re.S)
                names=[canonical(n) for n in (bus[3] if bus else ports).split(',')]
                if bus:
                    lo,hi=sorted((int(bus[1]),int(bus[2])))
                    names=[n+'['+str(i)+']' for n in names for i in range(lo,hi+1)]
                (ins if kind=='input' else outs).extend(names); continue
            assign=re.fullmatch(r'assign\s+(\S+)\s*=\s*(.*?)',stmt,re.S)
            if assign:
                aliases[canonical(assign[1])]=canonical(assign[2]); continue
            inst=re.fullmatch(r'(\w+)\s+([^\s(]+)\s*\((.*)\)',stmt,re.S)
            if inst is None: raise ValueError('Unsupported Verilog statement: '+stmt[:100])
            cell,instname,connections=inst.groups()
            if cell not in lib.cells: raise ValueError('Unsupported/sequential cell: '+cell)
            matches=list(re.finditer(r'\.(\w+)\s*\(\s*([^()]*)\s*\)',connections))
            if re.sub(r'\s|,','',re.sub(r'\.\w+\s*\([^()]*\)','',connections)):
                raise ValueError('Malformed named-pin connections')
            pins={m[1]:canonical(m[2]) for m in matches if m[2].strip()}
            if len(pins)!=len([m for m in matches if m[2].strip()]): raise ValueError('Duplicate pin')
            c=lib.cells[cell]
            if set(c.inputs)-pins.keys(): raise ValueError('Unconnected inputs: '+instname)
            if set(pins)-set(c.inputs)-set(c.functions): raise ValueError('Unknown pin: '+instname)
            gates.append(Gate(instname,cell,pins))
        return cls(lib,sorted(ins),sorted(outs),gates,aliases,name)

    def resolve(self, net):
        seen=set()
        while net in self.aliases:
            if net in seen: raise ValueError('Alias cycle')
            seen.add(net); net=self.aliases[net]
        return net

    def rebuild(self):
        if len(set(self.inputs))!=len(self.inputs) or len(set(self.outputs))!=len(self.outputs):
            raise ValueError('Duplicate port')
        producer={}
        for i,g in enumerate(self.gates):
            for p in self.lib.cells[g.cell].functions:
                if p in g.pins:
                    net=g.pins[p]
                    if net in producer or net in self.inputs or net in self.aliases:
                        raise ValueError('Multiple drivers: '+net)
                    producer[net]=i
        visiting=set(); done=set(); order=[]
        def visit(net):
            net=self.resolve(net)
            if net in self.inputs or net in (ZERO,ONE): return
            if net not in producer: raise ValueError('Undriven net: '+net)
            i=producer[net]
            if i in done:return
            if i in visiting: raise ValueError('Combinational cycle: '+net)
            visiting.add(i); g=self.gates[i]
            for p in self.lib.cells[g.cell].inputs: visit(g.pins[p])
            visiting.remove(i); done.add(i); order.append(g)
        for net in self.outputs: visit(net)
        self.gates=order
        self.producer={g.pins[p]:g for g in order for p in self.lib.cells[g.cell].functions if p in g.pins}
        self.level={n:0 for n in (*self.inputs,ZERO,ONE)}
        self.fanouts={}
        for g in order:
            c=self.lib.cells[g.cell]
            deps=[self.resolve(g.pins[p]) for p in c.inputs]
            level=1+max((self.level[n] for n in deps),default=0)
            for p in c.functions:
                if p in g.pins:self.level[g.pins[p]]=level
            for n in deps:self.fanouts.setdefault(n,[]).append(g)
        self.units=sum(self.lib.cells[g.cell].units for g in order)

    def copy(self):
        return Circuit(self.lib,self.inputs,self.outputs,[Gate(g.name,g.cell,dict(g.pins)) for g in self.gates],self.aliases,self.name)

    def materialize(self, replacements):
        """Simultaneous modifications; later cleanup recomputes physical liveness."""
        result=self.copy(); used=set(result.producer)|set(result.inputs)|set(result.aliases)
        serial=0
        def fresh():
            nonlocal serial
            while True:
                n='rs_'+str(serial); serial+=1
                if n not in used: used.add(n); return n
        for target,divisors,plan in replacements:
            for g in result.gates:
                for p in result.lib.cells[g.cell].functions:
                    if g.pins.get(p)==target:g.pins[p]=fresh()
            memo={}
            def emit(p):
                if p.name=='leaf': return divisors[p.leaf] if p.leaf>=0 else (ZERO,ONE)[p.leaf+2]
                if p in memo:return memo[p]
                c=result.lib.cells[p.name]; out=fresh()
                pins=dict(zip(c.inputs,(emit(child) for child in p.children)))
                pins[next(iter(c.functions))]=out
                result.gates.append(Gate(fresh(),c.name,pins));memo[p]=out;return out
            # Always provide two leaf entries, even for a one-divisor function.
            divisors=tuple(divisors)+(ZERO,)*(2-len(divisors))
            result.aliases[target]=emit(plan)
        result.rebuild()
        return result

    def simplify(self):
        """Exact constant/cofactor propagation and cheaper <=2-signal mapping."""
        current=self
        while True:
            chosen=[]
            for g in current.gates:
                c=current.lib.cells[g.cell]
                if len(c.functions)!=1:continue
                outpin=next(iter(c.functions));target=g.pins[outpin]
                signals=tuple(dict.fromkeys(current.resolve(g.pins[p]) for p in c.inputs
                                            if current.resolve(g.pins[p]) not in (ZERO,ONE)))
                if len(signals)>2:continue
                vals={ZERO:0,ONE:15,**dict(zip(signals,(10,12)))}
                truth=evaluate(c.functions[outpin],{'0':0,'1':15,**{p:vals[current.resolve(g.pins[p])] for p in c.inputs}},15)
                plan=current.lib.mapper.plans[truth]
                if plan.units<c.units:
                    chosen.append((target,signals,plan))
            if not chosen:return current
            trial=current.materialize(chosen)
            if trial.units>=current.units:raise AssertionError('Exact simplification failed to reduce area')
            current=trial

    def write(self,path):
        def fmt(n):return n+' ' if n.startswith('\\') else n
        nets=set(self.inputs)|set(self.outputs)
        for g in self.gates:nets.update(g.pins.values())
        nets.update(self.aliases);nets.update(self.aliases.values());nets-={ZERO,ONE}
        # Flatten bit-select port names into escaped identifiers for legal scalar Verilog.
        def ver(n):return fmt('\\'+n if '[' in n and not n.startswith('\\') else n)
        lines=['module '+self.name+' ('+', '.join(map(ver,self.inputs+self.outputs))+');',
               '  input '+', '.join(map(ver,self.inputs))+';', '  output '+', '.join(map(ver,self.outputs))+';']
        wires=sorted(nets-set(self.inputs)-set(self.outputs))
        if wires:lines.append('  wire '+', '.join(map(ver,wires))+';')
        for i,g in enumerate(self.gates):
            lines.append('  '+g.cell+' cell_'+str(i)+' ('+', '.join('.'+p+'('+ver(n)+')' for p,n in g.pins.items())+');')
        # Only emit aliases that can be referenced by a retained gate or output.
        needed=set(self.outputs)|{n for g in self.gates for n in g.pins.values()}
        for n in sorted(needed & self.aliases.keys()):
            lines.append('  assign '+ver(n)+' = '+ver(self.resolve(n))+';')
        lines.append('endmodule');Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
