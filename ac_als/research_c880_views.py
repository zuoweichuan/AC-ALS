"""Isolated ER experiment retaining exact fine-grain standard-cell search DAGs."""
from portable_tools import tool_command
import copy
import subprocess
import hashlib
from pathlib import Path
import er_search_resub as baseline
from er_repair_tools import ERWorker
from bacs_numeric import save
from prepare_resub_iscas import DEFAULT,normalized_author_blif,read
from bacs_data import read_blif
from numeric_logic_view import from_sop
from numeric_exact import ExactBackend,write_blif
from hybrid_identity import structural_id

def author_view(source,folder):
    raw=DEFAULT/'c880/author_initial/accurate_working.blif'
    mapping=read(DEFAULT/'baseline.json')['designs']['c880']['port_mapping']
    path=folder/'author_working_normalized.blif'
    normalized_author_blif(read_blif(raw),mapping,source,path)
    view=from_sop(path,source)
    write_blif(view,folder/'author_working_cells.blif');write_blif(source,folder/'source_for_cec.blif')
    q=lambda n:ExactBackend.quote(ExactBackend.linux(folder/n))
    cmd='cec %s %s'%(q('source_for_cec.blif'),q('author_working_cells.blif'))
    proc=subprocess.run(tool_command('/usr/local/bin/abc', '-c', cmd, distro='Ubuntu'),capture_output=True,timeout=15)
    log=(proc.stdout+proc.stderr).decode('utf-8',errors='replace');(folder/'view_cec.log').write_text(log,encoding='utf-8')
    if proc.returncode or 'Networks are equivalent' not in log:raise AssertionError('Accurate author work view CEC failed')
    return view

class Views:
    def add(self,circuit,origin,preserve):
        if len(self.active)>=4:raise ValueError('Branch slots full')
        bid='b%03d'%self.next_branch;self.next_branch+=1
        folder=self.folder/bid;folder.mkdir()
        options=copy.copy(self.args)
        options.logic_view=self.args.research_view=='expanded';options.sop_view=False;options.no_exact=False
        keep=options.logic_view or preserve
        w=ERWorker(circuit,self.order,self.inputs,self.refs,options,folder,remaining=self.remaining,
            preserve=keep,reference_source=self.source)
        if self.args.research_view=='author':
            w.current=author_view(self.source,folder);w.args.logic_view=True;w.preserve=True
            w.lowest_units=w.current.units;w.seen.add(structural_id(w.current));w.loss=0
            w.checkpoint('author_accurate_standard_cell_work_view',[]);w.annotate(len(w.history)-1)
            w.refine('author_work_initial_exact')
        self.workers[bid]=w;self.active.append(bid);self.origin[bid]=dict(origin,research_view=self.args.research_view)
        return bid

class MCTS(Views,baseline.MCTS):pass
class Agent(Views,baseline.Agent):pass

def run(a):
    # Invoke baseline validation without modifying any shared source or controller.
    original_mcts,original_agent=baseline.MCTS,baseline.Agent
    baseline.MCTS,baseline.Agent=MCTS,Agent
    try:
        code=baseline.run(a)
        own=Path(__file__);folder=Path(a.output)
        (folder/'source'/own.name).write_bytes(own.read_bytes())
        save(folder/'research_source.json',dict(path=str(own),sha256=hashlib.sha256(own.read_bytes()).hexdigest()))
        return code
    finally:baseline.MCTS,baseline.Agent=original_mcts,original_agent

if __name__=='__main__':
    p=baseline.parser();p.description=__doc__
    p.add_argument('--research-view',choices=('expanded','mapped','author'),default='expanded')
    a=p.parse_args()
    if a.design!='c880':p.error('This isolated validation is c880 only')
    raise SystemExit(run(a))
