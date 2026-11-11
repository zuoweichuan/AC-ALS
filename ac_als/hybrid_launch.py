"""Launch saved worker code so live algorithm edits cannot alter a running trial."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--tool',choices=['controller','provider','mcts','act','observe','verify'],default='controller')
 args,rest=p.parse_known_args();folder=Path(args.session).resolve();manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
 name={'controller':'hybrid_controller.py','provider':'hybrid_provider.py','mcts':'hybrid_mcts.py'}.get(args.tool,'hybrid_agent.py')
 script=folder/'source'/name
 if not script.exists() or hashlib.sha256(script.read_bytes()).hexdigest()!=manifest['code'][name]:raise RuntimeError('Invalid frozen entry point')
 env=dict(os.environ,NETLIST_FRONTIER_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
 command=[sys.executable,'-u','-B',str(script)]
 if args.tool not in ('controller','provider','mcts'):command.append(args.tool)
 result=subprocess.run([*command,'--session',str(folder),*rest],cwd=ROOT,env=env)
 raise SystemExit(result.returncode)
if __name__=='__main__':main()
