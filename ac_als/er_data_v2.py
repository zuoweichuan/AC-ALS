"""Same accurate ISCAS inputs and streams, plus isolated v2 source manifest."""
from er_data import ROOT, DESIGNS, source_path, matrix, preflight
from er_data import source_files as original_files
source_files=(*original_files,'er_data_v2.py','er_search_v2.py','er_repair_tools.py',
              'er_suite_v2.py','er_agent_knowledge_v2.json')
