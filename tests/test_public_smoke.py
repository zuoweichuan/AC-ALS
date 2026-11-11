import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ac_als'))
from logic import Library
from netlist import Circuit
from simulate import pack,trace,outputs,validate_numpy

TOY_LIB='''library(toy) {
cell(AND2) { area: 1.0; pin(A) {direction: input;} pin(B) {direction: input;}
pin(Y) {direction: output; function: "A & B";} }
cell(INV) { area: 1.0; pin(A) {direction: input;}
pin(Y) {direction: output; function: "!A";} }
}'''


class PublicSmokeTests(unittest.TestCase):
    def test_staged_sources_parse(self):
        for path in sorted((ROOT/'ac_als').glob('*.py')):
            with self.subTest(file=path.name):ast.parse(path.read_text(encoding='utf-8-sig'))

    def test_bit_parallel_and_numpy_simulators_agree_on_original_toy(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);lib_path=folder/'toy.lib';netlist=folder/'toy.v'
            lib_path.write_text(TOY_LIB,encoding='utf-8')
            netlist.write_text('module toy(a,b,y); input a,b; output y; AND2 g(.A(a),.B(b),.Y(y)); endmodule',encoding='utf-8')
            lib=Library(lib_path);c=Circuit.read(netlist,lib)
            matrix=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.uint8)
            values=outputs(c,trace(c,pack(matrix,c.inputs),len(matrix)))
            expected=np.array([0,0,0,1],dtype=np.uint8)
            np.testing.assert_array_equal(validate_numpy(c,matrix)[:,0],expected)
            self.assertEqual(values['y'],8)


if __name__=='__main__':unittest.main()
