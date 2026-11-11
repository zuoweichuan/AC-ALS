"""This bit-vector workload does not use BLAS; avoid unused worker stacks."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[name]='1'
