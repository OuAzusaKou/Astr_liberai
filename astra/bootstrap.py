"""Set reference precision before importing either policy or simulator entry."""
import runpy
import sys

import torch

torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
entry = sys.argv.pop(1)
sys.argv[0] = entry
runpy.run_path(entry, run_name='__main__')
