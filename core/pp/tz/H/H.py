import json
import numpy as np
from pyscf import lib, gto, scf, ao2mo, tools, fci, mcscf, cc
from pyscf.shciscf import shci, settings

atomstring = ""
atomstring += f"H 0 0 0;"

mol = gto.M(
atom = atomstring,
unit = "bohr",
cart = False,
basis = {
'H':gto.basis.parse(
'''
H S
23.843185  0.00411490
10.212443  0.01046440
4.374164  0.02801110
1.873529  0.07588620
0.802465  0.18210620
0.343709  0.34852140
0.147217  0.37823130
0.063055  0.11642410
H S
0.091791  1.00000000
H S
0.287637  1.00000000
H P
0.393954  1.00000000
H P
1.462694  1.00000000
H D
1.065841  1.00000000
'''
)
},
ecp = {
'H':gto.basis.parse_ecp(
'''
H nelec 0
H ul
1 21.24359 1.00000
3 21.24359 21.24359
2 21.77696 -10.85192
H S
2 1.00000 0.00000
'''
)
},
spin = 1,
verbose = 5
)
mf = scf.UHF(mol).newton()
norbs = mol.nao

dm = mf.get_init_guess()

dm = dm + (2.0 * np.random.rand(norbs, norbs) - 1.0) / 1000
mf.max_cycle = 100
mf.kernel(dm0 = dm)
mf.analyze()

mycc = mf.CCSD()
mycc.run()
print("\n")
print('CCSD correlation energy: ', mycc.e_corr)
print('CCSD total energy: ', mf.energy_tot() + mycc.e_corr)
