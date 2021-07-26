import json
import numpy as np
from pyscf import lib, gto, scf, ao2mo, tools, fci, mcscf, cc
from pyscf.shciscf import shci, settings

atomstring = ""
atomstring += f"C 0 0 0;"

mol = gto.M(
atom = atomstring,
unit = "bohr",
cart = False,
spin = 2,
basis = "ccpvdz",
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
et = mycc.ccsd_t()
print("\n")
print('CCSD correlation energy: ', mycc.e_corr)
print('CCSD total energy: ', mf.energy_tot() + mycc.e_corr)
print("\n")
print('CCSD(T) correlation energy: ', mycc.e_corr + et)
print('CCSD(T) total energy: ', mf.energy_tot() + mycc.e_corr + et)
