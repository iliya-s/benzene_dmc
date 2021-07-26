import json
import numpy as np
from pyscf import lib, gto, scf, ao2mo, tools, fci, mcscf, cc
from pyscf.shciscf import shci, settings

bondlengths = {'CC':2.636, 'CH':2.038}

atomstring = f""
for i in range(6):
    x = np.cos(i * np.pi / 3)
    y = np.sin(i * np.pi / 3)
    atomstring += f"C\t{bondlengths['CC']*x}\t{bondlengths['CC']*y}\t0.0;"
    atomstring += f"H\t{(bondlengths['CC']+bondlengths['CH'])*x}\t{(bondlengths['CC']+bondlengths['CH'])*y}\t0.0;"

mol = gto.M(
atom = atomstring,
unit = "bohr",
cart = False,
spin = 0,
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
