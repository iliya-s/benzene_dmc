import json
import numpy as np
from pyscf import lib, gto, scf, ao2mo, tools, fci, mcscf
from pyscf.shciscf import shci, settings

atomstring = f"H 0 0 0;"

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
0.139013  1.00000000
H P
0.740212  1.00000000
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

gto.write_gto(mol)
gto.write_pp(mol)

#RHF
mf = scf.RHF(mol).newton()
norbs = mol.nao

dm = mf.get_init_guess()

##uncomment to use current another set of orbitals as guess for density matrix
#mo_occ = np.zeros(norbs)
#nocc = mf.mol.nelectron // 2
#mo_occ[:nocc] = 2
#mo_coeff = np.zeros((norbs, norbs))
#
#f = open("hf.txt", 'r')
#row = 0
#for line in f:
#    col = 0
#    for coeff in line.split():
#        mo_coeff[row, col]  = float(coeff)
#        col = col + 1
#    row = row + 1
#f.close()
#dm = mf.make_rdm1(mo_coeff, mo_occ)

dm = dm + (2.0 * np.random.rand(norbs, norbs) - 1.0) / 1000
mf.max_cycle = 100
mf.kernel(dm0 = dm)
mf.analyze()

asAO = mol.search_ao_label(["H 1s"])
f = open("asAO.txt", 'w')
for i in range(len(asAO)):
    f.write(f'{asAO[i]}\t')

fileHF = open("hf.txt", 'w')
for i in range(norbs):
    for j in range(norbs):
        fileHF.write('%16.10e '%(mf.mo_coeff[i,j]))
    fileHF.write('\n')
