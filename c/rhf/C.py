import json
import numpy as np
from pyscf import lib, gto, scf, ao2mo, tools, fci, mcscf
from pyscf.shciscf import shci, settings

atomstring = f"C 0 0 0;"

mol = gto.M(
atom = atomstring,
unit = "bohr",
cart = False,
basis = {
'C':gto.basis.parse(
'''
C s
13.073594 0.0051583
6.541187 0.0603424
4.573411 -0.1978471
1.637494 -0.0810340
0.819297 0.2321726
0.409924 0.2914643
0.231300 0.4336405
0.102619 0.2131940
0.051344 0.0049848
C s
0.127852 1.000000
C p
9.934169 0.0209076
3.886955 0.0572698
1.871016 0.1122682
0.935757 0.2130082
0.468003 0.2835815
0.239473 0.3011207
0.117063 0.2016934
0.058547 0.0453575
0.029281 0.0029775
C p
0.149161 1.000000
C d
0.561160 1.000000
'''
)
},
ecp = {
'C':gto.basis.parse_ecp(
'''
C nelec 2
C ul
1 14.43502 4.00000
3 8.39889 57.74008
2 7.38188 -25.81955
C S
2 7.76079 52.13345
'''
)
},
spin = 2,
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

asAO = mol.search_ao_label(["C 2s", "C 2p"])
f = open("asAO.txt", 'w')
for i in range(len(asAO)):
    f.write(f'{asAO[i]}\t')

fileHF = open("hf.txt", 'w')
for i in range(norbs):
    for j in range(norbs):
        fileHF.write('%16.10e '%(mf.mo_coeff[i,j]))
    fileHF.write('\n')
