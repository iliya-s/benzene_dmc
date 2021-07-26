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
0.921552 1.000000
C s
0.132800 1.000000
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
0.376742 1.000000
C p
0.126772 1.000000
C d
1.141611 1.000000
C d
0.329486 1.000000
C f
0.773485 1.000000
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
