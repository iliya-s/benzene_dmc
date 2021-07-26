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
),
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
),
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
spin = 0,
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
