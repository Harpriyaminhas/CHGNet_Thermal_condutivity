import numpy as np
from conductivity import class_kappa

"""
An example script to predict lattice thermal condutivity at 300 K using CHGNet
"""

kappa = class_kappa()
kappa.get_minikappa_phonopy(mesh_in=[50, 50, 50],  
                            sc_mat=np.eye(3) * 2,  
                            pm_mat=np.eye(3),
                            list_temp=[300.0],
                            name_pcell="POSCAR-unitcell",
                            name_ifc2nd="FORCE_CONSTANTS",
                            list_taufactor=[2.0])
