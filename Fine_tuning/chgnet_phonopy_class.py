import os
import torch
import numpy as np
import pickle

from phonopy import Phonopy
from phonopy.file_IO import (
                                write_FORCE_CONSTANTS,
                                )

from chgnet.model.model import CHGNet
from pymatgen.core import Structure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from jarvis.core.kpoints import Kpoints3D

from pymatgen.io.phonopy import get_pmg_structure, get_phonopy_structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from pymatgen.io.ase import AseAtomsAdaptor


class chgnet_phonopy:
        def __init__(
                        self,
                        structure:Structure,
                        path='.',
                        supercell_dims=[2,2,2],
                        ):
                self.structure=structure
                self.phonopy_structure=get_phonopy_structure(self.structure)
                self.jarvis_atoms=JarvisAtomsAdaptor.get_atoms(self.structure)

                self.path=path

                self.supercell_dims=supercell_dims
                # create supercell attribute in object through the supercell function
                self.supercell=self.create_supercell()


               def create_supercell(
                        self
                        ):
                new_structure=self.structure.copy()
                new_structure.make_supercell(self.supercell_dims)
                supercell_name=os.path.join(self.path, 'SPOSCAR_'+str(self.supercell_dims[0])+str(self.supercell_dims[1])+str(self.supercell_dims[2]))
                new_structure.to(filename=supercell_name)
                return new_structure


        
        def get_jarvis_kpoints(
                                self,
                                line_density=20,
                                ):
                kpoints = Kpoints3D().kpath(self.jarvis_atoms, line_density=line_density)
                return kpoints


       
        def save_to_pickle(
                                self,
                                filename='chgnet_phonopy_attrs.pkl',
                                ):
                filename=os.path.join(self.path, filename)
                with open(filename, 'wb') as outp:
                        pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


        
        def get_phonon_fc2(
                                self,
                                displacement=0.01,
                                num_snapshots=None,
                                write_fc=True,
                                output_POSCARs=False,
                                pretrained_model=True,
                                trained_path='./'
                                ):
                if pretrained_model:
                #chgnet = CHGNet.load()
                #else:
                        chgnet_model_path = 'best_model.pth.tar'
                        state_dict = torch.load(chgnet_model_path)
                        chgnet = CHGNet()  # Instantiate the model architecture
                        chgnet.load_state_dict(state_dict)  # Load weights
                        chgnet.eval()
#                        chgnet = CHGNet.from_file(chgnet_model_path)
                phonon = Phonopy(self.phonopy_structure, [[self.supercell_dims[0], 0, 0], [0, self.supercell_dims[1], 0], [0, 0, self.supercell_dims[2]]])
                phonon.generate_displacements(distance=displacement, number_of_snapshots=num_snapshots)
                supercells = phonon.get_supercells_with_displacements()
                set_of_forces = []
                disp = 0
                for i_scell, scell in enumerate(supercells):
                        scell_pmg=get_pmg_structure(scell)
                        if output_POSCARs:
                                scell_pmg.to('POSCAR-'+"{0:0=3d}".format(i_scell+1)); print("{0:0=3d}".format(i_scell+1))#f"i_scell+1:03d"
                        scell_predictions=chgnet.predict_structure(scell_pmg)
                        forces = np.array(scell_predictions['f'])
                        disp = disp + 1
                        drift_force = forces.sum(axis=0)
                        for force in forces:
                                force -= drift_force / forces.shape[0]
                        set_of_forces.append(forces)

                phonon.produce_force_constants(forces=set_of_forces)

                if write_fc:
                        write_FORCE_CONSTANTS(
                                                phonon.get_force_constants(), filename="FORCE_CONSTANTS"
                                                )

                # save the phonon attribute to the object
                self.phonon=phonon


        
        def get_phonon_dos_bs(
                                self,
                                line_density=30,
                                units='THz',
                                stability_threshold=-0.1,
                                output_ph_band: bool=True,
                                phonopy_bands_dos_figname='phonopy_bands_dos.png',
                                dpi=100,
                                ):

                # freq_conversion_factor=1 # THz units
                # freq_conversion_factor=333.566830  # ThztoCm-1
                if units=='cm-1':
                        freq_conversion_factor=333.566830
                else:
                        freq_conversion_factor=1

                kpoints=self.get_jarvis_kpoints(line_density=line_density)
                lbls = kpoints.labels
                lbls_ticks = []
                freqs = []
                tmp_kp = []
                lbls_x = []
                count = 0
                stability=True
                for ii, k in enumerate(kpoints.kpts):
                        k_str = ",".join(map(str, k))
                        if ii == 0:
                                tmp = []
                                for i, freq in enumerate(self.phonon.get_frequencies(k)):
                                        tmp.append(freq)
                                        #print(freq)
                                        #for fs in freq:
                                        if freq < stability_threshold*freq_conversion_factor:
                                                stability=False

                                freqs.append(tmp)
                                tmp_kp.append(k_str)
                                lbl = "$" + str(lbls[ii]) + "$"
                                lbls_ticks.append(lbl)
                                lbls_x.append(count)
                                count += 1
                                # lbls_x.append(ii)

                        elif k_str != tmp_kp[-1]:
                                tmp_kp.append(k_str)
                                tmp = []
                                for i, freq in enumerate(self.phonon.get_frequencies(k)):
                                        tmp.append(freq)

                                        #for fs in freq:
                                        if freq < stability_threshold*freq_conversion_factor:
                                                stability=False

                                freqs.append(tmp)
                                lbl = lbls[ii]
                                if lbl != "":
                                        lbl = "$" + str(lbl) + "$"
                                        lbls_ticks.append(lbl)
                                        # lbls_x.append(ii)
                                        lbls_x.append(count)
                                count += 1
                                # lbls_x = np.arange(len(lbls_ticks))


                with open(os.path.join(self.path, 'stability'), 'w') as stable_file:
                        if stability==True:
                                stable_file.write('stable'); #print('stable')
                        elif stability==False:
                                stable_file.write('unstable'); #print('unstable')
#               """
#               exit()
#               print(output_ph_band)

                if output_ph_band:
                        freqs = np.array(freqs)
                        freqs = freqs * freq_conversion_factor
                        # print('freqs',freqs,freqs.shape)
                        the_grid = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
                        plt.rcParams.update({"font.size": 18})

                        plt.figure(figsize=(10, 5))
                        plt.subplot(the_grid[0])
                        for i in range(freqs.shape[1]):
                                plt.plot(freqs[:, i], lw=2, c="b")
                        for i in lbls_x:
                                plt.axvline(x=i, c="black")
                        plt.xticks(lbls_x, lbls_ticks)
                        # print('lbls_x',lbls_x,len(lbls_x))
                        # print('lbls_ticks',lbls_ticks,len(lbls_ticks))

                        if units=='cm-1':
                                plt.ylabel("Frequency (cm$^{-1}$)")
                        else:
                                plt.ylabel("Frequency (THz)")

                        plt.xlim([0, max(lbls_x)])

                        self.phonon.run_mesh([10, 10, 10], is_gamma_center=True, is_mesh_symmetry=False)
                        self.phonon.run_total_dos()
                        tdos = self.phonon._total_dos

                        # print('tods',tdos._frequencies.shape)
                        freqs, ds = tdos.get_dos()
                        freqs = np.array(freqs)
                        freqs = freqs * freq_conversion_factor
                        min_freq = -0.05 * freq_conversion_factor
                        max_freq = max(freqs)
                        plt.ylim([min_freq, max_freq])

                        plt.subplot(the_grid[1])
                        plt.fill_between(
                                        ds, freqs, color=(0.2, 0.4, 0.6, 0.6), edgecolor="k", lw=1, y2=0
                                        )
                        plt.xlabel("DOS")
                        # plt.plot(ds,freqs)
                        plt.yticks([])
                        plt.xticks([])
                        plt.ylim([min_freq, max_freq])
                        plt.xlim([0, max(ds)])
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.path, phonopy_bands_dos_figname), dpi=dpi)
                        plt.close()


       
        def mesh_bands(
                self,
                filename="orig_band.conf",
                line_density=30,
                BAND_POINTS=100,
                ):
                kpoints = Kpoints3D().kpath(self.jarvis_atoms, line_density=line_density)
                all_kp = kpoints._kpoints
                labels = kpoints._labels
                all_labels = ""
                all_lines = ""
                for lb in labels:
                        if lb == "":
                                lb = None
                        all_labels = all_labels + str(lb) + str(" ")
                for k in all_kp:
                        all_lines = (
                                all_lines
                                + str(k[0])
                                + str(" ")
                                + str(k[1])
                                + str(" ")
                                + str(k[2])
                                + str(" ")
                                )
                file = open(os.path.join(self.path, filename), "w")
                file.write('PRIMITIVE_AXES = AUTO\n')
                line = str("ATOM_NAME = ") + str(" ".join(list(set(self.jarvis_atoms.elements)))) + "\n"
                file.write(line)
                line = str("DIM = ") + " ".join(map(str, self.supercell_dims)) + "\n"
                file.write(line)
                line = str("FORCE_CONSTANTS = READ") + "\n"
                file.write(line)
                line = str("BAND= ") + str(all_lines) + "\n"
                file.write(line)
                #line = str("BAND_LABELS= ") + str(all_labels) + "\n"
                #file.write(line)
                file.close()


                ase_atoms=AseAtomsAdaptor.get_atoms(self.structure)
                bandpath_kpts = ase_atoms.cell.bandpath()._kpts
                kpts_str=str()
                for i in bandpath_kpts:
                        kpts_str+=str(i[0])+' '+str(i[1])+' '+ str(i[2])+'  '

                file = open(os.path.join(self.path, 'band.conf'), "w")
                file.write('PRIMITIVE_AXES = AUTO\n')
                line = str("ATOM_NAME = ") + str(" ".join(list(set(self.jarvis_atoms.elements)))) + "\n"
                file.write(line)
                line = str("DIM = ") + " ".join(map(str, self.supercell_dims)) + "\n"
                file.write(line)
                line = str("FORCE_CONSTANTS = READ") + "\n"
                file.write(line)
                line = str("BAND= ") + kpts_str + "\n"
                file.write(line)
                file.write('BAND_POINTS = '+str(BAND_POINTS)+'\n')
                #line = str("BAND_LABELS= ") + str(all_labels) + "\n"
                #file.write(line)
                file.close()



if __name__=="__main__":
        pmg_struc=Structure.from_file('POSCAR')
        chg_phon=chgnet_phonopy(pmg_struc)
        chg_phon.save_to_pickle()

        #with open('chgnet_data.pkl', 'rb') as inp:
        #       chgnet_obj= pickle.load(inp)

        chg_phon.get_phonon_fc2()
        chg_phon.get_phonon_dos_bs()
