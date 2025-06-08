<h1 align="center">CHGNet for Lattice Thermal Conductivity Prediction</h1> <h4 align="center">Universal CHGNet for Thermal Transport: Decoding Bonding Contributions to Low Lattice Thermal Conductivity </h4>
📦 Installation
Install the package from PyPI:

bash
Copy
Edit
pip install chgnet_thermal_conductivity
If PyPI installation fails or you wish to use the latest version from the main branch:

bash
Copy
Edit
pip install git+https://github.com/Harpriyaminhas/chgnet_thermal_transport.git
🔧 Fine-Tuning CHGNet
To adapt CHGNet for accurate thermal conductivity predictions:

Run Langevin Molecular Dynamics (MD) simulations at various temperatures to generate force-displacement datasets.

Train the model using train.py. This will fine-tune CHGNet and save the best model as best_model.pth.

Predict phonon properties using chgnet_phonopy.py, which:

Generates displaced structures

Calculates force constants

Computes phonon band structure and DOS using the fine-tuned model

📁 Thermal Conductivity Calculation Scripts
This package includes utilities for computing lattice thermal conductivity:

✅ Phonopy version 2.7.1 required

Replace the default group_velocity.py in Phonopy with the modified version provided to ensure compatibility.

✅ Script included: getkappa.py

Computes the minimum lattice thermal conductivity using phonon properties.

Example input data is provided in the example folder for guidance.


@misc{Harpriya_CHGNet_Thermal,
  author = {Minhas, Harpriya}, {Sharma, Rahul Kumar}, {Pathak, Biswarup},
  title = {Universal CHGNet for Thermal Transport: Decoding Bonding Contributions to Low Lattice Thermal Conductivity},
  year = {2025},
  url = {https://github.com/Harpriyaminhas/chgnet_thermal_transport}
}

@reference_article{deng2023chgnet,
  title = {CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
  author = {Deng, Bowen and Zhong, Pan and Jun, Kyungwha and Riebesell, Jan and Han, Kejun and Bartel, Christopher J and Ceder, Gerbrand},
  journal = {Nature Machine Intelligence},
  volume = {5},
  number = {9},
  pages = {1031--1041},
  year = {2023},
  publisher = {Nature Publishing Group}
}

