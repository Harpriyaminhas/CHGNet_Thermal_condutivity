import os
import argparse
import sys
from pymatgen.core import Structure
import pickle
from chgnet.utils import parse_vasp_dir, read_json, write_json
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.graph import CrystalGraphConverter
import torch
import numpy as np
import csv
import json

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def none_or_val(value):
    """Helper function to convert a string to None or an integer."""
    if value == 'None':
        return None
    return int(value)

def bool_vals(value):
    """Helper function to convert a string to a boolean."""
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

# Output directories
output_dirs = {
    "cif": "./outputs/cif",
    "graphs": "./outputs/graphs",
    "json": "./outputs/json",
    "pickle": "./outputs/pickle"
}
for dir_path in output_dirs.values():
    create_directory(dir_path)

# Step 1: Parse VASP directories from T100 to T700
root_dir = "/home/biswarup/Harpriya/screen/md/mp-1192303_Pr2Mn3SbS34/"
temperature_dirs = [f"T{i}" for i in range(100, 701, 100)]  # T100, T200, ..., T700

merged_dataset = {"structure": [], "energy_per_atom": [], "force": [], "stress": [], "magmom": []}

# Iterate over temperature directories
for temp_dir in temperature_dirs:
    temp_path = os.path.join(root_dir, temp_dir)
    if os.path.exists(temp_path):
        dataset_dict = parse_vasp_dir(temp_path, save_path=f"{output_dirs['json']}/chgnet_dataset_{temp_dir}.json")
        dataset_dict = read_json(f"{output_dirs['json']}/chgnet_dataset_{temp_dir}.json")

        # Merge datasets
        merged_dataset["structure"].extend(dataset_dict["structure"])
        merged_dataset["energy_per_atom"].extend(dataset_dict["energy_per_atom"])
        merged_dataset["force"].extend(dataset_dict["force"])
        merged_dataset["stress"].extend(dataset_dict.get("stress", []))
        merged_dataset["magmom"].extend(dataset_dict.get("magmom", []))

# Save merged dataset to JSON
write_json(merged_dataset, f"{output_dirs['json']}/merged_chgnet_dataset.json")

# Convert dataset to CHGNet StructureData format
structures = [Structure.from_dict(struct) for struct in merged_dataset["structure"]]
energies = merged_dataset["energy_per_atom"]
forces = merged_dataset["force"]
stresses = merged_dataset.get("stress") or None
magmoms = merged_dataset.get("magmom") or None  # Handle missing magnetization data

# Convert dataset to CHGNet StructureData format
dataset = StructureData(
    structures=structures,
    energies=energies,
    forces=forces,
    stresses=stresses,
    magmoms=magmoms,  # This will be None if magnetization data is absent
)

# Save structures in various formats
# JSON
dict_to_json = [struct.as_dict() for struct in structures]
write_json(dict_to_json, f"{output_dirs['json']}/CHGNet_structures.json")

# Pickle
with open(f"{output_dirs['pickle']}/CHGNet_structures.p", "wb") as f:
    pickle.dump(merged_dataset, f)

# CIF
for idx, struct in enumerate(structures):
    struct.to(filename=f"{output_dirs['cif']}/{idx}.cif")

# CHGNet graph
converter = CrystalGraphConverter()
for idx, struct in enumerate(structures):
    graph = converter(struct)
    graph.save(fname=f"{output_dirs['graphs']}/{idx}.pt")

# ------------------ Training Setup -------------------

# Create data loaders for training, validation, and testing
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset,
    batch_size=8,
    train_ratio=0.8,  # 80% for training
    val_ratio=0.1,    # 10% for validation
)

# Initialize the CHGNet model
chgnet = CHGNet.load()

# Freeze selected layers of the model if needed (optional)
def freeze_model_layers(chgnet):
    """
    Freeze selected layers of the CHGNet model so that they are not updated during training.
    """
    # Freeze embedding and basis expansion layers.
    for layer in [
        chgnet.atom_embedding,
        chgnet.bond_embedding,
        chgnet.angle_embedding,
        chgnet.bond_basis_expansion,
        chgnet.angle_basis_expansion,
    ]:
        for param in layer.parameters():
            param.requires_grad = False

    # Freeze all but the last atom convolution layers (assuming atom_conv_layers is a list)
    for module in chgnet.atom_conv_layers[:-1]:
        for param in module.parameters():
            param.requires_grad = False

    # Freeze bond convolution layers.
    for module in chgnet.bond_conv_layers:
        for param in module.parameters():
            param.requires_grad = False

    # Freeze angle layers.
    for module in chgnet.angle_layers:
        for param in module.parameters():
            param.requires_grad = False

freeze_model_layers(chgnet)

# Initialize the Trainer
trainer = Trainer(
    model=chgnet,
    targets="efs",  # Targets: energy ("e"), force ("f"), stress ("s")
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=50,
    learning_rate=1e-2,
    use_device="cpu",
    print_freq=6,
)

# Manually loop over epochs to log metrics and track the best model
epoch_metrics_history = []
num_epochs = trainer.epochs
best_val_error = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch} =====")
    # Run one training epoch
    train_metrics = trainer._train(train_loader, current_epoch=epoch)
    # Run validation epoch
    val_metrics = trainer._validate(val_loader)
    # Run test epoch
    test_metrics = trainer._validate(test_loader, is_test=True)

    # Prepare a dictionary to store metrics
    epoch_metrics = {"epoch": epoch}
    for target in trainer.targets:
        epoch_metrics[f"train_{target}_mae"] = train_metrics.get(target, np.nan)
        epoch_metrics[f"val_{target}_mae"] = val_metrics.get(target, np.nan)
        epoch_metrics[f"test_{target}_mae"] = test_metrics.get(target, np.nan)
    epoch_metrics_history.append(epoch_metrics)

    # Track best model using energy MAE ("e")
    current_val_error = val_metrics.get("e", float('inf'))
    if current_val_error < best_val_error:
        best_val_error = current_val_error
        best_model_state = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        print(f"Epoch {epoch}: New best validation energy MAE: {best_val_error:.6f}")

# Save epoch metrics to a CSV file
csv_filename = "epoch_metrics.csv"
with open(csv_filename, "w", newline="") as csvfile:
    fieldnames = list(epoch_metrics_history[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(epoch_metrics_history)
print(f"\nEpoch metrics logged to {csv_filename}")
print(json.dumps(epoch_metrics_history, indent=2))

# Manually save the best model if tracked, otherwise save the final model
if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth.tar")
    print("Best model saved to best_model.pth.tar")
else:
    print("No best model was tracked; saving the final model instead.")
    torch.save(trainer.model.state_dict(), f"final_model_epoch{num_epochs-1}.pth.tar")
    print(f"Final model saved to final_model_epoch{num_epochs-1}.pth.tar")

# ------------------ Saving DFT vs CHGNet Data -------------------

def save_dft_vs_chgnet_data(dft_energies, chgnet_energies, dft_forces, chgnet_forces, dft_stresses, chgnet_stresses, output_dir="outputs"):
    """
    Save DFT vs CHGNet energy, force, and stress data to text files.

    Parameters:
    - dft_energies (list or np.array): DFT energies for structures
    - chgnet_energies (list or np.array): CHGNet predicted energies for structures
    - dft_forces (list or np.array): DFT forces for structures
    - chgnet_forces (list or np.array): CHGNet predicted forces for structures
    - dft_stresses (list or np.array): DFT stresses for structures
    - chgnet_stresses (list or np.array): CHGNet predicted stresses for structures
    - output_dir (str): Directory to save text files
    """
    create_directory(output_dir)

    with open(os.path.join(output_dir, "dft_vs_chgnet_energy.txt"), "w") as f:
        f.write("DFT Energy (eV), CHGNet Energy (eV)\n")
        for dft, chgnet in zip(dft_energies, chgnet_energies):
            f.write(f"{dft},{chgnet}\n")

    with open(os.path.join(output_dir, "dft_vs_chgnet_forces.txt"), "w") as f:
        f.write("DFT Forces, CHGNet Forces\n")
        for dft, chgnet in zip(dft_forces, chgnet_forces):
            f.write(f"{dft},{chgnet}\n")

    with open(os.path.join(output_dir, "dft_vs_chgnet_stresses.txt"), "w") as f:
        f.write("DFT Stresses, CHGNet Stresses\n")
        for dft, chgnet in zip(dft_stresses, chgnet_stresses):
            f.write(f"{dft},{chgnet}\n")

# Saving the results
save_dft_vs_chgnet_data(
    merged_dataset["energy_per_atom"],
    merged_dataset["energy_per_atom"],
    merged_dataset["force"],
    merged_dataset["force"],
    merged_dataset["stress"],
    merged_dataset["stress"]
)
