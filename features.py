import numpy as np
import rdkit
import rdkit.Chem
import rdkit.Chem.Descriptors
import rdkit.Chem.Fragments
from tqdm import tqdm


class MolFeatures:
    mol_rep = {
        # Count atoms
        "C": lambda m: sum([x.GetSymbol() == "C" for x in m.GetAtoms()]),
        "N": lambda m: sum([x.GetSymbol() == "N" for x in m.GetAtoms()]),
        "O": lambda m: sum([x.GetSymbol() == "O" for x in m.GetAtoms()]),
        "H": lambda m: sum([x.GetSymbol() == "H" for x in m.GetAtoms()]),
        "S": lambda m: sum([x.GetSymbol() == "S" for x in m.GetAtoms()]),
        "F": lambda m: sum([x.GetSymbol() == "F" for x in m.GetAtoms()]),
        "Cl": lambda m: sum([x.GetSymbol() == "Cl" for x in m.GetAtoms()]),
        # Count bonds
        "NumAtoms": lambda m: sum([True for _ in m.GetAtoms()]),
        "AtomIsInRing": lambda m: sum([x.IsInRing() for x in m.GetAtoms()]),
        "AtomIsAromatic": lambda m: sum([x.GetIsAromatic() for x in m.GetAtoms()]),
        "NumBonds": lambda m: sum([True for _ in m.GetBonds()]),
        "BondIsConjugated": lambda m: sum([x.GetIsConjugated() for x in m.GetBonds()]),
        "BondIsAromatic": lambda m: sum([x.GetIsAromatic() for x in m.GetBonds()]),
        "NumRotatableBonds": lambda m: rdkit.Chem.Lipinski.NumRotatableBonds(m),
        # Fractions
        "fr_Al_COO": lambda m: rdkit.Chem.Fragments.fr_Al_COO(m),
        "fr_Ar_COO": lambda m: rdkit.Chem.Fragments.fr_Ar_COO(m),
        "fr_Al_OH": lambda m: rdkit.Chem.Fragments.fr_Al_OH(m),
        "fr_Ar_OH": lambda m: rdkit.Chem.Fragments.fr_Ar_OH(m),
        "fr_C_O_noCOO": lambda m: rdkit.Chem.Fragments.fr_C_O_noCOO(m),
        "fr_NH2": lambda m: rdkit.Chem.Fragments.fr_NH2(m),
        "fr_SH": lambda m: rdkit.Chem.Fragments.fr_SH(m),
        "fr_sulfide": lambda m: rdkit.Chem.Fragments.fr_sulfide(m),
        "fr_alkyl_halide": lambda m: rdkit.Chem.Fragments.fr_alkyl_halide(m),
        # Descriptors
        "ExactMolWt": lambda m: rdkit.Chem.Descriptors.ExactMolWt(m),
        "FpDensityMorgan3": lambda m: rdkit.Chem.Descriptors.FpDensityMorgan3(m),
        "FractionCSP3": lambda m: rdkit.Chem.Lipinski.FractionCSP3(m),
        "MolLogP": lambda m: rdkit.Chem.Crippen.MolLogP(m),
        "MolMR": lambda m: rdkit.Chem.Crippen.MolMR(m),
        # Custom structures
        "has_CN(C)C": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("CN(C)C"))),
        "has_CNC": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("CNC"))),  # or HasSubstructMatch()
        "has_C=NC": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("C=NC"))),
        "has_Thiophene": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("c1cScc1"))),
        "has_Pyrrole": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("c1cNcc1"))),
        "has_Benzimidazole": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("Cn1cnc2ccccc21"))),
        "has_Benzothiophene": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("c1ccc2sccc2c1"))),
        "has_Naphthalene": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("c1ccc2ccccc2c1"))),
        "has_Biphenyl": lambda m: len(m.GetSubstructMatches(rdkit.Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")))
    }

    def __init__(self, descriptor_list: list = None):
        default_list = [
            "C", "N", "O", "H", "S", "F", "Cl",
            "NumAtoms", "AtomIsInRing", "AtomIsAromatic",
            "NumBonds", "BondIsConjugated", "BondIsAromatic", "NumRotatableBonds",
            "fr_C_O_noCOO",
            # "fr_Al_COO", "fr_Ar_COO", "fr_Al_OH", "fr_Ar_OH",  "fr_NH2",
            # "fr_SH", "fr_sulfide", "fr_alkyl_halide"
            "ExactMolWt", "FpDensityMorgan3", "FractionCSP3",
            "MolLogP", "MolMR",
            "has_CN(C)C", "has_CNC", "has_C=NC", "has_Thiophene", "has_Pyrrole", "has_Benzimidazole",
            "has_Benzothiophene", "has_Naphthalene", "has_Biphenyl"
        ]
        self.descriptor_list = default_list if descriptor_list is None else descriptor_list

    def __call__(self, mol_list: list):
        self.info("Making features:")
        feat_list = []
        for i in tqdm(range(len(mol_list))):
            feat_list.append(self.map_descriptor(mol_list[i]))
        return np.array(feat_list)

    def info(self, *args, **kwargs):
        print("INFO:", *args, **kwargs)

    def map_descriptor(self, m):
        rep = [self.mol_rep[n](m) for n in self.descriptor_list]
        return rep


if __name__ == "__main__":
    mk = MolFeatures()
    print(mk([rdkit.Chem.MolFromSmiles("CC"), rdkit.Chem.MolFromSmiles("CCOS")]))
    print(mk.descriptor_list)