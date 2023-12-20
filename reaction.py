import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os
from rdkit.Chem import rdChemReactions
import pandas as pd
import yaml
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class ReactionAB:
    smarts = "[#6:1]-[$([B](-O)(-O)),$([B](-F)(-F)(-F))].[#6,#7:2]-[I,Br]>>[*:1]-[*:2]"
    rxn = rdChemReactions.ReactionFromSmarts(smarts)
    conf_folder = "ExpToSimulate"

    def __init__(self,
                 file_name_a: str = "Mol_Group_A_v5.xlsx",
                 file_name_b: str = "Mol_Group_B_v4.xlsx",
                 data_dir_path: str = "database",
                 id_column: str = "ID",
                 smile_column: str = "Smiles"):
        """Initialize class that makes reaction.
        

        Args:
            file_name_a (str): File path to database for reactants A.
            file_name_b (str): File path to database for reactants B.
            data_dir_path (str): (Relative) path to database directory.
        """
        self.data_path = os.path.realpath(data_dir_path)
        self.file_path_a = os.path.join(data_dir_path, file_name_a)
        self.file_path_b = os.path.join(data_dir_path, file_name_b)
        self.info("Reading excel files.")
        self.data_a = pd.read_excel(self.file_path_a, sheet_name=0, header=0)
        self.data_b = pd.read_excel(self.file_path_b, sheet_name=0, header=0)
        self.info("Reading structures.")
        mol_a = self.load_json_file(
            os.path.splitext(self.file_path_a)[0] + ".json")
        self.mol_a = {key: rdkit.Chem.MolFromMolBlock(value, removeHs=False) for key, value in mol_a.items()}
        mol_b = self.load_json_file(
            os.path.splitext(self.file_path_b)[0] + ".json")
        self.mol_b = {key: rdkit.Chem.MolFromMolBlock(value, removeHs=False) for key, value in mol_b.items()}
        self._id_column = id_column
        self._smile_column = smile_column

    def info(self, *args, **kwargs):
        print("INFO:", *args, **kwargs)

    def error(self, *args, **kwargs):
        print("INFO:", *args, **kwargs)

    def _log(self, *args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def load_json_file(fname):
        with open(fname, 'r') as json_file:
            file_read = json.load(json_file)
        return file_read

    @staticmethod
    def count_halogen(m, rgs: list = ['I', 'Br']):
        counts = 0
        for a in m.GetAtoms():
            if str(a.GetSymbol()) in rgs:
                counts = counts + 1
        return counts

    @staticmethod
    def count_B_reacts(m, rgs: list = ["OBO", "FB(F)F"]):
        groups = [rdkit.Chem.MolFromSmiles(x) for x in rgs]
        counts = 0
        for x in groups:
            counts = counts + len(m.GetSubstructMatches(x))
        return counts

    @staticmethod
    def count_A_reacts(m, rgs: list = ["CBr", "NBr", "CI", "NI"]):
        groups = [rdkit.Chem.MolFromSmiles(x) for x in rgs]
        counts = 0
        for x in groups:
            counts = counts + len(m.GetSubstructMatches(x))
        return counts

    def check_reactive_groups(self):
        # Check individual groups
        ra_groups = []
        rb_groups = []
        rh_groups = []
        for key, value in self.mol_a.items():
            ra_groups.append(self.count_A_reacts(value))
            rb_groups.append(self.count_B_reacts(value))
            rh_groups.append(self.count_halogen(value))
        self.info("Counts for X-Br in A:",
                  {x: y for x, y in zip(*np.unique(ra_groups, return_counts=True))})
        self.info("Counts for X-B(O)O in A:",
                  {x: y for x, y in zip(*np.unique(rb_groups, return_counts=True))})
        self.info("Compare additional Br's, miss:",
                  (np.array(ra_groups) - np.array(rh_groups))[(np.array(ra_groups) != np.array(rh_groups))])

        ra_groups = []
        rb_groups = []
        for key, value in self.mol_b.items():
            ra_groups.append(self.count_A_reacts(value))
            rb_groups.append(self.count_B_reacts(value))
        self.info("Counts for X-Br in B",
                  {x: y for x, y in zip(*np.unique(ra_groups, return_counts=True))})
        self.info("Counts for X-B(O)O in B",
                  {x: y for x, y in zip(*np.unique(rb_groups, return_counts=True))})

    @staticmethod
    def remove_free_water(m, info: str = None):
        water = rdkit.Chem.MolFromSmiles("O")
        water = rdkit.Chem.AddHs(water)  # important!
        has_removed = False
        for _ in m.GetAtoms():  # Remove multiple times but atmost #Atoms
            if m.HasSubstructMatch(water):
                m = rdkit.Chem.DeleteSubstructs(m, water)
                has_removed = True
            else:
                break
        if has_removed and info is not None:
            print("Removed free water from:", info)
        return m

    @staticmethod
    def remove_free_acid(m, info=None):
        hcl = rdkit.Chem.MolFromSmiles("Cl")
        hcl = rdkit.Chem.AddHs(hcl)  # important!
        hbr = rdkit.Chem.MolFromSmiles("Br")
        hbr = rdkit.Chem.AddHs(hbr)  # important!
        has_removed = False
        for _ in m.GetAtoms():  # Remove multiple times but atmost #Atoms
            if m.HasSubstructMatch(hcl):
                m = rdkit.Chem.DeleteSubstructs(m, hcl)
                has_removed = True
            elif m.HasSubstructMatch(hbr):
                m = rdkit.Chem.DeleteSubstructs(m, hbr)
                has_removed = True
            else:
                break
        if has_removed and info is not None:
            print("Removed free HCl, HBr from:", info)
        return m

    def run_reaction(self, ida: str, idb: str,
                     add_hydrogen: bool = True,
                     sanitize: bool = True,
                     make_conformers: bool = False,
                     optimize_conformer: bool = False,
                     useRandomCoords: bool = True,
                     maxAttempts: int = 100,
                     randomSeed: int = -1):
        rxn = self.rxn
        is_valid = True

        m1 = self.mol_a[ida]
        m2 = self.mol_b[idb]

        c1a = self.count_A_reacts(m1)
        c1b = self.count_B_reacts(m1)
        c2a = self.count_A_reacts(m2)
        c2b = self.count_B_reacts(m2)

        if c1a == 0 or c2b == 0:
            self.error("Error: missing reactive group for", ida, idb)
            is_valid = False
        if c1b > 0 or c2a > 0:
            self.error(
                "Error: X-Br group in B or Y-B(O)O in group A for", ida, idb)
            is_valid = False

        if not is_valid:
            return

        run_racts = 0
        prod = m1
        for _ in range(c1a):
            reacts = (m2, prod)
            products = rxn.RunReactants(reacts)
            if len(products) > 0:
                prod = products[0][0]
                run_racts += 1
            rdkit.Chem.SanitizeMol(prod)
        if run_racts != c1a:
            self.error("Error: expected additional reaction for", ida, idb)

        # Finished product
        m = prod
        if sanitize:
            rdkit.Chem.SanitizeMol(m)
        if add_hydrogen:
            m = rdkit.Chem.AddHs(m)
        if make_conformers:
            rdkit.Chem.RemoveStereochemistry(m)
            rdkit.Chem.AssignStereochemistry(m)
            rdkit.Chem.AllChem.EmbedMolecule(
                m, randomSeed=randomSeed, maxAttempts=maxAttempts,
                useRandomCoords=useRandomCoords)
        if optimize_conformer and make_conformers:
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(m)
            rdkit.Chem.AssignAtomChiralTagsFromStructure(m)
            rdkit.Chem.AssignStereochemistryFrom3D(m)
            rdkit.Chem.AssignStereochemistry(m)
        m.SetProp("_Name", ida + idb)

        return m

    def __getitem__(self, key: tuple):
        assert len(key) == 2, "Provide ['A...', 'B...'] keys"
        return self.run_reaction(key[0], key[1])

    def show_reaction(self, ida, idb):
        m1 = self.mol_a[ida]
        m2 = self.mol_b[idb]
        prod = self.run_reaction(ida, idb)
        sm1 = rdkit.Chem.MolToSmarts(rdkit.Chem.RemoveHs(m1), isomericSmiles=True)
        sm2 = rdkit.Chem.MolToSmarts(rdkit.Chem.RemoveHs(m2), isomericSmiles=True)
        res = rdkit.Chem.MolToSmarts(rdkit.Chem.RemoveHs(prod), isomericSmiles=True)
        cr = rdkit.Chem.AllChem.ReactionFromSmarts(sm1 + "." + sm2 + ">>" + res)
        img = rdkit.Chem.Draw.ReactionToImage(cr)
        return img

    def all_combos_possible(self):
        combos = []
        for x in self.mol_a.keys():
            for y in self.mol_b.keys():
                combos.append((x, y))
        return combos

    def run_combos(self, combos, **kwargs):
        mol_combos = []  # List here is better
        for i in tqdm(range(len(combos))):
            mol_combos.append(self.run_reaction(combos[i][0], combos[i][1],
                                                **kwargs))
        return mol_combos

    def run_all_combos(self, num_workers=1, batch_size=1000, **kwargs):
        combos = self.all_combos_possible()
        mol_combos = []  # List here is better

        def wrapp_run_kwargs(kwargs):
            return self.run_reaction(**kwargs)

        if num_workers == 1:
            for i in tqdm(range(len(combos))):
                mol_combos.append(self.run_reaction(combos[i][0], combos[i][1], **kwargs))
        else:
            self.info("Start parallel reactions:", flush=True)
            arg_list = [{"ida": x, "idb": y} for x, y in combos]
            for x in arg_list:
                x.update(kwargs)
            for i in range(0, len(arg_list), batch_size):
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    result = executor.map(wrapp_run_kwargs, arg_list[i:i + batch_size])
                mol_combos.append(list(result))
                self.info("Finished {} of {}".format(i + batch_size, len(arg_list)))
        return mol_combos

    @staticmethod
    def MolsToGridImageZoomed(mols, molsPerRow=3, subImgSize=(100, 100), legends=None,
                              grid=True,
                              **kwargs):
        nRows = len(mols) // molsPerRow
        if len(mols) % molsPerRow: nRows += 1
        fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
        full_image = Image.new('RGBA', fullSize)
        dy = subImgSize[1]
        dx = subImgSize[0]
        for ii, mol in enumerate(mols):
            if mol.GetNumConformers() == 0:
                rdkit.Chem.AllChem.Compute2DCoords(mol)
            column = ii % molsPerRow
            row = ii // molsPerRow
            offset = (column * subImgSize[0], row * subImgSize[1])
            d2d = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1])
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            sub = Image.open(BytesIO(d2d.GetDrawingText()))
            full_image.paste(sub, box=offset)

        fnt = ImageFont.truetype("arial", size=12)
        txt = Image.new("RGBA", full_image.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        if legends is not None:
            for ii, mol in enumerate(mols):
                column = ii % molsPerRow
                row = ii // molsPerRow
                offset = ((column + 0.5) * dx, (row + 1) * dy)
                d.text(offset, legends[ii], font=fnt, anchor="md", fill=(0, 0, 0, 255))

        # Grid
        for ii, mol in enumerate(mols):
            column = ii % molsPerRow
            row = ii // molsPerRow
            gp = np.array([column * dx, row * dy], dtype="int")
            if grid:
                d.line(([gp[0], gp[1],
                         gp[0] + dx, gp[1]]),
                       fill=(0, 0, 0, 255), width=1)
                d.line(([gp[0], min(gp[1] + dy, full_image.size[1] - 1),
                         gp[0] + dx, min(gp[1] + dy, full_image.size[1] - 1)]),
                       fill=(0, 0, 0, 255), width=1)
                d.line(([gp[0], gp[1],
                         gp[0], gp[1] + dy]), fill=(0, 0, 0, 255), width=1)
                d.line(([min(gp[0] + dx, full_image.size[0] - 1), gp[1],
                         min(gp[0] + dx, full_image.size[0] - 1), gp[1] + dy]), fill=(0, 0, 0, 255), width=1)

        out = Image.alpha_composite(full_image, txt)
        background = Image.new("RGB", out.size, (255, 255, 255))
        background.paste(out, mask=out.split()[3])  # 3 is the alpha channel
        return background

    def draw_to_pdf_products(self, reacts: list,
                             filepath: str = "ReactionProductList.pdf",
                             mols_per_page: int = 35,
                             mols_per_row: int = 5):
        mols = self.run_combos(reacts, make_conformers=False)
        mol_copy = [
            rdkit.Chem.MolFromMolBlock(rdkit.Chem.MolToMolBlock(x)) for x in mols]
        ids = [x[0] + x[1] for x in reacts]

        for m in mol_copy:
            rdkit.Chem.RemoveHs(m)
            rdkit.Chem.SanitizeMol(m)
            rdkit.Chem.AllChem.Compute2DCoords(m)

        grid_function = self.MolsToGridImageZoomed  # or rdkit.Chem.Draw.MolsToGridImage
        image_m = [grid_function(
            mol_copy[i:i + mols_per_page],
            molsPerRow=mols_per_row,
            subImgSize=(500, 500),
            legends=ids[i:i + mols_per_page]) for i in range(0, len(mols), mols_per_page)]

        image_m[0].save(filepath,
                        "PDF", dpi=(300, 300), save_all=True,
                        append_images=image_m[1:])

    def save_to_mol_folder_for_simulation(self, reacts: list, **kwargs):
        mols = self.run_combos(reacts, **kwargs)
        os.makedirs(self.conf_folder, exist_ok=True)
        react_dict = {}
        for x, m in zip(reacts, mols):
            folder = os.path.join(self.conf_folder, x[0] + x[1])
            os.makedirs(folder, exist_ok=True)
            rdkit.Chem.MolToMolFile(m, os.path.join(folder, "conf_guess.mol"))
            rdkit.Chem.MolToXYZFile(m, os.path.join(folder, "conf_guess.xyz"))

            react_dict[x[0] + x[1]] = rdkit.Chem.MolToSmiles(m)

        with open(os.path.join(
                self.conf_folder, "exp_to_do_stock.yaml"), 'w') as yaml_file:
            yaml.dump(react_dict, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    react = ReactionAB()
    react.check_reactive_groups()
#    print(react["A1", "B1"])
#    react.show_reaction("A1", "B1").show()
#    print("Number of possible reactions:", len(react.all_combos_possible()))
#    print("Run over a list of pairs:", flush=True)
#    react.run_combos([("A1", "B1"), ("A10", "B10"), ("A100", "B100"), ("A110", "B500")])
    # react.draw_to_pdf_products([("A1", "B1"), ("A10", "B10"), ("A100", "B100"), ("A110", "B500")])
    # Not practical, takes long.
    # all = react.run_all_combos(num_workers=12, make_conformers=True, optimize_conformer=True)
    
    
    
    
    
    
