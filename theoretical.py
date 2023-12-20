import pandas as pd
import numpy as np
import os


class TheoSimulation:
    def __init__(self, 
                 file_name: str = "Theo_simu.xlsx",
                 data_dir_path: str = "data",
                 descriptor_list: list = None):
        self.file_path = os.path.join(data_dir_path, file_name)
        self.data_path = data_dir_path
        self.theo = None
        self.load_data()
        self.descriptor_list = descriptor_list
        if descriptor_list is None:
            self.descriptor_list = ["dipole", "homo", "lumo", "gap", "energy", "a", "b", "c"]
        
    def load_data(self):
        self.theo = pd.read_excel(self.file_path, sheet_name=0, header=0)
        self.theo.set_index("ID", inplace=True)

    def labels_for_combos(self, combos: list):
        ab = [x[0]+x[1] for x in combos]
        values = self.theo.loc[ab]
        values = values[self.descriptor_list]
        return ab, np.array(values)
            
    
        

if __name__ == "__main__":
    data = TheoSimulation()
    print(data.labels_for_combos([("A9","B702"), ("A99","B172"), ("A1066","B2")]))
