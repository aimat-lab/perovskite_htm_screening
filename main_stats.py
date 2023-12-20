import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel("data/MolFeatures.xlsx")

alpha=0.75
# Plot models
fig, axsg = plt.subplots(1, 4, figsize=(17.25, 3.75))
axs = axsg.flatten()

axs[0].hist(data["C"]/data["NumAtoms"], bins=20, label="C", color="dimgray", alpha=alpha)
axs[0].hist(data["N"]/data["NumAtoms"], bins=20, label="N", color="#3b5b92", alpha=alpha)
axs[0].hist(data["H"]/data["NumAtoms"], bins=20, label="H", color="lightgrey", alpha=alpha)
axs[0].hist(data["O"]/data["NumAtoms"], bins=20, label="O", color="maroon", alpha=alpha)
axs[0].hist(data["S"]/data["NumAtoms"], bins=20, label="S", color="gold", alpha=alpha)
axs[0].hist(data["F"]/data["NumAtoms"], bins=20, label="F", color="seagreen", alpha=alpha)
axs[0].set_xlabel("Mole Fraction")
axs[0].set_ylabel("Counts")
axs[0].set_ylim([0., 25.])
axs[0].legend(loc="upper right")


axs[1].hist(data["AtomIsInRing"]/data["NumAtoms"], bins=20, label="Ring", color="#3b5b92", alpha=alpha)
axs[1].hist(data["AtomIsAromatic"]/data["NumAtoms"], bins=20, label="Aromatic", color="goldenrod", alpha=alpha)
axs[1].hist(data["NumRotatableBonds"]/data["NumBonds"], bins=20, label="Rotatable", color="seagreen", alpha=alpha)
axs[1].hist(data["BondIsConjugated"]/data["NumBonds"], bins=20, label="Conjugated", color="indianred", alpha=alpha)
axs[1].set_xlabel("Mole Fraction")
# axs[0].set_ylim([0., 25.])
axs[1].legend(loc="upper left")


axs[2].hist(data["homo"], bins=20, label="Homo", color="#3b5b92", alpha=alpha)
axs[2].hist(data["lumo"], bins=20, label="Lumo", color="goldenrod", alpha=alpha)
axs[2].hist(data["gap"], bins=20, label="Gap", color="seagreen", alpha=alpha)
axs[2].set_xlabel("Energy [eV]")
# axs[0].set_ylim([0., 25.])
axs[2].legend(loc="upper left")


axs[3].hist(data["ExactMolWt"], bins=20, label="Weight", color="#3b5b92", alpha=alpha)
axs[3].set_xlabel("Molar Mass [g/mol]")
# axs[0].set_ylim([0., 25.])
axs[3].legend(loc="upper left")

axs[0].text(
    -0.2, 1.0, 'A', fontsize=18, weight="bold",
    transform=axs[0].transAxes,
)


plt.savefig("data_analysis.png", bbox_inches='tight', pad_inches=0.1)
plt.show()
