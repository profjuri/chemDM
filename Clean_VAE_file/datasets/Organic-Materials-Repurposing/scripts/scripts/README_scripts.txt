- The scripts and data used to create the plots are compressed and provided in scripts.zip.
To extract the files using Ubuntu Linux CLI:
0) sudo apt install zip unzip (if not present)
1) unzip scripts.zip

- The data is split across five files which are provided in .csv format, and can be read through common programs or programming languages.

1) basis_funnel.csv has the columns: "ID,HOMO_B3_321Gd,LUMO_B3_321Gd,HOMO_321_cal,LUMO_321_cal"; these are detailed below:

ID: unique CSD identifier
HOMO_B3_321Gd: computed HOMO energy (eV, X-ray geometry, B3LYP/3-21G*)
LUMO_B3_321Gd: computed LUMO energy (eV, X-ray geometry, B3LYP/3-21G*)
HOMO_321_cal: calibrated HOMO_B3_321Gd (eV)
LUMO_321_cal: calibrated LUMO_B3_321Gd (eV)

This data is used in oscs_fig1dg.py to create figure 1d) & figure 1g).

2) calibs.csv has the columns: "ID","doi","smiles","DoU","NAts","NAtsBin","HOMO_indo","LUMO_indo","HOMO_pm7","LUMO_pm7","HOMO_qm_321","LUMO_qm_321","HOMO_qm_631","LUMO_qm_631"; these are detailed below:

ID: unique CSD identifier
doi: doi of the experimental paper which characterises the X-ray structure
smiles: the SMILES string
DoU: degree of unsaturation
NAts: number of heavy atoms
NAtsBin: corresponding bin for the number of heavy atoms
HOMO_indo: computed HOMO energy (eV, X-ray geometry, INDO)
LUMO_indo: computed LUMO energy (eV, X-ray geometry, INDO)
HOMO_pm7: computed HOMO energy (eV, X-ray geometry, PM7)
LUMO_pm7: computed LUMO energy (eV, X-ray geometry, PM7)
HOMO_qm_321: computed HOMO energy (eV, X-ray geometry, B3LYP/3-21G*)
LUMO_qm_321: computed LUMO energy (eV, X-ray geometry, B3LYP/3-21G*)
HOMO_qm_631: computed HOMO energy (eV, X-ray geometry, B3LYP/6-31G*)
LUMO_qm_631: computed LUMO energy (eV, X-ray geometry, B3LYP/6-31G*)

This data is used in oscs_fig1bcef.py to create figure 1b) & figure 1c) & figure 1e) & figure 1f).

3) PM7_funnel.csv has the columns: "ID,HOMO_PM7,LUMO_PM7,HOMO_pm7_cal,LUMO_pm7_cal"; these are detailed below:

ID: unique CSD identifier
HOMO_PM7: computed HOMO energy (eV, X-ray geometry, PM7)
LUMO_PM7: computed LUMO energy (eV, X-ray geometry, PM7)
HOMO_pm7_cal: calibrated HOMO_PM7 (eV)
LUMO_pm7_cal: calibrated LUMO_PM7 (eV)

This data is used in oscs_fig1dg.py to create figure 1d) & figure 1g). 

4) OSCs_date_hist.csv has the columns: "year,tot_depo,osc_depo,frac,frac_per"; these are detailed below:

year: year of entry submission
tot_depo: total entries submitted
osc_depo: number of entries defined as "organic semiconductors" submitted
frac: ratio of osc_depo and tot_depo
frac_per: frac * 100

This data is used in fig2a.py & fig2b.py to create figure 2a) & figure 2b). 

5) The remaining scripts oscs_fig3a.py, oscs_fig3b.py, oscs_fig4a.py and oscs_fig4b.py use the data provided in "CSD_EES_DB.csv" to create figure 3a) & figure 3b) & figure 4a) & figure 4b). 
"CSD_EES_DB.csv" is further detailed in "README.txt". 