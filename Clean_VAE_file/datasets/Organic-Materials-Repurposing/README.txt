- Data extracted from GAUSSIAN16 QM calculations are provided in .csv format (CSD_EES_DB.csv), which can be read through common programs or programming languages.
The columns are: "ID,doi,formula,NAts,SMILES,HOMO,LUMO,E(S1),f(S1),E(S2),f(S2),E(S3),f(S3),E(T1),E(T2),E(T3)"; these are detailed below:

ID: unique CSD identifier
doi: doi of the experimental paper which characterises the X-ray structure
formula: chemical formula
NAts: number of heavy atoms
SMILES: the SMILES string
HOMO: computed HOMO energy (eV, X-ray geometry, M06-2X/def2-SVP)
LUMO: computed LUMO energy (eV, X-ray geometry, M06-2X/def2-SVP)
E(S1): computed S1 energy (eV, X-ray geometry, M06-2X/def2-SVP)
f(S1): computed S1 oscillator strength (X-ray geometry, M06-2X/def2-SVP)
E(S2): computed S2 energy (eV, X-ray geometry, M06-2X/def2-SVP)
f(S2): computed S2 oscillator strength (X-ray geometry, M06-2X/def2-SVP)
E(S3): computed S3 energy (eV, X-ray geometry, M06-2X/def2-SVP)
f(S3): computed S3 oscillator strength (X-ray geometry, M06-2X/def2-SVP)
E(T1): computed T1 energy (eV, X-ray geometry, M06-2X/def2-SVP)
E(T2): computed T2 energy (eV, X-ray geometry, M06-2X/def2-SVP)
E(T3): computed T3 energy (eV, X-ray geometry, M06-2X/def2-SVP)

-GAUSSIAN16 QM output (.log) files are compressed and provided in CSD_TDDFT_LOGFILES.zip (1.1GB total). 
Each output file is named "CCDC_XXXXXX_td.log", where "XXXXXX" is the unique CSD identifier. 
To extract the files using Ubuntu Linux CLI:
0) sudo apt install zip unzip (if not present)
1) unzip CSD_TDDFT_LOGFILES.zip

- Multiwfn wavefunction (.wfn) files are compressed and provided across a set of 1GB multi-part .zip files (SPLIT_CSD_TDDFT_WFNFILES.zxx) (30GB total). 
This allows for sequential or partial download.
Each wavefunction file is named "CCDC_XXXXXX_td.wfn", where "XXXXXX" is the unique CSD identifier.
To extract the multi-part .zip files (when collected together in a single directory along with SPLIT_CSD_TDDFT_WFNFILES.zip) using Ubuntu Linux CLI:
0) sudo apt install zip unzip (if not present)
1) zip -F SPLIT_CSD_TDDFT_WFNFILES.zip --out CSD_TDDFT_WFNFILES.zip 
2) unzip CSD_TDDFT_WFNFILES.zip

For partial download, ensure that the file SPLIT_CSD_TDDFT_WFNFILES.zip is downloaded along with the other multi-part files of choice and 
replace step 1) with: zip -FF SPLIT_CSD_TDDFT_WFNFILES.zip --out CSD_TDDFT_WFNFILES.zip

This will prompt the following message:

"Fix archive (-FF) - salvage what can
        zip warning: could not open input archive: SPLIT_CSD_TDDFT_WFNFILES.zip
Scanning for entries...


Could not find:
  SPLIT_CSD_TDDFT_WFNFILES.zxx

Hit c      (change path to where this split file is)
    s      (skip this split)
    q      (abort archive - quit)
    e      (end this archive - no more splits)
    z      (look for .zip split - the last split)
 or ENTER  (try reading this split again):"

Where ".zxx" is the missing multi-part .zip file. 
Skip the missing multi-part files by entering "s" when prompted. Once all the multi-part .zip files are accounted for, end the archive by entering "e".
Then continue with step 2): unzip CSD_TDDFT_WFNFILES.zip

