#!/usr/bin/env python3
import pandas as pd
from rdkit import Chem
import pubchempy as pcp
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions
)

# ——— Helpers ——————————————————————————————————————————————————————————————

def enumerate_isomeric_smiles(smiles: str) -> list:
    """Return a list of all RDKit-generated isomeric SMILES, or empty list if invalid/missing."""
    if not isinstance(smiles, str) or not smiles.strip():
        return []
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    opts = StereoEnumerationOptions(unique=True)
    isomers = EnumerateStereoisomers(mol, options=opts)
    return [Chem.MolToSmiles(iso, isomericSmiles=False) for iso in isomers]


def get_pubchem_smiles(cid) -> str:
    """Fetch from PubChem; return isomeric if available, else canonical, else None."""
    try:
        cid_int = int(cid)
    except (ValueError, TypeError):
        return None
    if cid_int <= 0:
        return None
    try:
        comp = pcp.Compound.from_cid(cid_int)
        return comp.isomeric_smiles or comp.canonical_smiles
    except Exception:
        return None


def canonical_inchikey(smiles: str) -> str:
    """Return InChIKey for a SMILES, or None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    # sanitize and generate InChIKey
    try:
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        return None

# ——— Main ———————————————————————————————————————————————————————————————

def augment_smiles_table(input_csv: str, output_csv: str):
    # 1) load
    df = pd.read_csv(input_csv)

    # 2) compute RDKit isomers and PubChem SMILES
    df['rdkit_isomers']  = df['SMILES'].apply(enumerate_isomeric_smiles)
    df['pubchem_smiles'] = df['CID'].apply(get_pubchem_smiles)

    # 3) compute InChIKeys for matching
    df['pubchem_ik'] = df['pubchem_smiles'].apply(canonical_inchikey)

    def is_match(row):
        target_ik = row['pubchem_ik']
        if not target_ik:
            return False
        # check each RDKit isomer
        for sm in row['rdkit_isomers']:
            if canonical_inchikey(sm) == target_ik:
                return True
        return False

    df['match'] = df.apply(is_match, axis=1)

    # 4) report
    total   = len(df)
    matches = df['match'].sum()
    print(f"{matches}/{total} compounds where PubChem SMILES matches one of the RDKit isomeric SMILES via InChIKey equivalence")

    if matches < total:
        print("Mismatched entries:")
        for _, row in df[df['match'] == False].iterrows():
            cid    = row['CID']
            idx    = row['compound_id']
            rd_list = row['rdkit_isomers']
            pub_sm = row['pubchem_smiles']
            print(f"  ID {idx} (CID={cid}): PubChem='{pub_sm}', RDKit isomers={rd_list}")

    # 5) save
    df.to_csv(output_csv, index=False)
    print(f"Augmented table written to {output_csv}")


if __name__ == "__main__":
    # Hard‐coded paths
    import os
    print("cwd:", os.getcwd())

    input_csv  = "dataset/processed/updated_CID.csv"
    output_csv = "dataset/processed/CID_SMILES_check.csv"
    augment_smiles_table(input_csv, output_csv)
