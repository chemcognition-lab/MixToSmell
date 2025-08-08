import os
import pubchempy as pcp
import pandas as pd
import numpy as np 
import time

if __name__ == "__main__":

    cid_original = pd.read_csv(os.path.abspath("../raw/CID.csv"))
    task2_components = pd.read_csv(os.path.abspath("../raw/TASK2_Components_definition_fixed_V1.csv"))

    task2_cids = set(task2_components["molecule"].unique())
    missing_cids = task2_cids - set(cid_original["molecule"].unique())

    print("Missing the following CIDs in original CID file:", missing_cids)

    added_cids = {"molecule": [], "SMILES": []}

    for cid in missing_cids:
        try:
            compound = pcp.Compound.from_cid(str(cid))
            added_cids["molecule"].append(cid)
            added_cids["SMILES"].append(compound.isomeric_smiles)
        except:
            print(f"{cid} failed")
        time.sleep(1)
    
    new_cids = pd.DataFrame(added_cids)
    
    updated_cid = pd.concat([cid_original, new_cids], ignore_index=True)
    updated_cid = updated_cid.rename(columns={"molecule": "CID"})
    updated_cid["CID"] = updated_cid["CID"].apply(lambda x: int(x))

    # Remove negative CIDs (No SMILES)
    updated_cid = updated_cid[updated_cid["CID"] > 0]

    updated_cid = updated_cid.reset_index(drop=True)
    updated_cid.to_csv("updated_CID.csv", index=True, index_label="compound_id")
