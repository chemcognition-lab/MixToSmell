import os
import pandas as pd
import numpy as np 
from collections import Counter
from pandas.testing import assert_frame_equal

def merge_tasks(group):
    merged_row = group.iloc[0].copy()
    for col in task_cols:
        merged_row[col] = group[task_cols][col].max()
    return merged_row

def str_to_int_list(s):
    if pd.isna(s):
        return s 
    return [int(i) for i in str(s).split(';') if i]

def str_to_float_list(s):
    if pd.isna(s):
        return s 
    return [float(i) for i in str(s).split(';') if i]

def str_to_list(s):
    if pd.isna(s):
        return s 
    return [i for i in str(s).split(';') if i]

def id_to_compound_id(id_list, lookup_dict):
    if not isinstance(id_list, list):
        return lookup_dict.get(id_list, id_list)
    return [lookup_dict.get(i, i) for i in id_list]

def id_to_smiles(id_list, lookup_dict):
    if not isinstance(id_list, list):
        return id_list
    return [lookup_dict.get(i) for i in id_list]

def list_nan_to_nan(x):
    # If it's a list with a single NaN inside, replace with NaN
    if isinstance(x, list) and len(x) == 1 and pd.isna(x[0]):
        return np.nan
    return x

if __name__ == "__main__":
    print("\n=== PROCESSING KELLER ===")
    # Load the Keller dataset
    keller_data = pd.read_csv("./keller_combined.csv")
    
    print(f"Original Keller dataset shape: {keller_data.shape}")
    print(f"Columns: {list(keller_data.columns)}")
    
    # ===== PROCESS LABELS (keller_labels.csv) =====
    # Define descriptor columns    
    intensity_pleasantness = ["Intensity", "Pleasantness"]
    ratings = ["Acid", "Ammonia", "Bakery", "Burnt", "Chemical", "Cold", "Decayed", 
    "Fish", "Flower", "Fruit", "Garlic", "Grass", 
    "Musky", "Sour", "Spices", "Sweaty", "Sweet", "Warm", "Wood"]

    descriptor_cols = intensity_pleasantness + ratings
    
    # Create labels dataframe
    keller_labels = keller_data[["stimulus", "Dilution"] + descriptor_cols].copy()
    
    # Sort and reset index by both stimulus and dilution
    keller_labels = keller_labels.sort_values(['stimulus', "Dilution"]).reset_index(drop=True)

    # Convert to string first
    keller_labels['stimulus'] = keller_labels['stimulus'].astype(str)

    # Create masks for first and second occurrences
    first_occurrence = ~keller_labels.duplicated(subset=['stimulus'], keep='first')
    second_occurrence = keller_labels.duplicated(subset=['stimulus'], keep='first')

    # Add A/B prefixes
    keller_labels.loc[first_occurrence, 'stimulus'] = 'A' + keller_labels.loc[first_occurrence, 'stimulus']
    keller_labels.loc[second_occurrence, 'stimulus'] = 'B' + keller_labels.loc[second_occurrence, 'stimulus']
    
    # Add task indicator (since this is all one dataset type)
    keller_labels["keller"] = 1
    
    # Create rata column (list of all descriptor values)
    keller_labels["intensity_pleasantness"] = keller_labels[intensity_pleasantness].values.tolist()
    keller_labels["ratings"] = keller_labels[ratings].values.tolist()
    keller_labels["ratings_intensity_pleasantness"] = keller_labels[descriptor_cols].values.tolist()
    
    #remove dilution column
    keller_labels = keller_labels.drop(columns=["Dilution"])
    
    # Reorder columns
    preferred_order = ['stimulus', 'keller'] + descriptor_cols + ['ratings', 'intensity_pleasantness', 'ratings_intensity_pleasantness']
    keller_labels = keller_labels[preferred_order]

    
    print(f"Keller labels shape: {keller_labels.shape}")
    keller_labels.to_csv("./keller_labels.csv", index=False)
    
    # ===== PROCESS INPUTS (keller_inputs.csv) =====
    # Create inputs dataframe with stimulus and chemical information
    keller_inputs = keller_data[["stimulus", "Dilution", "CID"]].copy()

    # Sort and reset index
    keller_inputs = keller_inputs.sort_values(['stimulus', "Dilution"]).reset_index(drop=True)
    
    # Convert to string first
    keller_inputs['stimulus'] = keller_inputs['stimulus'].astype(str)

    # Create masks for first and second occurrences
    first_occurrence = ~keller_inputs.duplicated(subset=['stimulus'], keep='first')
    second_occurrence = keller_inputs.duplicated(subset=['stimulus'], keep='first')

    # Add A/B prefixes without underscore
    keller_inputs.loc[first_occurrence, 'stimulus'] = 'A' + keller_inputs.loc[first_occurrence, 'stimulus']
    keller_inputs.loc[second_occurrence, 'stimulus'] = 'B' + keller_inputs.loc[second_occurrence, 'stimulus']
    
    
    keller_inputs["dilution"] = keller_inputs["Dilution"].apply(lambda x: [float(x)] if pd.notna(x) else [])
    
    # Convert CID to list format (single compound)
    keller_inputs["CID"] = keller_inputs["CID"].apply(lambda x: [int(x)] if pd.notna(x) else [])
    
    # Get unique CIDs and map them to sequential IDs
    cid_values = [cid[0] if isinstance(cid, list) and len(cid) > 0 else cid for cid in keller_inputs["CID"]]
    unique_cids = pd.Series(cid_values).unique()
    cid_to_id_map = {cid: i for i, cid in enumerate(unique_cids)}

    # Map each row's CID to its corresponding ID
    keller_inputs["cmp_ids"] = [[cid_to_id_map[cid[0] if isinstance(cid, list) else cid]] for cid in keller_inputs["CID"]]
    
    
    #create dilution_info column with one-hot encoding for solvent as list item with 0
    keller_inputs["solvent_onehot"] = [[0]] * len(keller_inputs)
    keller_inputs["dilution_info"] = keller_inputs.apply(lambda row: [row["dilution"], row["solvent_onehot"]], axis=1)
    #remove solvent_onehot column
    keller_inputs = keller_inputs.drop(columns=["solvent_onehot"])
    
    # Select final columns for inputs
    final_input_cols = ["stimulus", "CID", "dilution", "cmp_ids", "dilution_info"]
    keller_inputs = keller_inputs[final_input_cols]
    
    
    print(f"Keller inputs shape: {keller_inputs.shape}")
    keller_inputs.to_csv("./keller_inputs.csv", index=False)
    
    # ===== PROCESS CID MAPPING (keller_CID.csv) =====
    # Create CID mapping file similar to updated_CID.csv
    cid_mapping = []
    
    for _, row in keller_data.iterrows():
        cid_mapping.append({
            "CID": row["CID"],
            "SMILES": row["IsomericSMILES"]
        })
    
    keller_cid = pd.DataFrame(cid_mapping)
    keller_cid = keller_cid.drop_duplicates(subset=["CID"])
    
    print(f"Keller CID mapping shape: {keller_cid.shape}")
    keller_cid.to_csv("./keller_CID.csv", index=False)
    
    # ===== SUMMARY =====
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Generated files:")
    print(f"- keller_labels.csv: {keller_labels.shape[0]} rows, {keller_labels.shape[1]} columns")
    print(f"- keller_inputs.csv: {keller_inputs.shape[0]} rows, {keller_inputs.shape[1]} columns")
    print(f"- keller_CID.csv: {keller_cid.shape[0]} rows, {keller_cid.shape[1]} columns")

    print("\n=== PROCESSING DREAM 2025 DATA ===")
    #-----Stack Task1 and Task2 labels (y) together
    task1_labels = pd.read_csv(os.path.abspath("../raw/TASK1_training.csv"))
    task2_labels_single = pd.read_csv(os.path.abspath("../raw/Task2_single_RATA_fixed_V2.csv"))
    task2_labels_mix = pd.read_csv(os.path.abspath("../raw/TASK2_Train_mixture_Dataset.csv"))

    # Remove rows not in other df, remove NoOdor row as always NA or 0
    task2_labels_single_extra_info = task2_labels_single[["stimulus", "components","molecule","dilution"]].copy()
    task2_labels_single = task2_labels_single.drop(columns=["components","molecule","dilution", "NoOdor"])

    task1_labels["Task1"] = 1
    task1_labels["Task2_single"] = 0
    task1_labels["Task2_mix"] = 0

    task2_labels_single["Task1"] = 0
    task2_labels_single["Task2_single"] = 1
    task2_labels_single["Task2_mix"] = 0

    task2_labels_mix["Task1"] = 0
    task2_labels_mix["Task2_single"] = 0
    task2_labels_mix["Task2_mix"] = 1

    task_labels = pd.concat([task1_labels, task2_labels_single, task2_labels_mix], ignore_index=True)

    task_cols = ["Task1", "Task2_single", "Task2_mix"]

    # Columns that must be the same (within tolerance)
    feature_cols = [col for col in task_labels.columns if col not in ["stimulus"] + task_cols]
    rounded = task_labels[feature_cols].round(6)

    # Create a hashable key from rounded features
    task_labels['_key'] = rounded.astype(str).agg('-'.join, axis=1)

    # Group by stimulus and key to find duplicates that match in all features (within tolerance)
    grouped = task_labels.groupby(['stimulus', '_key'])

    # Apply merging
    deduplicated = grouped.apply(merge_tasks).reset_index(drop=True)

    # Drop temporary key
    deduplicated = deduplicated.drop(columns=['_key'])

    preferred_order = ['stimulus', 'Task1', 'Task2_single', 'Task2_mix']
    other_columns = [col for col in deduplicated.columns if col not in preferred_order]

    # Reorder the DataFrame
    deduplicated = deduplicated[preferred_order + other_columns]

    rata_columns = [col for col in deduplicated.columns if col not in ["stimulus","Task1","Task2_single","Task2_mix","Intensity","Pleasantness"]]
    deduplicated[rata_columns] = deduplicated[rata_columns] / 5
    deduplicated["rata"] = deduplicated[rata_columns].values.tolist()

    intensity_columns = [col for col in deduplicated.columns if col in ["Intensity","Pleasantness"]]
    deduplicated[intensity_columns] = deduplicated[intensity_columns] / 10
    deduplicated["intensity_pleasantness"] = deduplicated[intensity_columns].values.tolist()

    rata_intensity_columns = [col for col in deduplicated.columns if col not in ["stimulus","Task1","Task2_single","Task2_mix", "rata", "intensity_pleasantness"]]
    deduplicated["rata_intensity_pleasantness"] = deduplicated[rata_intensity_columns].values.tolist()

    # Remove datapoints associated to negative CIDs (No SMILES)
    deduplicated = deduplicated[~deduplicated['stimulus'].isin(["AN759", "AN873", "AN874"])]

    deduplicated.to_csv("task_labels.csv", index=False)

    #-----Stack Task 1 and task 2 stimuli (X) definition together
    task1_inputs = pd.read_csv(os.path.abspath("../raw/TASK1_Stimulus_definition.csv"))
    # task2_inputs_single = task2_labels_single_extra_info
    task2_inputs_mix_stimuli = pd.read_csv(os.path.abspath("../raw/TASK2_Stimulus_definition_fixed_V1.csv"))
    task2_inputs_mix_cmp = pd.read_csv(os.path.abspath("../raw/TASK2_Components_definition_fixed_V1.csv"))

    task1_inputs = task1_inputs.rename(columns={"molecule": "CID"}).drop(columns=["Intensity_label"])
    task2_inputs_mix_cmp = task2_inputs_mix_cmp.rename(columns={"molecule": "CID"})
    # task2_inputs_single = task2_inputs_single.rename(columns={"molecule": "CID"})
    task2_inputs_mix_stimuli = task2_inputs_mix_stimuli.rename(columns={"id": "stimulus"})

    # Task1
    task1_inputs["components"] = pd.Series(dtype='str')
    task1_inputs["solvent"] = task1_inputs["solvent"].apply(lambda x: str.lower(x))

    # Task2 single
    # task2_inputs_single["solvent"] = pd.Series(dtype='str')
    # task2_inputs_single["components"] = task2_inputs_single["components"].astype("Int64")

    # for i, row in task2_inputs_single.iterrows():

    #     if not pd.isna(row["components"]):
    #         def_row = task2_inputs_mix_cmp[task2_inputs_mix_cmp["id"] == row["components"]]
    #         task2_inputs_single.loc[i, "solvent"] = str(def_row["solvent"].values[0])
    #     else:
    #         def_row = task1_inputs[task1_inputs["stimulus"] == row["stimulus"]]
    #         task2_inputs_single.loc[i, "solvent"] = str(def_row["solvent"].values[0])

    # Task2 mix
    task2_inputs_mix_stimuli["CID"] = pd.Series(dtype='str')
    task2_inputs_mix_stimuli["dilution"] = pd.Series(dtype='str')
    task2_inputs_mix_stimuli["solvent"] = pd.Series(dtype='str')

    for i, row in task2_inputs_mix_stimuli.iterrows():

        components = [int(x) for x in row["components"].split(";")]

        cid_list = []
        dilution_list = []
        solvent_list = []

        for cmp in components:
            def_row = task2_inputs_mix_cmp[task2_inputs_mix_cmp["id"] == cmp]

            cid_list.append(str(def_row["CID"].values[0]))
            dilution_list.append(str(def_row["dilution"].values[0]))
            solvent_list.append(str(def_row["solvent"].values[0]))

        task2_inputs_mix_stimuli.loc[i, "CID"] = ";".join(cid_list)
        task2_inputs_mix_stimuli.loc[i, "dilution"] = ";".join(dilution_list)
        task2_inputs_mix_stimuli.loc[i, "solvent"] = ";".join(solvent_list)

    # task_inputs = pd.concat([task1_inputs, task2_inputs_single, task2_inputs_mix_stimuli])
    task_inputs = pd.concat([task1_inputs, task2_inputs_mix_stimuli])
    task_inputs = task_inputs.applymap(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    task_inputs = task_inputs.drop_duplicates()

    task_inputs = task_inputs.sort_values(by="stimulus")
    duplicated = task_inputs.duplicated(subset=["stimulus","CID","dilution","solvent"], keep=False)
    task_inputs = task_inputs[~(duplicated & task_inputs["components"].isna())]

    # Removing data points if stimulus not in Train/LB/Test
    lb_1 = pd.read_csv(os.path.abspath("../submission_templates/TASK1_leaderboard_set_Submission_form.csv"))
    test_1 = pd.read_csv(os.path.abspath("../submission_templates/TASK1_test_set_Submission_form.csv"))

    lb_2 = pd.read_csv(os.path.abspath("../submission_templates/TASK2_Leaderboard_set_Submission_form.csv"))
    test_2 = pd.read_csv(os.path.abspath("../submission_templates/TASK2_Test_set_Submission_form.csv"))

    no_data_stimulus = set(task_inputs["stimulus"].unique()) - set(deduplicated["stimulus"].unique()) - set(lb_1["stimulus"].unique()) - set(lb_2["stimulus"].unique()) - set(test_1["stimulus"].unique()) - set(test_2["stimulus"].unique())

    print("Removing data points if stimulus not in Train/LB/Test:", no_data_stimulus)

    task_inputs = task_inputs[~task_inputs["stimulus"].isin(no_data_stimulus)]

    # Use lists
    task_inputs[["CID", "components"]] = task_inputs[["CID", "components"]].map(str_to_int_list)
    task_inputs["dilution"] = task_inputs["dilution"].apply(str_to_float_list)
    task_inputs["solvent"] = task_inputs["solvent"].apply(str_to_list)

    # Map solvent to unique numerical values
    solvents = [val for sublist in task_inputs["solvent"] for val in sublist]
    solvents_counts = Counter(solvents)
    sorted_solvents = [val for val, _ in solvents_counts.most_common()]
    solvent_to_id = {val: i for i, val in enumerate(sorted_solvents)}

    task_inputs["solvent_onehot"] = task_inputs["solvent"].apply(lambda lst: [solvent_to_id[val] for val in lst])

    task_inputs["dilution_info"] = task_inputs.apply(lambda row: [row["dilution"], row["solvent_onehot"]], axis=1)

    # Remove datapoints associated to negative CIDs (No SMILES)
    task_inputs = task_inputs[~task_inputs['stimulus'].isin(["AN759", "AN873", "AN874"])]

    # Add compound_id info using updated CID file
    id_to_smi = pd.read_csv("./updated_CID.csv")
    lookup_dict = dict(zip(id_to_smi["CID"], id_to_smi["compound_id"]))
    task_inputs["cmp_ids"] = task_inputs["CID"].apply(lambda x: id_to_smiles(x, lookup_dict))

    # Add SMILES info using updated CID file
    # cid_to_smi = pd.read_csv("./updated_CID.csv")
    # lookup_dict = dict(zip(cid_to_smi["CID"], cid_to_smi["SMILES"]))

    # task_inputs["SMILES"] = task_inputs["CID"].apply(lambda x: cid_to_smiles(x, lookup_dict))

    # Add molecule specific dilution ratio
    df_exploded = task_inputs.explode(["cmp_ids", "CID", "dilution", "solvent", "components", "solvent_onehot"]).drop(columns=["dilution_info"]).reset_index(drop=True)
    processed_chunks = []
    for cid in df_exploded["CID"].unique():
        tmp = df_exploded.loc[df_exploded["CID"] == cid].sort_values(by="dilution", ascending=True).reset_index(drop=True)

        # Get min and max
        d_min = tmp["dilution"].min()
        d_max = tmp["dilution"].max()

        # Linear interpolation
        tmp["dilution_weakness"] = np.interp(tmp["dilution"], [d_min, d_max], [0, 1])

        processed_chunks.append(tmp)

    df_exploded = pd.concat(processed_chunks, ignore_index=True)

    grouped = (
        df_exploded.groupby("stimulus", sort=True)
        .agg({col: (lambda x: list(x)) if col in ["cmp_ids", "CID", "dilution", "solvent", "components", "solvent_onehot", "dilution_weakness"] else "first"
              for col in df_exploded.columns if col != "stimulus"})
        .reset_index()
    )

    grouped["dilution_info"] = grouped.apply(lambda row: [row["dilution"], row["solvent_onehot"], row["dilution_weakness"]], axis=1)
    grouped["components"] = grouped["components"].apply(list_nan_to_nan)

    grouped.to_csv("task_inputs_dw.csv", index=False)
    task_inputs.to_csv("task_inputs.csv", index=False)
 

    # LB data
    lb_1["Task1"] = 1
    lb_1["Task2_mix"] = 0

    lb_2["Task1"] = 0
    lb_2["Task2_mix"] = 1


    task_lb = pd.concat([lb_1, lb_2], ignore_index=True)
    rata_columns = [col for col in task_lb.columns if col not in ["stimulus","Task1","Task2_single","Task2_mix","Intensity","Pleasantness"]]
    task_lb["rata"] = pd.Series(dtype='str')
    task_lb = task_lb.drop(columns=rata_columns)
    task_lb.to_csv("task_leaderboard.csv", index=False)

    print("\n=== PROCESSING COMPLETE ===")

