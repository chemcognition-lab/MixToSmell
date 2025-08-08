from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

from mix2smell.data.data import MixtureDataInfo
from mix2smell.data.dataset import Mix2SmellData
from mix2smell.data.splits import SplitLoader

from mix2smell.data.splits import create_train_val_test_split, create_kfold_split
import torch

class MixBySingle(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Mix_By_Single",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "mixtures_with_agg_single",
            label_csv_name: str = "label_multi_mixture",
            feature_col: str = ["mix_rata"],
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            data_dir = project_root / "dataset" / "processed"
        data_dir = Path(data_dir)

        super().__init__(
            name,
            description,
            id_column,
            fraction_column,
            output_column,
            str(data_dir),
            compound_csv_name,
            input_csv_name,
            label_csv_name,
            feature_col
        )

class MixBySingleTest(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Mix_By_Single_Test",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "test_mixtures_with_agg_single",
            label_csv_name: str = "task_test2",
            feature_col: str = ["mix_rata"],
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            data_dir = project_root / "dataset" / "processed"
        data_dir = Path(data_dir)

        super().__init__(
            name,
            description,
            id_column,
            fraction_column,
            output_column,
            str(data_dir),
            compound_csv_name,
            input_csv_name,
            label_csv_name,
            feature_col
        )

class MixByAllSingle(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Mix_By_All_Single",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "mixtures_with_agg_single",
            label_csv_name: str = "label_multi_mixture",
            feature_col: str = ["all_rata"],
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            data_dir = project_root / "dataset" / "processed"
        data_dir = Path(data_dir)

        super().__init__(
            name,
            description,
            id_column,
            fraction_column,
            output_column,
            str(data_dir),
            compound_csv_name,
            input_csv_name,
            label_csv_name,
            feature_col
        )

if __name__ == "__main__":

    # get the input and label definitions
    input_info = pd.read_csv("dataset/processed/task_inputs.csv").set_index("stimulus")
    label_info = pd.read_csv("dataset/processed/task_labels.csv").set_index("stimulus")

    # add a column to the input to put the averaged RATA of the corresponding single molecules within its mixture
    input_info["mix_rata"] = [[0]*53]*len(input_info)
    input_info["all_rata"] = [[[]]*53]*len(input_info)
    new_df = pd.DataFrame({}, columns = input_info.columns)
    new_label = pd.DataFrame({}, columns = label_info.columns)
    
    new_df_full = pd.DataFrame({}, columns = input_info.columns)


    rata_avg = pd.DataFrame({"CID":[-10000], "avg_rata":[[]], "obs":[-1], "all_rata":[{}]}).set_index("CID")
    for stimulus in input_info.index:
        cids = (input_info.loc[stimulus]["CID"].replace("[","").replace("]","").replace(" ","").split(","))
        if len(cids) == 1:
            # if it is = 1 we add to the rata_avg df
            # if it already exists within the df, we add by running sum
            cid = cids[0]
            if cid in rata_avg.index and stimulus in label_info.index:
                rata_avg.loc[cid]["avg_rata"] = (rata_avg.loc[cid]["avg_rata"] * rata_avg.loc[cid]["obs"] + np.array([float(x) for x in label_info.loc[stimulus]["rata_intensity_pleasantness"].replace("[","").replace("]","").replace(" ","").split(",")]))/(rata_avg.loc[cid]["obs"]+1)
                rata_avg.loc[cid]["obs"] = rata_avg.loc[cid]["obs"] + 1
            else:
                if stimulus in label_info.index:
                    rata_avg.loc[cid] = {"CID": cid, "avg_rata": np.array([float(x) for x in label_info.loc[stimulus]["rata_intensity_pleasantness"].replace("[","").replace("]","").replace(" ","").split(",")]), "obs": 1, "all_rata":[]}
                else:
                    rata_avg.loc[cid] = {"CID": cid, "avg_rata": np.zeros(53), "obs": 1, "all_rata":[]}
            if stimulus in label_info.index:
                # this one stores all the information of the rata
                dummy = [float(input_info.loc[stimulus]["dilution"].replace("[","").replace("]","").replace(" ",""))]
                dummy.extend([float(x) for x in label_info.loc[stimulus]["rata_intensity_pleasantness"].replace("[","").replace("]","").replace(" ","").split(",")])
                rata_avg.loc[cid]["all_rata"].append(dummy) # list of lists of form [dilution, rata]

    max_len = 0
    for stimulus in input_info.index:
        if stimulus in label_info.index:
            cid_list = input_info.loc[stimulus]["CID"].replace("[","").replace("]","").replace(" ","").split(",")
            if len(cid_list) != 1:
                input_info.at[stimulus,"mix_rata"] = list(float(x) for x in np.mean(np.array([rata_avg.loc[cid, "avg_rata"] for cid in cid_list]), axis = 0))
                input_info.at[stimulus,"all_rata"] = [str(x).replace("[","{").replace("]","}") for x in rata_avg.loc[cid, "all_rata"] for cid in cid_list] # list of lists of form [dilution, rata] (CID blind single molecule in mixture rata information)
                max_len = max(max_len, len(input_info.at[stimulus,"all_rata"]))
                new_df.loc[stimulus] = input_info.loc[stimulus]
                new_label.loc[stimulus] = label_info.loc[stimulus]

    new_df["stimulus"] = new_df.index
    new_label["stimulus"] = new_label.index
    new_df = new_df.reset_index()[["stimulus","CID","dilution","solvent","components","solvent_onehot","dilution_info","cmp_ids","mix_rata","all_rata"]]
    new_label = new_label.reset_index()[["stimulus","Task1","Task2_single","Task2_mix","Intensity","Pleasantness","Green","Cucumber","Herbal","Mint","Woody","Pine","Floral","Powdery","Fruity","Citrus","Tropical","Berry","Peach","Sweet","Caramellic","Vanilla","BrownSpice","Smoky","Burnt","Roasted","Grainy","Meaty","Nutty","Fatty","Coconut","Waxy","Dairy","Buttery","Cheesy","Sour","Fermented","Sulfurous","Garlic.Onion","Earthy","Mushroom","Musty",'Ammonia',"Fishy","Fecal","Rotten.Decay",'Rubber',"Phenolic","Animal","Medicinal","Cooling","Sharp","Chlorine",'Alcoholic','Plastic',"Ozone","Metallic","rata","intensity_pleasantness","rata_intensity_pleasantness"]]

    new_df.to_csv("dataset/processed/mixtures_with_agg_single.csv")
    new_label.to_csv("dataset/processed/label_multi_mixture.csv")


    t1_to_2 = MixBySingle()


    # Call the torch dataset wrapper found in src/mix2smell/data/dataset.py
    t1_to_2_task = Mix2SmellData(
        dataset=t1_to_2,
        task=["Task2_mix"],  # Specify which Task of the challenge you want to load (here, we load them all)
        featurization="rdkit2d_normalized_features",  # Specify which featurization you want to use, see FEATURIZATION_TYPE variable in src/mix2smell/data/featurization.py for more info
    )

    # # Get data stats
    print(t1_to_2_task.__len__())
    print(t1_to_2_task.__max_num_components__())
    print(t1_to_2_task.__num_unique_mixtures__())

    # # Get one data point
    data_point = t1_to_2_task.__getitem__(0)
    print("All tensors in a datapoint:", data_point.keys())

    # Contains the padded custom ID (NOT the ones used in the Dream original files)
    print("ids tensor shape:", data_point["ids"].shape)

    # Contains the padded dilution factor + the one-hot encoded solvent type
    print("fractions tensor shape:", data_point["fractions"].shape)

    # Contains the 51-dimensional label vector
    print("label tensor shape:", data_point["label"].shape)

    # Contains the padded molecular feature vectors, based on selected featurization type
    print("features shape:", data_point["features"].shape)

    # Load a split, previously made using the make_splits.py scripts. For now we only have random kfold split.
    split_loader = SplitLoader(dataset_name= t1_to_2_task.dataset.name,
                task=t1_to_2_task.task,
                cache_dir=t1_to_2_task.dataset.data_dir)
    
    seed = 0

    labels = (torch.stack([t["label"] for t in t1_to_2_task]) == 0).to(float)

    create_train_val_test_split(
        t1_to_2_task.dataset.name,
        task=t1_to_2_task.task,
        mixture_indices_tensor=t1_to_2_task.indices_tensor,
        target_label_tensor=labels,     # this stratifies it by label
        cache_dir=t1_to_2.data_dir,
        test_size=0.2,
        seed=seed,
    )

    create_kfold_split(
        dataset_name= t1_to_2_task.dataset.name,
        task=t1_to_2_task.task,
        mixture_indices_tensor=t1_to_2_task.indices_tensor,
        target_label_tensor=labels,      # this stratifies it by label
        cache_dir=t1_to_2.data_dir,
        n_splits=5,
        seed=seed,
    )

    train_indices, val_indices, test_indices = split_loader()


    t1_to_2 = MixByAllSingle()


    # Call the torch dataset wrapper found in src/mix2smell/data/dataset.py
    t1_to_2_task = Mix2SmellData(
        dataset=t1_to_2,
        task=["Task2_mix"],  # Specify which Task of the challenge you want to load (here, we load them all)
        featurization="rdkit2d_normalized_features",  # Specify which featurization you want to use, see FEATURIZATION_TYPE variable in src/mix2smell/data/featurization.py for more info
    )

    # # Get data stats
    print(t1_to_2_task.__len__())
    print(t1_to_2_task.__max_num_components__())
    print(t1_to_2_task.__num_unique_mixtures__())

    # # Get one data point
    data_point = t1_to_2_task.__getitem__(0)
    print("All tensors in a datapoint:", data_point.keys())

    # Contains the padded custom ID (NOT the ones used in the Dream original files)
    print("ids tensor shape:", data_point["ids"].shape)

    # Contains the padded dilution factor + the one-hot encoded solvent type
    print("fractions tensor shape:", data_point["fractions"].shape)

    # Contains the 51-dimensional label vector
    print("label tensor shape:", data_point["label"].shape)

    # Contains the padded molecular feature vectors, based on selected featurization type
    print("features shape:", data_point["features"].shape)

    # Load a split, previously made using the make_splits.py scripts. For now we only have random kfold split.
    split_loader = SplitLoader(dataset_name= t1_to_2_task.dataset.name,
                task=t1_to_2_task.task,
                cache_dir=t1_to_2_task.dataset.data_dir)
    
    seed = 0

    labels = (torch.stack([t["label"] for t in t1_to_2_task]) == 0).to(float)

    create_train_val_test_split(
        t1_to_2_task.dataset.name,
        task=t1_to_2_task.task,
        mixture_indices_tensor=t1_to_2_task.indices_tensor,
        target_label_tensor=labels,     # this stratifies it by label
        cache_dir=t1_to_2.data_dir,
        test_size=0.2,
        seed=seed,
    )

    create_kfold_split(
        dataset_name= t1_to_2_task.dataset.name,
        task=t1_to_2_task.task,
        mixture_indices_tensor=t1_to_2_task.indices_tensor,
        target_label_tensor=labels,      # this stratifies it by label
        cache_dir=t1_to_2.data_dir,
        n_splits=5,
        seed=seed,
    )

    train_indices, val_indices, test_indices = split_loader()

     # get the input and label definitions
    input_info = pd.read_csv("dataset/processed/task_inputs.csv").set_index("stimulus")
    label_info = pd.read_csv("dataset/processed/task_labels.csv").set_index("stimulus")
    test_info = pd.read_csv("dataset/processed/task_inputs.csv").set_index("stimulus")
    test_stimuli = [
        "AA322", "AA374", "AA444", "AA524", "AA616", "AA700", "AA988",
        "AB026", "AB379", "AB391", "AB451", "AB518", "AB591", "AB676",
        "AB771", "AB817", "AB920", "AB958", "AB981",
        "AC033", "AC040", "AC105", "AC340", "AC737",
        "AD038", "AD062", "AD101", "AD133", "AD317", "AD347", "AD386", "AD567", "AD635", "AD639", "AD754", "AD876", "AD967", "AD995",
        "AE192", "AE207", "AE241", "AE280", "AE459", "AE675", "AE775",
        "AG090", "AG126", "AG616", "AG805",
        "AH263", "AH447", "AH552", "AH564", "AH702",
        "AI130", "AI216", "AI635", "AI814", "AI990",
        "AJ008", "AJ356", "AJ380", "AJ487", "AJ495", "AJ507", "AJ573", "AJ626", "AJ728", "AJ755", "AJ840",
        "AK033", "AK052", "AK103", "AK308", "AK455", "AK488", "AK788",
        "AL090", "AL137", "AL280", "AL334", "AL453", "AL542", "AL650", "AL883", "AL950",
        "AM019", "AM103", "AM221", "AM281", "AM311", "AM451", "AM483", "AM523", "AM796", "AM882", "AM907", "AM943", "AM944",
        "AN095", "AN243", "AN479", "AN802",
        "AO033", "AO180", "AO268", "AO284", "AO305", "AO440", "AO542", "AO579", "AO596", "AO615", "AO665", "AO697", "AO718", "AO849", "AO919",
        "AP177", "AP257", "AP320", "AP325", "AP477", "AP738", "AP752", "AP848",
        "AQ310", "AQ862", "AQ882", "AQ810"
    ]



    # add a column to the input to put the averaged RATA of the corresponding single molecules within its mixture
    test_info["mix_rata"] = [[0]*53]*len(test_info)
    test_info["all_rata"] = [[[]]*53]*len(test_info)
    new_df = pd.DataFrame({}, columns = test_info.columns)
    new_label = pd.DataFrame({}, columns = label_info.columns)
    
    new_df_full = pd.DataFrame({}, columns = test_info.columns)


    rata_avg = pd.DataFrame({"CID":[-10000], "avg_rata":[[]], "obs":[-1], "all_rata":[{}]}).set_index("CID")
    for stimulus in test_info.index:
        cids = (test_info.loc[stimulus]["CID"].replace("[","").replace("]","").replace(" ","").split(","))
        if len(cids) == 1:
            # if it is = 1 we add to the rata_avg df
            # if it already exists within the df, we add by running sum
            cid = cids[0]
            if cid in rata_avg.index and stimulus in label_info.index:
                rata_avg.loc[cid]["avg_rata"] = (rata_avg.loc[cid]["avg_rata"] * rata_avg.loc[cid]["obs"] + np.array([float(x) for x in label_info.loc[stimulus]["rata_intensity_pleasantness"].replace("[","").replace("]","").replace(" ","").split(",")]))/(rata_avg.loc[cid]["obs"]+1)
                rata_avg.loc[cid]["obs"] = rata_avg.loc[cid]["obs"] + 1
            else:
                if stimulus in label_info.index:
                    rata_avg.loc[cid] = {"CID": cid, "avg_rata": np.array([float(x) for x in label_info.loc[stimulus]["rata_intensity_pleasantness"].replace("[","").replace("]","").replace(" ","").split(",")]), "obs": 1, "all_rata":[]}
                else:
                    rata_avg.loc[cid] = {"CID": cid, "avg_rata": np.zeros(53), "obs": 1, "all_rata":[]}
            if stimulus in label_info.index:
                # this one stores all the information of the rata
                dummy = [float(input_info.loc[stimulus]["dilution"].replace("[","").replace("]","").replace(" ",""))]
                dummy.extend([float(x) for x in label_info.loc[stimulus]["rata_intensity_pleasantness"].replace("[","").replace("]","").replace(" ","").split(",")])
                rata_avg.loc[cid]["all_rata"].append(dummy) # list of lists of form [dilution, rata]

    max_len = 0


    
    for stimulus in test_stimuli:
        cid_list = test_info.loc[stimulus]["CID"].replace("[","").replace("]","").replace(" ","").split(",")
        if len(cid_list) != 1:
            test_info.at[stimulus,"mix_rata"] = list(float(x) for x in np.mean(np.array([rata_avg.loc[cid, "avg_rata"] for cid in cid_list]), axis = 0))
            test_info.at[stimulus,"all_rata"] = [str(x).replace("[","{").replace("]","}") for x in rata_avg.loc[cid, "all_rata"] for cid in cid_list] # list of lists of form [dilution, rata] (CID blind single molecule in mixture rata information)
            max_len = max(max_len, len(test_info.at[stimulus,"all_rata"]))
            new_df.loc[stimulus] = test_info.loc[stimulus]
            # new_label.loc[stimulus] = [None for i in range(len(["stimulus","Task1","Task2_single","Task2_mix","Intensity","Pleasantness","Green","Cucumber","Herbal","Mint","Woody","Pine","Floral","Powdery","Fruity","Citrus","Tropical","Berry","Peach","Sweet","Caramellic","Vanilla","BrownSpice","Smoky","Burnt","Roasted","Grainy","Meaty","Nutty","Fatty","Coconut","Waxy","Dairy","Buttery","Cheesy","Sour","Fermented","Sulfurous","Garlic.Onion","Earthy","Mushroom","Musty",'Ammonia',"Fishy","Fecal","Rotten.Decay",'Rubber',"Phenolic","Animal","Medicinal","Cooling","Sharp","Chlorine",'Alcoholic','Plastic',"Ozone","Metallic","rata","intensity_pleasantness","rata_intensity_pleasantness"]))]

    new_df["stimulus"] = new_df.index
    # new_label["stimulus"] = new_label.index
    new_df = new_df.reset_index()[["stimulus","CID","dilution","solvent","components","solvent_onehot","dilution_info","cmp_ids","mix_rata","all_rata"]]
    # new_label = new_label.reset_index()[["stimulus","Task1","Task2_single","Task2_mix","Intensity","Pleasantness","Green","Cucumber","Herbal","Mint","Woody","Pine","Floral","Powdery","Fruity","Citrus","Tropical","Berry","Peach","Sweet","Caramellic","Vanilla","BrownSpice","Smoky","Burnt","Roasted","Grainy","Meaty","Nutty","Fatty","Coconut","Waxy","Dairy","Buttery","Cheesy","Sour","Fermented","Sulfurous","Garlic.Onion","Earthy","Mushroom","Musty",'Ammonia',"Fishy","Fecal","Rotten.Decay",'Rubber',"Phenolic","Animal","Medicinal","Cooling","Sharp","Chlorine",'Alcoholic','Plastic',"Ozone","Metallic","rata","intensity_pleasantness","rata_intensity_pleasantness"]]

    new_df.to_csv("dataset/processed/test_mixtures_with_agg_single.csv")
    # new_label.to_csv("dataset/processed/test_label_multi_mixture.csv")


    t1_to_2 = MixBySingleTest()


    # Call the torch dataset wrapper found in src/mix2smell/data/dataset.py
    t1_to_2_task = Mix2SmellData(
        dataset=t1_to_2,
        task=["Task2_mix"],  # Specify which Task of the challenge you want to load
        featurization="rdkit2d_normalized_features",  # Specify which featurization you want to use, see FEATURIZATION_TYPE variable in src/mix2smell/data/featurization.py for more info
    )

    # # Get data stats
    print(t1_to_2_task.__len__())
    print(t1_to_2_task.__max_num_components__())
    print(t1_to_2_task.__num_unique_mixtures__())

    # # Get one data point
    data_point = t1_to_2_task.__getitem__(0)
    print("All tensors in a datapoint:", data_point.keys())

    # Contains the padded custom ID (NOT the ones used in the Dream original files)
    print("ids tensor shape:", data_point["ids"].shape)

    # Contains the padded dilution factor + the one-hot encoded solvent type
    print("fractions tensor shape:", data_point["fractions"].shape)

    # Contains the 51-dimensional label vector
    print("label tensor shape:", data_point["label"].shape)

    # Contains the padded molecular feature vectors, based on selected featurization type
    print("features shape:", data_point["features"].shape)
