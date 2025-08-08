import pandas as pd
import json
import os
import ast
from typing import Optional
from pathlib import Path

COLUMN_PROPERTY = "property"
COLUMN_TASKS = ["Task1", "Task2_single", "Task2_mix"] + ['gslf'] + ['keller']
COLUMN_VALUE = "value"


class MixtureDataInfo:
    """
    Base class for Mixture Data Information
    """
    def __init__(
            self,
            name: str,
            description: str,
            id_column: list[str],
            fraction_column: list[str],
            output_column: str,
            data_dir: str,
            compound_csv_name: str,
            input_csv_name: str,
            label_csv_name: str, 
            feature_column: Optional[list[str]] = None,
    ):
        self.name = name
        self.description = description
        self.metadata = {
            "columns": {
                "id_column": id_column,
                "fraction_column": fraction_column,
                "output_column": output_column,
                "feature_column": feature_column,
            }
        }

        # Load compound data
        compounds_path = os.path.join(data_dir, f"{compound_csv_name}.csv")
        if not os.path.exists(compounds_path):
            raise FileNotFoundError(f"The file {compounds_path} does not exist.")
    
        self.compounds = pd.read_csv(compounds_path, index_col="CID")
        
        # Load and filter input
        self.data_dir = data_dir

        input_path = os.path.join(data_dir, f"{input_csv_name}.csv")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file {input_path} does not exist.")
        
        self.inputs = pd.read_csv(input_path)

        for i in ["id_column", "fraction_column"]:
            cols = self.metadata["columns"][i]
            for col in cols:
                if col is not None:
                    self.inputs[col] = self.inputs[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        label_path = os.path.join(data_dir, f"{label_csv_name}.csv")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"The file {label_path} does not exist.")
        self.labels = pd.read_csv(label_path)
        
        self.labels[self.metadata["columns"]["output_column"]] = self.labels[self.metadata["columns"]["output_column"]].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def export_metadata(self, filepath="dataset_metadata.json"):
        """Exports metadata to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.metadata, f, indent=4)
    
    def summary(self):
        """Prints metadata summary."""
        print(f"Dataset: {self.name}")
        print(f"Description: {self.description}\n")
        print("Column Information:")
        for col, desc in self.metadata["columns"].items():
            print(f"- {col}: {desc}")


class TrainingData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Training Data",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "task_inputs_dw",
            label_csv_name: str = "task_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )


class TrainingDataAll(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Training Data",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata_intensity_pleasantness",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "task_inputs_dw",
            label_csv_name: str = "task_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )

class TrainingDataIntensity(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Training Data",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "intensity_pleasantness",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "task_inputs_dw",
            label_csv_name: str = "task_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )

class LBData(MixtureDataInfo):
    # Leaderboard data
    def __init__(
            self,
            name: str = "LeaderBoard Data",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "task_inputs_dw",
            label_csv_name: str = "task_leaderboard",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )


class TestData(MixtureDataInfo):
    # Leaderboard data
    def __init__(
            self,
            name: str = "Test Data",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "rata",
            data_dir: str | Path | None = None,
            compound_csv_name: str = "updated_CID",
            input_csv_name: str = "task_inputs_dw",
            label_csv_name: str = "task_test",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )


class GSLFData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Goodscents Leffingwell",
            description: str = """TBD""",
            id_column: str = ["cmp_ids"],
            fraction_column: str = [],      # no dilution available
            output_column: str = "rata", 
            data_dir: str | Path | None = None,
            compound_csv_name: str = "gslf_CID",
            input_csv_name: str = "gslf_inputs",
            label_csv_name: str = "gslf_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )


class KellerData(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Keller2017",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "ratings", 
            data_dir: str | Path | None = None,
            compound_csv_name: str = "keller_CID",
            input_csv_name: str = "keller_inputs",
            label_csv_name: str = "keller_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )


class KellerDataAll(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Keller2017",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "ratings_intensity_pleasantness", 
            data_dir: str | Path | None = None,
            compound_csv_name: str = "keller_CID",
            input_csv_name: str = "keller_inputs",
            label_csv_name: str = "keller_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )


class KellerDataIntensity(MixtureDataInfo):
    def __init__(
            self,
            name: str = "Keller2017",
            description: str = """TBD""",
            id_column: list[str] = ["cmp_ids"],
            fraction_column: list[str] = ["dilution_info"],
            output_column: str = "intensity_pleasantness", 
            data_dir: str | Path | None = None,
            compound_csv_name: str = "keller_CID",
            input_csv_name: str = "keller_inputs",
            label_csv_name: str = "keller_labels",
    ):
        # Determine default data_dir at runtime if not provided
        if data_dir is None:
            # go up four levels: data.py -> data -> mix2smell -> src -> dream2025
            project_root = Path(__file__).resolve().parents[3]
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
        )

# Catalog mapping
DATA_CATALOG = {
    "training": TrainingData,
    "training_all": TrainingDataAll,
    "training_intensity": TrainingDataIntensity,
    "leaderboard": LBData,
    "gslf": GSLFData,
    "keller": KellerData,
    "keller_all": KellerDataAll,
    "keller_intensity": KellerDataIntensity,
}
