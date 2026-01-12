import pandas as pd
from .interfaces import IDataLoader

class CSVLoader(IDataLoader):
    """Simple CSV loader.

    Args:
        file_path: Path to CSV file
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Load a CSV into a pandas DataFrame.

        Raises:
            FileNotFoundError: if the CSV cannot be found at ``file_path``.
        """
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")
