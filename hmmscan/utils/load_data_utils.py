import os
import pandas as pd
from pathlib import Path


def load_data(directory, name, index_col=None, nrows=None):
    """Load data from a specified subdirectory, with specified name."""

    path = os.path.join(get_data_path(), directory, name)
    data = pd.read_csv(path, index_col=index_col, nrows=nrows)

    return data


def get_ae_path(subdir=None):
    """Return path stored in ae root directory."""

    path = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'shared-path.txt')

    # Read first line.
    with open(path, 'r') as f:
        if subdir is None:
            full_path = f.readline().strip()
        else:
            full_path = os.path.join(f.readline().strip(), subdir)

    # Make sure that the path exists
    os.makedirs(full_path, exist_ok=True)

    # Return the full path
    return full_path


def get_data_path():
    """Return path stored in ae root directory."""

    return get_ae_path('data')


def get_results_path():
    """Return path stored in ae root directory."""

    return get_ae_path('results')


def write_data_csv(df, subdir, name, **kwargs):
    write_index = kwargs.get('index', False)
    write_col_names = kwargs.get('col_names', True)

    path = os.path.join(get_data_path(), subdir, name + '.csv')
    os.makedirs(os.path.join(get_data_path(), subdir), exist_ok=True)
    df.to_csv(path, index=write_index, header=write_col_names)


def write_result_csv(df, subdir, name, **kwargs):
    """Write data to a specified subdirectory, with specified name."""
    write_index = kwargs.get('index', True)
    write_col_names = kwargs.get('col_names', True)

    os.makedirs(os.path.join(get_results_path(), subdir), exist_ok=True)
    path = os.path.join(get_results_path(), subdir, name + '.csv')
    df.to_csv(path, index=write_index, header=write_col_names)
