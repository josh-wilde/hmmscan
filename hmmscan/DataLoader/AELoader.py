import numpy as np
import pandas as pd
from hmmscan.utils.load_data_utils import load_data


class AELoader:
    """
    Holds raw data and sequences for a particular type of AE count
    Also holds metadata about these raw data
    Designed to be fed to the fit command of a BinomialModel object
    """
    def __init__(self, directory, sequence_name, ae_type, gap, use_gap_q):

        # Set some initial metadata
        self.metadata = self.set_metadata(directory, sequence_name, ae_type, gap, use_gap_q)
        self.raw_data = pd.DataFrame()
        self.filled_df = pd.DataFrame()
        self.sequences_single = []
        self.sequences = []

        # Load the raw data and the clean data
        # Decided that I would save the augmented raw data
        # And the sequences because it takes a long time to process

        # Get the raw data with all columns
        all_cols = load_data(self.metadata['dir'], self.metadata['sequence_name'] + '.csv')
        all_cols['prefix'] = all_cols['prefix'].fillna("")
        self.load_sequences(all_cols, use_gap_q)

        # Add more metadata
        self.add_metadata()

    @staticmethod
    def set_metadata(directory, sequence_name, ae_type, gap, use_gap_q):
        metadata = {
            'dir': directory,  # relative to the ae/data folder
            'sequence_name': sequence_name,  # name of input data file is defined as the sequence name
            'ae_type': ae_type,  # type of ae count, name of column to get
        }

        if use_gap_q:
            metadata['gap_q'] = gap
            # Set the gap when loading the raw data
        else:
            metadata['gap_q'] = np.nan
            metadata['gap'] = gap

        return metadata

    def set_gap_fill(self, all_cols, use_gap_q):
        """
        Will set the fill_before column to determine where 0s should be added
        Assumes that column num is the numeric lot number column
        """

        data = all_cols.sort_values(by=['lot_group', 'num']).rename(columns={self.metadata['ae_type']: 'ae'})
        data['lot_num_gap'] = data.groupby(['lot_group'])['num'].diff()
        data = data[['lot_num', 'lot_num_gap', 'lot_group', 'prefix', 'num', 'ae']].reset_index()
        data['lot_num'] = data['lot_num'].astype(str)

        # If using a quantile for the gap, then set the threshold here
        if use_gap_q:
            self.metadata['gap'] = data['lot_num_gap'].quantile(self.metadata['gap_q'])

        # Then maybe set NAs to something huge in lot_num_gap
        data['lot_num_gap'] = data['lot_num_gap'].fillna(value=1e10)

        # And if that difference needs to be filled
        data['fill_before'] = ((data['lot_num_gap'] <= self.metadata['gap']) & (data['lot_num_gap'] > 1))

        return data

    def load_sequences(self, all_cols, use_gap_q):
        # Takes in the raw data with all columns
        # The raw data must have a lot_group column, also lot_num, num, and valid ae column
        self.raw_data = self.set_gap_fill(all_cols, use_gap_q)

        # Get dataframe that includes filler lots
        self.filled_df = self.get_filled_df()

        # Break into sequences of 2 consecutive lots or more
        self.sequences_single = self.split_sequences_single()
        self.sequences = self.split_sequences()

    def get_filler_lots(self):
        """
        Input is df with lot_num and fill_before columns
        Output the list of lots with filled numbers
        """
        lots_to_fill: pd.DataFrame = self.raw_data[['lot_num', 'prefix', 'num', 'lot_group', 'fill_before']].copy()
        prior_row = lots_to_fill.iloc[0, :]
        filled_nums = [prior_row['num']]
        current_prefix = '' if len(prior_row['prefix']) == 0 else prior_row['prefix']
        filled_prefixes = [current_prefix]
        filled_lot_nums = [prior_row['lot_num']]
        filled_lot_group = [prior_row['lot_group']]
        len_num_portion = len(str(prior_row['lot_num'])) - len(current_prefix)

        for index, row in lots_to_fill.iloc[1:, :].iterrows():
            if row['fill_before']:
                current_fill_nums = range(prior_row['num']+1, row['num']+1)
                current_prefix = '' if len(prior_row['prefix']) == 0 else row['prefix']
                current_fill_lot_nums = [current_prefix + str(i).rjust(len_num_portion, '0') for i in current_fill_nums]
                filled_nums = filled_nums + list(current_fill_nums)
                filled_prefixes = filled_prefixes + [current_prefix for _ in range(len(current_fill_lot_nums))]
                filled_lot_nums = filled_lot_nums + current_fill_lot_nums
                filled_lot_group = filled_lot_group + [row['lot_group'] for _ in range(len(current_fill_lot_nums))]
            else:
                current_prefix = '' if len(prior_row['prefix']) == 0 else row['prefix']
                filled_nums = filled_nums + [row['num']]
                filled_prefixes = filled_prefixes + [current_prefix]
                filled_lot_nums = filled_lot_nums + [row['lot_num']]
                filled_lot_group = filled_lot_group + [row['lot_group']]

            prior_row = row

        return pd.DataFrame(
            {
                'num': filled_nums,
                'prefix': filled_prefixes,
                'lot_num': filled_lot_nums,
                'lot_group': filled_lot_group
            }
        )

    def get_filled_df(self):
        # Columns lot_num, ae_type, lot_num_gap, fill_before
        # Get dataframe with lots including filler lots
        filled = self.get_filler_lots()

        # Add back the AEs for the observed lots
        filled = pd.merge(filled, self.raw_data[['lot_num', 'ae']], how='left', on='lot_num')

        # Make filler lots zero AE
        filled['ae'].fillna(0, inplace=True)

        # Add the lot number gap
        filled['lot_num_gap'] = filled.groupby(['lot_group'])['num'].diff()
        filled['lot_num_gap'] = filled['lot_num_gap'].fillna(value=1e10)

        return filled

    def split_sequences_single(self):
        """
        Input: filled lot dataframe (lot_num, ae, lot_num_gap)
        Output: list of np arrays that store sequences with 2 or more sequential lots
        """

        sequences = []
        seq_start_idx = 0
        # seq_start_lot = self.filled_df['lot_num'].iloc(0)

        for index, row in self.filled_df.iterrows():
            if index == self.filled_df.shape[0] - 1:  # if at last row of df
                if row['lot_num_gap'] == 1:
                    # Just add one sequence because last lot is same sequence as next to last lot
                    sequences.append(np.array(self.filled_df['ae'].iloc[seq_start_idx:index+1]).reshape(-1, 1))
                else:
                    # Add two sequences - one for the sequence ending with second to last lot and one for last lot
                    sequences.append(np.array(self.filled_df['ae'].iloc[seq_start_idx:index]).reshape(-1, 1))
                    sequences.append(np.array(self.filled_df['ae'].iloc[index: index+1]).reshape(-1, 1))
            else:
                if row['lot_num_gap'] > 1:  # if gap is 1, then do nothing. Otherwise, mark end of sequence
                    sequences.append(np.array(self.filled_df['ae'].iloc[seq_start_idx: index]).reshape(-1, 1))
                    seq_start_idx = index
                    # seq_start_lot = row['lot_num']

        return sequences

    def split_sequences(self):
        """
        Input: filled lot dataframe (lot_num, ae, lot_num_gap)
        Output: list of np arrays that store sequences with 2 or more sequential lots
        """

        sequences = []
        seq_start_idx = 0
        # seq_start_lot = self.filled_df['lot_num'].iloc(0)

        for index, row in self.filled_df.iloc[1:, :].iterrows():
            if index == self.filled_df.shape[0] - 1:
                if row['lot_num_gap'] == 1:
                    sequences.append(np.array(self.filled_df['ae'].iloc[seq_start_idx:index+1]).reshape(-1, 1))

            elif row['lot_num_gap'] > 1:
                if index - seq_start_idx > 1:
                    sequences.append(np.array(self.filled_df['ae'].iloc[seq_start_idx:index]).reshape(-1, 1))

                seq_start_idx = index
                # seq_start_lot = row['lot_num']

            else:
                continue

        return sequences

    def add_metadata(self):
        self.metadata['n_obs_lots'] = self.raw_data.shape[0]
        self.metadata['n_lots_in_seqs'] = sum([len(s) for s in self.sequences])
        self.metadata['n_lots_in_seqs_single'] = sum([len(s) for s in self.sequences_single])
        self.metadata['first_lot_num'] = self.raw_data.loc[0, 'lot_num']
        self.metadata['last_lot_num'] = self.raw_data.loc[self.raw_data.shape[0]-1, 'lot_num']
        self.metadata['n_seqs'] = len(self.sequences)
        self.metadata['n_seqs_single'] = len(self.sequences_single)
        self.metadata['mean_seq_len'] = self.metadata['n_lots_in_seqs']/self.metadata['n_seqs']
        self.metadata['mean_seq_single_len'] = self.metadata['n_lots_in_seqs_single']/self.metadata['n_seqs_single']
        self.metadata['med_obs_gap'] = self.raw_data['lot_num_gap'].median()
        self.metadata['max_obs_gap'] = self.raw_data['lot_num_gap'].max()
