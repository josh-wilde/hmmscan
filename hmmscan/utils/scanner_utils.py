from typing import Dict, Union, List
import pandas as pd
import numpy as np
from hmmscan.BinomialModel.wrappers import BinModel
from hmmscan.DataLoader import AELoader, ManualLoader


def report_model_dl_data(
        model: BinModel,
        dl: Union[AELoader, ManualLoader],
        state_predictions: Union[List[int], None],
        output: Dict
):
    # Add metadata from data loader
    for dl_metaitem in dl.metadata:
        if dl_metaitem in output:
            output[dl_metaitem].append(dl.metadata[dl_metaitem])
        else:
            output[dl_metaitem] = [dl.metadata[dl_metaitem]]

    # Add metadata from the model
    for mdl_metaitem in model.metadata:
        if mdl_metaitem in output:
            output[mdl_metaitem].append(model.metadata[mdl_metaitem])
        else:
            output[mdl_metaitem] = [model.metadata[mdl_metaitem]]

    # Instantiate lists if necessary
    if 'log_prob' not in output:
        output['log_prob'] = []
        output['bic'] = []
        output['mmdl'] = []
        output['comp_params'] = []
        output['stat_dist'] = []
        output['trans_mat'] = []

    # Add model results and state predictions
    output['log_prob'].append(model.get_eval(dl, evaluation='log_prob'))
    output['bic'].append(model.get_eval(dl, evaluation='bic'))
    output['mmdl'].append(model.get_eval(dl, evaluation='mmdl'))
    output['comp_params'].append(model.get_comp_params())
    output['stat_dist'].append(model.get_stat_dist())
    output['trans_mat'].append(model.get_trans_mat())
    if state_predictions is not None:
        for lot_num, state in enumerate(state_predictions):
            output[lot_num] = state

    return output


def convert_output_dict_to_df(d, max_s=4, max_c=9):
    df = pd.DataFrame(d)

    if len(d) == 0:
        return df
    else:
        # Then parse the comp_params column
        comp_params_input = pd.DataFrame(d['comp_params']).rename(columns={0: 'param', 1: 'wt'})
        comp_params_df = parse_comp_params(comp_params_input, max_s, max_c)

        # Parse the stat_dist column
        stat_dist_df = parse_stat_dist(df['stat_dist'], max_s)

        # Parse the trans_mat column
        trans_mat_df = parse_trans_mat(df['trans_mat'], max_s)

        # Get rid of the columns I don't want
        df = df.drop(['comp_params', 'stat_dist', 'trans_mat'], axis=1)
        return pd.concat([df, comp_params_df, stat_dist_df, trans_mat_df], axis=1)


def parse_comp_params(df, max_s=5, max_c=5):
    output = {}
    output = parse_comp_param_col(df, output, 'param', max_s, max_c)
    output = parse_comp_param_col(df, output, 'wt', max_s, max_c)
    return pd.DataFrame(output)


def parse_comp_param_col(df, output, parse_col, max_s, max_c):
    for d in df[parse_col]:
        for s in ['s' + str(n_s) for n_s in range(max_s)]:
            for c in range(max_c):
                col_name = s + '_' + str(c) + '_' + parse_col
                if col_name in output.keys():
                    if d is np.nan:
                        output[col_name].append(np.nan)
                    else:
                        if s in d.keys():
                            if c < len(d[s]):
                                output[col_name].append(d[s][c])
                            else:
                                output[col_name].append(np.nan)
                        else:
                            output[col_name].append(np.nan)
                else:
                    output[col_name] = []
                    if d is np.nan:
                        output[col_name].append(np.nan)
                    else:
                        if s in d.keys():
                            if c < len(d[s]):
                                output[col_name].append(d[s][c])
                            else:
                                output[col_name].append(np.nan)
                        else:
                            output[col_name].append(np.nan)
    return output


def parse_stat_dist(col, max_s=5):
    output = {}
    for ps in col:
        for s in range(max_s):
            col_name = 's' + str(s) + '_stat'
            if col_name in output.keys():
                if s < len(ps):
                    output[col_name].append(ps[s])
                else:
                    output[col_name].append(np.nan)
            else:
                output[col_name] = []
                if s < len(ps):
                    output[col_name].append(ps[s])
                else:
                    output[col_name].append(np.nan)

    return pd.DataFrame(output)


def parse_trans_mat(col, max_s=5):
    output = {}
    for mat in col:
        for s_from in range(max_s):
            for s_to in range(max_s):
                col_name = 's' + str(s_from) + '_' + 's' + str(s_to)
                if col_name in output.keys():
                    if s_from < mat.shape[0] and s_to < mat.shape[1]:
                        output[col_name].append(mat[s_from, s_to])
                    else:
                        output[col_name].append(np.nan)
                else:
                    output[col_name] = []
                    if s_from < mat.shape[0] and s_to < mat.shape[1]:
                        output[col_name].append(mat[s_from, s_to])
                    else:
                        output[col_name].append(np.nan)

    return pd.DataFrame(output)
