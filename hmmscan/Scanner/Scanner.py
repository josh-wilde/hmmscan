# This needs to take in:
# Name

# fit_p options (T/F list)
# fixed_n (list of options or None)
# fixed_p (list of options or None)
# Gap thresholds (list)
# States and components (2 lists)

from typing import Dict, List, Union
import os
import pandas as pd

from hmmscan.DataLoader import AELoader, ManualLoader
from hmmscan.utils.scanner_utils import report_model_dl_data, convert_output_dict_to_df
from hmmscan.utils.load_data_utils import write_data_csv, get_results_path
from hmmscan.utils.model_initialization_utils import initialize_model


class Scanner:
    def __init__(
            self,
            states=None,
            components=None,
            fixed_ns=None,
            fixed_ps=None,
            directory=None,
            sequence_names=None,
            ae_types=None,
            gaps=None,
            gap_qs=None,
            init_types=None,
            rseeds=None,
            rand_init_ubs=None,
            init_file_name=None,
            init_file_row=None,
            ae_sequence=None
    ):
        # Save max number of folds, states, and components
        self.max_s = max(states)
        self.max_c = max(components)

        # Set some default list parameters
        if init_types is None:
            init_types = ['random']
        if rseeds is None:
            rseeds = [1]
        if rand_init_ubs is None:
            rand_init_ubs = [0.01]

        # Set the models
        # You can either fully specify the model initialization, or you can use quasi-random initializations
        if ('fully_specified' in init_types) and (len(init_types) == 1):
            self.models = self.initialize_fully_specified_model(
                states,
                components,
                fixed_ns,
                fixed_ps,
                sequence_names,
                ae_types,
                gaps,
                gap_qs,
                init_file_name,
                init_file_row
            )
        elif ('fully_specified' in init_types) and (len(init_types) > 1):
            raise ValueError(f"Can either do a fully specified initialization, or random/deterministic. Not both.")
        else:
            self.models = self.initialize_models(
                states, components, fixed_ns, fixed_ps, init_types, rseeds, rand_init_ubs
            )

        # Set the data loaders
        # This gives the option to specify the sequence of AE directly, or load it from a file
        self.dls = self.set_dataloaders(ae_sequence, directory, sequence_names, ae_types, gaps, gap_qs)

        # Total number of models to fit
        self.total_models = len(self.models) * len(self.dls)

    @staticmethod
    def initialize_fully_specified_model(
            states,
            components,
            fixed_ns,
            fixed_ps,
            sequence_names,
            ae_types,
            gaps,
            gap_qs,
            init_file_name,
            init_file_row
    ):
        # Get the series with the initialization information
        init_df: pd.DataFrame = pd.read_csv(os.path.join(get_results_path(), init_file_name))
        init_series: pd.Series = init_df.iloc[init_file_row]
        if (
            init_series['sequence_name'] in sequence_names
            and init_series['ae_type'] in ae_types
            and init_series['n_states'] in states
            and init_series['n_mix_comps'] in components
            and (init_series['gap'] in gaps or init_series['gap_q'] in gap_qs)
            and (init_series['fixed_p'] in fixed_ps or init_series['fixed_n'] in fixed_ns)
        ):
            return {
                sequence_names[0]: initialize_model(
                    init_series,
                    'fully_specified',
                    init_file_row=init_file_row,
                    init_file_name=init_file_name
                )
            }
        else:
            raise ValueError("Input parameters describing model do not match with the fully specified init series.")

    @staticmethod
    def initialize_models(
        states, components, fixed_ns, fixed_ps, init_types, rseeds, rand_init_ubs
    ):
        # These models are used for all of the sequences
        models = []
        if 'random' in init_types:
            init_list = [('random', rseed, ub) for rseed in rseeds for ub in rand_init_ubs]
        else:
            init_list = []
        if 'deterministic' in init_types:
            init_list = [('deterministic', -1, -1)] + init_list

        for init in init_list:
            for s in states:
                for c in components:
                    for p in fixed_ps:
                        models.append(
                            initialize_model(
                                {
                                    'n_states': s,
                                    'n_mix_comps': c,
                                    'n': None,
                                    'p': p,
                                    'fit_p': False,
                                    'init_type': init[0],
                                    'rseed': init[1],
                                    'rand_init_ub': init[2]
                                },
                                'minimal'
                            )
                        )
                    for n in fixed_ns:
                        models.append(
                            initialize_model(
                                {
                                    'n_states': s,
                                    'n_mix_comps': c,
                                    'n': n,
                                    'p': None,
                                    'fit_p': True,
                                    'init_type': init[0],
                                    'rseed': init[1],
                                    'rand_init_ub': init[2]
                                },
                                'minimal'
                            )
                        )

        return {'all': models}

    @staticmethod
    def set_dataloaders(ae_sequence, directory, sequence_names, ae_types, gaps, gap_qs):
        if ae_sequence is not None:
            if len(sequence_names) > 1 or len(ae_types) > 1:
                raise ValueError(
                    f"Provided a single sequence directly",
                    f", but there are {len(sequence_names)} sequence_names and {len(ae_types)} ae types"
                )
            dls = {
                sequence_names[0]: [ManualLoader(ae_sequence, directory, sequence_names[0], ae_types[0])]
            }
        else:
            dls = {seq_name: [] for seq_name in sequence_names}

            # Absolute gaps or gap quantiles
            if gap_qs is not None:
                gaps = gap_qs
                use_gap_q = True
            else:
                use_gap_q = False

            for seq_name in sequence_names:
                for ae_type in ae_types:
                    for gap in gaps:
                        new_loader = AELoader(directory, seq_name, ae_type, gap, use_gap_q)
                        dls[seq_name].append(new_loader)
                        write_data_csv(new_loader.filled_df, 'gap_filled/' + ae_type + '/gap_' + str(gap), seq_name)

        return dls

    def run(self, predict=False, **fit_kwargs):
        # Initialize output dict
        output: Dict = {}

        for seq_name in self.dls:
            print(f"-Sequence name: {seq_name}")
            dls = self.dls[seq_name]
            models = self.models.get('all', []) + self.models.get(seq_name, [])
            for dl_counter, dl in enumerate(dls):
                print(f"--Data loader: {dl_counter+1}/{len(dls)}")
                for model_counter, model in enumerate(models):
                    print(f"---Model counter: {model_counter+1}/{len(models)}")
                    # Fit the model
                    model.fit(dl, **fit_kwargs)
                    if predict:
                        state_predictions: Union[List[int], None] = model.predict_states(dl.sequences)
                    else:
                        state_predictions = None

                    # Save the output of the fitted model and dataloader
                    output: Dict = report_model_dl_data(model, dl, state_predictions, output)

        # Convert the dictionary to a dataframe
        return convert_output_dict_to_df(output, max_s=self.max_s, max_c=self.max_c)
