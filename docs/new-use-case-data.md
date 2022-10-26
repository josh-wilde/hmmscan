# User-Provided Input Data

This document provides instructions for using HMMScan with a user-provided use case sequence of per lot AE rates.

1. Follow all steps in the Installation section of [the readme for this repo](../README.md).
2. Create a file in `ae-project/data/use_case` that has the same structure as `ae-project/data/use_case/dfa_by_date_ex_iqr_outliers.csv`. This file should have the following columns:
   1. `lot_num`: This column contains a numeric lot number, ordered from first to last.
   2. `lot_group`: This can be used to designate that a single input sequence should be considered as multiple sequences that restart when the lot group increments. This is not used in the paper results. If the input data should be considered as one contiguous sequence, then label each row with 1 in this column. Otherwise, you can increment this column by 1 to separate the multiple sequences. Make sure to start with 1.  
   3. `prefix`: Leave this column blank. Included for legacy purposes.
   4. `num`: Same as `lot_num` column. Included for legacy purposes.
   5. `[ae_type]`: This column contains the per lot AE rates and can be named as desired. The title of the column should match the `ae_type` argument that is used during the HMMScan procedure. Multiple AE type columns can be included for the same lot sequence to test different ways of counting AEs. Each column should have a unique name. 
