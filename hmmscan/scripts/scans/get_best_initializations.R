library(tidyverse)

# Paths
results_basedirpath <- file.path(read_lines('shared-path.txt'), 'results')

input_subdir_stub <- 'scans/use_case/random_initializations'
output_subdir_stub <- 'scans/use_case/best_initializations'
fname <- args[1] # 'by_date_ex_iqr.csv'
input_fpath <- file.path(results_basedirpath, input_subdir_stub, fname)
output_fpath <- file.path(results_basedirpath, output_subdir_stub, fname)

# Make directory if necessary
dir.create(dirname(output_fpath))

# Pull in the dataframe and pick out only the best initialization for each state/comp
read_csv(input_fpath) %>% 
  select(-starts_with('...')) %>% 
  select(-fpaths) %>% 
  group_by(dir, sequence_name, ae_type, n_states, n_mix_comps) %>% 
  slice_min(order_by = bic, n = 1, with_ties = FALSE) %>% 
  write_csv(output_fpath)

