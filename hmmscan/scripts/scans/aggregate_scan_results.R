library(tidyverse)

# Paths
results_basedirpath <- file.path(read_lines('shared-path.txt'), 'results')

args <- commandArgs(trailingOnly=TRUE)
input_subdir_stub <- args[1] # 'scans/use_case/random_initializations/by_date_ex_iqr'
input_dirpath <- file.path(results_basedirpath, input_subdir_stub)

# Get all files from the directory
fnames <- dir(input_dirpath)
input_fpaths <- file.path(input_dirpath, fnames)

# Pull into one giant tibble
combined <- tibble(fpaths = input_fpaths) %>% 
  mutate(data = map(fpaths, read_csv)) %>% 
  unnest(data)

# Output
combined %>% 
  write_csv(paste0(input_dirpath, '.csv'))