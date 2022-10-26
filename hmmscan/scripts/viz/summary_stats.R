library(tidyverse)

# Paths
data_dirpath <- file.path(read_lines('shared-path.txt'), 'data', 'use_case')
fnames <- list.files(data_dirpath)

# Pull in the raw data
aes <- tibble(fname = fnames) %>% 
  mutate(data = map(fname, ~read_csv(file.path(data_dirpath, .)))) %>% 
  unnest(data)

# Summary statistics
aes %>% 
  mutate(dose_form = str_sub(fname, 1, 3),
         ae_type = ifelse(str_detect(fname, 'expedited'), 'expedited', ifelse(str_detect(fname, 'iqr'), 'serious_iqr', 'outliers_included')),
         aes_std = ifelse(ae_type == 'expedited', exp_no_admin_std, serious_std)) %>% 
  select(dose_form, ae_type, aes_std) %>% 
  group_by(dose_form, ae_type) %>% 
  summarise(min = min(aes_std),
            q25 = quantile(aes_std, 0.25),
            q50 = quantile(aes_std, 0.5),
            q75 = quantile(aes_std, 0.75),
            q95 = quantile(aes_std, 0.95),
            max = max(aes_std),
            mean_std = mean(aes_std),
            n_lots = n(),
            n_aes_standardized = sum(aes_std))

