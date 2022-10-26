library(tidyverse)

# Paths
validation_dirpath <- file.path(read_lines('shared-path.txt'), 'validation')
dfa_fits_fpath <- file.path(validation_dirpath, 'ci', 'fits', 'dfa_by_date_ex_iqr_outliers_serious_std_s3_c2.csv')
dfb_fits_fpath <- file.path(validation_dirpath, 'ci', 'fits', 'dfb_by_date_ex_iqr_outliers_serious_std_s3_c2.csv')
dfc_fits_fpath <- file.path(validation_dirpath, 'ci', 'fits', 'dfc_by_date_ex_iqr_outliers_serious_std_s2_c2.csv')

# Pull data
sample_fits <- bind_rows(
  read_csv(dfa_fits_fpath),
  read_csv(dfb_fits_fpath),
  read_csv(dfc_fits_fpath),
) %>% 
  select(-`...1`) %>% 
  select(-matches('^[0-9]{1,3}$'))

# Reshape to get the parameters for each state for each sample
sample_params <- sample_fits %>% 
  select(sequence_name, sample_id, n_states, n_mix_comps, matches('.*_(wt|param)')) %>% 
  pivot_longer(cols = matches('.*_(wt|param)')) %>% 
  replace_na(list(value = 0)) %>% 
  mutate(state = as.integer(str_sub(name, 2, 2)),
         component = as.integer(str_sub(name, 4, 4)), 
         value_type = str_sub(name, 6, -1)) %>% 
  select(-name) %>% 
  pivot_wider(names_from = value_type, values_from = value) %>% 
  group_by(sequence_name, sample_id, n_states, n_mix_comps, state) %>% 
  mutate(sum_wt = sum(wt)) %>% 
  filter(near(sum_wt, 1)) %>% 
  ungroup()
  
# Check to make sure that there are still the right number of states and mix components for each sample after filtering
# Should equal 100 for each of the three dose forms
sample_params %>% 
  mutate(sequence_name = str_remove(sequence_name, '_sample.*')) %>% 
  count(sequence_name, sample_id, n_states, n_mix_comps, state) %>% 
  rename(n_mix_comps_check=n) %>% 
  count(sequence_name, sample_id, n_states, n_mix_comps, n_mix_comps_check) %>% 
  rename(n_states_check=n) %>% 
  count(sequence_name, (n_states == n_states_check) & (n_mix_comps == n_mix_comps_check))

# This provides the 90% CI
sample_params %>% 
  mutate(sequence_name = str_remove(sequence_name, '_sample.*')) %>% 
  group_by(sequence_name, sample_id, n_states, n_mix_comps, state) %>% 
  summarise(mean = sum(param * wt) * 100000) %>% 
  ungroup() %>% 
  arrange(sequence_name, sample_id, n_states, n_mix_comps, mean) %>% 
  group_by(sequence_name, sample_id, n_states, n_mix_comps) %>% 
  mutate(ord_state = row_number()) %>% 
  ungroup() %>%
  group_by(sequence_name, n_states, n_mix_comps, ord_state) %>% 
  summarise(min_mean = min(mean),
            q025_mean = quantile(mean, 0.025),
            q05_mean = quantile(mean, 0.05),
            q25_mean = quantile(mean, 0.25),
            q50_mean = quantile(mean, 0.50),
            q75_mean = quantile(mean, 0.75),
            q95_mean = quantile(mean, 0.95),
            q975_mean = quantile(mean, 0.975),
            max_mean = max(mean),
            ) %>% 
  ungroup() %>% 
  select(sequence_name, n_states, q05_mean, q95_mean)
