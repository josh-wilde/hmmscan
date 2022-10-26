library(tidyverse)

# Paths
validation_dirpath <- file.path(read_lines('shared-path.txt'), 'validation')
sim_fits_fpath <- file.path(validation_dirpath, 'simulation', 'fits', 's2c1.csv')
sim_inputs_fpath <- file.path(validation_dirpath, 'simulation', 'full_simulation_input.csv')
true_paths_fpath <- file.path(validation_dirpath, 'simulation', 'state_paths', 's2c1.csv')

# Pull in the fits and reference info
fits <- read_csv(sim_fits_fpath)
sim_inputs <- read_csv(sim_inputs_fpath)

true_params <- sim_inputs %>% 
  select(eid, n_states_true=n_states, n_mix_comps_true=n_mix_comps) %>% 
  mutate(is_single_true = n_states_true == 1)

##############
# Calculate the HMMScan accuracy for each eid
##############
bic_results <- fits %>% 
  select(sequence_name, sample_id, bic, n_states, n_mix_comps) %>% 
  separate(sequence_name, into = c('eid', 'sample'), sep = '_sample') %>% 
  select(eid, sample_id, n_states, n_mix_comps, bic) %>% 
  group_by(eid, sample_id) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>% 
  ungroup() %>% 
  mutate(eid = as.numeric(str_remove(eid, 'eid_')))

hmm_accuracy <- bic_results %>% 
  mutate(is_single = n_states == 1) %>% 
  left_join(true_params) %>% 
  mutate(hmm_acc = as.integer(is_single == is_single_true)) %>% 
  group_by(eid) %>% 
  summarise(hmm_acc = mean(hmm_acc))

# Plot the HMMScan accuracy for each eid
# Save as 490x370 SVG
hmm_accuracy %>% 
  left_join(sim_inputs) %>% 
  ggplot(aes(x = n_lots_in_seqs, y = hmm_acc, color = factor(ovl))) + 
  geom_line(size=0.4) + 
  geom_point(size=0.75) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(factor(input_s0_stat))) + 
  labs(x = 'Sequence Length (number of lots)', 
       y = 'HMMScan Detection Accuracy',
       color = 'Overlapping\nCoefficient',
       #title = 'Generating HMM Structure: 2 states, 1 component',
       #subtitle = 'Rows: High-AE State Mean Sojourn Time; Columns: Low-AE State Stationary Prob.'
  ) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(input_s0_stat)) + 
  theme_gray(base_size = 8) + 
  coord_cartesian(ylim = c(0,1))

##############
# Calculate the path accuracy for each eid
##############

true_paths <- read_csv(true_paths_fpath) %>% 
  select(-`...1`) %>% 
  select(eid, sample_id, everything()) %>% 
  pivot_longer(cols = -c(eid, sample_id), names_to = 'lot_id', values_to = 'true_state') %>% 
  filter(!is.na(true_state)) %>% 
  mutate(eid = as.numeric(str_remove(eid, 'eid_')),
         lot_id = as.numeric(lot_id))

state_preds <- fits %>% 
  select(sequence_name, sample_id, bic, n_states, n_mix_comps, matches('^[0-9]{1,3}$')) %>% 
  separate(sequence_name, into = c('eid', 'sample'), sep = '_sample') %>% 
  select(eid, sample_id, n_states, n_mix_comps, bic, matches('^[0-9]{1,3}$')) %>% 
  group_by(eid, sample_id) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>% 
  ungroup() %>% 
  select(eid, sample_id, n_states, n_mix_comps, matches('^[0-9]{1,3}$')) %>%
  mutate(eid = as.numeric(str_remove(eid, 'eid_'))) %>% 
  pivot_longer(cols = matches('^[0-9]{1,3}$'), names_to = 'lot_id', values_to = 'state_pred') %>% 
  filter(!is.na(state_pred)) %>% 
  mutate(lot_id = as.numeric(lot_id)) %>% 
  left_join(true_params %>% select(-is_single_true)) %>% 
  left_join(true_paths) %>% 
  mutate(state_pred = state_pred + 1)

rm(true_paths)

state_path_summary_by_eid_and_sample <- state_preds %>% 
  mutate(is_state_pred_correct = true_state == state_pred) %>% 
  group_by(eid, sample_id) %>% 
  summarise(is_bic_state_correct = first(n_states) == first(n_states_true),
            is_bic_structure_correct = (first(n_states) == first(n_states_true)) & (first(n_mix_comps) == first(n_mix_comps_true)),
            n = n(),
            n_1 = sum(true_state == 1),
            n_2 = sum(true_state == 2),
            n_correct = sum(is_state_pred_correct),
            n_correct_1 = sum(is_state_pred_correct & (true_state == 1)),
            n_correct_2 = sum(is_state_pred_correct & (true_state == 2)),
            ) %>% 
  ungroup() %>% 
  mutate(frac_true_s2 = n_2 / n,
         path_acc = n_correct / n,
         path_acc_s1true = n_correct_1 / n_1,
         path_acc_s2true = n_correct_2 / n_2,
         path_balacc = ifelse((n_1 > 0) & (n_2 > 0), (path_acc_s1true + path_acc_s2true) / 2, NA))

state_path_summary_by_eid <- state_path_summary_by_eid_and_sample %>% 
  group_by(eid) %>% 
  summarise(frac_state_correct = mean(is_bic_state_correct),
            frac_structure_correct = mean(is_bic_structure_correct),
            mean_frac_true_s2 = mean(frac_true_s2),
            mean_path_acc = mean(path_acc),
            mean_path_acc_s1true = mean(path_acc_s1true, na.rm = T),
            mean_path_acc_s2true = mean(path_acc_s2true, na.rm = T),
            mean_path_balacc = mean(path_balacc, na.rm = T))

# Balanced accuracy plot
# Save as 490x370 SVG
state_path_summary_by_eid %>% 
  left_join(sim_inputs) %>% 
  ggplot(aes(x = n_lots_in_seqs, y = mean_path_balacc, color = factor(ovl))) + 
  geom_line(size=0.4) + 
  geom_point(size=0.75) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(factor(input_s0_stat))) + 
  labs(x = 'Sequence Length (number of lots)', 
       y = 'HMMScan Mean State Prediction Accuracy',
       color = 'Overlapping\nCoefficient',
       #title = 'Generating HMM Structure: 2 states, 1 component',
       #subtitle = 'Rows: High-AE State Mean Sojourn Time; Columns: Low-AE State Stationary Prob.'
  ) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(input_s0_stat)) + 
  theme_gray(base_size = 8) + 
  coord_cartesian(ylim = c(0.5,1))

# Low AE state accuracy
state_path_summary_by_eid %>% 
  left_join(sim_inputs) %>% 
  ggplot(aes(x = n_lots_in_seqs, y = mean_path_acc_s1true, color = factor(ovl))) + 
  geom_line(size=0.4) + 
  geom_point(size=0.75) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(factor(input_s0_stat))) + 
  labs(x = 'Sequence Length (number of lots)', 
       y = 'Mean Accuracy on Low-AE Hidden State',
       color = 'Overlapping\nCoefficient',
       #title = 'Generating HMM Structure: 2 states, 1 component',
       #subtitle = 'Rows: High=-AE State Mean Sojourn Time; Columns: Low-AE State Stationary Prob.'
  ) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(input_s0_stat)) + 
  theme_gray(base_size = 8) + 
  coord_cartesian(ylim = c(0.5,1))

# High AE state accuracy
state_path_summary_by_eid %>% 
  left_join(sim_inputs) %>% 
  ggplot(aes(x = n_lots_in_seqs, y = mean_path_acc_s2true, color = factor(ovl))) + 
  geom_line(size=0.4) + 
  geom_point(size=0.75) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(factor(input_s0_stat))) + 
  labs(x = 'Sequence Length (number of lots)', 
       y = 'Mean Accuracy on High-AE Hidden State',
       color = 'Overlapping\nCoefficient',
       #title = 'Generating HMM Structure: 2 states, 1 component',
       #subtitle = 'Rows: High-AE State Mean Sojourn Time; Columns: Low-AE State Stationary Prob.'
  ) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(input_s0_stat)) + 
  theme_gray(base_size = 8) + 
  coord_cartesian(ylim = c(0.5,1))

# Not balanced accuracy
# Balanced accuracy plot
state_path_summary_by_eid %>% 
  left_join(sim_inputs) %>% 
  ggplot(aes(x = n_lots_in_seqs, y = mean_path_acc, color = factor(ovl))) + 
  geom_line(size=0.4) + 
  geom_point(size=0.75) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(factor(input_s0_stat))) + 
  labs(x = 'Sequence Length (number of lots)', 
       y = 'Mean Hidden State Accuracy',
       color = 'Overlapping\nCoefficient',
       #title = 'Generating HMM Structure: 2 states, 1 component',
       #subtitle = 'Rows: High-AE State Mean Sojourn Time; Columns: Low-AE State Stationary Prob.'
  ) + 
  facet_grid(rows = vars(mean_sojourn), cols = vars(input_s0_stat)) + 
  theme_gray(base_size = 8) + 
  coord_cartesian(ylim = c(0.5,1))
