library(tidyverse)

# Paths
validation_dirpath <- file.path(read_lines('shared-path.txt'), 'validation')
sim_fits_fpath <- file.path(validation_dirpath, 'simulation', 'fits', 's1.csv')
sim_inputs_fpath <- file.path(validation_dirpath, 'simulation', 'full_simulation_input.csv')

# Pull in the fits
fits <- read_csv(sim_fits_fpath)
sim_inputs <- read_csv(sim_inputs_fpath)

true_params <- sim_inputs %>% 
  select(eid, n_states_true=n_states, n_mix_comps_true=n_mix_comps) %>% 
  mutate(is_single_true = n_states_true == 1)

# Calculate the HMMScan accuracy for each eid
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
# Save as 490x250 SVG
hmm_accuracy %>% 
  left_join(sim_inputs) %>% 
  ggplot(aes(x = n_lots_in_seqs, y = hmm_acc, color = factor(ovl_intra))) + 
  geom_line(size=0.8) + 
  geom_point(size=1) + 
  labs(x = 'Sequence Length (number of lots)', 
       y = 'HMMScan Accuracy',
       color = 'Overlapping\nCoefficient\n(intra-state)',
       #title = 'Generating HMM Structure: 1 state, 2 components'
  ) + 
  theme_gray(base_size = 10) + 
  coord_cartesian(ylim = c(0,1))
