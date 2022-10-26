library(tidyverse)
library(glue)

# Paths
results_dirpath <- file.path(read_lines('shared-path.txt'), 'results')
best_init_fpath <- file.path(results_dirpath, 'scans', 'use_case', 'best_initializations', 'by_date_ex_iqr.csv')
predictions_dirpath <- file.path(results_dirpath, 'state_prediction')

# Pull in data
best_inits <- read_csv(best_init_fpath)

# Extract the state params for the best model for each sequence_name that we want
state_params <- best_inits %>% 
  filter(dir == 'use_case',
         str_detect(sequence_name, 'ex_iqr_outliers'),
         ae_type == 'serious_std'
         ) %>%
  group_by(sequence_name, ae_type) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>%
  ungroup() %>% 
  select(dir, sequence_name, ae_type, matches('s[0-3]_[0-8]_.*')) %>% 
  pivot_longer(cols = matches('s[0-3]_[0-8]_.*')) %>% 
  replace_na(list(value=0)) %>% 
  separate(col = 'name', into = c('state', 'comp', 'param_type')) %>% 
  mutate(state = as.integer(str_remove(state, 's')),
         comp = as.integer(comp)) %>% 
  pivot_wider(names_from = param_type, values_from = value) %>% 
  filter(wt > 0)

# Reorder the states by mean AE
state_ordering <- state_params %>%  
  group_by(sequence_name, ae_type, state) %>% 
  summarise(mean = sum(param * wt)) %>% 
  ungroup() %>% 
  arrange(sequence_name, ae_type, mean) %>% 
  group_by(sequence_name, ae_type) %>% 
  mutate(new_state = row_number()) %>% 
  select(state, new_state)

##############
# Visualize the state-specific distributions
##############

# State-specific distributions
state_dists <- state_params %>% 
  left_join(state_ordering) %>% 
  select(-state) %>% 
  rename(state=new_state) %>% 
  crossing(tibble(x = seq(0,100000))) %>% 
  mutate(prob_binom_wt = pmap_dbl(list(p = param, wt = wt, x = x), function(p, wt, x) {wt * dbinom(x, 100000, p)})) %>% 
  group_by(sequence_name, ae_type, state, x) %>% 
  summarise(prob_binom = sum(prob_binom_wt)) %>% 
  ungroup()

# Figure saved as SVG 490x200 if not including Dose Form C
state_dists %>% 
  mutate(dose_form_label = ifelse(str_detect(sequence_name, 'dfa'), 'Dose Form A', 
                                  ifelse(str_detect(sequence_name, 'dfb'), 'Dose Form B', 'Dose Form C'))) %>% 
  filter(((dose_form_label == 'Dose Form A') & (x < 100)) 
         | ((dose_form_label == 'Dose Form B') & (x < 50)) 
         | ((dose_form_label == 'Dose Form C') & (x < 30)) ) %>% 
  filter(dose_form_label != 'Dose Form C') %>% 
  ggplot(aes(x = x, y = prob_binom, fill = factor(state))) + 
  geom_area(alpha = 0.5, position = 'identity') + 
  facet_wrap(.~dose_form_label, ncol = 2, scales = 'free') +
  theme_bw(base_size = 9) +  
  labs(x = 'Reported AE rate per 100k doses',
       y = 'Probability Mass',
       fill = 'Hidden\nState',
       title = NULL # 'State-Specific Mixture Distributions' # 'Dose Form B'
  )

##############
# View the parameters for the states and transition probabilities
##############

# State dist means
state_params %>%  
  group_by(sequence_name, ae_type, state) %>% 
  summarise(mean = sum(param * wt)) %>% 
  ungroup() %>% 
  arrange(sequence_name, ae_type, mean) %>% 
  group_by(sequence_name, ae_type) %>% 
  mutate(new_state = row_number(),
         scaled_mean = mean * 100000) %>% 
  select(-mean)

# Stationary probabilities
stat_probs <- best_inits %>% 
  filter(dir == 'use_case',
         str_detect(sequence_name, 'ex_iqr_outliers'),
         ae_type == 'serious_std'
  ) %>%
  group_by(sequence_name, ae_type) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>%
  ungroup() %>% 
  select(dir, sequence_name, ae_type, ends_with('_stat')) %>% 
  pivot_longer(cols = ends_with('_stat'), names_to = 'state', values_to = 'stat_prob') %>%
  mutate(state = as.integer(str_remove(str_remove(state, '_stat'), 's'))) %>% 
  filter(!is.na(stat_prob)) %>% 
  left_join(state_ordering) %>% 
  arrange(dir, sequence_name, new_state)

stat_probs

trans_probs <- best_inits %>% 
  filter(dir == 'use_case',
         str_detect(sequence_name, 'ex_iqr_outliers'),
         ae_type == 'serious_std'
  ) %>%
  group_by(sequence_name, ae_type) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>%
  ungroup() %>% 
  select(dir, sequence_name, ae_type, matches('s[0-3]_s[0-3]')) %>% 
  pivot_longer(cols = matches('s[0-3]_s[0-3]'), names_to = 'from_to', values_to = 'prob') %>% 
  mutate(from_to = str_remove_all(from_to, 's')) %>% 
  separate(col = from_to, into = c('from', 'to'), sep = '_', convert = TRUE) %>% 
  left_join(state_ordering %>% rename(from=state, from_new=new_state)) %>% 
  left_join(state_ordering %>% rename(to=state, to_new=new_state)) %>% 
  filter(!is.na(prob)) %>% 
  select(-c(from, to, dir, ae_type)) %>% 
  arrange(from_new, to_new)

trans_probs %>% 
  filter(str_detect(sequence_name, 'dfa'))

trans_probs %>% 
  filter(str_detect(sequence_name, 'dfb'))

##############
# Visualize the state predictions
##############

# Extract the state params for the best model for each sequence_name that we want
predictions_state_params <- best_inits %>% 
  filter(dir == 'use_case',
         str_detect(sequence_name, 'ex_iqr_outliers|ex_iqr_expedited'),
         ae_type %in% c('serious_std', 'exp_no_admin_std')
  ) %>%
  group_by(sequence_name, ae_type) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>%
  ungroup() %>% 
  select(dir, sequence_name, ae_type, matches('s[0-3]_[0-8]_.*')) %>% 
  pivot_longer(cols = matches('s[0-3]_[0-8]_.*')) %>% 
  replace_na(list(value=0)) %>% 
  separate(col = 'name', into = c('state', 'comp', 'param_type')) %>% 
  mutate(state = as.integer(str_remove(state, 's')),
         comp = as.integer(comp)) %>% 
  pivot_wider(names_from = param_type, values_from = value) %>% 
  filter(wt > 0)

# Reorder the states by mean AE
predictions_state_ordering <- predictions_state_params %>%  
  group_by(sequence_name, ae_type, state) %>% 
  summarise(mean = sum(param * wt)) %>% 
  ungroup() %>% 
  arrange(sequence_name, ae_type, mean) %>% 
  group_by(sequence_name, ae_type) %>% 
  mutate(new_state = row_number()) %>% 
  select(state, new_state)

# Pull in all of the prediction files
predictions <- best_inits %>% 
  filter(dir == 'use_case',
         str_detect(sequence_name, 'ex_iqr_outliers|ex_iqr_expedited'),
         ae_type %in% c('serious_std', 'exp_no_admin_std')
  ) %>% 
  group_by(sequence_name, ae_type) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>% 
  select(dir, sequence_name, ae_type, n_states, n_mix_comps) %>% 
  mutate(pred_fname = file.path(predictions_dirpath, dir, glue('{sequence_name}_{ae_type}_s{n_states}c{n_mix_comps}.csv'))) %>% 
  mutate(predictions = map(pred_fname, ~{read_csv(.) %>% mutate(lot_id = row_number())})) %>% 
  unnest(predictions) %>% 
  mutate(ae_rate = ifelse(ae_type == 'exp_no_admin_std', exp_no_admin_std, serious_std)) %>% 
  select(-c(pred_fname, serious_std, exp_no_admin_std)) %>% 
  ungroup() %>% 
  left_join(predictions_state_ordering %>% rename(viterbi=state, viterbi_ordered=new_state))

# Saved as SVG 490x375
predictions %>% 
  mutate(dose_form_label = case_when(
    sequence_name == 'dfa_by_date_ex_iqr_outliers' ~ '(a) Dose Form A',
    sequence_name == 'dfb_by_date_ex_iqr_outliers' ~ '(b) Dose Form B',
    sequence_name == 'dfc_by_date_ex_iqr_outliers' ~ '(c) Dose Form C',
    sequence_name == 'dfa_by_date_ex_iqr_expedited' ~ '(d) Dose Form A\n(expedited reports)',
    sequence_name == 'dfb_by_date_ex_iqr_expedited' ~ '(e) Dose Form B\n(expedited reports)',
    sequence_name == 'dfc_by_date_ex_iqr_expedited' ~ '(f) Dose Form C\n(expedited reports)',
  )) %>% 
  ggplot(aes(x = lot_id, y = ae_rate, color = factor(viterbi_ordered))) +
  facet_wrap(.~dose_form_label, nrow = 2, ncol = 3, scales = 'free') + 
  geom_point(size = 0.8) + 
  theme_bw(base_size = 9) + 
  theme(legend.position = 'bottom') + 
  labs(color = 'Predicted State',
       x = 'Lot ID, ordered by packaging date',
       y = 'AE rate, per 100k doses')
         
