library(tidyverse)
library(glue)
library(cowplot)
library(scales)

# Paths
validation_dirpath <- file.path(read_lines('shared-path.txt'), 'validation')
use_case_fits_fpath <- file.path(validation_dirpath, 'use_case', 'fits', 'all_fits.csv')

##############
# Extract the best model structures for the dose forms
##############
fits <- read_csv(use_case_fits_fpath)

results <- fits %>% 
  select(sequence_name, sample_id, gen_n_states, gen_n_mix_comps, n_states, n_mix_comps, eval_n_states, eval_n_mix_comps, bic) %>% 
  group_by(sequence_name, sample_id) %>% 
  slice_min(bic, n = 1, with_ties = FALSE) %>% 
  filter((gen_n_states != eval_n_states) | (gen_n_mix_comps != eval_n_mix_comps)) %>% 
  mutate(eval_is_lowest = ifelse((n_states == eval_n_states) & (n_mix_comps == eval_n_mix_comps), TRUE, FALSE)) %>% 
  mutate(dose_form_label = ifelse(str_detect(sequence_name, 'dfa'), '(a) Dose Form A',
                                  ifelse(str_detect(sequence_name, 'dfb'), '(b) Dose Form B', '(c) Dose Form C'))) %>% 
  group_by(dose_form_label, gen_n_states, gen_n_mix_comps) %>% 
  summarise(n_samples = n(), 
            n_misclassified = sum(eval_is_lowest),
            frac_misclassified = n_misclassified/n_samples)

# Output as 490x200 SVG
results %>% 
  ggplot(aes(x = as.factor(gen_n_states), y = as.factor(gen_n_mix_comps))) +
  geom_tile(aes(fill = frac_misclassified), color = 'white', width = 0.9, height = 0.9) + 
  geom_text(aes(label = comma(frac_misclassified, accuracy = 0.001)),
            size = 2, fontface = 'bold', color = 'white') + 
  facet_wrap(.~dose_form_label, nrow = 1) +
  labs(x = 'Hidden states (sampling model)',
       y = 'Mixture components per state\n(sampling model)',
       title = NULL, 
       subtitle = NULL) + 
  theme_cowplot(font_size = 8) +
  theme(strip.background = element_blank()) + 
  theme(legend.position = 'none')
