library(tidyverse)
library(cowplot)
library(viridis)
library(scales)

# Paths
results_dirpath <- file.path(read_lines('shared-path.txt'), 'results')
best_init_fpath <- file.path(results_dirpath, 'scans', 'use_case', 'best_initializations', 'by_date_ex_iqr.csv')

# Pull in data
best_inits <- read_csv(best_init_fpath)

# Create the plot for the correct sequence name and ae_type
plot_df <- best_inits %>% 
  filter(str_detect(sequence_name, 'ex_iqr_outliers'), 
         ae_type == 'serious_std')

# Dose Form A
dfa_bic_plot <- plot_df %>% 
  filter(str_detect(sequence_name, 'dfa')) %>% 
  ggplot(aes(x = factor(n_states), y = factor(n_mix_comps))) +
  geom_tile(aes(fill = bic), color = 'white', width = 0.9, height = 0.9) +
  geom_text(aes(label = comma(bic, accuracy = 1)),
            size = 7/.pt, 
            fontface = 'bold', color = 'white') +
  labs(x = 'Hidden states',
       y = 'Mixture components per state',
       fill = 'BIC',
       title = 'Dose Form A') + 
  theme_cowplot(font_size = 10) + 
  theme(strip.background = element_blank(), 
        legend.text = element_text(size = 7),
        legend.title = element_text(size = 7),
        legend.position = 'bottom',
        legend.key.width = unit(0.5, 'cm')) +
  guides(fill = guide_legend(override.aes = list(size = 1), nrow = 4)) +
  scale_fill_viridis(colors = plasma(5), 
                     direction = 1, 
                     limits = c(NA, 4500), 
                     name = 'BIC', 
                     labels = comma)

dfb_bic_plot <- plot_df %>% 
  filter(str_detect(sequence_name, 'dfb')) %>% 
  ggplot(aes(x = factor(n_states), y = factor(n_mix_comps))) +
  geom_tile(aes(fill = bic), color = 'white', width = 0.9, height = 0.9) +
  geom_text(aes(label = comma(bic, accuracy = 1)),
            size = 7/.pt, 
            fontface = 'bold', color = 'white') +
  labs(x = 'Hidden states',
       y = NULL, # 'Mixture components per state',
       fill = 'BIC',
       title = 'Dose Form B') + 
  theme_cowplot(font_size = 10) + 
  theme(strip.background = element_blank(), 
        legend.position = 'bottom',
        legend.text = element_text(size = 7),
        legend.title = element_text(size = 7),
        legend.key.width = unit(0.5, 'cm')) +
  guides(fill = guide_legend(override.aes = list(size = 1), nrow = 4)) +
  scale_fill_viridis(colors = plasma(5), 
                     direction = 1, 
                     limits = c(NA, 2400), 
                     name = 'BIC', 
                     labels = comma)

dfc_bic_plot <- plot_df %>% 
  filter(str_detect(sequence_name, 'dfc')) %>% 
  ggplot(aes(x = factor(n_states), y = factor(n_mix_comps))) +
  geom_tile(aes(fill = bic), color = 'white', width = 0.9, height = 0.9) +
  geom_text(aes(label = comma(bic, accuracy = 1)),
            size = 7/.pt, 
            fontface = 'bold', color = 'white') +
  labs(x = 'Hidden states',
       y = NULL, #'Mixture components per state',
       fill = 'BIC',
       title = 'Dose Form C') + 
  theme_cowplot(font_size = 10) + 
  theme(strip.background = element_blank(), 
        legend.position = 'bottom',
        legend.text = element_text(size = 7),
        legend.title = element_text(size = 7),
        legend.key.width = unit(0.5, 'cm')) +
  guides(fill = guide_legend(override.aes = list(size = 1), nrow = 4)) +
  scale_fill_viridis(colors = plasma(5), 
                     direction = 1, 
                     limits = c(NA, 1000), 
                     name = 'BIC', 
                     labels = comma)

# This is displayed and saved as 490x350 SVG file for paper
plot_grid(dfa_bic_plot, dfb_bic_plot, dfc_bic_plot, nrow = 1, axis = 'tb')

