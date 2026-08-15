# Figure B - Clarke Error Grid (CEG) on out-of-fold FPG predictions.
#
# Source data:
#   ceg_oof_zero_shot.csv   (columns: license, y_true, y_pred_oof)
#
# Output:
#   figureB_clarke_error_grid.png
#   printed zone summary table (Points, %, A+B cumulative %)
#
# Requires: install.packages(c("ega", "ggplot2", "dplyr", "tidyr"))

library(ega)
library(ggplot2)
library(dplyr)
library(tidyr)


data_df <- read.csv("/Users/hannahkim/논문/data_requested/ceg_oof_zero_shot.csv")
reference_bg <- data_df$y_true
test_bg      <- data_df$y_pred_oof

ZOOM_LIMIT <- 260
SMALLER_POINT_SIZE <- 0.5

# --- 1. Classify points into Clarke zones ---
clarke_zones <- getClarkeZones(reference_bg, test_bg)
clarke_zones_factor <- factor(clarke_zones, levels = c("A", "B", "C", "D", "E"))
clarke_zone_counts  <- table(clarke_zones_factor)
clarke_total_points <- length(clarke_zones)
clarke_perc <- round(clarke_zone_counts / clarke_total_points * 100, 1)

perc_A <- clarke_perc["A"]
perc_B <- clarke_perc["B"]
perc_C <- clarke_perc["C"]
perc_D <- clarke_perc["D"]
perc_E <- clarke_perc["E"]

label_text <- paste(
  "Zones:",
  paste0("A = ", perc_A, "%"),
  paste0("B = ", perc_B, "%"),
  paste0("C = ", perc_C, "%"),
  paste0("D = ", perc_D, "%"),
  paste0("E = ", perc_E, "%"),
  sep = "\n"
)

# --- 2. Build the CEG plot with identity line, zoom, and zone-% annotation ---
ceg_plot <- plotClarkeGrid(
  reference_bg,
  test_bg,
  title = "Clarke Error Grid",
  xlab = "Reference fasting plasma glucose (mg/dL)",
  ylab = "Predicted fasting plasma glucose (mg/dL)",
  pointsize = SMALLER_POINT_SIZE
)

ceg_plot_final <- ceg_plot +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  coord_cartesian(xlim = c(0, ZOOM_LIMIT), ylim = c(0, ZOOM_LIMIT)) +
  annotate("text",
           x = ZOOM_LIMIT * 0.15,
           y = ZOOM_LIMIT * 0.25,
           label = label_text,
           hjust = 1,
           vjust = 1,
           size = 3,
           fontface = "bold",
           lineheight = 1.2,
           colour = "black")

print(ceg_plot_final)
ggsave("figureB_clarke_error_grid.png", plot = ceg_plot_final, width = 6, height = 6, dpi = 300)

# --- 3. Zone summary table (Points, %, A+B cumulative %) ---
zone_levels <- c("A", "B", "C", "D", "E")
zones_factor <- factor(clarke_zones, levels = zone_levels)

zone_df <- as.data.frame(table(zones_factor)) %>%
  rename(Zone = zones_factor, Points = Freq) %>%
  mutate(
    Percentage = round(Points / clarke_total_points * 100, 1),
    AB_Cumulative = case_when(Zone %in% c("A", "B") ~ Percentage, TRUE ~ 0)
  )

ab_sum <- sum(zone_df$AB_Cumulative)

final_table <- zone_df %>%
  mutate(
    AB_Cumulative = case_when(Zone %in% c("A", "B") ~ ab_sum, TRUE ~ NaN),
    `%` = Percentage,
    `AB Sum %` = AB_Cumulative
  ) %>%
  select(Zone, Points, `%`, `AB Sum %`)

total_row <- data.frame(
  Zone = "Total",
  Points = sum(final_table$Points),
  `%` = 100.0,
  `AB Sum %` = NaN
) %>%
  mutate_all(as.character)

final_table_display <- final_table %>%
  mutate_all(as.character) %>%
  bind_rows(total_row)

print(final_table_display)