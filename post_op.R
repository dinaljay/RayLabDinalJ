#Clear console 
cat("\014");

library(ggplot2)
library(tidyverse)
library(xlsx)
library(readxl)

setwd("/home/functionalspinelab/Desktop/JNS_paper")
rm(list = ls())
colors = c("red2", "forestgreen", "blue3")

postop <- read_xlsx("/home/functionalspinelab/Downloads/Post_op_data.xlsx", col_names = TRUE, 
                   sheet = "Accuracy")

post_plot <- ggplot(data=postop, aes(x=Feature_set, y = Performance, fill = Metric))+
  geom_bar(stat = "identity", color = "black", width = 0.85, position = position_dodge2(reverse = TRUE, preserve = "single"))+
  guides(fill = guide_legend(reverse = TRUE))+
  geom_errorbar(aes(ymin = Lower, ymax = Upper), stat = "identity", colour = "black", size = 0.6, width = 0.85, position = position_dodge2(preserve = "single", reverse = TRUE))

mytheme3 <- theme(legend.text = element_text(family = "Helvetica", size = rel(1.5), color="black"), 
                  axis.title = element_text(family = "Helvetica", size = rel(1.5), color="black"), 
                  axis.text = element_text(family = "Helvetica", size = rel(1.5), color="black"), 
                  axis.line = element_line(size = 1,colour = "black"), 
                  axis.ticks = element_line(colour="grey",size = rel(1.4)),
                  panel.grid.major = element_line(colour="grey",size = rel(1.0)), 
                  panel.grid.minor = element_line(colour = "grey", size = rel(0.5)), 
                  panel.background = element_rect(fill = "white"), 
                  legend.key = element_rect(fill = "white"),
                  legend.position = ("bottom"),
                  legend.title = element_blank(), 
                  plot.title = element_text(face = "bold", size = rel(1.7),family = "Helvetica", hjust = 0.5))

plot_fin <- print(post_plot + mytheme3+ ggtitle("Accuracy")
                  + scale_x_discrete(limits=c("DTI", "DBSI"))
                  + labs(y="Performance (%)", x = "", colour = "Feature_set")
                  + scale_fill_manual(values = colors)
                  + scale_y_continuous(expand = c(0,0), limits = c(0,100)))

ggsave("Accuracy.tiff", plot_fin, width = 7, height = 5.5, units = 'in', dpi=300)

