#Clear console 
cat("\014");

library(ggplot2)
library(tidyverse)
library(xlsx)

setwd("/home/functionalspinelab/Desktop/Paper images(processed)/DBSI/Prognostication")
rm(list = ls())
colors = c("yellow", "green", "blueviolet", "darkcyan", "brown")

evaluator <- read_xlsx("/home/functionalspinelab/Downloads/Test2.xlsx", col_names = TRUE, 
                   sheet = "F1 Score")

pre_plot <- ggplot(data=evaluator, aes(x=Input, y = Performance, fill = Comparison))+
  geom_bar(stat = "identity", color = "black", position = position_dodge2(reverse = TRUE))+
  guides(fill = guide_legend(reverse = TRUE))

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

plot_fin <- print(pre_plot + mytheme3+ ggtitle("F1 Score")
      + scale_x_discrete(limits=c("Clinical + DTI", "Clinical + DBSI"))
      + labs(y="Performance (%)", x = "", colour = "Input")
      + scale_fill_manual(values = colors)
      + scale_y_continuous(expand = c(0,0), limits = c(0,100)))

