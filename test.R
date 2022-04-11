#Clear console 
cat("\014");

library(ggplot2)
library(tidyverse)
library(xlsx)

setwd("/home/functionalspinelab/Desktop/JNS_paper")
rm(list = ls())
colors = c("red", "blue")
  
preop <- read_xlsx("/home/functionalspinelab/Downloads/Test.xlsx", col_names = TRUE, 
                   sheet = "Preop")

postop <- read_xlsx("/home/functionalspinelab/Downloads/Test.xlsx", col_names = TRUE, 
                    sheet = "Postop")

pre_plot <- ggplot(data=preop, aes(x=Comparison, y = Performance, fill = Input))+
  geom_bar(stat = "identity", color = "black", position = position_dodge2(reverse = TRUE))+
  guides(fill = guide_legend(reverse = TRUE))+
  scale_y_continuous(expand = c(0,0),
                     limits = c(0,100))

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

pre_plot_fin <- print(pre_plot + mytheme3+ ggtitle("Pre-Op Classifier Performance")
      + scale_x_discrete(limits=c("Accuracy","Precision", "Recall", "F1 Score", "AUC × 100"))
      + labs(y="Performance (%)", x = "", colour = "Input")
      + scale_fill_manual(values = colors))

ggsave("Figure_4.tiff", pre_plot_fin, width = 7, height = 5.5, units = 'in', dpi=300)

post_plot <- ggplot(data=postop, aes(x=Comparison, y = Performance, fill = Input))+
  geom_bar(stat = "identity", color = "black", position = position_dodge2(reverse=TRUE))+
  guides(fill = guide_legend(reverse = TRUE))+
  scale_y_continuous(expand = c(0,0),
                     limits = c(0,100))

post_plot_fin <- print(post_plot + mytheme3+ ggtitle("Evaluation of prognostication model")
      + scale_x_discrete(limits=c("Accuracy","Precision", "Recall", "F1 Score", "AUC × 100"))
      + labs(y="Performance (%)", x = "", colour = "Input")
      + scale_fill_manual(values = colors))

ggsave("Figure_6.tiff", post_plot_fin, width = 7, height = 5.5, units = 'in', dpi=300)
