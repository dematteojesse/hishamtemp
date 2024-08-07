dt <- read.table("Z:/Student Folders/Hisham_Temmar/tcFNNPaper/Results/variance_online/combined_df_only_successfultrials.csv", sep=",", header=TRUE)

nest <- aov(BitRate ~ Day / factor(Algorithm), data=dt)
summary(nest)

library(lsr)
etaSquared(nest)

