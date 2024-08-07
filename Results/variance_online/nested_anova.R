# dt <- read.table("Z:/Student Folders/Hisham_Temmar/tcFNNPaper/Results/variance_online/combined_df.csv", sep=",", header=TRUE)
dt <- read.table("Z:/Student Folders/Hisham_Temmar/tcFNNPaper/Results/variance_online/metrics_successful_full.csv", 
                 sep=",", header=TRUE)

# nest <- aov(BitRate ~ Day / factor(Decoder), data=dt, subset=Day != "Day2")
nest <- aov(BitRate ~ Day / factor(Run), data=dt)
summary(nest)

library(lsr)
etaSquared(nest)

# Try an ANOVA per day
# Days with multiple decoders
day1 <- subset(dt, Day == "Day1")
day2 <- subset(dt, Day == "Day2")
# Days with only one decoder
day3 <- subset(dt, Day == "Day3")
day4 <- subset(dt, Day == "Day4")

# Run ANOVA per day

## Homogeneity of Variance
boxplot(BitRate~Run, data=day1)
boxplot(BitRate~Run, data=day2)
boxplot(BitRate~Run, data=day3)
boxplot(BitRate~Run, data=day4)
# ONE-WAY ANOVA 
resday1 <- aov(BitRate ~ Run, data = day1)   #aov is a wrapper for lm()
summary(resday1)
resday2 <- aov(BitRate ~ Run, data = day2)   #aov is a wrapper for lm()
summary(resday2)
resday3 <- aov(BitRate ~ Run, data = day3)   #aov is a wrapper for lm()
summary(resday3)
resday4 <- aov(BitRate ~ Run, data = day4)   #aov is a wrapper for lm()
summary(resday4)
