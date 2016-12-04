library(softImpute)
require(softImpute)

# Load the data
original_data <- read.csv(file = "4-5-6-7-8-9-10-visits.csv", header = TRUE, sep = ",")

# Remove Odd visits
D <- subset(original_data, original_data[,2] == 'V04' | original_data[,2] == 'V06'| original_data[,2] == 'V08'| original_data[,2] == 'V10' )

# Save the new data to a csv
#write.csv(D, file = "~/Dropbox/Graduate School/Fall-2016/Advanced-ML/project/data/new_data/even_visits_unto_v10.csv")

# Remove the first 2 columns i.e patno and eventid
newD <- D[, -which(names(D) %in% c("PATNO","EVENT_ID"))]

# convert data frame to a matrix
newD <- data.matrix(newD)

# Do matrix completion
rnk <- 100
lmda <- 10

fit1=softImpute(newD,rank=30,lambda=30)
ans1 <- complete(newD, fit1)
# take the original 2 col from D and add them to ans1
ans <- cbind(D[,1:2], ans1)

# join rows to make a wide matrix

# Get the label matrix from v12

