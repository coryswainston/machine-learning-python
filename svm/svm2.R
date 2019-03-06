library(e1071)

letters <- read.csv("letters.csv")

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(letters)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

lTest <- letters[testRows,]
lTrain <- letters[-testRows,]

bestAccuracy <- 0
bestC <- 0
bestGamma <- 0

for (cexp in seq(1,5,2)) {
  cValue <- 2^cexp
  for (gexp in seq(-5,3,2)) {
    gamma <- 2^gexp
    model <- svm(letter~., data = lTrain, kernel = "radial", gamma = gamma, cost = cValue)
    prediction <- predict(model, lTest[,-1])
    
    # Produce a confusion matrix
    agreement <- prediction == lTest$letter
    accuracy <- length(agreement[agreement==TRUE])/length(agreement)
    
    print(accuracy)
    
    if (accuracy > bestAccuracy) {
      bestC <- cValue
      bestGamma <- gamma
      bestAccuracy <- accuracy
    }
  }
}


print(bestAccuracy)
print(bestC)
print(bestGamma)