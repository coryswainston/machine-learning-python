library(e1071)

vowels <- read.csv("vowel.csv")

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(vowels)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

vTest <- vowels[testRows,]
vTrain <- vowels[-testRows,]

bestAccuracy <- 0
bestC <- 0
bestGamma <- 0
bestCexp <- 0
bestGexp <- 0

for (cexp in seq(-5,15,2)) {
  cValue <- 2^cexp
  for (gexp in seq(-15,3,2)) {
    gamma <- 2^gexp
    model <- svm(Class~., data = vTrain, kernel = "radial", gamma = gamma, cost = cValue)
    prediction <- predict(model, vTest[,-13])
    
    # Produce a confusion matrix
    confusionMatrix <- table(pred = prediction, true = vTest$Class)
    agreement <- prediction == vTest$Class
    
    accuracy <- length(agreement[agreement==TRUE])/length(agreement)
    
    if (accuracy > bestAccuracy) {
      bestCexp <- cexp
      bestGexp <- gexp
      bestGamma <- gamma
      bestC <- cValue
      bestAccuracy <- accuracy
    }
  }
}

for (cexp in seq(bestCexp - 1,bestCexp + 1, 0.25)) {
  cValue <- 2^cexp
  for (gexp in seq(bestGexp - 1,bestGexp + 1, 0.25)) {
    gamma <- 2^gexp
    model <- svm(Class~., data = vTrain, kernel = "radial", gamma = gamma, cost = cValue)
    prediction <- predict(model, vTest[,-13])
    
    # Produce a confusion matrix
    confusionMatrix <- table(pred = prediction, true = vTest$Class)
    agreement <- prediction == vTest$Class
    
    accuracy <- length(agreement[agreement==TRUE])/length(agreement)
    
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