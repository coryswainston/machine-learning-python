library(arules)
data(Groceries)

itemFrequencyPlot(Groceries, topN=10, type="absolute", main="Item Frequency")

rules <- apriori(Groceries, parameter = list(supp = 0.005, conf = 0.3))
rules_conf <- sort(rules, by="lift", decreasing = TRUE)
inspect(rules_conf)

