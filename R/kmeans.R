source(file.path("R","global.R"))

news <- readRDS(file.path("output","labels.rds"))

bow <- readRDS(file.path("output","bow.rds"))
sbow <- scale(as.matrix(bow))

file <- file.path("output","bow_kmeans.rds")
x <- list()

x$res_cat <- biganalytics::bigkmeans(x = sbow, iter.max=100, centers = 41, nstart = 10)
x$ari_cat <- mclust::adjustedRandIndex(x$res_cat$cluster,news$category)
x$accaret_cat <- confusionMatrix(
  factor(x$res_cat$cluster-1), reference=factor(news$label_cat), mode = "everything")

x$res_mcat <- biganalytics::bigkmeans(x = sbow, iter.max=100, centers = 34, nstart = 10)
x$ari_mcat <- mclust::adjustedRandIndex(x$res_mcat$cluster,news$mcategory)
x$accaret_mcat <- confusionMatrix(
  factor(x$res_mcat$cluster-1), reference=factor(news$label_mcat), mode = "everything")

saveRDS(x, file=file)
