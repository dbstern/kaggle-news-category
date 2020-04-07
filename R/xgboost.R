source(file.path("R","global.R"))

set.seed(1)
train <- sample(c(T,F), nrow(news), c(.1,.9), replace = TRUE)

bow <- readRDS(file.path("output","bow.rds"))
news <- readRDS(file.path("output","labels.rds"))

fit_xgboost <- function(data, train, file_out) {
  fit <- list()
  params <- list(
    lambda = 0.01,
    objective = "multi:softmax",
    metrics = "merror",
    num_class = length(unique(getinfo(data,"label"))),
    early_stopping_rounds = 10
  )

  dtrain <- xgboost::slice(data, idxset = which(train))
  fit$xgbcv <- xgboost::xgb.cv(params = params, data = dtrain,
                               nround = 10000, nfold = 2, verbose = FALSE)

  nround <- which.min(fit$xgbcv$evaluation_log$test_merror_mean)
  fit$xgb <- xgboost::xgb.train(params = params, data = dtrain,
                                nround = nround, verbose = FALSE)
  rm(dtrain); gc()

  dtest <- xgboost::slice(data, idxset = which(!train))
  fit$test <- as.numeric(predict(fit$xgb, dtest, type = "response"))
  fit$acc <- sum(fit$test==xgboost::getinfo(dtest, 'label'))/sum(!train)
  fit$accaret <- confusionMatrix(
    factor(fit$test),
    factor(xgboost::getinfo(dtest, 'label')),
    mode = "everything"
    )
  saveRDS(fit, file = file_out)
  rm(dtest); gc()

  return(fit)
}

x <- xgb.DMatrix(bow)
xgboost::setinfo(object = x, 'label', news$label_cat)
fit_file <- file.path("output","bowxgb_cat.rds")
fit_cat <- fit_xgboost(data = x, train = train, file_out = fit_file)

x <- xgb.DMatrix(bow)
xgboost::setinfo(object = x, 'label', news$label_mcat)
fit_file <- file.path("output","bowxgb_mcat.rds")
fit_mcat <- fit_xgboost(data = x, train = train, file_out = fit_file)
