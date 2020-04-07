source(file.path("R","global.R"))

## load dataset ----
news_file <- file.path("input","News_Category_Dataset_v2.json")
news <- readLines(con = news_file) %>%
  paste0(., collapse = ",") %>%
  paste0("[",.,"]") %>%
  fromJSON(.) %>%
  dplyr::mutate(., label_cat = as.integer(as.factor(category))-1)

## merge categories ----
merged_category <- list(
  "ARTS" = c("ARTS","ARTS & CULTURE","CULTURE & ARTS"),
  "FOOD & DRINK" = c("FOOD & DRINK","TASTE"),
  "PARENTING" = c("PARENTING","PARENTS"),
  "STYLE" = c("STYLE","STYLE & BEAUTY"),
  "WORLDPOST" = c("THE WORLDPOST","WORLD NEWS","WORLDPOST")
) 
news <- dplyr::mutate(news, mcategory = category)
for(i in names(merged_category)) {
  news <- dplyr::mutate(
    news, 
    mcategory = ifelse(mcategory %in% merged_category[[i]], i, mcategory),
    label_mcat = as.integer(as.factor(mcategory))-1
  )
}

## prepare text for bow ----
news$txt <- paste(news$headline, news$short_description) %>%
  tolower() %>%
  # remove everything that is not a number or letter
  stringr::str_replace_all(.,"[^a-zA-Z\\s]", " ") %>%
  removeNumbers() %>%
  stripWhitespace() %>%
  enc2utf8(.) # encode as utf-8

## bow ----
bow <- quanteda::dfm(news$txt, verbose = TRUE)
countspace <- numeric(nrow(news))
for(i in 1:nrow(news))
  countspace[i] <- length(gregexpr(" ", news$txt[i])[[1]])
bow <- cbind(countspace, bow)
colnames(bow)[1] <- "</s>"

bow <- bow[, colSums(bow) > 999]
rbow <- rowSums(bow)

idx_rm <- which(rbow == 1)
bow <- bow[-idx_rm,]
file <- file.path("output","bow.rds")
saveRDS(bow, file = file)

file <- file.path("output","labels.rds")
idx_keep <- row.names(bow) %>%
  str_replace(.,"text","") %>% as.numeric(.)
news <- news[idx_keep,]
labels <- news %>% dplyr::select(., category,mcategory,label_cat,label_mcat)
saveRDS(labels, file = file)
