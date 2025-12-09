####################################### Multidimensional distributions ############################

rm(list=ls())
############### Function #########
library(statip)
library(matrixcalc)
hellinger_dist <- function(p, q,num=1) {
  res<-hellinger(p,q,1)
  
  return(res)
}
hellinger_df <- function(original_df, synthetic_df) {
  if (!all(names(original_df) == names(synthetic_df))) {
    stop("Both data frames must have the same columns in the same order.")
  }
  
  distances <- numeric(length = ncol(original_df))
  
  for (i in seq_along(original_df)) {
    x <- original_df[[i]]
    y <- synthetic_df[[i]]
    
    
    
    distances[i] <- tryCatch(
      {
        hellinger_dist(x, y, 1)   # main attempt
      },
      error = function(e) {
        message("Error in column ", i, ": ", e$message)
        return(NULL)               # fallback value
      })
  }
  
  # Return average distance across variables
  mean(distances)
}

pcd <- function(real_df, synth_df, method = "pearson") {
  # Ensure same structure
  if (!all(names(real_df) == names(synth_df))) {
    stop("Both data frames must have the same column names and order.")
  }
  
  # Keep only numeric columns (correlation only makes sense for numeric variables)
  real_num <- real_df[sapply(real_df, is.numeric)]
  synth_num <- synth_df[sapply(synth_df, is.numeric)]
  
  # Compute correlation matrices
  corr_real <- cor(real_num, method = method, use = "pairwise.complete.obs")
  corr_synth <- cor(synth_num, method = method, use = "pairwise.complete.obs")
  
  # Compute Frobenius norm of their difference
  diff_mat <- corr_real - corr_synth
  pcd_value <- frobenius.norm(diff_mat)
  
  #pcd_value<-1 - FDist2(corr_real,corr_synth)/2
  
  return(pcd_value)
}

#matrixcalc ---> frobenius.norm(X)

# Propensity Mean Squared Error (pMSE) metric
propensity_score <- function(real_df, synth_df) {
  # Ensure same columns
  if (!all(names(real_df) == names(synth_df))) {
    stop("Both data frames must have the same column names and order.")
  }
  
  # Add indicator: 0 = real, 1 = synthetic
  real_df$label <- 0
  synth_df$label <- 1
  
  # Combine
  combined <- rbind(real_df, synth_df)
  combined$label <- as.factor(combined$label)
  
  # Simple preprocessing: remove non-numeric columns or convert factors
  for (col in names(combined)) {
    if (is.character(combined[[col]])) {
      combined[[col]] <- as.factor(combined[[col]])
    }
  }
  
  # Fit logistic regression model
  model <- glm(label ~ ., data = combined, family = binomial)
  
  # Predict propensity scores
  p_hat <- predict(model, type = "response")
  
  # Compute pMSE
  N <- length(p_hat)
  pMSE <- mean((p_hat - 0.5)^2)
  
  return(pMSE)
}
############ Original set ###########
original<-read.csv('C:/Users/30697/Desktop/PREGSAFE/Data/Complete/Dataset_filtered.csv',check.names=FALSE)




########### Synthetic sets #############

options<-c('Original','borderline_smote','enn_smote','kmeans_smote','smote','svm_smote','tomek_smote','GAN')
sets<-list()
for(option in options){
  sets[[option]]<-list()
}
for(fold in 1:10){
  for(option in options){
    if(option=='Original'){
      df<-read.csv(paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_',fold,'.csv'),check.names=FALSE)
      df_wanted<-df[,c("MA","Conception ART 01","Wt pre","BMI pre","Wgain","GA days")]
      sets[[option]][[fold]]<-df_wanted
    }else if(option=='GAN'){
      df<-read.csv(paste0('C:/Users/30697/Desktop/PREGSAFE/Data/GAN_AUTO/GAN_',fold,'.csv'),check.names=FALSE)
      cases<-df[df$GDM01==1,]
      all_cases<-nrow(cases)
      df_wanted<-cases[424:all_cases,c("MA","Conception ART 01","Wt pre","BMI pre","Wgain","GA days")]
      
      
      sets[[option]][[fold]]<-df_wanted
    }else{
    df<-read.csv(paste0('C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_',fold,'_',option,'.csv'),check.names=FALSE)
    cases<-df[df$GDM01==1,]
    all_cases<-nrow(cases)
    df_wanted<-cases[424:all_cases,c("MA","Conception ART 01","Wt pre","BMI pre","Wgain","GA days")]
    sets[[option]][[fold]]<-df_wanted
    }
  
  
  }
}



##### 1. Univariate fidelitty or Attribute Fidelity Hellinger distance
hellinger_res<-c()
SDG_arr<-c()
fold_arr<-c()
#comparison_all<-c()
comparation_options<-c('borderline_smote','enn_smote','kmeans_smote','smote','svm_smote','tomek_smote','GAN')
for(fold in 1:10){
  df2<-sets[['Original']][[fold]]
  for(option in comparation_options){
    df1<-sets[[option]][[fold]]
    res<-round(hellinger_df(df1,df2),3)
    hellinger_res<-c(hellinger_res,res)
    SDG_arr<-c(SDG_arr,option)
    fold_arr<-c(fold_arr,fold)
  }
 

}
df_hell<-data.frame(SDG=SDG_arr,Fold=fold_arr,Hellinger=hellinger_res)


#### 2. Pairwise correlation difference (PCD)
pcd_res<-c()
for(fold in 1:10){
  df2<-sets[['Original']][[fold]]
  for(option in comparation_options){
    df1<-sets[[option]][[fold]]
    res<-round(pcd(df2,df1,"pearson"),3)
    pcd_res<-c(pcd_res,res)
    
  }
  
  
}
df_hell$PCD<-pcd_res




####### 3. Propensity Score
prop_res<-c()
for(fold in 1:10){
  
  df<-read.csv(paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_',fold,'.csv'),check.names=FALSE)
  df<-df[df$GDM01==1,]
  df2<-df[,c("MA","Conception ART 01","Wt pre","BMI pre","Wgain","GA days")]
  for(option in comparation_options){
    df1<-sets[[option]][[fold]]
    option_j<-c()
    nrows_df2<-nrow(df2)
    for(j in 1:1000){
      df1_iter<-df1[sample(1:nrows_df2,nrows_df2),]
      res<-round(propensity_score(df2,df1_iter),4)
      option_j<-c(option_j,res)
    }
    prop_res<-c(prop_res,mean(res))
    
  }
  
  
}
df_hell$Propensity<-prop_res
df_final<-df_hell
save(df_final,file='C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/Intrisinc_quality_res.Rda')



library(dplyr)
df<-df_final %>%
  group_by(SDG) %>%
  summarize(
    median_MeanHellingerDist=round(quantile(Hellinger,probs=0.5),3),
    Q1_MeanHellingerDist=round(quantile(Hellinger,probs=0.25),3),
    Q3_MeanHellingerDist=round(quantile(Hellinger,probs=0.75),3),
    
    median_PCD=round(quantile(PCD,probs=0.5),3),
    Q1_PCD=round(quantile(PCD,probs=0.25),3),
    Q3_PCD=round(quantile(PCD,probs=0.75),3),
    
    median_Prop=round(quantile(Propensity,probs=0.5),3),
    Q1_Prop=round(quantile(Propensity,probs=0.25),3),
    Q3_Prop=round(quantile(Propensity,probs=0.75),3)
    
    
  )





















