########################## Code for experiments ########################

########### Libraries  #################
library(caret)
library(MLmetrics)
library(dplyr)
library(tidyr)
library(mltools)
library(recipes)

########### Functions ################
Model_training<-function(train,target_col,algorithms=c('RF','SVM','KNN','NB','XGB'),number_iterations=50,optimization_metric='balAcc'){
  
  
  set.seed(123)
  number_iterations<-10
  input_y <- target_col
  input_y <- factor(input_y, levels = c(0, 1), labels = c("neg", "pos"))
  
 
  input_x <- as.matrix(train)

  ctrl <- trainControl(
    method = "cv",
    number = 4,
    search = "random",
    summaryFunction = summary_fnc,
    classProbs = TRUE,
    verboseIter = TRUE,
    allowParallel = TRUE
  )
  
  # Control for final model training
  tune_control <- trainControl(
    method = "cv",
    number = 4,
    classProbs = TRUE,
    summaryFunction = summary_fnc,
    verboseIter = FALSE,
    allowParallel = TRUE
  )
  
  
  ################# XGB model ####################
  xgb_model_clinical<-train(
    x = input_x,
    y = input_y,          
    method = "xgbTree",
    trControl = ctrl,
    tuneLength =number_iterations,
    metric=optimization_metric
  )
  
  (final_grid<- expand.grid(
    nrounds = xgb_model_clinical$bestTune$nrounds,
    eta = xgb_model_clinical$bestTune$eta,
    max_depth = xgb_model_clinical$bestTune$max_depth,
    gamma = xgb_model_clinical$bestTune$gamma,
    colsample_bytree = xgb_model_clinical$bestTune$colsample_bytree,
    min_child_weight = xgb_model_clinical$bestTune$min_child_weight,
    subsample = xgb_model_clinical$bestTune$subsample
  ))
  
  (xgb_model<- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = TRUE
  ))
  
  
  ################################## RF model #############################
  rf_model_clinical <- train(
    x = input_x,
    y = input_y,          
    method = "rf",
    trControl = ctrl,
    tuneLength =number_iterations,
    metric=optimization_metric
  )
  (final_grid<- expand.grid(
    mtry=rf_model_clinical$bestTune$mtry
  ))
  (rf_model <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = final_grid,
    method = "rf",
    verbose = TRUE
  ))
  
  
  ############################# KNN ##################
  k_values <- c(3, 5, 7, 9, 11, 13, 15)
  
  # Train the KNN model using the specified grid
  knn_model_clinical <- train(
    x = input_x,
    y = input_y,          
    method = "knn",
    trControl = ctrl,
    tuneGrid = expand.grid(k = k_values),
    metric = optimization_metric
  )
  
  # Extract the best k
  (final_grid <- expand.grid(
    k = knn_model_clinical$bestTune$k
  ))
  
  # Refit the model using the best k
  (knn_model <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = final_grid,
    method = "knn"
  ))
  
  
  
  #################################### SVM #############################
  # sigma_values <- c(0.001, 0.01, 0.05, 0.1, 0.5)   # example sigma values
  # C_values <- c(0.25, 0.5, 1, 2, 4)                # example cost values
  # 
  # tune_grid <- expand.grid(
  #   sigma = sigma_values,
  #   C = C_values
  # )
  # 
  # # Train SVM model using the manual grid
  # svm_model_clinical <- train(
  #   x = input_x,
  #   y = input_y,          
  #   method = "svmRadial",
  #   trControl = ctrl,
  #   tuneGrid = tune_grid,
  #   metric = optimization_metric
  # )
  # 
  # # Extract the best parameters
  # (final_grid <- expand.grid(
  #   sigma = svm_model_clinical$bestTune$sigma,
  #   C = svm_model_clinical$bestTune$C
  # ))
  # 
  # Refit the model with the best parameters
  (svm_model <- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    method = "svmRadial",
    metric = optimization_metric,
    probs = TRUE
  ))
  
  ################################### NB ###############################
  nb_model_clinical <- train(
    x = input_x,
    y = input_y,
    method = "naive_bayes",      # use the naive Bayes method
    trControl = ctrl,
    tuneLength = number_iterations,
    metric = optimization_metric
  )
  
  # Extract the best tuning parameters
  (final_grid <- expand.grid(
    laplace = nb_model_clinical$bestTune$laplace,
    usekernel = nb_model_clinical$bestTune$usekernel,
    adjust = nb_model_clinical$bestTune$adjust
  ))
  
  # Refit the model using the best parameters
  (nb_model<- caret::train(
    x = input_x,
    y = input_y,
    trControl = tune_control,
    tuneGrid = final_grid,
    method = "naive_bayes",
    metric = optimization_metric,
    probs = TRUE
  ))
  
  
  return(list(XGB_model=xgb_model,RF_model=rf_model,KNN_model=knn_model,SVM_model=svm_model,NB_model=nb_model))
  
}

summary_fnc <- function(data, lev = NULL, model = NULL) {
  # Ensure prediction and true labels are factors
  obs <- factor(data$obs, levels = lev)
  pred <- factor(data$pred, levels = lev)
  
  cm <- caret::confusionMatrix(pred, obs)
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  
  bal_acc <- mean(c(sens, spec), na.rm = TRUE)
  acc <- MLmetrics::Accuracy(pred, obs)
  f1 <- FBeta_Score(obs, pred, positive = "pos", beta = 1)
  f2 <- FBeta_Score(obs, pred, positive = "pos", beta = 2)
  f05 <- FBeta_Score(obs, pred, positive = "pos", beta = 0.5)
  
  # AUC Calculation (assumes the column name for positive class is 'pos')
  if ("pos" %in% colnames(data)) {
    auc <- pROC::roc(response = obs, predictor = data[["pos"]], levels = rev(lev), direction = ">")$auc
  } else {
    auc <- NA
  }
  
  out <- c(
    balAcc = bal_acc,
    F1Score = f1,
    Spec = spec,
    Sens = sens,
    Accuracy = acc,
    F2Score = f2,
    F05Score = f05,
    AUC = auc
  )
  return(out)
}




######### Main code #################
options<-c('Original','borderline_smote','enn_smote','kmeans_smote','smote','svm_smote','tomek_smote')#'GAN')
for(fold in 1:10){
  for(option in options){
  print(option)
  tryCatch({
  if(option=='Original'){
    path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_',fold,'.csv')
    train_set<-read.csv(path_train,check.names=FALSE)
    colnames(train_set)<-make.names(colnames(train_set))
    
  }else if(option=='GAN'){
    #path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/CTGAN_',fold,'.csv') 
    #train_set<-read.csv(path_train,check.names=FALSE)
    #colnames(train_set)<-make.names(colnames(train_set))
  }else{
  path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_',fold,'_',option,'.csv')
  train_set<-read.csv(path_train,check.names=FALSE)
  colnames(train_set)<-make.names(colnames(train_set))
  
  }
  test_set<-read.csv(paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Test_set_',fold,'.csv'),check.names = FALSE)
  colnames(test_set)<-make.names(colnames(test_set))
  train_set<-train_set[,colnames(test_set)]
  
  
  train_set$Conception.ART.01<-as.factor(ifelse(train_set$Conception.ART.01>0.5,1,0))
  ################ Convert to factor columns
  for(col in colnames(train_set)){
    if(length(unique(train_set[,col]))<=8){
      train_set[,col]<-as.factor(train_set[,col])
      test_set[,col]<-factor(test_set[,col],levels=levels(train_set[,col]))

    }
  }
  
  ############### One hot encoding
  colnames(train_set)<-make.names(colnames(train_set))
  colnames(test_set)<-make.names(colnames(test_set))
  
  train<-train_set[!colnames(train_set) %in% c('GDM01')]
  for(col in colnames(train)){
    if(is.factor(train[,col])){
      train[,col]<-as.numeric(train[,col])
    }
    
  }
  
  test<-test_set[!colnames(test_set) %in% c('GDM01')]
  for(col in colnames(test)){
    if(is.factor(test[,col])){
      test[,col]<-as.numeric(test[,col])
    }
  }
  ############# Train models
  res_ML <- Model_training(train,train_set[,'GDM01'],number_iterations = 10)
  
  save(res_ML,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Results/',option,'_models_',fold,'.Rda'))
  
  save(test_set,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Results/',option,'_test_set_',fold,'.Rda'))
  }, error = function(e) {
    message(paste("⚠️ Error in option:", option, "— skipping."))
    message("Error details:", e$message)
  })
    
  }
}

options<-c('GAN')
for(fold in 1:10){
  for(option in options){
    print(option)
    tryCatch({
      if(option=='GAN'){
        path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/GAN_AUTO/GAN_',fold,'.csv') 
        train_set<-read.csv(path_train,check.names=FALSE)
        colnames(train_set)<-make.names(colnames(train_set))
      }else{
        path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_',fold,'_',option,'.csv')
        train_set<-read.csv(path_train,check.names=FALSE)
        colnames(train_set)<-make.names(colnames(train_set))
        
      }
      test_set<-read.csv(paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Test_set_',fold,'.csv'),check.names = FALSE)
      colnames(test_set)<-make.names(colnames(test_set))
      train_set<-train_set[,colnames(test_set)]
      
      
      train_set$Conception.ART.01<-as.factor(ifelse(train_set$Conception.ART.01>0.5,1,0))
      ################ Convert to factor columns
      for(col in colnames(train_set)){
        if(length(unique(train_set[,col]))<=8){
          train_set[,col]<-as.factor(train_set[,col])
          test_set[,col]<-factor(test_set[,col],levels=levels(train_set[,col]))
          
        }
      }
      
      ############### One hot encoding
      colnames(train_set)<-make.names(colnames(train_set))
      colnames(test_set)<-make.names(colnames(test_set))
      
      train<-train_set[!colnames(train_set) %in% c('GDM01')]
      for(col in colnames(train)){
        if(is.factor(train[,col])){
          train[,col]<-as.numeric(train[,col])
        }
        
      }
      
      test<-test_set[!colnames(test_set) %in% c('GDM01')]
      for(col in colnames(test)){
        if(is.factor(test[,col])){
          test[,col]<-as.numeric(test[,col])
        }
      }
      ############# Train models
      res_ML <- Model_training(train,train_set[,'GDM01'],number_iterations = 10)
      
      save(res_ML,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Results/',option,'_models_',fold,'.Rda'))
      
      save(test_set,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Results/',option,'_test_set_',fold,'.Rda'))
    }, error = function(e) {
      message(paste("⚠️ Error in option:", option, "— skipping."))
      message("Error details:", e$message)
    })
    
  }
}
