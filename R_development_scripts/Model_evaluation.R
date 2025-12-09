#################################################### Evaluation code #########################################
rm(list=ls())
######## Libraries ###########
library(MLmetrics)
library(caret)
library(pROC)
library(PRROC)



######## Functions ###########
Evaluation_metrics<-function(dataset,response,res_ML){
  res_all<-NULL
  response<-factor(response,labels=c("neg","pos"))
  
  for(name in names(res_ML)){
    model_name<-strsplit(name,split='_')[[1]][1]
    
    model<-res_ML[[name]]
    
    if(model_name=='XGB'){
    dataset_temp<-dataset[,model$finalModel$feature_names]
    }else if(model_name=='SVM'){
      dataset_temp<-dataset[,predictors(model)]
    }else{
      dataset_temp<-dataset
    }
    predictions<-predict(model,newdata=dataset_temp)
    if(any(is.na(predictions))){
      row_na<-rep(NA,8)
      row<-c(model_name,row_na)
      res_all<-rbind(res_all,row)
      next
    }
    
    cm<- caret::confusionMatrix(predictions,factor(response),positive="pos",mode=c("prec_recall"))
    predictions_performance<-data.frame(Results=c(cm$overall[1],
                                                  cm$byClass[1:11]))
    
    results_balanced_accuracy<-predictions_performance['Balanced Accuracy',]
    results_prec<-predictions_performance['Precision',]
    results_acc<-predictions_performance['Accuracy',]
    results_f1<-predictions_performance['F1',]
    results_sensitivity<-predictions_performance['Sensitivity',]
    results_specificity<-predictions_performance['Specificity',]
    
    
    probs<-predict(model,newdata=dataset_temp,type="prob")
    probs_class_0<-probs[,2][response == "pos"]
    probs_class_1<-probs[,2][response == "neg"]
    
    roc_obj <- roc(response, probs[,2],levels=c("neg","pos"))
    # Compute PR-AUC using PRROC
    pr_curve <- pr.curve(scores.class0 = probs_class_0,
                         scores.class1 = probs_class_1, curve = TRUE)
    PR_auc <- pr_curve$auc.integral  # Extract PR-AUC
    
    ROC_auc <- as.numeric(roc_obj$auc)
    
    
    metrics<-round(c(results_balanced_accuracy,results_acc,results_specificity,results_sensitivity,
               results_f1,results_prec,ROC_auc,PR_auc),3)
    
    row<-c(model_name,metrics)
    
    res_all<-rbind(res_all,row)
  }
  
  res<-data.frame(res_all)
  colnames(res)<-c('Model','Balanced.Accuracy','Accuracy','Specificity','Sensitivity','F1','Precision','ROC.AUC','PR.AUC')
  
  for(col in colnames(res)[2:9]){
    res[,col]<-as.numeric(res[,col])
  }
  row.names(res)<-NULL
  return(res)
  }



####### Main code ############

res_train<-NULL

res_test<-NULL

path_res<-'C:/Users/30697/Desktop/PREGSAFE/Data/Results/'


options<-c('Original','borderline_smote','enn_smote','kmeans_smote','smote','svm_smote','tomek_smote')#,'GAN')#,'GAN','TVAE')

for(fold in 1:10){
  print(fold)
 for(option in options){
   print(option)
   
   test_path<-paste0(path_res,option,'_test_set_',fold,'.Rda')
   model_path<-paste0(path_res,option,'_models_',fold,'.Rda')
   
   
  load(test_path)
  colnames(test_set)<-make.names(colnames(test_set))
   
  load(model_path)
   
   if(option=='Original'){
     path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_',fold,'.csv')
     train_set<-read.csv(path_train,check.names=FALSE)
     colnames(train_set)<-make.names(colnames(train_set))
   }else if(option=='GAN'){
     path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/CTGAN_',fold,'.csv')
     train_set<-read.csv(path_train,check.names=FALSE)
     colnames(train_set)<-make.names(colnames(train_set))
   }else{
     path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_',fold,'_',option,'.csv')
     train_set<-read.csv(path_train,check.names=FALSE)
     colnames(train_set)<-make.names(colnames(train_set))
   }
  
   for(col in colnames(train_set)){
     if(length(unique(train_set[,col]))<=8){
       train_set[,col]<-as.factor(train_set[,col])

     }
   }
   ############### One hot encoding
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
   
   
   ############### Train  Evaluation ################
   res_train_fold<-Evaluation_metrics(train,train_set[,'GDM01'],res_ML)
   
   res_train_fold$Fold<-rep(fold,nrow(res_train_fold))
   res_train_fold$SDG<-rep(option,nrow(res_train_fold))
   res_train<-rbind(res_train,res_train_fold)
   
   ############## Test Evaluation ##################
   res_test_fold<-Evaluation_metrics(test,test_set[,'GDM01'],res_ML)
   
   res_test_fold$Fold<-rep(fold,nrow(res_test_fold))
   res_test_fold$SDG<-rep(option,nrow(res_test_fold))
   
   res_test<-rbind(res_test,res_test_fold)
 } 
}





save(res_test,file='C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/CV_results_test.Rda')
save(res_train,file='C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/CV_results_train.Rda')


options<-c('GAN')#,'GAN','TVAE')

for(fold in 1:10){
  print(fold)
  for(option in options){
    print(option)
    
    test_path<-paste0(path_res,option,'_test_set_',fold,'.Rda')
    model_path<-paste0(path_res,option,'_models_',fold,'.Rda')
    
    
    load(test_path)
    colnames(test_set)<-make.names(colnames(test_set))
    
    load(model_path)
    
    if(option=='Original'){
      path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_',fold,'.csv')
      train_set<-read.csv(path_train,check.names=FALSE)
      colnames(train_set)<-make.names(colnames(train_set))
    }else if(option=='GAN'){
      path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/GAN_AUTO/GAN_',fold,'.csv')
      train_set<-read.csv(path_train,check.names=FALSE)
      colnames(train_set)<-make.names(colnames(train_set))
    }else{
      path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_',fold,'_',option,'.csv')
      train_set<-read.csv(path_train,check.names=FALSE)
      colnames(train_set)<-make.names(colnames(train_set))
    }
    
    for(col in colnames(train_set)){
      if(length(unique(train_set[,col]))<=8){
        train_set[,col]<-as.factor(train_set[,col])
        
      }
    }
    ############### One hot encoding
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
    
    
    ############### Train  Evaluation ################
    res_train_fold<-Evaluation_metrics(train,train_set[,'GDM01'],res_ML)
    
    res_train_fold$Fold<-rep(fold,nrow(res_train_fold))
    res_train_fold$SDG<-rep(option,nrow(res_train_fold))
    res_train<-rbind(res_train,res_train_fold)
    
    ############## Test Evaluation ##################
    res_test_fold<-Evaluation_metrics(test,test_set[,'GDM01'],res_ML)
    
    res_test_fold$Fold<-rep(fold,nrow(res_test_fold))
    res_test_fold$SDG<-rep(option,nrow(res_test_fold))
    
    res_test<-rbind(res_test,res_test_fold)
  } 
}

res_GAN<-res_test
res_GAN_train<-res_train


load('C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/CV_results_test.Rda')
res_test_f<-rbind(res_test,res_GAN)
load('C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/CV_results_train.Rda')
res_train_f<-rbind(res_train,res_GAN_train)


res_test_f<-res_test_f[,c("Model","SDG","Fold","Balanced.Accuracy","Sensitivity","ROC.AUC")]
save(res_test_f,file='C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/CV_test.Rda')
save(res_train_f,file='C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/CV_train.Rda')
