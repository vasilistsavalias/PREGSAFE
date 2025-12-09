################################## Script for variables selection / train-test splitting #########################
library(readxl)

data<-read_excel('C:/Users/30697/Desktop/PREGSAFE/Data/128.xlsx')

dd<-data[sample(row.names(data),size=nrow(data)),]
########## Remove the PI Ut A NAs --- only 5 patients missing values
data_na<-data[!is.na(data$`PI Ut A`),]

############## Remove features not intented for diagnosis #########
feats_to_exclude<-c("Ν2","N1","PID","PID1","PI Ut A","Time of Diagnosis","Early Diagnosis 01","Treatment","Insulin01","Exam date","EDD US","DOB","W","D","days in 2nd tr")
dataset<-data_na[!colnames(data_na) %in% feats_to_exclude]


dataset<-dataset[colnames(dataset) %in% c('GDM01',"Thyroid all","Hypoth01",'MA','Parity','Conception ART 01','Smoking01',
                                          'BMI pre','Wgain','GA days',"PI Ut A","Wt pre")]

############ Univariate LR results ##################
res<-NULL
predictors<-setdiff(colnames(dataset),'GDM01')
for(predictor in predictors){
  form <- as.formula(paste('GDM01', '~', paste0('`', predictor, '`')))
  res_lr<-glm(form,family='binomial',data=dataset)
  OR <- round(as.numeric(exp(coef(res_lr))[2]),3)
  
  # 95% CI for coefficients (log-odds)
  ci <- confint.default(res_lr)  # Wald-type CI (fast)
  # or: confint(fit)          # profile likelihood CI (slower but more accurate)
  
  # Convert to OR scale
  OR_CI <- exp(ci)[2,]
  lower_CI<-round(as.numeric(OR_CI[1]),3)
  upper_CI<-round(as.numeric(OR_CI[2]),3)
  
  train_temp<-dataset[!is.na(dataset[[predictor]]),]
  res_null<-glm(as.formula(paste0('GDM01',' ~ 1')),family='binomial',data=train_temp)
  res_anova<-anova(res_null,res_lr,test="Chisq")
  pval <- round(res_anova$`Pr(>Chi)`[2],3)
  
  
  df_predictor<-data.frame(Predictor=predictor,OR=OR,LowerOR=lower_CI,UpperOR=upper_CI,p_value=pval)
  res<-rbind(res,df_predictor)
}



##### Feature selection: Boruta ###########
library(caret)

folds <- createFolds(dataset$GDM01, 
                     k = 10,         # number of folds
                     list = TRUE,    # return indices in a list
                     returnTrain = TRUE)  

train_sets<-list()
test_sets<-list()
for(fold in 1:length(folds)){
  train_idx<-folds[[fold]]
  train_i<-dataset[train_idx,]
  test_i<-dataset[-train_idx,]
  
  train_sets[[fold]]<-train_i
  test_sets[[fold]]<-test_i
}


boruta_selected<-list()
library(Boruta)
library(dplyr)


for(i in 1:10){
  train_all_i<-train_sets[[i]]
  test_all_i<-test_sets[[i]]
  
  train_all_i <- train_all_i %>%
    mutate(across(where(is.character), as.factor))
  
  train_all_1<-train_all_i$Parity
  
  train_all_3<-train_all_i$`GA days`
  
  train_all_i$`GA days`<-train_all_1
  train_all_i$Parity<-train_all_3
  
  boruta_decision<-Boruta(GDM01 ~ .,data=train_all_i,pValue=0.05,maxRuns=500)
  selected_features_fold <-colnames(train_all_i)[2:ncol(train_all_i)][boruta_decision$finalDecision=='Confirmed']
  boruta_selected[[i]]<-selected_features_fold
}







################ Next step selected the independent features from Boruta list #####################
indep_feats_fold<-list()
for(i in 1:10){
  boruta_fold<-boruta_selected[[i]]
  temp_feats<-c('GDM01')
  if(any(c('Thyroid all','Type','Hypoth01') %in% boruta_fold)){
    temp_feats<-c(temp_feats,'Thyroid all')
  }
  if(any(c("MA","MA>35 01") %in% boruta_fold)){
    temp_feats<-c(temp_feats,"MA")
  }
  if(any(c("Parity", "Parity01") %in% boruta_fold)){
    #temp_feats<-c(temp_feats,"Parity")
  }
 
  temp_feats<-c(temp_feats,"Conception ART 01")
  
  if("Smoking01" %in% boruta_fold){
    temp_feats<-c(temp_feats,"Smoking01")
  }
  if(any(c("Wt pre","Ht","BMI pre","BMIcat","UW" ,"NW","OW","OB1","OB2","OB3") %in% boruta_fold)){
    temp_feats<-c(temp_feats,"Wt pre","BMI pre")
  }
  if(any(c("Wt now","Wgain","Wgcat","WGmin","WGmax","WGav","WGextra","WG>av","Wgcat less","Wgcat normal","Wgcat more" ) %in% boruta_fold)){
    temp_feats<-c(temp_feats,"Wgain")
  }
  if("GA days" %in% boruta_fold){
    temp_feats<-c(temp_feats,"GA days")
  }
  if("PI Ut A" %in% boruta_fold){
    temp_feats<-c(temp_feats,"PI Ut A")
  }
  indep_feats_fold[[i]]<-temp_feats
}






for(i in 1:10){
  train_all_i<-train_sets[[i]]
  #write.csv(train_all_i,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_all',i,'.csv'),row.names = FALSE)
  
  test_all_i<-test_sets[[i]]
  #write.csv(test_all_i,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Test_set_all',i,'.csv'),row.names = FALSE)
  
  
  train_i<-train_all_i[colnames(train_all_i) %in% feats]
  write.csv(train_i,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_',i,'.csv'),row.names = FALSE)
  
  test_i<-test_all_i[colnames(test_all_i) %in% feats]
  write.csv(test_i,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Test_set_',i,'.csv'),row.names = FALSE)
  
}


save(boruta_selected,file='C:/Users/30697/Desktop/PREGSAFE/Code/Selected_features_Boruta.Rda')






