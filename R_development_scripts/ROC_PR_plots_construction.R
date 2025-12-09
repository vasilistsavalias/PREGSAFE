############################################## ROC PR plots ##################################
#################################################### Evaluation code #########################################
rm(list=ls())
######## Libraries ###########
library(MLmetrics)
library(caret)
library(pROC)
library(PRROC)



######## Functions ###########
Evaluation_metrics_ROC_PR<-function(dataset,response,res_ML){
  response<-factor(response,labels=c("neg","pos"))
  
  class_all<-c()
  ls_for_probs<-list()
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
    
    model_list_res<-list()
    probs<-predict(model,newdata=dataset_temp,type="prob")
    
    probs_class_0<-probs[,2][response == "pos"]
    probs_class_1<-probs[,2][response == "neg"]
    
    model_list_res[['ROC_probs']]<-probs[,2]
    model_list_res[['Prob_class_0']]<-probs_class_0
    model_list_res[['Prob_class_1']]<-probs_class_1
    
    ls_for_probs[[model_name]]<-model_list_res
  }
  
  return(list(Results=ls_for_probs,Class=response))
}



####### Main code ############

res_train<-NULL

res_test<-NULL

#options<-c('Original','borderline_smote','enn_smote','kmeans_smote','smote','svm_smote','tomek_smote','GAN')#,'GAN','TVAE')
options<-c('Original','borderline_smote','smote','GAN')#,'GAN','TVAE')

path_res<-'C:/Users/30697/Desktop/PREGSAFE/Data/Results/'

models_wanted<-c('NB_tomek_smote','NB_borderline_SMOTE','NB_smote','NB_enn_smote','NB_svm_smote','KNN_svm_smote','SVM_svm_smote')
res_list<-list()
all_class<-c()
for(fold in 1:10){
  print(fold)
  for(model_opt in models_wanted){
    options<-strsplit(model_opt,split="_")[[1]]
    if(length(options)==2){
      option<-options[2]
    }else{
      option<-paste0(options[2],'_',options[3])
    }
    test_path<-paste0(path_res,option,'_test_set_',fold,'.Rda')
    model_path<-paste0(path_res,option,'_models_',fold,'.Rda')
    
    
    load(test_path)
    load(model_path)
    
    path_train<-paste0('C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_',fold,'_',option,'.csv')
    train_set<-read.csv(path_train,check.names=FALSE)
    for(col in colnames(train_set)){
      if(length(unique(train_set[,col]))<=8){
        train_set[,col]<-as.factor(train_set[,col])
        
      }
    }
    ############### One hot encoding
    colnames(train_set)<-make.names(colnames(train_set))
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
    
    model_option<-paste0(options[1],'_model')
    #res_ML<-res_ML$RF_model
    res_ML_end<-list()
    res_ML_end[[model_option]]<-res_ML[[model_option]]
    ############### Test  Evaluation ################
    res_train_fold<-Evaluation_metrics_ROC_PR(test,test_set[,'GDM01'],res_ML_end)
    
    
    probs<-res_train_fold$Results
    class<-res_train_fold$Class
    res_list[[paste0(model_option,'_',option,'_',fold)]]<-probs

  }
  all_class<-c(all_class,class)
  
}



############### ROC Curves ##########
ls_for_roc<-list()
ls_for_pr_class0<-list()
ls_for_pr_class1<-list()
for(fold in 1:10){
  for(model_opt in models_wanted){
    options<-strsplit(model_opt,split="_")[[1]]
    if(length(options)==2){
      option<-options[2]
    }else{
      option<-paste0(options[2],'_',options[3])
    }
    name<-paste0(options[1],'_model_',option,'_',fold)
    wanted<-res_list[[name]]
    for(name in names(wanted)){
      name_n<-paste0(name,'_',option)
      if(name_n %in% names(ls_for_roc)){
        ls_for_roc[[name_n]]<-c(ls_for_roc[[name_n]],wanted[[name]]$ROC_probs)
        ls_for_pr_class0[[name_n]]<-c(ls_for_pr_class0[[name_n]],wanted[[name]]$Prob_class_0)
        ls_for_pr_class1[[name_n]]<-c(ls_for_pr_class1[[name_n]],wanted[[name]]$Prob_class_1)
        
      }else{
        ls_for_roc[[name_n]]<-wanted[[name]]$ROC_probs
        ls_for_pr_class0[[name_n]]<-wanted[[name]]$Prob_class_0
        ls_for_pr_class1[[name_n]]<-wanted[[name]]$Prob_class_1
      }
    }
  }
}


all_class<-factor(all_class)
levels(all_class)<-c(0,1)



roc_list <- list()
auc_tbl  <- tibble(Model = character(), AUC = numeric())


for(name in names(ls_for_roc)) {
  sc<-ls_for_roc[[name]]
  roc_obj <- pROC::roc(response = all_class, predictor = sc, quiet = TRUE)
  roc_list[[name]] <- roc_obj
  auc_tbl <- bind_rows(auc_tbl, tibble(Model = name, AUC = as.numeric(pROC::auc(roc_obj))))
}

auc_tbl$Model<-c('NB/Tomek SMOTE','NB/Borderline SMOTE','NB/SMOTE','NB/ENN SMOTE','NB/SVM SMOTE','KNN/SVM SMOTE','SVM/SVM SMOTE')
names(roc_list)<-c('NB/Tomek SMOTE','NB/Borderline SMOTE','NB/SMOTE','NB/ENN SMOTE','NB/SVM SMOTE','KNN/SVM SMOTE','SVM/SVM SMOTE')

# prettify legend labels as "name (AUC=0.873)" and order by AUC
auc_tbl <- auc_tbl %>% arrange(desc(AUC)) %>%
  mutate(Label = sprintf("%s (AUC=%.3f)", Model, AUC))
name_map <- setNames(auc_tbl$Label, auc_tbl$Model)
names(roc_list) <- name_map[names(roc_list)]


# ---- 2. Convert to data frame for ggplot ----
df_roc <- purrr::map_dfr(names(roc_list), function(name) {
  roc_df <- data.frame(
    FPR = 1 - roc_list[[name]]$specificities,
    TPR = roc_list[[name]]$sensitivities,
    Model = name
  )
  roc_df
})

# ---- 3. Create nice labels with AUC ----
auc_tbl <- auc_tbl %>%
  arrange(desc(AUC)) %>%
  mutate(Label = sprintf("%s (AUC = %.3f)", Model, AUC))
#df_roc$Model <- factor(df_roc$Model, levels = auc_tbl$Model, labels = auc_tbl$Label)

# ---- 4. Plot with your requested style ----
p4<-ggplot(df_roc, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.5) +                                # thicker lines
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_brewer(palette = "Dark2") +                # nice distinct colors
  labs(title = "ROC Curves",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_bw() +
  theme(
    legend.position = c(0.7, 0.2),
    legend.background = element_rect(fill = "white", color = "black"),
    legend.text = element_text(size = 13, face = "bold"),
    legend.title = element_blank(),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text  = element_text(size = 12, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )
library(tidyverse)
setwd('C:/Users/30697/Desktop')
tiff('ROC.jpeg',units="in",width=10,height=10,res=300)
p4
dev.off()


### 95% CIs:: 
roc1<-roc_list$`SVM/SVM SMOTE (AUC=0.631)`
ci.auc(roc1)


### Delong tests::
list_results_delong_numerical<-list()
names_all<-names(roc_list)
for(name in names_all){
  roc1<-roc_list[[name]]
  rest_names <-setdiff(names_all,name)
  for(second_name in rest_names){
    roc2<-roc_list[[second_name]]
    res<-roc.test(roc1,roc2,method="delong",boot.n=1000)
    test<-paste(name,second_name,sep="~")
    list_results_delong_numerical[[test]]=res
  }
  names_all<-setdiff(names_all,name)
}

auc_1<-c()
auc_2<-c()
z<-c()
confint_1<-c()
confint_2<-c()
p_value<-c()
comparison_names<-c()
for(name in names(list_results_delong_numerical)){
  comp<-list_results_delong_numerical[[name]]
  comparison_names<-c(comparison_names,name)
  auc_1<-c(auc_1,as.numeric(comp$estimate[1]))
  auc_2<-c(auc_2,as.numeric(comp$estimate[2]))
  z<-c(z,as.numeric(comp$statistic))
  confint_1<-c(confint_1,comp$conf.int[1])
  confint_2<-c(confint_2,comp$conf.int[2])
  p_value<-c(p_value,comp$p.value)
}
names_all<-c('ROC Comparison','AUC (1)','AUC (2)','Statistic (Z)','P-Value','Lower CI','Upper CI')
res_delong_numerical<-data.frame(`ROC Comparison`=comparison_names,`AUC(1)`=auc_1,`AUC(2)`=auc_2,`Statistic (Z)`=z,`P-Value`=p_value,
                                 `Lower CI`=confint_1,`Upper CI`=confint_2)
colnames(res_delong_numerical)<-names_all

#res_delong_numerical$`P-Value`<-p.adjust(res_delong_numerical$`P-Value`,method='fdr')

res_delong_numerical[2:7]<-round(res_delong_numerical[2:7],3)
library(flextable)

ft <- flextable(res_delong_numerical)
ft <- autofit(ft) 

library(officer)

doc <- read_docx()         # create new Word document
doc <- body_add_flextable(doc, ft)
print(doc, target = "C:/Users/30697/Desktop/table_output.docx")
