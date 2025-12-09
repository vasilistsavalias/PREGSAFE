########################################### Engineer and prepare the dataset per fold for the ML pipeline #######################
######### Libraries ##########




########## Feats selected by Boruta per fold ###########
load('C:/Users/30697/Desktop/PREGSAFE/Code/Selected_features_Boruta.Rda')
smote_path<-'C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants'


################ Test sets ########################
original_path<-'C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Test_set_all'
for(i in 1:10){
  origin<-paste0(original_path,i,'.csv')
  data<-read.csv(origin,check.names = FALSE)
  test<-data[colnames(data) %in% c(boruta_selected[[i]],'GDM01')]
  write.csv(test,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/Test_set_',i,'.csv'),row.names = FALSE)
}



############# Original train sets ########################
original_path<-'C:/Users/30697/Desktop/PREGSAFE/Data/Folds/Train_set_all'
for(i in 1:10){
  origin<-paste0(original_path,i,'.csv')
  data<-read.csv(origin,check.names = FALSE)
  train<-data[colnames(data) %in% c(boruta_selected[[i]],'GDM01')]
  write.csv(train,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/Original_train_set_',i,'.csv'),row.names = FALSE)
}


dataset<-read.csv('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/Original_train_set_1.csv')
########### SMOTE variants ###############################

####### Function for engineering ######
engineering_feats<-function(dataset_selected,all_feats){
  MA35 <- as.numeric(dataset_selected$MA>35)
  dataset_selected$`MA>35 01`<-MA35
  
  BMIpre<-round((dataset_selected$`Wt pre`)*10000/dataset_selected$Ht^2,1)
  dataset_selected$`BMI pre`<-BMIpre
  
  BMIcat<-ifelse(BMIpre<18.5,"UW",ifelse(BMIpre<=24.9,"NW",
                                         ifelse(BMIpre<=29.9,"OW",
                                                ifelse(BMIpre<=34.9,"OB1",
                                                       ifelse(BMIpre<=39.9,"OB2","OB3"))
                                         )))
  dataset_selected$BMIcat <- BMIcat
  
  ### UW category
  UW<-as.numeric(BMIcat=="UW")
  dataset_selected$UW <- UW
  ### NW category
  NW<-as.numeric(BMIcat=="NW")
  dataset_selected$NW <- NW
  ### OW category
  OW<-as.numeric(BMIcat=="OW")
  dataset_selected$OW <- OW
  
  ### OB1
  OB1<-as.numeric(BMIcat=="OB1")
  dataset_selected$OB1 <- OB1
  ### OB2
  OB2<-as.numeric(BMIcat=="OB2")
  dataset_selected$OB2 <- OB2
  ### OB3
  OB3<-as.numeric(BMIcat=="OB3")
  dataset_selected$OB3 <-OB3
  
  
  Wtnow<-dataset_selected$`Wt pre`+dataset_selected$Wgain
  dataset_selected$`Wt now` <-Wtnow
  
  
  BMIgrp_IOM <- ifelse(BMIpre < 18.5, "UW",
                       ifelse(BMIpre <= 24.9, "NW",
                              ifelse(BMIpre <= 29.9, "OW", "OB")))
  
  # Weeks elapsed in 2nd/3rd trimester (clip at 0 in case GA<98)
  weeks_after14 <- pmax(0, (dataset_selected$`GA days` - 98) / 7)
  rate_min <- ifelse(BMIgrp_IOM == "UW", 0.44,
                     ifelse(BMIgrp_IOM == "NW", 0.35,
                            ifelse(BMIgrp_IOM == "OW", 0.23, 0.17)))
  rate_max <- ifelse(BMIgrp_IOM == "UW", 0.58,
                     ifelse(BMIgrp_IOM == "NW", 0.50,
                            ifelse(BMIgrp_IOM == "OW", 0.33, 0.27)))
  ft_min <- 0.5 ######## Need approval --- Need to check with the clinicians. 652 patients are not using the 0.5 and 2.0 values for calculation of WG_min and WGmax
  ft_max <- 2.0
  
  # Expected cumulative gain range at the visit
  ## WGmin
  WG_min <- ft_min + weeks_after14 * rate_min
  dataset_selected$`WGmin`<-WG_min
  ## WG max
  WG_max <- ft_max + weeks_after14 * rate_max
  dataset_selected$`WGmax`<-WG_max
  ## WG av
  WG_av  <- (WG_min + WG_max) / 2
  dataset_selected$WGav <- WG_av
  
  
  ############################################### ALL THE OTHER DEPENDS ON IT ##################################
  ## WG extra
  WG_extra <- dataset_selected$Wgain - WG_max
  dataset_selected$WGextra <- WG_extra
  # WG > av
  WG_av <- dataset_selected$Wgain - WG_av
  dataset_selected$`WG>av` <- WG_av
  #Wgcat categories
  Wgcat <- ifelse(dataset_selected$Wgain < WG_min, "Less",
                  ifelse(dataset_selected$Wgain > WG_max, "More",
                         "Normal"))
  dataset_selected$Wgcat <- Wgcat
  
  #Wgcate less
  Wgcat_less<-as.numeric(Wgcat=="Less")
  dataset_selected$`Wgcat less` <- Wgcat_less
  #Wgcate more
  Wgcat_more<-as.numeric(Wgcat=="More")
  dataset_selected$`Wgcat more` <- Wgcat_more
  #Wgcate normal
  Wgcat_normal<-as.numeric(Wgcat=="Normal")
  dataset_selected$`Wgcat normal` <- Wgcat_normal
  
  
  wanted_data<-dataset_selected[colnames(dataset_selected) %in% c(all_feats,'GDM01')]
  
  return(wanted_data)
  
  
}


original_path<-'C:/Users/30697/Desktop/PREGSAFE/Data/SMOTE_variants/Train_set_'
smote_choices<-c('borderline_smote','enn_smote','kmeans_smote','smote','svm_smote','tomek_smote')
for(choice in smote_choices){
  for(fold in 1:10){
    choice<-'kmeans_smote'
    path_choice<-paste0(original_path,fold,'_',choice,'.csv')
    data_choice<-read.csv(path_choice,check.names = FALSE)
    #feats_boruta<-boruta_selected[[fold]]
    
    #dataset_wanted<-engineering_feats(data_choice,feats_boruta)
    
    dataset_wanted<-data_choice
    dataset_wanted$`Conception ART 01`<-as.factor(ifelse(dataset_wanted$`Conception ART 01`>0.5,1,0))
    levels(dataset_wanted$`Conception ART 01`)<-c(0,1)
    write.csv(dataset_wanted,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/',choice,'_train_set_',fold,'.csv'),row.names = FALSE)
    
  }
}

























################## CTGAN and Autoencoders ############
res_all<-list()
path<-'C:/Users/30697/Desktop/PREGSAFE/Data/GAN_AUTO/Train_set_'
for(i in 1:10){
  path_ctgan<-paste0(path,i,'/','synthetic_data_CTGAN.csv')
  #path_auto<-paste0(path,i,'/','synthetic_data_TVAE.csv')
  path_orig<-paste0(path,i,'/','original_data.csv')
  
  
  orig_data<-read.csv(path_orig,check.names = FALSE)
  ctgan_data<-read.csv(path_ctgan,check.names = FALSE)
  #auto_data<-read.csv(path_auto,check.names = FALSE)
  
  ctgan_i<-rbind(orig_data,ctgan_data)
  #auto_i<-rbind(orig_data,auto_data)
  
  feats_boruta<-boruta_selected[[i]]
  ctgan_wanted<-engineering_feats(ctgan_i,feats_boruta)
  write.csv(ctgan_wanted,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/','GAN','_train_set_',i,'.csv'),row.names = FALSE)
  
  #auto_wanted<-engineering_feats(auto_i,feats_boruta)
  #write.csv(auto_wanted,file=paste0('C:/Users/30697/Desktop/PREGSAFE/Data/ML_sets/','TVAE','_train_set_',i,'.csv'),row.names = FALSE)
  
  
}
