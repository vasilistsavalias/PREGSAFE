################################ Scott-Knott analysis results #######################
library(ScottKnott)
library(reshape2)
library(psych)
library(ggplot2)
library(ggExtra)
library(ggthemes)
library(lmerTest)
library(lsmeans)
library(emmeans)
library("multcompView")
library(dplyr)
library("ScottKnott")

setwd("C:/Current Work Space 26_04_21/Research 2025/PREGSAFE")

load("C:/Current Work Space 26_04_21/Research 2025/PREGSAFE/CV_test.Rda")


res_test<-res_test_f[c('Model','SDG','Fold',setdiff(colnames(res_test_f),c('Model','SDG','Fold')))]


###################################### Test #################################
Dataset<-res_test
colnames(Dataset) <- c("Algorithm",
                       "Technique",
                       "Fold",
                       "Bal.Accuracy",
                       "Sensitivity",
                       "ROC.AUC")

rm(list=setdiff(ls(), "Dataset"))

Dataset$Algorithm <- factor(Dataset$Algorithm)
Dataset$Technique <- factor(Dataset$Technique, levels = c("Original",
                                                          "borderline_smote",
                                                          "enn_smote",
                                                          'GAN',
                                                          "kmeans_smote",
                                                          "smote",
                                                          "svm_smote",
                                                          "tomek_smote"))

Dataset$Fold <- factor(Dataset$Fold)

# Dataset[is.na(Dataset)]<-0

StatisticalComparison <- function (Dataset, Metric = "Bal.Accuracy"){
  ### Exploratory Analytics for AUC
  ExploratoryTable <- describeBy(Dataset[,paste0(Metric)], list(Dataset$Algorithm,
                                                                Dataset$Technique),
                                 mat=T)
  
  ExploratoryTable <- ExploratoryTable[,c("group1","group2","n","mean","sd","median","min","max")]
  
  ExploratoryTable <- ExploratoryTable[order(ExploratoryTable$group1,decreasing = F),]
  
  
  ExploratoryTable[,c("mean","sd","median","min","max")]<- lapply(ExploratoryTable[,c("mean","sd","median","min","max")],
                                                                  round, 4)
  
  Boxplots <- ggplot(data = Dataset ,aes(y = Technique , x = Dataset[,Metric])) +
    geom_boxplot()+
    facet_wrap(.~Algorithm)+
    #scale_fill_viridis_d( option = "D", direction = -1)+  
    theme_bw()+
    removeGrid(x=TRUE, y=TRUE)+
    guides(fill = guide_legend(override.aes = list(alpha = 1,color="black")))+
    theme(legend.position="none",
          strip.text.x = element_text(size = 16, hjust = 0.5, vjust = 0.5, face = 'bold'),
          strip.text.y = element_text(size = 16, hjust = 0.5, vjust = 0.5, face = 'bold'),
          
          legend.title=element_blank(),
          legend.text = element_text(size=30),
          #axis.text.x = element_blank(),
          axis.text.x = element_text(face="bold", color="black", size=15),
          axis.text.y = element_text(face="bold", color="black", size=15),
          #axis.title.x=element_blank(),
          axis.title.x = element_text(face="bold", color="black", size=20),
          axis.title.y = element_text(face="bold", color="black", size=20),
          plot.title = element_text(face="bold", color = "black", size=20))+
    labs(x = Metric, y = "Technique")
  
  
  # LME model with interaction 
  ModelInteraction <- lmer(Dataset[,Metric]~Algorithm*Technique+
                             # Preprocess*Dataset+
                             (1 | Fold),
                           data = Dataset[c("Algorithm","Technique","Fold",Metric)],
                           REML=FALSE)
  
  
  # LME model with main effects
  ModelMain <- lmer(Dataset[,Metric] ~ Algorithm+Technique+
                      # Preprocess*Dataset+
                      (1 | Fold), 
                    data = Dataset[c("Algorithm","Technique","Fold",Metric)],
                    REML=FALSE)
  
  # Comparison of Interaction vs. Main Effects Models
  ModelComparison <- anova(ModelInteraction,ModelMain)
  
  ModelComparison.pvalue <- ModelComparison$`Pr(>Chisq)`
  
  
  if (ModelComparison.pvalue[2]>0.05){
    print("Insignificant interaction term")
    
    ModelFinal <- lmer(Dataset[,Metric] ~ Algorithm+Technique+
                         # Preprocess*Dataset+
                         (1 | Fold), 
                       data = Dataset[c("Algorithm","Technique","Fold",Metric)],
                       REML=TRUE)
    
    AnovaTable <- anova(ModelFinal,ddf="Kenward-Roger")
    
    AnovaTable <- AnovaTable [,c(3:6)]
    
    plot(ModelFinal)
    qqnorm(resid(ModelFinal))
    
    # #### Scott Knott
    formula.SK <- paste(Metric,"~","Algorithm+Technique+(1|Fold)")
    
    anv <- lmer(as.formula(formula.SK), data=Dataset[c("Algorithm","Technique","Fold",Metric)])
    
    ### SK results for Algorithm #### 
    SKResults.Algorithm <- ScottKnott::SK(x=anv,
                                          y=Metric,
                                          which='Algorithm')
    
    SKResultsStatistics.Algorithm <- data.frame(SKResults.Algorithm$info)
    
    Algorithm <- SKResultsStatistics.Algorithm [,1]
    Measure <- SKResultsStatistics.Algorithm [,2]
    Lower <- SKResultsStatistics.Algorithm[,4]
    Upper <- SKResultsStatistics.Algorithm[,5]
    
    ClusterMatrix <- SKResults.Algorithm$out$Result
    ClusterMatrix <- ClusterMatrix[-1]
    ClusterMatrix <- cbind(ClusterMatrix, Group = apply(ClusterMatrix, 1, 
                                                        function(x) paste0(na.omit(x), collapse = "")))
    
    Groups.SK.Algorithm <- data.frame(Algorithm,Measure,Lower,Upper,ClusterMatrix$Group)
    
    colnames(Groups.SK.Algorithm)[5] <- c("Cluster")
    
    ### SK results for Technique #### 
    SKResults.Technique <- ScottKnott::SK(x=anv,
                                          y=Metric,
                                          which='Technique')
    
    SKResultsStatistics.Technique <- data.frame(SKResults.Technique$info)
    
    Technique <- SKResultsStatistics.Technique [,1]
    Measure <- SKResultsStatistics.Technique [,2]
    Lower <- SKResultsStatistics.Technique[,4]
    Upper <- SKResultsStatistics.Technique[,5]
    
    ClusterMatrix <- SKResults.Technique$out$Result
    ClusterMatrix <- ClusterMatrix[-1]
    ClusterMatrix <- cbind(ClusterMatrix, Group = apply(ClusterMatrix, 1, 
                                                        function(x) paste0(na.omit(x), collapse = "")))
    
    Groups.SK.Technique <- data.frame(Technique,Measure,Lower,Upper,ClusterMatrix$Group)
    
    colnames(Groups.SK.Technique)[5] <- c("Cluster")
    
    list(
      ExploratoryTable = ExploratoryTable,
      Boxplots = Boxplots, 
      Groups.SK.Algorithm = Groups.SK.Algorithm,
      Groups.SK.Technique = Groups.SK.Technique)
    
    
  } else {
    print("Significant interaction term was identified")
    
    ModelFinal <- lmer(Dataset[,Metric] ~ Algorithm*Technique+
                         # Preprocess*Dataset+
                         (1 | Fold), 
                       data = Dataset[c("Algorithm","Technique","Fold",Metric)],
                       REML=TRUE)
    
    AnovaTable <- anova(ModelFinal,ddf="Kenward-Roger")
    
    AnovaTable <- AnovaTable [,c(3:6)]
    
    plot(ModelFinal)
    qqnorm(resid(ModelFinal))
    
    # #### Scott Knott
    formula.SK <- paste(Metric,"~","Algorithm*Technique+(1 | Fold)")
    
    
    
    anv <- lmer(as.formula(formula.SK), data=Dataset[c("Algorithm","Technique","Fold",Metric)])
    
    
    
    ############# New ####################
    
    library(ScottKnott)
    
    Dataset$Algorithm_Technique <- interaction(
      Dataset$Algorithm, Dataset$Technique, sep = "/", drop = TRUE
    )
    
    # build a proper formula
    formula.SK <- as.formula(paste(Metric, "~ Algorithm_Technique"))
    
    # fit the ANOVA
    anv_AOV <- aov(formula.SK, data = Dataset)
    
    # run Scott?Knott
    SKResults <- ScottKnott::SK(anv_AOV)
    ##########################################
    
    ### SK results for Algorithm #### 
    Groups.SK.Algorithm <- data.frame()
    
    for(i in 1:nlevels(Dataset$Algorithm)){
      SKResults.Algorithm <- ScottKnott::SK(x=anv,
                                            y=Metric,
                                            which='Algorithm:Technique',
                                            fl1=i)
      
      
      SKResultsStatistics.Algorithm <- data.frame(SKResults.Algorithm$info)
      
      Algorithm <- SKResultsStatistics.Algorithm [,1]
      Technique <- SKResultsStatistics.Algorithm [,2]
      Measure <- SKResultsStatistics.Algorithm [,3]
      Lower <- SKResultsStatistics.Algorithm[,6]
      Upper <- SKResultsStatistics.Algorithm[,7]
      
      ClusterMatrix <- SKResults.Algorithm$out$Result
      ClusterMatrix <- ClusterMatrix[-1]
      ClusterMatrix <- cbind(ClusterMatrix, Group = apply(ClusterMatrix, 1, 
                                                          function(x) paste0(na.omit(x), collapse = "")))
      
      Groups.SK.Algorithm0 <- data.frame(Algorithm,Technique,Measure,Lower,Upper,ClusterMatrix$Group)
      
      colnames(Groups.SK.Algorithm0)[6] <- c("Cluster")
      Groups.SK.Algorithm <- rbind(Groups.SK.Algorithm, Groups.SK.Algorithm0)
    }
    
    Dataset$Interaction <- interaction(Dataset$Algorithm, 
                                       Dataset$Technique, 
                                       sep = "/",
                                       drop=T)
    
    ModelFinalInteraction <- lmer(Dataset[,Metric] ~ Interaction+
                                    # Preprocess*Dataset+
                                    (1 | Fold), 
                                  data = Dataset[c("Interaction","Fold",Metric)],
                                  REML=TRUE)
    
    AnovaTableInteraction <- anova(ModelFinalInteraction,ddf="Kenward-Roger")
    
    AnovaTableInteraction <- AnovaTableInteraction [,c(3:6)]
    
    # #### Scott Knott
    formula.SK.Interaction <- paste(Metric,"~","Interaction+(1 | Fold)")
    
    anvInteraction <- lmer(as.formula(formula.SK.Interaction), 
                           data=Dataset[c("Interaction","Fold",Metric)])
    
    ### SK results for Algorithm #### 
    SKResults.Interaction <- ScottKnott::SK(x=anvInteraction,
                                            y=Metric,
                                            which='Interaction')
    
    
    SKResultsStatistics.Interaction <- data.frame(SKResults.Interaction$info)
    
    AlgorithmxTechnique.Interaction <- SKResultsStatistics.Interaction [,1]
    Measure.Interaction <- SKResultsStatistics.Interaction [,2]
    Lower.Interaction <- SKResultsStatistics.Interaction[,4]
    Upper.Interaction <- SKResultsStatistics.Interaction[,5]
    
    ClusterMatrix.Interaction <- SKResults.Interaction$out$Result
    ClusterMatrix.Interaction <- ClusterMatrix.Interaction[-1]
    ClusterMatrix.Interaction <- cbind(ClusterMatrix.Interaction, Group = apply(ClusterMatrix.Interaction, 1, 
                                                                                function(x) paste0(na.omit(x), collapse = "")))
    
    Groups.SK.Interaction <- data.frame(AlgorithmxTechnique.Interaction,
                                        Measure.Interaction,
                                        Lower.Interaction,
                                        Upper.Interaction,
                                        ClusterMatrix.Interaction$Group)
    
    colnames(Groups.SK.Interaction)[5] <- c("Cluster")
    
    
    list(
      ExploratoryTable = ExploratoryTable,
      Boxplots = Boxplots,
      Groups.SK.Algorithm =Groups.SK.Algorithm,
      Groups.SK.Interaction = Groups.SK.Interaction) 
  }
  
}


#MultCompTable <- difflsmeans(ModelAUCMainFinal, test.effs="Algorithm")
# lsm.Algorithm.Technique.AUC <- emmeans(ModelAUCMainFinal, 
#                                          list(pairwise ~ Algorithm+Technique), type="response")
# 
# LSMeans4Models.Algorithm.AUC <- data.frame(lsm.Algorithm.Technique.AUC$`emmeans of Algorithm, Technique`)
# 
# 
# PostHocAnalysisAE <- data.frame(lsm.Algorithm.Technique.AUC$`pairwise differences of Algorithm, Technique`)
# PostHocAnalysisAE <-  PostHocAnalysisAE[complete.cases(PostHocAnalysisAE),]
# 

### Example for calling the function
Balanced_Accuracy_Results <- StatisticalComparison (Dataset, Metric = "Bal.Accuracy")
Balanced_Accuracy_Results <- StatisticalComparison (Dataset, Metric = "Sensitivity")




load('C:/Users/30697/Desktop/PREGSAFE/Results/Latest/Balanced Accuracy Results.RData')
load('C:/Users/30697/Desktop/PREGSAFE/Results/Latest/Sensitivity Results.RData')
load('C:/Users/30697/Desktop/PREGSAFE/Results/Latest/AUC ROC Results.RData')








wanted<-Balanced_Accuracy_Results$Groups.SK.Interaction
wanted[,c(2,4)]<-round(wanted[,c(2,4)],3)

wanted_col<-c()
for(i in 1:40){
  wanted_row<-wanted[i,]
  median1<-wanted_row$Measure.Interaction
  q1<-wanted_row$Lower.Interaction
  q3<-wanted_row$Upper.Interaction
  
  i_want<-paste0(median1," [",q1," ",q3,"]")
  wanted_col<-c(wanted_col,i_want)
}

wanted$Col<-wanted_col
library(flextable)

ft <- flextable(wanted)
ft <- autofit(ft) 

library(officer)

doc <- read_docx()         # create new Word document
doc <- body_add_flextable(doc, ft)
print(doc, target = "C:/Users/30697/Desktop/PREGSAFE/Results/Latest/table_output.docx")

wanted<-Balanced_Accuracy_Results$Groups.SK.Algorithm
wanted[,c(3,5)]<-round(wanted[,c(3,5)],3)


wanted_col<-c()
for(i in 1:40){
  wanted_row<-wanted[i,]
  median1<-wanted_row$Measure
  q1<-wanted_row$Lower
  q3<-wanted_row$Upper
  
  i_want<-paste0(median1," [",q1," ",q3,"]")
  wanted_col<-c(wanted_col,i_want)
}

wanted$Col<-wanted_col
library(flextable)

ft <- flextable(wanted)
ft <- autofit(ft) 

library(officer)

doc <- read_docx()         # create new Word document
doc <- body_add_flextable(doc, ft)
print(doc, target = "C:/Users/30697/Desktop/PREGSAFE/Results/Latest/table_output_per_algorithm.docx")




wanted<-Sensitivity_Results$Groups.SK.Interaction
wanted[,c(2,4)]<-round(wanted[,c(2,4)],3)
wanted_col<-c()
for(i in 1:40){
  wanted_row<-wanted[i,]
  median1<-wanted_row$Measure.Interaction
  q1<-wanted_row$Lower.Interaction
  q3<-wanted_row$Upper.Interaction
  
  i_want<-paste0(median1," [",q1," ",q3,"]")
  wanted_col<-c(wanted_col,i_want)
}
wanted$Col<-wanted_col

library(flextable)

ft <- flextable(wanted)
ft <- autofit(ft) 

library(officer)

doc <- read_docx()         # create new Word document
doc <- body_add_flextable(doc, ft)
print(doc, target = "C:/Users/30697/Desktop/PREGSAFE/Results/Latest/table_output_sensitivity.docx")




wanted<-Sensitivity_Results$Groups.SK.Algorithm
wanted[,c(3,5)]<-round(wanted[,c(3,5)],3)


wanted_col<-c()
for(i in 1:40){
  wanted_row<-wanted[i,]
  median1<-wanted_row$Measure
  q1<-wanted_row$Lower
  q3<-wanted_row$Upper
  
  i_want<-paste0(median1," [",q1," ",q3,"]")
  wanted_col<-c(wanted_col,i_want)
}

wanted$Col<-wanted_col

library(flextable)

ft <- flextable(wanted)
ft <- autofit(ft) 

library(officer)

doc <- read_docx()         # create new Word document
doc <- body_add_flextable(doc, ft)
print(doc, target = "C:/Users/30697/Desktop/PREGSAFE/Results/Latest/table_output_sensitivity_per_algorithm.docx")




load('C:/Users/30697/Desktop/PREGSAFE/Data/Metrics_res/Intrisinc_quality_res.Rda')
library(dplyr)
df_2<-df_final%>%
  group_by(SDG) %>%
  summarise(
    med_hellinger = median(Hellinger, na.rm = TRUE),
    q1_hellinger   = as.numeric(quantile(Hellinger,na.rm=TRUE)[2]),
    q3_hellinger = as.numeric(quantile(Hellinger,na.rm=TRUE)[4]),
    med_PCD = median(PCD, na.rm = TRUE),
    q1_PCD   = as.numeric(quantile(PCD,na.rm=TRUE)[2]),
    q3_PCD= as.numeric(quantile(PCD,na.rm=TRUE)[4]),
    med_Propens = median(Propensity, na.rm = TRUE),
    q1_Propens   = as.numeric(quantile(Propensity,na.rm=TRUE)[2]),
    q3_Propens = as.numeric(quantile(Propensity,na.rm=TRUE)[4])
    
    
  )


library(flextable)
df_2[,3:10]<-round(df_2[,3:10],3)
ft <- flextable(df_2)
ft <- autofit(ft) 

library(officer)

doc <- read_docx()         # create new Word document
doc <- body_add_flextable(doc, ft)
print(doc, target = "C:/Users/30697/Desktop/PREGSAFE/Results/Latest/table_output_Intr2.docx")


