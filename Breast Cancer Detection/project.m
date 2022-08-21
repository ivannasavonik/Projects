%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AEML - Course Project
Submitted by -
 - Vivek Bhargava 
 - Ivanna Savonik 

No. of functions used -
- We have used a total of 23 functions and their .m scripts are attached
  in the main folder 
- The main program file is named as main.m file
- The main data csv file is called wdbc.csv and is attachecd in the main
 folder

Other details -


  Data description
  (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
  ------------------------------------------------------------------

  > 

    Number of Instances
    -------------------
    class B -> 357
    class M -> 212
 	
 
    Number of Attributes - 30
    -------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
fprintf('\n******************** LOADING DATA ********************')
clear all;
close all;
clc

dataset = readtable('wdbc.csv');
[n,k] = size(dataset);

X_all = table2array(dataset(:,3:k));
classes = (dataset(:, 2));
%

indOfClassB = find(strcmp(classes.Var2, 'B')==1);
indOfClassM = setdiff(1:n,indOfClassB)';
X_B = table2array(dataset(indOfClassB,3:k));
X_M = table2array(dataset(indOfClassM,3:k));

rng(1001)

%Y = strcmp(table2array(classes),'M'); % returns 1 if M 0 if B
Y = table2array(dataset(:,2));
% Y = 1 or 'TRUE' ==> Malignant (Positive)
% Y = 0 or 'FALSE' ==> Benign (Negative)
[X_train, X_test, Y_train, Y_test] = splitSample(X_all,Y, 0.8, true);

[X_train_norm, mu_X_train, sigma_X_train] = featureNormalize(X_train);
% Now normalizing testing data using value from training standards
X_test_norm = (X_test - mu_X_train)./sigma_X_train;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 1 - Model Training - Preliminary Exercise
% Comments - 
% In this part we train various Models with all 30 features in it. The idea
% is to gauge the accuracy of various models for our dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n***************** Running Part 1 *****************')

% Part 1-a) Training a LDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LDAtrainedClassifier, LDA_Accuracy] = LDAtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_LDA_train,score_LDA_train] = LDAtrainedClassifier.predictFcn(X_train_norm);
[X2_LDA,Y2_LDA,T2_LDA,AUC2_LDA] = perfcurve(Y_train,score_LDA_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_LDA,score_LDA] = LDAtrainedClassifier.predictFcn(X_test_norm);
[X1_LDA,Y1_LDA,T1_LDA,AUC1_LDA] = perfcurve(Y_test,score_LDA(:,2),'M');

LDA_Accuracy_test = sum(strcmp(Y_test, Y_fit_LDA))/length(Y_test);
%
%{
% Confusion Chart
figure
confLDA = confusionchart(Y_test,Y_fit_LDA);
%}

% Part 1-b) Training a QDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[QDAtrainedClassifier, QDA_Accuracy] = QDAtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_QDA_train,score_QDA_train] = QDAtrainedClassifier.predictFcn(X_train_norm);
[X2_QDA,Y2_QDA,T2_QDA,AUC2_QDA] = perfcurve(Y_train,score_QDA_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_QDA,score_QDA] = QDAtrainedClassifier.predictFcn(X_test_norm);
[X1_QDA,Y1_QDA,T1_QDA,AUC1_QDA] = perfcurve(Y_test,score_QDA(:,2),'M');

QDA_Accuracy_test = sum(strcmp(Y_test, Y_fit_QDA))/length(Y_test);

%{
% Confusion Chart
figure
confQDA = confusionchart(Y_test,Y_fit_QDA)
%}
%
% Part 1-c) Training a GNB Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[GNBtrainedClassifier, GNB_Accuracy] = GNBtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_GNB_train,score_GNB_train]= GNBtrainedClassifier.predictFcn(X_train_norm);
[X2_GNB,Y2_GNB,T2_GNB,AUC2_GNB] = perfcurve(Y_train,score_GNB_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_GNB,score_GNB]= GNBtrainedClassifier.predictFcn(X_test_norm);
[X1_GNB,Y1_GNB,T1_GNB,AUC1_GNB] = perfcurve(Y_test,score_GNB(:,2),'M');

GNB_Accuracy_test = sum(strcmp(Y_test, Y_fit_GNB))/length(Y_test);

%{
% Confusion Chart
figure
confGNB = confusionchart(Y_test,Y_fit_GNB)
%}
%
% Part 1-d) Training a Logistic Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LogistictrainedClassifier, Logistic_Accuracy] = LogistictrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Logistic_train,score_Logistic_train] = LogistictrainedClassifier.predictFcn(X_train_norm);
[X2_Logistic,Y2_Logistic,T2_Logistic,AUC2_Logistic] = perfcurve(Y_train,score_Logistic_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_Logistic,score_Logistic] = LogistictrainedClassifier.predictFcn(X_test_norm);
[X1_Logistic,Y1_Logistic,T1_Logistic,AUC1_Logistic] = perfcurve(Y_test,score_Logistic(:,2),'M');

Logistic_Accuracy_test = sum(strcmp(Y_test, Y_fit_Logistic))/length(Y_test);

%{
% Confusion Chart
figure
confLogistic = confusionchart(Y_test,Y_fit_Logistic)
%}

%
% Part 1-e) Training a Linear SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier, Linear_SVM_Accuracy] = Linear_SVMtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Linear_SVM_train, score_Linear_SVM_train] = Linear_SVMtrainedClassifier.predictFcn(X_train_norm);
[X2_Linear_SVM,Y2_Linear_SVM,T2_Linear_SVM,AUC2_Linear_SVM] = perfcurve(Y_train,score_Linear_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Linear_SVM, score_Linear_SVM] = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
[X1_Linear_SVM,Y1_Linear_SVM,T1_Linear_SVM,AUC1_Linear_SVM] = perfcurve(Y_test,score_Linear_SVM(:,2),'M');

Linear_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Linear_SVM))/length(Y_test);



%{
% Confusion Chart
figure
conf_LinearSVM = confusionchart(Y_test,Y_fit_Linear_SVM)
%}

%
% Part 1-f) Training a Quadratic SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Quadratic_SVMtrainedClassifier, Quadratic_SVM_Accuracy] = Quadratic_SVMtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Quadratic_SVM_train,score_Quadratic_SVM_train] = Linear_SVMtrainedClassifier.predictFcn(X_train_norm);
[X2_Quadratic_SVM,Y2_Quadratic_SVM,T2_Quadratic_SVM,AUC2_Quadratic_SVM] = perfcurve(Y_train,score_Quadratic_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Quadratic_SVM,score_Quadratic_SVM] = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
[X1_Quadratic_SVM,Y1_Quadratic_SVM,T1_Quadratic_SVM,AUC1_Quadratic_SVM] = perfcurve(Y_test,score_Quadratic_SVM(:,2),'M');

Quadratic_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Quadratic_SVM))/length(Y_test);

% Confusion Chart
%{
figure
confLinearSVM = confusionchart(Y_test,Y_fit_Quadratic_SVM)
%}


% Part 1-g) Training a Gaussian SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Gaussian_SVM_trainedClassifier, Gaussian_SVM_Accuracy] = Gaussian_SVM_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Gaussian_SVM_train,score_Gaussian_SVM_train] = Gaussian_SVM_trainedClassifier.predictFcn(X_train_norm);
[X2_Gaussian_SVM,Y2_Gaussian_SVM,T2_Gaussian_SVM,AUC2_Gaussian_SVM] = perfcurve(Y_train,score_Gaussian_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Gaussian_SVM,score_Gaussian_SVM] = Gaussian_SVM_trainedClassifier.predictFcn(X_test_norm);
[X1_Gaussian_SVM,Y1_Gaussian_SVM,T1_Gaussian_SVM,AUC1_Gaussian_SVM] = perfcurve(Y_test,score_Gaussian_SVM(:,2),'M');

Gaussian_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Gaussian_SVM))/length(Y_test);






% Part 1-h) Training a Classification Tree
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[CTreetrainedClassifier, CTreeAccuracy] = Classification_Tree_trainClassifier(X_train_norm, Y_train);


% Calculating AUC for trained data
[Y_fit_Ctree_train,score_Ctree_train]= CTreetrainedClassifier.predictFcn(X_train_norm);
[X2_Ctree,Y2_Ctree,T2_Ctree,AUC2_Ctree] = perfcurve(Y_train,score_Ctree_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Ctree,score_Ctree]= CTreetrainedClassifier.predictFcn(X_test_norm);
[X1_Ctree,Y1_Ctree,T1_Ctree,AUC1_Ctree] = perfcurve(Y_test,score_Ctree(:,2),'M');

Ctree_Accuracy_test = sum(strcmp(Y_test, Y_fit_Ctree))/length(Y_test);

view(CTreetrainedClassifier.ClassificationTree,'Mode','graph')



% Part 1-i) Training a Random Forest Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[RForest_trainedClassifier, RForest_Accuracy] = RForest_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_RForest_train,score_RForest_train]= RForest_trainedClassifier.predictFcn(X_train_norm);
[X2_RForest,Y2_RForest,T2_RForest,AUC2_RForest] = perfcurve(Y_train,score_RForest_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_RForest,score_RForest]= RForest_trainedClassifier.predictFcn(X_test_norm);
[X1_RForest,Y1_RForest,T1_RForest,AUC1_RForest] = perfcurve(Y_test,score_RForest(:,2),'M');

RForest_Accuracy_test = sum(strcmp(Y_test, Y_fit_RForest))/length(Y_test);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comapiring Results with all 30 features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model_names= {'LDA', 'QDA', 'GNB', 'Logistic', 'Linear SVM', 'Quadratic SVM','Gaussian SVM','Ctree','RForest'};

% Compairing Accuracy
Accuracy_Training = [LDA_Accuracy,QDA_Accuracy,GNB_Accuracy, Logistic_Accuracy,Linear_SVM_Accuracy,Quadratic_SVM_Accuracy,Gaussian_SVM_Accuracy,CTreeAccuracy,RForest_Accuracy];
Accuracy_Testing = [LDA_Accuracy_test, QDA_Accuracy_test, GNB_Accuracy_test, Logistic_Accuracy_test, Linear_SVM_Accuracy_test, Quadratic_SVM_Accuracy_test,Gaussian_SVM_Accuracy_test,Ctree_Accuracy_test,RForest_Accuracy_test];

Table_Accuracy = [Accuracy_Training; Accuracy_Testing];
Table1 = array2table(Table_Accuracy,'VariableNames',Model_names,'RowNames',{'Training Data','Testing Data'})

% Compairing Area Under the Curve
AUC2_Training_data = [AUC2_LDA, AUC2_QDA, AUC2_GNB, AUC2_Logistic, AUC2_Linear_SVM, AUC2_Quadratic_SVM,AUC2_Gaussian_SVM,AUC2_Ctree,AUC2_RForest];
AUC1_Testing_data = [AUC1_LDA, AUC1_QDA, AUC1_GNB, AUC1_Logistic, AUC1_Linear_SVM, AUC1_Quadratic_SVM,AUC1_Gaussian_SVM,AUC1_Ctree,AUC1_RForest];


Area_Under_Curve_Table = [AUC2_Training_data;AUC1_Testing_data];
Table2 = array2table(Area_Under_Curve_Table,'VariableNames',Model_names,'RowNames',{'Training Data','Testing Data'})

% Plotting for Accuracy - for Training Data


figure
bar(Accuracy_Training)
title('Accuracy - Trained Models')
xlabel('Model')
ylabel('Accuracy')
xticklabels(Model_names)
xtickangle(45)

% Plotting for Accuracy - for Testing Data
figure
bar(Accuracy_Testing)
title('Accuracy - Out of Sample Data')
xlabel('Model')
ylabel('Accuracy')
xticklabels(Model_names)
xtickangle(45)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 2 - Feature Selection and Model Training - Quadratic Discriminant
% Model
% Comments - 
% In this section we build up on feature Selection. We try our hands from
% the most basic analysis and in each subsequent step we try to move
% towards an efficient selection of Features.
% - We start with basic filtering techniques and subsequently we build up
% on Forward subset selection 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n******************** Running Part 2 ********************')
% Part 2-1 - Filtering - Evaluating feature using p-values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h,p,ci,stat] = ttest2(X_M,X_B,'Vartype','unequal');
[ZZZZZ,featureIdxSortbyP] = sort(p,2);

figure
ecdf(p);
xlabel('P value');
ylabel('CDF value')
title('CDF of diiferent p-values')

nfs = 1:size(X_train_norm,2);


for i = 1:length(nfs)
   fs = featureIdxSortbyP(1:nfs(i));
   Mdl = fitcdiscr(X_train_norm(:,fs), Y_train,'DiscrimType','Quadratic');
   label = predict(Mdl,X_train_norm(:,fs));
   QDA_Misclassification(:,i)= sum(~strcmp(Y_train, label))/length(Y_train); % Resubs error
   
   
   label2 = predict(Mdl,X_test_norm(:,fs));
   QDA_Misclassification_testingdata(:,i) = sum(~strcmp(Y_test, label2))/length(Y_test);
   
end

figure
plot(1:length(nfs),QDA_Misclassification,'r^')
hold on
plot(1:length(nfs),QDA_Misclassification_testingdata)
xlabel('Number of Features');
ylabel('MCE')
legend({'MCE on the training set' 'MCE on the test set'});
title('Misclassification  - Training and Testing data')
hold off






% Part 2-2a)  Forward Subset Selection - QDA 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classf = @(xtrain,ytrain,xtest,ytest)sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
% fun1 = @(XT,yT,Xt,yt) sum(~strcmp(yt, predict((fitcdiscr(XT,yT,'DiscrimType','Quadratic')),Xt)))
fprintf('\n***************** Forward Subset Selection QDA *****************')
fun = @(XT,yT,Xt,yt)loss(fitcdiscr(XT,yT,'DiscrimType','Quadratic'),Xt,yt);

FivefoldCVP = cvpartition(Y_train,'kfold',5);
opts = statset('Display','iter');
%[fsLocal, history] = sequentialfs(fun,X_train_norm,Y_train,'cv',FivefoldCVP,'options',opts);


% Forward Subset Selection - Using cross Validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[fsCVfor30,historyCV] = sequentialfs(fun,X_train_norm,Y_train,'cv',FivefoldCVP,'Nf',30,'options',opts);

figure
plot(historyCV.Crit,'o');
xlabel('Number of Features');
ylabel('MCE');
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
title('Forward Selection with CV- QDA');

% Finding Minimum CV error
A = find(historyCV.Crit==min(historyCV.Crit));
Min_index_A = A(1); % If there are multiple CV error with same min value, we choose the first one
fsLocal = historyCV.In(Min_index_A,:);

% Calculating the Testing error using the model
Mdl_FS = fitcdiscr(X_train_norm(:,fsLocal), Y_train,'DiscrimType','Quadratic');

label_FS = predict(Mdl_FS,X_train_norm(:,fsLocal));
Misclassification_training_FS= sum(~strcmp(Y_train, label_FS))/length(Y_train); % Resubs error
      
label_test_FS = predict(Mdl_FS,X_test_norm(:,fsLocal));
Misclassification_test_FS= sum(~strcmp(Y_test, label_test_FS))/length(Y_test);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Part 2-2b)  Forward Subset Selection - SVM Linear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%classf = @(xtrain,ytrain,xtest,ytest)sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
%fun1 = @(XT,yT,Xt,yt) sum(~strcmp(yt, predict((fitcdiscr(XT,yT,'DiscrimType','Quadratic')),Xt)))

fun_svm = @(XT,yT,Xt,yt)loss(fitcsvm(XT,yT),Xt,yt);

%[fsLocal_svm, history_svm] = sequentialfs(fun_svm,X_train_norm,Y_train,'cv',FivefoldCVP,'options',opts);
fprintf('\n***************Forward Subset Selection Linear SVM***************')
[fsCV_svm,historyCV_svm] = sequentialfs(fun_svm,X_train_norm,Y_train,'cv',FivefoldCVP,'Nf',30,'options',opts);


figure
plot(historyCV_svm.Crit,'o');
xlabel('Number of Features');
ylabel('MCE');
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
title('Forward Selection with CV - Linear SVM');

% Finding Minimum CV error
B = find(historyCV_svm.Crit==min(historyCV_svm.Crit));
Min_index_B = B(1); % If there are multiple CV error with same min value, we choose the first one
fsLocal_svm = historyCV_svm.In(Min_index_B,:);


% Calculating the Testing error using the model
Mdl_FS_svm = fitcsvm(X_train_norm(:,fsLocal_svm), Y_train,'KernelFunction','linear');

label_FS = predict(Mdl_FS_svm,X_train_norm(:,fsLocal_svm));
Misclassification_training_FS_svm= sum(~strcmp(Y_train, label_FS))/length(Y_train); % Resubs error
      
label_test_FS = predict(Mdl_FS_svm,X_test_norm(:,fsLocal_svm));
Misclassification_test_FS_svm = sum(~strcmp(Y_test, label_test_FS))/length(Y_test);


%{
SVMModel = fitcsvm(X_train_norm(:,fsLocal_svm),Y_train);
sv = SVMModel.SupportVectors;

figure
gscatter(X_train_norm(:,2),X_train_norm(:,3),Y_train)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('Feature A','Feature B','Support Vector')
hold off
%}



% Part 2-2c)  Forward Subset Selection - SVM_Gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%classf = @(xtrain,ytrain,xtest,ytest)sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
%fun1 = @(XT,yT,Xt,yt) sum(~strcmp(yt, predict((fitcdiscr(XT,yT,'DiscrimType','Quadratic')),Xt)))
fprintf('\n************** Forward Subset Selection Gaussian SVM **************')
fun_svm = @(XT,yT,Xt,yt)loss(fitcsvm(XT,yT,'KernelFunction','gaussian'),Xt,yt)

%[fsLocal_svm_Gaussian, history_svm_Gaussian] = sequentialfs(fun_svm,X_train_norm,Y_train,'cv',FivefoldCVP,'options',opts);

[fsCV_svm_Gaussian,historyCV_svm_Gaussian] = sequentialfs(fun_svm,X_train_norm,Y_train,'cv',FivefoldCVP,'Nf',30,'options',opts);


figure
plot(historyCV_svm_Gaussian.Crit,'o');
xlabel('Number of Features');
ylabel('MCE');
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
title('Forward Selection with CV- Gaussian SVM');


% Finding Minimum CV error
C = find(historyCV_svm_Gaussian.Crit==min(historyCV_svm_Gaussian.Crit))
Min_index_C = C(1) % If there are multiple CV error with same min value, we choose the first one
fsLocal_svm_Gaussian = historyCV_svm_Gaussian.In(Min_index_C,:)



% Calculating the Testing error using the model
Mdl_FS_svm_gaussian = fitcsvm(X_train_norm(:,fsLocal_svm_Gaussian), Y_train,'KernelFunction','gaussian');
label_FS = predict(Mdl_FS_svm_gaussian,X_train_norm(:,fsLocal_svm_Gaussian));
Misclassification_training_FS_svm_gaussian= sum(~strcmp(Y_train, label_FS))/length(Y_train); % Resubs error
      
label_test_FS = predict(Mdl_FS_svm_gaussian,X_test_norm(:,fsLocal_svm_Gaussian));
Misclassification_test_FS_svm_gaussian = sum(~strcmp(Y_test, label_test_FS))/length(Y_test);


% Compairing Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Accuracy_testing_FS_Models = [(1-Misclassification_test_FS),(1-Misclassification_test_FS_svm),(1-Misclassification_test_FS_svm_gaussian)];
Acc_without_FS = [QDA_Accuracy_test,Linear_SVM_Accuracy_test,Gaussian_SVM_Accuracy_test];

Acc_comp_FS = [Accuracy_testing_FS_Models',Acc_without_FS'];

figure
bar(Acc_comp_FS)
ylim([0.9 1.02])
xticklabels({'QDA','Linear SVM','Gaussian SVM'})
legend({'Forward Selection','All Features'},'location','best','FontSize',14)
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Georgia' , 'fontsize' , 14)
title('Prediction Accuracy comparison - Forward Selection vs All features')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 3 - Feature Selection and Model Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n******************** Running Part 3 ********************')
% Doing PCA Analysis
% Finding the variability explained by the Principal Components 
[wcoeff_PCA,score_PCA,latent_PCA,tsquared_PCA,explained_PCA] = pca(X_train_norm,'VariableWeights','variance');

figure()
pareto(explained_PCA)
xlabel('Principal Component')
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
ylabel('Variance Explained (%)')



% Part 3-a) Training a LDA Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LDAtrainedClassifier_PCA, LDA_Accuracy_PCA] = LDA_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_LDA_train_PCA,score_LDA_train_PCA] = LDAtrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_LDA_PCA,Y2_LDA_PCA,T2_LDA_PCA,AUC2_LDA_PCA] = perfcurve(Y_train,score_LDA_train_PCA(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_LDA_PCA,score_LDA_PCA] = LDAtrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_LDA_PCA,Y1_LDA_PCA,T1_LDA_PCA,AUC1_LDA_PCA] = perfcurve(Y_test,score_LDA_PCA(:,2),'M');

LDA_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_LDA_PCA))/length(Y_test);



% Part 3-b) Training a QDA Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[QDAtrainedClassifier_PCA, QDA_Accuracy_PCA] = QDA_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_QDA_train_PCA,score_QDA_train_PCA] = QDAtrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_QDA_PCA,Y2_QDA_PCA,T2_QDA_PCA,AUC2_QDA_PCA] = perfcurve(Y_train,score_QDA_train_PCA(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_QDA_PCA,score_QDA_PCA] = QDAtrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_QDA_PCA,Y1_QDA_PCA,T1_QDA_PCA,AUC1_QDA_PCA] = perfcurve(Y_test,score_QDA_PCA(:,2),'M');

QDA_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_QDA_PCA))/length(Y_test);



% Part 3-c) Training a GNB Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[GNBtrainedClassifier_PCA, GNB_Accuracy_PCA] = GNB_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_GNB_train_PCA,score_GNB_train_PCA]= GNBtrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_GNB_PCA,Y2_GNB_PCA,T2_GNB_PCA,AUC2_GNB_PCA] = perfcurve(Y_train,score_GNB_train_PCA(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_GNB_PCA,score_GNB_PCA]= GNBtrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_GNB_PCA,Y1_GNB_PCA,T1_GNB_PCA,AUC1_GNB_PCA] = perfcurve(Y_test,score_GNB_PCA(:,2),'M');

GNB_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_GNB_PCA))/length(Y_test);




% Part 3-d) Training a Logistic Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LogistictrainedClassifier_PCA, Logistic_Accuracy_PCA] = Logistic_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Logistic_train_PCA,score_Logistic_train_PCA] = LogistictrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_Logistic_PCA,Y2_Logistic_PCA,T2_Logistic_PCA,AUC2_Logistic_PCA] = perfcurve(Y_train,score_Logistic_train_PCA(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_Logistic_PCA,score_Logistic_PCA] = LogistictrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_Logistic_PCA,Y1_Logistic_PCA,T1_Logistic_PCA,AUC1_Logistic_PCA] = perfcurve(Y_test,score_Logistic_PCA(:,2),'M');

Logistic_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_Logistic_PCA))/length(Y_test);



% Part 3-e) Training a Linear SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier_PCA, Linear_SVM_Accuracy_PCA] = Linear_SVM_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Linear_SVM_train_PCA, score_Linear_SVM_train_PCA] = Linear_SVMtrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_Linear_SVM_PCA,Y2_Linear_SVM_PCA,T2_Linear_SVM_PCA,AUC2_Linear_SVM_PCA] = perfcurve(Y_train,score_Linear_SVM_train_PCA(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Linear_SVM_PCA, score_Linear_SVM_PCA] = Linear_SVMtrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_Linear_SVM_PCA,Y1_Linear_SVM_PCA,T1_Linear_SVM_PCA,AUC1_Linear_SVM_PCA] = perfcurve(Y_test,score_Linear_SVM_PCA(:,2),'M');

Linear_SVM_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_Linear_SVM_PCA))/length(Y_test);



% Part 3-f) Training a Quadratic SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Quadratic_SVMtrainedClassifier_PCA, Quadratic_SVM_Accuracy_PCA] = Quadratic_SVM_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Quadratic_SVM_train_PCA,score_Quadratic_SVM_train_PCA] = Quadratic_SVMtrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_Quadratic_SVM_PCA,Y2_Quadratic_SVM_PCA,T2_Quadratic_SVM_PCA,AUC2_Quadratic_SVM_PCA] = perfcurve(Y_train,score_Quadratic_SVM_train_PCA(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Quadratic_SVM_PCA,score_Quadratic_SVM_PCA] = Quadratic_SVMtrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_Quadratic_SVM_PCA,Y1_Quadratic_SVM_PCA,T1_Quadratic_SVM_PCA,AUC1_Quadratic_SVM_PCA] = perfcurve(Y_test,score_Quadratic_SVM_PCA(:,2),'M');

Quadratic_SVM_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_Quadratic_SVM_PCA))/length(Y_test);



% Part 3-g) Training a Gaussian SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Gaussian_SVM_trainedClassifier_PCA, Gaussian_SVM_Accuracy_PCA] = Gaussian_SVM_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Gaussian_SVM_train_PCA,score_Gaussian_SVM_train_PCA] = Gaussian_SVM_trainedClassifier_PCA.predictFcn(X_train_norm);
[X2_Gaussian_SVM_PCA,Y2_Gaussian_SVM_PCA,T2_Gaussian_SVM_PCA,AUC2_Gaussian_SVM_PCA] = perfcurve(Y_train,score_Gaussian_SVM_train_PCA(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Gaussian_SVM_PCA,score_Gaussian_SVM_PCA] = Gaussian_SVM_trainedClassifier_PCA.predictFcn(X_test_norm);
[X1_Gaussian_SVM_PCA,Y1_Gaussian_SVM_PCA,T1_Gaussian_SVM_PCA,AUC1_Gaussian_SVM_PCA] = perfcurve(Y_test,score_Gaussian_SVM_PCA(:,2),'M');

Gaussian_SVM_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_Gaussian_SVM_PCA))/length(Y_test);



% Part 3-h) Training a Classification Tree Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[CTreetrainedClassifier_PCA, CTreeAccuracy_PCA] = Classification_Tree_PCA_trainClassifier(X_train_norm, Y_train);


% Calculating AUC for trained data
[Y_fit_Ctree_train_PCA,score_Ctree_train_PCA]= CTreetrainedClassifier_PCA.predictFcn(X_train_norm);
[X2_Ctree_PCA,Y2_Ctree_PCA,T2_Ctree_PCA,AUC2_Ctree_PCA] = perfcurve(Y_train,score_Ctree_train_PCA(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Ctree_PCA,score_Ctree_PCA]= CTreetrainedClassifier_PCA.predictFcn(X_test_norm);
[X1_Ctree_PCA,Y1_Ctree_PCA,T1_Ctree_PCA,AUC1_Ctree_PCA] = perfcurve(Y_test,score_Ctree_PCA(:,2),'M');

Ctree_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_Ctree_PCA))/length(Y_test);


%view(CTreetrainedClassifier_PCA.ClassificationTree,'Mode','graph')



% Part 3-i) Training a Random Forest Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[RForest_trainedClassifier_PCA, RForest_Accuracy_PCA] = RForest_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_RForest_train_PCA,score_RForest_train_PCA]= RForest_trainedClassifier_PCA.predictFcn(X_train_norm);
[X2_RForest_PCA,Y2_RForest_PCA,T2_RForest_PCA,AUC2_RForest_PCA] = perfcurve(Y_train,score_RForest_train_PCA(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_RForest_PCA,score_RForest_PCA]= RForest_trainedClassifier_PCA.predictFcn(X_test_norm);
[X1_RForest_PCA,Y1_RForest_PCA,T1_RForest_PCA,AUC1_RForest_PCA] = perfcurve(Y_test,score_RForest_PCA(:,2),'M');

RForest_Accuracy_test_PCA = sum(strcmp(Y_test, Y_fit_RForest_PCA))/length(Y_test);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comapiring Results with Principal Components
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Model_names= {'LDA', 'QDA', 'GNB', 'Logistic', 'Linear SVM', 'Quadratic SVM','Gaussian SVM','Ctree','RForest'};

% Compairing Accuracy
Accuracy_Training_PCA = [LDA_Accuracy_PCA,QDA_Accuracy_PCA,GNB_Accuracy_PCA, Logistic_Accuracy_PCA,Linear_SVM_Accuracy_PCA,Quadratic_SVM_Accuracy_PCA,Gaussian_SVM_Accuracy_PCA,CTreeAccuracy_PCA,RForest_Accuracy_PCA];
Accuracy_Testing_PCA = [LDA_Accuracy_test_PCA, QDA_Accuracy_test_PCA, GNB_Accuracy_test_PCA, Logistic_Accuracy_test_PCA, Linear_SVM_Accuracy_test_PCA, Quadratic_SVM_Accuracy_test_PCA,Gaussian_SVM_Accuracy_test_PCA,Ctree_Accuracy_test_PCA,RForest_Accuracy_test_PCA];

Table_Accuracy_PCA = [Accuracy_Training_PCA; Accuracy_Testing_PCA];
Table3 = array2table(Table_Accuracy_PCA,'VariableNames',Model_names,'RowNames',{'Training Data - PCA','Testing Data - PCA'})

% Compairing Area Under the Curve
AUC2_Training_data_PCA = [AUC2_LDA_PCA, AUC2_QDA_PCA, AUC2_GNB_PCA, AUC2_Logistic_PCA, AUC2_Linear_SVM_PCA, AUC2_Quadratic_SVM_PCA,AUC2_Gaussian_SVM_PCA,AUC2_Ctree_PCA,AUC2_RForest_PCA];
AUC1_Testing_data_PCA = [AUC1_LDA_PCA, AUC1_QDA_PCA, AUC1_GNB_PCA, AUC1_Logistic_PCA, AUC1_Linear_SVM_PCA, AUC1_Quadratic_SVM_PCA,AUC1_Gaussian_SVM_PCA,AUC1_Ctree_PCA,AUC1_RForest_PCA];


Area_Under_Curve_Table_PCA = [AUC2_Training_data_PCA;AUC1_Testing_data_PCA];
Table4 = array2table(Area_Under_Curve_Table_PCA,'VariableNames',Model_names,'RowNames',{'Training Data - PCA','Testing Data - PCA'})

% Plotting for Accuracy - for Training Data


figure
bar(Accuracy_Training_PCA)
ylabel('Accuracy')
ylim([0.8 1.02])
xticklabels(Model_names)
xtickangle(45)
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
title('Accuracy - Trained Models - Using PCA')

% Plotting for Accuracy - for Testing Data
figure
bar(Accuracy_Testing_PCA)
ylim([0.8 1.02])
ylabel('Accuracy')
xticklabels(Model_names)
xtickangle(45)
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
title('Accuracy - Out of Sample Data - Using PCA')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 4 PartA - Hyper Parameter Optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n******************** Running Part 4 ********************')

 %Hyper parameter Optimization
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hypopts = struct('ShowPlots',false,'Verbose',0); %

% For Polynomial SVM
mdls_SVM_opt = fitcsvm(X_train_norm,Y_train,'KernelFunction','linear','Standardize','on', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
results_SVM_opt = mdls_SVM_opt.HyperparameterOptimizationResults;

% For Gaussian SVM
mdls_SVM_Gaussian_opt = fitcsvm(X_train_norm,Y_train,'KernelFunction','gaussian','Standardize','on', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
results_SVM_Gaussian_opt = mdls_SVM_Gaussian_opt.HyperparameterOptimizationResults;

% For Classification Tree
mdls_tree_opt = fitctree(X_train_norm,Y_train, ...
    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', hypopts);
results_tree_opt = mdls_tree_opt.HyperparameterOptimizationResults;


figure
plot(results_SVM_opt.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
hold on
plot(results_tree_opt.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
hold on
plot(results_SVM_Gaussian_opt.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
hold off
names_Model = {'SVM-Polynomial','Decision Tree','SVM-Gaussian'};
a = get (gca, 'XTickLabel' );
set (gca, 'XTickLabel' , a, 'FontName' , 'Times' , 'fontsize' , 18)
legend(names_Model,'Location','northeast')
title('Bayesian Optimization')
xlabel('Number of Iterations')
ylabel('Minimum Objective Value')


% Checking it on Testing Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For Poly SVM
[label_SVM_opt,score_SVM_opt] = predict(mdls_SVM_opt,X_test_norm); 

% For Ctree
[label_tree_opt,score_tree_opt] = predict(mdls_tree_opt,X_test_norm);

% For Gaussian SVM
[label_SVM_Gaussian_opt,score_SVM_Gaussian_opt] = predict(mdls_SVM_Gaussian_opt,X_test_norm);


figure
subplot(1,3,1)
c_svm_opt = confusionchart(Y_test,label_SVM_opt);
title(names_Model{1})
subplot(1,3,2)
c_tree_opt = confusionchart(Y_test,label_tree_opt);
title(names_Model{2})
subplot(1,3,3)
c_svm_gaussian_opt = confusionchart(Y_test,label_SVM_Gaussian_opt);
title(names_Model{3})


% PLOTTING ROC CURVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For SVM
[X1_SVM_opt,Y1_SVM_opt,T1_SVM_opt,AUC1_SVM_opt] = perfcurve(Y_test,score_SVM_opt(:,2),'M');

% For Ctree
[X1_Ctree_opt,Y1_Ctree_opt,T1_Ctree_opt,AUC1_Ctree_opt] = perfcurve(Y_test,score_tree_opt(:,2),'M');

% For SVM Gaussian
[X1_SVM_Gaussian_opt,Y1_SVM_Gaussian_opt,T1_SVM_Gaussian_opt,AUC1_SVM_Gaussian_opt] = perfcurve(Y_test,score_SVM_Gaussian_opt(:,2),'M');


figure
plot(X1_SVM_opt,Y1_SVM_opt)
hold on
plot(X1_Ctree_opt,Y1_Ctree_opt)
hold on
plot (X1_SVM_Gaussian_opt,Y1_SVM_Gaussian_opt)
hold off
legend(names_Model,'Location','northeast')
title('ROC Curves')
xlabel('False Positive Rate')
ylabel('True Positive Rate')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 4 PartB - Hyper Parameter Optimization with PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Part 4B i) Training a Linear SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier_PCA_Opt, Linear_SVM_Accuracy_PCA_Opt] = Linear_SVM_PCA_Opt_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Linear_SVM_train_PCA_Opt, score_Linear_SVM_train_PCA_Opt] = Linear_SVMtrainedClassifier_PCA_Opt.predictFcn(X_train_norm);
[X2_Linear_SVM_PCA_Opt,Y2_Linear_SVM_PCA_Opt,T2_Linear_SVM_PCA_Opt,AUC2_Linear_SVM_PCA_Opt] = perfcurve(Y_train,score_Linear_SVM_train_PCA_Opt(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Linear_SVM_PCA_Opt, score_Linear_SVM_PCA_Opt] = Linear_SVMtrainedClassifier_PCA_Opt.predictFcn(X_test_norm);
[X1_Linear_SVM_PCA_Opt,Y1_Linear_SVM_PCA_Opt,T1_Linear_SVM_PCA_Opt,AUC1_Linear_SVM_PCA_Opt] = perfcurve(Y_test,score_Linear_SVM_PCA_Opt(:,2),'M');

Linear_SVM_Accuracy_test_PCA_Opt = sum(strcmp(Y_test, Y_fit_Linear_SVM_PCA_Opt))/length(Y_test);



% Part 4B ii) Training a Gaussian SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Gaussian_SVM_trainedClassifier_PCA_Opt, Gaussian_SVM_Accuracy_PCA_Opt] = Gaussian_SVM_PCA_Opt_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Gaussian_SVM_train_PCA_Opt,score_Gaussian_SVM_train_PCA_Opt] = Gaussian_SVM_trainedClassifier_PCA_Opt.predictFcn(X_train_norm);
[X2_Gaussian_SVM_PCA_Opt,Y2_Gaussian_SVM_PCA_Opt,T2_Gaussian_SVM_PCA_Opt,AUC2_Gaussian_SVM_PCA_Opt] = perfcurve(Y_train,score_Gaussian_SVM_train_PCA_Opt(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Gaussian_SVM_PCA_Opt,score_Gaussian_SVM_PCA_Opt] = Gaussian_SVM_trainedClassifier_PCA_Opt.predictFcn(X_test_norm);
[X1_Gaussian_SVM_PCA_Opt,Y1_Gaussian_SVM_PCA_Opt,T1_Gaussian_SVM_PCA_Opt,AUC1_Gaussian_SVM_PCA_Opt] = perfcurve(Y_test,score_Gaussian_SVM_PCA_Opt(:,2),'M');

Gaussian_SVM_Accuracy_test_PCA_Opt = sum(strcmp(Y_test, Y_fit_Gaussian_SVM_PCA_Opt))/length(Y_test);





% Part 4B iii) Training a Classification Tree Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[CTreetrainedClassifier_PCA_Opt, CTreeAccuracy_PCA_Opt] = Classification_Tree_PCA_Opt_trainClassifier(X_train_norm, Y_train);


% Calculating AUC for trained data
[Y_fit_Ctree_train_PCA_Opt,score_Ctree_train_PCA_Opt]= CTreetrainedClassifier_PCA_Opt.predictFcn(X_train_norm);
[X2_Ctree_PCA_Opt,Y2_Ctree_PCA_Opt,T2_Ctree_PCA_Opt,AUC2_Ctree_PCA_Opt] = perfcurve(Y_train,score_Ctree_train_PCA_Opt(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Ctree_PCA_Opt,score_Ctree_PCA_Opt]= CTreetrainedClassifier_PCA_Opt.predictFcn(X_test_norm);
[X1_Ctree_PCA_Opt,Y1_Ctree_PCA_Opt,T1_Ctree_PCA_Opt,AUC1_Ctree_PCA_Opt] = perfcurve(Y_test,score_Ctree_PCA_Opt(:,2),'M');

Ctree_Accuracy_test_PCA_Opt = sum(strcmp(Y_test, Y_fit_Ctree_PCA_Opt))/length(Y_test);


% view(CTreetrainedClassifier_PCA_Opt.ClassificationTree,'Mode','graph')



% Plotting graphs for Testing Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accuracy
Accuracy_PCA_Opt = [Linear_SVM_Accuracy_test_PCA_Opt,Ctree_Accuracy_test_PCA_Opt,Gaussian_SVM_Accuracy_test_PCA_Opt];


% Confusion Matrix
figure
subplot(1,3,1)
c_svm_opt_PCA = confusionchart(Y_test,Y_fit_Linear_SVM_PCA_Opt);
title(names_Model{1})
subplot(1,3,2)
c_tree_opt_PCA = confusionchart(Y_test,Y_fit_Ctree_PCA_Opt);
title(names_Model{2})
subplot(1,3,3)
c_svm_gaussian_opt_PCA = confusionchart(Y_test,Y_fit_Gaussian_SVM_PCA_Opt);
title(names_Model{3})


%}





