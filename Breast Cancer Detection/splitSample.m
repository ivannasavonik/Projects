function [X_trainSample, X_testSample, Y_trainSample, Y_testSample] = splitSample(X_sample, Y_sample, trainSize, permute)
% splits the sample into a training sample and a test sample
% trainSize is a number between 0 and 1 that decides which percentage of the sample is to be used for training
% permute is a boolean value such that if true then the sample is randomly permuted before being split, if false then no permutation is used

[n,d] = size(X_sample);
train_size = ceil(trainSize * n);
all_ind = [1:n];


if permute == true
    
    rng('default')
    ind_train1 = randperm(n,train_size);
    ind_test = (setdiff(randperm(n),ind_train1));
    
    
    
    X_testSample = X_sample(ind_test,:);
    X_trainSample = X_sample(ind_train1,:);
    
    Y_testSample = Y_sample(ind_test,:);
    Y_trainSample = Y_sample(ind_train1,:);
    
else
    
    ind_train1 = [1:train_size];
    ind_test = (setdiff(all_ind,ind_train1));
    
    X_testSample = X_sample(ind_test,:);
    X_trainSample = X_sample(ind_train1,:);
    
    Y_testSample = Y_sample(ind_test,:);
    Y_trainSample = Y_sample(ind_train1,:);
    

    
    
    
%{
rng('default')
ind_test = randperm(n,test_size)
n_test = length(ind_test)

ind_train = (setdiff(randperm(n),ind_test))
n_train = length(ind_train)
%}
    
end

 
