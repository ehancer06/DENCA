% for runnig this code, please install the PlatEMO toolbox
clc;
clear;

load 'datasets'/colon_tumor.mat;
%load veri_vga16.mat;

%[~,~,H]=feat_sel_sim([train,ytrain], 'luca', 2);
%H=(H-min(H))/(max(H)-min(H));
%ntrain = rescale(train,'InputMin',min(train),'InputMax',max(train));%normalize(train,'range');
ncaMdl = fscnca(train,ytrain,'FitMethod','exact','Verbose',1, ...
    'Solver','lbfgs','IterationLimit',50);
H   = (ncaMdl.FeatureWeights)'; 
H=(H-min(H))/(max(H)-min(H));
W=0.05;
Params{1}=H;
Params{2}=W;
d=size(train,2);
% ,
options = optimoptions('particleswarm','SwarmSize',30,'MaxIterations',150,'FunctionTolerance',0.000001,'Display','iter');
fitnessfcn = @(x)FilterSz(x, Params);


PRO = UserProblem('objFcn',fitnessfcn,'N',30,'maxFE',4500,'D',d);
ALG1 = DE();


parfor i=1:30
    a=cputime; ALG1.Solve(PRO); cpuu(i)=cputime-a; 
    finalresult=ALG1.result{end,end};
    [minval,index] = min([finalresult(1,:).obj]);
    gbest = finalresult(1,index).dec;
    Best(i,:)=gbest;
    Funval(i,:)=minval;
    Dim(i)=sum(gbest>0.5);

    Mdl= fitcknn(train(:,gbest>0.5), ytrain,'NumNeighbors',5);
    predicted = predict(Mdl, test(:,gbest>0.5));
    [acc_knn,precision_knn,recall_knn,f1_knn]=calculatemetrics(ytest,predicted);
    acc(i,:)=acc_knn;
    precision(i,:)=precision_knn;
    recall(i,:)=recall_knn;
    f1(i,:)=f1_knn;
    
    Mdlsvm = fitcecoc(train(:,gbest>0.5), ytrain);%,'KernelFunction', 'rbf','KernelScale', 'auto', 'BoxConstraint', 1
    predicted = predict(Mdlsvm, test(:,gbest>0.5));
    [acc_svm,precision_svm,recall_svm,f1_svm]=calculatemetrics(ytest,predicted);
    accsvm(i,:)=acc_svm;
    precisionsvm(i,:)=precision_svm;
    recallsvm(i,:)=recall_svm;
    f1svm(i,:)=f1_svm;
end

clear train test ytrain ytest;
