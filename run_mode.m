clear;
clc;

datasets=["dna.mat"]; % you can add more datasets here

for dat=1:length(datasets)
load(strcat("datasets\", datasets(dat)))
ncaMdl = fscnca(train,ytrain,'FitMethod','exact','Verbose',1, ...
    'Solver','lbfgs','IterationLimit',250);
H   = (ncaMdl.FeatureWeights)'; 
H=(H-min(H))/(max(H)-min(H));
MODEDat.H = H;

%% Variables regarding the optimization problem
MODEDat.NOBJ = 2;                          % Number of objectives
MODEDat.NRES = 0;                          % Number of constraints
MODEDat.NVAR   = size(train,2);                       % Numero of decision variables
MODEDat.mop = str2func('CostFunction');    % Cost function
MODEDat.CostProblem='FiterNCA';               % Cost function instance
MODEDat.FieldD =[zeros(MODEDat.NVAR,1)...
                    ones(MODEDat.NVAR,1)]; % Initialization bounds
MODEDat.Initial=[zeros(MODEDat.NVAR,1)...
                    ones(MODEDat.NVAR,1)]; % Optimization bounds
%% Variables regarding the optimization algorithm
MODEDat.XPOP = 30;%5*MODEDat.NOBJ;             % Population size
MODEDat.Esc = 0.5;                         % Scaling factor
MODEDat.Pm= 0.2;                           % Croosover Probability
%
%% Other variables
%
MODEDat.InitialPop=[];                     % Initial population (if any)
MODEDat.MAXGEN =10000;                     % Generation bound
MODEDat.MAXFUNEVALS = 5000;%150*MODEDat.NVAR...  % Function evaluations bound
    %*MODEDat.NOBJ;                         
MODEDat.SaveResults='yes';                 % Write 'yes' if you want to 
                                           % save your results after the
                                           % optimization process;
                                           % otherwise, write 'no';
%% Initialization (don't modify)
MODEDat.CounterGEN=0;
MODEDat.CounterFES=0;
%% Put here the variables required by your code (if any).
%
%
%
%% 
%
Chromsome=[];

parfor i=1:30
t = cputime; OUT=MODE_(MODEDat); e = cputime-t; cpuu(i)=e;
POP=OUT.Xpop;
DIM=sum(OUT.Xpop>0.5,2);
[acc, precision, recall, f1] = calculate_knnmulti(train,ytrain,test,ytest,POP);
Chromsome(i,:,:)=OUT.Xpop;
Dim(i,:)=DIM; Acc(i,:)=acc; Precision(i,:)=precision; Recall(i,:)=recall; F1(i,:)=f1;
end
T=-Acc(:);
T(:,2)=Dim(:);
f = non_domination_sort_mod(T, 2, 0);
avgf = averagefront(f);
clear train test ytrain ytest;
f(:,1)=-f(:,1)*100;
avgf(:,1)=-avgf(:,1)*100;

save(strcat("results\", datasets(dat))); 
clearvars -except datasets
end


