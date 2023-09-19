function J=CostFunction(X,Dat)
if strcmp(Dat.CostProblem,'FiterNCA')
    J=FilterNCA(X,Dat);
end

%% Write your own cost function here....
function J=FilterNCA(X,Dat)
Xpop=size(X,1);
Nvar=Dat.NVAR;
M=Dat.NOBJ;
Rel=Dat.H;
K=Nvar+1-M;
J=ones(Xpop,M);
for xpop=1:Xpop
    x=logical(X(xpop,:)>0.5);
    if sum(x)==0
        J(xpop,1)=inf;
        J(xpop,2)=inf;
        
    else
        S=find(x==1);
        J(xpop,1) = -sum(Rel(:,S));
        J(xpop,2) = sum(x(:,S)); 
    end
end
    




%% Release and bug report:
%
% November 2012: Initial release