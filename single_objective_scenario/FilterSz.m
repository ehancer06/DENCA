% This function calculate trade-off between an accurasy and a dimension %
% for feature selection problem                                          %
function y = FilterSz(x, Param)

   
    x=logical(x>0.5);
    if sum(x)==0
        y=inf;
        return;
    end
    S=find(x==1);

    Rel=Param{1};
    W=Param{2};
    
    y= -W*sum(Rel(:,S))+(1-W)*sum(x)/size(x,2);
    %y= sum(Rel(:,S))/sum(x);

end

