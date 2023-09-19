function [ACC, Precision, Recall, F1] = calculate_knnmulti(train,ytrain,test,ytest,POP)
for xpop=1:size(POP,1)
    Mdl= fitcknn(train(:,POP(xpop,:)>0.5), ytrain,'NumNeighbors',5);
    predicted = predict(Mdl, test(:,POP(xpop,:)>0.5));
    [acc_knn,precision_knn,recall_knn,f1_knn]=calculatemetrics(ytest,predicted);
    ACC(1,xpop)=acc_knn; Precision(1,xpop)=precision_knn; Recall(1,xpop)=recall_knn; F1(1,xpop)=f1_knn;
end

end
