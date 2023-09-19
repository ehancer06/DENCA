function result=averagefront(f)
 
index=1;
for i=1:max(f(:,2))
    a= f(:,2)==i;
    display(sum(a(:)));
    i
    if sum(a(:))~=0
        result(index,:)= [mean(f(a,1)) i];
        index=index+1;
    end
end

end