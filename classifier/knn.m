function [ test_Y ] = knn( X, Y, test_X )
k = 7;
 for i = 1:length(test_X)
     for j = 1:length(X)
        dist(i,j) = norm(test_X(i)-X(j));
     end
     [B,idx] = mink(dist(i,:),k);
     result(i,:) = Y(idx);
 end
 for i = 1:length(test_X)
     if(sum(result(i,:))>=k/2)
         test_Y(i) = 1;
     else
         test_Y(i) = 0;
     end
 end
end