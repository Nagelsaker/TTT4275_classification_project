
tic
Mdl = fitcknn(trainv, trainlab);
predicted_labels = predict(Mdl,testv);

conf = confusionmat(testlab,predicted_labels);

k = 1;
num_misclassified = 0;
num_classified = 0;
while num_misclassified < 3 && num_classified < 3
    true_label = testlab(k);
    predicted_label = predicted_labels(k);
    if true_label ~= predicted_label;
        if num_misclassified == 0
            x1 = zeros(28,28); x1(:)= testv(k,:);
        end
        if num_misclassified == 1
            x2 = zeros(28,28); x2(:)= testv(k,:);
        end
        if num_misclassified == 2
            x3 = zeros(28,28); x3(:)= testv(k,:);
        end
        num_misclassified = num_misclassified + 1;
    end
    if predicted_label == true_label
        if num_classified == 0
            x4 = zeros(28,28); x4(:)= testv(k,:);
        end
        if num_classified == 1
            x5 = zeros(28,28); x5(:)= testv(k,:);
        end
        if num_classified == 2
            x6 = zeros(28,28); x6(:)= testv(k,:);
        end
        num_classified = num_classified + 1;        
    end
    k = k + 1;
end

imwrite(x1,'misclassified_1.png')
imwrite(x2,'misclassified_2.png')
imwrite(x3,'misclassified_3.png')
imwrite(x4,'classified_1.png')
imwrite(x5,'classified_2.png')
imwrite(x6,'classified_3.png')

error_count = 0;
for k=1:num_test
    if testlab(k) ~= predicted_labels(k)
        error_count = error_count + 1;        
    end
end
error_rate = error_count / num_test; 
toc


