
%Classifiying the test images
Mdl = fitcknn(trainv, trainlab);
predicted_labels = predict(Mdl,testv);

conf = confusionmat(testlab,predicted_labels);
cm = confusionchart(testlab, predicted_labels);

%Extracting three misclassified images and three correctly classified
%images
k = 1;
num_misclassified = 0;
num_classified = 0;
while num_misclassified < 3 || num_classified < 3
    true_label = testlab(k);
    predicted_label = predicted_labels(k);
    if true_label ~= predicted_label;
        if num_misclassified == 0
            x1 = zeros(28,28); x1(:)= testv(k,:);
            x1_true_label = true_label;
            x1_predicted_label = predicted_label;
        end
        if num_misclassified == 1
            x2 = zeros(28,28); x2(:)= testv(k,:);
            x2_true_label = true_label;
            x2_predicted_label = predicted_label;
        end
        if num_misclassified == 2
            x3 = zeros(28,28); x3(:)= testv(k,:);
            x3_true_label = true_label;
            x3_predicted_label = predicted_label;
        end
        num_misclassified = num_misclassified + 1;
    end
    if predicted_label == true_label
        if num_classified == 0
            x4 = zeros(28,28); x4(:)= testv(k,:);
            x4_label = true_label;
        end
        if num_classified == 1
            x5 = zeros(28,28); x5(:)= testv(k,:);
            x5_label = true_label;
        end
        if num_classified == 2
            x6 = zeros(28,28); x6(:)= testv(k,:);
            x6_label = true_label;
        end
        num_classified = num_classified + 1;        
    end
    k = k + 1;
end

%Saving example images
imwrite(x1,'misclassified_1.png')
imwrite(x2,'misclassified_2.png')
imwrite(x3,'misclassified_3.png')
imwrite(x4,'classified_1.png')
imwrite(x5,'classified_2.png')
imwrite(x6,'classified_3.png')


%Finding error rate
error_count = 0;
for k=1:num_test
    if testlab(k) ~= predicted_labels(k)
        error_count = error_count + 1;        
    end
end
error_rate = error_count / num_test; 



