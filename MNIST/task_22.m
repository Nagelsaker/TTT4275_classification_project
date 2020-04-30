%problem 2

%Dividing the set of training vectors into a sets containing only training
%vectors from one class. 
trainv_0 = zeros(8000,vec_size);
trainv_1 = zeros(8000,vec_size);
trainv_2 = zeros(8000,vec_size);
trainv_3 = zeros(8000,vec_size);
trainv_4 = zeros(8000,vec_size);
trainv_5 = zeros(8000,vec_size);
trainv_6 = zeros(8000,vec_size);
trainv_7 = zeros(8000,vec_size);
trainv_8 = zeros(8000,vec_size);
trainv_9 = zeros(8000,vec_size);
row_0 = 1;
row_1 = 1;
row_2 = 1;
row_3 = 1;
row_4 = 1;
row_5 = 1;
row_6 = 1;
row_7 = 1;
row_8 = 1;
row_9 = 1;


for k=1:num_test
    if trainlab(k)== 0
        trainv_0(row_0,:) = trainv(k,:);
        row_0 = row_0 + 1;
    end
    if trainlab(k)== 1
        trainv_1(row_1,:) = trainv(k,:);
        row_1 = row_1 + 1;
    end
    if trainlab(k)== 2
        trainv_2(row_2,:) = trainv(k,:);
        row_2 = row_2 + 1;
    end
    if trainlab(k)== 3
        trainv_3(row_3,:) = trainv(k,:);
        row_3 = row_3 + 1;
    end
    if trainlab(k)== 4
        trainv_4(row_4,:) = trainv(k,:);
        row_4 = row_4 + 1;
    end
    if trainlab(k)== 5
        trainv_5(row_5,:) = trainv(k,:);
        row_5 = row_5 + 1;
    end
    if trainlab(k)== 6
        trainv_6(row_6,:) = trainv(k,:);
        row_6 = row_6 + 1;
    end
    if trainlab(k)== 7
        trainv_7(row_7,:) = trainv(k,:);
        row_7 = row_7 + 1;
    end
    if trainlab(k)== 8
        trainv_8(row_8,:) = trainv(k,:);
        row_8 = row_8 + 1;
    end
    if trainlab(k)== 9
        trainv_9(row_9,:) = trainv(k,:);
        row_9 = row_9 + 1;
    end
end

trainv_0 = trainv_0(1:row_0,:);
trainv_1 = trainv_1(1:row_1,:);
trainv_2 = trainv_2(1:row_2,:);
trainv_3 = trainv_3(1:row_3,:);
trainv_4 = trainv_4(1:row_4,:);
trainv_5 = trainv_5(1:row_5,:);
trainv_6 = trainv_6(1:row_6,:);
trainv_7 = trainv_7(1:row_7,:);
trainv_8 = trainv_8(1:row_8,:);
trainv_9 = trainv_9(1:row_9,:);

%Clustering of training vectors
M = 64;
[idx_0,C_0] = kmeans(trainv_0,M);
[idx_1,C_1] = kmeans(trainv_1,M);
[idx_2,C_2] = kmeans(trainv_2,M);
[idx_3,C_3] = kmeans(trainv_3,M);
[idx_4,C_4] = kmeans(trainv_4,M);
[idx_5,C_5] = kmeans(trainv_5,M);
[idx_6,C_6] = kmeans(trainv_6,M);
[idx_7,C_7] = kmeans(trainv_7,M);
[idx_8,C_8] = kmeans(trainv_8,M);
[idx_9,C_9] = kmeans(trainv_9,M);

C = [C_0;C_1;C_2;C_3;C_4;C_5;C_6;C_7;C_8;C_9];
trainlab_2 = [0*ones(M,1);1*ones(M,1);2*ones(M,1);3*ones(M,1);4*ones(M,1);5*ones(M,1);
            6*ones(M,1);7*ones(M,1);8*ones(M,1);9*ones(M,1)];

%Classifying the test images
Mdl = fitcknn(C, trainlab_2);
%Mdl = fitcknn(C, trainlab_2,'NumNeighbors',7); %Uncomment this to use
%KNN-classifier
predicted_labels = predict(Mdl,testv);
conf = confusionmat(testlab,predicted_labels);
cm = confusionchart(testlab,predicted_labels);

%Findning error rate
error_count = 0;
for k=1:num_test
    if testlab(k) ~= predicted_labels(k)
        error_count = error_count + 1;        
    end
end
error_rate = error_count / num_test;