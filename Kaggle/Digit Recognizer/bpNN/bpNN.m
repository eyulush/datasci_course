%% Kaggle Digit Recognizer

%  Instructions
%  ------------
% 
%  

%% ======================================================================
%  STEP 0: Load the data and setup the parameters
%  Initialization
clear ; close all; clc

addpath(genpath('../lib'));
DEBUG = 1;

trainData = csvread('../data/train.csv',1,0);
trainLabels = trainData(:,1);             % size 42000 x 1      
trainData = trainData(:,2:end);         % size 42000 x 784
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

[numCases,inputSize]  = size(trainData);  
hiddenSize = 200;            % 25 hidden units
num_labels = 10;           % 10 labels, from 1 to 10   
                                      % (note that we have mapped "0" to label 10)

if DEBUG == 1   % for DEBUG, use 30% data in training data as test
    propRatio = 0.7; 
    numCases= round(numCases*propRatio);
    
    testData = trainData(numCases+1:end,:);
    testLabels = trainLabels(numCases+1:end,:);
    trainData = trainData(1:numCases,:);
    trainLabels = trainLabels(1:numCases,:);
else
    testData = csvread('../data/test.csv',1,0);
end

% PCA/ZCA whitening does not improve. result 93.92

% STEP 1 : Build and Train the Neural Network

%  Randomly initialize the weights
%  the hidden layer weights size hiddenSize x  (inputSize + 1)
%  the output layer weights size num_labels x (hiddenSize +1)
%  +1 is for bias
initial_Theta1 = randInitWeights(hiddenSize, inputSize);
initial_Theta2 = randInitWeights(num_labels, hiddenSize);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  lambda for weight decay
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   inputSize, ...
                                   hiddenSize, ...
                                   num_labels, trainData, trainLabels, lambda);

%  set MaxIter to 100, it can be changed to larger number. e.g. 400
options = optimset('MaxIter', 400);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

Theta2 = reshape(nn_params((1 + (hiddenSize * (inputSize + 1))):end), ...
                 num_labels, (hiddenSize + 1));

 %% Step 2 : Predict
 
pred = predict(Theta1, Theta2, testData);

if DEBUG == 1
    fprintf('\nTotal number: %f Training Set Accuracy: %f\n', length(testLabels), mean(double(pred == testLabels)) * 100);
    
    % fprintf('\nErrors \n');
    pred_error = [pred(any((pred ~= testLabels),2)),testLabels(any((pred ~= testLabels),2))];

    hist(testLabels(any((pred ~= testLabels),2)))
else
    pred(pred == 10) = 0; % Remap 10 back to 0 since our labels need to start from 1
    % write the prediction into file
    result = [[1:28000]',pred];
    
    filename = 'result.csv';
    fid = fopen(filename, 'w');
    fprintf(fid, 'ImageId,Label\n');
    fclose(fid);

    dlmwrite(filename, result, '-append');
end


                                     