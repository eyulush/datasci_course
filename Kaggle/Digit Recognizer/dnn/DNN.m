%% Kaggle Digit Recognizer

%  Instructions
%  ------------
% 

%% ======================================================================
%  STEP 0: Load the data and setup the parameters
%  Initialization
clear ; close all; clc

addpath(genpath('../lib'));
DEBUG = 0;

% define a currPhrase as 
% 1 - training 1st hidden layer autoencoder
% 2 - training 2nd hidden layer autoencoder
% 3 - training output layer
% 4 - fine tuning
% 5 - output real test result
currPhrase = 5;

% load the data
trainData = csvread('../data/train.csv',1,0);
trainLabels = trainData(:,1);             % size 42000 x 1      
trainData = trainData(:,2:end);         % size 42000 x 784
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
trainData = trainData/255;              % mapping to [0-1]

[numCases,inputSize]  = size(trainData);  
numClasses = 10;          % 10 labels, from 1 to 10   
                                      % (note that we have mapped "0" to label 10)
                                      
% try to load or init parameters
if exist('saves','dir') ~= 7
    mkdir('saves');
end

if exist('saves\dnn_parameters.mat','file')  == 2
    % load parameter, including
    % hiddenSizeL1,hiddenSizeL2, currIter, aeTheta1
    load('saves\dnn_parameters.mat');
    if stepAe1Iter == 10
        stepAe1Iter = 30;
    end    
else
    hiddenSizeL1 = 200;    % Layer 1 Hidden Size
    hiddenSizeL2 = 200;    % Layer 2 Hidden Size
    currAe1Iter = 0;
    stepAe1Iter = 10;
    currAe2Iter = 0;
    stepAe2Iter = 200;
    save('saves/dnn_parameters.mat','hiddenSizeL1','hiddenSizeL2','currAe1Iter','stepAe1Iter','currAe2Iter','stepAe2Iter');
        
    ae1Theta = initializeParameters(hiddenSizeL1, inputSize);
    ae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1); 
    save('saves/dnn_parameters.mat','ae1Theta','ae2Theta','-append');
    
    sparsityParam = 0.1;    % desired average activation of the hidden units.
                                         % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                                         %  in the lecture notes). 
    lambda = 3e-3;            % weight decay parameter       
    beta = 3;                       % weight of sparsity penalty term  
    save('saves/dnn_parameters.mat','sparsityParam','lambda','beta','-append');
end

if DEBUG == 1   % for DEBUG, use 30% data in training data as test
    propRatio = 0.7; 
    numCases= round(numCases*propRatio);
    
    testData = trainData(numCases+1:end,:);
    testLabels = trainLabels(numCases+1:end,:);
    trainData = trainData(1:numCases,:);
    trainLabels = trainLabels(1:numCases,:);
else
    testData = csvread('../data/test.csv',1,0);
    testData = testData/255;              % mapping to [0-1]
end

% finally trainData and testData should be inputSize x numCases for function
% sparseAutoencoderCost and feedforward
trainData = trainData';
testData = testData';
                                 
% STEP 1 : 1st layer autoencoder training
if currPhrase == 1 || currPhrase == 5
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
    options.maxIter = stepAe1Iter;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';

    % Self-unsupervised-training
    [ae1Theta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                                   ae1Theta, options);
                               
    % update the saved parameters
    currAe1Iter = currAe1Iter + stepAe1Iter;
    if DEBUG == 1
        save('saves/dnn_parameters','currAe1Iter','ae1Theta','-append');    
    end
    
    % 1st layer autoencoder feature extraction
    trainFeaturesL1 = feedForwardAutoencoder(ae1Theta, hiddenSizeL1, inputSize, ...
                                       trainData);
    testFeaturesL1 = feedForwardAutoencoder(ae1Theta, hiddenSizeL1, inputSize, ...
                                       testData);
    if DEBUG == 1
        save('saves/dnn_parameters','trainFeaturesL1','testFeaturesL1','-append');   
    end
    trainFeatures = trainFeaturesL1;
    testFeatures = testFeaturesL1;
end

if currPhrase == 2 || currPhrase == 5
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
    options.maxIter = stepAe2Iter;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';

    % Self-unsupervised-training
    [ae2Theta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, trainFeaturesL1), ...
                                   ae2Theta, options);
                               
    % update the saved parameters
    currAe2Iter = currAe2Iter + stepAe2Iter;
    if DEBUG == 1
        save('saves/dnn_parameters','currAe2Iter','ae2Theta','-append');    
    end
    
    % 2nd layer autoencoder feature extraction
    trainFeaturesL2 = feedForwardAutoencoder(ae2Theta, hiddenSizeL2, hiddenSizeL1, ...
                                       trainFeaturesL1);
    testFeaturesL2 = feedForwardAutoencoder(ae2Theta, hiddenSizeL2, hiddenSizeL1, ...
                                       testFeaturesL1);
    if DEBUG == 1
        save('saves/dnn_parameters','trainFeaturesL2','testFeaturesL2','-append');   
    end
    trainFeatures = trainFeaturesL2;
    testFeatures = testFeaturesL2;
end


% STEP 3: Train the softmax classifier
if currPhrase == 3 || currPhrase == 5
    trainFeatures = [trainFeaturesL1;trainFeaturesL2];
    testFeatures = [testFeaturesL1;testFeaturesL2];
end

softmax_lambda = 1e-5;       % weight decay parameter       
softmax_maxIter = 400;

options.maxIter = softmax_maxIter;
softmaxModel = softmaxTrain(size(trainFeatures,1), numClasses, softmax_lambda, ...
                            trainFeatures, trainLabels, options);  

% STEP 4: fine tuning

% STEP 4: Testing with test data
% inputData is testFeatures
[pred] = softmaxPredict(softmaxModel, testFeatures);    

if DEBUG == 1
    testAccurancy = 100*mean(pred(:) == testLabels(:)); 
        
    fprintf('Test Accuracy: %f%%\n', testAccurancy);
    save('saves/dnn_parameters.mat','testAccurancy','-append');
else
    pred(pred == 10) = 0; % Remap 10 back to 0 since our labels need to start from 1
    % write the prediction into file
    result = [[1:28000];pred]';
    
    filename = 'result.csv';
    fid = fopen(filename, 'w');
    fprintf(fid, 'ImageId,Label\n');
    fclose(fid);

    dlmwrite(filename, result, '-append');
end