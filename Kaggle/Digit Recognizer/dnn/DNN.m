%% Kaggle Digit Recognizer

%  Instructions
%  ------------
% 

%% ======================================================================
%  STEP 0: Load the data and setup the parameters
%  Initialization
clear ; close all; clc

addpath(genpath('../lib'));
DEBUG = 1;  % if DEBUG, the formal result will not be produced, otherwise, result.csv generated.

% define a currPhrase as 
% 1 - training 1st hidden layer autoencoder
% 2 - training 2nd hidden layer autoencoder
% 3 - training output layer, maybe to get an result
% 4 - fine tuning
currPhrase = 4;

% flag for split training data for validation
validFlag = 0;

% load the data
trainData = csvread('../data/train.csv',1,0);
trainLabels = trainData(:,1);             % size 42000 x 1      
trainData = trainData(:,2:end);         % size 42000 x 784
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
trainData = trainData/255;              % mapping to [0-1]

[numCases,inputSize]  = size(trainData);  
numClasses = 10;          % 10 labels, from 1 to 10   
                                      % (note that we have mapped "0" to label 10)
                                      
% parameters
currAe1Iter = 0;
currAe2Iter = 0;
currAe4Iter = 0;
 
stepAe1Iter = 200;
stepAe2Iter = 5;
stepAe4Iter = 200;



% try to load or init parameters
if exist('saves','dir') ~= 7
    mkdir('saves');
end

if exist('saves\dnn_parameters.mat','file')  == 2
    % load parameter, including
    % hiddenSizeL1,hiddenSizeL2, currIter, aeTheta1
    load('saves\dnn_parameters.mat'); 
else
    hiddenSizeL1 = 200;    % Layer 1 Hidden Size
    hiddenSizeL2 = 200;    % Layer 2 Hidden Size
    save('saves/dnn_parameters.mat','hiddenSizeL1','hiddenSizeL2');
        
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

if validFlag == 1   % for DEBUG, use 30% data in training data as test
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

% remove the whitening operation
% if DEBUG == 1 &&  exist('saves\whitenData.mat','file')  == 2
%     load('saves\whitenData.mat');
% else
%     % extra data preprocessing, zca whitening
%     [pcaWhitenData,U] = pca_whitening([trainData,testData],1,0);
%     zcaWhitenData = U*pcaWhitenData;
%     trainData = zcaWhitenData(:,1:numCases);
%     testData = zcaWhitenData(:,numCases+1:end);
%     if DEBUG == 1
%         save('saves\whitenData.mat','trainData','testData');
%     end
% end

% STEP 1 : 1st layer autoesncoder training
if currPhrase == 1 || DEBUG == 0
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

if currPhrase == 2 || DEBUG == 0
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

% remove this combining hidden layer features
if currPhrase == 3
     trainFeatures = [trainFeaturesL1;trainFeaturesL2];
     testFeatures = [testFeaturesL1;testFeaturesL2];
end

if currPhrase == 4
     trainFeatures = trainFeaturesL2;
     testFeatures = testFeaturesL2;
end

softmax_lambda = 1e-4;       % weight decay parameter       
softmax_maxIter = 400;

options.maxIter = softmax_maxIter;
softmaxModel = softmaxTrain(size(trainFeatures,1), numClasses, softmax_lambda, ...
                            trainFeatures, trainLabels, options);  

% STEP 4: fine tuning
if currPhrase == 4
    % Initialize the stack using the parameters learned
    stack = cell(2,1);
    stack{1}.w = reshape(ae1Theta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
    stack{1}.b = ae1Theta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
    stack{2}.w = reshape(ae2Theta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
    stack{2}.b = ae1Theta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

    % Initialize the parameters for the deep model
    [stackparams, netconfig] = stack2params(stack);
    stackedAETheta = [softmaxModel.optTheta(:) ; stackparams ];

    if exist('saves/stackedAEOptTheta.mat','file') == 2
        load('saves/stackedAEOptTheta.mat');
    else
        stackedAEOptTheta = stackedAETheta;
        if DEBUG == 1
            save('saves/stackedAEOptTheta.mat', 'stackedAEOptTheta');
        end
    end
    
    % Training
    options = struct;
    options.Method = 'lbfgs';
    options.maxIter = stepAe4Iter;
    options.display = 'on';

    [stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
                         numClasses, netconfig,1e-4, trainData, trainLabels),...
                        stackedAEOptTheta,options);
                    
    save('saves/stackedAEOptTheta.mat', 'stackedAEOptTheta','-append');    
    
    currAe4Iter = currAe4Iter + stepAe4Iter;
    save('saves/dnn_parameters.mat','currAe4Iter','-append');  
end

% STEP 5: Testing with test data
% inputData is testFeatures
if currPhrase < 4
    [pred] = softmaxPredict(softmaxModel, testFeatures);    
else
    [pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);
end

if validFlag == 1
    testAccurancy = 100*mean(pred(:) == testLabels(:)); 
        
    fprintf('Test Accuracy: %f%%\n', testAccurancy);
    save('saves/dnn_parameters.mat','testAccurancy','-append');
else
    if currPhrase > 2
        pred(pred == 10) = 0; % Remap 10 back to 0 since our labels need to start from 1
        % write the prediction into file
        result = [[1:28000];pred]';
    
        filename = 'result.csv';
        fid = fopen(filename, 'w');
        fprintf(fid, 'ImageId,Label\n');
        fclose(fid);

        dlmwrite(filename, result, '-append');
    end
end