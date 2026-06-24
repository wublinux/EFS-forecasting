function fisin = initializeFIS(timeLag, numInputMFs)
%INITIALIZEFIS Initialize Sugeno-type Fuzzy Inference System
%   Creates FIS structure with specified number of inputs and membership functions

    inputRange = [0 1];   
    outputRange = [0 1];  
    
    % Create Sugeno FIS type 2
    fisin = sugfistype2;  
    
    % Configure input structure
    numInputs = timeLag + 2;  % Time lag sales + temperature + humidity
    
    for i = 1:numInputs
        fisin = addInput(fisin, inputRange, 'NumMFs', numInputMFs);
        for j = 1:numInputMFs
            fisin.Inputs(i).MembershipFunctions(j).LowerScale = 1; 
            fisin.Inputs(i).MembershipFunctions(j).LowerLag = 0;   
        end
    end
    
    % Assign semantic names to inputs
    inputNames = cell(1, numInputs);
    for i = 1:timeLag
        inputNames{i} = sprintf('Lag%d_Sales', i);
    end
    inputNames{timeLag+1} = 'Temperature';
    inputNames{timeLag+2} = 'Humidity';
    
    for i = 1:numInputs
        fisin.Inputs(i).Name = inputNames{i};
    end
    
    % Configure output membership functions
    numOutputMFs = numInputMFs^numInputs; 
    fisin = addOutput(fisin, outputRange, 'NumMFs', numOutputMFs);
end
