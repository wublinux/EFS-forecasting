function fis = initializeFIS(featureNames, numInputMFs)
%INITIALIZEFIS Create an interval type-2 Sugeno FIS for normalized inputs.

arguments
    featureNames (1,:) string
    numInputMFs (1,1) double {mustBeInteger,mustBePositive} = 2
end

fis = sugfistype2(Name="AdaptForecast");
for inputIndex = 1:numel(featureNames)
    fis = addInput(fis, [0 1], Name=featureNames(inputIndex), NumMFs=numInputMFs);
    for mfIndex = 1:numInputMFs
        fis.Inputs(inputIndex).MembershipFunctions(mfIndex).LowerScale = 1;
        fis.Inputs(inputIndex).MembershipFunctions(mfIndex).LowerLag = 0;
        if numInputMFs == 2
            labels = ["Low", "High"];
            fis.Inputs(inputIndex).MembershipFunctions(mfIndex).Name = labels(mfIndex);
        else
            fis.Inputs(inputIndex).MembershipFunctions(mfIndex).Name = "Level_" + mfIndex;
        end
    end
end

maxRules = numInputMFs ^ numel(featureNames);
fis = addOutput(fis, [0 1], Name="Demand", NumMFs=maxRules);
end

