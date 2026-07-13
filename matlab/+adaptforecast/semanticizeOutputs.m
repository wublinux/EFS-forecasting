function fis = semanticizeOutputs(fis)
%SEMANTICIZEOUTPUTS Assign stable business-readable labels on [0,1].

boundaries = [0 .2 .4 .6 .8 1];
labels = ["Critical_Low", "Low", "Medium", "High", "Peak"];
seen = containers.Map("KeyType", "char", "ValueType", "double");
for mfIndex = 1:numel(fis.Outputs(1).MembershipFunctions)
    mf = fis.Outputs(1).MembershipFunctions(mfIndex);
    parameters = mf.Parameters;
    mfType = string(mf.Type);
    if mfType == "constant"
        center = parameters(1);
    elseif mfType == "linear"
        center = parameters(end);
    else
        center = mean(parameters);
    end
    group = find(center >= boundaries(1:end-1) & center <= boundaries(2:end), 1, "first");
    if isempty(group)
        baseName = "Outside_Range";
    else
        baseName = labels(group);
    end
    key = char(baseName);
    if isKey(seen, key)
        seen(key) = seen(key) + 1;
        newName = baseName + "_" + seen(key);
    else
        seen(key) = 1;
        newName = baseName;
    end
    fis.Outputs(1).MembershipFunctions(mfIndex).Name = newName;
end
end
