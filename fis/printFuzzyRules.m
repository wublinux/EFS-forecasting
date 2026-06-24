function printFuzzyRules(fis)
%PRINTFUZZYRULES Print human-readable fuzzy rule list
%   Displays all rules in the FIS with input conditions and output consequences

    inputNames = {fis.Inputs.Name};  
    outputName = fis.Outputs(1).Name; 

    for i = 1:length(fis.Rules)
        rule = fis.Rules(i); 
        antecedents = parseAntecedent(rule.Antecedent, inputNames, fis);
        consequentMF = fis.Outputs(1).MembershipFunctions(rule.Consequent).Name;
        
        fprintf('Rule %2d: IF %s \n     THEN %s is %s (Weight=%.1f)\n\n', ...
                i, antecedents, outputName, consequentMF, rule.Weight);
    end
end

function str = parseAntecedent(antecedent, inputNames, fis)
%PARSEANTECEDENT Convert antecedent array to natural language string
    parts = {}; 
    for j = 1:length(antecedent)
        if antecedent(j) == 0
            continue; 
        end
        
        inputVar = inputNames{j};
        mfIndex = antecedent(j);
        mfName = fis.Inputs(j).MembershipFunctions(mfIndex).Name;
        parts{end+1} = sprintf('%s is %s', inputVar, mfName);
    end
    str = strjoin(parts, ' AND ');
end
