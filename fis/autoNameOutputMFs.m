function fis = autoNameOutputMFs(fis)
%AUTONAMEOUTPUTMFS Automatically name output membership functions
%   Assigns semantic names to output MFs based on their center values

    % Define output groups with ranges and names
    outputGroups = {
        [0.0, 0.2],   'Critical_Low';
        [0.2, 0.4],   'Low';
        [0.4, 0.6],   'Medium';
        [0.6, 0.8],   'High';
        [0.8, 1.0],   'Peak'
    };

    outputMFs = fis.Outputs(1).MembershipFunctions;
    centers = zeros(1, length(outputMFs));
    
    % Calculate center for each membership function
    for i = 1:length(outputMFs)
        params = outputMFs(i).Parameters;
        switch outputMFs(i).Type
            case 'trimf'
                centers(i) = params(2);
            case 'gaussmf'
                centers(i) = params(1);
            case 'trapmf'
                centers(i) = mean(params(2:3));
            case 'constant'
                centers(i) = params(1);
            otherwise
                centers(i) = mean(params);
        end
    end

    % Assign names based on center value ranges
    nameCounter = containers.Map(); 
    for i = 1:length(outputMFs)
        groupName = '';
        
        % Find appropriate group for this MF
        for g = 1:size(outputGroups, 1)
            if centers(i) >= outputGroups{g, 1}(1) && centers(i) <= outputGroups{g, 1}(2)
                groupName = outputGroups{g, 2};
                break;
            end
        end
        
        % Handle edge cases
        if isempty(groupName)
            groupName = 'Unknown';
        end
        
        % Ensure unique names
        if ~isKey(nameCounter, groupName)
            nameCounter(groupName) = 1;
            newName = groupName;
        else
            nameCounter(groupName) = nameCounter(groupName) + 1;
            newName = sprintf('%s_%d', groupName, nameCounter(groupName));
        end
        
        fis.Outputs(1).MembershipFunctions(i).Name = newName;
    end
end
