function fisout3 = advancedTuning(fisout2, trnX, trnY, config)
%ADVANCEDTUNING Perform advanced tuning with asymmetric lag (Stage 3)
%   Enables asymmetric lag parameters for fine-tuning

    % Get tunable settings
    [in, ~] = getTunableSettings(fisout2, 'AsymmetricLag', true);
    
    % Configure membership functions for advanced tuning
    for i = 1:length(in)
        for j = 1:length(in(i).MembershipFunctions)
            in(i).MembershipFunctions(j).UpperParameters.Free = false; 
            in(i).MembershipFunctions(j).LowerScale.Free = true;  
            in(i).MembershipFunctions(j).LowerLag.Free = true;    
        end
    end
    
    % Configure for tuning optimization
    options = tunefisOptions;
    options.OptimizationType = 'tuning'; 
    
    rng('default'); 
    
    if config.enableTuning
        fisout3 = tunefis(fisout2, in, trnX, trnY, options);
    else
        fisout3 = fisout2;
    end
end
