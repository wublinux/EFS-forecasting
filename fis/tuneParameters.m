function fisout2 = tuneParameters(fisout1, trnX, trnY, config)
%TUNEPARAMETERS Tune FIS parameters (Stage 2)
%   Locks certain parameters and performs parameter optimization

    % Get tunable settings with asymmetric lag enabled
    [in, out] = getTunableSettings(fisout1, 'AsymmetricLag', true);
    
    % Lock membership function parameters
    for i = 1:length(in)
        for j = 1:length(in(i).MembershipFunctions)
            in(i).MembershipFunctions(j).LowerScale.Free = false; 
            in(i).MembershipFunctions(j).LowerLag.Free = false;   
        end
    end
    
    % Configure for tuning optimization
    options = tunefisOptions;
    options.OptimizationType = 'tuning'; 
    
    rng('default'); 
    
    if config.enableTuning
        fisout2 = tunefis(fisout1, [in; out], trnX, trnY, options);
    else
        fisout2 = fisout1;
    end
end
