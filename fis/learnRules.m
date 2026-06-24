function fisout1 = learnRules(fisin, trnX, trnY, config)
%LEARNRULES Learn fuzzy rules using genetic algorithm (Stage 1)
%   Optimizes rule structure using GA

    options = tunefisOptions;
    options.Method = 'ga';               
    options.OptimizationType = 'learning';
    options.NumMaxRules = config.numInputMFs^(config.timeLag + 2); 
    options.MethodOptions.PopulationSize = config.gaPopulationSize;  
    options.MethodOptions.CrossoverFraction = config.gaCrossoverFraction;
    options.MethodOptions.MaxGenerations = config.gaMaxGenerations; 
    options.UseParallel = true;          
    
    rng('default');         
    
    if config.enableTuning
        fisout1 = tunefis(fisin, [], trnX, trnY, options);
    else
        fisout1 = fisin;
    end
end
