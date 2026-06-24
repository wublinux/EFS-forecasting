function evaluateModel(fis, vldX, vldY)
%EVALUATEMODEL Evaluate model performance and display results
%   Calculates metrics and generates visualization plots

    % Calculate performance metrics
    [rmse, actY] = calculateRMSE(fis, vldX, vldY);
    [mae, ~] = calculateMAE(fis, vldX, vldY);
    
    fprintf('Performance Metrics:\n');
    fprintf('  RMSE: %.6f\n', rmse);
    fprintf('  MAE:  %.6f\n', mae);
    
    % Plot actual vs predicted results
    plotActualAndExpectedResultsWithMetrics(fis, vldX, vldY);
    
    % Print fuzzy rules
    fprintf('\nFuzzy Rules:\n');
    printFuzzyRules(fis);
    
    % Visualize rule activation
    plotRuleActivation(fis, vldX);
end

function [rmse, actY] = calculateRMSE(fis, x, y)
%CALCULATERMSE Calculate Root Mean Square Error
    evalOptions = evalfisOptions("EmptyOutputFuzzySetMessage","none", ...
        "NoRuleFiredMessage","none", "OutOfRangeInputValueMessage","none");

    actY = evalfis(fis, x, evalOptions);
    del = actY - y;
    rmse = sqrt(mean(del.^2));
end

function [mae, actY] = calculateMAE(fis, x, y)
%CALCULATEMAE Calculate Mean Absolute Error
    evalOptions = evalfisOptions("EmptyOutputFuzzySetMessage", "none", ...
        "NoRuleFiredMessage", "none", "OutOfRangeInputValueMessage", "none");

    actY = evalfis(fis, x, evalOptions);
    mae = mean(abs(actY - y));
end

function plotActualAndExpectedResultsWithMetrics(fis, vldX, vldY)
%PLOTACTUALANDEXPECTEDRESULTSWITHMETRICS Plot comparison of actual vs predicted
    [rmse, actY] = calculateRMSE(fis, vldX, vldY);
    [mae, ~] = calculateMAE(fis, vldX, vldY);

    figure('Name', 'Sales Prediction Results');
    plot([actY vldY], 'LineWidth', 1.5);
    axis([0 length(vldY) min(vldY)-0.01 max(vldY)+0.13]);
    xlabel('Sample Index');
    ylabel('Signal Value');
    title(['RMSE = ' num2str(rmse) ', MAE = ' num2str(mae)]);
    legend(["Predicted Output" "Actual Output"], 'Location', "northeast");
    grid on;
end

function plotRuleActivation(fis, data)
%PLOTRULEACTIVATION Visualize rule activation intensities
    N = size(data, 1);                    
    NR = numel(fis.Rules);               
    combinedFiring = zeros(N, NR);       

    for k = 1:N                         
        [~, ~, ~, ruleFiring] = evalfis(fis, data(k,:));
        combinedFiring(k,:) = sqrt(ruleFiring(:,1) .* ruleFiring(:,2)); 
    end

    figure('Name', 'Rule Activation Map');
    h = heatmap(combinedFiring, 'Colormap', parula);
    xlabel('Rule Index');
    ylabel('Sample Index');
    title('Rule Activation Intensity Map');
end
