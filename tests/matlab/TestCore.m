classdef TestCore < matlab.unittest.TestCase
    methods (Test)
        function initializesExpectedDimensions(testCase)
            fis = adaptforecast.initializeFIS(["sales_lag_1", "sales_lag_2", ...
                "sales_lag_3", "avg_temp", "humidity", "rainfall"], 2);
            testCase.verifyClass(fis, "sugfistype2");
            testCase.verifyNumElements(fis.Inputs, 6);
            testCase.verifyNumElements(fis.Outputs(1).MembershipFunctions, 64);
        end

        function metricsAreCorrect(testCase)
            metrics = adaptforecast.evaluate([10; 20], [12; 18], (1:20)', 1);
            testCase.verifyEqual(metrics.rmse, 2, AbsTol=1e-12);
            testCase.verifyEqual(metrics.mae, 2, AbsTol=1e-12);
            testCase.verifyEqual(metrics.mase, 2, AbsTol=1e-12);
        end

        function predictionShapeMatchesInput(testCase)
            fis = sugfistype2(NumInputs=1, NumInputMFs=2, NumOutputs=1, NumOutputMFs=2);
            fis = addRule(fis, [1 1 1 1; 2 2 1 1]);
            input = table(datetime(2017,3,(1:3))', [0.1; 0.5; 0.9], ...
                VariableNames=["date", "sales_lag_1"]);
            [prediction, activation] = adaptforecast.predict(fis, input, "sales_lag_1");
            testCase.verifySize(prediction, [3 2]);
            testCase.verifyHeight(activation, height(input));
            testCase.verifyWidth(activation, 1 + numel(fis.Rules));
            testCase.verifyEqual(activation.date, input.date);
            testCase.verifyEqual(string(activation.Properties.VariableNames(2:end)), ...
                "rule_" + (1:numel(fis.Rules)));
        end

        function noRuleFiringIsHandledWithoutWarningOrShapeLoss(testCase)
            fis = sugfis(Name="NoRuleFiringFixture");
            fis = addInput(fis, [0 1], Name="x");
            fis = addMF(fis, "x", "trimf", [0 .1 .2], Name="Low");
            fis = addOutput(fis, [0 1], Name="y");
            fis = addMF(fis, "y", "constant", .2, Name="LowDemand");
            fis = addRule(fis, "x==Low => y=LowDemand");
            input = table(datetime(2017, 3, 1)', .9, VariableNames=["date", "x"]);

            [prediction, activation] = adaptforecast.predict(fis, input, "x");

            testCase.verifySize(prediction, [1 2]);
            testCase.verifySize(activation, [1 2]);
            testCase.verifyEqual(activation.rule_1, 0, AbsTol=1e-12);
        end
    end
end
