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
            testCase.verifySize(activation, [3 3]);
        end
    end
end
