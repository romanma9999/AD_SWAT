function data = load_htm_results_data(filename)

Array=csvread(filename,1,0);
data.inputs = Array(:, 1);
data.pred1 = Array(:, 2);
data.pred5 = Array(:, 3);
data.anomaly = Array(:,4);
data.anomalylikelihood = Array(:,5);

end
