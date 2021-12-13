function plot_multiple_htm_results(data)
norm_inputs = data(1).inputs/max(data(1).inputs);
x = 1:length(norm_inputs);

plot_multiple(x,norm_inputs,data,1);
plot_multiple(x,norm_inputs,data,2);
plot_multiple(x,norm_inputs,data,3);

end

function plot_multiple(x,norm_inputs,data,plot_type)
anomalylikelihoodThreshold = 0.6;
figure;
n = numel(data);
for i = 1:n
     subplot(n,1,i)
     plot(x, norm_inputs, 'black');
     hold on
    plot_data(x,norm_inputs,data(i),anomalylikelihoodThreshold,plot_type);
end
end


