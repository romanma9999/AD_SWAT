function plot_multiple_htm_results(data,P)
norm_inputs = data(1).inputs/max(data(1).inputs);
x = P.Time;

plot_multiple(x,norm_inputs,data);

end

function plot_multiple(x,norm_inputs,data)
anomalylikelihoodThreshold = 0.6;
figure;
n = numel(data);
axes_arr = []
for i = 1:n
      axes_arr(i) = subplot(n,1,i);
      yyaxis left
      plot(x, norm_inputs, 'black','LineWidth',2);
      grid on;
      hold on
      yyaxis right
      plot_data(x,data(i),anomalylikelihoodThreshold,data(i).plot_type);
end
linkaxes(axes_arr,'x');
end


