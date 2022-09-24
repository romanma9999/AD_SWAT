function plot_params(names, values)
channel_names =categorical(names);

plot_bar(values(1,:),channel_names, 'channel name','window length')
plot_bar(values(2,:),channel_names, 'channel name','sdr size')

end

