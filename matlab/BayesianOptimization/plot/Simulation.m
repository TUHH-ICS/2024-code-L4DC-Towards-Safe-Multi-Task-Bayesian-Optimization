%---------------------------------------------------------------------------------------------
% For Paper,
% "A Safe Bayesian Optimization Algorithm for Tuning the Optical Synchronization System at European XFEL"
% by Jannis O. Lübsen, Maximilian Schütte, Sebastian Schulz, Annika Eichler
% Copyright (c) Institute of Control Systems, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Author(s): Jannis Lübsen
%--------------------------------------------------------------------------------------------

clear all
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot, 'defaultLegendInterpreter','latex')
set(groot, 'defaultTextInterpreter','latex')

%% Comparison Plot of the Simulation
y=[];
f_alpha=0.15;
load("matlab/data/output_files.mat")
data_dim1_1 = data;
% load("data_dim1_moSaOpt2.mat")
% data_dim1_2 = data;
% load("data_dim2_safeOpt.mat")
% data_dim1_3 = data;
% load("data_dim2_moSaOpt.mat")
% data_dim2_1 = data;

fig = figure(3);
fig.Units='centimeters';
fig.Position(3:end)= [8.8,5.5];
hold on
y = data_dim1_1(:,9);
[Y,std_Y]=getvals(y);
X=1:length(Y);
[yopt,xopt]=min(Y);
p1=plot(X,Y,'-','Color',[0 0.4470 0.7410],LineWidth=1.5);
p11 = plot(xopt,yopt,'*','Color',[0 0.4470 0.7410],'MarkerSize',10);
fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0 0.4470 0.7410],'FaceAlpha',2*f_alpha,'EdgeColor','none');
save("data/matlab_data/matlabSim_num_laser5","X","Y","std_Y")
% y = data_dim1_2(:,9);
% [Y,std_Y]=getvals(y);
% X=1:length(Y);
% p2=plot(X,Y,'-','Color',[0.9290 0.6940 0.1250],LineWidth=1.5);
% [yopt,xopt]=min(Y);
% p21 = plot(xopt,yopt,'*','Color',[0.9290 0.6940 0.1250],'MarkerSize',10);
% fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0.9290 0.6940 0.1250],'FaceAlpha',2*f_alpha,'EdgeColor','none');

% y = data_dim1_3(:,9);
% [Y,std_Y]=getvals(y);
% X=1:length(Y);
% p3=plot(X,Y,'--','Color',[0 0 0],LineWidth=1.5);
% [yopt,xopt]=min(Y);
% p31 = plot(xopt,yopt,'*','Color',[0 0 0],'MarkerSize',10);
% fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0 0 0],'FaceAlpha',f_alpha,'EdgeColor','none');
% 
% 
% y = data_dim2_1(:,9);
% [Y,std_Y]=getvals(y);
% X=1:length(Y);
% p4=plot(X,Y,'--','Color',[1 0 0],LineWidth=1.5);
% [yopt,xopt]=min(Y);
% p41 = plot(xopt,yopt,'*','Color',[1 0 0],'MarkerSize',10);
% fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[1 0 0],'FaceAlpha',f_alpha,'EdgeColor','none');

grid on
xlim([0,400])
ylim([10,35])
ax = gca;
set(gca,'TickLabelInterpreter','latex');
xlabel("Iteration $n$","Interpreter","latex")
ylabel("$$J_{opt}(n)$$ [fs]",'Interpreter','latex')
% l1=legend([p1,p2],'$\texttt{SafeOpt}$','$\texttt{MoSaOpt}$','Interpreter','latex','NumColumns',1);
%title(l1,'$\texttt{LineBO}$ + ','Interpreter','latex')
a=axes('Position',get(ax,'position'),'visible','off');
ax.Box = 'on';
% l2 = legend(a,[p3,p4],'$\texttt{SafeOpt}$','$\texttt{MoSaOpt}$','Interpreter','latex','NumColumns',1);
% title(l2,'$\texttt{PlaneBO}$ + ','Interpreter','latex')
% l2.Units='centimeters';
% l1.Units='centimeters';
% l2.Position(1:2)=[l1.Position(1)-2.8,l1.Position(2)];
hold off


function [Y,std_Y] = getvals(y)
yt=cell(1,length(y));
for i = 1:length(y)
    temp = y{i};
    if iscell(temp)
        temp = temp(~cellfun('isempty',temp));
        temp = temp{end};
        yt{i}=temp;
    else
        yt{i} = temp;
    end
end
len=max(cellfun('length',yt));
for i = 1:length(yt)
    temp = ones(len,1)*min(yt{i});
    temp(1:length(yt{i})) = yt{i};
    for j=1:len
        yt1(i,j)=min(temp(1:j));
    end
end
std_Y=std(yt1);
Y = mean(yt1,1);
end
