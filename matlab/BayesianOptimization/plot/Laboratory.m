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
% set(groot, 'defaultAxesFontsize',7.5)
% set(groot, 'defaultTextFontsize',8.25)
% set(groot, 'defaultLegendFontSize',6.5)
%%
data1 = "LineBO_Lab_1sec/data";

data2="LineBO_descent_Lab_1sec/data";

data3="Nelder_Mead_Lab_1sec/data";

data4 = "LineBO_Lab/data";

data5="LineBO_descent_Lab/data";

data6="Nelder_Mead_Lab/data";

x=[];
Lab=1;
if ~(data1=="")&&exist("data1",'var')
    load(data1)
    data_dim1_1 = data(:,2);
    par_dim = data(:,1);
    for i = 1:size(data_dim1_1,1)
        par_temp = par_dim{i};
        data_temp = data_dim1_1{i};
        c=cellfun("isempty",data_temp);
        par_temp = par_temp(~c);
        data_temp = data_temp(~c);
        [y(1,i),I]=min(cell2mat(data_temp(end)));
        par_temp = cell2mat(par_temp(end));
        x(end+1,:) = par_temp(I,:);
    end
end

if ~(data2=="")&&exist("data2",'var')
    load(data2)
    data_dim1_2 = data(:,2);
    par_dim = data(:,1);
    for i = 1:size(data_dim1_2,1)
        par_temp = par_dim{i};
        data_temp = data_dim1_2{i};
        c=cellfun("isempty",data_temp);
        data_temp = data_temp(~c);
        par_temp = par_temp(~c);
        [y(2,i),I]=min(cell2mat(data_temp(end)));
        par_temp = cell2mat(par_temp(end));
        x(end+1,:) = par_temp(I,:);
    end
end
if ~(data3=="")&&exist("data3",'var')
    load(data3)
    data_dim1_3 = data(1:end,1);
end
if exist("data4",'var')&&~(data4=="")
    load(data4)
    data_dim2_1 = data(:,2);
    par_dim = data(:,1);
    for i = 1:size(data_dim2_1,1)
        par_temp = par_dim{i};
        data_temp = data_dim2_1{i};
        c=cellfun("isempty",data_temp);
        par_temp = par_temp(~c);
        data_temp = data_temp(~c);
        [y(1,i),I]=min(cell2mat(data_temp(end)));
        par_temp = cell2mat(par_temp(end));
        x(end+1,:) = par_temp(I,:);
    end
end
if exist("data5",'var')&&~(data5=="")
    load(data5)
    data_dim2_2 = data(:,2);
    par_dim = data(:,1);
    for i = 1:size(data_dim2_2,1)
        par_temp = par_dim{i};
        data_temp = data_dim2_2{i};
        c=cellfun("isempty",data_temp);
        data_temp = data_temp(~c);
        par_temp = par_temp(~c);
        [y(2,i),I]=min(cell2mat(data_temp(end)));
        par_temp = cell2mat(par_temp(end));
        x(end+1,:) = par_temp(I,:);
    end
end
if exist("data6",'var')&&~(data6=="")
    load(data6)
    data_dim2_3 = data(1:end,1);
end
%%

f_alpha = 0.15;
fig1=figure(1);
% fig1.Units = 'centimeters';
% fig1.Position(3:4)=[8.5,4];
ax=gca;
ax.Box='on';
ax.FontSize=10;

hold on
y = data_dim1_1;
[Y,std_Y]=getvals(y);
X=1:length(Y);
[yopt,xopt]=min(Y);
p1=plot(X,Y,'-','Color',[0 0.4470 0.7410],LineWidth=1.5);
p11 = plot(xopt,yopt,'*','Color',[0 0.4470 0.7410],'MarkerSize',10);
p12=fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0 0.4470 0.7410],'FaceAlpha',2*f_alpha,'EdgeColor','none');


y = data_dim1_2;
[Y,std_Y]=getvals(y);
X=1:length(Y);
p2=plot(X,Y,'-','Color',[0.8500 0.3250 0.0980],LineWidth=1.5);
[yopt,xopt]=min(Y);
p21 = plot(xopt,yopt,'*','Color',[0.8500 0.3250 0.0980],'MarkerSize',10);
p22 = fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0.8500 0.3250 0.0980],'EdgeColor','none');
p22.FaceAlpha = 2*f_alpha;

y = data_dim1_3;
[Y,std_Y]=getvals(y);
X=1:length(Y);
p3=plot(X,Y,'-','Color','k'	,LineWidth=1.5);
[yopt,xopt]=min(Y);
p31 = plot(xopt,yopt,'*','Color','k'	,'MarkerSize',10);
p32 = fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0 0 0],'EdgeColor','none');
p32.FaceAlpha = f_alpha;
grid on
xlim([0 130])
ax = gca;
set(gca,'TickLabelInterpreter','latex');
xlabel("Iteration $n$","Interpreter","latex")
ylabel("$$J_{opt}(n)$$ [fs]",'Interpreter','latex')


fig=figure(2);
% fig.Units = 'centimeters';
% fig.Position(3:4)=[8.5,4];
ax=gca;
ax.Box='on';
ax.FontSize=10;
% fig.Position(1)=fig1.Position(1)+fig1.Position(3)+0.8
hold on
y = data_dim2_1;
[Y,std_Y]=getvals(y);
X=1:length(Y);
p4=plot(X,Y,'-','Color',[0 0.4470 0.7410]	,LineWidth=1.5);
[yopt,xopt]=min(Y);
p41 = plot(xopt,yopt,'*','Color',[0 0.4470 0.7410]	,'MarkerSize',10);
p42 = fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0 0.4470 0.7410],'EdgeColor','none');
p42.FaceAlpha = 2*f_alpha;

y = data_dim2_2;
[Y,std_Y]=getvals(y);
X=1:length(Y);
p5=plot(X,Y,'-','Color',[0.8500 0.3250 0.0980]	,LineWidth=1.5);
[yopt,xopt]=min(Y);
p51 = plot(xopt,yopt,'*','Color',[0.8500 0.3250 0.0980]	,'MarkerSize',10);
p52=fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],[0.8500 0.3250 0.0980],'FaceAlpha',2*f_alpha,'EdgeColor','none');
y = data_dim2_3;
[Y,std_Y]=getvals(y);
X=1:length(Y);
p6=plot(X,Y,'-','Color','k'	,LineWidth=1.5);
[yopt,xopt]=min(Y);
p61 = plot(xopt,yopt,'*','Color','k'	,'MarkerSize',10);
p62 = fill([X,flip(X,2)],[Y+std_Y,flip(Y-std_Y,2)],'k','EdgeColor','none');
p62.FaceAlpha = f_alpha;
ax=gca;
ax.Box='on';
grid on
xlim([0 130])
%
if Lab
    l1=legend([p1,p2,p3],'\texttt{LineBO} random','\texttt{LineBO} descent', 'Nelder-Mead','Interpreter','latex','NumColumns',2,'Location','Northeast','fontsize',8)
    l2=legend([p4,p5,p6],'\texttt{LineBO} random','\texttt{LineBO} descent', 'Nelder-Mead','Interpreter','latex','NumColumns',2,'Location','Northeast','fontsize',8)
    ax = gca;
    set(gca,'TickLabelInterpreter','latex');
    xlabel("Iteration $n$","Interpreter","latex")
    ylabel("$$J_{opt}(n)$$ [fs]",'Interpreter','latex')
    l1.Position(1)=l1.Position(1)+0.02;
    l2.Position(1)=l2.Position(1)+0.02;
    %     ax.Box='on';
else
    ylim([10,30])
    hold off
    grid on
    xlim([0,1000])
    ylim([8,35])
    ax = gca;
    set(gca,'TickLabelInterpreter','latex');
    xlabel("iteration $n$","Interpreter","latex")
    ylabel("$$J_{opt}(n)$$ [fs]",'Interpreter','latex')
    l1=legend([p1,p2],'SafeOpt','MoSaOpt','Interpreter','latex','NumColumns',1,'Location','Northeast','fontsize',8);
    title(l1,'LineBO +','Interpreter','latex')
    a=axes('Position',get(ax,'position'),'visible','off');
    ax.Box = 'on';
    l2 = legend(a,[p4,p6],'SafeOpt','MoSaOpt','Interpreter','latex','NumColumns',1,'fontsize',8);
    title(l2,'PlaneBO + ','Interpreter','latex')
    l2.Units='centimeters';
    l1.Units='centimeters';
    l2.Position(1:2)=[l1.Position(1)-3,l1.Position(2)];
end


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
