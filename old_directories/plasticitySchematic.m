
fpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/thesisFigures';

silReduction = [1.1, 1.05, 1.025, 1.00];
NSR = length(silReduction);
cmap = [zeros(NSR,2), linspace(0,1,NSR)'];

% Plasticity 
figure(3); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.63 0.54 0.22 0.34]);
line([0.95 1.05],[1,1],'color','k','linewidth',1.0);

for sra = 1:NSR
    plot(1, silReduction(sra), 'marker','o','color',cmap(sra,:),'markerfacecolor',cmap(sra,:),'markersize',12);
end
line([1.075 1.09],1.1*[1,1],'color','k','linewidth',1.5);
text(1.1, 1.1, '110%','Fontsize',24,'HorizontalAlignment','Left','VerticalAlignment','Middle');
line([1.075 1.09],1.05*[1,1],'color','k','linewidth',1.5);
text(1.1, 1.05, '105%','Fontsize',24,'HorizontalAlignment','Left','VerticalAlignment','Middle');
line([1.075 1.09],1.00*[1,1],'color','k','linewidth',1.5);
text(1.1, 1.00, '100%','Fontsize',24,'HorizontalAlignment','Left','VerticalAlignment','Middle');
text(1, 1.2,'Apical','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

line([0.95 1.05]-0.2,[1,1],'color','k','linewidth',1.0);
plot(0.8, 1.1, 'marker','o','color','k','markerfacecolor','k','markersize',12);
text(1-0.2, 1.2,'Basal','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

text(0.925, 1.3, 'Depression/Potentiation Ratio', 'Fontweight','Bold','fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

xlim([0.7 1.2]);
ylim([0.8 1.35]);

set(gca,'visible','off')
print(gcf,'-painters',fullfile(fpath,'depressionRatioCustomCorrelations'),'-djpeg');



%% -- create specific example --

fpath = '/Users/landauland/Dropbox/SabatiniLab/stdp-modeling/thesisFigures';

silReduction = [1];
NSR = length(silReduction);
cmap = zeros(100,3);%[zeros(NSR,2), linspace(0,1,NSR)'];

% Plasticity 
figure(3); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.63 0.54 0.22 0.34]);
line([0.95 1.05],[1,1],'color','k','linewidth',1.0);

for sra = 1:NSR
    plot(1, silReduction(sra), 'marker','o','color',cmap(sra,:),'markerfacecolor',cmap(sra,:),'markersize',12);
end
line([1.075 1.09],1.1*[1,1],'color','k','linewidth',1.5);
text(1.1, 1.1, '110%','Fontsize',24,'HorizontalAlignment','Left','VerticalAlignment','Middle');
line([1.075 1.09],1.05*[1,1],'color','k','linewidth',1.5);
text(1.1, 1.05, '105%','Fontsize',24,'HorizontalAlignment','Left','VerticalAlignment','Middle');
line([1.075 1.09],1.00*[1,1],'color','k','linewidth',1.5);
text(1.1, 1.00, '100%','Fontsize',24,'HorizontalAlignment','Left','VerticalAlignment','Middle');
text(1, 1.2,'Apical','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

line([0.95 1.05]-0.2,[1,1],'color','k','linewidth',1.0);
plot(0.8, 1.1, 'marker','o','color','k','markerfacecolor','k','markersize',12);
text(1-0.2, 1.2,'Basal','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

text(0.925, 1.3, 'Depression/Potentiation Ratio', 'Fontweight','Bold','fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

xlim([0.7 1.2]);
ylim([0.8 1.35]);

set(gca,'visible','off')
print(gcf,'-painters',fullfile(fpath,'OneHighOneLow'),'-djpeg');

%% -- 


fpath = '/Users/landauland/Documents/Research/o2/stdpModels/cf1BasicSilent';

NSD = 3; 
cmap = [zeros(NSD,2), linspace(0,1,NSD)'];


% Plasticity 
figure(3); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.63 0.54 0.22 0.34]);


text(1.125, 1.1, '- 110%','Fontsize',16,'HorizontalAlignment','Center','VerticalAlignment','Middle');
line([0.95 1.05]-0.2,[1,1],'color','k','linewidth',1.0);
plot(0.8, 1.1, 'marker','o','color','k','markerfacecolor','k','markersize',12);
text(1-0.2, 1.2,'Active','Fontsize',24,'Fontweight','Bold','HorizontalAlignment','Center','VerticalAlignment','Middle');

text(0.925, 1.3, 'Depression/Potentiation Ratio', 'fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

xlim([0.7 1.2]);
ylim([0.9 1.35]);

set(gca,'visible','off')
% print(gcf,'-painters',fullfile(fpath,'depressionRatioCustomCorrelationsJustActive'),'-djpeg');