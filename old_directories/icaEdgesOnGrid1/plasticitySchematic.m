
fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';

NSD = 3; 
cmap = [zeros(NSD,2), linspace(0,1,NSD)'];

activeRatios = [1.1, 1.05];
silentRatios = [1.1, 1.05, 1.025];

useActive = 2; 
useSilent = 2; 

% Plasticity 
figure(3); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.63 0.54 0.22 0.34]);
line([0.95 1.05],[1,1],'color','k','linewidth',1.0);
plot(1, silentRatios(useSilent), 'marker','o','color',cmap(3,:),'markerfacecolor',cmap(3,:),'markersize',12);
text(1.125, 1.1, '- 110%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1.125, 1.05, '- 105%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1.125, 1.0, '- 100%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1, 1.2,'Silent','Fontsize',24,'Fontweight','Bold','HorizontalAlignment','Center','VerticalAlignment','Middle');

line([0.95 1.05]-0.2,[1,1],'color','k','linewidth',1.0);
plot(0.8, activeRatios(useActive), 'marker','o','color','k','markerfacecolor','k','markersize',12);
text(1-0.2, 1.2,'Active','Fontsize',24,'Fontweight','Bold','HorizontalAlignment','Center','VerticalAlignment','Middle');

text(0.925, 1.3, 'Depression/Potentiation Ratio', 'fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

xlim([0.7 1.2]);
ylim([0.9 1.35]);

set(gca,'visible','off')
print(gcf,'-painters',fullfile(fpath,sprintf('depressionRatioCustomCorrelations_AD%d_SD%d',useActive,useSilent)),'-djpeg');



%% -- 

fpath = '/Users/landauland/Documents/Research/o2/songAbbott/oriPosMoving1';

NSD = 3; 
cmap = [zeros(NSD,2), linspace(0,1,NSD)'];


% Plasticity 
figure(3); clf; hold on;
set(gcf,'units','normalized','outerposition',[0.63 0.54 0.22 0.34]);


text(1.125, 1.1, '- 110%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1.125, 1.05, '- 105%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');
text(1.125, 1.0, '- 100%','Fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

line([0.95 1.05]-0.2,[1,1],'color','k','linewidth',1.0);
plot(0.8, 1.1, 'marker','o','color','k','markerfacecolor','k','markersize',12);
text(1-0.2, 1.2,'Active','Fontsize',24,'Fontweight','Bold','HorizontalAlignment','Center','VerticalAlignment','Middle');

text(0.925, 1.3, 'Depression/Potentiation Ratio', 'fontsize',24,'HorizontalAlignment','Center','VerticalAlignment','Middle');

xlim([0.7 1.2]);
ylim([0.9 1.35]);

set(gca,'visible','off')
print(gcf,'-painters',fullfile(fpath,'depressionRatioCustomCorrelationsJustActive'),'-djpeg');


