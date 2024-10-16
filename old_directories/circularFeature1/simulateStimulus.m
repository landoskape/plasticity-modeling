
% Initialize Stimulus
stimulus.numFeatures = 2;
stimulus.dNudge = [0.4 0.4];
stimulus.nudgeRate = [250 250];

stimulus.dJump = [0.75 0.75];
stimulus.jumpRate = [20 20];

stimulus.maxValue = [1 1];
stimulus.minValue = [-1 -1];

stimulus.value = mean([stimulus.maxValue; stimulus.minValue],1);

% Initialize Simulation
T = 5;
dt = 0.0001;
NT = round(T/dt);
features = zeros(NT,2);
features(1,:) = stimulus.value;

for nt = 2:NT
    stimulus = changeStimulus(stimulus,dt);
    features(nt,:) = stimulus.value;
end

figure(1); clf; 
set(gcf,'units','normalized','outerposition',[0.18 0.64 0.43 0.34]);
subplot(2,2,1); plot(features(:,1),'k');
subplot(2,2,3); plot(features(:,2),'k');
subplot(2,2,[2 4]); plot(features(:,1),features(:,2),'linestyle','none','marker','o','color','k');

%
ds = 10;
cmap = varycolor(NT/ds);
figure(2); clf; hold on;
dsFeatures = dsarray(features,ds);
for i = 1:NT/ds
    plot(dsFeatures(i,1),dsFeatures(i,2),'color',cmap(i,:),'linestyle','none','marker','o');
end
