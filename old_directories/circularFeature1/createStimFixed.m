simPrm.simLength    =  3;         % Time to simulate, (sec)
simPrm.stepSize     = .0001;        % Time step resolution, (sec)
simPrm.nTimePoints = round(simPrm.simLength/simPrm.stepSize);  % # of points in simulation
simPrm.savePoints = 50/simPrm.stepSize;
simPrm.nSaves = simPrm.nTimePoints/simPrm.savePoints;

%% Stimulus Parameters
stimulus.numFeatures = 1;
stimulus.circular = 1;

stimulus.dNudge = pi/4;
stimulus.nudgeRate = 1000;

stimulus.dJump = pi;
stimulus.jumpRate = 50;

stimulus.value = 0;

stimFixed = zeros(1,simPrm.nTimePoints);
for n = 1:simPrm.nTimePoints
   stimulus = changeStimulus(stimulus, simPrm.stepSize);
   stimFixed(n)=stimulus.value;
end

%save('stimFixed.mat','stimFixed');


%xxx = stimFixed;

%%


dT = simPrm.stepSize;

nudgeFeature = dT*stimulus.nudgeRate .* stimulus.dNudge.*randn(simPrm.nTimePoints,stimulus.numFeatures);
jumpFeature = stimulus.dJump .* randn(simPrm.nTimePoints,stimulus.numFeatures);
jumpProbability =  rand(simPrm.nTimePoints,stimulus.numFeatures)<(dT*stimulus.jumpRate);

stimulus.value = stimulus.value + cumsum(nudgeFeature + jumpFeature.*jumpProbability, 1);
plot(stimulus.value); 
