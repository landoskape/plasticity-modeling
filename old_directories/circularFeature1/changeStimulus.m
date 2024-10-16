function stimulus = changeStimulus(stimulus, dT)

nudgeFeature = dT*stimulus.nudgeRate .* stimulus.dNudge.*randn(1,stimulus.numFeatures);
jumpFeature = stimulus.dJump .* randn(1,stimulus.numFeatures);
jumpProbability =  rand(1,stimulus.numFeatures)<(dT*stimulus.jumpRate);

stimulus.value = stimulus.value + (nudgeFeature + jumpFeature.*jumpProbability);
  
