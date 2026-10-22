/** Completed runs request extra epochs; the worker resolves the actual total
 * after checking whether the decoded dataset has changed. */
export function isAdditionalRun(config, checkpoint) {
 return config.mode==='checkpoint'&&checkpoint&&checkpoint.training.epoch>=checkpoint.training.config.epochs;
}
export function trainingPlan(config, checkpoint) {
 if(!Number.isInteger(config.epochs)||config.epochs<1||config.epochs>1000)throw new Error('EPOCHS_RANGE');
 return {config:{...config},...(isAdditionalRun(config,checkpoint)?{additionalEpochs:config.epochs}:{})};
}
