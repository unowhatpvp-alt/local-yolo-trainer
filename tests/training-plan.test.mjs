import test from 'node:test';
import assert from 'node:assert/strict';
import {trainingPlan,isAdditionalRun} from '../src/data/training-plan.js';

test('paused checkpoints preserve their absolute target without repeating completed epochs',()=>{
 const config={mode:'checkpoint',epochs:30};
 const checkpoint={training:{epoch:12,config:{epochs:30}}};
 assert.equal(isAdditionalRun(config,checkpoint),false);
 assert.deepEqual(trainingPlan(config,checkpoint),{config});
 assert.equal(config.epochs,30);
});
test('completed checkpoints add the requested extra epochs on each continuation',()=>{
 const config={mode:'checkpoint',epochs:20};
 assert.deepEqual(trainingPlan(config,{training:{epoch:30,config:{epochs:30}}}),{config,additionalEpochs:20});
 assert.deepEqual(trainingPlan(config,{training:{epoch:50,config:{epochs:50}}}),{config,additionalEpochs:20});
 assert.deepEqual(trainingPlan({...config,mode:'scratch'},null),{config:{...config,mode:'scratch'}});
});
test('invalid requested counts are rejected; actual dataset-dependent totals are resolved by the worker',()=>{
 assert.throws(()=>trainingPlan({mode:'checkpoint',epochs:1001},null),/EPOCHS_RANGE/);
 // A changed dataset resets the worker epoch to zero. Do not add the old epoch
 // on the main thread: the worker must verify the content hash first.
 assert.equal(trainingPlan({mode:'checkpoint',epochs:30},{training:{epoch:990,config:{epochs:990}}}).additionalEpochs,30);
});
