/** Shared by production training and backend probes, so probes run the real graph. */
export function detectorForward(tf, get, input, architecture, resolution, classCount) {
  const scratch = architecture === 'tiny-grid-v1', grid = resolution / 8;
  let features = input;
  if (scratch) for (let layer = 1; layer <= 3; layer++) features = tf.relu(tf.add(tf.conv2d(features, get(`conv${layer}/kernel`), 2, 'same'), get(`conv${layer}/bias`)));
  const cells = tf.reshape(features, [-1, scratch ? 16 : 128]);
  const hidden = tf.relu(tf.add(tf.matMul(cells, get('head/hidden/kernel')), get('head/hidden/bias')));
  return tf.reshape(tf.add(tf.matMul(hidden, get('head/output/kernel')), get('head/output/bias')), [input.shape[0], grid, grid, 2, 5 + classCount]);
}

export function detectorLoss(tf, rawLogits, rawTargets, classCount) {
  // These are views of the same cells and channels. TF.js 4.22 GPU kernels
  // miscompiled rank-five broadcast arithmetic on a real RTX 5080; rank two
  // gives the identical CPU loss/gradients and correct GPU calculations.
  const logits = tf.reshape(rawLogits, [-1, 5 + classCount]), targets = tf.reshape(rawTargets, [-1, 5 + classCount]);
  const object = tf.slice(targets, [0, 4], [-1, 1]), objectLogits = tf.slice(logits, [0, 4], [-1, 1]);
  const positiveCount = tf.maximum(tf.sum(object), 1), negative = tf.sub(1, object);
  const bce = tf.sub(tf.softplus(objectLogits), tf.mul(object, objectLogits));
  // Dense detectors have hundreds of empty cells per object. Focal BCE keeps
  // easy background from dominating while penalizing hard false positives;
  // averaging background separately had made overconfident false boxes cheap.
  const probability = tf.sigmoid(objectLogits), error = tf.add(tf.mul(object, tf.sub(1, probability)), tf.mul(negative, probability));
  const alpha = tf.add(tf.mul(object, .25), tf.mul(negative, .75));
  const objectLoss = tf.div(tf.sum(tf.mul(tf.mul(alpha, tf.square(error)), bce)), positiveCount);
  const truth = tf.slice(targets, [0, 0], [-1, 4]), boxLogits = tf.slice(logits, [0, 0], [-1, 4]);
  // KL on bounded box parameters has a useful logit gradient even for tiny
  // targets. Squared sigmoid error could collapse a 4px width to nearly zero
  // while reporting a deceptively small loss.
  const safeTruth = tf.clipByValue(truth, 1e-7, 1 - 1e-7);
  const entropy = tf.neg(tf.add(tf.mul(safeTruth, tf.log(safeTruth)), tf.mul(tf.sub(1, safeTruth), tf.log(tf.sub(1, safeTruth)))));
  const boxDivergence = tf.maximum(tf.sub(tf.sub(tf.softplus(boxLogits), tf.mul(truth, boxLogits)), entropy), 0);
  const boxes = tf.div(tf.sum(tf.mul(boxDivergence, object)), tf.mul(positiveCount, 4));
  const classTruth = tf.slice(targets, [0, 5], [-1, classCount]);
  const classLogits = tf.slice(logits, [0, 5], [-1, classCount]);
  const classLoss = tf.div(tf.neg(tf.sum(tf.mul(tf.mul(classTruth, tf.logSoftmax(classLogits, -1)), object))), positiveCount);
  return tf.addN([objectLoss, tf.mul(boxes, 5), classLoss]);
}
