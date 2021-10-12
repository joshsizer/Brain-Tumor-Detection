import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { shuffle } from "./util";

export function readImage(path: string) {
  const imageBuffer = fs.readFileSync(path);
  const tfimage = tf.node.decodeImage(imageBuffer);
  //default #channel 4
  return tfimage;
}

const dataPath = path.join(__dirname, "../data");
const trainingPath = path.join(dataPath, "Training");
const testPath = path.join(dataPath, "Testing");
testPath;

const trainingPaths: { path: string; label: string }[] = [];
const allLabels = [];

for (const label of fs.readdirSync(trainingPath)) {
  const labelPath = path.join(trainingPath, label);
  if (!fs.lstatSync(labelPath).isDirectory()) {
    continue;
  }
  allLabels.push(label);
  for (const file of fs.readdirSync(labelPath)) {
    const fullImagePath = path.join(labelPath, file);
    trainingPaths.push({ path: fullImagePath, label });
  }
}

const labelsMap: { [label: string]: number } = {};
for (let i = 0; i < allLabels.length; i++) {
  labelsMap[allLabels[i]] = i;
}

shuffle(trainingPaths);

function labelToCategorical(label: string) {
  const toRet = [];
  for (const l in labelsMap) {
    if (l === label) {
      toRet.push(1.0);
    } else {
      toRet.push(0.0);
    }
  }
  return tf.tensor(toRet, undefined, "float32");
}

export function* data() {
  for (let i = 0; i < trainingPaths.length; i++) {
    // Generate one sample at a time.
    let img = readImage(trainingPaths[i].path);
    img = tf.image.resizeBilinear(img, [128, 128]);
    img = tf.cast(img, "float32");
    img = img.div(tf.scalar(255));
    const shape = img.shape;
    const desiredShape = tf.zeros([128, 128, 3]).shape;
    let areShapesEqual = true;
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] !== desiredShape[i]) {
        areShapesEqual = false;
        break;
      }
    }

    if (areShapesEqual) {
      yield img;
    } else {
      continue;
    }
    // let img = readImage(trainingPaths[i].path);
    // img = tf.cast(img, "float32");
    // img = tf.image.resizeBilinear(tfImage, [128, 128]).div(tf.scalar(255));
    // yield img;
  }
}

export function* labels() {
  for (let i = 0; i < trainingPaths.length; i++) {
    // Generate one sample at a time.
    let img = readImage(trainingPaths[i].path);
    img = tf.image.resizeBilinear(img, [128, 128]);
    img = tf.cast(img, "float32");
    img = img.div(tf.scalar(255));
    const shape = img.shape;
    const desiredShape = tf.zeros([128, 128, 3]).shape;
    let areShapesEqual = true;
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] !== desiredShape[i]) {
        areShapesEqual = false;
        break;
      }
    }
    if (areShapesEqual) {
      yield labelToCategorical(trainingPaths[i].label);
    } else {
      continue;
    }

    yield labelToCategorical(trainingPaths[i].label);
  }
}
