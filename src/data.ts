import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { shuffle } from "./util";

export default class BrainTumorData {
  trainingPaths: { path: string; label: string }[];
  testingPaths: { path: string; label: string }[];
  desiredImageShape: { height: number; width: number; channels: number };
  allLabels: string[];
  labelMap: { [label: string]: number };

  constructor(
    trainingPath: string,
    testingPath: string,
    desiredImageShape: { height: number; width: number; channels: number }
  ) {
    const { trainingPaths, testingPaths, allLabels } = this.getPathsAndLabels(
      trainingPath,
      testingPath
    );
    this.trainingPaths = trainingPaths;
    this.testingPaths = testingPaths;
    this.desiredImageShape = desiredImageShape;
    this.allLabels = allLabels;
    this.labelMap = this.createLabelMap(allLabels);

    shuffle(this.trainingPaths);
    shuffle(this.testingPaths);
  }

  data(which: "train" | "test") {
    // "this" is not defined when tensorflow calls
    // the returned function, so we pass along all the
    // dependancies here
    const paths = which === "train" ? this.trainingPaths : this.testingPaths;
    const desiredImageShape = this.desiredImageShape;
    return function* (): Iterator<tf.TensorContainer, any, undefined> {
      for (let i = 0; i < paths.length; i++) {
        // Generate one sample at a time.
        yield tf.tidy(() => {
          let img = readImage(paths[i].path, desiredImageShape.channels);
          img = tf.image.resizeBilinear(img, [
            desiredImageShape.height,
            desiredImageShape.width,
          ]);
          img = tf.cast(img, "float32");
          img = img.div(tf.scalar(255));
          return img;
        });
      }
    };
  }

  labels(which: "train" | "test") {
    const paths = which === "train" ? this.trainingPaths : this.testingPaths;
    const labelToCategorical = this.labelToCategorical;
    const labelMap = this.labelMap;
    return function* (): Iterator<tf.TensorContainer, any, undefined> {
      for (let i = 0; i < paths.length; i++) {
        yield labelToCategorical(paths[i].label, labelMap);
      }
    };
  }

  private getPathsAndLabels(
    trainingPath: string,
    testingPath: string
  ): {
    trainingPaths: { path: string; label: string }[];
    testingPaths: { path: string; label: string }[];
    allLabels: string[];
  } {
    const trainingPaths: { path: string; label: string }[] = [];
    const testingPaths: { path: string; label: string }[] = [];
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

    for (const label of fs.readdirSync(testingPath)) {
      const labelPath = path.join(testingPath, label);
      if (!fs.lstatSync(labelPath).isDirectory()) {
        continue;
      }
      for (const file of fs.readdirSync(labelPath)) {
        const fullImagePath = path.join(labelPath, file);
        testingPaths.push({ path: fullImagePath, label });
      }
    }

    return { trainingPaths, testingPaths, allLabels };
  }

  private createLabelMap(allLabels: string[]): { [label: string]: number } {
    const labelsMap: { [label: string]: number } = {};
    for (let i = 0; i < allLabels.length; i++) {
      labelsMap[allLabels[i]] = i;
    }
    return labelsMap;
  }

  private labelToCategorical(
    label: string,
    labelsMap: { [label: string]: number }
  ) {
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
}

export function readImage(path: string, channels?: number) {
  return tf.tidy(() => {
    const imageBuffer = fs.readFileSync(path);
    const tfimage = tf.node.decodeImage(imageBuffer, channels);
    //default #channel 4
    return tfimage;
  });
}
