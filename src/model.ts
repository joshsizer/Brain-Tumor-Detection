import * as tf from "@tensorflow/tfjs-node";

/**
 * Define a convolutional neural network by extending a
 * tf.Sequential model. Observe the constructor for network's
 * architecture.
 */
export default class BrainTumorModel extends tf.Sequential {
  constructor(inputImageShape: {
    height: number;
    width: number;
    channels: number;
  }) {
    super();

    const { height, width, channels } = inputImageShape;

    super.add(
      tf.layers.conv2d({
        inputShape: [width, height, channels],
        kernelSize: 5,
        filters: 4,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );

    super.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    super.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );

    super.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    super.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling",
      })
    );

    super.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    super.add(tf.layers.flatten());

    const NUM_OUTPUT_CLASSES = 4;
    super.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: "softmax",
      })
    );

    const optimizer = tf.train.adam();
    super.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }
}
