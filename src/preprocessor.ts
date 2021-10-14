import * as tf from "@tensorflow/tfjs-node";

export default function imagePreprocessor(desiredImageShape: {
  height: number;
  width: number;
  channels: number;
}): (img: tf.TensorContainer) => tf.TensorContainer {
  return (img: tf.TensorContainer): tf.TensorContainer => {
    return tf.tidy(() => {
      img = tf.image.resizeBilinear(img as tf.Tensor<tf.Rank.R3>, [
        desiredImageShape.height,
        desiredImageShape.width,
      ]);
      img = tf.cast(img, "float32");
      img = img.div(tf.scalar(255));
      return img;
    });
  };
}
