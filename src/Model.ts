
import * as tf from '@tensorflow/tfjs';

class Model {
  private model: tf.GraphModel | null = null;

  constructor() {
  }

  async load(path: string) {
    this.model = await tf.loadGraphModel(path);
  } 

  predict(input: tf.Tensor): number[] | null {
    if (this.model == null) {
      return null;
    }
    const model = this.model;

    const outputTensor = tf.tidy(() => {
      let outputTensor = model.predict(input) as tf.Tensor;
      return outputTensor;
    });
    const output = Array.from(outputTensor.dataSync())
    return output;
  }
}

export default Model;