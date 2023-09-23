
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import * as flatbuffers from 'flatbuffers';
import * as flatbuffer_tflite from './tflite_generated/tflite';

class QuantizedModel {
  private model: tflite.TFLiteModel | null = null;
  private input_quantization : {scale: number, zero_point: number};
  private output_quantization : {scale: number, zero_point: number};

  constructor() {
  }

  async load(path: string) {
    const res = await fetch(path);
    const buffer = new Uint8Array(await res.arrayBuffer());
    const flatbuf = new flatbuffers.ByteBuffer(buffer);
    const model = flatbuffer_tflite.Model.getRootAsModel(flatbuf);
    const subgraph = model.subgraphs(0);
    const inputs = subgraph?.inputs(0);
    const outputs = subgraph?.outputs(0);
    if (inputs == null || outputs == null) {
      return;
    }
    const input_params = subgraph?.tensors(inputs)?.quantization();
    const output_params = subgraph?.tensors(outputs)?.quantization();
    if (input_params == null || output_params == null) {
      return;
    }
    this.input_quantization = {scale: input_params.scale(0) || 1, zero_point: Number(input_params.zeroPoint(0))};
    this.output_quantization = {scale: output_params.scale(0) || 1, zero_point: Number(output_params.zeroPoint(0))};

    this.model = await tflite.loadTFLiteModel(buffer);
  } 

  predict(input: tf.Tensor): number[] | null {
    if (this.model == null) {
      return null;
    }
    const model = this.model;

    const outputTensor = tf.tidy(() => {
      const input_q = tf.cast(tf.add(tf.div(input, this.input_quantization.scale), this.input_quantization.zero_point), 'int32');
      let outputTensor = model.predict(input_q) as tf.Tensor;
      outputTensor = tf.mul(tf.sub(tf.cast(outputTensor, 'float32'), this.output_quantization.zero_point), this.output_quantization.scale);
      return outputTensor;
    });
    const output = Array.from(outputTensor.dataSync())
    return output;
  }
}

export default QuantizedModel;