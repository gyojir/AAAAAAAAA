
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import Chart from 'chart.js/auto';
import img2spctr_file from './../models/img2spctr_quantized_and_pruned.tflite';
import img2f0_file from './../models/img2f0_quantized_and_pruned.tflite';
import QuantizedModel from './QuantizedModel';
import { GetOneFrameSegment } from './synthesis';

// tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
tflite.setWasmPath('/static/tflite/');

(async () => {
  const img2spctr = new QuantizedModel();
  await img2spctr.load(img2spctr_file);
  
  const img2f0 = new QuantizedModel();
  await img2f0.load(img2f0_file);

  const {output, f0} = tf.tidy(() => {
    let img = tf.browser.fromPixels(document.querySelector('img'));
    img = tf.image.resizeBilinear(img, [71,71]);
    const inputTensor = tf.div(tf.expandDims(img), 255.0);

    const output = img2spctr.predict(inputTensor);
    const f0 = img2f0.predict(inputTensor);
    return {output, f0: f0[0]};
  });
  
  console.log(output);
  console.log(f0);

  const sp = output.map(x => 10 ** x);
  const response = GetOneFrameSegment(f0, 8000, sp, output.length);
    
  const canvas = document.querySelector('canvas');
  const ctx = canvas.getContext('2d')!;
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: response.map((_, i) => i),
      datasets: [{
        label: "x",
        data: response
      }]
    },
    options: {}
  });
})();