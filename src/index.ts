
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import Chart from 'chart.js/auto';
import img2spctr_file from './../models/img2spctr_quantized_and_pruned.tflite';
import img2f0_file from './../models/img2f0_quantized_and_pruned.tflite';
import QuantizedModel from './QuantizedModel';
import { GetOneFrameSegment } from './synthesis';
import { ele, createPulseGeneratorNode, element_wise_ave, ave, sleep, createFileFromUrl } from './misc';
import * as images from '../images/*.jpg';

console.warn = function() {}

// tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
tflite.setWasmPath('/static/tflite/');

const SamplingRate = 8000;

const img2spctr = new QuantizedModel;
const img2f0 = new QuantizedModel;
let input_data: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;
let spectrograms: number[][] = [];
let f0s: number[] = [];
let f0: number = -1;
let impulseResponse: number[] | null = null;
let isRealTimeMode = false;
const SpectrogramAverageNum = 2;
const f0AverageNum = 50;

let chart: Chart;

window.AudioContext = window.AudioContext || window.webkitAudioContext;
let audioContext: AudioContext | null = null;
let oscillator: OscillatorNode | null = null;
let processorNode: AudioWorkletNode | null = null;
let gainNode: GainNode | null = null;
let gainNodeSwap: GainNode | null = null;
let convolver: ConvolverNode | null = null;
let convolverSwap: ConvolverNode | null = null;
let responseBuf: AudioBuffer | null = null;
let isSoundPlaying = false;

let crossFade = 0.0;
let crossFadeIntervalId: NodeJS.Timeout | null = null;

const video = ele<HTMLVideoElement>('video');
const image = ele<HTMLImageElement>('#input-image');
const canvas = ele<HTMLCanvasElement>('#input-canvas');
const trigger = ele('.trigger');

async function start() {
  await loadModels();

  if (Object.keys(images).length > 0) {
    await loadImage(Object.values(images)[0])
    selectInputElement('image');
    trigger.classList.add('tap-here');
  }

  trigger.addEventListener('click', async (event) => {
    if (isSoundPlaying) {
      await stopSound();
      video.play();
    }
    else {
      trigger.classList.remove('tap-here');
      video.pause();
      await predict();
      await playSound();
      await updateSound();
    }
  });
  
  ele('#camera-button')?.addEventListener('click', async () => {
    await stopSound();
    await setupCam();
    selectInputElement('video');
  });

  ele('#input-file').addEventListener('change', async (e: Event) => {
    if (!(e.target instanceof HTMLInputElement) || !e.target.files) {
      return;
    }
    await loadImage(window.URL.createObjectURL(e.target.files[0]));
    selectInputElement('image');
    if (isSoundPlaying) {
      await stopSound();
      await predict();
      await playSound();
      await updateSound();
    }
  });

  // (async () => {
  //   while(true) {
  //     await predict();
  //     await sleep(20);
  //   }
  // })();

  // predictIntervalId = setInterval(predict, 500);
}

async function setupCam() {
  const constraints = {
    video: {
      width: { min: 224, ideal: 224, max: 224, },
      height: { min: 224, ideal: 224, max: 224, },
    }
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await new Promise(resolve => video.onplaying = resolve);
  updateVideo();
}
  
function updateVideo() {
  canvas.getContext('2d')?.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  requestAnimationFrame(updateVideo);     
}

function stopVideo() {
  const stream = video.srcObject;
  if (!(stream instanceof MediaStream)) {
    return;
  }
  const tracks = stream.getTracks();
  tracks.forEach(function (track) {
    track.stop();
  });
  video.srcObject = null;
}

async function loadModels() {
  await img2spctr.load(img2spctr_file);
  await img2f0.load(img2f0_file);
}

async function loadImage(path: string) {
  // input_data = ele<HTMLCanvasElement>('#input-canvas');
  image.src = path;
  // const image = new Image();
  // image.onload = () => {
  //   canvas.getContext('2d')?.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvas.width, canvas.height);
  // }
  await new Promise(resolve => image.onload = resolve);
}

async function predict() {
  if (!input_data) {
    return;
  }

  // モデル実行
  const output = tf.tidy(() => {
    let img = tf.browser.fromPixels(input_data);
    img = tf.image.resizeBilinear(img, [71,71]);
    const inputTensor = tf.div(tf.expandDims(img), 255.0);

    const sp = img2spctr.predict(inputTensor);
    const f0 = img2f0.predict(inputTensor);
    if (sp == null || f0 == null) {
      return undefined;
    }

    return {sp, f0: f0[0]};
  });

  if (output === undefined) {
    return;
  }

  // スペクトラム移動平均
  const sp = output.sp.map(x => 10 ** x);
  spectrograms.push(sp);
  if (isRealTimeMode) {
    spectrograms.splice(0, spectrograms.length - 1);
  }
  else if (spectrograms.length > SpectrogramAverageNum) {
    spectrograms.shift();
  }
  const sp_ave = element_wise_ave(spectrograms);

  // f0移動平均
  f0s.push(output.f0);
  if (isRealTimeMode) {
    f0s.splice(0, f0s.length - 1);
  }
  else if (f0s.length > f0AverageNum) {
    f0s.shift();
  }
  f0 = ave(f0s);
  
  // インパルス応答取得
  const response = GetOneFrameSegment(f0, SamplingRate, sp_ave, sp_ave.length);
  impulseResponse = response;

  /*
  // グラフ描画
  const canvas = ele<HTMLCanvasElement>('#graph');
  const ctx = canvas.getContext('2d')!;
  if(chart) {
    chart.data.datasets[0].data = output.sp;
    chart.update();
  }
  else {
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: output.sp.map((_, i) => i),
        datasets: [{
          label: 'x',
          data: output.sp
        }]
      },
      options: {
        // responsive: true,
        maintainAspectRatio: false,
      }
    });
  }
  */
}

async function updateSound() {
  if (impulseResponse &&
      audioContext &&
      oscillator &&
      processorNode &&
      gainNode &&
      gainNodeSwap) {
    oscillator.frequency.setValueAtTime(f0, audioContext.currentTime);

    if (!responseBuf) {
      // インパルス応答用バッファ作成
      responseBuf = audioContext.createBuffer(1, impulseResponse.length, SamplingRate);
    }
    // インパルス応答セット
    responseBuf.getChannelData(0).set(impulseResponse);

    // 一旦接続解除
    processorNode.disconnect();

    if (convolver) {
      // 古いConvolverNodeを退避
      if (convolverSwap) {
        convolverSwap.disconnect();
        convolverSwap = null;
      }
      convolverSwap = convolver;
      convolverSwap.disconnect();
      processorNode
        .connect(convolverSwap)
        .connect(gainNodeSwap);
    }

    // ConvolverNode作成
    convolver = new ConvolverNode(audioContext);
    convolver.buffer = responseBuf;
    processorNode
      .connect(convolver)
      .connect(gainNode);

    // 数ミリ秒かけてクロスフェード
    if (convolverSwap) {
      gainNode.gain.setValueAtTime(0.0, audioContext.currentTime);
      gainNodeSwap.gain.setValueAtTime(1.0, audioContext.currentTime);
      crossFade = 0.0;
      crossFadeIntervalId && clearInterval(crossFadeIntervalId);

      const _audioContext = audioContext;
      const _gainNode = gainNode;
      const _gainNodeSwap = gainNodeSwap;
      crossFadeIntervalId = setInterval(() => {
        crossFade += 0.01;
        if (crossFade > 1.0) {
          crossFade = 1.0;
          crossFadeIntervalId && clearInterval(crossFadeIntervalId);
        }
        const gain = crossFade ** 2;
        _gainNode.gain.setValueAtTime(gain, _audioContext.currentTime);
        _gainNodeSwap.gain.setValueAtTime(1.0 - gain, _audioContext.currentTime);
      }, 1);
    }
    else {
      gainNode.gain.setValueAtTime(1.0, audioContext.currentTime);
      gainNodeSwap.gain.setValueAtTime(0.0, audioContext.currentTime);
    }
  }
}


async function playSound() {
  if (!audioContext) {
    audioContext = new AudioContext({sampleRate: SamplingRate});
  }
  if (audioContext.state === 'suspended') {
    audioContext.resume();
  }
  
  if(isSoundPlaying) {
    return;
  }  
  isSoundPlaying = true;

  oscillator = new OscillatorNode(audioContext, {type: 'square', frequency: f0});
  processorNode = await createPulseGeneratorNode(audioContext);
  gainNode = new GainNode(audioContext, {gain: 0.0});
  gainNodeSwap = new GainNode(audioContext, {gain: 0.0});

  if (!processorNode) {
    return;
  }

  oscillator
    .connect(processorNode);
  gainNode
    .connect(audioContext.destination);
  gainNodeSwap
    .connect(audioContext.destination);
  
  oscillator.start();
};

async function stopSound() {
  oscillator?.stop();
  oscillator?.disconnect();
  processorNode?.disconnect();
  gainNode?.disconnect();
  gainNodeSwap?.disconnect();
  convolver?.disconnect();
  convolverSwap?.disconnect();
  convolver = null;
  convolverSwap = null;
  isSoundPlaying = false;
}

function selectInputElement(type: 'image' | 'video') {
  if (type === 'image') {
    image.classList.remove('hide');
    canvas.classList.add('hide');
    stopVideo();
    input_data = image;
  }
  else if (type === 'video') {
    image.classList.add('hide');
    canvas.classList.remove('hide');
    input_data = canvas;
  }
}

start();
