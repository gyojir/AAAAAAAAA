
import 'materialize-css/dist/css/materialize.min.css';
import 'materialize-css/dist/js/materialize.min';
import 'material-icons/iconfont/material-icons.css';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs';
import * as faceDetection from '@tensorflow-models/face-detection';
import Model from './Model';
import { GetOneFrameSegment } from './synthesis';
import { ele, createPulseGeneratorNode, element_wise_ave, ave } from './misc';
import * as images from '../images/*.jpg';
import MicroModal from 'micromodal';

const SamplingRate = 8000;

let modelLoaded = false;
let videoAnimReq: number;

const img2spctr = new Model;
const img2f0 = new Model;
let input_data: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;
let spectrograms: number[][] = [];
let f0s: number[] = [];
let f0: number = -1;
let impulseResponse: number[] | null = null;
let isRealTimeMode = false;
const SpectrogramAverageNum = 2;
const f0AverageNum = 50;

const AudioContext = window.AudioContext || window.webkitAudioContext;
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

let detector: faceDetection.FaceDetector | null = null;

const video = ele<HTMLVideoElement>('video');
const image = ele<HTMLImageElement>('#input-image');
const canvas = ele<HTMLCanvasElement>('#input-canvas');
const trigger = ele('.trigger');

async function start() {
  MicroModal.init();

  // サンプル画像ロード
  if (Object.keys(images).length > 0) {
    await loadImage(Object.values(images)[0])
    selectInputElement('image');
    trigger.classList.add('tap-here');
  }

  // メインのクリック処理
  trigger.addEventListener('click', async (event) => {
    if (isSoundPlaying) {
      await stopSound();
      if (video.srcObject != null) {
        await video.play();
      }
    }
    else {
      if (!modelLoaded) {
        if(window.confirm("start downloading model?")){
          MicroModal.show('modal-1');
          await loadModels();
          modelLoaded = true;
        }
        else {
          return;
        }
      }

      video.pause();
      await predict();
      await playSound();
      await updateSound();
      trigger.classList.remove('tap-here');
      MicroModal.close('modal-1');
    }
  });
  
  // カメラ起動
  ele('#camera-button')?.addEventListener('click', async () => {
    await stopSound();
    await setupCam();
    selectInputElement('video');
  });

  // 画像ロード
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
}

async function setupCam() {
  if (video.srcObject != null) {
    return;
  }

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
  videoAnimReq = requestAnimationFrame(updateVideo);     
}

function stopVideo() {
  const stream = video.srcObject;
  if (!(stream instanceof MediaStream)) {
    return;
  }
  cancelAnimationFrame(videoAnimReq);
  const tracks = stream.getTracks();
  tracks.forEach(function (track) {
    track.stop();
  });
  video.srcObject = null;
}

async function loadModels() {
  await img2spctr.load('./static/img2spctr/model.json');
  await img2f0.load('./static/img2f0/model.json');
  
  const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
  const detectorConfig = {
    runtime: 'tfjs',
  } as const;
  detector = await faceDetection.createDetector(model, detectorConfig);
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

async function faceDetect(input: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): Promise<{left: number, top: number, right: number, bottom: number}[]> {
  let faces: {left: number, top: number, right: number, bottom: number, col: number[]}[] = [];

  const detections = await detector?.estimateFaces(input) || [];
  for (let detection of detections) {
    faces.push({left: detection.box.xMin, top: detection.box.yMin, right: detection.box.xMax, bottom: detection.box.yMax, col: [0, 255, 0, 255]});
  }

  if (process.env.NODE_ENV === 'development')
  {
    const src = cv.imread(input);
    for (let face of faces)
    {
      cv.rectangle(src, {x: face.left, y: face.top}, {x: face.right, y: face.bottom}, face.col);
      cv.imshow('canvasOutput', src);
    }
    src.delete();
  }

  return faces;
}

async function predict() {
  if (!input_data) {
    return;
  }

  const faces = await faceDetect(input_data);
  if (faces.length == 0) {
    faces.push({left: 0, top: 0, right: input_data.width, bottom: input_data.height});
  }

  // モデル実行
  const output = tf.tidy(() => {
    let img = tf.browser.fromPixels(input_data);
    const box = [
      faces[0].top / input_data.height,
      faces[0].left / input_data.width,
      faces[0].bottom / input_data.height,
      faces[0].right / input_data.width,
    ];
    let imgs = tf.image.cropAndResize(tf.expandDims<tf.Tensor4D>(img), [box], [0], [71,71])
    // img = tf.image.resizeBilinear(img, [71,71]);
    const inputTensor = tf.div(imgs, 255.0);

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
  if (!isRealTimeMode) {
    spectrograms.splice(0, spectrograms.length - 1);
  }
  else if (spectrograms.length > SpectrogramAverageNum) {
    spectrograms.shift();
  }
  const sp_ave = element_wise_ave(spectrograms);

  // f0移動平均
  f0s.push(output.f0);
  if (!isRealTimeMode) {
    f0s.splice(0, f0s.length - 1);
  }
  else if (f0s.length > f0AverageNum) {
    f0s.shift();
  }
  f0 = ave(f0s);
  
  // インパルス応答取得
  const response = GetOneFrameSegment(f0, SamplingRate, sp_ave, sp_ave.length);
  impulseResponse = response;
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
  
  if(isSoundPlaying || oscillator != null) {
    return;
  }

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
  isSoundPlaying = true;
};

async function stopSound() {
  oscillator?.stop();
  oscillator?.disconnect();
  processorNode?.disconnect();
  gainNode?.disconnect();
  gainNodeSwap?.disconnect();
  convolver?.disconnect();
  convolverSwap?.disconnect();
  oscillator = null;
  processorNode = null;
  gainNode = null;
  gainNodeSwap = null;
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
