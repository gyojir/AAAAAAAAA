import workletUrl from 'worklet:./pulse-generator.ts';

export async function createPulseGeneratorNode(audioContext: AudioContext) {
  let processorNode: AudioWorkletNode;
  try {
    processorNode = new AudioWorkletNode(audioContext, "pulse-generator");
  } catch (e) {
    try {
      await audioContext.audioWorklet.addModule(workletUrl);
      processorNode = new AudioWorkletNode(audioContext, "pulse-generator");
    } catch (e) {
      console.log(`** Error: Unable to create worklet node: ${e}`);
      return null;
    }
  }
  return processorNode;
}

export function ele<E extends Element = Element>(selector: string) {
  return document.querySelector<E>(selector)!;
}

export function eles<E extends Element = Element>(selector: string) {
  return document.querySelectorAll<E>(selector);
}

export function element_wise_ave(arr: number[][]): number[] {
  if (arr.length == 0) {
    return [];
  }
  const sum = arr.reduce((prev, curr) => {
    return prev.map((x,i)=> x + (curr?.at(i) || 0));
  });
  return sum.map(x=> x / arr.length);
}

export function ave(arr: number[]) {
  if (arr.length == 0) {
    return 0;
  }
  return arr.reduce((prev,curr) => prev + curr) / arr.length;
}

export function normalize(arr: number[]) {
  const max = arr.reduce((prev, curr) => Math.abs(curr) > prev ? Math.abs(curr) : prev, 0)
  if (max < Number.EPSILON) {
    return arr;
  }
  return arr.map(e => e / max);
}

export function overlapAdd(arr: ArrayLike<number>, interval: number, sample: number) {
  const output = [...Array(sample)].map(e => 0);
  for (let i = 0; i < sample; i+=interval) {
    for (let j = 0; j < arr.length && i + j < sample; ++j) {
      output[i + j] += arr[j];
    }
  }
  return output;
}

export async function sleep(ms: number) {
  return await new Promise((res, req) => { 
    setTimeout(res, ms)
  });
}

export function createFileFromUrl(cv: any, path: string, url: string, callback: Function) {
  let request = new XMLHttpRequest();
  request.open('GET', url, true);
  request.responseType = 'arraybuffer';
  request.onload = function(ev) {
      if (request.readyState === 4) {
          if (request.status === 200) {
              let data = new Uint8Array(request.response);
              cv.FS_createDataFile('/', path, data, true, false, false);
              callback();
          } else {
              console.error('Failed to load ' + url + ' status: ' + request.status);
          }
      }
  };
  request.send();
};
