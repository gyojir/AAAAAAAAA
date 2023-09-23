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

export async function sleep(ms: number) {
  return await new Promise((res, req) => { 
    setTimeout(res, ms)
  });
}
