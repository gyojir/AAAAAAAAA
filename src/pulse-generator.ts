
class PulseGeneratorProcessor extends AudioWorkletProcessor {
  isOn: boolean = false;

  constructor() {
    super();
  }

  static get parameterDescriptors() {
    return [
      {
        name: "gain",
        defaultValue: 0.2,
        minValue: 0,
        maxValue: 1,
      },
    ];
  }

  process(inputList: Float32Array[][], outputList: Float32Array[][], parameters: Record<string, Float32Array>) {
    // const gain = parameters.gain[0];

    let input = inputList[0];
    let output = outputList[0];
    if (input.length == 0) {
      return true;
    }

    let sampleCount = input[0].length;

    for (let i = 0; i < sampleCount; i++) {
      let sample = input[0][i];
      if (!this.isOn && sample > 0) {
        output[0][i] = 1.0;
      }
      else {
        output[0][i] = 0.0;
      }
      this.isOn = sample > 0;
    }

    return true;
  }
}

registerProcessor("pulse-generator", PulseGeneratorProcessor);