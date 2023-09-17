// https://github.com/mmorise/World から合成処理を移植

/* ----------------------------------------------------------------- */
/*           WORLD: High-quality speech analysis,                    */
/*           manipulation and synthesis system                       */
/*           developed by M. Morise                                  */
/*           http://www.kisc.meiji.ac.jp/~mmorise/world/english/     */
/* ----------------------------------------------------------------- */
/*                                                                   */
/*  Copyright (c) 2010  M. Morise                                    */
/*                                                                   */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/* - Redistributions of source code must retain the above copyright  */
/*   notice, this list of conditions and the following disclaimer.   */
/* - Redistributions in binary form must reproduce the above         */
/*   copyright notice, this list of conditions and the following     */
/*   disclaimer in the documentation and/or other materials provided */
/*   with the distribution.                                          */
/* - Neither the name of the M. Morise nor the names of its          */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission.   */
/*                                                                   */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            */
/* CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       */
/* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          */
/* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          */
/* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS */
/* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          */
/* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   */
/* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     */
/* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON */
/* ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   */
/* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    */
/* OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                       */
/* ----------------------------------------------------------------- */

import * as math from 'mathjs'

const kMySafeGuardMinimum = 0.000000000001

namespace util {
  export function zeros(size: number): number[] {
    return [...Array(size)].map(()=>0);
  }
  
  export function zeros_complex(size: number): math.Complex[] {
    return [...Array(size)].map(()=>math.complex());
  }
  
  export function fft(rfft_data: math.Complex[] | number[]): math.Complex[] {
    return math.fft(rfft_data) as math.Complex[];
  }
  
  export function irfft(rfft_data: math.Complex[]): number[] {
    const fft_data = zeros_complex((rfft_data.length - 1) * 2);
    for (let i = 0; i <= fft_data.length / 2; ++i) {
      fft_data[i] = rfft_data[i];
    }
    for (let i = 1; i < fft_data.length / 2; ++i) {
      fft_data[fft_data.length - i] = math.conj(rfft_data[i]);
    }
    const ifft_data = math.ifft(fft_data);
    return ifft_data.map(x=>x.re);
  }

  export function fftshift(x: number[]): number[] {
    const ret = zeros(x.length);
    for (let i = 0; i < x.length; ++i) {
      ret[i] = x[(i + (x.length / 2)) % x.length];
    }
    return ret;
  }
}

function GetMinimumPhaseSpectrum(fft_size: number, log_spectrum: number[]) {
  //  Mirroring
  for (let i = fft_size / 2 + 1; i < fft_size; ++i) {
    log_spectrum[i] = log_spectrum[fft_size - i];
  }

  //  This fft_plan carries out "forward" FFT.
  //  To carriy out the Inverse FFT, the sign of imaginary part
  //  is inverted after FFT.
  const cepstrum = util.zeros_complex(fft_size);
  util.fft(log_spectrum).forEach((x, i) => cepstrum[i] = x);

  cepstrum[0].im *= -1.0;
  for (let i = 1; i < fft_size / 2; ++i) {
    cepstrum[i].re *= 2.0;
    cepstrum[i].im *= -2.0;
  }
  cepstrum[fft_size / 2].im *= -1.0;
  for (let i = fft_size / 2 + 1; i < fft_size; ++i) {
    cepstrum[i].re = 0.0;
    cepstrum[i].im = 0.0;
  }

  const minimum_phase_spectrum = util.zeros_complex(fft_size);
  util.fft(cepstrum).forEach((x, i) => minimum_phase_spectrum[i] = x);

  //  Since x is complex number, calculation of exp(x) is as following.
  //  Note: This FFT library does not keep the aliasing.
  for (let i = 0; i <= fft_size / 2; ++i) {
    const tmp = Math.exp(minimum_phase_spectrum[i].re / fft_size);
    minimum_phase_spectrum[i].re = tmp * Math.cos(minimum_phase_spectrum[i].im / fft_size);
    minimum_phase_spectrum[i].im = tmp * Math.sin(minimum_phase_spectrum[i].im / fft_size);
  }

  return minimum_phase_spectrum;
}

//-----------------------------------------------------------------------------
// GetSpectrumWithFractionalTimeShift() calculates a periodic spectrum with
// the fractional time shift under 1/fs.
//-----------------------------------------------------------------------------
function GetSpectrumWithFractionalTimeShift(fft_size: number, coefficient: number, spectrum: math.Complex[]) {
  for (let i = 0; i <= fft_size / 2; ++i) {
    const re = spectrum[i].re;
    const im = spectrum[i].im;
    const re2 = Math.cos(coefficient * i);
    const im2 = Math.sqrt(1.0 - re2 * re2);  // Math.sin(pshift)

    spectrum[i].re = re * re2 + im * im2;
    spectrum[i].im = im * re2 - re * im2;
  }
}

//-----------------------------------------------------------------------------
// GetPeriodicResponse() calculates a periodic response.
//-----------------------------------------------------------------------------
function GetPeriodicResponse(fft_size: number, spectrum: number[], aperiodic_ratio: number[], fractional_time_shift: number, fs: number) {
  const log_spectrum = util.zeros(fft_size)
  for (let i = 0; i <= fft_size / 2; ++i) {
    log_spectrum[i] = Math.log(spectrum[i] * (1.0 - aperiodic_ratio[i]) + kMySafeGuardMinimum) / 2.0;
  }
  const minimum_phase_spectrum = GetMinimumPhaseSpectrum(fft_size, log_spectrum);

  const tmp_minimum_phase_spectrum = util.zeros_complex(fft_size / 2 + 1);
  for (let i = 0; i <= fft_size / 2; ++i) {
    tmp_minimum_phase_spectrum[i] = minimum_phase_spectrum[i];
  }

  // apply fractional time delay of fractional_time_shift seconds
  // using linear phase shift
  const coefficient = 2.0 * math.pi * fractional_time_shift * fs / fft_size;
  GetSpectrumWithFractionalTimeShift(fft_size, coefficient, tmp_minimum_phase_spectrum);

  let waveform = util.irfft(tmp_minimum_phase_spectrum).map(x => x * fft_size);
  waveform = util.fftshift(waveform);

  // @todo RemoveDCComponent

  return waveform;
}

//-----------------------------------------------------------------------------
// GetOneFrameSegment() calculates a periodic and aperiodic response at a time.
//-----------------------------------------------------------------------------
export function GetOneFrameSegment(f0: number, fs: number, power_spectrum: number[], fft_size: number=1024, q1: number=-0.15) {
  const aperiodic_ratio = util.zeros(power_spectrum.length);

  let waveform = GetPeriodicResponse(fft_size, power_spectrum, aperiodic_ratio, 0, fs);

  // @todo GetAperiodicResponse
  
  const noise_size = fs/f0;
  const sqrt_noise_size = Math.sqrt(noise_size);
  waveform = waveform.map(x => x * sqrt_noise_size / fft_size);

  return waveform;
}
