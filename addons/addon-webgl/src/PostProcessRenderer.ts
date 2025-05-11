import { IColor } from 'common/Types';
import { createProgram } from './WebglUtils';
import { Disposable } from 'vs/base/common/lifecycle';
import { IRenderDimensions } from 'browser/renderer/shared/Types';
import { throwIfFalsy } from 'browser/renderer/shared/RendererUtils';
import { IWebGL2RenderingContext, IWebGLVertexArrayObject } from 'Types';

const postProcessVertexShaderSource = `#version 300 es
layout(location=0) in vec2 a_position;
out vec2 v_texCoord;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = (a_position + 1.0) * 0.5;
}`;

const postProcessFragmentShaderSource = `#version 300 es
precision highp float;

in vec2 v_texCoord;

// Uniforms from PixelShaderSettings (adapted for VFX.js)
uniform float time;         // Was u_Time
uniform vec2  resolution;   // Was u_Resolution, resolution of the target
uniform sampler2D src;      // Was u_shaderTexture, the input texture

// Custom uniforms for this specific shader (VFX.js can pass these)
uniform float u_Scale;        // UI Scale (specific to this shader's effects)
uniform vec4  u_Background;   // Background color as rgba (specific to this shader's effects)
uniform vec2  offset;         // Standard VFX.js offset uniform

#define ENABLE_CURVE            1
#define ENABLE_OVERSCAN         1
#define ENABLE_BLOOM            1
#define ENABLE_BLUR             1
#define ENABLE_GRAYSCALE        0
#define ENABLE_BLACKLEVEL       1
#define ENABLE_REFRESHLINE      1
#define ENABLE_SCANLINES        1
#define ENABLE_TINT             0
#define ENABLE_GRAIN            1

#define CURVE_INTENSITY         1.0

#define OVERSCAN_PERCENTAGE     0.02

#define BLOOM_OFFSET            0.0015
#define BLOOM_STRENGTH          0.8

#define BLUR_MULTIPLIER         1.05
#define BLUR_STRENGTH           0.2
#define BLUR_OFFSET             0.003

#define GRAYSCALE_INTENSITY     0
#define GRAYSCALE_GLEAM         0
#define GRAYSCALE_LUMINANCE     1
#define GRAYSCALE_LUMA          0

#define TINT_AMBER              vec3(1.0, 0.7, 0.0)
#define TINT_LIGHT_AMBER        vec3(1.0, 0.8, 0.0)
#define TINT_GREEN_1            vec3(0.2, 1.0, 0.0)
#define TINT_APPLE_II           vec3(0.2, 1.0, 0.2)
#define TINT_GREEN_2            vec3(0.0, 1.0, 0.2)
#define TINT_APPLE_IIc          vec3(0.4, 1.0, 0.4)
#define TINT_GREEN_3            vec3(0.0, 1.0, 0.4)
#define TINT_WARM               vec3(1.0, 0.9, 0.8)
#define TINT_COOL               vec3(0.8, 0.9, 1.0)

#define TINT_COLOR              TINT_AMBER

#define BLACKLEVEL_FLOOR        (TINT_COLOR / 40.0)


#define GRAIN_INTENSITY         0.02

#if ENABLE_BLOOM && (GRAYSCALE_GLEAM || GRAYSCALE_LUMA)
#undef GRAYSCALE_INTENSITY
#undef GRAYSCALE_GLEAM
#undef GRAYSCALE_LUMINANCE
#undef GRAYSCALE_LUMA
#define GRAYSCALE_INTENSITY     0
#define GRAYSCALE_GLEAM         0
#define GRAYSCALE_LUMINANCE     1
#define GRAYSCALE_LUMA          0
#endif

#if ENABLE_BLACKLEVEL && !ENABLE_TINT
#undef BLACKLEVEL_FLOOR
#define BLACKLEVEL_FLOOR vec3(0.05, 0.05, 0.05)
#endif

#if ENABLE_CURVE
vec2 transformCurve(vec2 uv_in) {
  vec2 uv_local = uv_in;
  uv_local -= 0.5;
  float r = (uv_local.x * uv_local.x + uv_local.y * uv_local.y) * CURVE_INTENSITY;
  uv_local *= 4.2 + r;
  uv_local *= 0.25;
  uv_local += 0.5;
  return uv_local;
}
#endif

#if ENABLE_OVERSCAN
vec2 calculateOverscanUV(vec2 screenuv_in) {
  vec2 uv_out = screenuv_in;
  uv_out -= 0.5;
  uv_out *= 1.0 / (1.0 - OVERSCAN_PERCENTAGE);
  uv_out += 0.5;
  return uv_out;
}
#endif

#if ENABLE_BLOOM
vec3 bloom(vec3 color, vec2 uv_sample) {
  vec3 bloom_effect = color - texture(src, uv_sample + vec2(-BLOOM_OFFSET, 0.0)).rgb;
  vec3 bloom_mask = bloom_effect * BLOOM_STRENGTH;
  return clamp(color + bloom_mask, 0.0, 1.0);
}
#endif

#if ENABLE_BLUR
const float blurWeights[9] = float[](0.0, 0.092, 0.081, 0.071, 0.061, 0.051, 0.041, 0.031, 0.021);

vec3 blurH(vec3 c, vec2 uv_sample) {
  vec3 screen_color = texture(src, uv_sample).rgb * 0.102;
  for (int i = 1; i < 9; i++) {
    screen_color += texture(src, uv_sample + vec2( float(i) * BLUR_OFFSET, 0.0)).rgb * blurWeights[i];
    screen_color += texture(src, uv_sample + vec2(-float(i) * BLUR_OFFSET, 0.0)).rgb * blurWeights[i];
  }
  return screen_color * BLUR_MULTIPLIER;
}

vec3 blurV(vec3 c, vec2 uv_sample) {
  vec3 screen_color = texture(src, uv_sample).rgb * 0.102;
  for (int i = 1; i < 9; i++) {
    screen_color += texture(src, uv_sample + vec2(0.0,  float(i) * BLUR_OFFSET)).rgb * blurWeights[i];
    screen_color += texture(src, uv_sample + vec2(0.0, -float(i) * BLUR_OFFSET)).rgb * blurWeights[i];
  }
  return screen_color * BLUR_MULTIPLIER;
}

vec3 blur(vec3 color, vec2 uv_sample) {
  vec3 blur_effect = (blurH(color, uv_sample) + blurV(color, uv_sample)) / 2.0 - color;
  vec3 blur_mask = blur_effect * BLUR_STRENGTH;
  return clamp(color + blur_mask, 0.0, 1.0);
}
#endif

#if ENABLE_GRAYSCALE
vec3 rgb2intensity(vec3 c) {
  return vec3((c.r + c.g + c.b) / 3.0);
}

#define GAMMA 2.2
vec3 gammaCorrect(vec3 c) {
  return pow(c, vec3(GAMMA));
}

vec3 invGammaCorrect(vec3 c) {
  return pow(c, vec3(1.0 / GAMMA));
}

vec3 rgb2gleam(vec3 c) {
  c = invGammaCorrect(c);
  c = rgb2intensity(c);
  return gammaCorrect(c);
}

vec3 rgb2luminance(vec3 c) {
  return vec3(0.2989 * c.r + 0.5866 * c.g + 0.1145 * c.b);
}

vec3 rgb2luma(vec3 c) {
  c = invGammaCorrect(c);
  c = vec3(0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b);
  return gammaCorrect(c);
}

vec3 grayscale(vec3 color_in) {
  vec3 processed_color = color_in;
  #if GRAYSCALE_INTENSITY
  processed_color.rgb = clamp(rgb2intensity(processed_color.rgb), 0.0, 1.0);
  #elif GRAYSCALE_GLEAM
  processed_color.rgb = clamp(rgb2gleam(processed_color.rgb), 0.0, 1.0);
  #elif GRAYSCALE_LUMINANCE
  processed_color.rgb = clamp(rgb2luminance(processed_color.rgb), 0.0, 1.0);
  #elif GRAYSCALE_LUMA
  processed_color.rgb = clamp(rgb2luma(processed_color.rgb), 0.0, 1.0);
  #else
  processed_color.rgb = vec3(1.0, 0.0, 1.0) - processed_color.rgb;
  #endif
  return processed_color;
}
#endif

#if ENABLE_BLACKLEVEL
vec3 blacklevel(vec3 color_in) {
  vec3 c = color_in;
	c.rgb -= BLACKLEVEL_FLOOR;
	c.rgb = clamp(c.rgb, 0.0, 1.0);
	c.rgb += BLACKLEVEL_FLOOR;
	return clamp(c, 0.0, 1.0);
}
#endif

#if ENABLE_REFRESHLINE
vec3 refreshLines(vec3 color_in, vec2 uv_screen) {
  vec3 c = color_in;
  float timeOver = mod(time / 5.0, 1.5) - 0.5;
  float refreshLineColorTint = timeOver - uv_screen.y;
  if(uv_screen.y > timeOver && uv_screen.y - 0.03 < timeOver ) c.rgb += (refreshLineColorTint * 2.0);
  return clamp(c, 0.0, 1.0);
}
#endif

#if ENABLE_SCANLINES
#define SCANLINE_FACTOR 0.3
#define SCALED_SCANLINE_PERIOD u_Scale // u_Scale is a custom uniform

float squareWave(float y_coord) {
  return 1.0 - (mod(floor(y_coord / max(SCALED_SCANLINE_PERIOD, 1.0)), 2.0)) * SCANLINE_FACTOR;
}

vec3 scanlines(vec3 color_in, vec2 frag_pos) {
  vec3 c = color_in;
  float wave = squareWave(frag_pos.y);

  if (length(c.rgb) < 0.2 && false) {
    return clamp(c + wave * 0.1, 0.0, 1.0);
  } else {
    return clamp(c * wave, 0.0, 1.0);
  }
}
#endif

#if ENABLE_TINT
vec3 tint(vec3 color_in) {
  vec3 c = color_in;
	c.rgb *= TINT_COLOR;
	return clamp(c, 0.0, 1.0);
}
#endif

#if ENABLE_GRAIN
#define a0  0.151015505647689
#define a1 -0.5303572634357367
#define a2  1.365020122861334
#define b0  0.132089632343748
#define b1 -0.7607324991323768

float permute(float x) {
  x *= (34.0 * x + 1.0);
  return 289.0 * fract(x * (1.0 / 289.0));
}

float do_rand(inout float state_val) {
  state_val = permute(state_val);
  return fract(state_val / 41.0);
}

vec3 grain(vec3 color_in, vec2 uv_screen) {
  vec3 c = color_in;
  vec3 m = vec3(uv_screen, mod(time, 5.0) / 5.0) + 1.0;
  float state = permute(permute(m.x) + m.y) + m.z;

  float p = 0.95 * do_rand(state) + 0.025;
  float q = p - 0.5;
  float r2 = q * q;

  float grain_val = q * (a2 + (a1 * r2 + a0) / (r2 * r2 + b1 * r2 + b0));
  c.rgb += GRAIN_INTENSITY * grain_val;

  return clamp(c, 0.0, 1.0);
}
#endif

// Output color
out vec4 outColor;

void main() {
  vec2 uv = v_texCoord; // Old way, remove

  vec2 screen_pixel_pos = gl_FragCoord.xy;

  vec4 color = vec4(1.0, 0.0, 1.0, -1.0); // Initial sentinel value

  #if ENABLE_CURVE
  uv = transformCurve(uv);

  if(uv.x <  -0.025 || uv.y <  -0.025 || uv.x >   1.025 || uv.y >   1.025) {
      outColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
  }
  if(uv.x <  -0.015 || uv.y <  -0.015 || uv.x >   1.015 || uv.y >   1.015) {
      outColor = vec4(0.03, 0.03, 0.03, 1.0);
      return;
  }
  if(uv.x <  -0.001 || uv.y <  -0.001 || uv.x >   1.001 || uv.y >   1.001) {
      outColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
  }
  #endif

  vec2 uv_for_texture_sample = uv;
  vec2 uv_for_screen_effects = uv;

  #if ENABLE_OVERSCAN
  vec2 overscan_uv_candidate = calculateOverscanUV(uv_for_texture_sample);

  if (overscan_uv_candidate.x < 0.0 || overscan_uv_candidate.x > 1.0 ||
      overscan_uv_candidate.y < 0.0 || overscan_uv_candidate.y > 1.0) {
    // Use the custom u_Background uniform here
    color = clamp(vec4(u_Background.rgb, 0.0) * 0.1, 0.0, 1.0);
  } else {
    uv_for_texture_sample = overscan_uv_candidate;
  }
  #endif

  if (color.a < 0.0) { // Check if color was set by overscan
    color = texture(src, uv_for_texture_sample);
  }

  #if ENABLE_BLOOM
  color.rgb = bloom(color.rgb, uv_for_texture_sample);
  #endif

  #if ENABLE_BLUR
  color.rgb = blur(color.rgb, uv_for_texture_sample);
  #endif

  #if ENABLE_GRAYSCALE
  color.rgb = grayscale(color.rgb);
  #endif

  #if ENABLE_BLACKLEVEL
  color.rgb = blacklevel(color.rgb);
  #endif

  #if ENABLE_REFRESHLINE
  color.rgb = refreshLines(color.rgb, uv_for_screen_effects);
  #endif

  #if ENABLE_SCANLINES
  color.rgb = scanlines(color.rgb, screen_pixel_pos);
  #endif

  #if ENABLE_TINT
  color.rgb = tint(color.rgb);
  #endif

  #if ENABLE_GRAIN
  color.rgb = grain(color.rgb, uv_for_screen_effects);
  #endif

  outColor = color;
}`;

export class PostProcessRenderer extends Disposable {
  private _program: WebGLProgram;
  private _vao: WebGLVertexArrayObject;
  private _vertexBuffer: WebGLBuffer;
  private _uTimeLocation: WebGLUniformLocation;
  private _uSrcLocation: WebGLUniformLocation;
  private _uScaleLocation: WebGLUniformLocation;
  private _uBackgroundLocation: WebGLUniformLocation;

  constructor(private _gl: IWebGL2RenderingContext, private _dimensions: IRenderDimensions) {
    super();

    this._program = throwIfFalsy(createProgram(
      _gl,
      postProcessVertexShaderSource,
      postProcessFragmentShaderSource
    ));

    this._uTimeLocation = throwIfFalsy(_gl.getUniformLocation(this._program, 'time'));
    this._uSrcLocation = throwIfFalsy(_gl.getUniformLocation(this._program, 'src'));
    this._uScaleLocation = throwIfFalsy(_gl.getUniformLocation(this._program, 'u_Scale'));
    this._uBackgroundLocation = throwIfFalsy(_gl.getUniformLocation(this._program, 'u_Background'));

    const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this._vertexBuffer = throwIfFalsy(_gl.createBuffer());
    this._vao = throwIfFalsy(_gl.createVertexArray());

    _gl.bindVertexArray(this._vao);
    _gl.bindBuffer(_gl.ARRAY_BUFFER, this._vertexBuffer);
    _gl.bufferData(_gl.ARRAY_BUFFER, vertices, _gl.STATIC_DRAW);
    _gl.enableVertexAttribArray(0);
    _gl.vertexAttribPointer(0, 2, _gl.FLOAT, false, 0, 0);
    _gl.bindVertexArray(null as unknown as IWebGLVertexArrayObject);
  }

  public render(texture: WebGLTexture, time: number, scale: number, background: IColor): void {
    const gl = this._gl;
    const bgColor = new Float32Array([
      ((background.rgba >> 24) & 0xff) / 255,
      ((background.rgba >> 16) & 0xff) / 255,
      ((background.rgba >> 8) & 0xff) / 255,
      (background.rgba & 0xff) / 255
    ]);

    gl.useProgram(this._program);
    gl.bindVertexArray(this._vao);

    gl.uniform1f(this._uTimeLocation, time);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(this._uSrcLocation, 0);
    gl.uniform1f(this._uScaleLocation, scale);
    gl.uniform4fv(this._uBackgroundLocation, bgColor);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null as unknown as IWebGLVertexArrayObject);
  }

  public dispose(): void {
    this._gl.deleteProgram(this._program);
    this._gl.deleteVertexArray(this._vao);
    this._gl.deleteBuffer(this._vertexBuffer);
    super.dispose();
  }
}
