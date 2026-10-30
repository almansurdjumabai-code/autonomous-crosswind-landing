/**
 * Ground-material enhancement for the glTF airfield surfaces (asphalt, grass, concrete).
 *
 * Patches three.js MeshStandardMaterial shaders in place (onBeforeCompile) with two extra samples
 * of the material's own textures:
 *
 *  - near-field DETAIL: the albedo and normal map re-sampled at `detailScale` x frequency and faded
 *    in by view distance (`detailNear`..`detailFar`), so close-up ground stays crisp instead of
 *    showing a magnified, blurry tile;
 *  - far-field MACRO: the albedo re-sampled at `macroScale` x frequency (offset), which modulates the
 *    base tile at landscape scale and hides its repetition.
 *
 * Both modulations are normalised by the texture's mean colour, so they add contrast and variety
 * without shifting the overall brightness. The mean is computed once on the CPU (linear RGB) and
 * passed as a uniform, so the per-fragment cost is exactly two extra albedo fetches and one extra
 * normal fetch. Nothing here is rebuilt per frame: the program is compiled once (stable cache key)
 * and the uniforms never change.
 */
import * as THREE from "three";

const meanCache = new WeakMap();

/** Mean linear-RGB colour of a texture's image (downsampled on a 2D canvas), cached per image. */
function textureMean(texture) {
  const img = texture?.image;
  if (!img) return new THREE.Vector3(0.5, 0.5, 0.5);
  if (meanCache.has(img)) return meanCache.get(img);
  const n = 32;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, n, n);
  const px = ctx.getImageData(0, 0, n, n).data;
  // Average in linear space, as the GPU sees the texture after sRGB decode.
  const srgb = texture.colorSpace === THREE.SRGBColorSpace;
  const sum = [0, 0, 0];
  for (let i = 0; i < px.length; i += 4) for (let k = 0; k < 3; k++) sum[k] += srgb ? srgbToLinear(px[i + k] / 255) : px[i + k] / 255;
  const count = px.length / 4;
  const mean = new THREE.Vector3(...sum.map((v) => Math.max(v / count, 1e-3)));
  meanCache.set(img, mean);
  return mean;
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

const DEFAULTS = {
  detailScale: 7.0,
  detailStrength: 0.55,
  detailNear: 4.0,
  detailFar: 120.0,
  macroScale: 0.085,
  macroStrength: 0.4,
  normalDetail: 0.7,
};

export function enhanceGroundMaterial(material, options = {}) {
  if (!material || material.userData.groundEnhanced) return material;
  const o = { ...DEFAULTS, ...options };
  material.userData.groundEnhanced = true;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uGdDetailScale = { value: o.detailScale };
    shader.uniforms.uGdDetailStrength = { value: o.detailStrength };
    shader.uniforms.uGdNear = { value: o.detailNear };
    shader.uniforms.uGdFar = { value: o.detailFar };
    shader.uniforms.uGdMacroScale = { value: o.macroScale };
    shader.uniforms.uGdMacroStrength = { value: o.macroStrength };
    shader.uniforms.uGdNormalDetail = { value: o.normalDetail };
    shader.uniforms.uGdMean = { value: textureMean(material.map) };
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
uniform float uGdDetailScale;
uniform float uGdDetailStrength;
uniform float uGdNear;
uniform float uGdFar;
uniform float uGdMacroScale;
uniform float uGdMacroStrength;
uniform float uGdNormalDetail;
uniform vec3 uGdMean;`,
      )
      .replace(
        "#include <map_fragment>",
        `#ifdef USE_MAP
  vec4 sampledDiffuseColor = texture2D( map, vMapUv );
  vec3 gdMean = uGdMean;
  float gdNearW = 1.0 - smoothstep( uGdNear, uGdFar, length( vViewPosition ) );
  vec3 gdDetail = texture2D( map, vMapUv * uGdDetailScale ).rgb / gdMean;
  vec3 gdMacro = texture2D( map, vMapUv * uGdMacroScale + vec2( 0.37, 0.71 ) ).rgb / gdMean;
  sampledDiffuseColor.rgb *= mix( vec3( 1.0 ), gdDetail, uGdDetailStrength * gdNearW );
  sampledDiffuseColor.rgb *= mix( vec3( 1.0 ), gdMacro, uGdMacroStrength );
  diffuseColor *= sampledDiffuseColor;
#endif`,
      )
      .replace(
        "vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;",
        `vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
  {
    float gdNearN = 1.0 - smoothstep( uGdNear, uGdFar, length( vViewPosition ) );
    vec3 gdN = texture2D( normalMap, vNormalMapUv * uGdDetailScale ).xyz * 2.0 - 1.0;
    // Whiteout/UDN blend of the detail normal, faded with distance.
    mapN = normalize( vec3( mapN.xy + gdN.xy * uGdNormalDetail * gdNearN, mapN.z ) );
  }`,
      );
  };
  material.customProgramCacheKey = () =>
    `ground:${o.detailScale}:${o.detailStrength}:${o.macroScale}:${o.macroStrength}:${o.normalDetail}`;
  material.needsUpdate = true;
  return material;
}

/** Set the same UV repeat on every texture of a material (map, normal, ORM). */
export function setTextureRepeat(material, repeat) {
  for (const tex of [material.map, material.normalMap, material.roughnessMap, material.metalnessMap, material.aoMap]) {
    if (!tex) continue;
    tex.repeat.set(repeat, repeat);
    tex.needsUpdate = true;
  }
}
