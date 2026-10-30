/**
 * Wind and airflow visualisation, all driven by the recorded telemetry (scrub-safe, nothing
 * accumulates between frames apart from the advection phases):
 *  - FlowLines: streamlines computed in Blender around the airframe (Rankine fuselage + elliptic
 *    lifting line, see airliner_model.py) with dashes travelling at the local flow speed; the whole
 *    set follows the recorded sideslip and angle of attack.
 *  - Streaks: air-mass tracers wrapped in a box around the aircraft. One layer shows the flow past
 *    the airframe; a low, long layer shows the crosswind itself sweeping across the runway.
 *  - Vortices: wing-tip and flap-tip vortex filaments shed along the flown path and carried by the wind.
 */
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { clock, useUI } from "../lib/store.js";
import { DEG } from "../lib/flightData.js";

const LINE_VERTEX = /* glsl */ `
  attribute vec3 aDir;
  attribute float aSide;
  attribute float aS;
  attribute float aFade;
  attribute float aSpeed;
  uniform float uWidth;
  varying float vS;
  varying float vFade;
  varying float vSpeed;
  varying float vSide;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vec3 dirView = normalize(mat3(modelViewMatrix) * aDir);
    vec3 perp = normalize(cross(dirView, normalize(mv.xyz)));
    mv.xyz += perp * aSide * max(uWidth, -mv.z * 0.0007);
    vS = aS;
    vFade = aFade;
    vSpeed = aSpeed;
    vSide = aSide;
    gl_Position = projectionMatrix * mv;
  }`;

const flowMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  side: THREE.DoubleSide,
  uniforms: { uWidth: { value: 0.016 }, uPhase: { value: 0 }, uOpacity: { value: 1 } },
  vertexShader: LINE_VERTEX,
  fragmentShader: /* glsl */ `
    uniform float uPhase;
    uniform float uOpacity;
    varying float vS;
    varying float vFade;
    varying float vSpeed;
    varying float vSide;
    void main() {
      // A soft-headed comet every 15 m of path, offset per line through vFade's fractional seed.
      float d = fract(vS / 15.0 - uPhase);
      float comet = pow(d, 6.0) * (1.0 - smoothstep(0.985, 1.0, d));
      float edge = 1.0 - vSide * vSide;
      float slow = clamp((vSpeed - 0.85) / 0.45, 0.0, 1.0);
      vec3 col = mix(vec3(0.22, 0.55, 1.0), vec3(0.75, 0.96, 1.0), slow);
      float a = (0.008 + 0.13 * comet) * edge * vFade * uOpacity;
      gl_FragColor = vec4(col, a);
      #include <colorspace_fragment>
    }`,
});

const filamentMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  side: THREE.DoubleSide,
  uniforms: { uWidth: { value: 0.02 } },
  vertexShader: LINE_VERTEX,
  fragmentShader: /* glsl */ `
    varying float vFade;
    varying float vSide;
    void main() {
      float edge = 1.0 - vSide * vSide;
      gl_FragColor = vec4(vec3(0.82, 0.92, 1.0), 0.07 * edge * vFade);
      #include <colorspace_fragment>
    }`,
});

/* ---------------------------------------------------------------------------------------------- */
function buildRibbonIndex(segmentsPerLine, lineCount) {
  const out = [];
  for (let l = 0; l < lineCount; l++) {
    const base = l * (segmentsPerLine + 1) * 2;
    for (let i = 0; i < segmentsPerLine; i++) {
      const a = base + i * 2;
      out.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  return out;
}

function FlowLines({ flow, rigged }) {
  const group = useRef();
  const geometry = useMemo(() => {
    const lines = flow.lines;
    const n = lines[0].length;
    const v = lines.length * n * 2;
    const pos = new Float32Array(v * 3);
    const dir = new Float32Array(v * 3);
    const side = new Float32Array(v);
    const arc = new Float32Array(v);
    const fade = new Float32Array(v);
    const speed = new Float32Array(v);
    const nominal = Math.hypot(...lines[0][1].map((c, k) => c - lines[0][0][k]));
    lines.forEach((line, l) => {
      const seed = (l * 0.6180339) % 1;
      let s = seed * 15;
      line.forEach((p, i) => {
        const a = line[Math.max(i - 1, 0)];
        const b = line[Math.min(i + 1, n - 1)];
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const len = Math.hypot(d[0], d[1], d[2]) || 1;
        const step = len / (b === a ? 1 : i === 0 || i === n - 1 ? 1 : 2);
        if (i > 0) s += Math.hypot(p[0] - line[i - 1][0], p[1] - line[i - 1][1], p[2] - line[i - 1][2]);
        const u = i / (n - 1);
        for (const [k, sd] of [[0, -1], [1, 1]]) {
          const o = (l * n + i) * 2 + k;
          pos.set(p, o * 3);
          dir.set([d[0] / len, d[1] / len, d[2] / len], o * 3);
          side[o] = sd;
          arc[o] = s;
          fade[o] = Math.min(u / 0.12, 1) * Math.min((1 - u) / 0.3, 1);
          speed[o] = step / nominal;
        }
      });
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aDir", new THREE.BufferAttribute(dir, 3));
    g.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
    g.setAttribute("aS", new THREE.BufferAttribute(arc, 1));
    g.setAttribute("aFade", new THREE.BufferAttribute(fade, 1));
    g.setAttribute("aSpeed", new THREE.BufferAttribute(speed, 1));
    g.setIndex(buildRibbonIndex(n - 1, lines.length));
    return g;
  }, [flow]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const st = useRef({ phase: 0, lastT: 0, beta: 0, alpha: 0, q: new THREE.Quaternion(), q2: new THREE.Quaternion() });
  const Y = useMemo(() => new THREE.Vector3(0, 1, 0), []);
  const Z = useMemo(() => new THREE.Vector3(0, 0, 1), []);

  useFrame((_, delta) => {
    const s = clock.sample;
    const gr = group.current;
    if (!s || !gr) return;
    const m = st.current;
    const simDt = Math.max(s.t - m.lastT, 0);
    m.lastT = s.t;
    // Dashes advect at airspeed, independent of the frame rate and frozen while paused.
    m.phase = (m.phase + Math.min(simDt, 0.2) * Math.max(s.ias, 0) / 15) % 1;
    flowMaterial.uniforms.uPhase.value = m.phase;
    const k = 1 - Math.exp(-Math.min(delta, 0.1) / 0.25);
    m.beta += (THREE.MathUtils.clamp(s.beta, -14, 14) - m.beta) * k;
    m.alpha += (THREE.MathUtils.clamp(s.alpha - flow.alpha_deg, -10, 10) - m.alpha) * k;
    // Rotate the frozen flow field about the reference point onto the current incidence.
    m.q.setFromAxisAngle(Y, -m.beta * DEG);
    m.q2.setFromAxisAngle(Z, -m.alpha * DEG);
    gr.quaternion.copy(m.q).multiply(m.q2);
    // Over the runway the flow ceases to be free-stream; let it fade with the rollout.
    flowMaterial.uniforms.uOpacity.value = THREE.MathUtils.clamp((s.ias - 6) / 12, 0, 1);
  });

  return (
    <primitive object={rigged.aircraft}>
      <group ref={group}>
        <mesh geometry={geometry} material={flowMaterial} frustumCulled={false} renderOrder={12} />
      </group>
    </primitive>
  );
}

/* ---------------------------------------------------------------------------------------------- */
function Streaks({ count, box, heightCenter, followHeight, width, opacity, mode, color }) {
  const mesh = useRef();
  const { geometry, material, drift } = useMemo(() => {
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute([0, -1, 0, 0, 1, 0, 1, -1, 0, 1, 1, 0], 3));
    g.setIndex([0, 1, 2, 1, 3, 2]);
    const offsets = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      offsets[i * 4] = Math.random() * box[0];
      offsets[i * 4 + 1] = Math.random() * box[1];
      offsets[i * 4 + 2] = Math.random() * box[2];
      offsets[i * 4 + 3] = Math.random();
    }
    g.setAttribute("aOffset", new THREE.InstancedBufferAttribute(offsets, 4));
    g.instanceCount = count;
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      uniforms: {
        uCenter: { value: new THREE.Vector3() },
        uDrift: { value: new THREE.Vector3() },
        uDir: { value: new THREE.Vector3(1, 0, 0) },
        uLen: { value: 1 },
        uWidth: { value: width },
        uBox: { value: new THREE.Vector3(...box) },
        uOpacity: { value: opacity },
        uColor: { value: new THREE.Color(color) },
      },
      vertexShader: /* glsl */ `
        attribute vec4 aOffset;
        uniform vec3 uCenter, uDrift, uDir, uBox;
        uniform float uLen, uWidth;
        varying float vA;
        varying float vU;
        varying float vV;
        void main() {
          // Tracers ride the air mass; the box wraps around the aircraft so the volume never empties.
          vec3 rel = mod(aOffset.xyz + uDrift - uCenter, uBox) - 0.5 * uBox;
          vec3 world = uCenter + rel;
          float lenScale = 0.55 + 0.9 * aOffset.w;
          world += uDir * (position.x - 0.5) * uLen * lenScale;
          vec4 mv = viewMatrix * vec4(world, 1.0);
          vec3 dirView = normalize(mat3(viewMatrix) * uDir);
          vec3 perp = normalize(cross(dirView, normalize(mv.xyz)));
          mv.xyz += perp * position.y * max(uWidth * (0.7 + 0.6 * aOffset.w), -mv.z * 0.0006);
          vec3 e = abs(rel) / (0.5 * uBox);
          // Tracers dissolve close to the lens instead of flashing past as wide beams.
          vA = (1.0 - smoothstep(0.65, 1.0, max(e.x, max(e.y, e.z)))) * (0.35 + 0.65 * aOffset.w) * smoothstep(12.0, 55.0, -mv.z);
          vU = position.x;
          vV = position.y;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        uniform float uOpacity;
        varying float vA;
        varying float vU;
        varying float vV;
        void main() {
          float along = smoothstep(0.0, 0.8, vU) * (1.0 - smoothstep(0.92, 1.0, vU));
          gl_FragColor = vec4(uColor, uOpacity * vA * along * (1.0 - vV * vV));
          #include <colorspace_fragment>
        }`,
    });
    return { geometry: g, material: mat, drift: new THREE.Vector3() };
  }, [box, color, count, opacity, width]);
  useEffect(() => () => {
    geometry.dispose();
    material.dispose();
  }, [geometry, material]);
  const st = useRef({ lastT: null, wind: new THREE.Vector3(), v: new THREE.Vector3(), dir: new THREE.Vector3(1, 0, 0) });

  useFrame((_, delta) => {
    const s = clock.sample;
    if (!s) return;
    const m = st.current;
    const simDt = m.lastT == null ? 0 : Math.max(Math.min(s.t - m.lastT, 0.2), 0);
    m.lastT = s.t;
    m.wind.set(s.wind[0], s.wind[1], s.wind[2]);
    drift.addScaledVector(m.wind, simDt);
    const u = material.uniforms;
    u.uDrift.value.copy(drift);
    u.uCenter.value.set(s.visPos[0], followHeight ? s.visPos[1] + heightCenter : heightCenter, s.visPos[2]);
    const k = 1 - Math.exp(-Math.min(delta, 0.1) / 0.2);
    if (mode === "flow") {
      // Direction of the air relative to the airframe: wind minus ground velocity along the track.
      const trk = s.track * DEG;
      m.v.set(s.wind[0] - s.gs * Math.sin(trk), s.wind[1] - s.vz, s.wind[2] + s.gs * Math.cos(trk));
      const speed = m.v.length();
      m.dir.lerp(m.v.normalize(), k).normalize();
      u.uDir.value.copy(m.dir);
      u.uLen.value = Math.max(speed * 0.13, 0.4);
      u.uOpacity.value = opacity * THREE.MathUtils.clamp(speed / 18, 0, 1);
    } else {
      const speed = m.wind.length();
      if (speed > 0.05) m.dir.lerp(m.v.copy(m.wind).divideScalar(speed), k).normalize();
      u.uDir.value.copy(m.dir);
      u.uLen.value = 4 + speed * 2.6;
      u.uOpacity.value = opacity * THREE.MathUtils.clamp(speed / 5, 0.15, 1.25);
    }
  });
  return <mesh ref={mesh} geometry={geometry} material={material} frustumCulled={false} renderOrder={11} />;
}

/* ---------------------------------------------------------------------------------------------- */
const VORTEX_POINTS = 90;
const VORTEX_DT = 0.08;
// Shed points in the aircraft frame (glTF): [x, y, z, swirl radius, phase, strength].
const SHED = [
  [-3.2, 4.9, -17.7, 0.28, 0, 1],
  [-3.2, 4.9, -17.7, 0.28, Math.PI, 0.65],
  [-3.2, 4.9, 17.7, 0.28, 0, 1],
  [-3.2, 4.9, 17.7, 0.28, Math.PI, 0.65],
  [-1.6, 1.85, -12, 0.2, 0, 0.55],
  [-1.6, 1.85, 12, 0.2, 0, 0.55],
];

function Vortices({ episode }) {
  const buffers = useMemo(() => {
    const v = SHED.length * VORTEX_POINTS * 2;
    const g = new THREE.BufferGeometry();
    const pos = new THREE.BufferAttribute(new Float32Array(v * 3), 3);
    const dir = new THREE.BufferAttribute(new Float32Array(v * 3), 3);
    const fade = new THREE.BufferAttribute(new Float32Array(v), 1);
    const side = new Float32Array(v);
    for (let i = 0; i < v; i++) side[i] = i % 2 ? 1 : -1;
    pos.setUsage(THREE.DynamicDrawUsage);
    dir.setUsage(THREE.DynamicDrawUsage);
    fade.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("position", pos);
    g.setAttribute("aDir", dir);
    g.setAttribute("aFade", fade);
    g.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
    g.setAttribute("aS", new THREE.BufferAttribute(new Float32Array(v), 1));
    g.setAttribute("aSpeed", new THREE.BufferAttribute(new Float32Array(v), 1));
    g.setIndex(buildRibbonIndex(VORTEX_POINTS - 1, SHED.length));
    return { g, pos, dir, fade, q: new THREE.Quaternion(), q2: new THREE.Quaternion(), a: new THREE.Vector3(), prev: new THREE.Vector3() };
  }, []);
  useEffect(() => () => buffers.g.dispose(), [buffers]);

  useFrame(() => {
    const s = clock.sample;
    const ep = episode;
    if (!s || !ep) return;
    const { pos, dir, fade, q, q2, a, prev } = buffers;
    const n = ep.n;
    const t0 = ep.t[0];
    const dtS = ep.t[1] - ep.t[0];
    const windX = s.wind[0];
    const windY = s.wind[1];
    const windZ = s.wind[2];
    SHED.forEach((sh, line) => {
      const swirlSign = sh[2] < 0 ? -1 : 1;
      for (let k = 0; k < VORTEX_POINTS; k++) {
        const age = k * VORTEX_DT;
        const tp = s.t - age;
        const o = (line * VORTEX_POINTS + k) * 2;
        let ax = 0;
        let ay = 0;
        let az = 0;
        let alpha = 0;
        if (tp >= t0) {
          // Interpolated shed pose along the flown path (allocation free, no spline needed behind the airframe).
          const f = Math.min((tp - t0) / dtS, n - 1.001);
          const i = Math.floor(f);
          const u = f - i;
          const P = ep.visPos;
          const Q = ep.visQuat;
          q.set(Q[i * 4], Q[i * 4 + 1], Q[i * 4 + 2], Q[i * 4 + 3]);
          q2.set(Q[i * 4 + 4], Q[i * 4 + 5], Q[i * 4 + 6], Q[i * 4 + 7]);
          q.slerp(q2, u);
          a.set(sh[0], sh[1], sh[2]).applyQuaternion(q);
          ax = a.x + P[i * 3] + (P[i * 3 + 3] - P[i * 3]) * u + windX * age;
          ay = a.y + P[i * 3 + 1] + (P[i * 3 + 4] - P[i * 3 + 1]) * u + windY * age - 0.55 * age;
          az = a.z + P[i * 3 + 2] + (P[i * 3 + 5] - P[i * 3 + 2]) * u + windZ * age;
          // The filament winds around a growing core as the vortex rolls up.
          const r = sh[3] + 0.22 * Math.sqrt(age);
          const phi = sh[4] + swirlSign * 7.5 * Math.sqrt(age);
          ay += r * Math.cos(phi);
          az += r * Math.sin(phi);
          ay = Math.max(ay, 0.15);
          // Ground effect ends the filament as it reaches the runway; age thins it out.
          const groundFade = THREE.MathUtils.clamp((ay - 0.2) / 2.5, 0, 1);
          alpha = sh[5] * Math.min(age / 0.25, 1) * (1 - age / (VORTEX_DT * VORTEX_POINTS)) * groundFade;
        }
        pos.setXYZ(o, ax, ay, az);
        pos.setXYZ(o + 1, ax, ay, az);
        fade.setX(o, alpha);
        fade.setX(o + 1, alpha);
        if (k === 0) prev.set(ax, ay, az);
        else {
          a.set(prev.x - ax, prev.y - ay, prev.z - az);
          const l = a.length() || 1;
          dir.setXYZ(o, a.x / l, a.y / l, a.z / l);
          dir.setXYZ(o + 1, a.x / l, a.y / l, a.z / l);
          if (k === 1) {
            dir.setXYZ(o - 2, a.x / l, a.y / l, a.z / l);
            dir.setXYZ(o - 1, a.x / l, a.y / l, a.z / l);
          }
          prev.set(ax, ay, az);
        }
      }
    });
    pos.needsUpdate = true;
    dir.needsUpdate = true;
    fade.needsUpdate = true;
  });
  return <mesh geometry={buffers.g} material={filamentMaterial} frustumCulled={false} renderOrder={12} />;
}

/* ---------------------------------------------------------------------------------------------- */
const FLOW_BOX = [190, 70, 190];
const GROUND_BOX = [320, 26, 320];

export default function WindFlow({ rig, rigged }) {
  const show = useUI((s) => s.showFlow);
  const episode = useUI((s) => s.episode);
  return (
    <group visible={show}>
      {rig.flow && <FlowLines flow={rig.flow} rigged={rigged} />}
      <Streaks count={170} box={FLOW_BOX} heightCenter={4} followHeight width={0.012} opacity={0.045} mode="flow" color="#cfe6ff" />
      <Streaks count={150} box={GROUND_BOX} heightCenter={6} followHeight={false} width={0.03} opacity={0.035} mode="wind" color="#e9f4ff" />
      <Vortices episode={episode} />
    </group>
  );
}
