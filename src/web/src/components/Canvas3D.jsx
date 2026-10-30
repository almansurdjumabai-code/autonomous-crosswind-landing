import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, Lightformer, OrbitControls, Sky, useGLTF } from "@react-three/drei";
import { Bloom, DepthOfField, EffectComposer, ToneMapping, Vignette } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import { clock, presetDpr, QUALITY, ui, useUI } from "../lib/store.js";
import { createSample, DEG, sampleEpisode } from "../lib/flightData.js";
import { enhanceGroundMaterial, setTextureRepeat } from "../lib/groundShader.js";
import WindFlow from "./WindFlow.jsx";

const BASE = import.meta.env.BASE_URL;
const MODEL_URL = `${BASE}models/scene.glb`;

// Fallback sun (~14 deg elevation, west-south-west); the rig manifest carries the direction the
// exporter used to bake the apron shadows, so the runtime and the bake always agree.
const DEFAULT_SUN = [-0.9, 0.25, 0.33];
const FOG_COLOR = new THREE.Color("#b9b2a8");
// Only objects the 96 m shadow box around the aircraft can reach: approach masts and runway light
// fixtures (merged into Static_Light_Fixture by the exporter), the navaid masts and the windsock.
const SHADOW_CASTER_PREFIXES = ["Static_Light_Fixture", "Navaid", "Windsock"];
const RUNWAY_LIGHT_MATERIALS = ["GLB_RWY_Light_White", "GLB_RWY_Light_Yellow", "GLB_RWY_Light_Green", "GLB_RWY_Light_Red", "GLB_Taxi_Light_Blue"];
const UP = new THREE.Vector3(0, 1, 0);
const NEG_Z = new THREE.Vector3(0, 0, -1);
const AIRFRAME_PREFIXES = [
  "Fuselage", "Wing", "Aileron", "Elevator", "Stabilizer", "Fin", "Dorsal", "Rudder", "Strut", "Oleo", "Axle",
  "Wheel", "Engine", "Fan", "Spoiler", "Gear", "Title", "Registration",
];
// Airframe parts too small or too enclosed to show in a 5 cm/texel shadow map.
const NO_SHADOW_PREFIXES = ["Registration", "Title", "Strobe", "Nav_", "Landing_Light", "Fan", "Engine_Duct", "Axle", "Wing_Canoe"];
const COCKPIT_HIDDEN = ["Fuselage", "Fuselage_Fairing", "Dorsal_Fin", "Registration_L", "Registration_R", "Title_L", "Title_R", "Landing_Light_Lens_N", "Gear_Nose_Doors", "Strobe_Top", "Strobe_Belly"];
const GEAR_INDEX = { nose: 0, main_left: 1, main_right: 2 };

useGLTF.preload(MODEL_URL);

/* ---------------------------------------------------------------------------------------------- */
/* Model + rig binding                                                                             */
/* ---------------------------------------------------------------------------------------------- */
function useRiggedScene(rig) {
  const { scene } = useGLTF(MODEL_URL);
  const gl = useThree((s) => s.gl);
  return useMemo(() => {
    const nodes = {};
    const lightMaterials = new Map();
    let strobeMaterial = null;
    // 8x is visually indistinguishable from 16x on the runway at approach angles and roughly halves
    // the texture bandwidth of the ground, which fills most of the screen on short final.
    const aniso = Math.min(gl.capabilities.getMaxAnisotropy(), 8);
    scene.traverse((o) => {
      if (o.name) nodes[o.name] = o;
      if (!o.isMesh) return;
      // Multi-material meshes load as a group whose primitives are named "<node>_<n>";
      // classify those by the owning node, everything else by its own name.
      const stem = o.name.replace(/_\d+$/, "");
      const owner = o.parent && o.parent.name === stem ? stem : o.name;
      const airframe = AIRFRAME_PREFIXES.some((p) => owner.startsWith(p));
      o.castShadow = (airframe && !NO_SHADOW_PREFIXES.some((p) => owner.startsWith(p))) || SHADOW_CASTER_PREFIXES.some((p) => owner.startsWith(p));
      o.receiveShadow = true;
      const m = o.material;
      if (!m) return;
      for (const tex of [m.map, m.normalMap, m.roughnessMap, m.metalnessMap, m.aoMap]) if (tex) tex.anisotropy = aniso;
      // Ground surfaces: near-field detail + far-field anti-tiling layers (see groundShader.js).
      if (m.name?.startsWith("GLB_Asphalt")) {
        m.normalScale = new THREE.Vector2(1.2, 1.2);
        m.envMapIntensity = 0.8;
        enhanceGroundMaterial(m, { detailScale: 6.3, detailStrength: 0.6, macroScale: 0.07, macroStrength: 0.35 });
      } else if (m.name?.startsWith("GLB_Grass")) {
        if (!m.userData.repeatSet) {
          setTextureRepeat(m, 4); // 10 m tiles instead of 40 m: crisp blades near the runway
          m.userData.repeatSet = true;
        }
        enhanceGroundMaterial(m, { detailScale: 7.7, detailStrength: 0.65, macroScale: 0.045, macroStrength: 0.5, normalDetail: 0.5 });
      } else if (m.name?.startsWith("GLB_Concrete")) {
        // Slab joints live on the tile border: keep the detail layer subtle so it never grids.
        enhanceGroundMaterial(m, { detailScale: 5.3, detailStrength: 0.2, macroScale: 0.11, macroStrength: 0.12, normalDetail: 0.0 });
      }
      if (m.metalness > 0.5) m.envMapIntensity = 1.35;
      if (m.name?.startsWith("GLB_Air_")) {
        for (const tex of [m.map, m.normalMap, m.roughnessMap, m.metalnessMap]) if (tex) tex.anisotropy = gl.capabilities.getMaxAnisotropy();
        m.envMapIntensity = 1.15;
      }
      if (m.name === "GLB_Strobe_White") strobeMaterial = m;
      if (RUNWAY_LIGHT_MATERIALS.includes(m.name)) lightMaterials.set(m, m.emissiveIntensity);
      if (m.name === "GLB_Grass_Terrain" || m.name === "GLB_Grass") {
        // Blade-scale relief only; a strong normal map reads as tiling ripples at grazing angles.
        m.normalScale = new THREE.Vector2(0.45, 0.45);
      }
      if (owner === "Mountains") {
        o.castShadow = false;
        m.envMapIntensity = 0.4;
      }
    });
    const surfaces = Object.entries(rig.surfaces).map(([name, s]) => ({
      node: nodes[name],
      axis: new THREE.Vector3(...s.hinge_axis_local).normalize(),
      sign: s.sign,
      key: { elevator_deg: "elev", aileron_deg: "ail", rudder_deg: "rud" }[s.channel],
    }));
    const gear = Object.entries(rig.gear).map(([name, g]) => {
      const hub = nodes[`Hub_${name}`];
      const attach = new THREE.Vector3(...g.attach_local);
      const rest = hub.position.clone();
      const stretch = [nodes[g.strut_node], nodes[g.spring_node]]
        .filter(Boolean)
        .map((n) => ({ node: n, s0: n.scale.clone(), len0: rest.distanceTo(attach) }));
      return {
        name,
        pen: 0,
        index: GEAR_INDEX[name],
        hub,
        rest,
        attach,
        stretch,
        wheel: nodes[g.wheel_node],
        radius: g.wheel_radius_m,
      };
    });
    return {
      scene,
      nodes,
      aircraft: nodes.Aircraft,
      surfaces,
      gear,
      fans: (rig.fans ?? []).map((n) => nodes[n]).filter(Boolean),
      spoilers: Object.entries(rig.spoilers ?? {}).map(([name, s]) => ({ node: nodes[name], axis: new THREE.Vector3(...s.hinge_axis_local).normalize() })),
      strobeMaterial,
      steer: nodes.Nose_Steer,
      lightMaterials,
      cockpitHidden: COCKPIT_HIDDEN.map((n) => nodes[n]).filter(Boolean),
    };
  }, [scene, rig, gl]);
}

/* ---------------------------------------------------------------------------------------------- */
/* Tire smoke: instanced soft billboards, emission scaled by touchdown sink rate                    */
/* ---------------------------------------------------------------------------------------------- */
const MAX_PARTICLES = 900;

const smokeMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  uniforms: { uColor: { value: new THREE.Color("#d9d6d0") } },
  vertexShader: /* glsl */ `
    attribute float aAlpha;
    attribute float aSeed;
    varying float vAlpha;
    varying float vSeed;
    varying vec2 vUv;
    void main() {
      vUv = uv;
      vAlpha = aAlpha;
      vSeed = aSeed;
      vec4 mv = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      float s = length(instanceMatrix[0].xyz);
      mv.xy += position.xy * s;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor;
    varying float vAlpha;
    varying float vSeed;
    varying vec2 vUv;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    float noise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
    }
    void main() {
      vec2 p = vUv - 0.5;
      float r = length(p) * 2.0;
      float n = noise(p * 4.0 + vSeed * 17.0) * 0.6 + noise(p * 9.0 - vSeed * 5.0) * 0.4;
      float a = smoothstep(1.0, 0.15, r + (n - 0.5) * 0.45) * vAlpha;
      if (a < 0.004) discard;
      vec3 col = uColor * (0.82 + 0.28 * (1.0 - r)) * (0.9 + 0.2 * n);
      gl_FragColor = vec4(col, a);
      #include <colorspace_fragment>
    }`,
});

function TireSmoke({ rigged, episodeRef, simRef }) {
  const meshRef = useRef();
  const state = useMemo(() => {
    const geo = new THREE.PlaneGeometry(1, 1);
    const alpha = new THREE.InstancedBufferAttribute(new Float32Array(MAX_PARTICLES), 1);
    const seed = new THREE.InstancedBufferAttribute(Float32Array.from({ length: MAX_PARTICLES }, () => Math.random()), 1);
    alpha.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aAlpha", alpha);
    geo.setAttribute("aSeed", seed);
    return {
      geo,
      alpha,
      particles: Array.from({ length: MAX_PARTICLES }, () => ({
        alive: false, p: new THREE.Vector3(), v: new THREE.Vector3(), age: 0, life: 3, s0: 0.3, s1: 2, a0: 1,
      })),
      cursor: 0,
      spray: [0, 0, 0], // remaining continuous-emission time per gear leg
      sprayVz: [0, 0, 0],
      lastT: -1,
      lastEpisode: null,
      tmpM: new THREE.Matrix4(),
      tmpQ: new THREE.Quaternion(),
      tmpS: new THREE.Vector3(),
      contact: new THREE.Vector3(),
      wind: new THREE.Vector3(),
    };
  }, []);

  const advance = (pt, dt, wind) => {
    const relax = 1 - Math.exp(-dt / 0.55);
    pt.age += dt;
    pt.v.x += (wind.x * 0.9 - pt.v.x) * relax;
    pt.v.z += (wind.z * 0.9 - pt.v.z) * relax;
    pt.v.y += (0.35 - pt.v.y) * relax;
    pt.p.addScaledVector(pt.v, dt);
  };

  // Reconstruct the plume after a forward seek: emit at the true contact positions along the
  // recorded path and fast-forward each particle by its age, so scrubbing is deterministic.
  const rebuild = (ep, t) => {
    const st = state;
    st.particles.forEach((p) => (p.alive = false));
    st.spray = [0, 0, 0];
    const inv = new THREE.Matrix4().copy(rigged.aircraft.matrixWorld).invert();
    const m = new THREE.Matrix4();
    const smp = createSample();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const vel = new THREE.Vector3();
    for (const g of rigged.gear) {
      const c = ep.contacts[g.index];
      if (!c || t < c.t || t - c.t > 6) continue;
      const rel = new THREE.Matrix4().multiplyMatrices(inv, g.hub.parent.matrixWorld);
      const local = g.rest.clone();
      local.y -= g.radius;
      local.applyMatrix4(rel);
      const m01 = Math.min(Math.abs(c.vz) / 1.5, 2.0);
      const scale = g.name === "nose" ? 0.35 : 1;
      const sprayDur = 0.35 + 0.45 * m01;
      const emitAt = (tau, count, vz) => {
        sampleEpisode(ep, c.t + tau, smp);
        q.set(smp.quat[0], smp.quat[1], smp.quat[2], smp.quat[3]);
        m.compose(pos.set(smp.pos[0], smp.pos[1], smp.pos[2]), q, new THREE.Vector3(1, 1, 1));
        const contact = local.clone().applyMatrix4(m);
        contact.y = Math.max(contact.y, 0);
        vel.set(0, 0, -smp.gs);
        st.wind.set(smp.wind[0], smp.wind[1], smp.wind[2]);
        for (let k = 0; k < count; k++) {
          const pt = spawn(contact, vel, st.wind, vz);
          const age = t - (c.t + tau);
          for (let a = 0; a < age; a += 1 / 30) advance(pt, Math.min(1 / 30, age - a), st.wind);
          if (pt.age >= pt.life) pt.alive = false;
        }
      };
      emitAt(0, Math.round((18 + 150 * m01) * scale), c.vz);
      for (let tau = 1 / 30; tau < Math.min(sprayDur, t - c.t); tau += 1 / 30) {
        emitAt(tau, Math.round((140 * m01 * scale) / 30 + 0.5), c.vz * 0.6);
      }
    }
  };

  const spawn = (pos, vAircraft, wind, vz) => {
    const st = state;
    const pt = st.particles[st.cursor];
    st.cursor = (st.cursor + 1) % MAX_PARTICLES;
    const m = Math.min(Math.abs(vz) / 1.5, 2.0);
    pt.alive = true;
    pt.age = 0;
    pt.life = 2.6 + 1.8 * Math.random() + 0.8 * m;
    pt.s0 = 0.18 + 0.32 * m;
    pt.s1 = 1.3 + 2.2 * m * (0.6 + 0.6 * Math.random());
    pt.a0 = 0.35 + 0.45 * Math.min(m, 1);
    // Pooled particles, no allocation: touchdown bursts spawn ~170 at once and must not churn the GC.
    pt.p.copy(pos);
    pt.p.x += (Math.random() - 0.5) * 0.3;
    pt.p.y += 0.05 + Math.random() * 0.1;
    pt.p.z += (Math.random() - 0.5) * 0.3;
    pt.v.copy(vAircraft).multiplyScalar(0.18 + 0.1 * Math.random());
    pt.v.x += (Math.random() - 0.5) * 1.2 + wind.x * 0.2;
    pt.v.z += (Math.random() - 0.5) * 1.2 + wind.z * 0.2;
    pt.v.y = 0.3 + Math.random() * 0.6;
    return pt;
  };

  useFrame(() => {
    const ep = episodeRef.current;
    const mesh = meshRef.current;
    if (!ep || !mesh) return;
    const st = state;
    const t = simRef.current.t;
    const dt = Math.max(0, Math.min(t - st.lastT, 0.1));
    // Scenario change, backwards scrub or forward seek: rebuild the plume for time t.
    if (st.lastEpisode !== ep || t < st.lastT - 1e-6 || t - st.lastT > 0.5) {
      rebuild(ep, t);
      st.lastEpisode = ep;
      st.lastT = t;
    }
    const s = clock.sample;
    if (s) st.wind.set(s.wind[0], s.wind[1], s.wind[2]);
    const vAircraft = simRef.current.velocity;

    // Touchdown events: each gear leg emits a burst scaled by the recorded sink rate at contact.
    for (const g of rigged.gear) {
      const c = ep.contacts[g.index];
      if (!c) continue;
      if (st.lastT < c.t && t >= c.t) {
        const vz = c.vz;
        const burst = Math.round(18 + 150 * Math.min(Math.abs(vz) / 1.5, 2.0)) * (g.name === "nose" ? 0.35 : 1);
        g.hub.getWorldPosition(st.contact);
        st.contact.y = Math.max(st.contact.y - g.radius, 0.0);
        for (let k = 0; k < burst; k++) spawn(st.contact, vAircraft, st.wind, vz);
        st.spray[g.index] = 0.35 + 0.45 * Math.min(Math.abs(vz) / 1.5, 2.0);
        st.sprayVz[g.index] = vz;
      }
      if (st.spray[g.index] > 0 && dt > 0) {
        st.spray[g.index] -= dt;
        const rate = 140 * Math.min(Math.abs(st.sprayVz[g.index]) / 1.5, 2.0) * (g.name === "nose" ? 0.3 : 1);
        const n = Math.floor(rate * dt + Math.random());
        g.hub.getWorldPosition(st.contact);
        st.contact.y = Math.max(st.contact.y - g.radius, 0.0);
        for (let k = 0; k < n; k++) spawn(st.contact, vAircraft, st.wind, st.sprayVz[g.index] * 0.6);
      }
    }

    // Integrate: velocity relaxes towards the wind (plus buoyant rise); size grows, density decays.
    let count = 0;
    for (const pt of st.particles) {
      if (!pt.alive) continue;
      if (dt > 0) advance(pt, dt, st.wind);
      if (pt.age >= pt.life) {
        pt.alive = false;
        continue;
      }
      const grow = 1 - Math.exp(-pt.age / 0.9);
      const size = pt.s0 + (pt.s1 - pt.s0) * grow;
      const fade = Math.min(pt.age / 0.08, 1) * Math.exp(-pt.age / 1.4) * (1 - pt.age / pt.life);
      st.tmpS.setScalar(size);
      st.tmpM.compose(pt.p, st.tmpQ, st.tmpS);
      mesh.setMatrixAt(count, st.tmpM);
      st.alpha.array[count] = pt.a0 * fade;
      count++;
    }
    mesh.count = count;
    mesh.visible = count > 0; // no empty instanced draw call for the whole approach
    mesh.instanceMatrix.needsUpdate = true;
    st.alpha.needsUpdate = true;
    st.lastT = t;
  });

  return (
    <instancedMesh ref={meshRef} args={[state.geo, smokeMaterial, MAX_PARTICLES]} frustumCulled={false} renderOrder={10} />
  );
}

/* ---------------------------------------------------------------------------------------------- */
/* Dynamic airfield lighting: sequenced approach flashers, PAPI, runway light intensity            */
/* ---------------------------------------------------------------------------------------------- */
function AirfieldLighting({ rigged, lighting }) {
  const flasherRef = useRef();
  const papiRef = useRef();
  const flashers = useMemo(() => lighting?.approach_flashers ?? [], [lighting]);
  const papi = useMemo(() => lighting?.papi ?? [], [lighting]);
  const hz = lighting?.flash_hz ?? 2;
  const tmp = useMemo(() => ({ m: new THREE.Matrix4(), c: new THREE.Color() }), []);
  const flasherMat = useMemo(() => new THREE.MeshBasicMaterial({ color: "#ffffff", toneMapped: false }), []);
  const papiMat = useMemo(() => new THREE.MeshBasicMaterial({ color: "#ffffff", toneMapped: false }), []);
  const sphere = useMemo(() => new THREE.SphereGeometry(0.32, 12, 8), []);
  const disc = useMemo(() => new THREE.CircleGeometry(0.22, 20), []);
  const RED = useMemo(() => new THREE.Color(9, 0.25, 0.12), []);
  const WHITE = useMemo(() => new THREE.Color(7, 6.4, 5.4), []);

  useEffect(() => {
    const fm = flasherRef.current;
    if (fm) {
      flashers.forEach((p, i) => {
        tmp.m.makeTranslation(p[0], p[1], p[2]);
        fm.setMatrixAt(i, tmp.m);
        fm.setColorAt(i, tmp.c.setRGB(0, 0, 0));
      });
      fm.instanceMatrix.needsUpdate = true;
      fm.computeBoundingSphere(); // static instances: let the renderer cull the whole bar
    }
    const pm = papiRef.current;
    if (pm) {
      papi.forEach((u, i) => {
        // Lenses face the approach (+Z, south) just in front of the housings.
        tmp.m.makeTranslation(u.position[0], u.position[1], u.position[2] + 0.05);
        pm.setMatrixAt(i, tmp.m);
        pm.setColorAt(i, WHITE);
      });
      pm.instanceMatrix.needsUpdate = true;
      pm.computeBoundingSphere();
    }
  }, [flashers, papi, tmp, WHITE]);

  useFrame(() => {
    const st = ui.get();
    // Sequenced flashers: far-to-near sweep at flash_hz, with a short dark gap between sweeps.
    const fm = flasherRef.current;
    if (fm && flashers.length) {
      const phase = (performance.now() / 1000) * hz;
      const sweep = (phase % 1) * flashers.length * 1.35;
      for (let i = 0; i < flashers.length; i++) {
        const on = Math.abs(sweep - i) < 0.6 ? 14 : 0;
        fm.setColorAt(i, tmp.c.setRGB(on, on, on * 0.95));
      }
      fm.instanceColor.needsUpdate = true;
    }
    // PAPI: each unit is white above its setting angle and red below it, with the narrow pink
    // transition sector of a real projector (+/- 0.05 deg).
    const pm = papiRef.current;
    const ac = rigged.aircraft.position;
    if (pm && papi.length) {
      papi.forEach((u, i) => {
        const dx = ac.x - u.position[0];
        const dz = ac.z - u.position[2];
        const elev = (Math.atan2(ac.y + 0.8 - u.position[1], Math.hypot(dx, dz)) * 180) / Math.PI;
        const w = THREE.MathUtils.clamp((elev - u.angle_deg + 0.05) / 0.1, 0, 1);
        tmp.c.copy(RED).lerp(WHITE, w);
        pm.setColorAt(i, tmp.c);
      });
      pm.instanceColor.needsUpdate = true;
    }
    // High-intensity runway lighting steps up as visibility drops.
    const step = 1 + 0.9 * (st.gustPct / 100);
    rigged.lightMaterials.forEach((base, mat) => {
      mat.emissiveIntensity = base * step;
    });
  });

  return (
    <>
      {flashers.length > 0 && <instancedMesh ref={flasherRef} args={[sphere, flasherMat, flashers.length]} />}
      {papi.length > 0 && <instancedMesh ref={papiRef} args={[disc, papiMat, papi.length]} />}
    </>
  );
}

/* ---------------------------------------------------------------------------------------------- */
/* Telemetry engine: advances the clock and drives every animated node at display rate              */
/* ---------------------------------------------------------------------------------------------- */
function FlightEngine({ rigged, episodeRef, simRef }) {
  const sample = useMemo(() => createSample(), []);
  const tmp = useMemo(
    () => ({ v: new THREE.Vector3(), d: new THREE.Vector3(), up: new THREE.Vector3(), prev: new THREE.Vector3(), q: new THREE.Quaternion() }),
    [],
  );
  const gl = useThree((s) => s.gl);
  const propAngle = useRef(0);
  const lastCamera = useRef(null);
  const smoothDt = useRef(1 / 60);

  useEffect(() => {
    gl.info.autoReset = false;
  }, [gl]);

  useFrame((_, delta) => {
    const t0 = performance.now();
    const st = ui.get();
    const ep = st.episode;
    clock.frame += 1;
    clock.fps = clock.fps * 0.92 + (1 / Math.max(delta, 1e-4)) * 0.08;
    clock.render.calls = gl.info.render.calls;
    clock.render.dpr = gl.getPixelRatio();
    clock.render.triangles = gl.info.render.triangles;
    gl.info.reset();
    if (!ep) return;

    const sim = simRef.current;
    if (episodeRef.current !== ep) {
      episodeRef.current = ep;
      sim.t = ep.t[0];
      sim.velocity.set(0, 0, 0);
      tmp.prev.set(ep.pos[0], ep.pos[1], ep.pos[2]);
    }
    // rAF timestamps jitter by a millisecond or two; at 35 m/s that is visible against the ground.
    // Filter the step, but follow real hitches at once so that the sim clock never drifts from wall time.
    const rawDt = Math.min(delta, 0.1);
    smoothDt.current = Math.abs(rawDt - smoothDt.current) > 0.012 ? rawDt : smoothDt.current + (rawDt - smoothDt.current) * 0.08;
    const dtWall = smoothDt.current;
    if (clock.seekTo != null) {
      sim.t = Math.min(Math.max(clock.seekTo, ep.t[0]), ep.tEnd);
      clock.seekTo = null;
    } else if (st.playing && st.autopilot === "ENGAGED") {
      sim.t += dtWall * st.speed;
      if (sim.t >= ep.tEnd) {
        sim.t = ep.tEnd;
        ui.set({ playing: false });
      }
    }
    sampleEpisode(ep, sim.t, sample);
    clock.t = sim.t;
    clock.sample = sample;
    clock.index = sample.i;
    clock.latencyMs = (sim.t - ep.t[sample.i]) * 1000;

    // Airframe pose.
    const ac = rigged.aircraft;
    tmp.v.set(sample.visPos[0], sample.visPos[1], sample.visPos[2]);
    const simDt = sim.t - (sim.lastT ?? sim.t);
    if (simDt > 1e-5) sim.velocity.subVectors(tmp.v, tmp.prev).divideScalar(simDt);
    tmp.prev.copy(tmp.v);
    sim.lastT = sim.t;
    ac.position.copy(tmp.v);
    ac.quaternion.set(sample.visQuat[0], sample.visQuat[1], sample.visQuat[2], sample.visQuat[3]);

    // Control surfaces about their true hinge axes.
    for (const s of rigged.surfaces) {
      if (!s.node) continue;
      s.node.quaternion.setFromAxisAngle(s.axis, s.sign * sample[s.key] * DEG * st.surfaceGain);
    }
    if (rigged.steer) rigged.steer.rotation.y = 10 * DEG * (sample.rud / 25);
    // Fans: the engines spool from ground idle to approach thrust, in sim time so a pause freezes them.
    const omega = (9 + 16 * Math.min(Math.max(sample.thr, 0), 1)) * 2 * Math.PI;
    propAngle.current = (propAngle.current + omega * Math.max(simDt, 0)) % (2 * Math.PI);
    for (const fan of rigged.fans) fan.rotation.x = propAngle.current;
    // Ground spoilers rise over 1.2 s from main-gear contact.
    const mainContact = ep.contacts[GEAR_INDEX.main_left]?.t ?? ep.contacts[GEAR_INDEX.main_right]?.t;
    const spoiler = mainContact == null ? 0 : THREE.MathUtils.smoothstep(sim.t, mainContact, mainContact + 1.2);
    for (const s of rigged.spoilers) s.node.quaternion.setFromAxisAngle(s.axis, spoiler * 42 * DEG);
    // Anti-collision strobes: double flash, 60 ms each, every 1.1 s.
    if (rigged.strobeMaterial) {
      const phase = sim.t % 1.1;
      rigged.strobeMaterial.emissiveIntensity = phase < 0.06 || (phase > 0.18 && phase < 0.24) ? 60 : 0.2;
    }
    ac.updateMatrixWorld(true);

    // Landing gear: wheels on the runway surface (same contact model as the simulator), struts
    // re-stretched to the hub. A leg that has touched extends as well as compresses, because the
    // long airliner wheelbase turns small pitch changes into real wheel-height changes.
    tmp.up.set(0, 1, 0).applyQuaternion(ac.quaternion);
    for (const g of rigged.gear) {
      tmp.d.copy(g.rest);
      tmp.d.y -= g.radius;
      g.hub.parent.localToWorld(tmp.d);
      const gap = -tmp.d.y / Math.max(tmp.up.y, 0.5);
      const touched = ep.contacts[g.index] && sim.t >= ep.contacts[g.index].t;
      const pen = THREE.MathUtils.clamp(gap, touched ? -0.5 : 0, 0.5);
      g.pen = Math.abs(simDt) > 0.2 ? pen : g.pen + (pen - g.pen) * (1 - Math.exp(-Math.max(simDt, 0) / 0.05));
      g.hub.position.copy(g.rest);
      g.hub.position.y += g.pen;
      tmp.d.subVectors(g.hub.position, g.attach);
      const len = tmp.d.length();
      tmp.d.normalize();
      for (const s of g.stretch) {
        s.node.quaternion.setFromUnitVectors(NEG_Z, tmp.d);
        s.node.scale.set(s.s0.x, s.s0.y, (s.s0.z * len) / s.len0);
      }
      const c = ep.contacts[g.index];
      if (g.wheel) g.wheel.rotation.z = c && sim.t >= c.t ? -(sample.x - ep.x[c.index]) / g.radius : 0;
    }

    // Cockpit view hides the cabin shell so the pilot sees out.
    if (lastCamera.current !== st.camera) {
      const hide = st.camera === "CAM-03";
      rigged.cockpitHidden.forEach((n) => (n.visible = !hide));
      lastCamera.current = st.camera;
    }
    clock.frameMs = performance.now() - t0;
  }, -2);

  return null;
}

/* ---------------------------------------------------------------------------------------------- */
/* Camera director: four modes, eased transitions, lag-filtered chase                               */
/* ---------------------------------------------------------------------------------------------- */
const CHASE_BACK = 85; // m behind along the runway
const CHASE_UP = 15; // m above the aircraft
const CHASE_FREE_X = 18; // m of lateral deviation shown before the camera starts to follow
const CHASE_VIEW = new THREE.Vector3(0, -Math.tan(10 * DEG), -1); // along the runway (-Z), 10 deg down
const easeInOutCubic = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

function CameraDirector({ rigged, focus }) {
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const mode = useUI((s) => s.camera);
  const controls = useRef();
  const st = useMemo(
    () => ({
      chase: new THREE.Vector3(60, 80, 2300),
      chaseInit: false,
      from: { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), fov: 38 },
      target: { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), fov: 38 },
      u: 1,
      look: new THREE.Vector3(),
      m: new THREE.Matrix4(),
      fwd: new THREE.Vector3(),
      right: new THREE.Vector3(),
      prevAc: new THREE.Vector3(),
      cockpitQ: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2),
      eye: new THREE.Vector3(),
    }),
    [],
  );

  // Mode switches are detected inside the frame loop rather than in an effect: a passive effect
  // can run a frame after the new mode is already rendering, which showed up as a one-frame jump
  // to the destination pose before the ease began.
  const beginTransition = () => {
    st.from.pos.copy(camera.position);
    st.from.quat.copy(camera.quaternion);
    st.from.fov = camera.fov;
    st.u = 0;
  };

  const lookPose = (pos, look, up, out) => {
    st.m.lookAt(pos, look, up);
    out.quat.setFromRotationMatrix(st.m);
    out.pos.copy(pos);
  };

  useFrame((_, delta) => {
    const ac = rigged.aircraft;
    const dt = Math.min(delta, 0.1);
    const P = ac.position;
    focus.copy(P);
    st.fwd.set(1, 0, 0).applyQuaternion(ac.quaternion);
    st.fwd.y = 0;
    if (st.fwd.lengthSq() < 1e-6) st.fwd.set(0, 0, -1);
    st.fwd.normalize();
    st.right.crossVectors(st.fwd, UP).normalize();

    // A teleport of the airframe (data load, seek, scenario switch) re-frames every camera mode
    // instead of easing across hundreds of metres.
    const teleported = st.prevAc.distanceToSquared(P) > 30 * 30;
    if (teleported) st.snap = true;
    // Chase camera state is always integrated so switching to it is seamless.
    // Runway-referenced chase: behind the aircraft along the runway axis and on the extended centreline,
    // so the centreline stays put on screen and the aircraft's offset from it reads directly. The camera
    // only slides sideways once the deviation would carry the aircraft under the side panels.
    st.eye.set(P.x - THREE.MathUtils.clamp(P.x, -CHASE_FREE_X, CHASE_FREE_X), P.y + CHASE_UP, P.z + CHASE_BACK);
    if (!st.chaseInit || teleported) {
      st.chase.copy(st.eye);
      st.chaseInit = true;
    }
    st.chase.lerp(st.eye, 1 - Math.exp(-dt / 0.45));

    if (st.mode !== mode) {
      // Ease between modes, but open the initial (or deep-linked) view already framed.
      if (st.mode === undefined) st.snap = true;
      else beginTransition();
      st.mode = mode;
    }

    const tgt = st.target;
    if (mode === "CAM-01") {
      // Fixed attitude, parallel to the runway: heading and bank changes never swing the view.
      st.look.copy(st.chase).add(CHASE_VIEW);
      lookPose(st.chase, st.look, UP, tgt);
      tgt.fov = 38;
    } else if (mode === "CAM-02") {
      st.eye.set(-58, 1.7, -372);
      lookPose(st.eye, P, UP, tgt);
      const dist = st.eye.distanceTo(P);
      tgt.fov = THREE.MathUtils.clamp((2 * Math.atan(26 / dist)) / DEG, 6, 50);
    } else if (mode === "CAM-03") {
      st.eye.set(14.9, 3.15, -0.48).applyQuaternion(ac.quaternion).add(P);
      tgt.pos.copy(st.eye);
      tgt.quat.copy(ac.quaternion).multiply(st.cockpitQ);
      tgt.fov = 62;
    } else {
      st.eye.copy(P).addScaledVector(st.fwd, -32).addScaledVector(st.right, -42).addScaledVector(UP, 14);
      lookPose(st.eye, P, UP, tgt);
      tgt.fov = 45;
    }

    if (st.snap) {
      st.u = 1;
      camera.position.copy(tgt.pos);
      camera.quaternion.copy(tgt.quat);
      camera.fov = tgt.fov;
      if (controls.current) controls.current.target.copy(P);
      st.prevAc.copy(P); // the follow step below must not re-apply the teleport displacement
      st.snap = false;
    }
    const orbiting = mode === "CAM-04" && st.u >= 1;
    if (controls.current) controls.current.enabled = orbiting;
    if (st.u < 1) {
      st.u = Math.min(1, st.u + dt / 1.25);
      const e = easeInOutCubic(st.u);
      camera.position.lerpVectors(st.from.pos, tgt.pos, e);
      camera.quaternion.slerpQuaternions(st.from.quat, tgt.quat, e);
      camera.fov = THREE.MathUtils.lerp(st.from.fov, tgt.fov, e);
      if (st.u >= 1 && mode === "CAM-04" && controls.current) controls.current.target.copy(P);
    } else if (orbiting) {
      camera.position.add(st.look.subVectors(P, st.prevAc));
      if (controls.current) {
        controls.current.target.copy(P);
        controls.current.update();
      }
    } else {
      camera.position.copy(tgt.pos);
      camera.quaternion.copy(tgt.quat);
      camera.fov = tgt.fov;
    }
    camera.near = mode === "CAM-03" ? 0.05 : 0.6;
    // External views place the airframe in the upper third, clear of the PFD; the cockpit view
    // stays centred so the HUD symbology is conformal.
    const lift = mode === "CAM-03" ? 0 : 0.14 * size.height;
    camera.setViewOffset(size.width, size.height, 0, lift, size.width, size.height);
    camera.updateProjectionMatrix();
    st.prevAc.copy(P);
  }, -1);

  return <OrbitControls ref={controls} enabled={false} enablePan={false} minDistance={24} maxDistance={320} enableDamping dampingFactor={0.08} />;
}

/* ---------------------------------------------------------------------------------------------- */
/* Environment, lighting, post-processing                                                          */
/* ---------------------------------------------------------------------------------------------- */
const SHADOW_HALF = 48; // m, half-size of the sun's orthographic shadow box

function Atmosphere({ rigged, sunDir }) {
  const sun = useRef();
  const fog = useRef();
  const quality = useUI((s) => s.quality);
  const mapSize = QUALITY[quality].shadowMap;
  useEffect(() => {
    // Resizing a shadow map requires discarding the old render target.
    const light = sun.current;
    if (!light) return;
    light.shadow.mapSize.set(mapSize, mapSize);
    light.shadow.map?.dispose();
    light.shadow.map = null;
  }, [mapSize]);
  const target = useMemo(() => new THREE.Object3D(), []);
  // Light-space axes for snapping the shadow box to whole texels: a box that slides with the
  // aircraft by sub-texel amounts makes every shadow edge crawl.
  const basis = useMemo(() => {
    const right = new THREE.Vector3().crossVectors(UP, sunDir).normalize();
    const up = new THREE.Vector3().crossVectors(sunDir, right).normalize();
    return { right, up, p: new THREE.Vector3() };
  }, [sunDir]);
  const scene = useThree((s) => s.scene);
  useEffect(() => {
    scene.add(target);
    return () => scene.remove(target);
  }, [scene, target]);

  useFrame(() => {
    const P = rigged.aircraft.position;
    const st = ui.get();
    if (sun.current) {
      const texel = (2 * SHADOW_HALF) / mapSize;
      const { right, up, p } = basis;
      const r = P.dot(right);
      const u = P.dot(up);
      p.copy(P)
        .addScaledVector(right, Math.round(r / texel) * texel - r)
        .addScaledVector(up, Math.round(u / texel) * texel - u);
      sun.current.position.copy(p).addScaledVector(sunDir, 180);
      target.position.copy(p);
      sun.current.target = target;
    }
    if (fog.current) {
      // Visibility degrades with gust intensity and crosswind-driven dust/haze.
      // About 0.7 transmittance to the mountains at 8 km in calm air; a few km visibility in severe gusts.
      const density = 0.00007 + 0.00024 * (st.gustPct / 100) + 0.000003 * st.crosswindKt;
      fog.current.density += (density - fog.current.density) * 0.05;
    }
  });

  return (
    <>
      <fogExp2 ref={fog} attach="fog" args={[FOG_COLOR, 0.0001]} />
      <Sky
        distance={20000}
        sunPosition={sunDir.clone().multiplyScalar(1000).toArray()}
        turbidity={7}
        rayleigh={2.4}
        mieCoefficient={0.0062}
        mieDirectionalG={0.86}
      />
      <hemisphereLight args={["#cdd9e8", "#5a4a33", 0.95]} />
      <directionalLight
        ref={sun}
        color="#ffd2a1"
        intensity={3.6}
        castShadow
        shadow-mapSize={[mapSize, mapSize]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.03}
        shadow-camera-left={-SHADOW_HALF}
        shadow-camera-right={SHADOW_HALF}
        shadow-camera-top={SHADOW_HALF}
        shadow-camera-bottom={-SHADOW_HALF}
        shadow-camera-near={1}
        shadow-camera-far={500}
      />
      <Environment resolution={256} frames={1}>
        <color attach="background" args={["#6f7f92"]} />
        <Lightformer form="rect" intensity={6} color="#ffc48a" position={[-30, 4, 12]} scale={[60, 8, 1]} target={[0, 0, 0]} />
        <Lightformer form="ring" intensity={1.4} color="#bcd2ec" position={[0, 40, 0]} scale={40} target={[0, 0, 0]} />
        <Lightformer form="rect" intensity={0.8} color="#3b3226" position={[0, -10, 0]} rotation-x={Math.PI / 2} scale={[100, 100, 1]} />
        <Lightformer form="rect" intensity={1.2} color="#90a6c4" position={[30, 10, -20]} scale={[40, 20, 1]} target={[0, 0, 0]} />
      </Environment>
    </>
  );
}

function PostFX({ focus }) {
  const camera = useUI((s) => s.camera);
  const quality = useUI((s) => s.quality);
  const preset = QUALITY[quality];
  const gl = useThree((st) => st.gl);
  const dof = preset.dof && camera !== "CAM-03";
  const dofRef = useRef(null);
  useEffect(() => {
    // Without the composer the renderer tone-maps directly; with it, the ToneMapping effect does.
    gl.toneMapping = preset.post ? THREE.NoToneMapping : THREE.AgXToneMapping;
    clock.render.passes = preset.post
      ? ["RENDER", "BLOOM", ...(dof ? ["DOF"] : []), ...(preset.smaa ? ["SMAA"] : []), ...(preset.msaa ? [`MSAA${preset.msaa}`] : []), "VIGNETTE", "AGX"]
      : ["RENDER", "AGX"];
  }, [gl, preset, dof]);
  // R3F copies a Vector3 prop only when it changes identity, so the autofocus target is
  // pushed into the effect every frame (after the camera director has updated `focus`).
  useFrame(() => {
    if (dofRef.current?.target) dofRef.current.target.copy(focus);
  });
  if (!preset.post) return null;
  if (preset.dof) {
    return (
      <EffectComposer key="cinematic" multisampling={preset.msaa} enableNormalPass={false}>
        <Bloom mipmapBlur intensity={0.85} luminanceThreshold={0.92} luminanceSmoothing={0.25} radius={0.75} />
        <DepthOfField ref={dofRef} target={focus} worldFocusRange={70} bokehScale={dof ? 2.4 : 0} height={720} />
        <Vignette eskil={false} offset={0.28} darkness={0.72} />
        <ToneMapping mode={ToneMappingMode.AGX} />
      </EffectComposer>
    );
  }
  // BALANCED: half-resolution bloom with a short mip chain, MSAA for clean airframe edges, no DOF.
  return (
    <EffectComposer key="balanced" multisampling={preset.msaa} enableNormalPass={false}>
      <Bloom mipmapBlur levels={5} resolutionScale={0.5} intensity={0.7} luminanceThreshold={0.95} luminanceSmoothing={0.2} radius={0.65} />
      <Vignette eskil={false} offset={0.28} darkness={0.68} />
      <ToneMapping mode={ToneMappingMode.AGX} />
    </EffectComposer>
  );
}

/**
 * Fixed pixel ratio per preset. Runtime resolution scaling is deliberately absent: it traded a few
 * FPS for a visibly soft, smeared viewport, and the scene now holds 60 FPS without it.
 */
function FixedResolution() {
  const quality = useUI((s) => s.quality);
  const setDpr = useThree((st) => st.setDpr);
  useEffect(() => {
    setDpr(presetDpr(quality));
  }, [quality, setDpr]);
  return null;
}

/**
 * Upload every texture and compile every program once the world is mounted, so the first frame a
 * hangar, the parked fleet or the runway comes into view (e.g. on a camera switch) costs nothing.
 */
function Prewarm() {
  const gl = useThree((st) => st.gl);
  const scene = useThree((st) => st.scene);
  const camera = useThree((st) => st.camera);
  useEffect(() => {
    const seen = new Set();
    scene.traverse((o) => {
      for (const m of [o.material].flat()) {
        if (!m || seen.has(m)) continue;
        seen.add(m);
        for (const v of Object.values(m)) if (v?.isTexture && !v.isRenderTargetTexture) gl.initTexture(v);
      }
    });
    gl.compileAsync(scene, camera).catch(() => {});
  }, [gl, scene, camera]);
  return null;
}

function World({ rig }) {
  const rigged = useRiggedScene(rig);
  const sunDir = useMemo(() => new THREE.Vector3(...(rig.lighting?.sun_direction ?? DEFAULT_SUN)).normalize(), [rig]);
  const episodeRef = useRef(null);
  const simRef = useRef({ t: 0, lastT: null, velocity: new THREE.Vector3() });
  const focus = useMemo(() => new THREE.Vector3(), []);
  return (
    <>
      <primitive object={rigged.scene} />
      <FlightEngine rigged={rigged} episodeRef={episodeRef} simRef={simRef} />
      <TireSmoke rigged={rigged} episodeRef={episodeRef} simRef={simRef} />
      <CameraDirector rigged={rigged} focus={focus} />
      <Atmosphere rigged={rigged} sunDir={sunDir} />
      <AirfieldLighting rigged={rigged} lighting={rig.lighting} />
      <WindFlow rig={rig} rigged={rigged} />
      <FixedResolution />
      <PostFX focus={focus} />
      <Prewarm />
    </>
  );
}

export default function Canvas3D({ rig }) {
  return (
    <Canvas
      shadows="percentage"
      dpr={presetDpr(ui.get().quality)}
      gl={{ antialias: false, powerPreference: "high-performance", toneMapping: THREE.NoToneMapping }}
      camera={{ fov: 38, near: 0.6, far: 24000, position: [60, 80, 2300] }}
      onCreated={({ gl }) => {
        clock.render.api = gl.capabilities.isWebGL2 ? "WEBGL2" : "WEBGL1";
      }}
    >
      <Suspense fallback={null}>{rig && <World rig={rig} />}</Suspense>
    </Canvas>
  );
}
