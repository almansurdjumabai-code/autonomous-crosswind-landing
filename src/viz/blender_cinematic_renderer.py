"""Cinematic teaser renderer for the autonomous crosswind landing system (Blender 4.2+ / 5.x).

Builds a complete, physically driven scene from a ``crosswind-landing-telemetry``
JSON episode (see ``src/utils/telemetry.py``) and renders a three-shot teaser:

1. **Low-angle tracking shot** of the aircraft crabbing and banking into the crosswind.
2. **Close-up** from a chase mount on the airframe: ailerons, elevator and rudder
   moving exactly as the controller commanded them during gust corrections.
3. **Chase camera through touchdown** with tire-smoke puffs at the exact
   main-gear contact instants, drifting with the simulated wind.

All motion comes from the telemetry: the airframe pose is keyframed at every
simulation sample with auto-clamped Bezier handles; control surfaces rotate about
their true hinge lines; landing-gear springs compress by the geometric
penetration of each wheel's contact point computed from the recorded pose (the
same contact model as the simulator); wheels spin up at touchdown; the
propeller speed follows the throttle; camera shake scales with the recorded
Dryden-turbulence + discrete-gust intensity.

Everything (airframe, runway with weathered markings, approach/edge/threshold
lights, PAPI, terrain, tree line, volumetric clouds and haze) is procedural, so
the script has no asset dependencies.

Usage (headless)::

    blender --background --python src/viz/blender_cinematic_renderer.py -- \\
        --telemetry results/telemetry_demo/baseline_xw7.5_s1_00000.json \\
        --output renders/crosswind_teaser.mov

    # fast look-development pass (720p, low samples, every 2nd frame):
    blender -b --python src/viz/blender_cinematic_renderer.py -- --telemetry ep.json --preview

    # build the scene only and save it for inspection:
    blender -b --python src/viz/blender_cinematic_renderer.py -- --telemetry ep.json --no-render --save-blend teaser.blend

Inside a running Blender session the script builds a new scene
(``CrosswindTeaser``) instead of touching the current one::

    import runpy; runpy.run_path("src/viz/blender_cinematic_renderer.py",
                                 run_name="__main__")  # reads sys.argv after "--"
    # or:  from blender_cinematic_renderer import RenderConfig, build_scene

Output defaults to a 3840x2160, 30 fps QuickTime container with the lossless
QuickTime Animation (QTRLE) codec, plus a ``<output>.timeline.json`` sidecar
(fps, frame range, shot boundaries, touchdown frame) for Remotion
composition. ``--codec`` also offers ``prores`` (ProRes 4444), ``ffv1`` (lossless,
MKV), ``png`` (lossless PNG-in-MOV), ``h264`` (lossy review proxy) and
``png-sequence`` (16-bit PNG frames).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector, noise

SCENE_NAME = "CrosswindTeaser"
COLLECTION_NAME = "CrosswindTeaser"


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class RenderConfig:
    telemetry: str
    output: str = "renders/crosswind_teaser.mov"
    engine: str = "EEVEE"  # "EEVEE" | "CYCLES"
    resolution: tuple[int, int] = (3840, 2160)
    fps: int = 30
    samples: int = 64
    codec: str = "qtrle"  # qtrle | prores | ffv1 | png | h264 | png-sequence
    mood: str = "golden"  # golden | storm
    preview: bool = False
    pre_roll_s: float = 14.0  # footage before touchdown
    post_roll_s: float = 3.0  # footage after touchdown (bounded by the episode)
    shot2_start_before_td_s: float = 6.5
    shot3_start_before_td_s: float = 2.2
    surface_gain: float = 1.0  # 1.0 = true deflections; >1 exaggerates for readability
    surface_source: str = "deflection"  # "deflection" (actuator state) | "action" (raw policy output)
    smoke_puffs_per_wheel: int = 36
    seed: int = 7
    render: bool = True
    save_blend: Optional[str] = None
    frame_step: int = 1
    frame_range: Optional[tuple[int, int]] = None  # render a sub-range (e.g. for farm splitting)
    aircraft: str = "airliner"  # airliner | light

    def apply_preview(self) -> None:
        self.resolution = (1280, 720)
        self.samples = 16
        self.codec = "h264"
        self.frame_step = 2
        self.smoke_puffs_per_wheel = 14
        if self.output.endswith(".mov"):
            self.output = self.output[:-4] + "_preview.mp4"


# =============================================================================
# Small utilities
# =============================================================================
def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def enum_items(struct, prop: str) -> list[str]:
    return [i.identifier for i in struct.bl_rna.properties[prop].enum_items]


def set_first(struct, prop: str, *candidates) -> Optional[object]:
    """Assign the first candidate value the RNA property accepts (version-tolerant)."""
    for value in candidates:
        try:
            setattr(struct, prop, value)
            return value
        except (TypeError, ValueError, AttributeError):
            continue
    print(f"[teaser] note: none of {candidates} accepted for {type(struct).__name__}.{prop}")
    return None


def set_socket(node, name: str, value) -> bool:
    for sock in node.inputs:
        if sock.name == name and getattr(sock, "enabled", True):
            try:
                sock.default_value = value
                return True
            except (TypeError, ValueError):
                continue
    return False


def link_object(obj: bpy.types.Object, coll: bpy.types.Collection, parent: Optional[bpy.types.Object] = None):
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def new_empty(name: str, coll, parent=None, location=(0, 0, 0), size: float = 0.3, kind: str = "PLAIN_AXES"):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = kind
    obj.empty_display_size = size
    obj.location = location
    return link_object(obj, coll, parent)


def mesh_object(name: str, bm: bmesh.types.BMesh, coll, parent=None, materials: Sequence = (), smooth: bool = False):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    for m in materials:
        me.materials.append(m)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    return link_object(obj, coll, parent)


# =============================================================================
# Animation helpers (legacy + layered Actions)
# =============================================================================
def _channelbag_fcurves(id_data):
    ad = id_data.animation_data
    if ad is None or ad.action is None:
        return []
    action = ad.action
    if hasattr(action, "fcurves") and not getattr(action, "is_action_layered", False):
        return action.fcurves
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(action, ad.action_slot)
    return bag.fcurves if bag is not None else []


def find_fcurve(id_data, data_path: str, index: int):
    for fc in _channelbag_fcurves(id_data):
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    return None


def keyframe_series(
    id_data,
    data_path: str,
    index: int,
    frames: Sequence[float],
    values: Sequence[float],
    interpolation: str = "BEZIER",
    handles: str = "AUTO_CLAMPED",
) -> None:
    """Bulk-create one F-curve (fast foreach_set path) with uniform interpolation."""
    n = len(frames)
    if n == 0:
        return
    is_array = hasattr(id_data.path_resolve(data_path), "__len__")
    if is_array:
        id_data.keyframe_insert(data_path, index=index, frame=float(frames[0]))
    else:
        id_data.keyframe_insert(data_path, frame=float(frames[0]))
        index = 0
    fc = find_fcurve(id_data, data_path, index)
    if fc is None:
        raise RuntimeError(f"could not create F-curve {data_path}[{index}] on {id_data.name}")
    kp = fc.keyframe_points
    while len(kp) > 1:
        kp.remove(kp[-1])
    kp.add(n - 1)
    co = np.empty(2 * n, dtype=np.float32)
    co[0::2] = frames
    co[1::2] = values
    kp.foreach_set("co", co)
    for p in kp:
        p.interpolation = interpolation
        p.handle_left_type = handles
        p.handle_right_type = handles
    fc.update()


def keyframe_vector(id_data, data_path: str, frames, values: np.ndarray, **kw) -> None:
    for i in range(values.shape[1]):
        keyframe_series(id_data, data_path, i, frames, values[:, i], **kw)


# =============================================================================
# Node building
# =============================================================================
class Nodes:
    """Terse node-graph builder working for shader and compositor trees."""

    def __init__(self, tree: bpy.types.NodeTree, clear: bool = True) -> None:
        self.tree = tree
        if clear:
            tree.nodes.clear()

    # Socket renames across Blender versions (e.g. texture "Fac" became "Factor" in 5.0).
    ALIASES = {"Fac": "Factor", "Factor": "Fac"}

    @classmethod
    def out(cls, node, name=0):
        if isinstance(name, int):
            return node.outputs[name]
        for candidate in (name, cls.ALIASES.get(name)):
            for s in node.outputs:
                if s.name == candidate and getattr(s, "enabled", True):
                    return s
        raise KeyError(f"{node.bl_idname} has no output {name!r}")

    @staticmethod
    def inp(node, name):
        if isinstance(name, int):
            return node.inputs[name]
        for s in node.inputs:
            if s.name == name and getattr(s, "enabled", True):
                return s
        raise KeyError(f"{node.bl_idname} has no input {name!r}")

    def new(self, bl_idname: str, inputs: Optional[dict] = None, **props):
        node = self.tree.nodes.new(bl_idname)
        for k, v in props.items():
            setattr(node, k, v)
        for k, v in (inputs or {}).items():
            sock = self.inp(node, k)
            if isinstance(v, bpy.types.NodeSocket):
                self.tree.links.new(v, sock)
            else:
                sock.default_value = v
        return node

    def link(self, a, b) -> None:
        self.tree.links.new(a, b)

    def math(self, op: str, a, b=None, clamp: bool = False):
        n = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                self.link(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    def map_range(self, value, f0, f1, t0=0.0, t1=1.0, clamp=True, interp="LINEAR"):
        n = self.new(
            "ShaderNodeMapRange",
            {"Value": value, "From Min": f0, "From Max": f1, "To Min": t0, "To Max": t1},
            clamp=clamp,
            interpolation_type=interp,
        )
        return n.outputs[0]

    def mix_color(self, fac, a, b, blend="MIX"):
        n = self.new("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        for name, v in (("Factor", fac), ("A", a), ("B", b)):
            sock = self.inp(n, name)
            if isinstance(v, bpy.types.NodeSocket):
                self.link(v, sock)
            else:
                sock.default_value = v
        return self.out(n, "Result")

    def mix_float(self, fac, a, b):
        n = self.new("ShaderNodeMix", data_type="FLOAT")
        for name, v in (("Factor", fac), ("A", a), ("B", b)):
            sock = self.inp(n, name)
            if isinstance(v, bpy.types.NodeSocket):
                self.link(v, sock)
            else:
                sock.default_value = v
        return self.out(n, "Result")

    def separate(self, vec):
        n = self.new("ShaderNodeSeparateXYZ", {"Vector": vec})
        return n.outputs[0], n.outputs[1], n.outputs[2]


def new_material(name: str) -> tuple[bpy.types.Material, Nodes, bpy.types.Node]:
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nb = Nodes(mat.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    return mat, nb, out


# =============================================================================
# Materials
# =============================================================================
class Materials:
    def __init__(self, mood: str) -> None:
        self.mood = mood
        self.wet = 0.75 if mood == "storm" else 0.35
        self.aluminium = self._aluminium()
        self.glass = self._glass()
        self.rubber = self._simple("Tire_Rubber", (0.018, 0.018, 0.02), 0.0, 0.62)
        self.chrome = self._simple("Strut_Chrome", (0.8, 0.8, 0.82), 1.0, 0.12)
        self.spring = self._simple("Spring_Steel", (0.55, 0.12, 0.08), 0.6, 0.35)
        self.prop_black = self._simple("Prop_Black", (0.02, 0.02, 0.02), 0.2, 0.4)
        self.prop_tip = self._simple("Prop_Tip_Yellow", (0.9, 0.62, 0.02), 0.0, 0.35)
        self.prop_disc = self._prop_disc()
        self.nav_red = self._emissive("Nav_Red", (1.0, 0.05, 0.03), 40.0)
        self.nav_green = self._emissive("Nav_Green", (0.05, 1.0, 0.25), 40.0)
        self.strobe = self._emissive("Strobe_White", (1.0, 1.0, 1.0), 0.0)
        self.landing_lens = self._emissive("Landing_Light_Lens", (1.0, 0.95, 0.85), 60.0)
        self.asphalt = self._asphalt()
        self.paint = self._runway_paint()
        self.grass = self._grass()
        self.foliage = self._simple("Foliage", (0.03, 0.06, 0.025), 0.0, 0.85)
        self.bark = self._simple("Bark", (0.05, 0.035, 0.025), 0.0, 0.9)
        self.fixture = self._simple("Light_Fixture", (0.03, 0.03, 0.03), 0.3, 0.6)
        self.light_white = self._emissive("RWY_Light_White", (1.0, 0.86, 0.62), 35.0)
        self.light_yellow = self._emissive("RWY_Light_Yellow", (1.0, 0.65, 0.12), 35.0)
        self.light_green = self._emissive("RWY_Light_Green", (0.1, 1.0, 0.35), 45.0)
        self.light_red = self._emissive("RWY_Light_Red", (1.0, 0.04, 0.02), 45.0)
        self.smoke = self._smoke()
        self.clouds = self._clouds()

    # -- generic -------------------------------------------------------------
    def _simple(self, name, color, metallic, roughness):
        mat, nb, out = new_material(name)
        bsdf = nb.new(
            "ShaderNodeBsdfPrincipled",
            {"Base Color": (*color, 1.0), "Metallic": metallic, "Roughness": roughness},
        )
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        return mat

    def _emissive(self, name, color, strength):
        mat, nb, out = new_material(name)
        bsdf = nb.new(
            "ShaderNodeBsdfPrincipled",
            {
                "Base Color": (*color, 1.0),
                "Emission Color": (*color, 1.0),
                "Emission Strength": strength,
                "Roughness": 0.2,
            },
        )
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        return mat

    # -- airframe ------------------------------------------------------------
    def _aluminium(self):
        """Polished aluminium with painted navy/orange cheat-lines and panel seams."""
        mat, nb, out = new_material("Airframe_Polished_Aluminium")
        tc = nb.new("ShaderNodeTexCoord")
        x, y, z = nb.separate(nb.out(tc, "Object"))
        # Cheat-line centre follows the fuselage centreline as it sweeps up to the tail.
        rise = nb.math("MULTIPLY", nb.math("MAXIMUM", nb.math("SUBTRACT", -1.6, x), 0.0), 0.055)
        zc = nb.math("ADD", rise, -0.16)
        dz = nb.math("SUBTRACT", z, zc)
        navy = nb.math("LESS_THAN", nb.math("ABSOLUTE", dz), 0.055)
        orange = nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SUBTRACT", dz, 0.095)), 0.018)
        on_fuselage = nb.math("LESS_THAN", nb.math("ABSOLUTE", y), 0.75)
        navy = nb.math("MULTIPLY", navy, on_fuselage)
        orange = nb.math("MULTIPLY", orange, on_fuselage)
        paint = nb.math("ADD", navy, orange, clamp=True)

        grain = nb.new("ShaderNodeTexNoise", {"Vector": nb.out(tc, "Object"), "Scale": 900.0, "Detail": 2.0})
        base = nb.mix_color(nb.math("MULTIPLY", nb.out(grain, "Fac"), 0.12), (0.83, 0.84, 0.86, 1), (0.6, 0.62, 0.65, 1))
        base = nb.mix_color(navy, base, (0.012, 0.03, 0.11, 1))
        base = nb.mix_color(orange, base, (0.95, 0.32, 0.03, 1))
        metallic = nb.math("SUBTRACT", 1.0, paint)
        rough = nb.mix_float(paint, nb.map_range(nb.out(grain, "Fac"), 0.3, 0.7, 0.12, 0.22), 0.28)

        bricks = nb.new(
            "ShaderNodeTexBrick",
            {"Vector": nb.out(tc, "Object"), "Scale": 1.2, "Mortar Size": 0.004, "Mortar Smooth": 0.3,
             "Brick Width": 0.9, "Row Height": 0.45, "Color1": (1, 1, 1, 1), "Color2": (1, 1, 1, 1),
             "Mortar": (0, 0, 0, 1)},
        )
        bump = nb.new("ShaderNodeBump", {"Strength": 0.25, "Distance": 0.002, "Height": nb.out(bricks, "Fac")})
        bsdf = nb.new(
            "ShaderNodeBsdfPrincipled",
            {
                "Base Color": base,
                "Metallic": metallic,
                "Roughness": rough,
                "Anisotropic": 0.35,
                "Coat Weight": nb.math("MULTIPLY", paint, 0.9),
                "Coat Roughness": 0.04,
                "Normal": bump.outputs[0],
            },
        )
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        return mat

    def _glass(self):
        mat, nb, out = new_material("Canopy_Glass")
        bsdf = nb.new(
            "ShaderNodeBsdfPrincipled",
            {"Base Color": (0.01, 0.015, 0.02, 1), "Metallic": 0.0, "Roughness": 0.03, "Coat Weight": 1.0,
             "Coat Roughness": 0.0, "Specular IOR Level": 0.8},
        )
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        return mat

    def _prop_disc(self):
        """Faint radial disc that reads as a spinning propeller arc under motion blur."""
        mat, nb, out = new_material("Prop_Disc")
        tc = nb.new("ShaderNodeTexCoord")
        _, y, z = nb.separate(nb.out(tc, "Object"))
        r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", y, y), nb.math("MULTIPLY", z, z)))
        alpha = nb.math("MULTIPLY", nb.map_range(r, 0.2, 0.95, 0.05, 0.12), nb.map_range(r, 0.9, 0.95, 1.0, 0.0))
        bsdf = nb.new("ShaderNodeBsdfPrincipled", {"Base Color": (0.05, 0.05, 0.05, 1), "Roughness": 0.4, "Alpha": alpha})
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        self._dithered(mat)
        return mat

    @staticmethod
    def _dithered(mat):
        if hasattr(mat, "surface_render_method"):
            set_first(mat, "surface_render_method", "DITHERED", "BLENDED")
        elif hasattr(mat, "blend_method"):
            set_first(mat, "blend_method", "HASHED", "BLEND")

    # -- runway ----------------------------------------------------------------
    def _asphalt(self):
        """Weathered, partly wet asphalt: aggregate, puddles, rubber deposits, lighter shoulders."""
        mat, nb, out = new_material("Runway_Asphalt")
        tc = nb.new("ShaderNodeTexCoord")
        obj = nb.out(tc, "Object")
        x, y, _ = nb.separate(obj)

        aggregate = nb.new("ShaderNodeTexVoronoi", {"Vector": obj, "Scale": 55.0, "Randomness": 1.0})
        stain = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 0.12, "Detail": 10.0, "Roughness": 0.62})
        fine = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 4.0, "Detail": 12.0})
        tone = nb.math("ADD", nb.math("MULTIPLY", nb.out(stain, "Fac"), 0.7), nb.math("MULTIPLY", nb.out(aggregate, "Distance"), 0.3))
        base = nb.mix_color(nb.map_range(tone, 0.25, 0.75), (0.028, 0.028, 0.03, 1), (0.085, 0.082, 0.08, 1))

        # Rubber deposits along the main-wheel tracks in the touchdown zone.
        track = nb.math("ABSOLUTE", nb.math("SUBTRACT", nb.math("ABSOLUTE", x), 1.4))
        track = nb.math("EXPONENT", nb.math("MULTIPLY", nb.math("MULTIPLY", track, track), -0.6))
        tdz = nb.math("MULTIPLY", nb.map_range(y, 90.0, 180.0), nb.map_range(y, 650.0, 900.0, 1.0, 0.0))
        rubber = nb.math("MULTIPLY", nb.math("MULTIPLY", track, tdz), nb.map_range(nb.out(fine, "Fac"), 0.3, 0.7, 0.55, 1.0))
        base = nb.mix_color(rubber, base, (0.008, 0.008, 0.009, 1))

        # Shoulders (outside the runway width) in lighter, older concrete.
        shoulder = nb.math("GREATER_THAN", nb.math("ABSOLUTE", x), 15.0)
        base = nb.mix_color(shoulder, base, (0.11, 0.105, 0.1, 1))

        # Standing water: large-scale noise thresholded into puddles.
        puddle_noise = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 0.045, "Detail": 6.0, "Roughness": 0.55})
        # Puddles collect in the slightly rutted wheel tracks; elsewhere the surface is only damp.
        lo = 0.70 - 0.17 * self.wet
        wet = nb.map_range(nb.math("ADD", nb.out(puddle_noise, "Fac"), nb.math("MULTIPLY", track, 0.12)), lo, lo + 0.035)
        damp = nb.map_range(nb.out(puddle_noise, "Fac"), lo - 0.25, lo)
        base = nb.mix_color(nb.math("MULTIPLY", wet, 0.6), base, (0.01, 0.01, 0.012, 1))
        rough = nb.mix_float(wet, nb.map_range(nb.out(fine, "Fac"), 0.2, 0.8, 0.55, 0.85), 0.07)
        rough = nb.math("SUBTRACT", rough, nb.math("MULTIPLY", damp, 0.18 * self.wet))

        bump_h = nb.math("ADD", nb.math("MULTIPLY", nb.out(aggregate, "Distance"), 1.0), nb.math("MULTIPLY", nb.out(fine, "Fac"), 0.4))
        bump = nb.new("ShaderNodeBump", {"Strength": nb.map_range(wet, 0.0, 1.0, 0.35, 0.02), "Distance": 0.004, "Height": bump_h})
        bsdf = nb.new("ShaderNodeBsdfPrincipled", {"Base Color": base, "Roughness": rough, "Normal": bump.outputs[0]})
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        return mat

    def _runway_paint(self):
        """Retro-reflective paint, eroded by traffic and darkened by tire rubber."""
        mat, nb, out = new_material("Runway_Paint_Weathered")
        tc = nb.new("ShaderNodeTexCoord")
        obj = nb.out(tc, "Object")
        x, y, _ = nb.separate(obj)
        wear = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 2.5, "Detail": 12.0, "Roughness": 0.7})
        chips = nb.new("ShaderNodeTexVoronoi", {"Vector": obj, "Scale": 18.0})
        erosion = nb.math("ADD", nb.out(wear, "Fac"), nb.math("MULTIPLY", nb.out(chips, "Distance"), 0.35))
        tdz = nb.math("MULTIPLY", nb.map_range(y, 90.0, 180.0), nb.map_range(y, 650.0, 900.0, 1.0, 0.0))
        centre = nb.map_range(nb.math("ABSOLUTE", x), 0.6, 3.0, 1.0, 0.0)
        erosion = nb.math("ADD", erosion, nb.math("MULTIPLY", nb.math("MULTIPLY", tdz, centre), 0.25))
        alpha = nb.map_range(erosion, 0.78, 0.7)
        dirt = nb.map_range(erosion, 0.4, 0.75)
        base = nb.mix_color(dirt, (0.82, 0.8, 0.74, 1), (0.32, 0.31, 0.29, 1))
        bsdf = nb.new(
            "ShaderNodeBsdfPrincipled",
            {"Base Color": base, "Roughness": nb.map_range(erosion, 0.3, 0.8, 0.35, 0.6), "Alpha": alpha,
             "Specular IOR Level": 0.6},
        )
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        self._dithered(mat)
        return mat

    def _grass(self):
        mat, nb, out = new_material("Terrain_Grass")
        tc = nb.new("ShaderNodeTexCoord")
        obj = nb.out(tc, "Object")
        large = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 0.004, "Detail": 8.0, "Roughness": 0.6})
        mid = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 0.08, "Detail": 6.0})
        small = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 6.0, "Detail": 4.0})
        dry, lush = ((0.17, 0.13, 0.05, 1), (0.04, 0.075, 0.02, 1)) if self.mood == "golden" else ((0.07, 0.07, 0.04, 1), (0.025, 0.045, 0.02, 1))
        base = nb.mix_color(nb.map_range(nb.out(large, "Fac"), 0.35, 0.65), lush, dry)
        base = nb.mix_color(nb.map_range(nb.out(mid, "Fac"), 0.4, 0.6, 0.0, 0.35), base, (0.09, 0.08, 0.04, 1))
        bump = nb.new("ShaderNodeBump", {"Strength": 0.4, "Distance": 0.02, "Height": nb.out(small, "Fac")})
        bsdf = nb.new("ShaderNodeBsdfPrincipled", {"Base Color": base, "Roughness": 0.92, "Normal": bump.outputs[0]})
        nb.link(bsdf.outputs[0], out.inputs["Surface"])
        return mat

    # -- volumes ---------------------------------------------------------------
    def _smoke(self):
        """Tire smoke: per-object density from Object Info alpha (animated), spherical falloff."""
        mat, nb, out = new_material("Tire_Smoke")
        tc = nb.new("ShaderNodeTexCoord")
        obj = nb.out(tc, "Object")
        info = nb.new("ShaderNodeObjectInfo")
        radius = nb.new("ShaderNodeVectorMath", {0: obj}, operation="LENGTH")
        falloff = nb.map_range(nb.out(radius, "Value"), 1.0, 0.25, interp="SMOOTHSTEP")
        billow = nb.new("ShaderNodeTexNoise", {"Vector": obj, "Scale": 2.2, "Detail": 6.0, "Roughness": 0.65})
        shape = nb.math("MULTIPLY", falloff, nb.map_range(nb.out(billow, "Fac"), 0.35, 0.75))
        density = nb.math("MULTIPLY", nb.math("MULTIPLY", shape, nb.out(info, "Alpha")), 14.0)
        vol = nb.new("ShaderNodeVolumePrincipled", {"Color": (0.78, 0.78, 0.8, 1), "Density": density, "Anisotropy": 0.35})
        nb.link(vol.outputs[0], out.inputs["Volume"])
        return mat

    def _clouds(self):
        mat, nb, out = new_material("Cloud_Deck")
        tc = nb.new("ShaderNodeTexCoord")
        obj = nb.out(tc, "Object")
        _, _, z = nb.separate(obj)
        mapping = nb.new("ShaderNodeMapping", {"Vector": obj, "Scale": (1.0, 1.0, 6.0)})
        n1 = nb.new("ShaderNodeTexNoise", {"Vector": mapping.outputs[0], "Scale": 4.0, "Detail": 8.0, "Roughness": 0.6, "Distortion": 0.5})
        coverage = (0.55, 0.72) if self.mood == "golden" else (0.4, 0.56)
        profile = nb.math("MULTIPLY", nb.map_range(z, -1.0, -0.4), nb.map_range(z, 0.2, 1.0, 1.0, 0.0))
        dens = nb.math("MULTIPLY", nb.map_range(nb.out(n1, "Fac"), *coverage), profile)
        peak = 0.012 if self.mood == "golden" else 0.02
        color = (1.0, 0.96, 0.92, 1) if self.mood == "golden" else (0.55, 0.58, 0.62, 1)
        vol = nb.new("ShaderNodeVolumePrincipled", {"Color": color, "Density": nb.math("MULTIPLY", dens, peak), "Anisotropy": 0.55})
        nb.link(vol.outputs[0], out.inputs["Volume"])
        return mat


# =============================================================================
# Telemetry
# =============================================================================
def _array(values) -> np.ndarray:
    """JSON rows (``null`` for non-finite values) -> float array with nulls as 0."""
    rows = [[np.nan if v is None else v for v in row] if isinstance(row, (list, tuple)) else (np.nan if row is None else row)
            for row in values]
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0)


class Telemetry:
    """Columnar view of a telemetry document with interpolation helpers."""

    def __init__(self, doc: dict) -> None:
        if doc.get("schema") != "crosswind-landing-telemetry":
            raise ValueError("not a crosswind-landing-telemetry document")
        fr = doc["frames"]
        if len(fr) < 10:
            raise ValueError("telemetry episode too short to animate")
        self.doc = doc
        self.t = _array([f["t"] for f in fr])
        self.loc = _array([f["blender"]["location"] for f in fr])
        self.quat = _array([f["blender"]["rotation_quaternion"] for f in fr])
        self.euler_deg = _array([f["ned"]["euler_deg"] for f in fr])
        self.deflection_deg = _array(
            [[f["controls"]["elevator_deg"], f["controls"]["aileron_deg"], f["controls"]["rudder_deg"]] for f in fr]
        )
        self.action = _array([f["controls"]["action"] for f in fr])
        self.throttle = _array([f["controls"]["throttle"] for f in fr])
        self.wind = _array([f["wind_blender"] for f in fr])
        turb = _array([f["wind_ned"]["turbulence"] for f in fr])
        gust = _array([f["wind_ned"]["gust"] for f in fr])
        self.gust_intensity = np.linalg.norm(turb + gust, axis=1)
        self.airspeed = _array([f["airdata"]["airspeed_mps"] for f in fr])
        self.touched = np.array([bool(f["touched_down"]) for f in fr])

        ac = doc["aircraft"]
        # FRD -> Blender model axes (FLU).
        self.gear_flu = {g["name"]: Vector((g["position_frd_m"][0], -g["position_frd_m"][1], -g["position_frd_m"][2])) for g in ac["gear"]}
        self.limits_deg = ac["control_limits_deg"]
        self.span = ac["span_m"]
        self.runway = doc["runway"]
        td = doc["episode"].get("touchdown")
        self.touchdown_time = float(td["time_s"]) if td else None
        self.metrics = doc["episode"].get("metrics", {})

        # Quaternion hemisphere continuity (exporter guarantees it; enforce anyway).
        for i in range(1, len(self.quat)):
            if float(np.dot(self.quat[i], self.quat[i - 1])) < 0.0:
                self.quat[i] = -self.quat[i]

    @classmethod
    def load(cls, path: str | Path) -> "Telemetry":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def interp(self, arr: np.ndarray, t: float) -> np.ndarray:
        if arr.ndim == 1:
            return np.interp(t, self.t, arr)
        return np.array([np.interp(t, self.t, arr[:, k]) for k in range(arr.shape[1])])

    def quat_at(self, t: float) -> Quaternion:
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a, b = Quaternion(self.quat[i]), Quaternion(self.quat[i + 1])
        u = (t - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-9)
        return a.slerp(b, float(np.clip(u, 0.0, 1.0)))

    def gear_world(self, name: str, i: int) -> Vector:
        return Vector(self.loc[i]) + Quaternion(self.quat[i]).to_matrix() @ self.gear_flu[name]

    def gear_compression(self, name: str) -> np.ndarray:
        """Geometric penetration of a gear contact point below the runway (z = 0)."""
        return np.array([max(0.0, -self.gear_world(name, i).z) for i in range(len(self.t))])

    def smoothed(self, arr: np.ndarray, window_s: float) -> np.ndarray:
        dt = float(np.median(np.diff(self.t)))
        k = max(1, int(round(window_s / dt)))
        kernel = np.hanning(2 * k + 1)
        kernel /= kernel.sum()
        pad = np.pad(arr, ((k, k), (0, 0)) if arr.ndim == 2 else (k, k), mode="edge")
        if arr.ndim == 1:
            return np.convolve(pad, kernel, mode="valid")
        return np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(arr.shape[1])], axis=1)


@dataclass
class Timeline:
    """Maps simulation time to (fractional) video frames."""

    t0: float
    t1: float
    fps: int
    frame_start: int = 1

    def frame(self, t):
        return self.frame_start + (np.asarray(t) - self.t0) * self.fps

    @property
    def frame_end(self) -> int:
        return int(math.floor(self.frame(self.t1)))


# =============================================================================
# Procedural geometry
# =============================================================================
def naca4(code: str, n: int = 28, f0: float = 0.0, f1: float = 1.0) -> list[tuple[float, float]]:
    """Closed NACA 4-digit loop (x/c, z/c), optionally restricted to the chord band [f0, f1].

    Ordered: upper surface f1 -> f0, then lower surface f0 -> f1.
    """
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    beta = np.linspace(0.0, math.pi, n)
    xs = f0 + (f1 - f0) * 0.5 * (1.0 - np.cos(beta))

    def surf(x):
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
        if m > 0 and x < p:
            yc, dyc = m / p**2 * (2 * p * x - x**2), 2 * m / p**2 * (p - x)
        elif m > 0:
            yc, dyc = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2), 2 * m / (1 - p) ** 2 * (p - x)
        else:
            yc, dyc = 0.0, 0.0
        th = math.atan(dyc)
        return (x - yt * math.sin(th), yc + yt * math.cos(th)), (x + yt * math.sin(th), yc - yt * math.cos(th))

    upper = [surf(x)[0] for x in xs[::-1]]
    # A full section shares its leading/trailing-edge points; a chord band is closed by
    # vertical edges at f0 and f1, so its lower surface keeps both end points.
    lower = [surf(x)[1] for x in xs[1:-1]] if (f0 == 0.0 and f1 == 1.0) else [surf(x)[1] for x in (xs[1:] if f0 == 0.0 else xs)]
    return upper + lower


def camber_z(code: str, x: float) -> float:
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    if m == 0:
        return 0.0
    return m / p**2 * (2 * p * x - x**2) if x < p else m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)


def loft(bm: bmesh.types.BMesh, rings: list[list[Vector]], cap_start: bool = True, cap_end: bool = True, mat_fn=None):
    """Connect equal-length closed rings with quads; optional n-gon caps."""
    verts = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for a, b in zip(verts[:-1], verts[1:]):
        for i in range(n):
            j = (i + 1) % n
            f = bm.faces.new((a[i], a[j], b[j], b[i]))
            if mat_fn is not None:
                f.material_index = mat_fn(f)
    if cap_start:
        bm.faces.new(list(reversed(verts[0])))
    if cap_end:
        bm.faces.new(verts[-1])
    return verts


def superellipse_ring(x: float, half_w: float, z_top: float, z_bot: float, n: int = 32, exponent: float = 2.6) -> list[Vector]:
    zc, b = 0.5 * (z_top + z_bot), 0.5 * (z_top - z_bot)
    pts = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        y = half_w * math.copysign(abs(c) ** (2.0 / exponent), c)
        z = zc + b * math.copysign(abs(s) ** (2.0 / exponent), s)
        pts.append(Vector((x, y, z)))
    return pts


def airfoil_ring(code: str, chord: float, x_le: float, y: float, z: float, f0=0.0, f1=1.0, n=26, plane: str = "XZ") -> list[Vector]:
    """Airfoil section placed in the model frame. plane XZ: span along Y; plane XY: span along Z (fins)."""
    pts = []
    for xc, zc in naca4(code, n, f0, f1):
        px = x_le - xc * chord
        if plane == "XZ":
            pts.append(Vector((px, y, z + zc * chord)))
        else:  # fin: thickness along Y, span along Z (passed as ``z``)
            pts.append(Vector((px, zc * chord, y)))
    return pts


def add_cylinder(bm, p0: Vector, p1: Vector, r0: float, r1: Optional[float] = None, segments: int = 16):
    r1 = r0 if r1 is None else r1
    d = p1 - p0
    length = d.length
    rot = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    mat = Matrix.Translation((p0 + p1) * 0.5) @ rot
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=r0, radius2=r1, depth=length, matrix=mat)


def add_box(bm, x0, x1, y0, y1, z0, z1):
    res = bmesh.ops.create_cube(bm, size=1.0)
    for v in res["verts"]:
        v.co = Vector(((x0 + x1) / 2 + v.co.x * (x1 - x0), (y0 + y1) / 2 + v.co.y * (y1 - y0), (z0 + z1) / 2 + v.co.z * (z1 - z0)))


# =============================================================================
# Aircraft
# =============================================================================
@dataclass
class AircraftRig:
    root: bpy.types.Object
    surfaces: dict = field(default_factory=dict)  # name -> (object, axis Vector, sign)
    gear_hubs: dict = field(default_factory=dict)  # gear name -> (hub empty, rest location)
    wheels: dict = field(default_factory=dict)  # gear name -> (wheel object, radius)
    nose_steer: Optional[bpy.types.Object] = None
    prop: Optional[bpy.types.Object] = None
    strobes: list = field(default_factory=list)
    focus_tail: Optional[bpy.types.Object] = None


FUSELAGE_STATIONS = (
    # x [m],  half width, z top, z bottom   (model axes: +X nose, +Y left, +Z up; origin = CG)
    (1.95, 0.30, 0.22, -0.30),
    (1.72, 0.44, 0.33, -0.42),
    (1.30, 0.52, 0.40, -0.50),
    (0.95, 0.55, 0.46, -0.52),
    (0.55, 0.56, 0.86, -0.52),
    (0.00, 0.56, 0.92, -0.50),
    (-0.90, 0.55, 0.92, -0.46),
    (-1.60, 0.50, 0.82, -0.40),
    (-2.60, 0.38, 0.56, -0.28),
    (-3.80, 0.25, 0.42, -0.12),
    (-5.00, 0.15, 0.33, 0.02),
    (-6.05, 0.06, 0.29, 0.12),
    (-6.30, 0.02, 0.25, 0.17),
)


class AircraftBuilder:
    """Procedural high-wing light aircraft (Cessna-172 proportions) with articulated parts."""

    WING_CODE = "2412"
    TAIL_CODE = "0012"
    WING_Z = 0.94
    DIHEDRAL = math.radians(1.73)

    def __init__(self, coll, mats: Materials, tel: Telemetry) -> None:
        self.coll = coll
        self.m = mats
        self.tel = tel
        self.half_span = 0.5 * tel.span

    # ---------------------------------------------------------------- helpers
    def wing_section(self, y: float) -> tuple[float, float, float]:
        """(chord, x_le, z) of the wing at span station |y|."""
        ay = abs(y)
        chord = 1.63 if ay <= 2.55 else 1.63 + (1.12 - 1.63) * (ay - 2.55) / (self.half_span - 2.55)
        x_le = 0.45 if ay <= 2.55 else 0.45 - 0.08 * (ay - 2.55) / (self.half_span - 2.55)
        return chord, x_le, self.WING_Z + ay * math.tan(self.DIHEDRAL)

    def _wing_ring(self, y, f0=0.0, f1=1.0):
        c, x_le, z = self.wing_section(y)
        return airfoil_ring(self.WING_CODE, c, x_le, y, z, f0, f1)

    def _hinge_point(self, y, frac):
        c, x_le, z = self.wing_section(y)
        return Vector((x_le - frac * c, y, z + camber_z(self.WING_CODE, frac) * c))

    def _surface_object(self, name, rings, hinge_a: Vector, hinge_b: Vector, parent, sign: float, mat):
        """Mesh whose origin lies on the hinge line; articulated with an axis-angle rotation."""
        bm = bmesh.new()
        origin = (hinge_a + hinge_b) * 0.5
        loft(bm, [[p - origin for p in ring] for ring in rings])
        obj = mesh_object(name, bm, self.coll, parent, [mat], smooth=True)
        obj.location = origin
        obj.rotation_mode = "AXIS_ANGLE"
        axis = (hinge_b - hinge_a).normalized()
        obj.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
        self._bevel(obj, 0.004)
        return obj, axis, sign

    @staticmethod
    def _subsurf(obj, levels=1, render_levels=2):
        mod = obj.modifiers.new("Subdivision", "SUBSURF")
        mod.levels = levels
        mod.render_levels = render_levels

    @staticmethod
    def _bevel(obj, width):
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = width
        mod.segments = 2
        mod.limit_method = "ANGLE"

    # ------------------------------------------------------------------ build
    def build(self) -> AircraftRig:
        root = new_empty("Aircraft", self.coll, size=2.0, kind="ARROWS")
        root.rotation_mode = "QUATERNION"
        rig = AircraftRig(root=root)
        self._fuselage(root)
        self._wings(root, rig)
        self._tail(root, rig)
        self._struts(root)
        self._gear(root, rig)
        self._propeller(root, rig)
        self._lights(root, rig)
        self._registration(root)
        return rig

    def _fuselage(self, root):
        bm = bmesh.new()
        rings = [superellipse_ring(*st) for st in FUSELAGE_STATIONS]

        def window_index(face):
            c = face.calc_center_median()
            # Vertical position of the face relative to its local cross-section.
            xs = [s[0] for s in FUSELAGE_STATIONS][::-1]
            zt = float(np.interp(c.x, xs, [s[2] for s in FUSELAGE_STATIONS][::-1]))
            zb = float(np.interp(c.x, xs, [s[3] for s in FUSELAGE_STATIONS][::-1]))
            rel = (c.z - 0.5 * (zt + zb)) / max(0.5 * (zt - zb), 1e-6)
            if 0.5 < c.x < 0.95 and rel > 0.05:
                return 1  # windshield
            if -1.75 < c.x < 0.5 and 0.18 < rel < 0.78 and abs(c.y) > 0.3:
                return 1  # side windows
            return 0

        loft(bm, rings, mat_fn=window_index)
        obj = mesh_object("Fuselage", bm, self.coll, root, [self.m.aluminium, self.m.glass], smooth=True)
        self._subsurf(obj)

    def _wings(self, root, rig):
        span = self.half_span
        ail0, ail1 = 2.75, span - 0.45
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            bm = bmesh.new()
            ys = lambda vals: [sgn * v for v in vals]  # noqa: E731
            # Inboard full-chord panel (flap region is part of the fixed wing).
            loft(bm, [self._wing_ring(y) for y in ys([0.0, 1.2, 2.55, ail0])])
            # Panel ahead of the aileron.
            loft(bm, [self._wing_ring(y, 0.0, 0.775) for y in ys([ail0, 4.0, ail1])])
            # Wing tip with a rounded, slightly drooped cap.
            tip_rings = [self._wing_ring(y) for y in ys([ail1, span - 0.08])]
            cap = [p.copy() for p in tip_rings[-1]]
            c, x_le, z = self.wing_section(span)
            centre = Vector((x_le - 0.45 * c, sgn * span, z))
            cap = [centre + (p - centre) * 0.55 + Vector((0, 0, -0.02)) for p in cap]
            for p in cap:
                p.y = sgn * span
            loft(bm, tip_rings + [cap])
            wing = mesh_object(f"Wing_{side}", bm, self.coll, root, [self.m.aluminium], smooth=True)
            self._bevel(wing, 0.004)

            rings = [self._wing_ring(y, 0.8, 1.0) for y in ys([ail0 + 0.02, ail1 - 0.02])]
            a, b = self._hinge_point(sgn * (ail0 + 0.02), 0.8), self._hinge_point(sgn * (ail1 - 0.02), 0.8)
            # Positive aileron rolls right-wing-down: left aileron trailing edge down.
            name = "Aileron_L" if sgn > 0 else "Aileron_R"
            obj, axis, _ = self._surface_object(name, rings, a, b, root, 1.0, self.m.aluminium)
            # Rotation by +angle about +Y raises a trailing edge (which lies aft, at -X).
            # Left: TE down -> angle = -delta_a about +Y.  Right: TE up -> +delta_a about +Y.
            if axis.y < 0:
                axis = -axis
                obj.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
            rig.surfaces[name] = (obj, axis, -1.0 if sgn > 0 else 1.0)

    def _tail(self, root, rig):
        # Horizontal stabiliser with a straight (spanwise) elevator hinge at 62 % chord.
        root_c, tip_c, frac = 1.35, 0.90, 0.62
        x_root = -4.95
        x_tip = x_root - frac * (root_c - tip_c)
        z = 0.18
        half = 1.72

        def ring(y, f0=0.0, f1=1.0):
            t = abs(y) / half
            c = root_c + (tip_c - root_c) * t
            xl = x_root + (x_tip - x_root) * t
            return airfoil_ring(self.TAIL_CODE, c, xl, y, z, f0, f1)

        bm = bmesh.new()
        loft(bm, [ring(y, 0.0, frac - 0.02) for y in (-half, -0.6, 0.0, 0.6, half)])
        stab = mesh_object("Stabilizer", bm, self.coll, root, [self.m.aluminium], smooth=True)
        self._bevel(stab, 0.003)
        hinge_x = x_root - frac * root_c
        rings = [ring(y, frac, 1.0) for y in (-half + 0.02, -0.6, 0.0, 0.6, half - 0.02)]
        obj, axis, _ = self._surface_object(
            "Elevator", rings, Vector((hinge_x, -half, z)), Vector((hinge_x, half, z)), root, 1.0, self.m.aluminium
        )
        rig.surfaces["Elevator"] = (obj, axis, -1.0)  # positive delta_e: trailing edge down

        # Vertical fin + rudder (swept hinge line).
        fin_root_c, fin_tip_c, z0, z1 = 1.55, 0.85, 0.28, 1.95
        xr, xt = -4.75, -5.70

        def fin_ring(zz, f0=0.0, f1=1.0):
            t = (zz - z0) / (z1 - z0)
            c = fin_root_c + (fin_tip_c - fin_root_c) * t
            xl = xr + (xt - xr) * t
            return airfoil_ring("0010", c, xl, zz, 0.0, f0, f1, plane="XY")

        bm = bmesh.new()
        loft(bm, [fin_ring(zz, 0.0, frac - 0.02) for zz in (z0 - 0.05, 1.0, z1)])
        fin = mesh_object("Fin", bm, self.coll, root, [self.m.aluminium], smooth=True)
        self._bevel(fin, 0.003)
        # Dorsal fillet blending the fin into the fuselage.
        bm = bmesh.new()
        loft(bm, [airfoil_ring("0010", c, xl, zz, 0.0, plane="XY") for c, xl, zz in ((2.4, -3.0, 0.38), (1.5, -4.6, 0.62))])
        mesh_object("Dorsal_Fin", bm, self.coll, root, [self.m.aluminium], smooth=True)

        def hinge(zz):
            t = (zz - z0) / (z1 - z0)
            c = fin_root_c + (fin_tip_c - fin_root_c) * t
            return Vector((xr + (xt - xr) * t - frac * c, 0.0, zz))

        rings = [fin_ring(zz, frac, 1.0) for zz in (z0 + 0.02, 1.0, z1 - 0.02)]
        obj, axis, _ = self._surface_object("Rudder", rings, hinge(z0 + 0.02), hinge(z1 - 0.02), root, 1.0, self.m.aluminium)
        if axis.z < 0:
            axis = -axis
            obj.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
        # Positive rudder: trailing edge left (+Y); +angle about +Z moves an aft TE to -Y.
        rig.surfaces["Rudder"] = (obj, axis, -1.0)
        rig.focus_tail = new_empty("Focus_Tail", self.coll, root, location=(-5.4, 0.0, 0.6), size=0.2)

    def _struts(self, root):
        bm = bmesh.new()
        for sgn in (1.0, -1.0):
            c, x_le, z = self.wing_section(2.5)
            add_cylinder(bm, Vector((-0.05, sgn * 0.52, -0.32)), Vector((x_le - 0.35 * c, sgn * 2.5, z - 0.05)), 0.045, 0.035, 12)
        mesh_object("Wing_Struts", bm, self.coll, root, [self.m.aluminium], smooth=True)

    def _gear(self, root, rig):
        """Gear legs and coil springs stretch between a fixed attach point and a hub empty
        that the animation translates by the compression of each leg."""
        specs = {
            "nose": (Vector((1.30, 0.0, -0.40)), 0.20, True),
            "main_left": (Vector((-0.30, 0.48, -0.44)), 0.24, False),
            "main_right": (Vector((-0.30, -0.48, -0.44)), 0.24, False),
        }
        for name, contact in self.tel.gear_flu.items():
            attach_p, radius, steerable = specs.get(name, (contact + Vector((0, 0, 0.6)), 0.22, False))
            hub_rest = contact + Vector((0.0, 0.0, radius))
            parent = root
            if steerable:
                steer = new_empty("Nose_Steer", self.coll, root, location=(contact.x, 0.0, 0.0), size=0.2)
                rig.nose_steer = steer
                parent = steer
                hub_rest = hub_rest - Vector((contact.x, 0.0, 0.0))
                attach_p = attach_p - Vector((contact.x, 0.0, 0.0))
            hub = new_empty(f"Hub_{name}", self.coll, parent, location=hub_rest, size=0.15)
            rig.gear_hubs[name] = (hub, hub_rest.copy())

            # Fixed attach point (looked up by name by the glTF exporter's rig manifest).
            new_empty(f"Attach_{name}", self.coll, parent, location=attach_p, size=0.1)
            rest_len = (hub_rest - attach_p).length
            # Strut (built along +Y so Stretch-To can drive it).
            bm = bmesh.new()
            add_cylinder(bm, Vector((0, 0, 0)), Vector((0, rest_len, 0)), 0.045 if steerable else 0.035, 0.03, 12)
            strut = mesh_object(f"Strut_{name}", bm, self.coll, parent, [self.m.chrome], smooth=True)
            strut.location = attach_p
            self._stretch_to(strut, hub, rest_len)
            # Coil spring as a beveled helix curve.
            spring = self._coil(f"Spring_{name}", rest_len * 0.62, 0.075 if steerable else 0.06, 7, parent)
            spring.location = attach_p
            self._stretch_to(spring, hub, rest_len * 0.62)

            # Wheel (tire + hub cap) and fairing ("speed pant").
            bm = bmesh.new()
            bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=radius, radius2=radius, depth=0.15,
                                        matrix=Matrix.Rotation(math.pi / 2, 4, "X"))
            bmesh.ops.bevel(bm, geom=[e for e in bm.edges if len(e.link_faces) == 2 and abs(e.calc_face_angle(0.0)) > 0.8],
                            offset=0.05, segments=4, affect="EDGES")
            wheel = mesh_object(f"Wheel_{name}", bm, self.coll, hub, [self.m.rubber], smooth=True)
            wheel.rotation_mode = "XYZ"
            rig.wheels[name] = (wheel, radius)
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0,
                                      matrix=Matrix.Diagonal((radius * 2.3, 0.13, radius * 1.05, 1.0)) @ Matrix.Translation((0.0, 0.0, 0.18)))
            # Remove the lower part so the tire shows below the fairing.
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -0.02], context="VERTS")
            fairing = mesh_object(f"Fairing_{name}", bm, self.coll, hub, [self.m.aluminium], smooth=True)
            fairing.location = (0.03, 0.0, 0.0)
            solid = fairing.modifiers.new("Solidify", "SOLIDIFY")
            solid.thickness = 0.01

    @staticmethod
    def _stretch_to(obj, target, rest_len):
        c = obj.constraints.new("STRETCH_TO")
        c.target = target
        c.rest_length = rest_len
        c.volume = "NO_VOLUME"
        c.keep_axis = "PLANE_X"

    def _coil(self, name, length, radius, turns, parent):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.011
        curve.bevel_resolution = 3
        spline = curve.splines.new("POLY")
        n = turns * 24
        spline.points.add(n)
        for k in range(n + 1):
            a = 2.0 * math.pi * turns * k / n
            spline.points[k].co = (radius * math.cos(a), length * k / n, radius * math.sin(a), 1.0)
        obj = bpy.data.objects.new(name, curve)
        obj.data.materials.append(self.m.spring)
        return link_object(obj, self.coll, parent)

    def _propeller(self, root, rig):
        prop = new_empty("Propeller", self.coll, root, location=(2.05, 0.0, -0.04), size=0.4)
        prop.rotation_mode = "XYZ"
        rig.prop = prop
        bm = bmesh.new()
        profile = [(0.0, 0.0), (0.06, 0.12), (0.14, 0.2), (0.24, 0.255), (0.32, 0.27)]
        rings = []
        for dx, r in profile:
            rings.append([Vector((0.32 - dx, r * math.cos(2 * math.pi * k / 24), r * math.sin(2 * math.pi * k / 24))) for k in range(24)])
        rings[0] = [Vector((0.32, 0.0, 0.0)) + (p - Vector((0.32, 0, 0))) * 0.02 for p in rings[1]]
        loft(bm, rings)
        mesh_object("Spinner", bm, self.coll, prop, [self.m.chrome], smooth=True)
        for blade_sign in (1.0, -1.0):
            bm = bmesh.new()
            rings = []
            for r in np.linspace(0.16, 0.95, 9):
                chord = 0.17 - 0.09 * (r - 0.16) / 0.79
                twist = math.radians(38.0 - 26.0 * (r - 0.16) / 0.79)
                ring = []
                for xc, zc in naca4("4412", 12):
                    u = (0.35 - xc) * chord  # along blade chord
                    v = zc * chord
                    # chord lies in the Y-X plane rotated by the twist; blade span along Z
                    ring.append(Vector((v * math.cos(twist) - u * math.sin(twist),
                                        (u * math.cos(twist) + v * math.sin(twist)) * blade_sign,
                                        blade_sign * r)))
                rings.append(ring)
            # material: tips yellow
            def tip_index(face):
                return 1 if abs(face.calc_center_median().z) > 0.86 else 0
            loft(bm, rings, mat_fn=tip_index)
            mesh_object(f"Blade_{'A' if blade_sign > 0 else 'B'}", bm, self.coll, prop,
                        [self.m.prop_black, self.m.prop_tip], smooth=True)
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, segments=64, radius=0.96, matrix=Matrix.Rotation(math.pi / 2, 4, "Y"))
        disc = mesh_object("Prop_Disc", bm, self.coll, root, [self.m.prop_disc])
        disc.location = (2.03, 0.0, -0.04)

    def _lights(self, root, rig):
        span = self.half_span
        for name, y, mat, color in (("Nav_Left", span, self.m.nav_red, (1.0, 0.05, 0.03)),
                                    ("Nav_Right", -span, self.m.nav_green, (0.05, 1.0, 0.25))):
            c, x_le, z = self.wing_section(abs(y))
            bm = bmesh.new()
            bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.045)
            bulb = mesh_object(name, bm, self.coll, root, [mat], smooth=True)
            bulb.location = (x_le - 0.2 * c, y + math.copysign(0.03, y), z)
            lamp = bpy.data.lights.new(name + "_Lamp", "POINT")
            lamp.color = color
            lamp.energy = 25.0
            lamp.shadow_soft_size = 0.05
            link_object(bpy.data.objects.new(name + "_Lamp", lamp), self.coll, root).location = bulb.location
        # Tail and belly strobes (white, 1 Hz double flash, animated later).
        for name, loc in (("Strobe_Tail", (-6.32, 0.0, 0.22)), ("Strobe_Fin", (-5.75, 0.0, 1.98))):
            bm = bmesh.new()
            bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.04)
            mesh_object(name, bm, self.coll, root, [self.m.strobe], smooth=True).location = loc
            lamp = bpy.data.lights.new(name + "_Lamp", "POINT")
            lamp.energy = 0.0
            lamp.shadow_soft_size = 0.02
            obj = link_object(bpy.data.objects.new(name + "_Lamp", lamp), self.coll, root)
            obj.location = loc
            rig.strobes.append(lamp)
        # Landing light in the cowling: volumetric beam through the haze.
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, segments=24, radius=0.07, matrix=Matrix.Rotation(math.pi / 2, 4, "Y"))
        mesh_object("Landing_Light_Lens", bm, self.coll, root, [self.m.landing_lens]).location = (1.93, 0.22, -0.18)
        spot = bpy.data.lights.new("Landing_Light", "SPOT")
        spot.energy = 6000.0
        spot.spot_size = math.radians(24.0)
        spot.spot_blend = 0.6
        spot.color = (1.0, 0.94, 0.82)
        spot.shadow_soft_size = 0.05
        if hasattr(spot, "volume_factor"):
            spot.volume_factor = 1.5
        obj = link_object(bpy.data.objects.new("Landing_Light", spot), self.coll, root)
        obj.location = (1.95, 0.22, -0.18)
        # Spot lights shine along local -Z: aim forward (+X) and 6 deg down.
        obj.rotation_euler = (0.0, -math.pi / 2 - math.radians(6.0), 0.0)

    def _registration(self, root):
        """Registration marks on both sides of the tail cone, laid flat against the tapering skin."""
        xs = [st[0] for st in FUSELAGE_STATIONS][::-1]
        hw = [st[1] for st in FUSELAGE_STATIONS][::-1]
        zc = [0.5 * (st[2] + st[3]) for st in FUSELAGE_STATIONS][::-1]
        x_fwd, x_aft = -3.0, -4.1
        hw_f, hw_a = float(np.interp(x_fwd, xs, hw)), float(np.interp(x_aft, xs, hw))
        z_mid = float(np.interp(0.5 * (x_fwd + x_aft), xs, zc))
        up = Vector((0.0, 0.0, 1.0))
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            front = Vector((x_fwd, sgn * (hw_f + 0.004), z_mid))
            aft = Vector((x_aft, sgn * (hw_a + 0.004), z_mid))
            # Text reads nose-to-tail on the left side and tail-to-nose on the right side.
            reading = (aft - front).normalized() if sgn > 0 else (front - aft).normalized()
            normal = reading.cross(up).normalized()
            curve = bpy.data.curves.new(f"Reg_{side}", "FONT")
            curve.body = "N172XW"
            curve.size = 0.2
            curve.extrude = 0.001
            curve.align_x = "CENTER"
            curve.align_y = "CENTER"
            obj = link_object(bpy.data.objects.new(f"Registration_{side}", curve), self.coll, root)
            obj.data.materials.append(self.m.prop_black)
            basis = Matrix((reading, up, normal)).transposed().to_4x4()
            obj.matrix_basis = Matrix.Translation((front + aft) * 0.5 + normal * 0.002) @ basis


# =============================================================================
# Environment
# =============================================================================
class EnvironmentBuilder:
    def __init__(self, coll, mats: Materials, tel: Telemetry, cfg: RenderConfig, sun_dir: Vector) -> None:
        self.coll = coll
        self.m = mats
        self.tel = tel
        self.cfg = cfg
        self.sun_dir = sun_dir
        rw = tel.runway
        self.length = float(rw["length_m"])
        self.width = float(rw["width_m"])
        self.aim = float(rw["aim_point_m"])
        self.gs = math.radians(float(rw["glideslope_deg"]))
        self.rng = random.Random(cfg.seed)

    def build(self, scene) -> None:
        self._runway()
        self._markings()
        self._lights()
        self._papi()
        self._terrain()
        self._trees()
        from airfield_assets import AirfieldBuilder  # sibling module (src/viz is on sys.path)

        AirfieldBuilder(self).build(scene)
        scene["alma_runway_half_width"] = self.width / 2
        self._clouds()
        self._sky_and_sun(scene)

    # -- runway ------------------------------------------------------------------
    def _runway(self):
        bm = bmesh.new()
        hw = self.width / 2 + 7.5
        # Dense enough to carry per-vertex surface variation (rubber deposits, patches) in the glTF export.
        bmesh.ops.create_grid(bm, x_segments=24, y_segments=400, size=0.5)
        for v in bm.verts:
            v.co.x *= 2 * hw
            v.co.y = -60.0 + (v.co.y + 0.5) * (self.length + 120.0)
        mesh_object("Runway", bm, self.coll, None, [self.m.asphalt])

    def _markings(self):
        L, hw = self.length, self.width / 2
        rects: list[tuple[float, float, float, float]] = []
        for k in range(4):  # threshold "piano keys"
            for s in (1, -1):
                xc = s * (2.7 + 3.6 * k)
                rects.append((xc - 0.9, xc + 0.9, 6.0, 36.0))
        rects += [(-hw, -hw + 0.9, 0.0, L), (hw - 0.9, hw, 0.0, L)]  # edge lines
        rects.append((-hw, hw, -0.9, 0.0))  # threshold bar
        y = 70.0
        while y + 30.0 < L - 40.0:  # centreline dashes 30 m, 20 m gaps
            rects.append((-0.45, 0.45, y, y + 30.0))
            y += 50.0
        rects += [(6.0, 10.0, self.aim, self.aim + 45.0), (-10.0, -6.0, self.aim, self.aim + 45.0)]  # aiming point
        for y0, n in ((150.0, 3), (450.0, 2), (600.0, 2), (750.0, 1)):  # touchdown-zone markings
            for k in range(n):
                for s in (1, -1):
                    x0 = 6.0 + k * 3.3
                    rects.append((x0, x0 + 1.8, y0, y0 + 22.5) if s > 0 else (-x0 - 1.8, -x0, y0, y0 + 22.5))
        rects += self._designator("36", 42.0)

        bm = bmesh.new()
        for x0, x1, y0, y1 in rects:
            vs = [bm.verts.new((x, y, 0.006)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            bm.faces.new(vs)
        mesh_object("Runway_Markings", bm, self.coll, None, [self.m.paint])

    @staticmethod
    def _designator(text: str, y0: float, h: float = 9.0, w: float = 3.0, stroke: float = 0.9):
        """Seven-segment runway designator, upright when read from the approach."""
        segs = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg",
                "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
        rects = []
        gap = 1.8
        total = len(text) * w + (len(text) - 1) * gap
        for i, ch in enumerate(text):
            x0 = -total / 2 + i * (w + gap)
            x1 = x0 + w
            ym = y0 + h / 2
            geo = {
                "a": (x0, x1, y0 + h - stroke, y0 + h), "g": (x0, x1, ym - stroke / 2, ym + stroke / 2), "d": (x0, x1, y0, y0 + stroke),
                "f": (x0, x0 + stroke, ym, y0 + h), "b": (x1 - stroke, x1, ym, y0 + h),
                "e": (x0, x0 + stroke, y0, ym), "c": (x1 - stroke, x1, y0, ym),
            }
            rects += [geo[s] for s in segs[ch]]
        return rects

    def _bulbs(self, name, positions, mat, radius=0.16):
        bm = bmesh.new()
        base = bmesh.new()
        for p in positions:
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=radius, matrix=Matrix.Translation(Vector(p) + Vector((0, 0, 0.32))))
            add_cylinder(base, Vector(p), Vector(p) + Vector((0, 0, 0.26)), 0.06, 0.05, 8)
        mesh_object(name, bm, self.coll, None, [mat], smooth=True)
        mesh_object(name + "_Fixtures", base, self.coll, None, [self.m.fixture])

    def _lights(self):
        L, hw = self.length, self.width / 2
        white, yellow = [], []
        for k in range(int(L // 60) + 1):
            y = min(k * 60.0, L)
            for s in (1, -1):
                (yellow if y > L - 600.0 else white).append((s * (hw + 1.5), y, 0.0))
        self._bulbs("Edge_Lights_White", white, self.m.light_white)
        self._bulbs("Edge_Lights_Yellow", yellow, self.m.light_yellow)
        xs = np.arange(-hw - 4.5, hw + 4.6, 3.0)
        self._bulbs("Threshold_Lights", [(x, -2.0, 0.0) for x in xs], self.m.light_green, 0.2)
        self._bulbs("End_Lights", [(x, L + 2.0, 0.0) for x in xs], self.m.light_red, 0.2)
        # Approach lighting system: 30 barrettes at 30 m spacing on frangible masts with crossbars,
        # plus crossbars at 300 m and 600 m.
        als = []
        masts = bmesh.new()
        for k in range(1, 31):
            y = -30.0 * k
            z = self.als_height(k)
            for dx in (-2.0, -1.0, 0.0, 1.0, 2.0):
                als.append((dx, y, z))
            add_cylinder(masts, Vector((0.0, y, -0.2)), Vector((0.0, y, z)), 0.07, 0.05, 8)
            add_box(masts, -2.3, 2.3, y - 0.05, y + 0.05, z - 0.08, z)
            if k in (10, 20):
                for x in np.arange(-15.0, 15.1, 1.5):
                    als.append((x, y, z))
                add_box(masts, -15.3, 15.3, y - 0.05, y + 0.05, z - 0.08, z)
                for x in (-12.0, -6.0, 6.0, 12.0):
                    add_cylinder(masts, Vector((x, y, -0.2)), Vector((x, y, z)), 0.06, 0.045, 8)
        mesh_object("Approach_Light_Masts", masts, self.coll, None, [self.m.fixture])
        self._bulbs("Approach_Lights", als, self.m.light_white, 0.2)

    def _papi(self):
        """PAPI on the left of the aim point: 2 red / 2 white on the 3 deg path."""
        bm_r, bm_w, bm_box = bmesh.new(), bmesh.new(), bmesh.new()
        for k in range(4):
            x = -(self.width / 2 + 15.0 + 9.0 * k)
            add_box(bm_box, x - 0.6, x + 0.6, self.aim - 0.4, self.aim + 0.4, 0.0, 0.7)
            target = bm_r if k < 2 else bm_w
            bmesh.ops.create_circle(target, cap_ends=True, segments=16, radius=0.17,
                                    matrix=Matrix.Translation((x, self.aim - 0.42, 0.42)) @ Matrix.Rotation(math.pi / 2, 4, "X"))
        mesh_object("PAPI_Housing", bm_box, self.coll, None, [self.m.fixture])
        mesh_object("PAPI_Red", bm_r, self.coll, None, [self.m.light_red])
        mesh_object("PAPI_White", bm_w, self.coll, None, [self.m.light_white])

    # -- terrain -----------------------------------------------------------------
    @staticmethod
    def als_height(k: int) -> float:
        """Height of approach-light barrette ``k`` (1 = nearest the threshold) above ground."""
        return 1.0 + 0.05 * k

    def terrain_height(self, x: float, y: float) -> float:
        # Flat aerodrome: 160 m west of the centreline, 520 m east (taxiway, apron, hangars, fuel farm).
        dx = max(0.0, x - 520.0) if x > 0 else max(0.0, -x - 160.0)
        dy = max(0.0, -1300.0 - y, y - (self.length + 450.0))
        d = math.hypot(dx, dy)
        m = smoothstep(0.0, 1400.0, d)
        p = Vector((x / 1100.0, y / 1100.0, 0.37))
        n = 0.0
        amp, freq = 1.0, 1.0
        for _ in range(5):
            n += amp * noise.noise(p * freq)
            amp *= 0.5
            freq *= 2.1
        return -0.04 + m * (14.0 + 70.0 * m * (0.5 + 0.5 * n))

    def _terrain(self):
        x0, x1 = -5000.0, 5000.0
        y0, y1 = -6000.0, self.length + 5000.0
        nx, ny = 180, 220
        bm = bmesh.new()
        verts = []
        for j in range(ny + 1):
            row = []
            y = y0 + (y1 - y0) * j / ny
            for i in range(nx + 1):
                x = x0 + (x1 - x0) * i / nx
                row.append(bm.verts.new((x, y, self.terrain_height(x, y))))
            verts.append(row)
        for j in range(ny):
            for i in range(nx):
                bm.faces.new((verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i]))
        mesh_object("Terrain", bm, self.coll, None, [self.m.grass], smooth=True)

    def _trees(self):
        bm_leaf, bm_trunk = bmesh.new(), bmesh.new()
        count = 0
        attempts = 0
        target = 260 if self.cfg.preview else 650
        while count < target and attempts < target * 20:
            attempts += 1
            x = self.rng.uniform(-3500.0, 3500.0)
            y = self.rng.uniform(-4500.0, self.length + 3500.0)
            if -260.0 < x < 620.0 and -1600.0 < y < self.length + 700.0:
                continue
            # cluster: accept with probability from low-frequency noise
            if noise.noise(Vector((x / 600.0, y / 600.0, 1.7))) < 0.05:
                continue
            z = self.terrain_height(x, y)
            h = self.rng.uniform(9.0, 19.0)
            add_cylinder(bm_trunk, Vector((x, y, z - 0.5)), Vector((x, y, z + h * 0.45)), 0.25 * h / 14, 0.15 * h / 14, 6)
            for k in range(2):
                r = h * self.rng.uniform(0.22, 0.32) * (1.0 - 0.25 * k)
                c = Vector((x + self.rng.uniform(-1, 1), y + self.rng.uniform(-1, 1), z + h * (0.55 + 0.25 * k)))
                bmesh.ops.create_icosphere(bm_leaf, subdivisions=2, radius=r,
                                           matrix=Matrix.Translation(c) @ Matrix.Diagonal((1.0, 1.0, 1.25, 1.0)))
            count += 1
        mesh_object("Tree_Foliage", bm_leaf, self.coll, None, [self.m.foliage], smooth=True)
        mesh_object("Tree_Trunks", bm_trunk, self.coll, None, [self.m.bark])

    def _clouds(self):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        obj = mesh_object("Cloud_Deck", bm, self.coll, None, [self.m.clouds])
        if self.cfg.mood == "golden":
            obj.location = (0.0, self.length / 2, 900.0)
            obj.scale = (6000.0, 7000.0, 260.0)
        else:
            obj.location = (0.0, self.length / 2, 520.0)
            obj.scale = (6000.0, 7000.0, 320.0)

    def _haze_volume(self):
        """Aerosol haze in a bounded domain with exponential height falloff.

        A *world* volume is deliberately avoided: EEVEE renders world volumes as an infinite
        atmosphere that absorbs all distant (sky) lighting. Extinction at ground level is
        6e-5 /m (~65 km visibility, golden hour) or 2.5e-4 /m (~15 km, storm), scale height 450 m.
        """
        mat, nb, out = new_material("Atmospheric_Haze")
        tc = nb.new("ShaderNodeTexCoord")
        _, _, z = nb.separate(nb.out(tc, "Object"))
        height_m = nb.math("MULTIPLY", nb.math("ADD", z, 1.0), 700.0)  # object z in [-1, 1] -> 0..1400 m
        falloff = nb.math("EXPONENT", nb.math("DIVIDE", height_m, -450.0))
        base = 0.00006 if self.cfg.mood == "golden" else 0.00025
        color = (1.0, 0.86, 0.72, 1) if self.cfg.mood == "golden" else (0.72, 0.76, 0.8, 1)
        vol = nb.new("ShaderNodeVolumePrincipled", {"Color": color, "Density": nb.math("MULTIPLY", falloff, base), "Anisotropy": 0.72})
        nb.link(vol.outputs[0], out.inputs["Volume"])
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        obj = mesh_object("Haze_Volume", bm, self.coll, None, [mat])
        obj.location = (0.0, self.length / 2, 690.0)
        obj.scale = (7000.0, 8000.0, 700.0)

    # -- sky ---------------------------------------------------------------------
    def _sky_and_sun(self, scene):
        world = bpy.data.worlds.new("Teaser_World")
        scene.world = world
        if world.node_tree is None:
            world.use_nodes = True
        nb = Nodes(world.node_tree)
        out = nb.new("ShaderNodeOutputWorld")
        sky = nb.new("ShaderNodeTexSky")
        set_first(sky, "sky_type", "MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE")
        elevation = math.asin(max(min(self.sun_dir.z, 1.0), -1.0))
        # Blender's physical sky places the solar disc at (sin r, -cos r) * cos(elevation)
        # (verified against renders), so r = atan2(dir.x, -dir.y) matches the sun lamp.
        sky.sun_elevation = elevation
        sky.sun_rotation = math.atan2(self.sun_dir.x, -self.sun_dir.y) % (2 * math.pi)
        sky.sun_disc = True
        sky.sun_size = math.radians(0.8)
        if hasattr(sky, "air_density"):
            sky.air_density = 1.0 if self.cfg.mood == "golden" else 1.6
            sky.aerosol_density = 2.5 if self.cfg.mood == "golden" else 5.0
        bg = nb.new("ShaderNodeBackground", {"Color": sky.outputs[0], "Strength": 0.32 if self.cfg.mood == "golden" else 0.12})
        nb.link(bg.outputs[0], out.inputs["Surface"])
        self._haze_volume()

        sun = bpy.data.lights.new("Sun", "SUN")
        sun.energy = 4.2 if self.cfg.mood == "golden" else 1.4
        sun.color = (1.0, 0.68, 0.42) if self.cfg.mood == "golden" else (0.78, 0.84, 0.95)
        sun.angle = math.radians(0.7)
        if hasattr(sun, "volume_factor"):
            sun.volume_factor = 1.0
        obj = link_object(bpy.data.objects.new("Sun", sun), self.coll)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = self.sun_dir.to_track_quat("Z", "Y")  # light shines along -Z


# =============================================================================
# Animation
# =============================================================================
class Animator:
    def __init__(self, tel: Telemetry, tl: Timeline, rig: AircraftRig, cfg: RenderConfig) -> None:
        self.tel = tel
        self.tl = tl
        self.rig = rig
        self.cfg = cfg
        margin = 1.5
        self.idx = np.where((tel.t >= tl.t0 - margin) & (tel.t <= tl.t1 + margin))[0]
        self.frames = tl.frame(tel.t[self.idx])

    def run(self) -> None:
        self.airframe()
        self.surfaces()
        self.gear()
        self.propeller()
        self.strobes()

    def airframe(self) -> None:
        root = self.rig.root
        keyframe_vector(root, "location", self.frames, self.tel.loc[self.idx])
        keyframe_vector(root, "rotation_quaternion", self.frames, self.tel.quat[self.idx])

    def surfaces(self) -> None:
        tel = self.tel
        if self.cfg.surface_source == "action":
            lim = tel.limits_deg
            defl = tel.action[:, :3] * np.array([lim["elevator"], lim["aileron"], lim["rudder"]])
        else:
            defl = tel.deflection_deg
        gain = self.cfg.surface_gain
        mapping = {"Elevator": 0, "Elevator_L": 0, "Elevator_R": 0, "Aileron_L": 1, "Aileron_R": 1, "Rudder": 2}
        for name, (obj, axis, sign) in self.rig.surfaces.items():
            ang = np.radians(defl[self.idx, mapping[name]]) * sign * gain
            keyframe_series(obj, "rotation_axis_angle", 0, self.frames, ang)

    def gear(self) -> None:
        tel = self.tel
        dt = np.gradient(tel.t)
        vel = np.gradient(tel.loc, axis=0) / dt[:, None]
        for name, (hub, rest) in self.rig.gear_hubs.items():
            comp = np.clip(tel.gear_compression(name), 0.0, 0.35)
            z = rest.z + comp[self.idx]
            loc = np.tile(np.array(rest), (len(self.idx), 1))
            loc[:, 2] = z
            keyframe_vector(hub, "location", self.frames, loc)
            # Wheel spin-up: angular velocity = forward ground speed / radius while loaded.
            wheel, radius = self.rig.wheels[name]
            heading = np.array([Quaternion(q).to_matrix() @ Vector((1, 0, 0)) for q in tel.quat])
            v_fwd = np.einsum("ij,ij->i", vel, heading)
            contact = comp > 1e-4
            ever = np.maximum.accumulate(contact)
            omega = np.where(ever, v_fwd / radius, 0.0)
            angle = np.concatenate(([0.0], np.cumsum(0.5 * (omega[1:] + omega[:-1]) * np.diff(tel.t))))
            # Rolling forward about +Y (model left axis) is a negative rotation for +X travel.
            keyframe_series(wheel, "rotation_euler", 1, self.frames, -angle[self.idx], interpolation="LINEAR")
        if self.rig.nose_steer is not None:
            lim = tel.limits_deg["rudder"]
            steer = np.radians(10.0) * tel.deflection_deg[:, 2] / lim
            keyframe_series(self.rig.nose_steer, "rotation_euler", 2, self.frames, steer[self.idx])

    def propeller(self) -> None:
        tel = self.tel
        rpm = 750.0 + 1950.0 * np.clip(tel.throttle, 0.0, 1.0)
        omega = rpm / 60.0 * 2.0 * math.pi
        angle = np.concatenate(([0.0], np.cumsum(0.5 * (omega[1:] + omega[:-1]) * np.diff(tel.t))))
        for prop in getattr(self.rig, "props", None) or [self.rig.prop]:
            keyframe_series(prop, "rotation_euler", 0, self.frames, angle[self.idx], interpolation="LINEAR")

    def strobes(self) -> None:
        """Double-flash anti-collision strobes: 60 ms flashes at 0 and 0.18 s every 1.1 s."""
        frames, values = [], []
        t = self.tl.t0
        while t < self.tl.t1:
            for off in (0.0, 0.18):
                for dtt, v in ((0.0, 0.0), (0.001, 1.0), (0.06, 0.0)):
                    frames.append(float(self.tl.frame(t + off + dtt)))
                    values.append(v)
            t += 1.1
        for lamp in self.rig.strobes:
            keyframe_series(lamp, "energy", 0, frames, [v * 900.0 for v in values], interpolation="CONSTANT")
        mat = bpy.data.materials.get("Strobe_White")
        if mat is not None:
            bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            sock = Nodes.inp(bsdf, "Emission Strength")
            path = sock.path_from_id("default_value")
            for f, v in zip(frames, values):
                sock.default_value = v * 400.0
                mat.node_tree.keyframe_insert(path, frame=f)
            for fc in _channelbag_fcurves(mat.node_tree):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


class TireSmoke:
    """Volumetric puffs spawned at each main-gear contact point at its first touchdown,
    advected by the recorded near-ground wind."""

    def __init__(self, coll, mats: Materials, tel: Telemetry, tl: Timeline, rig: AircraftRig, cfg: RenderConfig) -> None:
        self.coll, self.m, self.tel, self.tl, self.rig, self.cfg = coll, mats, tel, tl, rig, cfg
        self.rng = random.Random(cfg.seed + 11)

    def build(self) -> list[float]:
        tel = self.tel
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        me = bpy.data.meshes.new("Smoke_Puff")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(self.m.smoke)
        contact_times = []
        for name in tel.gear_flu:
            if name == "nose":
                continue
            comp = tel.gear_compression(name)
            hits = np.where(comp > 1e-4)[0]
            if hits.size == 0:
                continue
            i0 = int(hits[0])
            # Interpolate the exact contact instant between samples.
            t_c = tel.t[i0]
            if i0 > 0:
                z_prev = -tel.gear_world(name, i0 - 1).z
                z_now = -tel.gear_world(name, i0).z
                frac = (0.0 - z_prev) / (z_now - z_prev) if z_now != z_prev else 1.0
                t_c = tel.t[i0 - 1] + float(np.clip(frac, 0.0, 1.0)) * (tel.t[i0] - tel.t[i0 - 1])
            contact_times.append(t_c)
            self._emit(name, t_c, me)
        return contact_times

    def _contact_world(self, name: str, t: float) -> Vector:
        loc = Vector(self.tel.interp(self.tel.loc, t))
        return loc + self.tel.quat_at(t).to_matrix() @ self.tel.gear_flu[name]

    def _emit(self, name: str, t_c: float, me) -> None:
        n = self.cfg.smoke_puffs_per_wheel
        emit_span = 0.65
        wind = Vector(self.tel.interp(self.tel.wind, t_c))
        wind.z = 0.0
        for j in range(n):
            ts = t_c + emit_span * (j / max(n - 1, 1)) ** 1.4 + self.rng.uniform(0.0, 0.02)
            p0 = self._contact_world(name, ts)
            p0.z = 0.12
            v_ac = (self._contact_world(name, ts + 0.02) - self._contact_world(name, ts)) / 0.02
            v_ac.z = 0.0
            jitter = Vector((self.rng.gauss(0, 0.35), self.rng.gauss(0, 0.35), self.rng.uniform(0.25, 0.7)))
            obj = link_object(bpy.data.objects.new(f"Smoke_{name}_{j:03d}", me), self.coll)
            obj.rotation_euler = (self.rng.uniform(0, 6.3), self.rng.uniform(0, 6.3), self.rng.uniform(0, 6.3))
            strength = 1.0 - 0.6 * j / max(n - 1, 1)
            taus = [0.0, 0.08, 0.25, 0.5, 0.9, 1.5, 2.3, 3.4, 4.6]
            frames, locs, scales, alphas = [], [], [], []
            f_pre = float(self.tl.frame(ts)) - 0.01
            frames.append(f_pre)
            locs.append(np.array(p0))
            scales.append(0.001)
            alphas.append(0.0)
            for tau in taus:
                drag = v_ac * 0.22 * (1.0 - math.exp(-tau / 0.25))  # rubber smoke thrown forward, then drag
                p = p0 + drag + (wind * 0.85 + jitter) * tau
                p.z = 0.12 + 0.35 * tau + jitter.z * tau * 0.5
                s = 0.22 + 2.6 * (1.0 - math.exp(-tau / 1.1))
                a = strength * min(1.0, tau / 0.06) * math.exp(-tau / 1.5) if tau > 0 else 0.0
                frames.append(float(self.tl.frame(ts + tau)))
                locs.append(np.array(p))
                scales.append(s)
                alphas.append(a)
            frames.append(frames[-1] + 1.0)
            locs.append(locs[-1])
            scales.append(0.001)
            alphas.append(0.0)
            keyframe_vector(obj, "location", frames, np.array(locs))
            keyframe_vector(obj, "scale", frames, np.repeat(np.array(scales)[:, None], 3, axis=1))
            obj.color = (1.0, 1.0, 1.0, 0.0)
            keyframe_series(obj, "color", 3, frames, alphas)
            # The pre-spawn key must not ease into the spawn.
            for fc in _channelbag_fcurves(obj):
                fc.keyframe_points[0].interpolation = "CONSTANT"


# =============================================================================
# Cameras
# =============================================================================
def _shake_signal(t: np.ndarray, seed: int) -> np.ndarray:
    """Band-limited pseudo-random shake (3 axes), unit RMS, from incommensurate sines."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(t), 3))
    for axis in range(3):
        for f, a in ((0.9, 1.0), (2.3, 0.7), (4.7, 0.45), (9.1, 0.25), (15.3, 0.12)):
            out[:, axis] += a * np.sin(2 * math.pi * f * rng.uniform(0.85, 1.15) * t + rng.uniform(0, 2 * math.pi))
    return out / np.sqrt(np.mean(out**2, axis=0, keepdims=True))


@dataclass
class Shot:
    name: str
    t_start: float
    t_end: float
    camera: Optional[bpy.types.Object] = None


class CameraDirector:
    def __init__(self, coll, tel: Telemetry, tl: Timeline, rig: AircraftRig, cfg: RenderConfig, side: float, touchdown_t: float) -> None:
        self.coll, self.tel, self.tl, self.rig, self.cfg = coll, tel, tl, rig, cfg
        self.side = side  # +1: cameras east of the runway, -1: west
        self.td = touchdown_t
        self.smooth = tel.smoothed(tel.loc, 1.2)
        g = tel.smoothed(tel.gust_intensity, 0.25)
        self.gust = np.clip(g / 4.0, 0.0, 1.6)

    def _times(self, a: float, b: float) -> np.ndarray:
        step = 1.0 / self.cfg.fps
        return np.arange(a - 0.5, b + 0.5 + step, step)

    def _camera(self, name, lens, fstop, focus, parent=None, clip_end=12000.0):
        cam = bpy.data.cameras.new(name)
        cam.lens = lens
        cam.sensor_width = 36.0
        cam.clip_start = 0.1
        cam.clip_end = clip_end
        cam.dof.use_dof = True
        cam.dof.aperture_fstop = fstop
        cam.dof.focus_object = focus
        cam.dof.aperture_blades = 7
        obj = link_object(bpy.data.objects.new(name, cam), self.coll, parent)
        return obj

    def _apply_shake(self, cam_obj, times, base_deg: float, gust_deg: float, seed: int, base_loc=(0, 0, 0), impulse_t=None):
        sig = _shake_signal(times, seed)
        g = np.interp(times, self.tel.t, self.gust)
        amp = np.radians(base_deg + gust_deg * g)
        if impulse_t is not None:
            dt_imp = times - impulse_t
            amp += np.where(dt_imp >= 0.0, np.radians(1.6) * np.exp(-dt_imp / 0.35), 0.0)
        rot = sig * amp[:, None] * np.array([1.0, 0.6, 0.35])
        frames = self.tl.frame(times)
        cam_obj.rotation_mode = "XYZ"
        keyframe_vector(cam_obj, "rotation_euler", frames, rot)
        loc = np.array(base_loc)[None, :] + 0.02 * sig[:, ::-1] * (1.0 + g)[:, None]
        keyframe_vector(cam_obj, "location", frames, loc)

    def _tracking_rig(self, name, times, offset_fn, target_fn):
        rig = new_empty(f"{name}_Rig", self.coll, size=1.0)
        target = new_empty(f"{name}_Target", self.coll, size=0.5)
        cam_pos = np.array([offset_fn(t) for t in times])
        tgt_pos = np.array([target_fn(t) for t in times])
        frames = self.tl.frame(times)
        keyframe_vector(rig, "location", frames, cam_pos)
        keyframe_vector(target, "location", frames, tgt_pos)
        c = rig.constraints.new("TRACK_TO")
        c.target = target
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"
        return rig, target

    def smooth_at(self, t: float) -> Vector:
        return Vector([np.interp(t, self.tel.t, self.smooth[:, k]) for k in range(3)])

    def actual_at(self, t: float) -> Vector:
        return Vector(self.tel.interp(self.tel.loc, t))

    def build(self, shots: list[Shot]) -> None:
        s1, s2, s3 = shots
        side = self.side

        # Shot 1: low-angle tracking dolly on the downwind side, arcing in towards the nose.
        t1 = self._times(s1.t_start, s1.t_end)

        def off1(t):
            u = (t - s1.t_start) / max(s1.t_end - s1.t_start, 1e-6)
            p = self.smooth_at(t)
            # Starts wide and ahead-left of the nose, then dollies in alongside and below.
            q = Vector((side * (15.0 - 4.0 * u), 16.0 - 12.0 * u, p.z - 6.5 + 2.0 * u))
            out = Vector((p.x + q.x, p.y + q.y, max(q.z, 1.6)))
            return np.array(out)

        rig1, tgt1 = self._tracking_rig("Shot1", t1, off1, lambda t: np.array(self.actual_at(t) + Vector((0, 1.0, 0.4))))
        cam1 = self._camera("Cam_Shot1_LowTracking", 35.0, 5.6, self.rig.root, rig1)
        self._apply_shake(cam1, t1, 0.05, 0.75, self.cfg.seed + 1)
        s1.camera = cam1

        # Shot 2: airframe-mounted close-up behind the left tailplane, looking forward along
        # the empennage and the left aileron.
        t2 = self._times(s2.t_start, s2.t_end)
        ms = side_sign_model(side)
        mount = new_empty("Shot2_Mount", self.coll, self.rig.root, location=(-10.0, 3.6 * ms, 1.9), size=0.5)
        # Look forward past the elevator and rudder towards the aileron (camera looks along -Z, up +Y).
        look = Vector((-3.2, 1.9 * ms, 0.75)) - Vector(mount.location)
        mount.rotation_mode = "QUATERNION"
        mount.rotation_quaternion = look.to_track_quat("-Z", "Y")
        cam2 = self._camera("Cam_Shot2_ControlSurfaces", 28.0, 3.2, self.rig.focus_tail, mount)
        self._apply_shake(cam2, t2, 0.04, 0.55, self.cfg.seed + 2)
        s2.camera = cam2

        # Shot 3: lagging chase camera, slightly upwind of the centreline and well behind, on a
        # longer lens aimed between the aircraft and its tire-smoke trail; the crosswind carries
        # the smoke across frame and away from the lens.
        t3 = self._times(s3.t_start, s3.t_end)

        def off3(t):
            p = self.smooth_at(t - 0.35)
            u = (t - s3.t_start) / max(s3.t_end - s3.t_start, 1e-6)
            return np.array((p.x - side * (3.2 - 1.0 * u), p.y - 26.0 + 6.0 * u, max(p.z + 2.0, 2.4)))

        def aim3(t):
            p = self.actual_at(t)
            return np.array((p.x + side * 1.2, p.y - 5.0, max(p.z - 0.6, 0.6)))

        rig3, tgt3 = self._tracking_rig("Shot3", t3, off3, aim3)
        # EEVEE distributes volume froxels over the camera clip range; a short far clip keeps
        # the metre-scale smoke puffs resolved (the haze and sky still fill the horizon).
        cam3 = self._camera("Cam_Shot3_Chase", 50.0, 5.6, self.rig.root, rig3, clip_end=1200.0)
        self._apply_shake(cam3, t3, 0.06, 0.6, self.cfg.seed + 3, impulse_t=self.td)
        s3.camera = cam3


def side_sign_model(side_world: float) -> float:
    """Shot 2 mounts on the left tailplane (model +Y) when cameras sit west of a north runway."""
    return 1.0 if side_world < 0 else -1.0


# =============================================================================
# Render setup
# =============================================================================
def configure_render(scene, cfg: RenderConfig, tl: Timeline) -> Path:
    r = scene.render
    r.resolution_x, r.resolution_y = cfg.resolution
    r.resolution_percentage = 100
    r.fps = cfg.fps
    r.fps_base = 1.0
    scene.frame_start = tl.frame_start
    scene.frame_end = tl.frame_end
    if cfg.frame_range is not None:
        scene.frame_start = max(tl.frame_start, cfg.frame_range[0])
        scene.frame_end = min(tl.frame_end, cfg.frame_range[1])
    scene.frame_step = cfg.frame_step
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.5
    r.film_transparent = False

    if cfg.engine.upper() == "CYCLES":
        r.engine = "CYCLES"
        cy = scene.cycles
        cy.samples = max(cfg.samples * 4, 64)
        cy.use_denoising = True
        cy.max_bounces = 8
        cy.volume_bounces = 2
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                try:
                    prefs.compute_device_type = backend
                    prefs.get_devices()
                    if any(d.type == backend for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == backend
                        cy.device = "GPU"
                        break
                except TypeError:
                    continue
        except KeyError:
            pass
    else:
        set_first(r, "engine", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")
        ee = scene.eevee
        ee.taa_render_samples = cfg.samples
        for prop, value in (
            ("use_raytracing", True),  # screen-space / hybrid reflections
            ("use_fast_gi", True),  # horizon-scan ambient occlusion + GI
            ("fast_gi_distance", 6.0),
            ("use_shadows", True),
            ("shadow_ray_count", 2),
            ("shadow_step_count", 10),
            # Without a custom range EEVEE spreads the volume froxels over the full camera clip
            # range, which leaves metre-scale tire smoke unresolved near the lens.
            ("use_volume_custom_range", True),
            ("volumetric_start", 0.5),
            ("volumetric_end", 6000.0),
            ("volumetric_samples", 96 if not cfg.preview else 48),
            ("volumetric_sample_distribution", 0.9),
            ("use_volumetric_shadows", True),
            ("volumetric_shadow_samples", 24),
            ("volumetric_ray_depth", 8),
            ("motion_blur_steps", 4 if not cfg.preview else 1),
            # legacy EEVEE (<= 4.1) equivalents
            ("use_ssr", True),
            ("use_ssr_refraction", True),
            ("use_gtao", True),
            ("use_bloom", True),
        ):
            if hasattr(ee, prop):
                try:
                    setattr(ee, prop, value)
                except (TypeError, ValueError, AttributeError):
                    pass
        if hasattr(ee, "volumetric_tile_size"):
            set_first(ee, "volumetric_tile_size", "4" if not cfg.preview else "8")
        if hasattr(ee, "ray_tracing_method"):
            set_first(ee, "ray_tracing_method", "SCREEN")
        if hasattr(ee, "ray_tracing_options"):
            opts = ee.ray_tracing_options
            for prop, value in (("resolution_scale", "1"), ("use_denoise", True), ("screen_trace_quality", 0.75)):
                if hasattr(opts, prop):
                    try:
                        setattr(opts, prop, value)
                    except (TypeError, ValueError):
                        pass

    vs = scene.view_settings
    set_first(vs, "view_transform", "AgX", "Filmic")
    set_first(vs, "look", "AgX - Medium High Contrast", "AgX - Punchy", "Medium High Contrast", "None")
    vs.exposure = 0.15 if cfg.mood == "golden" else 0.35

    return configure_output(scene, cfg)


def configure_output(scene, cfg: RenderConfig) -> Path:
    r = scene.render
    out = Path(cfg.output).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    im = r.image_settings
    codec = cfg.codec.lower()
    if codec == "png-sequence":
        if hasattr(im, "media_type"):
            im.media_type = "IMAGE"
        im.file_format = "PNG"
        im.color_mode = "RGB"
        im.color_depth = "16"
        im.compression = 15
        seq_dir = out.with_suffix("")
        seq_dir.mkdir(parents=True, exist_ok=True)
        r.filepath = str(seq_dir / "frame_")
        return seq_dir
    if hasattr(im, "media_type"):
        im.media_type = "VIDEO"
    im.file_format = "FFMPEG"
    ff = r.ffmpeg
    container, ffcodec, suffix = {
        "qtrle": ("QUICKTIME", "QTRLE", ".mov"),
        "prores": ("QUICKTIME", "PRORES", ".mov"),
        "png": ("QUICKTIME", "PNG", ".mov"),
        "ffv1": ("MKV", "FFV1", ".mkv"),
        "h264": ("MPEG4", "H264", ".mp4"),
    }[codec]
    ff.format = container
    ff.codec = ffcodec
    im.color_mode = "RGB"
    if codec == "prores" and hasattr(ff, "ffmpeg_prores_profile"):
        set_first(ff, "ffmpeg_prores_profile", "4444_XQ", "4444", "HQ")
    if codec == "h264":
        set_first(ff, "constant_rate_factor", "HIGH")
        set_first(ff, "ffmpeg_preset", "GOOD")
    if codec in ("ffv1", "png", "qtrle") and hasattr(ff, "use_lossless_output"):
        ff.use_lossless_output = True
    ff.audio_codec = "NONE"
    out = out.with_suffix(suffix)
    r.filepath = str(out)
    r.use_file_extension = False
    return out


def configure_compositor(scene, cfg: RenderConfig) -> None:
    """Bloom, chromatic aberration, teal/orange grade and vignette (Blender 5 node-group compositor
    with a fallback for the legacy ``scene.node_tree`` API)."""
    if hasattr(scene, "compositing_node_group"):
        tree = bpy.data.node_groups.new("Teaser_Compositor", "CompositorNodeTree")
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        scene.compositing_node_group = tree
        nb = Nodes(tree)
        output = nb.new("NodeGroupOutput")
        out_socket = output.inputs[0]
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        nb = Nodes(tree)
        output = nb.new("CompositorNodeComposite")
        out_socket = output.inputs["Image"]
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    rl = nb.new("CompositorNodeRLayers")
    rl.scene = scene
    img = rl.outputs["Image"]

    glare = nb.new("CompositorNodeGlare")
    nb.link(img, glare.inputs[0])
    if any(s.name == "Type" for s in glare.inputs):
        set_socket(glare, "Type", "Bloom")
        set_socket(glare, "Quality", "High")
        set_socket(glare, "Threshold", 1.2)
        set_socket(glare, "Smoothness", 0.4)
        set_socket(glare, "Strength", 0.55 if cfg.mood == "golden" else 0.35)
        set_socket(glare, "Size", 0.75)
        set_socket(glare, "Saturation", 1.1)
    else:
        set_first(glare, "glare_type", "BLOOM", "FOG_GLOW")
        set_first(glare, "quality", "HIGH")
        glare.threshold = 1.2
        glare.mix = 0.0
    img = nb.out(glare, "Image")

    lens = nb.new("CompositorNodeLensdist")
    nb.link(img, lens.inputs[0])
    if not set_socket(lens, "Dispersion", 0.012):
        lens.inputs[2].default_value = 0.012
    set_socket(lens, "Distortion", -0.006)
    set_socket(lens, "Fit", True)
    img = lens.outputs[0]

    grade = nb.new("CompositorNodeColorBalance")
    nb.link(img, grade.inputs[0])
    if any(s.name == "Type" for s in grade.inputs):
        set_socket(grade, "Type", "Lift/Gamma/Gain")
    else:
        set_first(grade, "correction_method", "LIFT_GAMMA_GAIN")
    lift, gamma, gain = (
        ((0.97, 1.0, 1.04, 1), (1.0, 1.0, 1.0, 1), (1.05, 1.0, 0.94, 1))
        if cfg.mood == "golden"
        else ((0.96, 1.0, 1.05, 1), (0.98, 1.0, 1.03, 1), (1.0, 1.0, 1.02, 1))
    )
    for name, value in (("Lift", lift), ("Gamma", gamma), ("Gain", gain)):
        if not set_socket(grade, name, value):
            setattr(grade, name.lower(), value[:3])
    img = grade.outputs[0]

    mask = nb.new("CompositorNodeEllipseMask")
    set_socket(mask, "Size", (0.92, 0.82))
    if not any(s.name == "Size" for s in mask.inputs):
        mask.width, mask.height = 0.92, 0.82
    blur = nb.new("CompositorNodeBlur")
    nb.link(mask.outputs[0], blur.inputs[0])
    if not set_socket(blur, "Size", (220.0, 220.0)):
        blur.size_x = blur.size_y = 220
    vignette = nb.map_range(blur.outputs[0], 0.0, 1.0, 0.72, 1.0)
    mix = nb.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY") if hasattr(scene, "compositing_node_group") \
        else nb.new("CompositorNodeMixRGB", blend_type="MULTIPLY")
    if hasattr(scene, "compositing_node_group"):
        nb.inp(mix, "Factor").default_value = 1.0
        nb.link(img, nb.inp(mix, "A"))
        nb.link(vignette, nb.inp(mix, "B"))
        nb.link(nb.out(mix, "Result"), out_socket)
    else:
        nb.link(img, mix.inputs[1])
        nb.link(vignette, mix.inputs[2])
        nb.link(mix.outputs[0], out_socket)


# =============================================================================
# Orchestration
# =============================================================================
def prepare_scene() -> tuple[bpy.types.Scene, bpy.types.Collection]:
    """Headless: reuse and clear the default scene. Interactive: build a fresh scene and switch to it,
    leaving the user's other scenes untouched (re-running replaces the previous teaser scene)."""
    old = bpy.data.scenes.get(SCENE_NAME)
    if bpy.app.background:
        scene = bpy.context.scene
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        scene.name = SCENE_NAME
    else:
        if old is not None:
            for obj in list(old.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            for coll in list(old.collection.children):
                bpy.data.collections.remove(coll)
            bpy.data.scenes.remove(old)
        scene = bpy.data.scenes.new(SCENE_NAME)
        if bpy.context.window is not None:
            bpy.context.window.scene = scene
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.lights, bpy.data.cameras,
                  bpy.data.actions, bpy.data.worlds, bpy.data.node_groups):
        for item in list(block):
            if item.users == 0:
                block.remove(item)
    coll = bpy.data.collections.new(COLLECTION_NAME)
    scene.collection.children.link(coll)
    return scene, coll


def sun_direction(mood: str, side: float) -> Vector:
    """Low golden-hour sun behind the camera side (rim + side light); higher for overcast storm."""
    elev = math.radians(6.5 if mood == "golden" else 28.0)
    az = math.radians(200.0 if side < 0 else -20.0)  # measured CCW from +X (east)
    return Vector((math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev))).normalized()


def plan_shots(tel: Telemetry, cfg: RenderConfig) -> tuple[Timeline, list[Shot], float]:
    t_first, t_last = float(tel.t[0]), float(tel.t[-1])
    td = tel.touchdown_time
    if td is None:
        td = t_last - cfg.post_roll_s
        print("[teaser] warning: episode has no touchdown; shots are anchored to the episode end")
    t0 = max(t_first, td - cfg.pre_roll_s)
    t1 = min(t_last, td + cfg.post_roll_s)
    s2 = max(t0 + 1.0, td - cfg.shot2_start_before_td_s)
    s3 = max(s2 + 1.0, td - cfg.shot3_start_before_td_s)
    tl = Timeline(t0, t1, cfg.fps)
    shots = [Shot("Shot1_LowAngleTracking", t0, s2), Shot("Shot2_ControlSurfaces", s2, s3), Shot("Shot3_TouchdownChase", s3, t1)]
    return tl, shots, td


def write_timeline_sidecar(path: Path, cfg: RenderConfig, tl: Timeline, shots: list[Shot], td: float,
                           contacts: list[float], tel: Telemetry) -> Path:
    sidecar = path.parent / (path.stem + ".timeline.json")
    doc = {
        "schema": "crosswind-teaser-timeline",
        "schema_version": "1.0.0",
        "video": str(path),
        "fps": cfg.fps,
        "width": cfg.resolution[0],
        "height": cfg.resolution[1],
        "frame_start": tl.frame_start,
        "frame_end": tl.frame_end,
        "duration_in_frames": tl.frame_end - tl.frame_start + 1,
        "sim_time_start_s": tl.t0,
        "sim_time_end_s": tl.t1,
        "touchdown_frame": float(tl.frame(td)),
        "main_gear_contact_frames": [float(tl.frame(t)) for t in contacts],
        "shots": [
            {"name": s.name, "camera": s.camera.name if s.camera else None,
             "from_frame": int(round(float(tl.frame(s.t_start)))), "to_frame": int(round(float(tl.frame(s.t_end))))}
            for s in shots
        ],
        "episode_metrics": tel.metrics,
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("render",)},
    }
    sidecar.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return sidecar


def build_scene(cfg: RenderConfig) -> tuple[bpy.types.Scene, Path]:
    tel = Telemetry.load(cfg.telemetry)
    tl, shots, td = plan_shots(tel, cfg)
    scene, coll = prepare_scene()

    mean_wind_x = float(np.mean(tel.wind[:, 0]))
    side = -1.0 if mean_wind_x <= 0.0 else 1.0  # cameras on the downwind side (wind blows towards them)
    sun_dir = sun_direction(cfg.mood, side)

    mats = Materials(cfg.mood)
    EnvironmentBuilder(coll, mats, tel, cfg, sun_dir).build(scene)
    if cfg.aircraft == "airliner":
        import airliner_model

        rig = airliner_model.AirlinerBuilder(coll, mats, tel.gear_flu["main_left"].z).build()
    else:
        rig = AircraftBuilder(coll, mats, tel).build()
    Animator(tel, tl, rig, cfg).run()
    contacts = TireSmoke(coll, mats, tel, tl, rig, cfg).build()

    CameraDirector(coll, tel, tl, rig, cfg, side, td).build(shots)
    scene.timeline_markers.clear()
    for s in shots:
        m = scene.timeline_markers.new(s.name, frame=int(round(float(tl.frame(s.t_start)))))
        m.camera = s.camera
    scene.camera = shots[0].camera
    scene.frame_set(tl.frame_start)

    output = configure_render(scene, cfg, tl)
    configure_compositor(scene, cfg)
    sidecar = write_timeline_sidecar(output, cfg, tl, shots, td, contacts, tel)
    print(f"[teaser] scene built: frames {tl.frame_start}-{tl.frame_end} @ {cfg.fps} fps, "
          f"touchdown frame {float(tl.frame(td)):.1f}, output {output}, timeline {sidecar}")
    return scene, output


def render(scene: bpy.types.Scene) -> None:
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
        bpy.ops.render.render(animation=True)
    else:
        with bpy.context.temp_override(scene=scene):
            bpy.ops.render.render(animation=True, scene=scene.name)


def parse_args(argv: Optional[Sequence[str]] = None) -> RenderConfig:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_cinematic_renderer", description=__doc__.split("\n")[0])
    p.add_argument("--telemetry", required=True, help="crosswind-landing-telemetry JSON episode")
    p.add_argument("--output", default=RenderConfig.output)
    p.add_argument("--engine", choices=("EEVEE", "CYCLES"), default="EEVEE")
    p.add_argument("--resolution", type=int, nargs=2, default=(3840, 2160), metavar=("W", "H"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--codec", choices=("qtrle", "prores", "ffv1", "png", "h264", "png-sequence"), default="qtrle")
    p.add_argument("--mood", choices=("golden", "storm"), default="golden")
    p.add_argument("--surface-gain", type=float, default=1.0, help="control-surface deflection multiplier (1 = physical)")
    p.add_argument("--surface-source", choices=("deflection", "action"), default="deflection")
    p.add_argument("--pre-roll", type=float, default=14.0, help="seconds of approach before touchdown")
    p.add_argument("--post-roll", type=float, default=3.0, help="seconds after touchdown")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--preview", action="store_true", help="720p, low samples, H.264, every 2nd frame")
    p.add_argument("--frames", type=int, nargs=2, default=None, metavar=("START", "END"),
                   help="render only this frame sub-range (the timeline sidecar still describes the full cut)")
    p.add_argument("--no-render", action="store_true", help="build the scene only")
    p.add_argument("--save-blend", default=None, help="save the built scene to this .blend")
    a = p.parse_args(list(argv))
    cfg = RenderConfig(
        telemetry=a.telemetry, output=a.output, engine=a.engine, resolution=tuple(a.resolution), fps=a.fps,
        samples=a.samples, codec=a.codec, mood=a.mood, surface_gain=a.surface_gain, surface_source=a.surface_source,
        pre_roll_s=a.pre_roll, post_roll_s=a.post_roll, seed=a.seed, preview=a.preview, render=not a.no_render,
        save_blend=a.save_blend, frame_range=tuple(a.frames) if a.frames else None,
    )
    if cfg.preview:
        cfg.apply_preview()
        if a.codec != "qtrle":  # an explicit codec choice overrides the preview default
            cfg.codec = a.codec
        if tuple(a.resolution) != (3840, 2160):
            cfg.resolution = tuple(a.resolution)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    scene, output = build_scene(cfg)
    if cfg.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(cfg.save_blend).resolve()), copy=True)
        print(f"[teaser] saved {cfg.save_blend}")
    if cfg.render:
        render(scene)
        print(f"[teaser] rendered {output}")


if __name__ == "__main__":
    main()
