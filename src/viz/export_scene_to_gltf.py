"""Package the procedural crosswind-landing scene into a single glTF binary for WebGL.

Exports the runway (asphalt + weathered ICAO markings), the approach / edge /
threshold / end lighting grid, PAPI, terrain and tree line, and the articulated
aircraft built by ``blender_cinematic_renderer.py`` into ``public/models/scene.glb``.

glTF cannot carry Blender's procedural shader graphs, so the exporter
*re-materialises* the scene for export:

* Asphalt, grass, runway paint, apron concrete (with slab joints), corrugated
  hangar cladding (rib normal map) and chain-link fencing (alpha mask) get
  seamless PBR texture sets (albedo, occlusion/roughness/metallic, tangent-space
  normal) synthesised with FFT-filtered 1/f noise, which is periodic by
  construction and therefore tiles without seams; meshes receive metric UVs.
* Static lighting is baked into vertex colours: sun shadows and ambient occlusion
  cast by the buildings onto the apron (BVH ray casts towards the same sun
  direction the WebGL runtime uses), and runway surface wear (touchdown-zone
  rubber deposits, patch repairs, lighter shoulders). glTF multiplies COLOR_0
  into the base colour.
* Airframe, glass, rubber, chrome, prop and emissive light materials become
  constant glTF metallic-roughness materials (emission uses
  ``KHR_materials_emissive_strength``).
* Curves (gear coil springs) and text (registration) are converted to meshes;
  constraint-driven parts are baked at their rest pose.
* Animation is not exported: the aircraft is reset to the origin with all
  control surfaces, wheels, propeller and nose steering at zero, because the
  web app drives every node from the telemetry. Node names are preserved
  (``Aircraft``, ``Aileron_L``, ``Aileron_R``, ``Elevator``, ``Rudder``,
  ``Hub_*``, ``Strut_*``, ``Spring_*``, ``Wheel_*``, ``Nose_Steer``, ``Propeller``).

WebGL profile (default; ``--full-scene`` disables it):

* Declutter: hangars, the MRO building, perimeter fencing, the fuel farm and the apron
  floodlights are left out, so low-altitude flight sees only the runway, its lighting,
  navaids, taxiways, apron, tower and operations building. The static PAPI lamps are
  always left out because the web runtime draws its own dynamic PAPI.
* With no buildings left to cast shadows, the apron is not light-baked; it exports as a
  planar-decimated slab with the plain concrete material.
* Static merge: world meshes sharing one constant (untextured) material are joined into
  a single ``Static_<material>`` node, so all light fixtures are one draw call, all
  runway-white lamps another, and so on.

The original materials, names, actions and transforms are restored afterwards, so
running the exporter inside an interactive session leaves the scene untouched.
A sidecar ``scene.rig.json`` records hinge axes, gear rest geometry and runway
dimensions in the glTF frame (+X east, +Y up, +Z south; aircraft nose +X, up +Y,
left wing -Z).

Usage::

    blender -b --python src/viz/export_scene_to_gltf.py -- \\
        --telemetry results/telemetry_demo/baseline_xw7.5_s1_00000.json \\
        --output src/web/public/models/scene.glb

If a ``CrosswindTeaser`` scene already exists (interactive session) it is
exported as is; otherwise it is built first from ``--telemetry``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import bpy
import numpy as np
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import airliner_model  # noqa: E402
import blender_cinematic_renderer as teaser  # noqa: E402

EXCLUDE_PREFIXES = ("Smoke_", "Cam_", "Shot", "DBG_", "Haze_Volume", "Cloud_Deck", "Sun")
EXCLUDE_SUFFIXES = ("_Lamp",)
ARTICULATED = ("Aileron_L", "Aileron_R", "Elevator", "Elevator_L", "Elevator_R", "Rudder")
SPOILERS = ("Spoiler_L", "Spoiler_R")
TILE_M = {"Runway": 6.0, "Terrain": 40.0, "Runway_Markings": 3.0, "Taxiways": 6.0, "Apron": 5.0, "Apron_Link": 5.0}
WALL_TILE_M = {"Hangar_Shells": 1.2, "MRO_Walls": 1.2, "Fence_Mesh": 0.6}
# Sun direction shared with the WebGL runtime (Blender ENU, pointing towards the sun).
SUN_DIR_ENU = Vector((-0.9, -0.33, 0.25)).normalized()
SHADOW_CASTERS = ("Hangar_", "MRO_", "Ops_", "Tower_", "Fuel_Tanks", "Floodlight_Masts")
# Background clutter dropped by the WebGL profile (heavy near the runway, little visual value).
DECLUTTER_PREFIXES = ("Hangar_", "MRO_", "Fence_", "Fuel_Tanks", "Floodlight_", "Tree_", "Ops_", "Tower_")
# Replaced at runtime by the web app's dynamic PAPI.
WEB_REPLACED = ("PAPI_Red", "PAPI_White")
# Nodes the web app addresses by name; never folded into a merged static node.
KEEP_SEPARATE = ("Mountains", "Windsock")


@dataclass
class ExportConfig:
    output: str
    telemetry: Optional[str] = None
    texture_size: int = 1024
    trees: bool = True
    tree_decimate: float = 0.4
    terrain_decimate: float = 0.5
    seed: int = 11
    full_scene: bool = False


# =============================================================================
# Seamless texture synthesis
# =============================================================================
def periodic_fbm(size: int, beta: float, rng: np.random.Generator, low_cut: float = 1.0) -> np.ndarray:
    """Tileable fractal noise in [0, 1]: white noise shaped by a 1/f^beta spectrum."""
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    k = np.sqrt(fx**2 + fy**2) * size
    amp = np.where(k >= low_cut, 1.0 / np.maximum(k, 1e-9) ** beta, 0.0)
    spec = np.fft.fft2(rng.standard_normal((size, size))) * amp
    out = np.real(np.fft.ifft2(spec))
    out -= out.min()
    return out / max(out.max(), 1e-12)


def periodic_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    size = img.shape[0]
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-2.0 * (math.pi * sigma_px) ** 2 * (fx**2 + fy**2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * g))


def normal_from_height(h: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space normal map (OpenGL +Y convention, as glTF expects), periodic gradients."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * strength
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * strength
    n = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


_TEXTURE_DIR = Path(tempfile.mkdtemp(prefix="alma_glb_textures_"))


def make_image(name: str, rgb: np.ndarray, non_color: bool) -> bpy.types.Image:
    """Encode a synthesized texture to a real image file and load it file-backed and packed.

    The glTF exporter's re-encoding path writes *black* pixels for generated in-memory images
    (verified on Blender 5.2), so textures are encoded here (JPEG, or PNG when they carry alpha)
    and exported with ``export_image_format='AUTO'``, which embeds the original file bytes.
    """
    size = rgb.shape[0]
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    has_alpha = rgb.shape[-1] == 4
    if has_alpha:
        rgba = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    else:
        rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), np.ones((size, size, 1))], axis=-1).astype(np.float32)
    # Blender image rows run bottom-to-top; flip so texture space matches numpy's top-left origin.
    rgba = np.ascontiguousarray(rgba[::-1])
    staging = bpy.data.images.new(name + "__staging", size, size, alpha=has_alpha, float_buffer=False)
    staging.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    staging.pixels.foreach_set(rgba.ravel())
    path = _TEXTURE_DIR / (name + (".png" if has_alpha else ".jpg"))
    staging.filepath_raw = str(path)
    staging.file_format = "PNG" if has_alpha else "JPEG"
    scene = bpy.context.scene
    old_quality = scene.render.image_settings.quality
    scene.render.image_settings.quality = 92
    staging.save()
    scene.render.image_settings.quality = old_quality
    bpy.data.images.remove(staging)
    img = bpy.data.images.load(str(path))
    img.name = name
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    img.pack()
    return img


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(np.maximum(lin, 0.0), 1 / 2.4) - 0.055)


def asphalt_textures(size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stain = periodic_fbm(size, 1.6, rng, low_cut=2.0)
    grain = periodic_fbm(size, 0.6, rng, low_cut=24.0)
    stones = (rng.random((size, size)) > 0.985).astype(np.float64)
    stones = np.clip(periodic_blur(stones, 0.9) * 6.0, 0.0, 1.0)
    tone = 0.55 * stain + 0.3 * grain + 0.15 * stones
    lin = 0.022 + 0.06 * tone
    albedo = srgb_encode(np.stack([lin * 0.98, lin * 0.98, lin * 1.02], axis=-1))
    rough = 0.62 + 0.25 * grain - 0.22 * np.clip((stain - 0.72) / 0.12, 0.0, 1.0)  # damp low spots
    orm = np.stack([np.full_like(rough, 1.0), np.clip(rough, 0.05, 1.0), np.zeros_like(rough)], axis=-1)
    height = 0.7 * grain + 0.6 * stones
    return albedo, orm, normal_from_height(height, 6.0)


def grass_textures(size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patches = periodic_fbm(size, 1.8, rng, low_cut=1.0)
    blades = periodic_fbm(size, 0.4, rng, low_cut=48.0)
    dry = np.array([0.105, 0.098, 0.04])
    lush = np.array([0.07, 0.092, 0.032])
    # Low in-tile contrast: strong patches repeat as a visible waffle pattern from altitude. Macro variation is per-vertex.
    mix = np.clip((patches - 0.25) / 0.5, 0.0, 1.0)[..., None]
    lin = lush * (1 - mix) + dry * mix
    lin = lin * (0.75 + 0.5 * blades[..., None])
    orm = np.stack([np.ones_like(blades), 0.9 + 0.08 * blades, np.zeros_like(blades)], axis=-1)
    return srgb_encode(lin), orm, normal_from_height(blades, 3.0)


def paint_textures(size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wear = periodic_fbm(size, 1.2, rng, low_cut=3.0)
    chips = np.clip(periodic_blur((rng.random((size, size)) > 0.97).astype(np.float64), 1.2) * 5.0, 0.0, 1.0)
    erosion = np.clip((wear - 0.55) / 0.3 + 0.6 * chips, 0.0, 1.0)
    paint = np.array([0.82, 0.8, 0.74])
    asphalt = np.array([0.05, 0.05, 0.052])
    lin = paint * (1 - erosion[..., None]) + asphalt * erosion[..., None]
    orm = np.stack([np.ones_like(wear), 0.38 + 0.35 * erosion, np.zeros_like(wear)], axis=-1)
    return srgb_encode(lin), orm, normal_from_height(chips, 2.0)


def concrete_textures(size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 5 m slab per tile: joint lines on the tile border, stains, aggregate speckle."""
    stain = periodic_fbm(size, 1.5, rng, low_cut=1.5)
    grain = periodic_fbm(size, 0.5, rng, low_cut=32.0)
    idx = np.arange(size)
    edge = np.minimum(idx, size - 1 - idx)
    joint = np.clip(1.0 - np.minimum(edge[None, :], edge[:, None]) / 2.5, 0.0, 1.0)
    lin = 0.32 * (0.82 + 0.25 * stain + 0.08 * grain) * (1.0 - 0.55 * joint)
    albedo = srgb_encode(np.stack([lin, lin * 0.99, lin * 0.96], axis=-1))
    orm = np.stack([1.0 - 0.3 * joint, 0.78 + 0.12 * grain, np.zeros_like(grain)], axis=-1)
    return albedo, orm, normal_from_height(0.4 * grain - 1.5 * joint, 4.0)


def corrugated_textures(size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trapezoidal-rib cladding: 8 ribs per tile along U, vertical weathering streaks."""
    u = np.linspace(0.0, 1.0, size, endpoint=False)
    ribs = np.clip(np.sin(2.0 * math.pi * 8.0 * u) * 1.6, -1.0, 1.0)
    height = np.tile(ribs[None, :], (size, 1))
    streak_cols = periodic_blur(rng.standard_normal((1, size)).repeat(size, axis=0), 2.0)
    streaks = np.clip(0.5 + 0.35 * streak_cols / (np.abs(streak_cols).max() + 1e-9), 0.0, 1.0)
    dirt = periodic_fbm(size, 1.4, rng, low_cut=2.0)
    lin = 0.36 * (0.78 + 0.2 * streaks + 0.12 * dirt)
    albedo = srgb_encode(np.stack([lin * 0.96, lin, lin * 1.04], axis=-1))
    orm = np.stack([np.ones_like(lin), 0.38 + 0.2 * dirt, np.full_like(lin, 0.7)], axis=-1)
    return albedo, orm, normal_from_height(height, 1.8)


def chainlink_texture(size: int) -> np.ndarray:
    """RGBA diamond mesh; alpha 1 on the wire, 0 in the openings."""
    y, x = np.mgrid[0:size, 0:size] / size
    d1 = np.abs(((x + y) * 4.0) % 1.0 - 0.5)
    d2 = np.abs(((x - y) * 4.0) % 1.0 - 0.5)
    wire = ((d1 > 0.45) | (d2 > 0.45)).astype(np.float64)
    rgb = np.full((size, size, 3), srgb_encode(np.array(0.4)))
    return np.concatenate([rgb, wire[..., None]], axis=-1)


# =============================================================================
# Export materials
# =============================================================================
def pbr_material(name: str, color, metallic: float, roughness: float, emission=None, strength: float = 0.0,
                 alpha: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = alpha
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def textured_material(name: str, albedo: bpy.types.Image, orm: bpy.types.Image, normal: bpy.types.Image,
                      normal_strength: float = 1.0, vertex_color: bool = False) -> bpy.types.Material:
    """Principled BSDF wired exactly as the glTF exporter expects (ORM via Separate Color)."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    t_alb = nt.nodes.new("ShaderNodeTexImage")
    t_alb.image = albedo
    if vertex_color:
        # glTF: base colour = texture x COLOR_0 (baked shadows / surface wear).
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Col"
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(t_alb.outputs["Color"], mix.inputs[6])
        nt.links.new(vc.outputs["Color"], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    else:
        nt.links.new(t_alb.outputs["Color"], bsdf.inputs["Base Color"])
    t_orm = nt.nodes.new("ShaderNodeTexImage")
    t_orm.image = orm
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(t_orm.outputs["Color"], sep.inputs[0])
    nt.links.new(sep.outputs[1], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs[2], bsdf.inputs["Metallic"])
    t_nrm = nt.nodes.new("ShaderNodeTexImage")
    t_nrm.image = normal
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.inputs["Strength"].default_value = normal_strength
    nt.links.new(t_nrm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs[0], bsdf.inputs["Normal"])
    return mat


def build_export_materials(cfg: ExportConfig) -> dict[str, bpy.types.Material]:
    rng = np.random.default_rng(cfg.seed)
    s = cfg.texture_size
    sets = {
        "asphalt": asphalt_textures(s, rng),
        "grass": grass_textures(s, rng),  # full resolution: the grass is seen from a few metres on short final
        "paint": paint_textures(s, rng),  # runway markings stay crisp under the wheels
        "concrete": concrete_textures(max(s // 2, 256), rng),
        "corrugated": corrugated_textures(256, rng),
    }
    mats: dict[str, bpy.types.Material] = {}
    images = {}
    for key, (alb, orm, nrm) in sets.items():
        images[key] = (
            make_image(f"GLB_{key}_albedo", alb, False),
            make_image(f"GLB_{key}_orm", orm, True),
            make_image(f"GLB_{key}_normal", nrm, True),
        )
        mats[key] = textured_material(f"GLB_{key.title()}", *images[key])
    # Variants that multiply baked per-vertex lighting / wear into the base colour.
    mats["asphalt_runway"] = textured_material("GLB_Asphalt_Runway", *images["asphalt"], vertex_color=True)
    mats["concrete_baked"] = textured_material("GLB_Concrete_Apron", *images["concrete"], vertex_color=True)
    mats["grass_varied"] = textured_material("GLB_Grass_Terrain", *images["grass"], vertex_color=True)
    # Chain-link: alpha-mask material (Greater Than on the texture alpha exports as alphaMode MASK).
    link = bpy.data.materials.new("GLB_Fence_Mesh")
    if link.node_tree is None:
        link.use_nodes = True
    nt = link.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.5
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = make_image("GLB_chainlink", chainlink_texture(128), False)
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = 0.5
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
    nt.links.new(cut.outputs[0], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    if hasattr(link, "surface_render_method"):
        link.surface_render_method = "DITHERED"
    mats["Fence_Mesh"] = link
    # Constant PBR stand-ins keyed by the cinematic material names.
    table = {
        "Airframe_Polished_Aluminium": ((0.86, 0.87, 0.89), 1.0, 0.2, None, 0.0),
        "Canopy_Glass": ((0.01, 0.015, 0.02), 0.0, 0.04, None, 0.0),
        "Tire_Rubber": ((0.018, 0.018, 0.02), 0.0, 0.62, None, 0.0),
        "Strut_Chrome": ((0.8, 0.8, 0.82), 1.0, 0.12, None, 0.0),
        "Spring_Steel": ((0.55, 0.12, 0.08), 0.6, 0.35, None, 0.0),
        "Prop_Black": ((0.02, 0.02, 0.02), 0.2, 0.4, None, 0.0),
        "Prop_Tip_Yellow": ((0.9, 0.62, 0.02), 0.0, 0.35, None, 0.0),
        "Nav_Red": ((1.0, 0.05, 0.03), 0.0, 0.2, (1.0, 0.05, 0.03), 12.0),
        "Nav_Green": ((0.05, 1.0, 0.25), 0.0, 0.2, (0.05, 1.0, 0.25), 12.0),
        "Strobe_White": ((1.0, 1.0, 1.0), 0.0, 0.2, (1.0, 1.0, 1.0), 2.0),
        "Landing_Light_Lens": ((1.0, 0.95, 0.85), 0.0, 0.1, (1.0, 0.95, 0.85), 20.0),
        "Foliage": ((0.03, 0.06, 0.025), 0.0, 0.85, None, 0.0),
        "Bark": ((0.05, 0.035, 0.025), 0.0, 0.9, None, 0.0),
        "Light_Fixture": ((0.03, 0.03, 0.03), 0.3, 0.6, None, 0.0),
        "RWY_Light_White": ((1.0, 0.86, 0.62), 0.0, 0.2, (1.0, 0.86, 0.62), 10.0),
        "RWY_Light_Yellow": ((1.0, 0.65, 0.12), 0.0, 0.2, (1.0, 0.65, 0.12), 10.0),
        "RWY_Light_Green": ((0.1, 1.0, 0.35), 0.0, 0.2, (0.1, 1.0, 0.35), 12.0),
        "RWY_Light_Red": ((1.0, 0.04, 0.02), 0.0, 0.2, (1.0, 0.04, 0.02), 12.0),
        "Hangar_Door": ((0.3, 0.34, 0.38), 0.55, 0.5, None, 0.0),
        "Painted_Wall": ((0.62, 0.6, 0.55), 0.0, 0.8, None, 0.0),
        "Roof_Dark": ((0.12, 0.12, 0.13), 0.1, 0.7, None, 0.0),
        "Window_Glow": ((0.9, 0.7, 0.45), 0.0, 0.3, (1.0, 0.78, 0.5), 3.0),
        "Tower_Glass": ((0.03, 0.06, 0.07), 0.2, 0.05, None, 0.0),
        "Galvanized_Steel": ((0.5, 0.52, 0.53), 0.8, 0.5, None, 0.0),
        "Windsock_Orange": ((1.0, 0.33, 0.04), 0.0, 0.6, (1.0, 0.33, 0.04), 0.4),
        "Taxi_Paint_Yellow": ((0.85, 0.62, 0.04), 0.0, 0.5, None, 0.0),
        "Taxi_Light_Blue": ((0.1, 0.3, 1.0), 0.0, 0.2, (0.1, 0.3, 1.0), 10.0),
        "Floodlight_Lamp": ((1.0, 0.92, 0.78), 0.0, 0.2, (1.0, 0.9, 0.75), 8.0),
        "Fuel_Tank_White": ((0.78, 0.78, 0.76), 0.2, 0.5, None, 0.0),
    }
    for src, (col, met, rough, em, strength) in table.items():
        mats[src] = pbr_material(f"GLB_{src}", col, met, rough, em, strength)
    mats["Prop_Disc"] = pbr_material("GLB_Prop_Disc", (0.05, 0.05, 0.05), 0.0, 0.4, alpha=0.08)
    mats["Runway_Asphalt"] = mats["asphalt"]
    mats["Terrain_Grass"] = mats["grass"]
    mats["Runway_Paint_Weathered"] = mats["paint"]
    mats["Concrete_Apron"] = mats["concrete"]
    mats["Corrugated_Metal"] = mats["corrugated"]
    return mats


# Per-object material overrides (objects whose export material needs baked vertex data).
OBJECT_MATERIALS = {"Runway": "asphalt_runway", "Apron": "concrete_baked", "Terrain": "grass_varied"}


def base_name(mat_name: str) -> str:
    """Strip Blender's ``.001`` duplicate suffixes."""
    head, _, tail = mat_name.rpartition(".")
    return head if tail.isdigit() and head else mat_name


# =============================================================================
# Scene preparation / restoration
# =============================================================================
def metric_uvs(obj: bpy.types.Object, tile_m: float) -> None:
    me = obj.data
    uv = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    mw = obj.matrix_world
    coords = np.array([tuple(mw @ v.co) for v in me.vertices])
    loop_v = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    uvs = coords[loop_v][:, :2] / tile_m
    uv.data.foreach_set("uv", uvs.ravel().astype(np.float32))
    me.uv_layers.active = uv


def wall_uvs(obj: bpy.types.Object, tile_m: float) -> None:
    """UVs for vertical surfaces: U along the horizontal (x + y), V up (z)."""
    me = obj.data
    uv = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    mw = obj.matrix_world
    coords = np.array([tuple(mw @ v.co) for v in me.vertices])
    loop_v = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    c = coords[loop_v]
    uvs = np.stack([(c[:, 0] + c[:, 1]) / tile_m, c[:, 2] / tile_m], axis=1)
    uv.data.foreach_set("uv", uvs.ravel().astype(np.float32))
    me.uv_layers.active = uv


def set_vertex_colors(obj: bpy.types.Object, values: np.ndarray) -> None:
    """Write per-vertex grey/RGB values (linear, 0..1) into the face-corner attribute 'Col'."""
    me = obj.data
    if "Col" in me.color_attributes:
        me.color_attributes.remove(me.color_attributes["Col"])
    attr = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    loop_v = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    vals = values if values.ndim == 2 else np.repeat(values[:, None], 3, axis=1)
    rgba = np.concatenate([vals[loop_v], np.ones((len(loop_v), 1))], axis=1).astype(np.float32)
    attr.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = attr


def shadow_bvh(scene) -> Optional[BVHTree]:
    verts, polys = [], []
    for obj in scene.objects:
        if obj.type != "MESH" or not obj.name.startswith(SHADOW_CASTERS):
            continue
        mw = obj.matrix_world
        base = len(verts)
        verts.extend(mw @ v.co for v in obj.data.vertices)
        polys.extend([base + i for i in p.vertices] for p in obj.data.polygons)
    return BVHTree.FromPolygons(verts, polys) if polys else None


def bake_apron_lighting(obj: bpy.types.Object, bvh: Optional[BVHTree]) -> None:
    """Soft sun shadows (5 jittered rays) and short-range ambient occlusion (8 rays) per vertex."""
    me = obj.data
    mw = obj.matrix_world
    rng = np.random.default_rng(5)
    up = Vector((0.0, 0.0, 1.0))
    t1 = SUN_DIR_ENU.cross(up).normalized()
    t2 = SUN_DIR_ENU.cross(t1).normalized()
    sun_dirs = [SUN_DIR_ENU] + [(SUN_DIR_ENU + 0.02 * (a * t1 + b * t2)).normalized() for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    ao_dirs = []
    for k in range(8):
        az = 2 * math.pi * (k + rng.random() * 0.5) / 8
        el = math.radians(20 + 40 * rng.random())
        ao_dirs.append(Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))))
    vals = np.ones(len(me.vertices))
    if bvh is not None:
        for i, v in enumerate(me.vertices):
            o = mw @ v.co + Vector((0.0, 0.0, 0.05))
            shadow = sum(1 for d in sun_dirs if bvh.ray_cast(o, d, 400.0)[0] is not None) / len(sun_dirs)
            occl = sum(1 for d in ao_dirs if bvh.ray_cast(o, d, 30.0)[0] is not None) / len(ao_dirs)
            vals[i] = (1.0 - 0.62 * shadow) * (1.0 - 0.35 * occl)
    set_vertex_colors(obj, vals)


def paint_runway_wear(obj: bpy.types.Object, half_width: float, seed: int) -> None:
    """Rubber deposits in the touchdown zones, patch repairs, centreline staining, lighter shoulders."""
    me = obj.data
    mw = obj.matrix_world
    xy = np.array([tuple(mw @ v.co)[:2] for v in me.vertices])
    x, y = xy[:, 0], xy[:, 1]
    rng = np.random.default_rng(seed)
    length = float(y.max() - 60.0)
    tdz = np.clip((y - 90.0) / 80.0, 0, 1) * np.clip((900.0 - y) / 250.0, 0, 1)
    tdz_far = np.clip((y - (length - 900.0)) / 250.0, 0, 1) * np.clip((length - 90.0 - y) / 80.0, 0, 1)
    tracks = np.exp(-(((np.abs(x) - 1.4) / 1.6) ** 2))
    rubber = np.clip(tracks * (tdz + 0.6 * tdz_far), 0.0, 1.0)
    stain = 0.06 * np.exp(-((x / 3.0) ** 2))
    val = 0.86 - 0.5 * rubber - stain
    for _ in range(9):  # rectangular patch repairs of newer (darker) or older (lighter) asphalt
        cx, cy = rng.uniform(-half_width + 3, half_width - 3), rng.uniform(80.0, length - 80.0)
        w, h = rng.uniform(3.0, 9.0), rng.uniform(8.0, 30.0)
        inside = (np.abs(x - cx) < w / 2) & (np.abs(y - cy) < h / 2)
        val = np.where(inside, val * rng.choice([0.72, 1.12]), val)
    val = np.where(np.abs(x) > half_width, np.minimum(val * 1.18, 1.0), val)
    set_vertex_colors(obj, np.clip(val, 0.05, 1.0))


def paint_terrain_variation(obj: bpy.types.Object) -> None:
    """Landscape-scale tint (dry / lush fields, darker hollows) that breaks up texture tiling."""
    from mathutils import noise as mnoise

    me = obj.data
    mw = obj.matrix_world
    cols = np.empty((len(me.vertices), 3))
    for i, v in enumerate(me.vertices):
        w = mw @ v.co
        field = 0.5 + 0.5 * mnoise.noise(Vector((w.x / 420.0, w.y / 420.0, 0.7)))
        fine = 0.5 + 0.5 * mnoise.noise(Vector((w.x / 90.0, w.y / 90.0, 2.3)))
        dry = np.array([1.0, 0.93, 0.78])
        lush = np.array([0.72, 0.86, 0.68])
        c = lush + (dry - lush) * field
        cols[i] = c * (0.78 + 0.22 * fine)
    set_vertex_colors(obj, np.clip(cols, 0.0, 1.0))


def subtree(root: bpy.types.Object) -> list[bpy.types.Object]:
    out = [root]
    for child in root.children:
        out.extend(subtree(child))
    return out


def is_excluded(obj: bpy.types.Object) -> bool:
    return obj.name.startswith(EXCLUDE_PREFIXES) or obj.name.endswith(EXCLUDE_SUFFIXES) or obj.type in {"CAMERA", "LIGHT"}


class ExportSession:
    """Applies all temporary export-only modifications and undoes them on exit."""

    def __init__(self, scene: bpy.types.Scene, cfg: ExportConfig) -> None:
        self.scene = scene
        self.cfg = cfg
        self.saved_actions: list[tuple] = []
        self.saved_materials: list[tuple] = []
        self.saved_transforms: list[tuple] = []
        self.renamed: list[tuple] = []
        self.temp_objects: list[bpy.types.Object] = []
        self.temp_modifiers: list[tuple] = []
        self.temp_meshes: list[str] = []
        self.hidden: list[bpy.types.Object] = []
        self.export_objects: list[bpy.types.Object] = []
        self.rig: dict = {}

    # ------------------------------------------------------------------- enter
    def __enter__(self) -> "ExportSession":
        scene = self.scene
        scene.frame_set(scene.frame_start)
        aircraft = bpy.data.objects["Aircraft"]
        air_objs = [o for o in subtree(aircraft) if not is_excluded(o) and not o.name.startswith("Shot2_")]
        world_objs = [o for o in scene.objects if o.parent is None and o.type == "MESH" and not is_excluded(o)]
        if not self.cfg.trees:
            world_objs = [o for o in world_objs if not o.name.startswith("Tree_")]
        world_objs = [o for o in world_objs if not o.name.startswith(WEB_REPLACED)]
        if not self.cfg.full_scene:
            world_objs = [o for o in world_objs if not o.name.startswith(DECLUTTER_PREFIXES)]

        # Rest pose: freeze the frame-1 state (gear uncompressed), then zero all articulations.
        for obj in air_objs + [aircraft]:
            ad = obj.animation_data
            if ad is not None and ad.action is not None:
                self.saved_transforms.append((obj, obj.matrix_basis.copy(), obj.rotation_mode))
                self.saved_actions.append((ad, ad.action, getattr(ad, "action_slot", None)))
                ad.action = None
        aircraft.location = (0.0, 0.0, 0.0)
        aircraft.rotation_mode = "QUATERNION"
        aircraft.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        for name in ARTICULATED + SPOILERS:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            aa = obj.rotation_axis_angle
            obj.rotation_axis_angle = (0.0, aa[1], aa[2], aa[3])
        for obj in air_objs:
            if obj.name.startswith(("Wheel_", "Fan_")) or obj.name == "Propeller":
                obj.rotation_euler = (0.0, 0.0, 0.0)
            if obj.name == "Nose_Steer":
                obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

        # Curves / text -> meshes (glTF has no curve primitive); keep the node names.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        converted = []
        for obj in list(air_objs):
            if obj.type in {"CURVE", "FONT"}:
                eval_obj = obj.evaluated_get(depsgraph)
                me = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph)
                original_name = obj.name
                world = obj.matrix_world.copy()
                obj.name = original_name + "__src"
                self.renamed.append((obj, original_name))
                tmp = bpy.data.objects.new(original_name, me)
                self.scene.collection.objects.link(tmp)
                tmp.parent = obj.parent
                tmp.matrix_world = world
                self.temp_objects.append(tmp)
                converted.append((obj, tmp))
        for src, tmp in converted:
            air_objs[air_objs.index(src)] = tmp

        # Lighter terrain / foliage via export-only decimation.
        for obj in world_objs:
            ratio = self.cfg.tree_decimate if obj.name.startswith("Tree_") else (self.cfg.terrain_decimate if obj.name == "Terrain" else None)
            if ratio is not None and ratio < 1.0:
                mod = obj.modifiers.new("GLB_Decimate", "DECIMATE")
                mod.ratio = ratio
                self.temp_modifiers.append((obj, mod.name))
            if obj.name == "Apron" and not self.cfg.full_scene:
                # The dense grid only existed to carry the baked building shadows.
                mod = obj.modifiers.new("GLB_Planar", "DECIMATE")
                mod.decimate_type = "DISSOLVE"
                mod.angle_limit = math.radians(1.0)
                self.temp_modifiers.append((obj, mod.name))

        # Materials, UVs and baked vertex data.
        mats = build_export_materials(self.cfg)
        self.export_materials = mats
        bvh = shadow_bvh(self.scene)
        half_width = float(self.scene.get("alma_runway_half_width", 15.0))
        for obj in world_objs + air_objs:
            if obj.type != "MESH":
                continue
            if obj.name in TILE_M:
                metric_uvs(obj, TILE_M[obj.name])
            if obj.name in WALL_TILE_M:
                wall_uvs(obj, WALL_TILE_M[obj.name])
            if obj.name == "Apron" and self.cfg.full_scene:
                bake_apron_lighting(obj, bvh)
            if obj.name == "Runway":
                paint_runway_wear(obj, half_width, self.cfg.seed)
            if obj.name == "Terrain":
                paint_terrain_variation(obj)
            for slot in obj.material_slots:
                if slot.material is None:
                    continue
                key = OBJECT_MATERIALS.get(obj.name, base_name(slot.material.name))
                if obj.name == "Apron" and not self.cfg.full_scene:
                    key = "concrete"
                if key in mats:
                    self.saved_materials.append((slot, slot.material, slot.link))
                    slot.material = mats[key]

        if not self.cfg.full_scene:
            world_objs = self._merge_static(world_objs)
        self.export_objects = world_objs + air_objs
        self.rig = self._rig_manifest(air_objs)
        return self

    # ------------------------------------------------------------------ merge
    def _merge_static(self, world_objs: list) -> list:
        """Join world meshes that share one constant material into one temporary node each."""
        def constant(mat) -> bool:
            return mat.node_tree is None or not any(n.type == "TEX_IMAGE" for n in mat.node_tree.nodes)

        groups: dict = {}
        out = []
        for obj in world_objs:
            mats = {s.material for s in obj.material_slots if s.material is not None}
            mat = next(iter(mats)) if len(mats) == 1 else None
            if mat is None or obj.children or obj.name.startswith(KEEP_SEPARATE) or not constant(mat):
                out.append(obj)
            else:
                groups.setdefault(mat, []).append(obj)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        view_layer = self.scene.view_layers[0]
        for mat, objs in groups.items():
            if len(objs) == 1:
                out.extend(objs)
                continue
            parts = []
            for obj in objs:
                me = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
                me.transform(obj.matrix_world)
                me.materials.clear()
                me.materials.append(mat)
                me.polygons.foreach_set("material_index", np.zeros(len(me.polygons), dtype=np.int32))
                part = bpy.data.objects.new(obj.name + "__merge", me)
                self.scene.collection.objects.link(part)
                self.temp_meshes.append(me.name)
                parts.append(part)
            with bpy.context.temp_override(scene=self.scene, view_layer=view_layer, active_object=parts[0], object=parts[0],
                                           selected_objects=parts, selected_editable_objects=parts):
                bpy.ops.object.join()
            merged = parts[0]
            merged.name = "Static_" + base_name(mat.name).removeprefix("GLB_")
            self.temp_objects.append(merged)
            out.append(merged)
            print(f"[glb] merged {len(objs)} meshes -> {merged.name}: {', '.join(o.name for o in objs)}")
        return out

    # -------------------------------------------------------------------- rig
    @staticmethod
    def _to_gltf(v) -> list[float]:
        """Blender Z-up -> glTF Y-up: (x, y, z) -> (x, z, -y)."""
        return [round(float(v[0]), 5), round(float(v[2]), 5), round(float(-v[1]), 5)]

    def _rig_manifest(self, air_objs) -> dict:
        rig: dict = {"surfaces": {}, "gear": {}, "propeller": None}
        signs = {"Aileron_L": -1.0, "Aileron_R": 1.0, "Elevator": -1.0, "Elevator_L": -1.0, "Elevator_R": -1.0, "Rudder": -1.0}
        channel = {"Aileron_L": "aileron_deg", "Aileron_R": "aileron_deg", "Elevator": "elevator_deg", "Elevator_L": "elevator_deg",
                   "Elevator_R": "elevator_deg", "Rudder": "rudder_deg"}
        for name in ARTICULATED:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            aa = obj.rotation_axis_angle
            rig["surfaces"][name] = {
                "hinge_axis_local": self._to_gltf((aa[1], aa[2], aa[3])),
                "sign": signs[name],
                "channel": channel[name],
                "pivot": self._to_gltf(obj.location),
            }
        for obj in air_objs:
            if obj.name.startswith("Hub_"):
                gear = obj.name[len("Hub_"):]
                strut = bpy.data.objects.get(f"Strut_{gear}")
                wheel = bpy.data.objects.get(f"Wheel_{gear}")
                attach = bpy.data.objects.get(f"Attach_{gear}")
                rest = (obj.location - attach.location).length if attach else None
                rig["gear"][gear] = {
                    "hub_rest_local": self._to_gltf(obj.location),
                    "attach_local": self._to_gltf(attach.location) if attach else None,
                    "strut_rest_length_m": round(rest, 5) if rest else None,
                    "strut_node": strut.name if strut else None,
                    "spring_node": f"Spring_{gear}",
                    "wheel_node": wheel.name if wheel else None,
                    "wheel_radius_m": obj.get("wheel_radius", 0.20 if gear == "nose" else 0.24),
                    "contact_local": self._to_gltf(obj.matrix_local.translation - Vector((0, 0, obj.get("wheel_radius", 0.20 if gear == "nose" else 0.24)))),
                }
        rig["spoilers"] = {}
        for name in SPOILERS:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                aa = obj.rotation_axis_angle
                rig["spoilers"][name] = {"hinge_axis_local": self._to_gltf((aa[1], aa[2], aa[3]))}
        rig["fans"] = sorted(o.name for o in air_objs if o.name in ("Fan_L", "Fan_R"))
        if rig["fans"]:
            contact_z = bpy.data.objects["Hub_main_left"].location.z - bpy.data.objects["Hub_main_left"]["wheel_radius"]
            rig["flow"] = airliner_model.compute_streamlines(airliner_model.axis_height(contact_z))
        lighting = json.loads(self.scene.get("alma_lighting", "{}"))
        if lighting:
            lighting["approach_flashers"] = [self._to_gltf(p) for p in lighting.get("approach_flashers", [])]
            for unit in lighting.get("papi", []):
                unit["position"] = self._to_gltf(unit["position"])
        lighting["sun_direction"] = self._to_gltf(SUN_DIR_ENU)
        rig["lighting"] = lighting
        rig["nose_steer"] = {"node": "Nose_Steer", "axis_local": [0.0, 1.0, 0.0], "max_deg": 10.0}
        return rig

    # -------------------------------------------------------------------- exit
    def __exit__(self, *exc) -> None:
        for slot, mat, link in self.saved_materials:
            slot.material = mat
        for obj, mod_name in self.temp_modifiers:
            mod = obj.modifiers.get(mod_name)
            if mod is not None:
                obj.modifiers.remove(mod)
        for tmp in self.temp_objects:
            me = tmp.data
            bpy.data.objects.remove(tmp, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        for name in self.temp_meshes:  # join() frees the merged-in meshes itself; drop any leftovers
            me = bpy.data.meshes.get(name)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        for obj, name in self.renamed:
            obj.name = name
        for ad, action, slot in self.saved_actions:
            ad.action = action
            if slot is not None and hasattr(ad, "action_slot"):
                try:
                    ad.action_slot = slot
                except (TypeError, RuntimeError):
                    pass
        unique = {m.name: m for m in getattr(self, "export_materials", {}).values()}
        for mat in unique.values():
            if mat.users == 0:
                bpy.data.materials.remove(mat)
        for img in list(bpy.data.images):
            if img.name.startswith("GLB_") and img.users == 0:
                bpy.data.images.remove(img)
        self.scene.frame_set(self.scene.frame_current)


# =============================================================================
# Export
# =============================================================================
def gltf_export(filepath: Path, scene: bpy.types.Scene, objects: Sequence[bpy.types.Object]) -> None:
    props = {p.identifier for p in bpy.ops.export_scene.gltf.get_rna_type().properties}
    wanted = {
        "filepath": str(filepath),
        "export_format": "GLB",
        "use_selection": True,
        "export_apply": True,
        "export_yup": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_tangents": False,
        "export_materials": "EXPORT",
        "export_image_format": "AUTO",  # embed the pre-encoded JPEG/PNG bytes as they are
        "export_cameras": False,
        "export_lights": False,
        "export_extras": True,
        "export_animations": False,
        "export_skins": False,
        "export_morph": False,
        "export_draco_mesh_compression_enable": False,
        "export_vertex_color": "MATERIAL",
    }
    kwargs = {k: v for k, v in wanted.items() if k in props}
    view_layer = scene.view_layers[0]
    for obj in scene.objects:
        obj.select_set(False, view_layer=view_layer)
    for obj in objects:
        obj.hide_set(False, view_layer=view_layer)
        obj.select_set(True, view_layer=view_layer)
    view_layer.objects.active = objects[0]
    override = {"scene": scene, "view_layer": view_layer}
    if bpy.context.window is not None:
        override["window"] = bpy.context.window
    with bpy.context.temp_override(**override):
        bpy.ops.export_scene.gltf(**kwargs)


def export_scene(cfg: ExportConfig) -> Path:
    scene = bpy.data.scenes.get(teaser.SCENE_NAME)
    if scene is None:
        if not cfg.telemetry:
            raise SystemExit(f"no '{teaser.SCENE_NAME}' scene in this session; pass --telemetry to build it")
        rcfg = teaser.RenderConfig(telemetry=cfg.telemetry, render=False)
        scene, _ = teaser.build_scene(rcfg)
    out = Path(cfg.output).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ExportSession(scene, cfg) as session:
        gltf_export(out, scene, session.export_objects)
        rw = scene.objects.get("Runway")
        rig = dict(session.rig)
        rig.update({
            "schema": "alma-autoland-rig",
            "schema_version": "1.0.0",
            "frame": "glTF Y-up: +X east, +Y up, +Z south; aircraft nose +X, up +Y, left wing -Z",
            "glb": out.name,
            "nodes": sorted(o.name for o in session.export_objects),
            "runway_bbox_m": [list(map(float, rw.dimensions))] if rw else None,
        })
        out.with_suffix(".rig.json").write_text(json.dumps(rig, indent=2), encoding="utf-8")
    print(f"[glb] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.rig.json').name}")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> ExportConfig:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="export_scene_to_gltf", description=__doc__.split("\n")[0])
    p.add_argument("--output", default="src/web/public/models/scene.glb")
    p.add_argument("--telemetry", default=None, help="episode used to build the scene when it does not exist yet")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--no-trees", action="store_true")
    p.add_argument("--tree-decimate", type=float, default=0.4)
    p.add_argument("--terrain-decimate", type=float, default=0.5)
    p.add_argument("--full-scene", action="store_true",
                   help="export every building and fence, light-baked apron, no static merging (pre-WebGL-profile output)")
    a = p.parse_args(list(argv))
    return ExportConfig(output=a.output, telemetry=a.telemetry, texture_size=a.texture_size, trees=not a.no_trees,
                        tree_decimate=a.tree_decimate, terrain_decimate=a.terrain_decimate, full_scene=a.full_scene)


if __name__ == "__main__":
    export_scene(parse_args())
