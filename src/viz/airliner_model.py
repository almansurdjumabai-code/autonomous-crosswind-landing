"""Procedural narrow-body airliner (737-800 proportions) for the WebGL scene and the teaser.

Fictional "ALMA AIR" livery. Everything is generated: lofted geometry, painted 4K PBR texture sets
(windows, doors, panel lines, weathering; albedo + ORM + normal) and aerodynamic streamlines from a
potential-flow model (Rankine fuselage + elliptic lifting line). Node names match the rig the
exporter and the web app already address (``Aircraft``, ``Aileron_*``, ``Rudder``, ``Hub_*``,
``Strut_*``, ``Wheel_*``, ``Nose_Steer``), plus ``Elevator_L/R``, ``Spoiler_L/R`` and ``Fan_L/R``.

Model axes: +X nose, +Y left, +Z up, origin at the telemetry reference point. The simulated contact
height is kept, so the wheels meet the runway exactly where the flight model says; the main gear sits
at the simulated main-gear station and the airframe is built around it at true scale.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import blender_cinematic_renderer as teaser

LENGTH = 39.5
RADIUS = 1.88
X_NOSE = 17.6
SEMI_SPAN = 17.9
MAIN_RADIUS = 0.56
NOSE_RADIUS = 0.39
MAIN_Y = 2.87
WHEELBASE = 15.6
ENGINE_Y = 4.84

# s = distance from the nose tip [m], half width, top and bottom relative to the fuselage axis.
SECTIONS = np.array([
    (0.00, 0.02, -0.34, -0.38), (0.12, 0.40, -0.04, -0.68), (0.45, 0.78, 0.22, -1.00),
    (1.00, 1.12, 0.52, -1.30), (1.80, 1.42, 0.90, -1.58), (2.80, 1.66, 1.32, -1.78),
    (3.80, 1.80, 1.64, -1.90), (5.00, 1.87, 1.90, -1.98), (7.00, 1.88, 2.00, -2.00),
    (27.0, 1.88, 2.00, -2.00), (29.0, 1.85, 2.00, -1.72), (31.0, 1.70, 1.95, -1.26),
    (33.0, 1.45, 1.86, -0.72), (35.0, 1.15, 1.72, -0.18), (37.0, 0.80, 1.55, 0.38),
    (38.6, 0.46, 1.38, 0.82), (39.3, 0.22, 1.26, 1.02), (39.5, 0.04, 1.18, 1.10),
])
EXPONENT = 2.15

_TEX_DIR = Path(tempfile.mkdtemp(prefix="alma_airliner_textures_"))


# =============================================================================
# Fuselage section lookup
# =============================================================================
def _smoothed_sections(count: int = 800) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, LENGTH, count)
    table = np.stack([np.interp(s, SECTIONS[:, 0], SECTIONS[:, k]) for k in (1, 2, 3)], axis=1)
    kernel = np.exp(-0.5 * (np.arange(-14, 15) / 5.0) ** 2)
    kernel /= kernel.sum()
    pad = np.pad(table, ((14, 14), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, k], kernel, mode="valid") for k in range(3)], axis=1)
    # The tips must close: blend back to the exact table over the first and last metre.
    w = np.clip(np.minimum(s, LENGTH - s) / 1.0, 0.0, 1.0)[:, None]
    return s, table * (1 - w) + smooth * w


_S, _SEC = _smoothed_sections()


def section(s):
    return (np.interp(s, _S, _SEC[:, 0]), np.interp(s, _S, _SEC[:, 1]), np.interp(s, _S, _SEC[:, 2]))


def section_point(s, theta):
    """(y, z) relative to the fuselage axis; theta = 0 on top, pi/2 on the left (+Y)."""
    hw, top, bot = section(s)
    c, sn = np.cos(theta), np.sin(theta)
    e = 2.0 / EXPONENT
    y = hw * np.sign(sn) * np.abs(sn) ** e
    z = np.where(c >= 0, top, -bot) * np.sign(c) * np.abs(c) ** e
    return y, z


def surface_y(s, z):
    """Half width of the fuselage at station s and height z (for laying decals on the skin)."""
    hw, top, bot = section(s)
    h = top if z >= 0 else -bot
    return hw * max(1.0 - abs(z / max(h, 1e-3)) ** EXPONENT, 0.0) ** (1.0 / EXPONENT)


# =============================================================================
# Texture helpers
# =============================================================================
def fbm(h: int, w: int, beta: float, rng, low_cut: float = 2.0) -> np.ndarray:
    fy, fx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
    k = np.sqrt(fx**2 + fy**2)
    spec = np.fft.fft2(rng.standard_normal((h, w))) * np.where(k >= low_cut, 1.0 / np.maximum(k, 1e-6) ** beta, 0.0)
    out = np.real(np.fft.ifft2(spec))
    out -= out.min()
    return out / max(out.max(), 1e-9)


def srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def normal_map(h: np.ndarray, strength: float) -> np.ndarray:
    # Rows run top-down while V runs up, so dh/dv = -dh/drow.
    du = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * strength
    drow = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * strength
    n = np.stack([-du, drow, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def to_image(name: str, rgb: np.ndarray, non_color: bool) -> bpy.types.Image:
    """Encode to JPEG and load file-backed (the glTF exporter blackens generated in-memory images)."""
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    h, w = rgb.shape[:2]
    rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), np.ones((h, w, 1))], axis=-1).astype(np.float32)
    staging = bpy.data.images.new(name + "__staging", w, h, alpha=False, float_buffer=False)
    staging.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    staging.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).ravel())
    path = _TEX_DIR / f"{name}.jpg"
    staging.filepath_raw = str(path)
    staging.file_format = "JPEG"
    scene = bpy.context.scene
    old_quality = scene.render.image_settings.quality
    scene.render.image_settings.quality = 90
    staging.save()
    scene.render.image_settings.quality = old_quality
    bpy.data.images.remove(staging)
    img = bpy.data.images.load(str(path))
    img.name = name
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    img.pack()
    return img


def textured(name: str, albedo, orm, normal, coat: float = 0.0, normal_strength: float = 1.0) -> bpy.types.Material:
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
    if coat > 0.0 and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.08
    return mat


def constant(name: str, color, metallic: float, roughness: float, coat: float = 0.0) -> bpy.types.Material:
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
    if coat > 0.0 and "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Coat Roughness"].default_value = 0.08
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def box_sdf(px, py, cx, cy, hx, hy, r):
    """Signed distance to a rounded rectangle (negative inside)."""
    qx = np.abs(px - cx) - hx + r
    qy = np.abs(py - cy) - hy + r
    return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - r


def fill(d, aa):
    return np.clip(0.5 - d / aa, 0.0, 1.0)


NAVY = np.array([0.012, 0.035, 0.11])
TEAL = np.array([0.0, 0.32, 0.42])
GOLD = np.array([0.78, 0.46, 0.08])
WHITE = np.array([0.82, 0.83, 0.84])


def paint_fuselage(w: int = 4096, h: int = 1024):
    """Painted in airframe coordinates (station, height, side), so nothing stretches on the skin."""
    rng = np.random.default_rng(3)
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h  # first row = top of the image = V 1
    U, V = np.meshgrid(u, v)
    S = U * LENGTH
    theta = (V - 0.5) * 2.0 * math.pi
    Y, Z = section_point(S, theta)
    aY = np.abs(Y)
    aa_s, aa_z = LENGTH / w * 1.5, 12.0 / h * 1.5

    weather = fbm(h, w // 2, 1.5, rng)
    weather = np.repeat(weather, 2, axis=1)
    speck = fbm(h, w, 0.5, rng, low_cut=64.0)

    # Livery: white crown, navy belly with a rising sweep towards the tail, teal and gold cheat lines.
    sweep = -0.62 + 0.9 * np.clip((S - 24.0) / 15.5, 0.0, 1.0) ** 1.6
    belly = fill(Z - sweep, aa_z)
    cheat_t = fill(np.abs(Z - (sweep + 0.13)) - 0.055, aa_z)
    cheat_g = fill(np.abs(Z - (sweep + 0.25)) - 0.022, aa_z)
    alb = np.tile(WHITE, (h, w, 1))
    for mask, col in ((belly, NAVY), (cheat_t, TEAL), (cheat_g, GOLD)):
        alb = alb * (1 - mask[..., None]) + col * mask[..., None]
    alb *= (0.965 + 0.05 * weather)[..., None]

    rough = 0.3 + 0.1 * weather + 0.04 * speck
    metal = np.zeros_like(rough)
    height = np.zeros_like(rough)
    occ = np.ones_like(rough)

    # Panel lines: frames every 0.508 m, stringer lap joints, a heavier production break at the wing.
    frames = np.abs(((S + 0.254) % 0.508) - 0.254)
    laps = np.abs(((Z + 0.32) % 0.64) - 0.32)
    fwd_of_cone = (S > 4.0) & (S < 37.5)
    frame_line = np.where(fwd_of_cone, fill(frames - 0.006, aa_s), 0.0)
    lap_line = np.where(fwd_of_cone, fill(laps - 0.007, aa_z), 0.0)
    height -= 0.35 * frame_line + 0.55 * lap_line
    alb *= (1.0 - 0.16 * lap_line - 0.05 * frame_line)[..., None]
    for st in (4.2, 12.4, 21.6, 28.0, 33.6):
        line = fill(np.abs(S - st) - 0.012, aa_s)
        height -= 0.9 * line
        alb *= (1.0 - 0.35 * line)[..., None]

    # Doors and exits: cut lines, frame shadow, handle; they break the window row.
    door_sites = [(6.45, 0.43, 0.15, 0.96), (33.0, 0.40, 0.25, 0.86), (15.65, 0.26, 0.42, 0.50), (16.65, 0.26, 0.42, 0.50)]
    window_gap = np.zeros_like(S, dtype=bool)
    for sc, hs, zc, hz in door_sites:
        d = box_sdf(S, Z, sc, zc, hs, hz, 0.12)
        outline = fill(np.abs(d) - 0.012, aa_s)
        height -= 1.2 * outline
        alb *= (1.0 - 0.6 * outline)[..., None]
        handle = fill(box_sdf(S, Z, sc + hs * 0.55, zc + 0.05, 0.05, 0.03, 0.01), aa_s)
        alb *= (1.0 - 0.7 * handle)[..., None]
        window_gap |= np.abs(S - sc) < hs + 0.14
    for sc, hs, zc, hz in ((10.5, 0.62, -1.05, 0.42), (27.6, 0.62, -1.0, 0.42)):  # cargo doors, lower lobe
        outline = fill(np.abs(box_sdf(S, Z, sc, zc, hs, hz, 0.08)) - 0.012, aa_s)
        height -= outline
        alb *= (1.0 - 0.5 * outline)[..., None]

    # Cabin windows.
    pitch = 0.508
    win_s = np.floor((S - 7.6) / pitch + 0.5) * pitch + 7.6
    wd = box_sdf(S, Z, win_s, 0.56, 0.115, 0.165, 0.09)
    in_row = (S > 7.3) & (S < 31.8) & ~window_gap & (aY > 0.5)
    glass = np.where(in_row, fill(wd, aa_s), 0.0)
    gasket = np.where(in_row, fill(np.abs(wd) - 0.018, aa_s), 0.0)
    alb = alb * (1 - gasket[..., None]) + np.array([0.28, 0.29, 0.3]) * gasket[..., None]
    alb = alb * (1 - glass[..., None]) + np.array([0.006, 0.009, 0.013]) * glass[..., None]
    rough = rough * (1 - glass) + 0.05 * glass
    metal = metal * (1 - glass) + 0.5 * glass
    height += 0.5 * gasket - 0.4 * glass

    # Cockpit glazing: a tapered band split by the centre post and the side-pane pillars.
    z_low = 0.52 + 0.20 * np.clip((S - 2.2) / 2.0, 0.0, 1.0)
    z_high = 1.02 + 0.36 * np.clip((S - 2.2) / 2.0, 0.0, 1.0)
    _, sec_top, _ = section(S)
    z_high = np.minimum(z_high, sec_top * 0.74)
    band = (S > 2.15) & (S < 4.75) & (Z > z_low) & (Z < z_high)
    pillar = (np.abs(S - 3.10) < 0.045) | (np.abs(S - 3.92) < 0.04) | (aY < 0.05)
    cockpit = (band & ~pillar).astype(np.float64)
    bezel = (band & pillar).astype(np.float64)
    alb = alb * (1 - cockpit[..., None]) + np.array([0.004, 0.007, 0.012]) * cockpit[..., None]
    alb = alb * (1 - bezel[..., None]) + np.array([0.02, 0.022, 0.025]) * bezel[..., None]
    rough = rough * (1 - cockpit) + 0.03 * cockpit
    metal = metal * (1 - cockpit) + 0.65 * cockpit

    # Radome seam, soot behind the APU, staining under the windows and behind the wing root.
    seam = fill(np.abs(S - 1.35) - 0.012, aa_s)
    height -= seam
    alb *= (1.0 - 0.3 * seam)[..., None]
    soot = np.clip((S - 36.6) / 2.9, 0.0, 1.0) ** 1.5
    alb *= (1.0 - 0.62 * soot * (0.6 + 0.4 * weather))[..., None]
    streak = np.clip(1.0 - np.abs(Z + 0.2) / 1.0, 0.0, 1.0) * np.clip((S - 15.0) / 8.0, 0.0, 1.0) * np.clip((31.0 - S) / 6.0, 0.0, 1.0)
    alb *= (1.0 - 0.07 * streak * weather)[..., None]

    orm = np.stack([occ, np.clip(rough, 0.03, 1.0), metal], axis=-1)
    return srgb(alb), orm, normal_map(height, 3.0)


def paint_wing(w: int = 2048, h: int = 1024):
    """Skin atlas: U along the span, V around the section (0..0.5 upper, 0.5..1 lower)."""
    rng = np.random.default_rng(4)
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v)
    weather = fbm(h, w, 1.6, rng)
    fine = fbm(h, w, 0.6, rng, low_cut=48.0)
    upper = V < 0.5
    base = np.where(upper[..., None], np.array([0.42, 0.44, 0.47]), np.array([0.5, 0.51, 0.53]))
    alb = base * (0.86 + 0.26 * weather + 0.06 * fine)[..., None]
    rough = 0.42 + 0.2 * weather
    height = 0.05 * fine
    ribs = np.abs(((U * 17.0 + 0.25) % 0.5) - 0.25) / 17.0
    spars = np.abs(((V * 12.0 + 0.5) % 1.0) - 0.5) / 12.0
    rib_line = fill(ribs - 0.00035, 0.0005)
    spar_line = fill(spars - 0.0009, 0.001)
    height -= 0.6 * rib_line + 0.9 * spar_line
    alb *= (1.0 - 0.14 * rib_line - 0.2 * spar_line)[..., None]
    walkway = upper & (U < 0.3) & (np.abs(V - 0.24) < 0.03)
    alb = np.where(walkway[..., None], alb * 0.55, alb)
    orm = np.stack([np.ones_like(rough), np.clip(rough, 0.05, 1.0), np.full_like(rough, 0.25)], axis=-1)
    return srgb(alb), orm, normal_map(height, 2.5)


def paint_fin(w: int = 1024, h: int = 1024):
    """Planar over the fin: U = (x + 22) / 10, V = (z - 3) / 8. Navy with swept teal and gold chevrons."""
    rng = np.random.default_rng(5)
    x = -22.0 + 10.0 * (np.arange(w) + 0.5) / w
    z = 3.0 + 8.0 * (1.0 - (np.arange(h) + 0.5) / h)
    X, Z = np.meshgrid(x, z)
    weather = fbm(h, w, 1.5, rng)
    alb = np.tile(NAVY, (h, w, 1))
    along = X + 0.95 * (Z - 3.0)
    aa = 0.03
    for offset, half, col in ((-16.6, 0.42, TEAL), (-15.5, 0.17, GOLD), (-14.7, 0.06, WHITE)):
        mask = fill(np.abs(along - offset) - half, aa)[..., None]
        alb = alb * (1 - mask) + col * mask
    alb *= (0.92 + 0.14 * weather)[..., None]
    rough = 0.28 + 0.12 * weather
    height = np.zeros_like(rough)
    for k in range(8):
        line = fill(np.abs(Z - (3.5 + k)) - 0.008, 0.012)
        height -= 0.6 * line
        alb *= (1.0 - 0.12 * line)[..., None]
    orm = np.stack([np.ones_like(rough), rough, np.zeros_like(rough)], axis=-1)
    return srgb(alb), orm, normal_map(height, 2.0)


# =============================================================================
# Geometry helpers
# =============================================================================
def grid_object(name, P, coll, parent, mats, *, closed=True, cap_start=False, cap_end=False, uv=None, mat_index=None,
                sharp_deg=38.0, location=(0.0, 0.0, 0.0)):
    """Quad grid from rings P[i, j]. closed: rings wrap around; uv has one extra wrap column then."""
    P = np.asarray(P, dtype=np.float64)
    ni, nj = P.shape[:2]
    offset = np.asarray(location, dtype=np.float64)
    verts = (P - offset).reshape(-1, 3)
    faces, loop_uv, idx = [], [], []
    last_j = nj if closed else nj - 1
    for i in range(ni - 1):
        for j in range(last_j):
            j1 = (j + 1) % nj
            faces.append((i * nj + j, i * nj + j1, (i + 1) * nj + j1, (i + 1) * nj + j))
            if uv is not None:
                jb = j + 1
                loop_uv += [uv[i, j], uv[i, jb], uv[i + 1, jb], uv[i + 1, j]]
            if mat_index is not None:
                idx.append(mat_index(i, j))
    base = len(faces)
    extra = []
    verts = list(map(tuple, verts))
    if cap_start or cap_end:
        for flag, i in ((cap_start, 0), (cap_end, ni - 1)):
            if not flag:
                continue
            c = P[i].mean(axis=0) - offset
            verts.append(tuple(c))
            cv = len(verts) - 1
            for j in range(last_j):
                j1 = (j + 1) % nj
                extra.append((cv, i * nj + j, i * nj + j1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces + extra)
    me.update()
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        data = np.zeros((len(me.loops), 2), dtype=np.float32)
        data[: len(loop_uv)] = np.asarray(loop_uv, dtype=np.float32)
        # Caps sample the first texel row; they are small and face away from the camera.
        layer.data.foreach_set("uv", data.ravel())
    for m in mats:
        me.materials.append(m)
    if mat_index is not None:
        full = np.zeros(len(me.polygons), dtype=np.int32)
        full[: base] = np.asarray(idx, dtype=np.int32)
        me.polygons.foreach_set("material_index", full)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    if hasattr(me, "set_sharp_from_angle"):
        me.set_sharp_from_angle(angle=math.radians(sharp_deg))
    obj = bpy.data.objects.new(name, me)
    teaser.link_object(obj, coll, parent)
    obj.location = tuple(offset)
    return obj


def airfoil(n: int, thickness: float, camber: float) -> np.ndarray:
    """Closed NACA-style section, x 1 -> 0 -> 1 over n points (upper first), blunt rear-loaded nose."""
    beta = np.linspace(0.0, math.pi, n // 2 + 1)
    x = 0.5 * (1.0 + np.cos(beta))  # 1 -> 0
    yt = 5.0 * thickness * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    p = 0.42
    yc = np.where(x < p, camber / p**2 * (2 * p * x - x**2), camber / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    upper = np.stack([x, yc + yt], axis=1)
    lower = np.stack([x[::-1], (yc - yt)[::-1]], axis=1)
    return np.concatenate([upper, lower[1:-1]], axis=0)


def ring_uv(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n + 1)


def wing_chord(y: float) -> tuple[float, float]:
    """(chord, x_le) of the reference planform at span |y|."""
    ay = abs(y)
    knots_y = (0.0, 1.88, 5.7, 17.1)
    knots_c = (7.9, 7.3, 4.55, 1.25)
    knots_x = (6.75, 5.8, 3.75, -2.04)
    return float(np.interp(ay, knots_y, knots_c)), float(np.interp(ay, knots_y, knots_x))


def axis_height(contact_z: float) -> float:
    """Fuselage axis above the model origin; belly about 1.45 m off the ground."""
    return contact_z + 1.45 + RADIUS + 0.12


class AirlinerBuilder:
    def __init__(self, coll, mats: "teaser.Materials", contact_z: float = -1.0) -> None:
        self.coll = coll
        self.m = mats
        self.contact_z = contact_z
        self.zc = axis_height(contact_z)

    # -------------------------------------------------------------- materials
    def _materials(self):
        self.skin = textured("GLB_Air_Fuselage", *self._images("fuselage", paint_fuselage), coat=0.6, normal_strength=0.8)
        self.wing_skin = textured("GLB_Air_Wing", *self._images("wing", paint_wing))
        self.fin_paint = textured("GLB_Air_Fin", *self._images("fin", paint_fin), coat=0.6)
        self.navy = constant("GLB_Air_Navy", tuple(NAVY), 0.0, 0.3, coat=0.5)
        self.cowl = constant("GLB_Air_Cowl", tuple(WHITE), 0.0, 0.3, coat=0.5)
        self.lip = constant("GLB_Air_Lip", (0.85, 0.86, 0.88), 1.0, 0.16)
        self.duct = constant("GLB_Air_Duct", (0.02, 0.022, 0.025), 0.3, 0.55)
        self.titanium = constant("GLB_Air_Fan", (0.55, 0.56, 0.6), 1.0, 0.24)
        self.spinner = constant("GLB_Air_Spinner", (0.06, 0.06, 0.07), 0.8, 0.28)
        self.core = constant("GLB_Air_Core", (0.32, 0.31, 0.3), 0.9, 0.42)
        self.fairing = constant("GLB_Air_Fairing", (0.5, 0.51, 0.53), 0.0, 0.38, coat=0.3)
        self.gear_grey = constant("GLB_Air_Gear", (0.42, 0.43, 0.45), 0.2, 0.42)
        self.rim = constant("GLB_Air_Rim", (0.62, 0.63, 0.65), 0.9, 0.3)
        self.decal_dark = constant("GLB_Air_Decal_Navy", tuple(NAVY), 0.0, 0.32, coat=0.5)
        self.white_mark = constant("GLB_Air_Decal_White", (0.86, 0.87, 0.88), 0.0, 0.32, coat=0.5)

    @staticmethod
    def _images(key, painter):
        alb, orm, nrm = painter()
        return (to_image(f"GLB_air_{key}_albedo", alb, False), to_image(f"GLB_air_{key}_orm", orm, True),
                to_image(f"GLB_air_{key}_normal", nrm, True))

    # ------------------------------------------------------------------ build
    def build(self) -> "teaser.AircraftRig":
        self._materials()
        root = teaser.new_empty("Aircraft", self.coll, size=4.0, kind="ARROWS")
        root.rotation_mode = "QUATERNION"
        rig = teaser.AircraftRig(root=root)
        self._fuselage(root)
        self._decals(root)
        self._wing_fairing(root)
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            self._wing(root, rig, side, sgn)
            self._engine(root, rig, side, sgn)
        self._tail(root, rig)
        self._gear(root, rig)
        self._lights(root, rig)
        rig.focus_tail = teaser.new_empty("Focus_Tail", self.coll, root, location=(-19.0, 0.0, self.zc + 2.0), size=0.3)
        return rig

    # --------------------------------------------------------------- fuselage
    def _fuselage(self, root):
        nj = 96
        s = np.concatenate([
            np.linspace(0.0, 1.0, 34) ** 2 * 6.0,
            np.linspace(6.0, 28.0, 45)[1:],
            np.linspace(28.0, LENGTH, 42)[1:],
        ])
        theta = np.linspace(-math.pi, math.pi, nj + 1)
        P = np.zeros((len(s), nj, 3))
        uv = np.zeros((len(s), nj + 1, 2))
        for i, si in enumerate(s):
            y, z = section_point(si, theta)
            P[i, :, 0] = X_NOSE - si
            P[i, :, 1] = y[:-1]
            P[i, :, 2] = self.zc + z[:-1]
            uv[i, :, 0] = si / LENGTH
            uv[i, :, 1] = (theta + math.pi) / (2 * math.pi)
        grid_object("Fuselage", P, self.coll, root, [self.skin], uv=uv, cap_start=True, cap_end=True, sharp_deg=70.0)
        # APU exhaust.
        t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        rings = []
        for x, r in ((-21.7, 0.24), (-22.05, 0.2), (-22.08, 0.15), (-21.8, 0.12)):
            rings.append(np.stack([np.full_like(t, x), r * np.cos(t), self.zc + 1.14 + r * np.sin(t)], axis=1))
        grid_object("Fuselage_APU", np.stack(rings), self.coll, root, [self.core], cap_end=True)

    def _decals(self, root):
        """Titles and registration: font outlines draped over the actual skin curvature."""
        font = None
        for path in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"):
            if Path(path).exists():
                font = bpy.data.fonts.load(path, check_existing=True)
                break
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            self._drape_text(root, f"Registration_{side}", "UP-A7380", 32.3, 1.03, 0.34, sgn, font, self.decal_dark)
            self._drape_text(root, f"Title_{side}", "ALMA AIR", 19.6, 1.28, 1.05, sgn, font, self.decal_dark)

    def _drape_text(self, root, name, body, station, z_rel, size, sgn, font, mat):
        curve = bpy.data.curves.new(name, "FONT")
        curve.body = body
        curve.size = size
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.fill_mode = "BOTH"
        if font is not None:
            curve.font = font
        tmp = bpy.data.objects.new(name + "__tmp", curve)
        self.coll.objects.link(tmp)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(depsgraph), depsgraph=depsgraph)
        self.coll.objects.unlink(tmp)
        bpy.data.objects.remove(tmp)
        bpy.data.curves.remove(curve)
        # Dense enough to follow the curvature; text reads nose-to-tail on the left, tail-to-nose on the right.
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bmesh.ops.subdivide_edges(bm, edges=[e for e in bm.edges if e.calc_length() > 0.18], cuts=1, use_grid_fill=False)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        for v in bm.verts:
            tx, tz = v.co.x, v.co.y
            s = station + (tx if sgn > 0 else -tx)
            z = z_rel + tz
            v.co = Vector((X_NOSE - s, sgn * (surface_y(s, z) + 0.018), self.zc + z))
        bm.normal_update()
        bm.to_mesh(me)
        bm.free()
        me.materials.clear()
        me.materials.append(mat)
        obj = bpy.data.objects.new(name, me)
        teaser.link_object(obj, self.coll, root)

    def _wing_fairing(self, root):
        """Belly fairing: flattened ellipsoid under the centre section, with the gear wells' dark bays."""
        nj = 48
        xs = np.linspace(-1.0, 1.0, 34)
        rings = []
        for t in xs:
            f = math.sqrt(max(1.0 - t**2, 1e-4))
            x = 1.6 + 8.4 * t
            hw = 2.42 * f ** 0.7
            depth = 1.05 * f ** 0.8
            a = np.linspace(0, 2 * math.pi, nj, endpoint=False)
            rings.append(np.stack([np.full(nj, x), hw * np.sin(a), self.zc - 1.15 + depth * np.cos(a)], axis=1))
        grid_object("Fuselage_Fairing", np.stack(rings), self.coll, root, [self.fairing], sharp_deg=75.0)

    # ------------------------------------------------------------------- wing
    def _wing_z(self, ay: float) -> float:
        return self.zc - 1.28 + (ay - 1.2) * math.tan(math.radians(6.0))

    def _section(self, y, f0=0.0, f1=1.0, n=48):
        """One ring for span station y: points between chord fractions f0..f1, closed at the cut."""
        ay = abs(y)
        chord, x_le = wing_chord(ay)
        t = np.clip(ay / SEMI_SPAN, 0.0, 1.0)
        thick = float(np.interp(t, (0.0, 0.3, 1.0), (0.155, 0.125, 0.105)))
        pts = airfoil(n, thick, 0.022)
        xf = np.clip(pts[:, 0], f0, f1)
        twist = math.radians(float(np.interp(t, (0.0, 1.0), (2.0, -1.2))))
        x_rel, z_rel = xf * chord, pts[:, 1] * chord
        cx = 0.3 * chord
        x = x_le - (cx + (x_rel - cx) * math.cos(twist) + z_rel * math.sin(twist))
        z = self._wing_z(ay) + z_rel * math.cos(twist) - (x_rel - cx) * math.sin(twist)
        return np.stack([x, np.full_like(x, y), z], axis=1)

    def _panel(self, name, parent, stations, f0, f1, sgn, mats, location=(0.0, 0.0, 0.0), xform=None, cap=True):
        n = 48
        rings = np.stack([self._section(sgn * y, f0, f1, n) for y in stations])
        if xform is not None:
            rings = xform(rings)
        uv = np.zeros((len(stations), rings.shape[1] + 1, 2))
        for i, y in enumerate(stations):
            uv[i, :, 0] = abs(y) / SEMI_SPAN
            uv[i, :, 1] = ring_uv(rings.shape[1])
        return grid_object(name, rings, self.coll, parent, mats, uv=uv, cap_start=cap, cap_end=cap, location=location)

    def _hinged(self, rig, name, sgn_channel_sign, parent, stations, f0, f1, sgn):
        """Control surface with its origin on the true hinge line, articulated in axis-angle."""
        y0, y1 = stations[0], stations[-1]
        a, b = self._hinge_at(sgn * y0, f0), self._hinge_at(sgn * y1, f0)
        origin = (a + b) * 0.5
        obj = self._panel(name, parent, stations, f0, f1, sgn, [self.wing_skin], location=tuple(origin))
        axis = (b - a).normalized()
        if axis.y < 0:
            axis = -axis
        obj.rotation_mode = "AXIS_ANGLE"
        obj.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
        return obj, axis

    def _hinge_at(self, y, f):
        ay = abs(y)
        chord, x_le = wing_chord(ay)
        return Vector((x_le - f * chord, y, self._wing_z(ay) + 0.015 * chord))

    def _wing(self, root, rig, side, sgn):
        ys = lambda a, b, n: list(np.linspace(a, b, n))  # noqa: E731
        spans = ys(0.9, 5.7, 7) + ys(5.7, 17.1, 18)[1:]
        # Structural box between slats and flaps (full chord where nothing is deployed).
        self._panel(f"Wing_{side}", root, spans, 0.015, 0.70, sgn, [self.wing_skin], cap=False)

        # Leading-edge slats, drooped and extended for landing.
        def slat_xform(rings):
            out = rings.copy()
            ang = math.radians(22.0)
            for i in range(len(out)):
                pivot = out[i].copy()
                px = pivot[:, 0].min()  # aft cut of the slat
                pz = pivot[np.argmin(pivot[:, 0]), 2]
                dx, dz = out[i, :, 0] - px, out[i, :, 2] - pz
                out[i, :, 0] = px + dx * math.cos(ang) + dz * math.sin(ang) + 0.14
                out[i, :, 2] = pz - dx * math.sin(ang) + dz * math.cos(ang) - 0.07
            return out

        self._panel(f"Wing_Slat_Inner_{side}", root, ys(2.1, ENGINE_Y - 0.55, 5), 0.0, 0.12, sgn, [self.wing_skin], xform=slat_xform)
        self._panel(f"Wing_Slat_Outer_{side}", root, ys(ENGINE_Y + 0.55, 16.7, 14), 0.0, 0.12, sgn, [self.wing_skin], xform=slat_xform)

        # Double-slotted landing flaps, Fowler-extended and deflected 30 degrees.
        def flap_xform(drop, shift):
            def apply(rings):
                out = rings.copy()
                ang = math.radians(drop)
                for i in range(len(out)):
                    px = out[i, :, 0].max()
                    pz = out[i, np.argmax(out[i, :, 0]), 2]
                    dx, dz = out[i, :, 0] - px, out[i, :, 2] - pz
                    out[i, :, 0] = px + dx * math.cos(ang) + dz * math.sin(ang) - shift
                    out[i, :, 2] = pz + dx * math.sin(ang) + dz * math.cos(ang) - 0.05 - shift * 0.22
                return out
            return apply

        for name, a, b in (("Inner", 1.95, ENGINE_Y + 0.2), ("Outer", ENGINE_Y + 0.25, 11.9)):
            stations = ys(a, b, 8)
            self._panel(f"Wing_Flap_{name}_{side}", root, stations, 0.71, 0.88, sgn, [self.wing_skin], xform=flap_xform(24.0, 0.42))
            self._panel(f"Wing_Flap_{name}_Aft_{side}", root, stations, 0.88, 1.0, sgn, [self.wing_skin], xform=flap_xform(38.0, 0.98))
        # Trailing panel outboard of the flaps (ahead of and beyond the aileron).
        self._panel(f"Wing_Tip_Fixed_{side}", root, ys(12.0, 17.1, 8), 0.70, 0.76, sgn, [self.wing_skin], cap=False)

        name = f"Aileron_{side}"
        obj, axis = self._hinged(rig, name, 0, root, ys(12.05, 16.4, 7), 0.77, 1.0, sgn)
        rig.surfaces[name] = (obj, axis, -1.0 if sgn > 0 else 1.0)
        self._panel(f"Wing_Tip_Aft_{side}", root, ys(16.45, 17.1, 3), 0.76, 1.0, sgn, [self.wing_skin])

        # Spoiler panels, flush on the upper surface and raised after touchdown.
        y0, y1 = 3.1, 11.6
        a, b = self._hinge_at(sgn * y0, 0.60), self._hinge_at(sgn * y1, 0.60)
        for p in (a, b):
            ay = abs(p.y)
            chord, _ = wing_chord(ay)
            p.z = self._wing_z(ay) + 0.07 * chord
        origin = (a + b) * 0.5
        rings = []
        for y in ys(y0, y1, 7):
            chord, x_le = wing_chord(y)
            z = self._wing_z(y) + 0.076 * chord
            rings.append([(x_le - f * chord, sgn * y, z + dz * chord) for f, dz in ((0.60, 0.002), (0.70, -0.012), (0.70, -0.022), (0.60, -0.012))])
        rings = np.array(rings)
        spoiler = grid_object(f"Spoiler_{side}", rings, self.coll, root, [self.wing_skin], cap_start=True, cap_end=True,
                              location=tuple(origin), sharp_deg=30.0)
        axis = (b - a).normalized()
        if axis.y < 0:
            axis = -axis
        spoiler.rotation_mode = "AXIS_ANGLE"
        spoiler.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)

        # Flap-track canoe fairings.
        for k, y in enumerate((3.4, 6.6, 9.3, 11.6)):
            chord, x_le = wing_chord(y)
            te = x_le - chord
            self._canoe(f"Wing_Canoe_{k}_{side}", root, x_le - 0.5 * chord, te - 0.75, sgn * y, self._wing_z(y) - 0.06, 0.3)

        self._winglet(root, side, sgn)

    def _canoe(self, name, parent, x0, x1, y, z, radius):
        nj = 16
        t = np.linspace(0.0, 1.0, 16)
        a = np.linspace(0, 2 * math.pi, nj, endpoint=False)
        rings = []
        for u in t:
            r = radius * (np.sin(math.pi * (0.04 + 0.86 * u) ** 0.7) ** 0.5 + 0.04)
            rings.append(np.stack([np.full(nj, x0 + (x1 - x0) * u), y + r * 0.72 * np.sin(a), z - r * 0.6 - r * 1.15 * (np.cos(a) * 0.5 + 0.5) - 0.45 * u * radius * 2], axis=1))
        grid_object(name, np.stack(rings), self.coll, parent, [self.fairing], sharp_deg=70.0, cap_start=True, cap_end=True)

    def _winglet(self, root, side, sgn):
        """Blended winglet continuing the planform: the section's span axis rolls up over a radius."""
        n = 40
        y0 = 17.1
        c0, x0 = wing_chord(y0)
        z0 = self._wing_z(y0)
        count = 20
        rings = []
        uv = np.zeros((count, n + 1, 2))
        pts = airfoil(n, 0.1, 0.012)
        for i in range(count):
            u = i / (count - 1)
            ang = math.radians(78.0) * min(u / 0.45, 1.0)
            if u < 0.45:
                rad = 0.8
                dy = rad * math.sin(ang)
                dz = rad * (1.0 - math.cos(ang))
            else:
                rad = 0.8
                dy = rad * math.sin(ang) + (u - 0.45) / 0.55 * 2.1 * math.cos(ang)
                dz = rad * (1.0 - math.cos(ang)) + (u - 0.45) / 0.55 * 2.1 * math.sin(ang)
            chord = c0 + (0.33 - c0) * u ** 0.9
            x_le = x0 - 1.25 * u ** 1.1
            xr = pts[:, 0] * chord
            tr = pts[:, 1] * chord
            x = x_le - xr
            yy = sgn * (y0 + dy - tr * math.sin(ang))
            zz = z0 + dz + tr * math.cos(ang)
            rings.append(np.stack([x, yy, zz], axis=1))
            uv[i, :, 0] = u
            uv[i, :, 1] = ring_uv(n)
        grid_object(f"Wing_Winglet_{side}", np.stack(rings), self.coll, root, [self.navy], cap_end=True)

    # ----------------------------------------------------------------- engine
    def _engine(self, root, rig, side, sgn):
        y = sgn * ENGINE_Y
        zc = self.contact_z + 1.56
        nj = 56
        a = np.linspace(0, 2 * math.pi, nj, endpoint=False)
        flat = np.where(np.cos(a) < 0, 0.86, 1.0)  # CFM56 flattened lower cowl for ground clearance

        def rev(x, r, squash=True):
            return np.stack([np.full(nj, x), y + r * np.sin(a), zc + r * np.cos(a) * (flat if squash else 1.0)], axis=1)

        # Outer cowl from the inner lip, over the highlight, to the fan-cowl trailing edge, then back
        # through the duct; a single closed revolve gives a continuous highlight reflection.
        x_in = 6.95
        outer = [(x_in - 0.60, 0.75), (x_in - 0.28, 0.77), (x_in - 0.08, 0.815), (x_in, 0.865), (x_in - 0.02, 0.92), (x_in - 0.10, 0.965),
                 (x_in - 0.32, 1.005), (x_in - 0.8, 1.035), (x_in - 1.5, 1.04), (x_in - 2.4, 1.0), (x_in - 3.15, 0.9), (x_in - 3.55, 0.8)]
        rings = np.stack([rev(x, r) for x, r in outer])
        lip_rows = 5

        def cowl_mat(i, j):
            if i < lip_rows:
                return 1
            return 2 if i == 7 else 0

        grid_object(f"Engine_Cowl_{side}", rings, self.coll, root, [self.cowl, self.lip, self.navy], mat_index=cowl_mat, sharp_deg=50.0)
        # Inlet duct (dark acoustic liner) and bypass duct exit.
        duct = np.stack([rev(x, r) for x, r in ((x_in - 0.6, 0.75), (x_in - 1.25, 0.72), (x_in - 2.1, 0.66), (x_in - 3.55, 0.8))])
        grid_object(f"Engine_Duct_{side}", duct, self.coll, root, [self.duct], sharp_deg=70.0)
        # Core cowl and primary nozzle tapering to the plug.
        core = np.stack([rev(x, r, False) for x, r in ((x_in - 2.3, 0.62), (x_in - 3.5, 0.62), (x_in - 4.2, 0.48), (x_in - 4.75, 0.33),
                                                         (x_in - 4.8, 0.27), (x_in - 4.7, 0.25))])
        grid_object(f"Engine_Core_{side}", core, self.coll, root, [self.core], sharp_deg=45.0)
        plug = np.stack([rev(x, r, False) for x, r in ((x_in - 4.65, 0.25), (x_in - 5.0, 0.2), (x_in - 5.45, 0.08), (x_in - 5.6, 0.01))])
        grid_object(f"Engine_Plug_{side}", plug, self.coll, root, [self.core], sharp_deg=60.0)

        # Pylon: airfoil strut from the cowl crown to the wing's underside.
        stations = []
        for z_t, x_f, x_a in ((zc + 0.8, x_in - 0.85, x_in - 4.5), (zc + 1.25, x_in - 1.75, x_in - 4.9)):
            secs = airfoil(24, 0.065, 0.0)
            stations.append(np.stack([x_f - secs[:, 0] * (x_f - x_a), y + secs[:, 1] * (x_f - x_a), np.full(len(secs), z_t)], axis=1))
        grid_object(f"Engine_Pylon_{side}", np.stack(stations), self.coll, root, [self.cowl], cap_start=True, cap_end=True, sharp_deg=30.0)

        # Fan: titanium blades on a rotating node, spinner with the painted swirl mark.
        fan_x = x_in - 0.95
        fan = teaser.new_empty(f"Fan_{side}", self.coll, root, location=(fan_x, y, zc), size=0.3)
        fan.rotation_mode = "XYZ"
        self._fan_blades(fan, side)
        if sgn > 0:
            rig.prop = fan
        rig.props = getattr(rig, "props", []) + [fan]

    def _fan_blades(self, fan, side):
        blades = 24
        bm = bmesh.new()
        for k in range(blades):
            rot = 2 * math.pi * k / blades
            rings = []
            for r in np.linspace(0.24, 0.69, 7):
                u = (r - 0.24) / 0.45
                chord = 0.22 - 0.05 * u
                twist = math.radians(55.0 - 34.0 * u)
                ring = []
                for c, t in ((-0.5, 0.0), (0.0, 0.012), (0.5, 0.0), (0.0, -0.012)):
                    tang = chord * c * math.cos(twist) + t * math.sin(twist)
                    axial = -chord * c * math.sin(twist) + t * math.cos(twist) - 0.04 * u
                    py, pz = r, tang
                    ring.append(Vector((axial, py * math.cos(rot) - pz * math.sin(rot), py * math.sin(rot) + pz * math.cos(rot))))
                rings.append(ring)
            teaser.loft(bm, rings)
        me = bpy.data.meshes.new(f"Fan_Blades_{side}")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(self.titanium)
        me.shade_smooth()
        if hasattr(me, "set_sharp_from_angle"):
            me.set_sharp_from_angle(angle=math.radians(35.0))
        teaser.link_object(bpy.data.objects.new(f"Fan_Blades_{side}", me), self.coll, fan)

        nj = 36
        a = np.linspace(0, 2 * math.pi, nj, endpoint=False)
        prof = [(0.34, 0.003), (0.27, 0.05), (0.17, 0.12), (0.06, 0.2), (-0.06, 0.25), (-0.26, 0.26)]
        rings = np.stack([np.stack([np.full(nj, x), r * np.sin(a), r * np.cos(a)], axis=1) for x, r in prof])

        def swirl(i, j):
            phase = (j / nj * 2.0 + i * 0.09) % 1.0
            return 1 if 0.0 < phase < 0.1 and i < 4 else 0

        grid_object(f"Fan_Spinner_{side}", rings, self.coll, fan, [self.spinner, self.white_mark], mat_index=swirl, sharp_deg=60.0)

    # ------------------------------------------------------------------- tail
    def _tail(self, root, rig):
        zs = self.zc + 0.42
        half = 7.1

        def plan(ay):
            t = ay / half
            chord = 4.5 + (1.45 - 4.5) * t
            xl = -15.3 - t * half * math.tan(math.radians(32.0))
            return chord, xl, zs + ay * math.tan(math.radians(7.0))

        def ring(y, f0=0.0, f1=1.0, n=36):
            chord, xl, z = plan(abs(y))
            pts = airfoil(n, 0.11, 0.0)
            xf = np.clip(pts[:, 0], f0, f1)
            return np.stack([xl - xf * chord, np.full(n, y), z + pts[:, 1] * chord], axis=1)

        for side, sgn in (("L", 1.0), ("R", -1.0)):
            stations = np.linspace(0.8, half, 11)
            P = np.stack([ring(sgn * y, 0.0, 0.70) for y in stations])
            uv = np.zeros((len(stations), P.shape[1] + 1, 2))
            uv[:, :, 1] = ring_uv(P.shape[1])
            uv[:, :, 0] = (stations / half)[:, None] * 0.6
            grid_object(f"Stabilizer_{side}", P, self.coll, root, [self.wing_skin], uv=uv, cap_end=True)

            st = np.linspace(1.5, half - 0.05, 9)
            P = np.stack([ring(sgn * y, 0.71, 1.0) for y in st])
            ca, xa, za = plan(st[0])
            cb, xb, zb = plan(st[-1])
            a = Vector((xa - 0.71 * ca, sgn * st[0], za))
            b = Vector((xb - 0.71 * cb, sgn * st[-1], zb))
            origin = (a + b) * 0.5
            uv = np.zeros((len(st), P.shape[1] + 1, 2))
            uv[:, :, 1] = ring_uv(P.shape[1])
            uv[:, :, 0] = (st / half)[:, None] * 0.6
            name = f"Elevator_{side}"
            obj = grid_object(name, P, self.coll, root, [self.wing_skin], uv=uv, cap_start=True, cap_end=True, location=tuple(origin))
            axis = (b - a).normalized()
            if axis.y < 0:
                axis = -axis
            obj.rotation_mode = "AXIS_ANGLE"
            obj.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
            rig.surfaces[name] = (obj, axis, -1.0)

        z0 = self.zc + 1.05
        z1 = self.zc + 7.05
        frac = 0.70

        def fin_plan(z):
            t = (z - z0) / (z1 - z0)
            return 6.2 + (1.55 - 6.2) * t, -13.6 - (z - z0) * math.tan(math.radians(38.0))

        def fin_ring(z, f0=0.0, f1=1.0, n=40):
            chord, xl = fin_plan(z)
            pts = airfoil(n, 0.105, 0.0)
            xf = np.clip(pts[:, 0], f0, f1)
            return np.stack([xl - xf * chord, pts[:, 1] * chord, np.full(n, z)], axis=1)

        def fin_uv(P):
            uv = np.zeros((P.shape[0], P.shape[1] + 1, 2))
            for i in range(P.shape[0]):
                closed = np.concatenate([P[i], P[i][:1]], axis=0)
                uv[i, :, 0] = (closed[:, 0] + 22.0) / 10.0
                uv[i, :, 1] = (closed[:, 2] - 3.0) / 8.0
            return uv

        zs_fin = np.linspace(z0 - 0.15, z1, 13)
        P = np.stack([fin_ring(z, 0.0, frac - 0.01) for z in zs_fin])
        grid_object("Fin", P, self.coll, root, [self.fin_paint], uv=fin_uv(P), cap_end=True)

        zr = np.linspace(z0 + 0.15, z1 - 0.04, 9)
        P = np.stack([fin_ring(z, frac, 1.0) for z in zr])

        def hinge(z):
            chord, xl = fin_plan(z)
            return Vector((xl - frac * chord, 0.0, z))

        a, b = hinge(zr[0]), hinge(zr[-1])
        origin = (a + b) * 0.5
        rudder = grid_object("Rudder", P, self.coll, root, [self.fin_paint], uv=fin_uv(P), cap_start=True, cap_end=True, location=tuple(origin))
        axis = (b - a).normalized()
        if axis.z < 0:
            axis = -axis
        rudder.rotation_mode = "AXIS_ANGLE"
        rudder.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
        rig.surfaces["Rudder"] = (rudder, axis, -1.0)

        # Fin tip cap and the dorsal fillet blending the fin into the crown.
        tip = np.stack([fin_ring(z1), fin_ring(z1 + 0.09) * np.array([1, 0.5, 1]) + np.array([-0.05, 0, 0])])
        grid_object("Fin_Tip", tip, self.coll, root, [self.navy], cap_end=True)
        rings = []
        for u in np.linspace(0.0, 1.0, 9):
            c = 8.3 - 2.4 * u
            xl = -8.2 - 5.6 * u
            z = self.zc + 1.82 + (z0 + 0.45 - (self.zc + 1.82)) * u
            pts = airfoil(28, 0.012 + 0.05 * u, 0.0)
            rings.append(np.stack([xl - pts[:, 0] * c, pts[:, 1] * c, np.full(28, z)], axis=1))
        grid_object("Dorsal_Fin", np.stack(rings), self.coll, root, [self.cowl], cap_start=True, cap_end=True, sharp_deg=70.0)

    # ------------------------------------------------------------------- gear
    def _wheel(self, name, hub, radius, width, track):
        """Twin wheels on one axle: treaded tire (grooved revolve), rim and hub."""
        nj = 48
        a = np.linspace(0, 2 * math.pi, nj, endpoint=False)
        wheel = teaser.new_empty(f"Wheel_{name}", self.coll, hub, size=0.1)
        wheel.rotation_mode = "XYZ"
        prof_t = [(0.0, 0.48), (0.08, 0.72), (0.2, 0.9), (0.3, 0.965), (0.36, 0.985), (0.385, 1.0), (0.46, 1.0), (0.47, 0.975), (0.5, 0.975)]
        profile = prof_t + [(1.0 - w, r) for w, r in reversed(prof_t[:-1])]
        for k, sy in enumerate((1.0, -1.0)):
            yc = sy * track
            rings = np.stack([np.stack([radius * r * np.sin(a), np.full(nj, yc + width * (w - 0.5)), radius * r * np.cos(a)], axis=1) for w, r in profile])
            grid_object(f"Wheel_Tire_{name}_{k}", rings, self.coll, wheel, [self.m.rubber], sharp_deg=50.0)
            rim_prof = [(0.03, 0.5), (0.12, 0.47), (0.2, 0.2), (0.2, 0.06)]
            inner = sy * (width * 0.5)
            outer_face = np.stack([np.stack([radius * r * np.sin(a), np.full(nj, yc + inner * (1.0 - 2.0 * w) * 0.98), radius * r * np.cos(a)], axis=1) for w, r in rim_prof])
            grid_object(f"Wheel_Rim_{name}_{k}", outer_face, self.coll, wheel, [self.rim], sharp_deg=30.0, cap_end=True)
        # Axle (does not need to spin; a thin cylinder hidden between the rims).
        bm = bmesh.new()
        teaser.add_cylinder(bm, Vector((0, track + 0.05, 0)), Vector((0, -track - 0.05, 0)), radius * 0.13, radius * 0.13, 16)
        teaser.mesh_object(f"Axle_{name}", bm, self.coll, hub, [self.gear_grey], smooth=True)
        return wheel

    def _leg(self, rig, name, parent, attach, hub_rest, radius, strut_r, width, track, *, steer=False):
        hub = teaser.new_empty(f"Hub_{name}", self.coll, parent, location=hub_rest, size=0.2)
        rig.gear_hubs[name] = (hub, hub_rest.copy())
        teaser.new_empty(f"Attach_{name}", self.coll, parent, location=attach, size=0.15)
        length = (hub_rest - attach).length
        bm = bmesh.new()
        teaser.add_cylinder(bm, Vector((0, 0, 0)), Vector((0, length, 0)), strut_r, strut_r * 0.62, 20)
        strut = teaser.mesh_object(f"Strut_{name}", bm, self.coll, parent, [self.m.chrome], smooth=True)
        strut.location = attach
        c = strut.constraints.new("STRETCH_TO")
        c.target = hub
        c.rest_length = length
        c.volume = "NO_VOLUME"
        c.keep_axis = "PLANE_X"
        # Oleo outer cylinder, fixed in the upper half so it hides the sliding piston.
        d = (hub_rest - attach).normalized()
        bm = bmesh.new()
        teaser.add_cylinder(bm, attach, attach + d * length * 0.56, strut_r * 1.45, strut_r * 1.35, 20)
        teaser.mesh_object(f"Oleo_{name}", bm, self.coll, parent, [self.gear_grey], smooth=True)
        wheel = self._wheel(name, hub, radius, width, track)
        rig.wheels[name] = (wheel, radius)
        wheel["wheel_radius"] = radius
        hub["wheel_radius"] = radius
        return hub

    def _gear(self, root, rig):
        z0 = self.contact_z
        # Main gear: wing-mounted trailing legs, wheels on the simulated main-gear station.
        for name, sgn in (("main_left", 1.0), ("main_right", -1.0)):
            hub_rest = Vector((-0.3, sgn * MAIN_Y, z0 + MAIN_RADIUS))
            attach = Vector((-0.75, sgn * 2.52, self._wing_z(2.52) - 0.1))
            self._leg(rig, name, root, attach, hub_rest, MAIN_RADIUS, 0.11, 0.38, 0.43)
            d = (hub_rest - attach).normalized()
            mid = attach + d * (hub_rest - attach).length * 0.42
            bm = bmesh.new()
            teaser.add_cylinder(bm, mid, Vector((-0.25, sgn * 1.36, self.zc - 1.0)), 0.05, 0.04, 12)     # side strut
            teaser.add_cylinder(bm, mid, Vector((-1.75, sgn * 2.7, self._wing_z(2.7) - 0.1)), 0.05, 0.04, 12)  # drag strut
            teaser.mesh_object(f"Gear_Brace_{name}", bm, self.coll, root, [self.gear_grey], smooth=True)
            # Main-gear door riding on the upper leg.
            bm = bmesh.new()
            side = Vector((0.75, 0.0, 0.0))
            p0, p1 = attach + Vector((0, sgn * 0.14, 0)), mid + Vector((0, sgn * 0.14, 0))
            face = [bm.verts.new(p0 - side * 0.8), bm.verts.new(p0 + side), bm.verts.new(p1 + side * 0.55), bm.verts.new(p1 - side * 0.55)]
            bm.faces.new(face)
            bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.025)
            teaser.mesh_object(f"Gear_Door_{name}", bm, self.coll, root, [self.fairing], smooth=False)
        # Nose gear on a steering node.
        x_n = -0.3 + WHEELBASE
        steer = teaser.new_empty("Nose_Steer", self.coll, root, location=(x_n, 0.0, 0.0), size=0.3)
        rig.nose_steer = steer
        hub_rest = Vector((0.0, 0.0, z0 + NOSE_RADIUS))
        attach = Vector((0.0, 0.0, self.zc - 1.55))
        self._leg(rig, "nose", steer, attach, hub_rest, NOSE_RADIUS, 0.075, 0.25, 0.24)
        bm = bmesh.new()
        teaser.add_cylinder(bm, Vector((x_n, 0, self.zc - 1.75)), Vector((x_n + 1.1, 0, self.zc - 1.95)), 0.04, 0.035, 10)
        for sgn in (1.0, -1.0):
            pts = [(x_n + 0.8, sgn * 0.42, self.zc - 1.72), (x_n - 0.8, sgn * 0.42, self.zc - 1.72), (x_n - 0.55, sgn * 0.44, self.zc - 2.12), (x_n + 0.55, sgn * 0.44, self.zc - 2.12)]
            bm.faces.new([bm.verts.new(p) for p in pts])
        teaser.mesh_object("Gear_Nose_Doors", bm, self.coll, root, [self.fairing], smooth=False)

    # ----------------------------------------------------------------- lights
    def _bulb(self, name, loc, mat, radius, root):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=radius)
        obj = teaser.mesh_object(name, bm, self.coll, root, [mat], smooth=True)
        obj.location = loc
        return obj

    def _lights(self, root, rig):
        tip_y, tip_x = SEMI_SPAN - 0.3, wing_chord(17.1)[1] - 0.25
        tip_z = self._wing_z(17.1) + 0.08
        for name, sgn, mat, color in (("Nav_Left", 1.0, self.m.nav_red, (1.0, 0.05, 0.03)), ("Nav_Right", -1.0, self.m.nav_green, (0.05, 1.0, 0.25))):
            self._bulb(name, (tip_x, sgn * tip_y, tip_z), mat, 0.08, root)
            lamp = bpy.data.lights.new(name + "_Lamp", "POINT")
            lamp.color = color
            lamp.energy = 80.0
            teaser.link_object(bpy.data.objects.new(name + "_Lamp", lamp), self.coll, root).location = (tip_x, sgn * tip_y, tip_z)
        for name, loc in (("Strobe_Tail", (-22.15, 0.0, self.zc + 1.14)), ("Strobe_Top", (0.0, 0.0, self.zc + 2.05)),
                          ("Strobe_Belly", (2.0, 0.0, self.zc - 2.24)), ("Strobe_Wing_L", (tip_x - 0.6, tip_y, tip_z)),
                          ("Strobe_Wing_R", (tip_x - 0.6, -tip_y, tip_z))):
            self._bulb(name, loc, self.m.strobe, 0.07, root)
            lamp = bpy.data.lights.new(name + "_Lamp", "POINT")
            lamp.energy = 0.0
            obj = teaser.link_object(bpy.data.objects.new(name + "_Lamp", lamp), self.coll, root)
            obj.location = loc
            rig.strobes.append(lamp)
        # Inboard landing lights in the wing roots, plus the nose-gear taxi light.
        for name, loc in (("Landing_Light_Lens_L", (5.5, 2.25, self._wing_z(2.25) + 0.02)), ("Landing_Light_Lens_R", (5.5, -2.25, self._wing_z(2.25) + 0.02)),
                          ("Landing_Light_Lens_N", (X_NOSE - 2.2, 0.0, self.zc - 2.1))):
            bm = bmesh.new()
            bmesh.ops.create_circle(bm, cap_ends=True, segments=24, radius=0.13, matrix=Matrix.Rotation(math.pi / 2, 4, "Y"))
            teaser.mesh_object(name, bm, self.coll, root, [self.m.landing_lens]).location = loc
        spot = bpy.data.lights.new("Landing_Light", "SPOT")
        spot.energy = 90000.0
        spot.spot_size = math.radians(22.0)
        spot.spot_blend = 0.6
        spot.color = (1.0, 0.94, 0.82)
        obj = teaser.link_object(bpy.data.objects.new("Landing_Light", spot), self.coll, root)
        obj.location = (5.6, 0.0, self._wing_z(2.25))
        obj.rotation_euler = (0.0, -math.pi / 2 - math.radians(5.0), 0.0)


# =============================================================================
# Aerodynamic streamlines: Rankine fuselage + elliptic lifting line, integrated in the body frame
# =============================================================================
def _segment_velocity(p: np.ndarray, a: np.ndarray, b: np.ndarray, gamma: np.ndarray, core: float = 0.45) -> np.ndarray:
    """Biot-Savart for straight vortex segments a->b, with a Lamb-Oseen-style core."""
    r0 = b - a
    r1 = p[None, :] - a
    r2 = p[None, :] - b
    cross = np.cross(r1, r2)
    c2 = (cross**2).sum(axis=1) + (core * np.linalg.norm(r0, axis=1)) ** 2
    n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
    dot = (r0 * (r1 / n1[:, None] - r2 / n2[:, None])).sum(axis=1)
    return ((gamma * dot / (4.0 * math.pi * c2))[:, None] * cross).sum(axis=0)


def compute_streamlines(zc: float, alpha_deg: float = 6.0, cl: float = 1.15) -> dict:
    """Streamlines around the airframe in landing configuration; U = 1, positions in metres."""
    alpha = math.radians(alpha_deg)
    u_inf = np.array([-math.cos(alpha), 0.0, math.sin(alpha)])
    # Fuselage: axial source line with strength dA/ds.
    s = np.linspace(0.0, LENGTH, 60)
    hw, top, bot = section(s)
    area = math.pi * hw * 0.5 * (top - bot)
    ds = s[1] - s[0]
    sigma = np.gradient(area, ds) * ds * math.cos(alpha)
    src = np.stack([X_NOSE - s, np.zeros_like(s), np.full_like(s, zc) + 0.5 * (top + bot)], axis=1)
    # Wing: horseshoe vortices on the quarter chord, elliptic loading, wake trailing along the flow.
    b = 2.0 * SEMI_SPAN
    gamma0 = 2.0 * cl * 125.0 / (math.pi * b)
    edges = np.sin(np.linspace(-math.pi / 2, math.pi / 2, 41)) * (SEMI_SPAN - 0.4)
    def lifting_point(y):
        c, xl = wing_chord(y)
        return np.array([xl - 0.25 * c, y, zc - 1.28 + (abs(y) - 1.2) * math.tan(math.radians(6.0))])
    pts = np.stack([lifting_point(y) for y in edges])
    gam = gamma0 * np.sqrt(np.maximum(1.0 - (edges / SEMI_SPAN) ** 2, 0.0))
    wake = np.array([-1.0, 0.0, math.sin(alpha) * 0.3]) * 250.0
    # Bound segments carry the mid-segment loading; trailing legs carry the jump in loading.
    bound_a, bound_b = pts[:-1], pts[1:]
    bound_g = 0.5 * (gam[:-1] + gam[1:])
    jump = np.diff(np.concatenate([[0.0], bound_g, [0.0]]))
    trail_a, trail_b = pts, pts + wake

    def velocity(p):
        v = u_inf.copy()
        r = p[None, :] - src
        d = np.sqrt((r**2).sum(axis=1) + 0.25)
        v += ((sigma / (4.0 * math.pi * d**3))[:, None] * r).sum(axis=0)
        v += _segment_velocity(p, bound_a, bound_b, -bound_g)
        v += _segment_velocity(p, trail_b, trail_a, jump)
        return v

    seeds = []
    x0 = X_NOSE + 11.0
    # Over and under the wing, fuselage crown, and dense rows around each tip where the vortex forms.
    for y in (0.0, 2.6, 5.2, 7.8, 10.4, 13.0, 15.6, 18.2):
        for z in (-2.7, 1.0, 2.6, 4.6):
            for sgn in ((1.0, -1.0) if y > 0 else (1.0,)):
                seeds.append((x0, sgn * y, zc + z))
    for dy in (-1.3, -0.5, 0.3, 1.1, 1.9):
        for dz in (-0.2, 0.9):
            for sgn in (1.0, -1.0):
                seeds.append((x0, sgn * (SEMI_SPAN + dy), zc + 1.1 + dz))
    lines = []
    step = 1.05
    for seed in seeds:
        p = np.array(seed, dtype=np.float64)
        pts_out = [p.copy()]
        for _ in range(72):
            k1 = velocity(p)
            k2 = velocity(p + 0.5 * step * k1)
            p = p + step * k2
            pts_out.append(p.copy())
        lines.append(np.round(np.array(pts_out), 3))
    return {"lines": [[[float(q[0]), float(q[2]), float(-q[1])] for q in line[::2]] for line in lines], "alpha_deg": alpha_deg, "cl": cl}


# =============================================================================
# Stand-alone preview (live Blender session; leaves the user's own scene alone)
# =============================================================================
def build_preview(scene_name: str = "Airliner_Preview") -> bpy.types.Scene:
    old = bpy.data.scenes.get(scene_name)
    if old is not None:
        for obj in list(old.collection.all_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.scenes.remove(old)
    scene = bpy.data.scenes.new(scene_name)
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    coll = bpy.data.collections.new(scene_name)
    scene.collection.children.link(coll)
    mats = teaser.Materials("golden")
    AirlinerBuilder(coll, mats).build()
    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, -1.0))
    ground = bpy.context.active_object
    ground.name = "Preview_Ground"
    ground.data.materials.append(constant("Preview_Asphalt", (0.05, 0.05, 0.055), 0.0, 0.8))
    sun = bpy.data.lights.new("Preview_Sun", "SUN")
    sun.energy = 4.0
    sun.angle = math.radians(2.0)
    sun.color = (1.0, 0.86, 0.68)
    sun_obj = bpy.data.objects.new("Preview_Sun", sun)
    coll.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(62), 0.0, math.radians(-40))
    world = bpy.data.worlds.new("Preview_World")
    if world.node_tree is None:
        world.use_nodes = True
    next(n for n in world.node_tree.nodes if n.type == "BACKGROUND").inputs[0].default_value = (0.55, 0.68, 0.85, 1.0)
    scene.world = world
    cam = bpy.data.cameras.new("Preview_Cam")
    cam.lens = 50.0
    cam_obj = bpy.data.objects.new("Preview_Cam", cam)
    coll.objects.link(cam_obj)
    cam_obj.location = (45.0, -55.0, 12.0)
    target = Vector((-2.0, 0.0, 2.5))
    cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj
    return scene
