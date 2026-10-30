"""Procedural airfield infrastructure for the crosswind-landing scene (Blender bpy).

Adds what turns a lone runway into a working aerodrome. Everything is placed in
the scene's ENU frame (runway along +Y from the threshold at the origin) on the
east side of the runway, clear of the approach and camera corridors:

* parallel taxiway with connectors, yellow centrelines, hold-short markings and
  blue edge lights; concrete apron;
* three arched (barrel-roof) maintenance hangars, an MRO building with a gable
  roof, an operations building with a lit window band, a control tower with a
  glazed cab, fuel-farm tanks and apron floodlight masts;
* perimeter security fence (posts, top rail, tension wires and chain-link panels);
* approach-light crossbars raised on frangible masts (replacing ground fixtures);
* windsock, ILS localizer antenna array and glideslope antenna with shelters;
* a distant mountain range built from ridged multifractal noise on a polar grid,
  vertex-coloured by altitude and slope (forest, rock, snow).

Dynamic-lighting metadata (sequenced-flasher and PAPI unit positions, PAPI
angles) is stored on the scene as the ``alma_lighting`` custom property so the
glTF exporter can hand it to the WebGL runtime.
"""

from __future__ import annotations

import json
import math
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _material(name: str, color, metallic: float = 0.0, roughness: float = 0.6, emission=None, strength: float = 0.0,
              vertex_color: bool = False) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    if vertex_color:
        attr = nt.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = "Col"
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _box(bm, x0, x1, y0, y1, z0, z1):
    res = bmesh.ops.create_cube(bm, size=1.0)
    for v in res["verts"]:
        v.co = Vector(((x0 + x1) / 2 + v.co.x * (x1 - x0), (y0 + y1) / 2 + v.co.y * (y1 - y0), (z0 + z1) / 2 + v.co.z * (z1 - z0)))
    return res


def _cyl(bm, p0: Vector, p1: Vector, r0: float, r1: float | None = None, segments: int = 8):
    r1 = r0 if r1 is None else r1
    d = p1 - p0
    rot = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=r0, radius2=r1, depth=d.length,
                          matrix=Matrix.Translation((p0 + p1) * 0.5) @ rot)


class AirfieldBuilder:
    """Builds aerodrome infrastructure around the runway described by ``env`` (EnvironmentBuilder)."""

    TAXI_X = 95.0
    TAXI_W = 18.0
    APRON = (120.0, 300.0, 380.0, 1150.0)  # x0, x1, y0, y1

    def __init__(self, env) -> None:
        self.env = env
        self.coll = env.coll
        self.length = env.length
        self.width = env.width
        self.rng = random.Random(env.cfg.seed + 101)
        self.m = {
            "concrete": _material("Concrete_Apron", (0.36, 0.355, 0.34), 0.0, 0.82),
            "corrugated": _material("Corrugated_Metal", (0.55, 0.58, 0.6), 0.7, 0.42),
            "door": _material("Hangar_Door", (0.3, 0.34, 0.38), 0.55, 0.5),
            "wall": _material("Painted_Wall", (0.62, 0.6, 0.55), 0.0, 0.8),
            "roof": _material("Roof_Dark", (0.12, 0.12, 0.13), 0.1, 0.7),
            "window": _material("Window_Glow", (0.9, 0.7, 0.45), 0.0, 0.3, (1.0, 0.78, 0.5), 4.0),
            "tower_glass": _material("Tower_Glass", (0.03, 0.06, 0.07), 0.0, 0.05),
            "steel": _material("Galvanized_Steel", (0.5, 0.52, 0.53), 0.8, 0.5),
            "chainlink": _material("Fence_Mesh", (0.42, 0.44, 0.45), 0.7, 0.55),
            "windsock": _material("Windsock_Orange", (1.0, 0.33, 0.04), 0.0, 0.6, (1.0, 0.33, 0.04), 0.4),
            "taxi_paint": _material("Taxi_Paint_Yellow", (0.85, 0.62, 0.04), 0.0, 0.5),
            "taxi_light": _material("Taxi_Light_Blue", (0.1, 0.3, 1.0), 0.0, 0.2, (0.1, 0.3, 1.0), 30.0),
            "flood": _material("Floodlight_Lamp", (1.0, 0.92, 0.78), 0.0, 0.2, (1.0, 0.9, 0.75), 25.0),
            "tank": _material("Fuel_Tank_White", (0.78, 0.78, 0.76), 0.2, 0.5),
            "mountain": _material("Mountain_Vertex", (0.2, 0.2, 0.2), 0.0, 0.92, vertex_color=True),
        }

    # ------------------------------------------------------------------ helpers
    def ground(self, x: float, y: float) -> float:
        return self.env.terrain_height(x, y)

    def mesh(self, name: str, bm, mat, smooth: bool = False):
        me = bpy.data.meshes.new(name)
        bm.normal_update()
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        obj = bpy.data.objects.new(name, me)
        self.coll.objects.link(obj)
        return obj

    # -------------------------------------------------------------------- build
    def build(self, scene) -> None:
        self._taxiways()
        self._apron()
        self._hangars()
        self._mro_building()
        self._operations_building()
        self._control_tower()
        self._fuel_farm()
        self._floodlights()
        self._fence()
        self._navaids()
        self._mountains()
        scene["alma_lighting"] = json.dumps(self.lighting_manifest())

    # ------------------------------------------------------------ ground areas
    def _taxiways(self):
        L = self.length
        hw = self.TAXI_W / 2
        bm = bmesh.new()
        paint = bmesh.new()
        lights = []
        z = 0.012
        # Parallel taxiway.
        _box(bm, self.TAXI_X - hw, self.TAXI_X + hw, 40.0, L - 40.0, -0.05, z)
        for y in np.arange(40.0, L - 40.0, 3.0):
            _box(paint, self.TAXI_X - 0.08, self.TAXI_X + 0.08, y, y + 2.2, z, z + 0.004)
        for y in np.arange(60.0, L - 40.0, 30.0):
            lights += [(self.TAXI_X - hw - 0.8, y), (self.TAXI_X + hw + 0.8, y)]
        # Connectors with hold-short markings 45 m from the runway centreline.
        for yc in (150.0, 800.0, 1450.0):
            x0 = self.width / 2 + 7.5
            _box(bm, x0, self.TAXI_X - hw, yc - hw, yc + hw, -0.05, z)
            for xx in np.arange(x0, self.TAXI_X - hw, 3.0):
                _box(paint, xx, xx + 2.2, yc - 0.08, yc + 0.08, z, z + 0.004)
            hold = 45.0
            for k, off in enumerate((0.0, 0.45, 1.2, 1.65)):
                x = hold + off
                if k < 2:  # two solid lines
                    _box(paint, x, x + 0.15, yc - hw, yc + hw, z, z + 0.004)
                else:  # two dashed lines
                    for yy in np.arange(yc - hw, yc + hw, 1.8):
                        _box(paint, x, x + 0.15, yy, yy + 0.9, z, z + 0.004)
            for xx in np.arange(x0 + 15.0, self.TAXI_X - hw, 15.0):
                lights += [(xx, yc - hw - 0.8), (xx, yc + hw + 0.8)]
        self.mesh("Taxiways", bm, self.env.m.asphalt)
        self.mesh("Taxiway_Markings", paint, self.m["taxi_paint"])
        bulbs, bases = bmesh.new(), bmesh.new()
        for x, y in lights:
            bmesh.ops.create_icosphere(bulbs, subdivisions=1, radius=0.13, matrix=Matrix.Translation((x, y, 0.33)))
            _cyl(bases, Vector((x, y, 0.0)), Vector((x, y, 0.26)), 0.05, 0.04, 6)
        self.mesh("Taxiway_Lights", bulbs, self.m["taxi_light"], smooth=True)
        self.mesh("Taxiway_Light_Fixtures", bases, self.env.m.fixture)

    def _apron(self):
        x0, x1, y0, y1 = self.APRON
        bm = bmesh.new()
        nx, ny = int((x1 - x0) / 2.5), int((y1 - y0) / 2.5)
        bmesh.ops.create_grid(bm, x_segments=nx, y_segments=ny, size=0.5)
        for v in bm.verts:
            v.co.x = x0 + (v.co.x + 0.5) * (x1 - x0)
            v.co.y = y0 + (v.co.y + 0.5) * (y1 - y0)
            v.co.z = 0.02
        self.mesh("Apron", bm, self.m["concrete"])
        # Apron link to the taxiway.
        bm = bmesh.new()
        _box(bm, self.TAXI_X + self.TAXI_W / 2, x0, 520.0, 1080.0, -0.05, 0.015)
        self.mesh("Apron_Link", bm, self.m["concrete"])

    # --------------------------------------------------------------- buildings
    def _hangars(self):
        """Barrel-roof hangars: arch spans Y, depth along X, doors facing the apron (west)."""
        x_door, depth, span, rise, wall = 312.0, 52.0, 44.0, 9.0, 7.0
        shell, doors, ends = bmesh.new(), bmesh.new(), bmesh.new()
        segs = 24
        for yc in (450.0, 530.0, 610.0):
            base_z = 0.0
            arch = []
            for k in range(segs + 1):
                a = math.pi * k / segs
                y = yc - span / 2 * math.cos(a)
                z = base_z + wall + rise * math.sin(a)
                arch.append((y, z))
            # Side walls + roof as one extruded profile (outer shell), corrugation via material.
            profile = [(yc - span / 2, base_z)] + arch + [(yc + span / 2, base_z)]
            front = [shell.verts.new((x_door, y, z)) for y, z in profile]
            back = [shell.verts.new((x_door + depth, y, z)) for y, z in profile]
            for i in range(len(profile) - 1):
                shell.faces.new((front[i], front[i + 1], back[i + 1], back[i]))
            # Gable end walls (rear closed, front: frame around the door).
            rear = [ends.verts.new((x_door + depth, y, z)) for y, z in profile]
            ends.faces.new(list(reversed(rear)))
            door_h = wall + rise * 0.55
            frame = [ends.verts.new((x_door, y, z)) for y, z in profile]
            ends.faces.new(frame)
            _box(doors, x_door - 0.35, x_door - 0.05, yc - span / 2 + 2.0, yc + span / 2 - 2.0, 0.0, door_h)
            # Door panel seams.
            for k in range(1, 6):
                yy = yc - span / 2 + 2.0 + k * (span - 4.0) / 6
                _box(doors, x_door - 0.42, x_door - 0.34, yy - 0.06, yy + 0.06, 0.0, door_h)
        self.mesh("Hangar_Shells", shell, self.m["corrugated"])
        self.mesh("Hangar_Ends", ends, self.m["wall"])
        self.mesh("Hangar_Doors", doors, self.m["door"])

    def _gable(self, name_prefix, x0, x1, y0, y1, h, ridge, wall_mat, roof_mat):
        walls, roof = bmesh.new(), bmesh.new()
        _box(walls, x0, x1, y0, y1, 0.0, h)
        xm = (x0 + x1) / 2
        o = 0.6
        v = [roof.verts.new(p) for p in (
            (x0 - o, y0 - o, h), (xm, y0 - o, h + ridge), (x1 + o, y0 - o, h),
            (x0 - o, y1 + o, h), (xm, y1 + o, h + ridge), (x1 + o, y1 + o, h))]
        roof.faces.new((v[0], v[1], v[4], v[3]))
        roof.faces.new((v[1], v[2], v[5], v[4]))
        tri0 = [walls.verts.new(p) for p in ((x0, y0, h), (xm, y0, h + ridge), (x1, y0, h))]
        tri1 = [walls.verts.new(p) for p in ((x0, y1, h), (x1, y1, h), (xm, y1, h + ridge))]
        walls.faces.new(tri0)
        walls.faces.new(tri1)
        self.mesh(f"{name_prefix}_Walls", walls, wall_mat)
        self.mesh(f"{name_prefix}_Roof", roof, roof_mat)

    def _mro_building(self):
        self._gable("MRO", 310.0, 346.0, 680.0, 770.0, 10.0, 3.5, self.m["corrugated"], self.m["roof"])
        doors = bmesh.new()
        for yc in (700.0, 730.0, 755.0):
            _box(doors, 309.7, 310.0, yc - 6.0, yc + 6.0, 0.0, 7.0)
        self.mesh("MRO_Doors", doors, self.m["door"])

    def _operations_building(self):
        x0, x1, y0, y1, h = 305.0, 330.0, 820.0, 950.0, 8.5
        body, glow, roof = bmesh.new(), bmesh.new(), bmesh.new()
        _box(body, x0, x1, y0, y1, 0.0, h)
        _box(roof, x0 - 0.4, x1 + 0.4, y0 - 0.4, y1 + 0.4, h, h + 0.6)
        for z0 in (1.4, 5.2):  # two storeys of window bands on the apron facade
            for y in np.arange(y0 + 2.0, y1 - 2.0, 4.0):
                _box(glow, x0 - 0.06, x0 + 0.02, y, y + 3.0, z0, z0 + 1.6)
        self.mesh("Ops_Building", body, self.m["wall"])
        self.mesh("Ops_Roof", roof, self.m["roof"])
        self.mesh("Ops_Windows", glow, self.m["window"])

    def _control_tower(self):
        cx, cy = 232.0, 1215.0
        shaft, cab, roof, glow = bmesh.new(), bmesh.new(), bmesh.new(), bmesh.new()
        _cyl(shaft, Vector((cx, cy, 0.0)), Vector((cx, cy, 26.0)), 3.2, 2.6, 16)
        _box(shaft, cx - 7.0, cx + 7.0, cy - 5.0, cy + 5.0, 0.0, 5.0)  # base building
        _cyl(cab, Vector((cx, cy, 26.0)), Vector((cx, cy, 30.5)), 5.2, 5.8, 8)  # glazed cab
        _cyl(roof, Vector((cx, cy, 30.5)), Vector((cx, cy, 31.3)), 6.4, 6.4, 8)
        _cyl(roof, Vector((cx, cy, 31.3)), Vector((cx, cy, 38.0)), 0.08, 0.05, 6)  # antenna
        _cyl(roof, Vector((cx, cy, 25.4)), Vector((cx, cy, 26.0)), 6.0, 6.0, 8)  # catwalk slab
        _cyl(glow, Vector((cx, cy, 26.4)), Vector((cx, cy, 26.9)), 5.25, 5.4, 8)  # lit console band
        self.mesh("Tower_Shaft", shaft, self.m["wall"])
        self.mesh("Tower_Cab", cab, self.m["tower_glass"])
        self.mesh("Tower_Roof", roof, self.m["roof"])
        self.mesh("Tower_Console", glow, self.m["window"])
        beacon = bmesh.new()
        bmesh.ops.create_icosphere(beacon, subdivisions=2, radius=0.25, matrix=Matrix.Translation((cx, cy, 38.2)))
        self.mesh("Tower_Beacon", beacon, self.env.m.light_red, smooth=True)

    def _fuel_farm(self):
        bm = bmesh.new()
        for x, y in ((385.0, 990.0), (385.0, 1018.0), (408.0, 1004.0)):
            _cyl(bm, Vector((x, y, 0.0)), Vector((x, y, 8.0)), 9.0, 9.0, 24)
            _cyl(bm, Vector((x, y, 8.0)), Vector((x, y, 9.2)), 9.0, 2.0, 24)
        self.mesh("Fuel_Tanks", bm, self.m["tank"], smooth=True)

    def _floodlights(self):
        poles, heads = bmesh.new(), bmesh.new()
        x0, x1, y0, y1 = self.APRON
        for x in (x0 + 4.0, x1 - 4.0):
            for y in np.arange(y0 + 40.0, y1, 180.0):
                _cyl(poles, Vector((x, y, 0.0)), Vector((x, y, 22.0)), 0.35, 0.18, 10)
                _box(poles, x - 2.2, x + 2.2, y - 0.4, y + 0.4, 21.6, 22.0)
                for dx in (-1.6, -0.55, 0.55, 1.6):
                    _box(heads, x + dx - 0.4, x + dx + 0.4, y - 0.3, y + 0.3, 21.0, 21.6)
        self.mesh("Floodlight_Masts", poles, self.m["steel"])
        self.mesh("Floodlight_Heads", heads, self.m["flood"])

    # ------------------------------------------------------------------- fence
    def _fence(self):
        """Perimeter security fence. The fence line lies entirely on the flat aerodrome ground, so
        rails, wires and the chain-link panel run continuously along each side (cheap geometry);
        posts stand every 4 m."""
        L = self.length
        corners = [(-140.0, -1150.0), (480.0, -1150.0), (480.0, L + 350.0), (-140.0, L + 350.0)]
        posts, rails, mesh = bmesh.new(), bmesh.new(), bmesh.new()
        h = 2.4
        z0 = self.ground(corners[0][0], corners[0][1])
        for (xa, ya), (xb, yb) in zip(corners, corners[1:] + corners[:1]):
            a, b = Vector((xa, ya, z0)), Vector((xb, yb, z0))
            length = (b - a).length
            d = (b - a).normalized()
            n = int(length / 4.0)
            for k in range(n + 1):
                p = a + d * (k * length / n)
                _cyl(posts, p - Vector((0.0, 0.0, 0.2)), p + Vector((0.0, 0.0, h + 0.35)), 0.04, 0.04, 4)
            for zz, r in ((h, 0.025), (h * 0.66, 0.006), (h * 0.33, 0.006), (0.15, 0.006)):
                _cyl(rails, a + Vector((0.0, 0.0, zz)), b + Vector((0.0, 0.0, zz)), r, r, 4)
            out = Vector((-d.y, d.x, 0.0)) * 0.3  # barbed-wire outrigger strands
            for s_off, zz in ((0.0, h + 0.15), (1.0, h + 0.28)):
                _cyl(rails, a + out * s_off + Vector((0.0, 0.0, zz)), b + out * s_off + Vector((0.0, 0.0, zz)), 0.005, 0.005, 3)
            # Chain-link: one panel per side; the exporter tiles an alpha-mask texture along it.
            q = [mesh.verts.new(v) for v in (a + Vector((0, 0, 0.05)), b + Vector((0, 0, 0.05)), b + Vector((0, 0, h)), a + Vector((0, 0, h)))]
            mesh.faces.new(q)
        self.mesh("Fence_Posts", posts, self.m["steel"])
        self.mesh("Fence_Rails", rails, self.m["steel"])
        self.mesh("Fence_Mesh", mesh, self.m["chainlink"])

    # ----------------------------------------------------------------- navaids
    def _navaids(self):
        L = self.length
        steel, sock, shelter = bmesh.new(), bmesh.new(), bmesh.new()
        # Windsock near the threshold, left of the runway.
        wx, wy = -42.0, 140.0
        _cyl(steel, Vector((wx, wy, 0.0)), Vector((wx, wy, 7.0)), 0.09, 0.06, 8)
        _cyl(steel, Vector((wx, wy, 7.0)), Vector((wx, wy, 7.0)) + Vector((0.0, 0.0, 0.01)), 0.5, 0.5, 12)
        # Sock: truncated cone along -X (blowing towards the west in the prevailing crosswind).
        _cyl(sock, Vector((wx, wy, 6.9)), Vector((wx - 3.6, wy, 6.5)), 0.45, 0.2, 16)
        # ILS localizer array beyond the stop end.
        ly = L + 260.0
        for x in np.arange(-15.0, 15.1, 2.5):
            _cyl(steel, Vector((x, ly, 0.0)), Vector((x, ly, 3.2)), 0.05, 0.05, 6)
            _box(steel, x - 0.9, x + 0.9, ly - 0.05, ly + 0.05, 2.6, 2.7)
            _box(steel, x - 0.9, x + 0.9, ly - 0.05, ly + 0.05, 1.6, 1.7)
        _box(steel, -15.5, 15.5, ly - 0.1, ly + 0.1, 0.4, 0.5)
        _box(shelter, 20.0, 26.0, ly - 2.5, ly + 2.5, 0.0, 2.8)
        # Glideslope antenna abeam the aim point.
        gx, gy = -125.0, self.env.aim
        _cyl(steel, Vector((gx, gy, 0.0)), Vector((gx, gy, 16.0)), 0.18, 0.12, 8)
        for z in (5.0, 10.0, 15.0):
            _box(steel, gx - 0.15, gx + 0.15, gy - 1.1, gy + 1.1, z, z + 0.9)
        _box(shelter, gx - 9.0, gx - 4.0, gy - 2.0, gy + 2.0, 0.0, 2.6)
        self.mesh("Navaid_Masts", steel, self.m["steel"])
        self.mesh("Windsock", sock, self.m["windsock"], smooth=True)
        self.mesh("Navaid_Shelters", shelter, self.m["wall"])

    # --------------------------------------------------------------- mountains
    def _mountains(self):
        cx, cy = 0.0, self.length / 2
        r_in, r_out = 4600.0, 12500.0
        n_a, n_r = 240, 28
        bm = bmesh.new()
        col = bm.loops.layers.color.new("Col")
        grid = []
        heights = []
        for j in range(n_r + 1):
            u = j / n_r
            r = r_in + (r_out - r_in) * (u ** 0.8)
            row = []
            hrow = []
            for i in range(n_a):
                th = 2.0 * math.pi * i / n_a
                x, y = cx + r * math.cos(th), cy + r * math.sin(th)
                # Range is highest to the north and east (behind the runway end / airfield side).
                sector = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(th + 0.6))
                env_h = _smoothstep(r_in, r_in + 2600.0, r) * (1.0 - 0.3 * _smoothstep(r_out - 2500.0, r_out, r))
                p = Vector((x / 2600.0, y / 2600.0, 3.1))
                ridged, amp, freq = 0.0, 1.0, 1.0
                for _ in range(6):
                    nval = 1.0 - abs(noise.noise(p * freq))
                    ridged += amp * nval * nval
                    amp *= 0.5
                    freq *= 2.05
                z = -40.0 + env_h * sector * 1250.0 * (ridged / 1.97)
                row.append(bm.verts.new((x, y, z)))
                hrow.append(z)
            grid.append(row)
            heights.append(hrow)
        for j in range(n_r):
            for i in range(n_a):
                i2 = (i + 1) % n_a
                bm.faces.new((grid[j][i], grid[j][i2], grid[j + 1][i2], grid[j + 1][i]))
        bm.normal_update()
        for f in bm.faces:
            slope = 1.0 - abs(f.normal.z)
            for loop in f.loops:
                z = loop.vert.co.z
                n = 0.5 + 0.5 * noise.noise(loop.vert.co * 0.004)
                forest = np.array([0.03, 0.055, 0.028])
                rock = np.array([0.2, 0.19, 0.18]) * (0.8 + 0.4 * n)
                snow = np.array([0.88, 0.9, 0.93])
                w_rock = _smoothstep(260.0, 520.0, z + 120.0 * (n - 0.5)) if slope < 0.55 else 1.0
                c = forest * (1 - w_rock) + rock * w_rock
                w_snow = _smoothstep(820.0, 980.0, z + 150.0 * (n - 0.5)) * (1.0 - _smoothstep(0.45, 0.7, slope))
                c = c * (1 - w_snow) + snow * w_snow
                loop[col] = (*c, 1.0)
        obj = self.mesh("Mountains", bm, self.m["mountain"], smooth=True)
        obj.data.color_attributes.active_color = obj.data.color_attributes["Col"]

    # ----------------------------------------------------------------- lighting
    def lighting_manifest(self) -> dict:
        """Positions (ENU, metres) for the WebGL dynamic lights."""
        rabbit = [[0.0, -30.0 * k, self.env.als_height(k) + 0.32] for k in range(30, 0, -1)]  # far -> near
        papi = []
        angles = (3.5, 3.1667, 2.8333, 2.5)  # nearest unit has the highest setting angle
        for k, ang in enumerate(angles):
            x = -(self.width / 2 + 15.0 + 9.0 * k)
            papi.append({"position": [x, self.env.aim - 0.42, 0.42], "angle_deg": ang})
        return {"approach_flashers": rabbit, "flash_hz": 2.0, "papi": papi,
                "aim_point_m": self.env.aim, "glideslope_deg": math.degrees(self.env.gs)}
