# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that Newton collides the body pairs you expect, and how deep they sink.

A spec is ``"<regex A>:<regex B>"`` matched (``re.search``) against Newton
body labels (USD prim paths). Every body pair (a, b) with a in A and b in B
should collide.

The static audit runs once, after the model is finalized. For each expected
body pair it counts the shape pairs in ``model.shape_contact_pairs`` (the
explicit broad phase collides only these). A pair that is missing gets a
reason: collision group, world, or a filter pair. Bodies in A or B with no
collision shape are listed too. Visual-only shapes (COLLIDE_SHAPES off) are
ignored.

The runtime monitor reads the Newton contact buffer after each env step. Its
depth is the signed gap between the contact points, less both surface
thicknesses. Negative is penetration. Contacts come from the last substep's
collide(); body poses from the current state, so depths are off by at most one
substep of motion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_ENV_RE = re.compile(r"/env_(\d+)/")


def _short(label: str) -> str:
    """Drop the ``/World/envs/env_N/`` prefix."""
    match = _ENV_RE.search(label)
    return label[match.end() :] if match else label


def _env_of(label: str) -> int | None:
    match = _ENV_RE.search(label)
    return int(match.group(1)) if match else None


def parse_spec(spec: str) -> tuple[re.Pattern, re.Pattern]:
    if ":" not in spec:
        raise ValueError(f"--expect-collide '{spec}' must look like '<regex A>:<regex B>'")
    left, right = spec.split(":", 1)
    return re.compile(left), re.compile(right)


def _group_pair(a: int, b: int) -> bool:
    """Newton's ``test_group_pair``."""
    if a == 0 or b == 0:
        return False
    if a > 0:
        return a == b or b < 0
    return a != b


@dataclass
class _PairStats:
    contacts: int = 0
    steps: int = 0
    steps_over_tol: int = 0
    last_step_over_tol: int = -1
    worst_depth: float = float("inf")
    worst_step: int = -1
    worst_shapes: tuple[str, str] = ("", "")


@dataclass
class CollisionAudit:
    specs: list[str]
    env_index: int = 0
    tolerance: float = 0.005
    verbose: bool = False
    _bodies_a: set[int] = field(default_factory=set)
    _bodies_b: set[int] = field(default_factory=set)
    _stats: dict[tuple[int, int], _PairStats] = field(default_factory=dict)
    _warned: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self):
        from isaaclab_newton.physics import NewtonManager

        self.model = NewtonManager.get_model()
        m = self.model
        self.body_label = list(m.body_label)
        self.shape_label = list(m.shape_label)
        self.shape_body = m.shape_body.numpy()
        self.shape_flags = m.shape_flags.numpy()
        self.shape_group = m.shape_collision_group.numpy()
        self.shape_world = m.shape_world.numpy()
        self.shape_type = m.shape_type.numpy()
        self.shape_gap = m.shape_gap.numpy() if m.shape_gap is not None else np.zeros(len(self.shape_label))
        pairs = m.shape_contact_pairs.numpy().reshape(-1, 2) if m.shape_contact_pairs is not None else []
        self.contact_pairs = {(int(min(a, b)), int(max(a, b))) for a, b in pairs}
        self.filter_pairs = {(min(a, b), max(a, b)) for a, b in m.shape_collision_filter_pairs}
        self.patterns = [parse_spec(s) for s in self.specs]

    # ------------------------------------------------------------------ static

    def _env_bodies(self, pattern: re.Pattern) -> list[int]:
        out = []
        for i, label in enumerate(self.body_label):
            env = _env_of(label)
            if env is not None and env != self.env_index:
                continue
            if pattern.search(label):
                out.append(i)
        return out

    def _shapes_of(self, body: int) -> list[int]:
        """Collision shapes only; visual-only meshes have COLLIDE_SHAPES off by design."""
        from newton import ShapeFlags

        collide = (self.shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        return [int(s) for s in np.nonzero((self.shape_body == body) & collide)[0]]

    def _why_not(self, s0: int, s1: int) -> str:
        g0, g1 = int(self.shape_group[s0]), int(self.shape_group[s1])
        if not _group_pair(g0, g1):
            return f"collision groups {g0} vs {g1}"
        w0, w1 = int(self.shape_world[s0]), int(self.shape_world[s1])
        if w0 != -1 and w1 != -1 and w0 != w1:
            return f"worlds {w0} vs {w1}"
        if (min(s0, s1), max(s0, s1)) in self.filter_pairs:
            return "explicit filter pair (e.g. joint parent/child or disabled self-collision)"
        return "not in shape_contact_pairs (unknown reason)"

    def _geo(self, s: int) -> str:
        from newton import GeoType

        try:
            return GeoType(int(self.shape_type[s])).name
        except ValueError:
            return str(int(self.shape_type[s]))

    def report_static(self) -> bool:
        """Print the pair table. Returns False when an expected pair cannot collide."""
        ok = True
        print("[COLLISION-AUDIT] ===== static pair check =====", flush=True)
        for spec, (pa, pb) in zip(self.specs, self.patterns):
            bodies_a, bodies_b = self._env_bodies(pa), self._env_bodies(pb)
            self._bodies_a.update(bodies_a)
            self._bodies_b.update(bodies_b)
            print(f"[COLLISION-AUDIT] spec '{spec}': {len(bodies_a)} bodies in A, {len(bodies_b)} in B")
            if not bodies_a or not bodies_b:
                print("[COLLISION-AUDIT]   !! a side matched no body. Body labels look like:")
                for label in self.body_label[:: max(1, len(self.body_label) // 15)]:
                    print(f"[COLLISION-AUDIT]      {label}")
                ok = False
                continue
            for side, bodies in (("A", bodies_a), ("B", bodies_b)):
                bare = [_short(self.body_label[b]) for b in bodies if not self._shapes_of(b)]
                if bare:
                    print(f"[COLLISION-AUDIT]   {side} bodies with no collision shape (pass through anything): {bare}")
                if self.verbose:
                    for b in bodies:
                        shapes = self._shapes_of(b)
                        geos = sorted({self._geo(s) for s in shapes})
                        print(f"[COLLISION-AUDIT]   {side} {_short(self.body_label[b])}: {len(shapes)} {geos}")
            for a in bodies_a:
                for b in bodies_b:
                    if a == b:
                        continue
                    sa, sb = self._shapes_of(a), self._shapes_of(b)
                    if not sa or not sb:
                        continue
                    total = len(sa) * len(sb)
                    enabled = sum((min(x, y), max(x, y)) in self.contact_pairs for x in sa for y in sb)
                    if enabled == total:
                        continue
                    ok = ok and enabled > 0
                    reasons: dict[str, int] = {}
                    for x in sa:
                        for y in sb:
                            if (min(x, y), max(x, y)) not in self.contact_pairs:
                                r = self._why_not(x, y)
                                reasons[r] = reasons.get(r, 0) + 1
                    mark = "!!" if enabled == 0 else "~ "
                    print(
                        f"[COLLISION-AUDIT]   {mark} {_short(self.body_label[a])} x {_short(self.body_label[b])}: "
                        f"{enabled}/{total} shape pairs collide; missing: {reasons}"
                    )
        print(f"[COLLISION-AUDIT] static result: {'OK' if ok else 'PROBLEMS FOUND'}", flush=True)
        return ok

    # ----------------------------------------------------------------- runtime

    def step(self, step: int) -> None:
        from isaaclab_newton.physics import NewtonManager

        contacts = NewtonManager.get_contacts()
        if contacts is None:
            return
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n == 0:
            return
        s0 = contacts.rigid_contact_shape0.numpy()[:n]
        s1 = contacts.rigid_contact_shape1.numpy()[:n]
        b0, b1 = self.shape_body[s0], self.shape_body[s1]
        a_set, b_set = self._bodies_a, self._bodies_b
        keep = np.array(
            [(x in a_set and y in b_set) or (x in b_set and y in a_set) for x, y in zip(b0, b1)], dtype=bool
        )
        if not keep.any():
            return
        idx = np.nonzero(keep)[0]
        body_q = NewtonManager.get_state_0().body_q.numpy()
        p0 = contacts.rigid_contact_point0.numpy()[idx]
        p1 = contacts.rigid_contact_point1.numpy()[idx]
        nrm = contacts.rigid_contact_normal.numpy()[idx]
        m0 = contacts.rigid_contact_margin0.numpy()[idx]
        m1 = contacts.rigid_contact_margin1.numpy()[idx]
        w0 = _to_world(body_q, b0[idx], p0)
        w1 = _to_world(body_q, b1[idx], p1)
        depth = np.einsum("ij,ij->i", nrm, w1 - w0) - (m0 + m1)

        seen: set[tuple[int, int]] = set()
        for k, i in enumerate(idx):
            x, y = int(b0[i]), int(b1[i])
            key = (x, y) if x in a_set else (y, x)
            st = self._stats.setdefault(key, _PairStats())
            st.contacts += 1
            if key not in seen:
                st.steps += 1
                seen.add(key)
            if depth[k] < st.worst_depth:
                st.worst_depth = float(depth[k])
                st.worst_step = step
                st.worst_shapes = (_short(self.shape_label[s0[i]]), _short(self.shape_label[s1[i]]))
            if depth[k] < -self.tolerance and st.last_step_over_tol != step:
                st.steps_over_tol += 1
                st.last_step_over_tol = step
            if depth[k] < -self.tolerance and key not in self._warned:
                self._warned.add(key)
                print(
                    f"[COLLISION-AUDIT] step={step} PENETRATION {-depth[k] * 1000:.1f} mm "
                    f"{_short(self.body_label[key[0]])} x {_short(self.body_label[key[1]])} "
                    f"(shapes {st.worst_shapes[0]} / {st.worst_shapes[1]})",
                    flush=True,
                )

    def report_runtime(self) -> None:
        print("[COLLISION-AUDIT] ===== runtime contacts (A x B) =====")
        if not self._stats:
            print("[COLLISION-AUDIT] no A x B contact was ever generated.")
            print("[COLLISION-AUDIT] If the parts visibly overlap, the collision shapes are elsewhere:")
            print("[COLLISION-AUDIT] run again with --visualizer newton --collision-only.")
            return
        rows = sorted(self._stats.items(), key=lambda kv: kv[1].worst_depth)
        for (a, b), st in rows:
            flag = "!!" if st.worst_depth < -self.tolerance else "  "
            print(
                f"[COLLISION-AUDIT] {flag} {_short(self.body_label[a])} x {_short(self.body_label[b])}: "
                f"steps={st.steps} (>{self.tolerance * 1000:.0f} mm: {st.steps_over_tol}) contacts={st.contacts} worst={st.worst_depth * 1000:+.1f} mm "
                f"@step {st.worst_step} ({st.worst_shapes[0]} / {st.worst_shapes[1]})"
            )
        print(f"[COLLISION-AUDIT] depth < 0 is penetration; flagged beyond {self.tolerance * 1000:.1f} mm.", flush=True)


def _to_world(body_q: np.ndarray, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Body-frame points to world. ``body_q`` rows are (px, py, pz, qx, qy, qz, qw); body -1 is world."""
    out = points.copy()
    mask = bodies >= 0
    if not mask.any():
        return out
    q = body_q[bodies[mask]]
    pos, quat = q[:, :3], q[:, 3:]
    v = points[mask]
    u, w = quat[:, :3], quat[:, 3:4]
    t = 2.0 * np.cross(u, v)
    out[mask] = pos + v + w * t + np.cross(u, t)
    return out
