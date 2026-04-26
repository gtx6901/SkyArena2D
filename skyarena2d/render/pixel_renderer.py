from __future__ import annotations

import math

import numpy as np

from ..core.config import EnvConfig
from ..core.state import EnvState
from . import palette
from .sprites import draw_cross, draw_detector, draw_fighter, draw_heading_tick

try:
    import pygame
except Exception:  # pragma: no cover - fallback path is tested through rgb_array
    pygame = None


class PixelRenderer:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.render_width = int(config.render.width)
        self.render_height = int(config.render.height)
        self.debug_overlay = bool(config.render.debug_overlay)
        self.scoreboard_height = 80
        self._window = None
        self._clock = None
        self._pygame_ready = False

    def toggle_debug_overlay(self) -> None:
        self.debug_overlay = not self.debug_overlay

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = int(np.clip((x / max(1.0, self.config.map.width)) * self.render_width, 0, self.render_width - 1))
        map_render_height = self.render_height - self.scoreboard_height
        sy = self.scoreboard_height + int(
            np.clip(
                (y / max(1.0, self.config.map.height)) * map_render_height,
                0,
                map_render_height - 1,
            )
        )
        return sx, sy

    def _ensure_pygame(self) -> None:
        if pygame is None:
            raise RuntimeError("pygame-ce is required for human rendering")
        if self._pygame_ready:
            return
        pygame.init()
        pygame.display.init()
        self._clock = pygame.time.Clock()
        self._pygame_ready = True

    def _draw_grid(self, surface) -> None:
        if pygame is None:
            return
        step = 40
        for x in range(0, self.render_width, step):
            pygame.draw.line(surface, palette.GRID, (x, self.scoreboard_height), (x, self.render_height), 1)
        for y in range(self.scoreboard_height, self.render_height, step):
            pygame.draw.line(surface, palette.GRID, (0, y), (self.render_width, y), 1)

    @staticmethod
    def _draw_dashed_line(
        surface,
        color: tuple[int, int, int],
        p1: tuple[int, int],
        p2: tuple[int, int],
        dash_len: int = 6,
        gap_len: int = 4,
        width: int = 1,
    ) -> None:
        """Draw a dashed line from p1 to p2."""
        if pygame is None:
            return
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        ux = dx / dist
        uy = dy / dist
        pos = 0.0
        drawing = True
        while pos < dist:
            seg_len = dash_len if drawing else gap_len
            seg_end = min(pos + seg_len, dist)
            if drawing:
                start_pt = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
                end_pt = (int(p1[0] + ux * seg_end), int(p1[1] + uy * seg_end))
                pygame.draw.line(surface, color, start_pt, end_pt, width)
            pos = seg_end
            drawing = not drawing

    def _draw_cone(
        self,
        surface,
        center: tuple[int, int],
        heading_deg: float,
        fov_deg: float,
        radius_px: int,
        color_rgba: tuple[int, int, int, int],
    ) -> None:
        if pygame is None or radius_px <= 1:
            return
        overlay = pygame.Surface((self.render_width, self.render_height), pygame.SRCALPHA)
        if fov_deg >= 359.9:
            pygame.draw.circle(overlay, color_rgba, center, radius_px, 1)
        else:
            steps = max(6, int(fov_deg // 8))
            start = heading_deg - fov_deg * 0.5
            pts = [center]
            for i in range(steps + 1):
                a = math.radians(start + (fov_deg * i / steps))
                px = int(center[0] + math.cos(a) * radius_px)
                py = int(center[1] + math.sin(a) * radius_px)
                pts.append((px, py))
            pygame.draw.polygon(overlay, color_rgba, pts)
        surface.blit(overlay, (0, 0))

    def _draw_jammer(self, surface, center: tuple[int, int], radius_px: int, color: tuple[int, int, int]) -> None:
        if pygame is None or radius_px <= 1:
            return
        segments = 24
        for i in range(segments):
            if i % 2 == 0:
                continue
            a0 = (2 * math.pi * i) / segments
            a1 = (2 * math.pi * (i + 1)) / segments
            p0 = (int(center[0] + math.cos(a0) * radius_px), int(center[1] + math.sin(a0) * radius_px))
            p1 = (int(center[0] + math.cos(a1) * radius_px), int(center[1] + math.sin(a1) * radius_px))
            pygame.draw.line(surface, color, p0, p1, 1)

    def _draw_edges(self, surface, state: EnvState) -> None:
        if pygame is None or state.cache is None:
            return

        if self.config.render.show_visible_edges:
            red_vis = np.argwhere(state.cache.red_visible)
            for i, j in red_vis.tolist():
                p1 = self._world_to_screen(float(state.red.pos[i, 0]), float(state.red.pos[i, 1]))
                p2 = self._world_to_screen(float(state.blue.pos[j, 0]), float(state.blue.pos[j, 1]))
                pygame.draw.line(surface, palette.EDGE_VISIBLE, p1, p2, 1)

            blue_vis = np.argwhere(state.cache.blue_visible)
            for i, j in blue_vis.tolist():
                p1 = self._world_to_screen(float(state.blue.pos[i, 0]), float(state.blue.pos[i, 1]))
                p2 = self._world_to_screen(float(state.red.pos[j, 0]), float(state.red.pos[j, 1]))
                pygame.draw.line(surface, palette.EDGE_VISIBLE, p1, p2, 1)

        if self.config.render.show_fireable_edges:
            red_fire = np.argwhere(state.cache.red_fireable_long | state.cache.red_fireable_short)
            for i, j in red_fire.tolist():
                p1 = self._world_to_screen(float(state.red.pos[i, 0]), float(state.red.pos[i, 1]))
                p2 = self._world_to_screen(float(state.blue.pos[j, 0]), float(state.blue.pos[j, 1]))
                self._draw_dashed_line(surface, palette.EDGE_FIREABLE, p1, p2)

            blue_fire = np.argwhere(state.cache.blue_fireable_long | state.cache.blue_fireable_short)
            for i, j in blue_fire.tolist():
                p1 = self._world_to_screen(float(state.blue.pos[i, 0]), float(state.blue.pos[i, 1]))
                p2 = self._world_to_screen(float(state.red.pos[j, 0]), float(state.red.pos[j, 1]))
                self._draw_dashed_line(surface, palette.EDGE_FIREABLE, p1, p2)

    def _draw_missiles(self, surface, state: EnvState) -> None:
        if pygame is None or state.cache is None or not self.config.render.show_target_allocations:
            return
        for launch in state.cache.launch_records:
            if not launch.valid:
                continue
            if launch.attacker_side == "red":
                src = state.red.pos[launch.attacker_idx]
                dst = state.blue.pos[launch.target_idx]
                color = palette.MISSILE_LONG if launch.missile_type == "long" else palette.MISSILE_SHORT
            else:
                src = state.blue.pos[launch.attacker_idx]
                dst = state.red.pos[launch.target_idx]
                color = palette.MISSILE_LONG if launch.missile_type == "long" else palette.MISSILE_SHORT
            p1 = self._world_to_screen(float(src[0]), float(src[1]))
            p2 = self._world_to_screen(float(dst[0]), float(dst[1]))
            pygame.draw.line(surface, color, p1, p2, 2)
            pygame.draw.circle(surface, color, p2, 3)

    def _draw_team(self, surface, state: EnvState, side: str) -> None:
        if pygame is None:
            return
        team = state.red if side == "red" else state.blue
        color = palette.RED if side == "red" else palette.BLUE
        cone = palette.RED_SOFT if side == "red" else palette.BLUE_SOFT

        for i in range(team.total_units):
            x, y = self._world_to_screen(float(team.pos[i, 0]), float(team.pos[i, 1]))
            if not team.alive[i]:
                draw_cross(surface, x, y, 4, palette.WRECK)
                continue

            if self.debug_overlay and self.config.render.show_radar and team.radar_on[i]:
                range_px = int((team.radar_range[i] / max(1.0, self.config.map.width)) * self.render_width)
                self._draw_cone(surface, (x, y), float(team.heading[i]), float(team.radar_fov_deg[i]), range_px, cone)

            if self.debug_overlay and self.config.render.show_jamming and team.jammer_on[i]:
                range_px = int((team.jammer_range[i] / max(1.0, self.config.map.width)) * self.render_width)
                self._draw_jammer(surface, (x, y), range_px, color)

            if team.unit_type[i] == 0:
                draw_fighter(surface, x, y, float(team.heading[i]), color, size=7)
            else:
                draw_detector(surface, x, y, color, radius=5)
            draw_heading_tick(surface, x, y, float(team.heading[i]), 10, palette.WHITE)

    def _draw_scoreboard(self, surface, state: EnvState, metrics: dict | None) -> None:
        if pygame is None:
            return
        # Dark background bar
        pygame.draw.rect(surface, palette.SCOREBOARD_BG, (0, 0, self.render_width, self.scoreboard_height))

        try:
            font = pygame.font.SysFont(None, 20)
        except Exception:
            return

        max_steps = getattr(self.config, "max_steps", 2000)

        # Row 1: step, winner, alive counts, kills
        red_alive = state.red.alive_count
        red_total = state.red.num_fighters
        blue_alive = state.blue.alive_count
        blue_total = state.blue.num_fighters
        red_kills = state.red.kills
        blue_kills = state.blue.kills
        red_missiles = state.red.remaining_missiles()
        blue_missiles = state.blue.remaining_missiles()

        # Row 2: attempt / selected / invalid counts
        if metrics is not None:
            red_attempt = metrics.get("red_attempted_edges", "---")
            blue_attempt = metrics.get("blue_attempted_edges", "---")
            red_select = metrics.get("red_selected_edges", "---")
            blue_select = metrics.get("blue_selected_edges", "---")
            red_invalid = metrics.get("red_invalid_fire_count", "---")
            blue_invalid = metrics.get("blue_invalid_fire_count", "---")
        else:
            red_attempt = blue_attempt = "---"
            red_select = blue_select = "---"
            red_invalid = blue_invalid = "---"

        # Row 3: detailed metrics
        if metrics is not None:
            red_fire_edges = metrics.get("red_fireable_edges", "---")
            blue_fire_edges = metrics.get("blue_fireable_edges", "---")
            exch = metrics.get("expected_exchange_proxy", None)
            exch_str = f"{exch:+.1f}" if exch is not None else "---"
            sel_exch = metrics.get("selected_expected_exchange", None)
            sel_exch_str = f"{sel_exch:+.1f}" if sel_exch is not None else "---"
            red_exec = metrics.get("red_fire_execution_rate_given_opportunity", None)
            blue_exec = metrics.get("blue_fire_execution_rate_given_opportunity", None)
            red_exec_str = f"{red_exec:.2f}" if red_exec is not None else "---"
            blue_exec_str = f"{blue_exec:.2f}" if blue_exec is not None else "---"
            contact = metrics.get("first_contact_step", None)
            contact_str = str(contact) if contact is not None else "---"
            fire_opp = metrics.get("first_fire_opportunity_step", None)
            fire_opp_str = str(fire_opp) if fire_opp is not None else "---"
        else:
            red_fire_edges = blue_fire_edges = "---"
            exch_str = sel_exch_str = "---"
            red_exec_str = blue_exec_str = "---"
            contact_str = fire_opp_str = "---"

        W = palette.SCOREBOARD_TEXT
        R = palette.SCOREBOARD_RED
        B = palette.SCOREBOARD_BLUE

        # Build row 1 as segments: (text, color)
        row1_segments = [
            (f"step={state.step_count}/{max_steps}  winner={state.winner}  ", W),
            (f"RED:{red_alive}/{red_total}", R),
            ("  ", W),
            (f"BLUE:{blue_alive}/{blue_total}", B),
            ("  kills R:", W),
            (f"{red_kills}", R),
            (" B:", W),
            (f"{blue_kills}", B),
            ("  missiles R:", W),
            (f"{red_missiles}", R),
            (" B:", W),
            (f"{blue_missiles}", B),
        ]

        row2_segments = [
            ("attempt R:", W),
            (f"{red_attempt}", R),
            (" B:", W),
            (f"{blue_attempt}", B),
            ("  selected R:", W),
            (f"{red_select}", R),
            (" B:", W),
            (f"{blue_select}", B),
            ("  invalid R:", W),
            (f"{red_invalid}", R),
            (" B:", W),
            (f"{blue_invalid}", B),
        ]

        row3_segments = [
            ("fireable R:", W),
            (f"{red_fire_edges}", R),
            (" B:", W),
            (f"{blue_fire_edges}", B),
            (f"  exch={exch_str}  sel_exch={sel_exch_str}  exec R:", W),
            (f"{red_exec_str}", R),
            (" B:", W),
            (f"{blue_exec_str}", B),
            (f"  contact={contact_str}  fire_opp={fire_opp_str}", W),
        ]

        def render_row(segments, y_offset: int) -> None:
            x = 4
            for text, color in segments:
                surf = font.render(text, True, color)
                surface.blit(surf, (x, y_offset))
                x += surf.get_width()

        render_row(row1_segments, 6)
        render_row(row2_segments, 24)
        render_row(row3_segments, 42)

    def _render_surface(self, state: EnvState, metrics: dict | None = None):
        if pygame is None:
            return None
        surface = pygame.Surface((self.render_width, self.render_height))
        surface.fill(palette.BACKGROUND)
        # Fill scoreboard area with its own background
        pygame.draw.rect(surface, palette.SCOREBOARD_BG, (0, 0, self.render_width, self.scoreboard_height))
        self._draw_grid(surface)
        self._draw_team(surface, state, "red")
        self._draw_team(surface, state, "blue")

        if self.debug_overlay:
            self._draw_edges(surface, state)
            self._draw_missiles(surface, state)

        self._draw_scoreboard(surface, state, metrics)
        return surface

    def _fallback_rgb(self, state: EnvState) -> np.ndarray:
        h, w = self.render_height, self.render_width
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Scoreboard area: dark color
        frame[: self.scoreboard_height, :, 0] = palette.SCOREBOARD_BG[0]
        frame[: self.scoreboard_height, :, 1] = palette.SCOREBOARD_BG[1]
        frame[: self.scoreboard_height, :, 2] = palette.SCOREBOARD_BG[2]

        # Map area: background color
        frame[self.scoreboard_height :, :, 0] = palette.BACKGROUND[0]
        frame[self.scoreboard_height :, :, 1] = palette.BACKGROUND[1]
        frame[self.scoreboard_height :, :, 2] = palette.BACKGROUND[2]

        # Grid lines in map area only
        for x in range(0, w, 40):
            frame[self.scoreboard_height :, x : x + 1, :] = np.array(palette.GRID, dtype=np.uint8)
        for y in range(self.scoreboard_height, h, 40):
            frame[y : y + 1, :, :] = np.array(palette.GRID, dtype=np.uint8)

        def draw_dot(px: int, py: int, color: tuple[int, int, int]) -> None:
            for oy in range(-2, 3):
                for ox in range(-2, 3):
                    xx = np.clip(px + ox, 0, w - 1)
                    yy = np.clip(py + oy, 0, h - 1)
                    frame[yy, xx] = np.array(color, dtype=np.uint8)

        for team, color in ((state.red, palette.RED), (state.blue, palette.BLUE)):
            for i in range(team.total_units):
                if not team.alive[i]:
                    continue
                x, y = self._world_to_screen(float(team.pos[i, 0]), float(team.pos[i, 1]))
                draw_dot(x, y, color)

        return frame

    def render(self, state: EnvState, mode: str = "rgb_array", metrics: dict | None = None) -> np.ndarray | None:
        if mode not in {"human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {mode}")

        if pygame is None:
            if mode == "human":
                raise RuntimeError("pygame-ce is required for human mode")
            return self._fallback_rgb(state)

        self._ensure_pygame()
        surface = self._render_surface(state, metrics)
        assert surface is not None

        if mode == "human":
            if self._window is None:
                self._window = pygame.display.set_mode((self.render_width, self.render_height))
                pygame.display.set_caption("SkyArena2D")
            pygame.event.pump()
            self._window.blit(surface, (0, 0))
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(60)
            return None

        arr = pygame.surfarray.array3d(surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self) -> None:
        if pygame is not None and self._pygame_ready:
            if self._window is not None:
                pygame.display.quit()
                self._window = None
            pygame.quit()
            self._pygame_ready = False
