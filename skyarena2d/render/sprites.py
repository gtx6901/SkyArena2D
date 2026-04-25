from __future__ import annotations

import math


def _rotated_points(points: list[tuple[float, float]], angle_deg: float) -> list[tuple[float, float]]:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def draw_fighter(surface, x: int, y: int, heading_deg: float, color: tuple[int, ...], size: int) -> None:
    import pygame

    base = [(size, 0.0), (-size * 0.6, -size * 0.55), (-size * 0.6, size * 0.55)]
    pts = _rotated_points(base, heading_deg)
    pts_int = [(int(x + px), int(y + py)) for px, py in pts]
    pygame.draw.polygon(surface, color, pts_int)


def draw_detector(surface, x: int, y: int, color: tuple[int, ...], radius: int) -> None:
    import pygame

    pygame.draw.circle(surface, color, (x, y), radius)


def draw_heading_tick(surface, x: int, y: int, heading_deg: float, length: int, color: tuple[int, ...]) -> None:
    import pygame

    theta = math.radians(heading_deg)
    x2 = int(x + math.cos(theta) * length)
    y2 = int(y + math.sin(theta) * length)
    pygame.draw.line(surface, color, (x, y), (x2, y2), 1)


def draw_cross(surface, x: int, y: int, size: int, color: tuple[int, ...]) -> None:
    import pygame

    pygame.draw.line(surface, color, (x - size, y - size), (x + size, y + size), 1)
    pygame.draw.line(surface, color, (x - size, y + size), (x + size, y - size), 1)
