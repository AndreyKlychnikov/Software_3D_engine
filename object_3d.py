import pygame as pg
from matrix_functions import *
from numba import njit


@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))


class Object3D:
    def __init__(self, render, vertexes, faces):
        self.render = render
        self.vertexes = vertexes
        self.rotate_x(math.pi)
        self.rotate_y(math.pi * 0.4)
        self.vertexes[:, :3] = (self.vertexes[:, :3] + 2) * 60 + np.array([200, 150, 0])
        self.faces = faces

        self.font = pg.font.SysFont("Arial", 30, bold=True)
        # self.color_faces = [(pg.Color("orange"), face) for face in self.faces]
        # self.movement_flag, self.draw_vertexes = True, False
        # self.label = ""

    def draw(self):
        frame_buffer = np.zeros((800, 900))
        z_buffer = np.zeros((800, 900))

        for i, face in enumerate(self.faces):
            poly = self.vertexes[face]

            left = round(min([v[0] for v in poly]))
            right = round(max([v[0] for v in poly]))
            top = round(max([v[1] for v in poly]))
            bottom = round(min([v[1] for v in poly]))

            A, B, C = np.linalg.solve(poly[:3, :3], np.array([-1, -1, -1]))
            if C == 0:
                continue
            if np.array([A, B, C]) @ np.array([0, 0, -1]) < 0:
                continue

            for j in range(len(poly) - 1):
                self.draw_line(frame_buffer, z_buffer, poly[j], poly[j + 1])
            self.draw_line(frame_buffer, z_buffer, poly[-1], poly[0])

            poly[:, :2] = poly[:, :2] + 1
            for y in range(bottom, top):
                for x in range(left, right):
                    if is_point_in_path(x, y, poly):
                        z = -(A * x + B * y + 1) / C - 1
                        if z > z_buffer[x, y]:
                            z_buffer[x, y] = z
                            frame_buffer[x, y] = 1

        # for x, y in np.argwhere(frame_buffer == 1):
        #     pg.draw.line(self.render.screen, pg.Color("white"), (x, y), (x, y), 1)

        for x, y in np.argwhere(frame_buffer == 2):
            pg.draw.line(self.render.screen, pg.Color("orange"), (x, y), (x, y), 1)

    def draw_line(self, frame_buffer, z_buffer, p1, p2):
        if tuple(p1[:2]) == tuple(p2[:2]):
            z = max(p1[3], p2[3])
            x, y = round(p1[0]), round(p2[1])
            if z > z_buffer[x, y]:
                z_buffer[x, y] = z
                frame_buffer[x, y] = 2
            return

        left = p1
        right = p2

        if round(right[0]) == round(left[0]):
            if right[1] < left[1]:
                left, right = right, left
            x = int(left[0])
            for y, z in zip(
                range(int(left[1]), int(right[1]) + 1),
                np.linspace(left[2], right[2], int(abs(right[1] - left[1]) + 1)),
            ):
                y = int(round(y))
                if z >= z_buffer[x, y]:
                    z_buffer[x, y] = z
                    frame_buffer[x, y] = 2
            return

        if right[0] < left[0]:
            left, right = right, left

        a, b = np.linalg.solve(((p1[0], 1), (p2[0], 1)), (p1[1], p2[1]))

        if right[0] - left[0] >= abs(right[1] - left[1]):
            left[0] = round(left[0])
            right[0] = round(right[0])
            for x, z in zip(
                range(int(left[0]), int(right[0]) + 1),
                np.linspace(left[2], right[2], int(right[0] - left[0] + 1)),
            ):
                y = a * x + b
                x, y = int(x), int(round(y))
                if z >= z_buffer[x, y]:
                    z_buffer[x, y] = z
                    frame_buffer[x, y] = 2
        else:
            first_y, last_y = a * left[0] + b, a * right[0] + b
            first_y, last_y = int(round(first_y)), int(round(last_y))
            first_z, last_z = left[2], right[2]
            first_x, last_x = left[0], right[0]
            if first_y > last_y:
                first_y, last_y = last_y, first_y
                first_z, last_z = last_z, first_z
                first_x, last_x = last_x, first_x

            for x, y, z in zip(
                np.linspace(first_x, last_x, last_y - first_y + 1),
                range(first_y, last_y + 1),
                np.linspace(first_z, last_z, last_y - first_y + 1),
            ):
                x, y = int(round(x)), int(round(y))
                if z >= z_buffer[x, y]:
                    z_buffer[x, y] = z
                    frame_buffer[x, y] = 2

    def translate(self, pos):
        self.vertexes = self.vertexes @ translate(pos)

    def scale(self, scale_to: float):
        self.vertexes = self.vertexes @ scale(scale_to)

    def rotate_x(self, angle):
        self.vertexes = self.vertexes @ rotate_x(angle)

    def rotate_y(self, angle):
        self.vertexes = self.vertexes @ rotate_y(angle)

    def rotate_z(self, angle):
        self.vertexes = self.vertexes @ rotate_z(angle)

    def control(self):
        key = pg.key.get_pressed()
        base_delta = 0.1
        if key[pg.K_a]:
            self.translate(np.array([-base_delta, 0, 0]))
        if key[pg.K_d]:
            self.translate(np.array([base_delta, 0, 0]))
        if key[pg.K_w]:
            self.translate(np.array([0, base_delta, 0]))
        if key[pg.K_s]:
            self.translate(np.array([0, -base_delta, 0]))
        if key[pg.K_q]:
            self.translate(np.array([0, 0, base_delta]))
        if key[pg.K_e]:
            self.translate(np.array([0, 0, -base_delta]))

        base_angle = math.pi / 180

        modded = pg.key.get_mods()
        if modded:
            base_angle *= -1

        if key[pg.K_x]:
            self.rotate_x(base_angle)
        if key[pg.K_z]:
            self.rotate_z(base_angle)
        if key[pg.K_y]:
            self.rotate_y(base_angle)

        if key[pg.K_EQUALS]:
            self.scale(1.01)
        if key[pg.K_MINUS]:
            self.scale(0.99)


class Axes(Object3D):
    def __init__(self, render):
        super().__init__(
            render,
            vertexes=np.array([(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]),
            faces=np.array([(0, 1), (0, 2), (0, 3)]),
        )
        self.colors = [pg.Color("red"), pg.Color("green"), pg.Color("blue")]
        self.color_faces = [
            (color, face) for color, face in zip(self.colors, self.faces)
        ]
        self.draw_vertexes = False
        self.label = "XYZ"


@njit(fastmath=True)
def is_point_in_path(x: int, y: int, poly) -> bool:
    """Determine if the point is in the path.

    Args:
      x -- The x coordinates of point.
      y -- The y coordinates of point.
      poly -- a list of tuples [(x, y), (x, y), ...]

    Returns:
      True if the point is in the path.
    """
    num = len(poly)
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and (
            x
            < poly[i][0]
            + (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1])
        ):
            c = not c
        j = i
    return c
