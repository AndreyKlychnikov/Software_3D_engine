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
        self.faces = faces

        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        self.movement_flag, self.draw_vertexes = True, False
        self.label = ''

    def draw(self):
        self.screen_projection()
        # self.movement()

    def movement(self):
        if self.movement_flag:
            self.rotate_y(-(pg.time.get_ticks() % 0.005))

    def screen_projection(self):
        vertexes = self.vertexes @ self.render.camera.camera_matrix()
        vertexes = vertexes @ self.render.projection.projection_matrix
        vertexes /= vertexes[:, -1].reshape(-1, 1)
        vertexes[(vertexes > 2) | (vertexes < -2)] = 0
        vertexes = vertexes @ self.render.projection.to_screen_matrix
        vertexes = vertexes[:, :2]

        for index, color_face in enumerate(self.color_faces):
            color, face = color_face
            polygon = vertexes[face]
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                pg.draw.polygon(self.render.screen, color, polygon, 1)
                if self.label:
                    text = self.font.render(self.label[index], True, pg.Color('white'))
                    self.render.screen.blit(text, polygon[-1])

        if self.draw_vertexes:
            for vertex in vertexes:
                if not any_func(vertex, self.render.H_WIDTH, self.render.H_HEIGHT):
                    pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)

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
        super().__init__(render,
                         vertexes=np.array([(0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]),
                         faces=np.array([(0, 1), (0, 2), (0, 3)]))
        self.colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]
        self.color_faces = [(color, face) for color, face in zip(self.colors, self.faces)]
        self.draw_vertexes = False
        self.label = 'XYZ'
