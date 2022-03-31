from typing import Sequence

import numpy as np
import moderngl

from .obj import Obj


def orthographic_matrix(left, right, bottom, top, near, far):
    return np.array((
        (2 / (right - left), 0, 0, -(right + left) / (right - left)),
        (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
        (0, 0, -2 / (far - near), -(far + near) / (far - near)),
        (0, 0, 0, 1),
    ))


def projection_matrix(K, w, h, near=10., far=10000.):  # 1 cm to 10 m
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[:2, :3] = K[:2, :3]
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:
    orth = orthographic_matrix(-.5, w - .5, -.5, h - .5, near, far)
    return orth @ persp @ view


class ObjCoordRenderer:
    def __init__(self, objs: Sequence[Obj], w: int, h: int = None, device_idx=0):
        self.objs = objs
        if h is None:
            h = w
        self.h, self.w = h, w
        self.ctx = moderngl.create_context(standalone=True, backend='egl', device_index=device_idx)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.simple_framebuffer((w, h), components=4, dtype='f4')
        self.near, self.far = 10., 10000.,

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform vec3 offset;
                uniform float scale;
                uniform mat4 mvp;
                in vec3 in_vert;
                out vec3 color;
                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    color = (in_vert - offset) / scale;
                }
                """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                in vec3 color;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
                """,
        )

        self.vaos = []
        for obj in self.objs:
            vertices = obj.mesh.vertices[obj.mesh.faces].astype('f4')  # (n, 3)
            vao = self.ctx.simple_vertex_array(self.prog, self.ctx.buffer(vertices), 'in_vert')
            self.vaos.append(vao)

    def read(self):
        return np.frombuffer(self.fbo.read(components=4, dtype='f4'), 'f4').reshape((self.h, self.w, 4))

    def read_depth(self):
        depth = np.frombuffer(self.fbo.read(attachment=-1, dtype='f4'), 'f4').reshape(self.h, self.w)
        neg_mask = depth == 1
        near, far = 10., 10000.  # TODO: use projection matrix instead of the default values
        depth = 2 * depth - 1
        depth = 2 * near * far / (far + near - depth * (far - near))
        depth[neg_mask] = 0
        return depth

    def render(self, obj_idx, K, R, t, clear=True, read=True, read_depth=False):
        obj = self.objs[obj_idx]
        mv = np.concatenate((
            np.concatenate((R, t), axis=1),
            [[0, 0, 0, 1]],
        ))
        mvp = projection_matrix(K, self.w, self.h, self.near, self.far) @ mv
        self.prog['mvp'].value = tuple(mvp.T.astype('f4').reshape(-1))
        self.prog['scale'].value = obj.scale
        self.prog['offset'].value = tuple(obj.offset.astype('f4'))

        self.fbo.use()
        if clear:
            self.ctx.clear()
        self.vaos[obj_idx].render(mode=moderngl.TRIANGLES)
        if read_depth:
            return self.read_depth()
        elif read:
            return self.read()
        else:
            return None

    @staticmethod
    def extract_mask(model_coords_img: np.ndarray):
        return model_coords_img[..., 3] == 255

    def denormalize(self, model_coords: np.ndarray, obj_idx: int):
        return model_coords * self.objs[obj_idx].scale + self.objs[obj_idx].offset
