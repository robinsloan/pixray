# RS CHIAROSCURO SPRAY DRAWER

from DrawingInterface import DrawingInterface

import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
import PIL.ImageFilter
import json

from util import str2bool

def bound(value, low, high):
    return max(low, min(high, value))

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

class NoiseDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--init_from_json", type=int, help="json points!", default=None, dest='init_from_json')

        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()

        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.init_from_json = settings.init_from_json

        canvas_width, canvas_height = self.canvas_width, self.canvas_height

        shapes = []
        shape_groups = []

        # background slate
        p0 = [0, 0]
        p1 = [canvas_width, canvas_height]
        path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
        shapes.append(path)
        bg_color = torch.tensor([25.0/255.0, 25.0/255.0, 50.0/255.0, 1.0])
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes)-1]), stroke_color = None, fill_color = bg_color)
        shape_groups.append(path_group)

        if self.init_from_json:
            print("init from json points!")
            with open(self.init_from_json) as json_file:
                init_points = json.load(json_file)
                for point_and_color in init_points:
                    # part 0 is [x, y], part 1 is [r, g, b, a]
                    point_radius = torch.tensor(0.5)
                    path = pydiffvg.Circle(radius = point_radius, center = torch.tensor(point_and_color[0]))
                    shapes.append(path)

                    group_ids = []
                    group_ids.append(len(shapes)-1)

                    point_color = torch.tensor(point_and_color[1])
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor(group_ids), fill_color = point_color, stroke_color = None)
                    shape_groups.append(path_group)

        else:
            # VERY important, determines density!
            scaled_num_points = round(canvas_width * canvas_height / 12)

            blob_size = int(canvas_width * 0.2) # hard-coded, oh well
            blob_sigma = 0.8
            center_x = canvas_width/2
            center_y = canvas_height/2

            """
            init_image = PIL.Image.open("/content/gdrive/MyDrive/pixray/guide-tri.png")
            init_image_rgb = init_image.convert('RGB')
            init_image_rgb = init_image_rgb.resize((canvas_width, canvas_height), PIL.Image.LANCZOS)
            init_image_rgb = init_image_rgb.filter(PIL.ImageFilter.GaussianBlur(radius = 32))
            init_data = np.array(init_image_rgb)
            """

            num_points_placed = 0

            while num_points_placed < scaled_num_points:
                point_ids = []
                point_radius = torch.tensor(0.5)
                point_x = center_x + random.gauss(0, blob_sigma) * blob_size
                point_y = center_y + random.gauss(0, blob_sigma) * blob_size
                point_x = clamp(0, point_x, canvas_width-1)
                point_y = clamp(0, point_y, canvas_height-1)

                if False:
                  orig_color = init_data[round(point_y), round(point_x), 1] # green only
                  if orig_color < 128:
                    continue # this one isn't suitable, too dark

                point_center = torch.tensor([point_x, point_y])
                path = pydiffvg.Circle(radius=point_radius, center=point_center)
                shapes.append(path)
                point_ids.append(len(shapes)-1)

                point_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])

                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor(point_ids), fill_color=point_color, stroke_color=None)
                shape_groups.append(path_group)

                num_points_placed += 1

        # diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

        points_vars = []
        color_vars = []

        for path in shapes[1:]: # [1:] to exclude bg rect
            if hasattr(path, 'center'):
                path.center.requires_grad = True
                points_vars.append(path.center)
            if hasattr(path, 'points'): # not used, currently
                path.points.requires_grad = True
                points_vars.append(path.points)

        print("points_vars length:")
        print(len(points_vars))

        print("shape_groups length:")
        print(len(shape_groups))

        for group in shape_groups[1:]: # [1:] to exclude bg rect
            if group.fill_color != None:
                group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)

        print("color_vars length:")
        print(len(color_vars))

        self.points_vars = points_vars
        self.color_vars = color_vars
        self.img = img
        self.shapes = shapes
        self.shape_groups  = shape_groups
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def load_model(self, settings, device):
        # use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        device = torch.device('cuda')
        pydiffvg.set_device(device)

    def get_opts(self, decay_divisor):
        # optimizers
        points_optim = torch.optim.Adam(self.points_vars, lr=1.0/decay_divisor)
        opts = [points_optim]
        if len(self.color_vars) > 0:
            color_optim = torch.optim.Adam(self.color_vars, lr=0.01/decay_divisor)
            opts.append(color_optim)
        return opts

    def rand_init(self, toksX, toksY):
        # TODO
        pass

    def init_from_tensor(self, init_tensor):
        # TODO
        pass

    def reapply_from_tensor(self, new_tensor):
        # TODO
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        # TODO
        return 5

    def synth(self, cur_iteration):
        # damn I am glad I didn't have to figure this out myself
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = render(self.canvas_width, self.canvas_height, 2, 2, cur_iteration, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        print("called to_image... surprising!")
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 254)
        # img = np.repeat(img, 4, axis=0)
        # img = np.repeat(img, 4, axis=1)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        with torch.no_grad():
            for group in self.shape_groups[1:]:
                if group.fill_color != None:
                    group.fill_color.data.clamp_(0.0, 1.0)

                    # REMEMBER YOU ARE CLAMPING ALPHA HERE
                    # but I think that's what I want... no transparent dots
                    group.fill_color.data[3] = 1.0

    def get_z(self):
        return None

    def get_z_copy(self):
        return None

    def set_z(self, new_z):
        return None

    @torch.no_grad()
    def to_svg(self, svg_filename):
        pydiffvg.save_svg(svg_filename, self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)

    @torch.no_grad()
    def to_json(self, json_filename):
        with open(json_filename, 'w') as outfile:
            points_list = [point.tolist() for point in self.points_vars]
            colors_list = [group.fill_color.data.tolist() for group in self.shape_groups[1:]]
            # i love a good array zip!!
            combo_list = list(zip(points_list, colors_list))
            json.dump(combo_list, outfile, indent=4)