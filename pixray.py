import argparse
import math
from urllib.request import urlopen
import sys
import os
import json
import subprocess
import glob
from braceexpand import braceexpand
from types import SimpleNamespace

import os.path

from omegaconf import OmegaConf

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
torch.backends.cudnn.benchmark = False
# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)
# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP, RAdam
from perlin_numpy import generate_fractal_noise_2d

from CLIP import clip
import kornia
import kornia.augmentation as K
import numpy as np
import imageio
import re
import random

from einops import rearrange

from PIL import ImageFile, Image, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

# or 'border'
global_padding_mode = "reflection"
global_aspect_width = 1

from util import map_number, palette_from_string, real_glob

from vqgan import VqganDrawer

class_table = {
    "vqgan": VqganDrawer
}

from clipdrawer import ClipDrawer
from pixeldrawer import PixelDrawer
from linedrawer import LineDrawer
# update class_table if these import OK
class_table.update({
    "line_sketch": LineDrawer,
    "pixel": PixelDrawer,
    "clipdraw": ClipDrawer
})

import matplotlib.colors
# only needed for palette stuff

# this is enabled when not in the master branch
# print("warning: running unreleased future version")

# https://stackoverflow.com/a/39662359
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'Shell':
            return True   # Seems to be what co-lab does
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

IS_NOTEBOOK = isnotebook()

if IS_NOTEBOOK:
    from IPython import display
    from tqdm.notebook import tqdm
    from IPython.display import clear_output
else:
    from tqdm import tqdm

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    # print(f"parsed vals is {vals}")
    return vals[0], float(vals[1]), float(vals[2])

from typing import cast, Dict, List, Optional, Tuple, Union

# override class to get padding_mode
class MyRandomPerspective(K.RandomPerspective):
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return kornia.geometry.warp_perspective(
            input, transform, (height, width),
             mode=self.resample.name.lower(), align_corners=self.align_corners, padding_mode=global_padding_mode
        )

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, clip_view=None):
        global global_aspect_width

        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.clip_view = clip_view
        self.cutn_zoom = int(2*cutn/3) # "two thirds of the cutouts will be zooms"
        self.transforms = None

        augmentations = []
        if global_aspect_width != 1:
            augmentations.append(K.RandomCrop(size=(self.cut_size,self.cut_size), p=1.0, cropping_mode="resample", return_transform=True))
        augmentations.append(MyRandomPerspective(distortion_scale=0.2, p=0.7, return_transform=True))
        augmentations.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,0.8),  ratio=(0.8,1.2), cropping_mode='resample', p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.7, return_transform=True))
        self.augs_zoom = nn.Sequential(*augmentations)

        augmentations = []
        if global_aspect_width == 1:
            n_s = 0.95
            n_t = (1-n_s)/2
            augmentations.append(K.RandomAffine(degrees=0, translate=(n_t, n_t), scale=(n_s, n_s), p=1.0, return_transform=True))
        elif global_aspect_width > 1:
            n_s = 1/global_aspect_width
            n_t = (1-n_s)/2
            augmentations.append(K.RandomAffine(degrees=0, translate=(0, n_t), scale=(0.9*n_s, n_s), p=1.0, return_transform=True))
        else:
            n_s = global_aspect_width
            n_t = (1-n_s)/2
            augmentations.append(K.RandomAffine(degrees=0, translate=(n_t, 0), scale=(0.9*n_s, n_s), p=1.0, return_transform=True))

        # augmentations.append(K.CenterCrop(size=(self.cut_size,self.cut_size), p=1.0, cropping_mode="resample", return_transform=True))
        augmentations.append(K.CenterCrop(size=self.cut_size, cropping_mode='resample', p=1.0, return_transform=True))
        augmentations.append(K.RandomPerspective(distortion_scale=0.2, p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.7, return_transform=True))
        self.augs_wide = nn.Sequential(*augmentations)

        self.noise_fac = 0.1

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input, spot=None):
        global global_aspect_width, cur_iteration
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        mask_indexes = None

        for _ in range(self.cutn):
            # Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2

            if mask_indexes is not None:
                cutout[0][mask_indexes] = 0.0 # 0.5

            if global_aspect_width != 1:
                if global_aspect_width > 1:
                    cutout = kornia.geometry.transform.rescale(cutout, (1, global_aspect_width))
                else:
                    cutout = kornia.geometry.transform.rescale(cutout, (1/global_aspect_width, 1))

            cutouts.append(cutout)

        if self.transforms is not None:
            batch1 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[:self.cutn_zoom], dim=0), self.transforms[:self.cutn_zoom],
                (self.cut_size, self.cut_size), padding_mode=global_padding_mode)
            batch2 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[self.cutn_zoom:], dim=0), self.transforms[self.cutn_zoom:],
                (self.cut_size, self.cut_size), padding_mode='zeros')
            batch = torch.cat([batch1, batch2])
        else:
            batch1, transforms1 = self.augs_zoom(torch.cat(cutouts[:self.cutn_zoom], dim=0))
            batch2, transforms2 = self.augs_wide(torch.cat(cutouts[self.cutn_zoom:], dim=0))
            batch = torch.cat([batch1, batch2])
            self.transforms = torch.cat([transforms1, transforms2])

            if self.clip_view and cur_iteration % 50 == 0:
                for j in range(self.cutn):
                    TF.to_pil_image(batch[j].cpu()).save(f'{self.clipview}clipview_cut_{j:02d}.png')

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def rebuild_optimisers(args):
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay
    global drawer

    drop_divisor = 10 ** num_loss_drop
    new_opts = drawer.get_opts(drop_divisor)
    if new_opts == None:
        # legacy

        dropped_learning_rate = args.learning_rate/drop_divisor;
        # print(f"Optimizing with {args.optimiser} set to {dropped_learning_rate}")

        # Set the optimiser
        to_optimize = [ drawer.get_z() ]
        if args.optimiser == "Adam":
            opt = optim.Adam(to_optimize, lr=dropped_learning_rate)        # LR=0.1
        elif args.optimiser == "AdamW":
            opt = optim.AdamW(to_optimize, lr=dropped_learning_rate)       # LR=0.2
        elif args.optimiser == "Adagrad":
            opt = optim.Adagrad(to_optimize, lr=dropped_learning_rate) # LR=0.5+
        elif args.optimiser == "Adamax":
            opt = optim.Adamax(to_optimize, lr=dropped_learning_rate)  # LR=0.5+?
        elif args.optimiser == "DiffGrad":
            opt = DiffGrad(to_optimize, lr=dropped_learning_rate)      # LR=2+?
        elif args.optimiser == "AdamP":
            opt = AdamP(to_optimize, lr=dropped_learning_rate)     # LR=2+?
        elif args.optimiser == "RAdam":
            opt = RAdam(to_optimize, lr=dropped_learning_rate)     # LR=2+?

        new_opts = [opt]

    return new_opts


def do_init(args):
    global opts, perceptors, normalize, cutoutsTable, cutoutSizeTable
    global z_orig, z_targets, z_labels, init_image_tensor, target_image_tensor
    global gside_X, gside_Y, overlay_image_rgba
    global pmsTable, pmsImageTable, pImages, device, spotPmsTable, spotOffPmsTable
    global drawer

    # do seed first!
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    int_seed = int(seed)%(2**30)
    print('Using seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(int_seed)
    random.seed(int_seed)

    # Do it (init that is)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    drawer = class_table[args.drawer](args)
    drawer.load_model(args, device)

    num_resolutions = drawer.get_num_resolutions()

    jit = True if float(torch.__version__[:3]) < 1.8 else False
    f = 2**(num_resolutions - 1)

    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f

    drawer.rand_init(toksX, toksY)

    # save sideX, sideY in globals (need if using overlay)
    gside_X = sideX
    gside_Y = sideY

    download_root = "/root/.cache/clip"
    if os.path.exists("/content/gdrive/MyDrive/pixray"):
        download_root = "/content/gdrive/MyDrive/pixray"
    if os.path.exists("/home/robin/pixray/models"):
        download_root = "/home/robin/pixray/models"

    for clip_model in args.clip_models:
        perceptor = clip.load(clip_model, jit=jit, download_root=download_root)[0].eval().requires_grad_(False).to(device)
        perceptors[clip_model] = perceptor

        cut_size = perceptor.visual.input_resolution
        cutoutSizeTable[clip_model] = cut_size
        if not cut_size in cutoutsTable:
            make_cutouts = MakeCutouts(cut_size, args.num_cuts, clip_view=args.clip_view)
            cutoutsTable[cut_size] = make_cutouts

    z_orig = drawer.get_z_copy()

    pmsTable = {}
    pmsImageTable = {}
    spotPmsTable = {}
    spotOffPmsTable = {}
    for clip_model in args.clip_models:
        pmsTable[clip_model] = []
        pmsImageTable[clip_model] = []
        spotPmsTable[clip_model] = []
        spotOffPmsTable[clip_model] = []
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                      std=[0.26862954, 0.26130258, 0.27577711])

    # CLIP tokenize/encode
    # NR: Weights / blending
    for prompt in args.prompts:
        for clip_model in args.clip_models:
            pMs = pmsTable[clip_model]
            perceptor = perceptors[clip_model]
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for vect_prompt in args.vector_prompts:
        f1, weight, stop = parse_prompt(vect_prompt)
        # vect_promts are by nature tuned to 10% of a normal prompt
        weight = 0.1 * weight
        if 'http' in f1:
            # note: this is currently untested...
            infile = urlopen(f1)
        elif 'json' in f1:
            infile = f1
        else:
            infile = f"vectors/{f1}.json"
            if not os.path.exists(infile):
                infile = f"pixray/vectors/{f1}.json"
        with open(infile) as f_in:
            vect_table = json.load(f_in)
        for clip_model in args.clip_models:
            pMs = pmsTable[clip_model]
            v = np.array(vect_table[clip_model])
            embed = torch.FloatTensor(v).to(device).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for clip_model in args.clip_models:
        pImages = pmsImageTable[clip_model]
        for path in args.image_prompts:
            img = Image.open(path)
            pil_image = img.convert('RGB')
            img = resize_image(pil_image, (sideX, sideY))
            pImages.append(TF.to_tensor(img).unsqueeze(0).to(device))

    opts = rebuild_optimisers(args)

    # Output for the user
    print('Using device:', device)
    print('Optimising using:', args.optimiser)

    if args.prompts:
        print('Using text prompts:', args.prompts)
    if args.image_prompts:
        print('Using #image prompts:', len(args.image_prompts))

# dreaded globals (for now)
z_orig = None
z_targets = None
z_labels = None
opts = None
drawer = None
perceptors = {}
normalize = None
cutoutsTable = {}
cutoutSizeTable = {}
init_image_tensor = None
target_image_tensor = None
pmsTable = None
spotPmsTable = None
spotOffPmsTable = None
pmsImageTable = None
gside_X=None
gside_Y=None
overlay_image_rgba=None
device=None
cur_iteration=None
cur_anim_index=None
anim_output_files=[]
anim_cur_zs=[]
anim_next_zs=[]
best_loss = None
best_iter = None
best_z = None
num_loss_drop = 0
max_loss_drops = 2
iter_drop_delay = 20

@torch.no_grad()
def checkdrop(args, iter, losses):
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay
    global drawer

    drop_loss_time = False

    loss_sum = sum(losses)
    is_new_best = False
    num_cycles_not_best = 0
    if (loss_sum < best_loss):
        is_new_best = True
        best_loss = loss_sum
        best_iter = iter
        best_z = drawer.get_z_copy()
    else:
        num_cycles_not_best = iter - best_iter
        if num_cycles_not_best >= iter_drop_delay:
            drop_loss_time = True
    return drop_loss_time

@torch.no_grad()
def checkin(args, iter, losses):
    global drawer
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay

    num_cycles_not_best = iter - best_iter

    if losses is not None:
        losses_str = ', '.join(f'{loss.item():2.3g}' for loss in losses)
        writestr = f'iter: {iter}, loss: {sum(losses).item():1.3g}, losses: {losses_str}'
    else:
        writestr = f'iter: {iter}, finished'

    writestr = f'{writestr} (-{num_cycles_not_best}=>{best_loss:2.4g})'
    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{args.prompts}')
    timg = drawer.synth(cur_iteration)
    img = TF.to_pil_image(timg[0].cpu())
    # img = drawer.to_image()
    output_with_iter = args.output.replace("ITER", str(cur_iteration))
    img.save(args.output_path + output_with_iter, pnginfo=info)

    if args.output_svg:
        try:
            drawer.to_svg(args.output_path + args.output_svg)
        except AttributeError:
            print("You specified an output SVG file, but it looks like your drawer doesn't offer that (or there is an error in the function)")

    if args.output_json:
        try:
            drawer.to_json(args.output_path + args.output_json)
        except AttributeError:
            print("You specified an output JSON file, but it looks like your drawer doesn't offer that (or there is an error in the function)")

    if IS_NOTEBOOK and iter % args.display_every == 0:
        if args.display_clear:
            clear_output()
        display.display(display.Image(args.output_path + args.output))

    tqdm.write(writestr)

def ascend_txt(args):
    global cur_iteration, cur_anim_index, perceptors, normalize, cutoutsTable, cutoutSizeTable
    global z_orig, z_targets, z_labels, init_image_tensor, target_image_tensor, drawer
    global pmsTable, pmsImageTable, spotPmsTable, spotOffPmsTable, global_padding_mode

    out = drawer.synth(cur_iteration);

    result = []

    if (cur_iteration % 2 == 0):
        global_padding_mode = "reflection"
    else:
        global_padding_mode = "border"

    cur_cutouts = {}
    for cutoutSize in cutoutsTable:
        make_cutouts = cutoutsTable[cutoutSize]
        cur_cutouts[cutoutSize] = make_cutouts(out)

    for clip_model in args.clip_models:
        perceptor = perceptors[clip_model]
        cutoutSize = cutoutSizeTable[clip_model]
        transient_pMs = []

        pMs = pmsTable[clip_model]
        iii = perceptor.encode_image(normalize( cur_cutouts[cutoutSize] )).float()
        for prompt in pMs:
            result.append(prompt(iii))

        # If there are image prompts we make cutouts for those each time
        # so that they line up with the current cutouts from augmentation
        make_cutouts = cutoutsTable[cutoutSize]

        pImages = pmsImageTable[clip_model]

        for timg in pImages:
            # note: this caches and reuses the transforms - a bit of a hack but it works

            if args.image_prompt_shuffle:
                # print("Disabling cached transforms")
                make_cutouts.transforms = None

            # print("Building throwaway image prompts")
            # new way builds throwaway Prompts
            batch = make_cutouts(timg)
            embed = perceptor.encode_image(normalize(batch)).float()
            if args.image_prompt_weight is not None:
                transient_pMs.append(Prompt(embed, args.image_prompt_weight).to(device))
            else:
                transient_pMs.append(Prompt(embed).to(device))

        for prompt in transient_pMs:
            result.append(prompt(iii))

    if args.enforce_palette_annealing and args.target_palette:
        target_palette = torch.FloatTensor(args.target_palette).requires_grad_(False).to(device)
        _pixels = cur_cutouts[cutoutSize].permute(0,2,3,1).reshape(-1,3)
        palette_dists = torch.cdist(target_palette, _pixels, p=2)
        best_guesses = palette_dists.argmin(axis=0)
        diffs = _pixels - target_palette[best_guesses]
        palette_loss = torch.mean( torch.norm( diffs, 2, dim=1 ) )*cur_cutouts[cutoutSize].shape[0]
        result.append( palette_loss*cur_iteration/args.enforce_palette_annealing )

    if args.smoothness > 0 and args.smoothness_type:
        _pixels = cur_cutouts[cutoutSize].permute(0,2,3,1).reshape(-1,cur_cutouts[cutoutSize].shape[2],3)
        gyr, gxr = torch.gradient(_pixels[:,:,0])
        gyg, gxg = torch.gradient(_pixels[:,:,1])
        gyb, gxb = torch.gradient(_pixels[:,:,2])
        sharpness = torch.sqrt(gyr**2 + gxr**2+ gyg**2 + gxg**2 + gyb**2 + gxb**2)
        if args.smoothness_type=='clipped':
            sharpness = torch.clamp( sharpness, max=0.5 )
        elif args.smoothness_type=='log':
            sharpness = torch.log( torch.ones_like(sharpness)+sharpness )
        sharpness = torch.mean( sharpness )

        result.append( sharpness*args.smoothness )

    if args.saturation:
        # based on the old "percepted colourfulness" heuristic from Hasler and Süsstrunk’s 2003 paper
        # https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images
        _pixels = cur_cutouts[cutoutSize].permute(0,2,3,1).reshape(-1,3)
        rg = _pixels[:,0]-_pixels[:,1]
        yb = 0.5*(_pixels[:,0]+_pixels[:,1])-_pixels[:,2]
        rg_std, rg_mean = torch.std_mean(rg)
        yb_std, yb_mean = torch.std_mean(yb)
        std_rggb = torch.sqrt(rg_std**2 + yb_std**2)
        mean_rggb = torch.sqrt(rg_mean**2 + yb_mean**2)
        colorfullness = std_rggb+.3*mean_rggb

        result.append( -colorfullness*args.saturation/5.0 )

    for cutoutSize in cutoutsTable:
        # clear the transform "cache"
        make_cutouts = cutoutsTable[cutoutSize]
        make_cutouts.transforms = None

    # main init_weight uses spherical loss
    if args.target_images is not None and args.target_image_weight > 0:
        if cur_anim_index is None:
            cur_z_targets = z_targets
        else:
            cur_z_targets = [ z_targets[cur_anim_index] ]
        for z_target in cur_z_targets:
            f_z = drawer.get_z()
            if f_z is not None:
                f = f_z.reshape(1,-1)
                f2 = z_target.reshape(1,-1)
                cur_loss = spherical_dist_loss(f, f2) * args.target_image_weight
                result.append(cur_loss)

    if args.target_weight_pix:
        if target_image_tensor is None:
            print("OOPS TIT is 0")
        else:
            cur_loss = F.l1_loss(out, target_image_tensor) * args.target_weight_pix
            result.append(cur_loss)

    if args.image_labels is not None:
        for z_label in z_labels:
            f = drawer.get_z().reshape(1,-1)
            f2 = z_label.reshape(1,-1)
            cur_loss = spherical_dist_loss(f, f2) * args.image_label_weight
            result.append(cur_loss)

    # main init_weight uses spherical loss
    if args.init_weight:
        f = drawer.get_z().reshape(1,-1)
        f2 = z_orig.reshape(1,-1)
        cur_loss = spherical_dist_loss(f, f2) * args.init_weight
        result.append(cur_loss)

    # these three init_weight variants offer mse_loss, mse_loss in pixel space, and cos loss
    if args.init_weight_dist:
        cur_loss = F.mse_loss(z, z_orig) * args.init_weight_dist / 2
        result.append(cur_loss)

    if args.init_weight_pix:
        if init_image_tensor is None:
            print("OOPS IIT is 0")
        else:
            cur_loss = F.l1_loss(out, init_image_tensor) * args.init_weight_pix / 2
            result.append(cur_loss)

    if args.init_weight_cos:
        f = drawer.get_z().reshape(1,-1)
        f2 = z_orig.reshape(1,-1)
        y = torch.ones_like(f[0])
        cur_loss = F.cosine_embedding_loss(f, f2, y) * args.init_weight_cos
        result.append(cur_loss)

    return result

def re_average_z(args):
    global gside_X, gside_Y
    global device, drawer

    # old_z = z.clone()
    cur_z_image = drawer.to_image()
    cur_z_image = cur_z_image.convert('RGB')
    if overlay_image_rgba:
        # print("applying overlay image")
        cur_z_image.paste(overlay_image_rgba, (0, 0), overlay_image_rgba)
        cur_z_image.save("overlaid.png")
    cur_z_image = cur_z_image.resize((gside_X, gside_Y), Image.LANCZOS)
    drawer.reapply_from_tensor(TF.to_tensor(cur_z_image).to(device).unsqueeze(0) * 2 - 1)

# torch.autograd.set_detect_anomaly(True)

def train(args, cur_it):
    global drawer, opts
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay

    lossAll = None
    if cur_it < args.iterations:
        # this is awkward, but train is in also in charge of saving, so...
        rebuild_opts_when_done = False

        for opt in opts:
            # opt.zero_grad(set_to_none=True)
            opt.zero_grad()

        # print("drops at ", args.learning_rate_drops)

        # num_batches = args.batches * (num_loss_drop + 1)
        num_batches = args.batches
        for i in range(num_batches):
            lossAll = ascend_txt(args)

            if i == 0:
                if cur_it in args.learning_rate_drops:
                    print("Dropping learning rate")
                    rebuild_opts_when_done = True
                else:
                    did_drop = checkdrop(args, cur_it, lossAll)
                    if args.auto_stop is True:
                        rebuild_opts_when_done = disabl

            if i == 0 and cur_it % args.save_every == 0:
                checkin(args, cur_it, lossAll)

            loss = sum(lossAll)
            loss.backward()

        for opt in opts:
            opt.step()

        drawer.clip_z()

    if cur_it == args.iterations:
        # this resetting to best is currently disabled
        # drawer.set_z(best_z)
        checkin(args, cur_it, lossAll)
        return False
    if rebuild_opts_when_done:
        num_loss_drop = num_loss_drop + 1
        # this resetting to best is currently disabled
        # drawer.set_z(best_z)
        # always checkin (and save) after resetting z
        # checkin(args, cur_it, lossAll)
        if num_loss_drop > max_loss_drops:
            return False
        best_iter = cur_it
        best_loss = 1e20
        opts = rebuild_optimisers(args)
    return True

imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

def do_run(args):
    global cur_iteration, cur_anim_index
    global anim_cur_zs, anim_next_zs, anim_output_files

    cur_iteration = 0

    try:
        keep_going = True
        with tqdm() as pbar:
            while keep_going:
                try:
                    keep_going = train(args, cur_iteration)
                    if cur_iteration == args.iterations:
                        break
                    cur_iteration += 1
                    pbar.update()
                except RuntimeError as e:
                    print("Oops: runtime error: ", e)
                    print("Try reducing --num-cuts to save memory")
                    raise e
    except KeyboardInterrupt:
        pass

# this dictionary is used for settings in the notebook
global_pixray_settings = {}

def setup_parser(vq_parser):
    # Create the parser
    # vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

    # Add the arguments
    vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=[], dest='prompts')
    vq_parser.add_argument("-vp",   "--vector_prompts", type=str, help="Vector prompts", default=[], dest='vector_prompts')
    vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts", default=[], dest='image_prompts')
    vq_parser.add_argument("-ipw",  "--image_prompt_weight", type=float, help="Weight for image prompt", default=None, dest='image_prompt_weight')
    vq_parser.add_argument("-ips",  "--image_prompt_shuffle", type=bool, help="Shuffle image prompts", default=False, dest='image_prompt_shuffle')
    vq_parser.add_argument("-il",   "--image_labels", type=str, help="Image prompts", default=None, dest='image_labels')
    vq_parser.add_argument("-ilw",  "--image_label_weight", type=float, help="Weight for image prompt", default=1.0, dest='image_label_weight')
    vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=None, dest='iterations')
    vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=10, dest='save_every')
    vq_parser.add_argument("-de",   "--display_every", type=int, help="Display image iterations", default=20, dest='display_every')
    vq_parser.add_argument("-dc",   "--display_clear", type=bool, help="Clear display when updating", default=False, dest='display_clear')
    vq_parser.add_argument("-qua",  "--quality", type=str, help="draft, normal, best", default="normal", dest='quality')
    vq_parser.add_argument("-asp",  "--aspect", type=str, help="widescreen, square", default="widescreen", dest='aspect')
    vq_parser.add_argument("-ezs",  "--ezsize", type=str, help="small, medium, large", default=None, dest='ezsize')
    vq_parser.add_argument("-sca",  "--scale", type=float, help="scale (instead of ezsize)", default=None, dest='scale')
    vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height)", default=None, dest='size')
    vq_parser.add_argument("-ti",   "--target_images", type=str, help="Target images", default=None, dest='target_images')
    vq_parser.add_argument("-tiw",  "--target_image_weight", type=float, help="Target images weight", default=1.0, dest='target_image_weight')
    vq_parser.add_argument("-twp",  "--target_weight_pix", type=float, help="Target weight pix loss", default=0., dest='target_weight_pix')
    vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight (main=spherical)", default=None, dest='init_weight')
    vq_parser.add_argument("-iwd",  "--init_weight_dist", type=float, help="Initial weight dist loss", default=0., dest='init_weight_dist')
    vq_parser.add_argument("-iwc",  "--init_weight_cos", type=float, help="Initial weight cos loss", default=0., dest='init_weight_cos')
    vq_parser.add_argument("-iwp",  "--init_weight_pix", type=float, help="Initial weight pix loss", default=0., dest='init_weight_pix')
    vq_parser.add_argument("-m",    "--clip_models", type=str, help="CLIP model", default=None, dest='clip_models')
    vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.2, dest='learning_rate')
    vq_parser.add_argument("-lrd",  "--learning_rate_drops", nargs="*", type=float, help="When to drop learning rate (relative to iterations)", default=[75], dest='learning_rate_drops')
    vq_parser.add_argument("-as",   "--auto_stop", type=bool, help="Auto stopping", default=False, dest='auto_stop')
    vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=None, dest='num_cuts')
    vq_parser.add_argument("-bats", "--batches", type=int, help="How many batches of cuts", default=1, dest='batches')
    vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
    vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser (Adam, AdamW, Adagrad, Adamax, DiffGrad, AdamP or RAdam)", default='Adam', dest='optimiser')
    vq_parser.add_argument("-opath","--output_path", type=str, help="Output path", default="./", dest='output_path')
    vq_parser.add_argument("-o",    "--output", type=str, help="Output file", default="output.png", dest='output')
    vq_parser.add_argument("-osvg", "--output_svg", type=str, help="Output file for raw SVG", default=None, dest='output_svg')
    vq_parser.add_argument("-ojson","--output_json", type=str, help="Output file for points as JSON", default=None, dest='output_json')
    vq_parser.add_argument("-d",    "--deterministic", type=bool, help="Enable cudnn.deterministic?", default=False, dest='cudnn_determinism')
    vq_parser.add_argument("-epw",  "--enforce_palette_annealing", type=int, help="enforce palette annealing, 0 -- skip", default=5000, dest='enforce_palette_annealing')
    vq_parser.add_argument("-tp",   "--target_palette", type=str, help="target palette", default=None, dest='target_palette')
    vq_parser.add_argument("-tpl",  "--target_palette_length", type=int, help="target palette length", default=16, dest='target_palette_length')
    vq_parser.add_argument("-smo",  "--smoothness", type=float, help="encourage smoothness, 0 -- skip", default=0, dest='smoothness')
    vq_parser.add_argument("-est",  "--smoothness_type", type=str, help="enforce smoothness type: default/clipped/log", default='default', dest='smoothness_type')
    vq_parser.add_argument("-sat",  "--saturation", type=float, help="encourage saturation, 0 -- skip", default=0, dest='saturation')
    vq_parser.add_argument("-cview","--clip_view", type=float, help="Directory to spit out files showing what CLIP is seeing", default=None, dest='clip_view')
    vq_parser.add_argument("-nd",   "--noise_density", type=float, help="Noise density for noisedrawer", default=0.08, dest='noise_density')

    return vq_parser

def process_args(vq_parser, namespace=None):
    global global_aspect_width
    global cur_iteration, cur_anim_index, anim_output_files, anim_cur_zs, anim_next_zs;
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay

    if namespace == None:
      # command line: use ARGV to get args
      args = vq_parser.parse_args()
    elif isnotebook():
      args = vq_parser.parse_args(args=[], namespace=namespace)
    else:
      # sometimes there are both settings and cmd line
      args = vq_parser.parse_args(namespace=namespace)

    if args.cudnn_determinism:
       torch.backends.cudnn.deterministic = True

    quality_to_clip_models_table = {
        'draft': 'ViT-B/32',
        'normal': 'ViT-B/32,ViT-B/16',
        'better': 'RN50,ViT-B/32,ViT-B/16',
        'best': 'RN50x4,ViT-B/32,ViT-B/16'
    }

    quality_to_iterations_table = {
        'draft': 200,
        'normal': 300,
        'better': 400,
        'best': 500
    }
    quality_to_scale_table = {
        'draft': 1,
        'normal': 2,
        'better': 3,
        'best': 4
    }

    # this should be replaced with logic that does somethings
    # smart based on available memory (eg: size, num_models, etc)
    quality_to_num_cuts_table = {
        'draft': 40,
        'normal': 40,
        'better': 40,
        'best': 40
    }

    if args.quality not in quality_to_clip_models_table:
        print("Qualitfy setting not understood, aborting -> ", args.quality)
        exit(1)

    if args.clip_models is None:
        args.clip_models = quality_to_clip_models_table[args.quality]
    if args.iterations is None:
        args.iterations = quality_to_iterations_table[args.quality]
    if args.num_cuts is None:
        args.num_cuts = quality_to_num_cuts_table[args.quality]
    if args.ezsize is None and args.scale is None:
        args.scale = quality_to_scale_table[args.quality]

    size_to_scale_table = {
        'small': 1,
        'medium': 2,
        'large': 4
    }

    aspect_to_size_table = {
        'square': [150, 150],
        'widescreen': [200, 112]
    }

    if args.size is not None:
        global_aspect_width = args.size[0] / args.size[1]
    elif args.aspect == "widescreen":
        global_aspect_width = 16/9
    else:
        global_aspect_width = 1

    # determine size if not set
    if args.size is None:
        size_scale = args.scale
        if size_scale is None:
            if args.ezsize in size_to_scale_table:
                size_scale = size_to_scale_table[args.ezsize]
            else:
                print("EZ Size not understood, aborting -> ", args.ezsize)
                exit(1)
        if args.aspect in aspect_to_size_table:
            base_size = aspect_to_size_table[args.aspect]
            base_width = int(size_scale * base_size[0])
            base_height = int(size_scale * base_size[1])
            args.size = [base_width, base_height]
        else:
            print("aspect not understood, aborting -> ", args.aspect)
            exit(1)

    # Split text prompts using the pipe character
    if args.prompts:
        args.prompts = [phrase.strip() for phrase in args.prompts.split("|")]

    # Split target images using the pipe character
    if args.image_prompts:
        args.image_prompts = real_glob(args.image_prompts)

    # Split text prompts using the pipe character
    if args.vector_prompts:
        args.vector_prompts = [phrase.strip() for phrase in args.vector_prompts.split("|")]

    if args.target_palette is not None:
        args.target_palette = palette_from_string(args.target_palette)

    clip_models = args.clip_models.split(",")
    args.clip_models = [model.strip() for model in clip_models]

    if args.learning_rate_drops is None:
        args.learning_rate_drops = []
    else:
        args.learning_rate_drops = [int(map_number(n, 0, 100, 0, args.iterations-1)) for n in args.learning_rate_drops]

    # reset global animation variables
    cur_iteration = 0
    best_iter = cur_iteration
    best_loss = 1e20
    num_loss_drop = 0
    max_loss_drops = len(args.learning_rate_drops)
    iter_drop_delay = 12
    best_z = None

    cur_anim_index=None
    anim_output_files=[]
    anim_cur_zs=[]
    anim_next_zs=[]

    return args

def reset_settings():
    global global_pixray_settings
    global_pixray_settings = {}

def add_settings(**kwargs):
    global global_pixray_settings
    for k, v in kwargs.items():
        if v is None:
            # just remove the key if it is there
            global_pixray_settings.pop(k, None)
        else:
            global_pixray_settings[k] = v

def get_settings():
    global global_pixray_settings
    return global_pixray_settings.copy()

def apply_settings():
    global global_pixray_settings
    settingsDict = None

    # first pass - just get the drawer
    # Create the parser
    vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')
    vq_parser.add_argument("--drawer", type=str, help="clipdraw, pixeldraw, etc", default="vqgan", dest='drawer')
    settingsDict = SimpleNamespace(**global_pixray_settings)
    settings_core, unknown = vq_parser.parse_known_args(namespace=settingsDict)

    vq_parser = setup_parser(vq_parser)
    class_table[settings_core.drawer].add_settings(vq_parser)

    if len(global_pixray_settings) > 0:
        # check for any bogus entries in the settings
        dests = [d.dest for d in vq_parser._actions]
        for k in global_pixray_settings:
            if not k in dests:
                raise ValueError(f"Requested setting not found, aborting: {k}={global_pixray_settings[k]}")

        # convert dictionary to easyDict
        # which can be used as an argparse namespace instead
        # settingsDict = easydict.EasyDict(global_pixray_settings)
        settingsDict = SimpleNamespace(**global_pixray_settings)

    settings = process_args(vq_parser, settingsDict)
    return settings

def command_line_override():
    global global_pixray_settings
    settingsDict = None
    vq_parser = setup_parser()
    settings = process_args(vq_parser)
    return settings

def main():
    settings = apply_settings()
    do_init(settings)
    do_run(settings)

if __name__ == '__main__':
    main()