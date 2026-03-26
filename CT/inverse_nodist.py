import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from training.pos_embedding import Pos_Embedding
import scipy.io
from diffusers import AutoencoderKL
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sys
from inverse_operators import *
from denoise_padding import denoisedFromPatches, getIndices, denoisedOverlap, denoisedTile
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def makeFigures(noisy2, denoised2, orig2, i, imsize=256):
    channels = len(denoised2[:,0,0])
    dir = '/home/akheirandish3/PaDIS/'
    denoised = torch.clone(denoised2)
    noisy = torch.clone(noisy2)
    orig = orig2.copy()
    orig = np.transpose(orig, (1,2,0))

    denoised = torch.squeeze(denoised).cpu().numpy()
    orig = np.squeeze(orig)
    noisy = torch.squeeze(noisy).cpu().numpy()

    if channels > 1:
        noisy = np.transpose(noisy, (1,2,0))
        denoised = np.transpose(denoised, (1,2,0))

    noisy = np.clip(noisy, 0, 1)
    denoised = np.clip(denoised, 0,1)
    orig = np.clip(orig, 0, 1)

    noisypsnr = psnr(noisy, orig, data_range=1)
    denoisedpsnr = psnr(denoised, orig, data_range=1)
    t1 = 'FBP recon'
    t2 = 'Diffusion recon'

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1),plt.imshow(noisy, cmap='gray'),plt.axis('off'),plt.title(str(noisypsnr))
    plt.subplot(1,3,2),plt.imshow(denoised, cmap='gray'),plt.axis('off'),plt.title(str(denoisedpsnr))
    plt.subplot(1,3,3),plt.imshow(orig, cmap='gray'),plt.axis('off')

    plt.show()
    plt.savefig(dir + str(i) + '.png')
    plt.close('all')

def pinv(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):
    w = len(latents[0,0,0,:])
    fbp = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    return torch.nn.functional.pad(fbp, (pad, pad, pad, pad), "constant", 0)

def pc_sampling(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):
    w = len(latents[0,0,0,:])
    patches = w // psize + 1
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_init = sigma_max*torch.randn(3, w, w).to(device = latents.device)
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0)

    for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))):
        if i == num_steps-1:
            break
        indices = getIndices(spaced, patches, pad, psize)
        D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False)
        D = torch.squeeze(D, dim=0)
        score = (D-x)/t_cur**2
        x = x + (t_cur**2 - t_next**2) * score
        z = randn_like(x)
        x = x + torch.sqrt(t_cur**2 - t_next**2) * z #predictor step

        Ainput = noisy-inverseop.A(x[:,pad:pad+w, pad:pad+w].to(dtype=torch.float32))
        stepsize = zeta/torch.norm(Ainput)
        x[:,pad:pad+w, pad:pad+w] += stepsize * inverseop.AT(Ainput)

        if i < num_steps-1:
            z = randn_like(x)
            D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False)
            D = torch.squeeze(D, dim=0)
            score = (D-x)/t_next**2
            r = 0.16
            eps = 2*r*torch.norm(z)/torch.norm(score)
            x = x + eps * score
            x = x + torch.sqrt(2*eps)*z #corrector step

            Ainput = noisy-inverseop.A(x[:,pad:pad+w, pad:pad+w].to(dtype=torch.float32))
            stepsize = zeta/torch.norm(Ainput)* min(40, t_cur*200)
            x[:,pad:pad+w, pad:pad+w] += stepsize * inverseop.AT(Ainput)

        if i%5 == 0 or i == num_steps-1:
            makeFigures(x_init, x[:,pad:pad+w, pad:pad+w].detach(), clean, i)
    return x

def measurement_cond_fn(measurement, x_prev, x0hat, inverseop, pad=24, w=256):
    difference = measurement - inverseop.A(x0hat[:,pad:pad+w, pad:pad+w]).to(dtype=torch.float32)
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
    return norm_grad

# def dps(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
#               clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
#               S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,):
#     w = len(latents[0,0,0,:])
#     patches = w // psize + 1
#     spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
#     x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
#     step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
#     t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
#     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

#     x = sigma_max*torch.randn_like(x_init).to(device=latents.device)
#     x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0).requires_grad_()
#     for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))):
#         alpha = 0.5*t_cur**2
#         for j in range(10):
#             indices = getIndices(spaced, patches, pad, psize)
#             D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False)
#             D = torch.squeeze(D, dim=0)
#             score = (D-x)/t_cur**2
#             z = randn_like(x)

#             x0hat = D
#             norm_grad = measurement_cond_fn(noisy, x, x0hat, inverseop, pad=pad)
#             x = x - zeta * norm_grad

#             if i < num_steps - 1:
#                 x = x + alpha/2 * score + torch.sqrt(alpha) * z
#             else:
#                 x = x + alpha/2 * score
#         # if i%5 == 0 or i == num_steps-1:
#         #     makeFigures(x_init, x[:,pad:pad+w, pad:pad+w].detach(), clean, i)
#     return x.detach()

def langevin(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, ddnm=False):
    w = len(latents[0,0,0,:])
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    #t_steps = torch.from_numpy(np.geomspace(sigma_max, sigma_min, num=num_steps)).to(latents.device)
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    #print(t_steps)
    x = x_init #might initialize with pure noise, depends
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", 0)
    x = sigma_max * torch.randn_like(x)

    patches = w // psize + 1
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    #print(x.shape)
    for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))):
        alpha = 1*t_cur**2
        for j in range(10):
            indices = getIndices(spaced, patches, pad, psize)
            D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False)
            D = torch.squeeze(D, dim=0)
            z = randn_like(x)

            if ddnm:
                Dsmall = D[:,pad:pad+w, pad:pad+w]
                x0hat = inverseop.Adagger(noisy) + Dsmall - inverseop.Adagger(inverseop.A(Dsmall))
                x0hat = torch.nn.functional.pad(x0hat, (pad, pad, pad, pad), "constant", 0)
                score = (x0hat-x)/t_cur**2
            else:
                score = (D-x)/t_cur**2
                Ainput = noisy-inverseop.A(x[:,pad:pad+w, pad:pad+w].to(dtype=torch.float32))
                stepsize = zeta/torch.norm(Ainput)* min(40, t_cur*200)
                x[:,pad:pad+w, pad:pad+w] += stepsize * inverseop.AT(Ainput)

            if i < num_steps - 1:
                x = x + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x = x + alpha/2 * score
        if i%2 == 0 or i == num_steps-1:
            makeFigures(x_init, x[:,pad:pad+w, pad:pad+w], clean, i)
    return x

def dps(net, latents, latents_pos, inverseop, noisy=None, randn_like = torch.randn_like, num_steps=18,
              clean=None, sigma_min=0.005, sigma_max = 0.05, rho=7, zeta=0.3, pad=64, psize=64,
              S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, save_steps=True):
    w = len(latents[0,0,0,:])
    patches = w // psize + 1
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x = sigma_max*torch.randn_like(x_init).to(device=latents.device)
    x = torch.nn.functional.pad(x_init, (pad, pad, pad, pad), "constant", 0).requires_grad_()
    
    # Store intermediate steps
    intermediate_steps = []
    step_intervals = max(1, num_steps//10)  # Save 5 steps evenly distributed
    
    for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))):
        alpha = 0.5*t_cur**2
        for j in range(10):
            indices = getIndices(spaced, patches, pad, psize)
            D = denoisedFromPatches(net, torch.unsqueeze(x, 0), t_cur, latents_pos, None, indices, t_goal=0, wrong=False)
            D = torch.squeeze(D, dim=0)
            score = (D-x)/t_cur**2
            z = randn_like(x)

            x0hat = D
            norm_grad = measurement_cond_fn(noisy, x, x0hat, inverseop, pad=pad)
            x = x - zeta * norm_grad

            if i < num_steps - 1:
                x = x + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x = x + alpha/2 * score
        
        # Save intermediate steps at specific intervals
        if save_steps and (i % step_intervals == 0 or i == num_steps-1):
            step_result = x[:, pad:pad+w, pad:pad+w].detach().cpu().numpy()
            intermediate_steps.append({
                'step': i,
                'sigma': t_cur.item(),
                'image': step_result
            })
            
    return x.detach(), intermediate_steps

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

#----------------------------------------------------------------------------

@click.command()
#directory based options
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--image_dir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--image_size',                help='Sample resolution', metavar='INT',                                 type=int, default=None)
@click.option('--pad',                help='Pad width', metavar='INT',                                 type=int, default=None)
@click.option('--psize',                help='Patch size', metavar='INT',                                 type=int, default=None)

#inverse operator options
@click.option('--views',                help='Number of CT views', metavar='INT',                                type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--blursize',                help='Size of blur kernel', metavar='INT',                                type=click.IntRange(min=1), default=31, show_default=True)
@click.option('--channels',                help='Image channels', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--name',                  help='Experiment type', metavar='ct_parbeam|ct_fanbeam|denoise',             type=click.Choice(['ct_parbeam', 'ct_fanbeam', 'lact', 'denoise', 'deblur_uniform', 'super']))
@click.option('--sigma',                help='Noise of measurement', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--scale',                help='Superresolution scale', metavar='INT',                                type=click.IntRange(min=1), default=2, show_default=True)

@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)

#solver options
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--zeta',                help='Step size', metavar='FLOAT',                          type=click.FloatRange(min=0), default=1.0, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
# @click.option('--device_id', 'device_id', help='CUDA device ID', metavar='INT', type=int, default=0, show_default=True)

# def main(network_pkl, image_size, outdir, image_dir, name, views, blursize, scale,channels, sigma, pad, psize,
#          device=device, **sampler_kwargs):
#     # Load network.
#     print("=== GPU INFORMATION ===")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
    
#     if torch.cuda.is_available():
#         print(f"Current device: {device}")
#         print(f"Current GPU ID: {torch.cuda.current_device()}")
#         print(f"GPU name: {torch.cuda.get_device_name(device)}")
#         print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
#         print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
#         print(f"Memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
#     # Check environment variables
#     import os
#     print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
#     print("========================")
    
#     # Load network.
#     print(f'Loading network from "{network_pkl}"...')
#     with dnnlib.util.open_url(network_pkl, verbose=False) as f:
#         net = pickle.load(f)['ema'].to(device)
    
#     # Check which device the network is on
#     print(f"Network loaded on device: {next(net.parameters()).device}")
    
#     print(f'Loading network from "{network_pkl}"...')
#     with dnnlib.util.open_url(network_pkl, verbose=False) as f:
#         net = pickle.load(f)['ema'].to(device)

#     files = os.listdir(image_dir)
#     png_files = [file for file in files if file.endswith('.png')]
#     print(f"Found {len(png_files)} image files: {png_files[:5]}")  # Show first 5 files
#     # png_files = png_files[:1]

#     inverseop = InverseOperator(image_size, name, views=views, channels=channels, blursize=blursize, scale_factor=scale)

#     x_start = 0
#     y_start = 0
#     resolution = image_size + 2*pad
#     x_pos = torch.arange(x_start, x_start+resolution).view(1, -1).repeat(resolution, 1)
#     y_pos = torch.arange(y_start, y_start+resolution).view(-1, 1).repeat(1, resolution)
#     x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
#     y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
#     latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
#     latents_pos = latents_pos.unsqueeze(0).repeat(1, 1, 1, 1)

#     allclean = np.zeros((len(png_files), image_size, image_size, channels))
#     allrecon = np.zeros((len(png_files), image_size, image_size, channels))
#     print(f'Generating images to "{outdir}"...')
#     totpsnr = 0
#     totssim = 0
#     psnrarr = []
#     ssimarr = []

#     for loop in tqdm.tqdm(range(len(png_files))):
#         clean = PIL.Image.open(os.path.join(image_dir, png_files[loop]))
#         clean = np.asarray(clean)/255
#         if channels == 1:
#             clean = np.expand_dims(clean, 0)
#         elif channels == 3:
#             clean = np.transpose(clean, (2,0,1))
#         print(clean.min(), clean.max())
#         print('clean shape: ', clean.shape)

#         print(f'Now doing image "{png_files[loop]}"')

#         xclean = torch.from_numpy(clean).to(device=device)
#         noisy_y = inverseop.A(xclean)
#         print('clean: ', xclean.shape)
#         print('noisy: ', noisy_y.shape)
#         noisy_y = noisy_y + sigma*torch.randn_like(noisy_y)
#         # scipy.io.savemat('proj.mat', {'proj': noisy_y.cpu().numpy()})

#         latents = torch.randn([1, channels, image_size, image_size], device=device)

#         sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
#         images = dps(net, latents, latents_pos, inverseop, clean=clean, noisy=noisy_y, pad=pad, psize=psize, **sampler_kwargs)

#         images = torch.clamp(images, min=0, max=1)
#         images = images[:, pad:pad+image_size, pad:pad+image_size]
#         images = torch.permute(images, (1,2,0))
#         images = images.cpu().numpy()
#         cleantmp = np.transpose(clean, (1,2,0))
#         thispsnr = psnr(images, cleantmp, data_range=1)
#         print('psnr for this image: ', thispsnr)
#         myssim = ssim(images, cleantmp, channel_axis=2, data_range=1)
#         print('ssim for this image: ', myssim)
#         totpsnr += thispsnr
#         totssim += myssim
#         psnrarr.append(thispsnr)
#         ssimarr.append(myssim)

#         allclean[loop, :,:,:] = cleantmp
#         allrecon[loop,:,:,:] = images

#     print('average psnr: ', totpsnr/(len(png_files)))
#     print('average ssim: ', totssim/(len(png_files)))
#     # Save final results
#     np.save(os.path.join(outdir, 'clean_images.npy'), allclean)
#     np.save(os.path.join(outdir, 'reconstructed_images.npy'), allrecon)
#     np.save(os.path.join(outdir, 'psnr_values.npy'), np.array(psnrarr))
#     np.save(os.path.join(outdir, 'ssim_values.npy'), np.array(ssimarr))



def main(network_pkl, image_size, outdir, image_dir, name, views, blursize, scale,channels, sigma, pad, psize,
         device=device, **sampler_kwargs):
    # Load network.
    print("=== GPU INFORMATION ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current device: {device}")
        print(f"Current GPU ID: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
    # Check environment variables
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print("========================")
    
    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)
    
    # Check which device the network is on
    print(f"Network loaded on device: {next(net.parameters()).device}")
    
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)

    files = os.listdir(image_dir)
    png_files = [file for file in files if file.endswith('.png')]
    print(f"Found {len(png_files)} image files: {png_files[:5]}")  # Show first 5 files
    # png_files = png_files[:1]

    inverseop = InverseOperator(image_size, name, views=views, channels=channels, blursize=blursize, scale_factor=scale)

    x_start = 0
    y_start = 0
    resolution = image_size + 2*pad
    x_pos = torch.arange(x_start, x_start+resolution).view(1, -1).repeat(resolution, 1)
    y_pos = torch.arange(y_start, y_start+resolution).view(-1, 1).repeat(1, resolution)
    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
    latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
    latents_pos = latents_pos.unsqueeze(0).repeat(1, 1, 1, 1)
    
    allclean = np.zeros((len(png_files), image_size, image_size, channels))
    allrecon = np.zeros((len(png_files), image_size, image_size, channels))
    all_intermediate_steps = []  # Store all intermediate steps
    
    print(f'Generating images to "{outdir}"...')
    totpsnr = 0
    totssim = 0
    psnrarr = []
    ssimarr = []

    for loop in tqdm.tqdm(range(len(png_files))):
        clean = PIL.Image.open(os.path.join(image_dir, png_files[loop]))
        clean = np.asarray(clean)/255
        if channels == 1:
            clean = np.expand_dims(clean, 0)
        elif channels == 3:
            clean = np.transpose(clean, (2,0,1))
        print(clean.min(), clean.max())
        print('clean shape: ', clean.shape)

        print(f'Now doing image "{png_files[loop]}"')

        xclean = torch.from_numpy(clean).to(device=device)
        noisy_y = inverseop.A(xclean)
        print('clean: ', xclean.shape)
        print('noisy: ', noisy_y.shape)
        noisy_y = noisy_y + sigma*torch.randn_like(noisy_y)

        latents = torch.randn([1, channels, image_size, image_size], device=device)

        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        
        # Get both final result and intermediate steps
        images, intermediate_steps = dps(net, latents, latents_pos, inverseop, clean=clean, noisy=noisy_y, 
                                       pad=pad, psize=psize, save_steps=True, **sampler_kwargs)

        # Store intermediate steps for this image
        image_steps = {
            'image_name': png_files[loop],
            'steps': intermediate_steps
        }
        all_intermediate_steps.append(image_steps)

        images = torch.clamp(images, min=0, max=1)
        images = images[:, pad:pad+image_size, pad:pad+image_size]
        images = torch.permute(images, (1,2,0))
        images = images.cpu().numpy()
        cleantmp = np.transpose(clean, (1,2,0))
        thispsnr = psnr(images, cleantmp, data_range=1)
        print('psnr for this image: ', thispsnr)
        myssim = ssim(images, cleantmp, channel_axis=2, data_range=1)
        print('ssim for this image: ', myssim)
        totpsnr += thispsnr
        totssim += myssim
        psnrarr.append(thispsnr)
        ssimarr.append(myssim)

        allclean[loop, :,:,:] = cleantmp
        allrecon[loop,:,:,:] = images

    print('average psnr: ', totpsnr/(len(png_files)))
    print('average ssim: ', totssim/(len(png_files)))
    
    # Save final results
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'clean_images.npy'), allclean)
    np.save(os.path.join(outdir, 'reconstructed_images.npy'), allrecon)
    np.save(os.path.join(outdir, 'psnr_values.npy'), np.array(psnrarr))
    np.save(os.path.join(outdir, 'ssim_values.npy'), np.array(ssimarr))
    
    # Save intermediate steps
    # import pickle
    # with open(os.path.join(outdir, 'intermediate_steps.pkl'), 'wb') as f:
    #     pickle.dump(all_intermediate_steps, f)
    
    # Also save intermediate steps as numpy arrays for easier loading
    for img_idx, img_steps in enumerate(all_intermediate_steps):
        img_name = img_steps['image_name'].replace('.png', '')
        steps_array = np.array([step['image'] for step in img_steps['steps']])
        step_numbers = [step['step'] for step in img_steps['steps']]
        sigmas = [step['sigma'] for step in img_steps['steps']]
        
        np.save(os.path.join(outdir, f'{img_name}_intermediate_steps.npy'), steps_array)
        np.save(os.path.join(outdir, f'{img_name}_step_numbers.npy'), np.array(step_numbers))
        np.save(os.path.join(outdir, f'{img_name}_sigmas.npy'), np.array(sigmas))
    
    print(f"Saved intermediate steps for {len(all_intermediate_steps)} images")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
