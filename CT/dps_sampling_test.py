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

# --- UNCHANGED HELPER FUNCTIONS (makeFigures, pinv, pc_sampling, etc.) ---
# ... (makeFigures, pinv, pc_sampling, langevin, dps, dps_modified functions are unchanged)

#----------------------------------------------------------------------------
# NEW: Batch-aware measurement conditioning function
def measurement_cond_fn_batch(measurement, x_prev, x0hat, inverseop, pad=24, w=256):
    """
    Calculates the gradient of the data consistency term for a batch of samples.
    """
    # x0hat has shape [B, C, H_pad, W_pad], crop to [B, C, H, W]
    x0hat_cropped = x0hat[:, :, pad:pad+w, pad:pad+w]

    # Assume inverseop.A can handle a batch of images
    # measurement has shape [B, ...], A(x0hat_cropped) will also be [B, ...]
    difference = measurement - inverseop.A(x0hat_cropped).to(dtype=torch.float32)

    # Calculate norm for each sample in the batch
    # difference.shape[0] is the batch size (num_runs)
    norm = torch.linalg.norm(difference.view(difference.shape[0], -1), dim=1)

    # Compute per-sample gradients by taking the gradient of the sum of norms
    # This is a standard and efficient way to compute batch gradients
    norm_grad = torch.autograd.grad(outputs=norm.sum(), inputs=x_prev)[0]
    return norm_grad

#----------------------------------------------------------------------------
# NEW: Batch-aware DPS sampler
def dps_modified_new_update_batch(net, latents, latents_pos, inverseop, noisy=None, num_steps=18,
                                 clean=None, sigma_min=0.005, sigma_max=0.05, rho=7, zeta=0.3, pad=64, psize=64,
                                 S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, save_steps=True):
    """
    Performs DPS sampling on a whole batch of initializations for a single measurement.
    All tensors like `latents`, `noisy`, `x` are expected to have a leading batch dimension of `num_runs`.
    """
    w = latents.shape[-1]
    num_runs = latents.shape[0] # The batch size is the number of runs

    patches = w // psize + 1
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)

    # inverseop.Adagger must be able to handle a batch of noisy measurements
    x_init = torch.clamp(inverseop.Adagger(noisy), min=0, max=1)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Initial noise is generated for the entire batch
    # x = sigma_max * torch.randn_like(x_init, device=latents.device)
    batch_seeds = list(range(num_runs))  # Generate a list of seeds with length num_runs
    if batch_seeds is not None:
        x_list = []
        for i, seed in enumerate(batch_seeds):
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(seed)
            x_single = sigma_max * torch.randn(
                x_init[i:i+1].shape, 
                device=latents.device, 
                dtype=x_init.dtype,
                generator=generator
            )
            x_list.append(x_single)
        x = torch.cat(x_list, dim=0)
    else:
        x = sigma_max * torch.randn_like(x_init, device=latents.device)
    
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", 0).requires_grad_(True)
    
    # measurement_adjustment_steps = []
    diffusion_model_steps = []
    measurement_updates = []
    diffusion_updates = []

    for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps-1, desc="Batch Sampling Steps"):
        alpha = 0.5 * t_cur ** 2
        
        num_langevin_steps = 10
        # These tensors now store updates for the entire batch at each inner step
        measurement_updates_tensor = torch.zeros(num_langevin_steps, num_runs, 1, 256, 256, device=device)
        diffusion_updates_tensor = torch.zeros(num_langevin_steps, num_runs, 1, 256, 256, device=device)

        for j in range(num_langevin_steps):
            # All operations inside this loop are now batch-wise
            D_batch = []
            indices = getIndices(spaced, patches, pad, psize)
            
            for b in range(num_runs):
                # Extract single sample from batch
                x_single = x[b:b+1]  # Keep batch dimension of 1
                latents_pos_single = latents_pos[b:b+1]  # Keep batch dimension of 1
                
                # Call denoisedFromPatches for single sample
                D_single = denoisedFromPatches(net, x_single, t_cur, latents_pos_single, None, indices, t_goal=0, wrong=False)
                D_batch.append(D_single)
            
            # Concatenate results back into batch
            D = torch.cat(D_batch, dim=0)  # Shape: [num_runs, C, H_pad, W_pad]
            score = (D - x) / t_cur ** 2
            z = torch.randn_like(x)

            x_before_measurement = x.clone()
            
            # MEASUREMENT ADJUSTMENT (BATCH)
            x0hat = D
            norm_grad = measurement_cond_fn_batch(noisy, x, x0hat, inverseop, pad=pad, w=w)
            x = (x - zeta * norm_grad)
            measurement_update = x - x_before_measurement

            if save_steps:
                measurement_updates_tensor[j] = measurement_update[:, :, pad:pad+w, pad:pad+w]

            x_before_diffusion = x.clone()

            # DIFFUSION MODEL UPDATE (BATCH)
            if i < num_steps - 1:
                x = (x + alpha / 2 * score + torch.sqrt(alpha) * z).requires_grad_(True)
            else:
                x = (x + alpha / 2 * score).requires_grad_(True)
            
            # diffusion_update = x - x_before_diffusion
            
            if save_steps:
                diffusion_updates_tensor[j] = alpha / 2 * score[:, :, pad:pad+w, pad:pad+w]
        
        # Save intermediate steps for the batch at the end of inner iterations
        if save_steps and j == num_langevin_steps - 1:
            with torch.no_grad():
                # measurement_result = x[:, :, pad:pad+w, pad:pad+w].detach().cpu().numpy()
                diffusion_result = x[:, :, pad:pad+w, pad:pad+w].detach().cpu().numpy()
                
                # measurement_adjustment_steps.append({'step': i, 'sigma': t_cur.item(), 'image_batch': measurement_result})
                diffusion_model_steps.append({'step': i, 'sigma': t_cur.item(), 'update_batch': diffusion_result})
                measurement_updates.append({'step': i, 'sigma': t_cur.item(), 'update_batch': measurement_updates_tensor.sum(dim=0).detach().cpu().numpy()})
                diffusion_updates.append({'step': i, 'sigma': t_cur.item(), 'update_batch': diffusion_updates_tensor.sum(dim=0).detach().cpu().numpy()})

    return x.detach(), {
        # 'measurement_adjustment_steps': measurement_adjustment_steps,
        'diffusion_model_steps': diffusion_model_steps,
        'measurement_updates': measurement_updates,
        'diffusion_updates': diffusion_updates
    }

# ... (StackedRandomGenerator, parse_int_list, set_requires_grad are unchanged)
@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--image_dir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--image_size',                help='Sample resolution', metavar='INT',                                 type=int, default=None)
@click.option('--pad',                help='Pad width', metavar='INT',                                 type=int, default=None)
@click.option('--psize',                help='Patch size', metavar='INT',                                 type=int, default=None)
@click.option('--views',                help='Number of CT views', metavar='INT',                                type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--blursize',                help='Size of blur kernel', metavar='INT',                                type=click.IntRange(min=1), default=31, show_default=True)
@click.option('--channels',                help='Image channels', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--name',                  help='Experiment type', metavar='ct_parbeam|ct_fanbeam|denoise',             type=click.Choice(['ct_parbeam', 'ct_fanbeam', 'lact', 'denoise', 'deblur_uniform', 'super']))
@click.option('--sigma',                help='Noise of measurement', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--scale',                help='Superresolution scale', metavar='INT',                                type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
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
@click.option('--num_runs', 'num_runs',    help='Number of sampling runs per measurement (run as a batch)', metavar='INT', type=click.IntRange(min=1), default=5, show_default=True)



def main(network_pkl, image_size, outdir, image_dir, name, views, blursize, scale, channels, sigma, pad, psize,
         num_runs, device=device, **sampler_kwargs):
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)

    files = os.listdir(image_dir)
    png_files = [file for file in files if file.endswith('.png')]
    print(f"Found {len(png_files)} image files.")

    inverseop = InverseOperator(image_size, name, views=views, channels=channels, blursize=blursize, scale_factor=scale)

    resolution = image_size + 2 * pad
    x_pos = torch.arange(0, resolution).view(1, -1).repeat(resolution, 1)
    y_pos = torch.arange(0, resolution).view(-1, 1).repeat(1, resolution)
    latents_pos_single = torch.stack([(x_pos / (resolution - 1) - 0.5) * 2., (y_pos / (resolution - 1) - 0.5) * 2.], dim=0).to(device)
    latents_pos_single = latents_pos_single.unsqueeze(0) # Shape: [1, 2, H_pad, W_pad]

    allclean = np.zeros((len(png_files), image_size, image_size, channels))
    allrecon_mean = np.zeros((len(png_files), image_size, image_size, channels))
    
    print(f'Generating images to "{outdir}" using a batch size of {num_runs} per image...')
    
    psnr_all_runs_avg_list = []
    ssim_all_runs_avg_list = []
    psnr_mean_recon_list = []
    ssim_mean_recon_list = []

    for loop_idx, filename in enumerate(tqdm.tqdm(png_files, desc="Processing Images")):
        clean = PIL.Image.open(os.path.join(image_dir, filename))
        clean = np.asarray(clean) / 255.0
        if channels == 1:
            clean = np.expand_dims(clean, 0)
        else:
            clean = np.transpose(clean, (2, 0, 1))
        
        cleantmp_transposed = np.transpose(clean, (1, 2, 0))

        print(f'\nProcessing "{filename}" ({loop_idx+1}/{len(png_files)})')

        xclean = torch.from_numpy(clean).to(device=device)
        noisy_y_single = inverseop.A(xclean) + sigma * torch.randn_like(inverseop.A(xclean))
        print(f"Debug - noisy_y_single shape: {noisy_y_single.shape}")

        # --- PREPARE BATCH FOR `num_runs` ---
        # The batch dimension will now represent the different sampling runs
        latents_batch = torch.randn([num_runs, channels, image_size, image_size], device=device)
        latents_pos_batch = latents_pos_single.repeat(num_runs, 1, 1, 1)
        noisy_y_batch = noisy_y_single.unsqueeze(0).expand(num_runs, *noisy_y_single.shape)
        print(f"Debug - noisy_y_batch shape: {noisy_y_batch.shape}")

        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        
        # --- RUN SAMPLING ON THE ENTIRE BATCH AT ONCE ---
        images_batch, step_data = dps_modified_new_update_batch(net, latents_batch, latents_pos_batch, inverseop,
                                                                clean=clean, noisy=noisy_y_batch, pad=pad, psize=psize,
                                                                save_steps=True, **sampler_kwargs)
        
        # --- AGGREGATE AND ANALYZE BATCH RESULTS ---
        images_batch = torch.clamp(images_batch, min=0, max=1)[:, :, pad:pad+image_size, pad:pad+image_size]
        
        # Calculate mean reconstruction
        mean_image = torch.mean(images_batch, dim=0) # Shape: [C, H, W]
        mean_image_np = mean_image.permute(1, 2, 0).cpu().numpy()
        
        # Calculate metrics for the mean reconstruction
        psnr_mean = psnr(mean_image_np, cleantmp_transposed, data_range=1)
        ssim_mean = ssim(mean_image_np, cleantmp_transposed, channel_axis=2, data_range=1)
        psnr_mean_recon_list.append(psnr_mean)
        ssim_mean_recon_list.append(ssim_mean)

        # Calculate metrics for each individual run in the batch
        run_psnrs, run_ssims = [], []
        for i in range(num_runs):
            run_img_np = images_batch[i].permute(1, 2, 0).cpu().numpy()
            run_psnrs.append(psnr(run_img_np, cleantmp_transposed, data_range=1))
            run_ssims.append(ssim(run_img_np, cleantmp_transposed, channel_axis=2, data_range=1))
        
        psnr_all_runs_avg_list.append(np.mean(run_psnrs))
        ssim_all_runs_avg_list.append(np.mean(run_ssims))

        print(f"  Metrics for '{filename}':")
        if num_runs > 1:
            print(f"    Avg PSNR across {num_runs} runs: {np.mean(run_psnrs):.4f} (Std: {np.std(run_psnrs):.4f})")
            print(f"    Avg SSIM across {num_runs} runs: {np.mean(run_ssims):.4f} (Std: {np.std(run_ssims):.4f})")
        print(f"    PSNR of Mean Reconstruction: {psnr_mean:.4f}")
        print(f"    SSIM of Mean Reconstruction: {ssim_mean:.4f}")
        
        allclean[loop_idx, :, :, :] = cleantmp_transposed
        allrecon_mean[loop_idx, :, :, :] = mean_image_np
        
        # --- SAVE BATCH RESULTS ---
        os.makedirs(outdir, exist_ok=True)
        img_name_base = filename.replace('.png', '')
        np.save(os.path.join(outdir, f'{img_name_base}_mean_recon.npy'), mean_image_np)

        # Complete the saving logic for measurement_updates and diffusion_updates
        for run_idx in range(num_runs):
            run_img_name = f'{img_name_base}_run{run_idx}'
            
            if 'measurement_updates' in step_data and step_data['measurement_updates']:
            # Stack updates from all steps: shape becomes (num_steps, num_runs, C, H, W)
                all_updates = np.array([s['update_batch'] for s in step_data['measurement_updates']])
            
                # Calculate the mean across the 'num_runs' dimension (axis=1)
                mean_updates = np.mean(all_updates, axis=1)  # Shape: (num_steps, C, H, W)
                
                # Save the mean updates
                np.save(os.path.join(outdir, f'{img_name_base}_mean_measurement_updates.npy'), mean_updates)

                # Save the sigma values (only needs to be done once)
                sigma_values = np.array([s['sigma'] for s in step_data['measurement_updates']])
                np.save(os.path.join(outdir, f'{img_name_base}_sigma_values.npy'), sigma_values)

            if 'diffusion_updates' in step_data and step_data['diffusion_updates']:
            # Stack updates from all steps: shape becomes (num_steps, num_runs, C, H, W)
                all_updates = np.array([s['update_batch'] for s in step_data['diffusion_updates']])
            
                # Calculate the mean across the 'num_runs' dimension (axis=1)
                mean_updates = np.mean(all_updates, axis=1)  # Shape: (num_steps, C, H, W)
                
                # Save the mean updates
                np.save(os.path.join(outdir, f'{img_name_base}_mean_diffusion_updates.npy'), mean_updates)

                # Save the sigma values (only needs to be done once)
                # sigma_values = np.array([s['sigma'] for s in step_data['diffusion_updates']])
                # np.save(os.path.join(outdir, f'{img_name_base}_sigma_values.npy'), sigma_values)
            if 'diffusion_model_steps' in step_data and step_data['diffusion_model_steps']:
            # Stack updates from all steps: shape becomes (num_steps, num_runs, C, H, W)
                all_updates = np.array([s['update_batch'] for s in step_data['diffusion_model_steps']])
            
                # Calculate the mean across the 'num_runs' dimension (axis=1)
                mean_updates = np.mean(all_updates, axis=1)  # Shape: (num_steps, C, H, W)
                
                # Save the mean updates
                np.save(os.path.join(outdir, f'{img_name_base}_mean_diffusion_steps.npy'), mean_updates)
            # Only save measurement_updates since that's what we have
            # if 'measurement_updates' in step_data and step_data['measurement_updates']:
            #     mu_array = np.array([s['update_batch'][run_idx, ...] for s in step_data['measurement_updates']])
            #     print(mu_array.shape)
            #     np.save(os.path.join(outdir, f'{run_img_name}_measurement_updates.npy'), mu_array)
            # # Save the sigma values for each step
            #     sigma_values = np.array([s['sigma'] for s in step_data['measurement_updates']])
            #     np.save(os.path.join(outdir, f'{run_img_name}_sigma_values.npy'), sigma_values)

    print("\n--- OVERALL RESULTS ---")
    print(f"Average PSNR of Mean Reconstructions: {np.mean(psnr_mean_recon_list):.4f}")
    print(f"Average SSIM of Mean Reconstructions: {np.mean(ssim_mean_recon_list):.4f}")
    if num_runs > 1:
        print(f"Overall Average of individual run PSNRs: {np.mean(psnr_all_runs_avg_list):.4f}")
        print(f"Overall Average of individual run SSIMs: {np.mean(ssim_all_runs_avg_list):.4f}")

    # np.save(os.path.join(outdir, 'clean_images.npy'), allclean)
    np.save(os.path.join(outdir, 'reconstructed_mean_images.npy'), allrecon_mean)
    np.save(os.path.join(outdir, 'psnr_values_mean_recon.npy'), np.array(psnr_mean_recon_list))
    # np.save(os.path.join(outdir, 'ssim_values_mean_recon.npy'), np.array(ssim_mean_recon_list))
    print(f"\nSaved all final and intermediate results to '{outdir}'")

if __name__ == "__main__":
    main()