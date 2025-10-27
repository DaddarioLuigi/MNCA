import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import json
import sys
import pickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mix_NCA.utils_images import train_nca, standard_update_net
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.RobustnessAnalysis import RobustnessAnalysis


def load_emoji(path, target_size=40, padding=6):
    img = Image.open(path).convert('RGBA')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    padded_tensor = torch.nn.functional.pad(
        img_tensor,
        (padding, padding, padding, padding),
        mode='constant',
        value=0
    )
    return padded_tensor.float()


def download_emoji(emoji_code, save_dir='data/emojis/'):
    os.makedirs(save_dir, exist_ok=True)
    url = f'https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/{emoji_code.lower()}.png'
    save_path = os.path.join(save_dir, f'{emoji_code}.png')

    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download emoji: {response.status_code}")


def run_experiment(emoji_code, output_dir, neighborhood_sizes,
                   total_steps=8000, print_every=200, n_runs=50, robust_steps=100,
                   pool_size=1000, batch_size=8):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TARGET_SIZE = 40
    N_CHANNELS = 16
    BATCH_SIZE = batch_size
    POOL_SIZE = pool_size
    NUM_STEPS = [30, 50]
    SEED_LOC = (20, 20)
    GAMMA = 0.2
    DECAY = 3e-5
    DROPOUT = 0.2
    HIDDEN_DIM = 128
    TOTAL_STEPS = total_steps
    PRINT_EVERY = print_every
    SEED = 3
    MILESTONES = [4000, 6000, 7000]
    LEARNING_RATE = 1e-3
    N_RULES = 6

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    base_dir = os.path.join(output_dir, f"experiment_{emoji_code}_robustness")
    os.makedirs(base_dir, exist_ok=True)

    emoji_path = download_emoji(emoji_code)
    target = load_emoji(emoji_path, target_size=TARGET_SIZE).to(DEVICE)

    for nb_size in neighborhood_sizes:
        print(f"\n=== Neighborhood size: {nb_size}x{nb_size} ===")
        exp_dir = os.path.join(base_dir, f"NB_{nb_size}")
        os.makedirs(exp_dir, exist_ok=True)

        model = ExtendedNCA(
            update_net=standard_update_net(N_CHANNELS * 3, HIDDEN_DIM, N_CHANNELS, device=DEVICE),
            state_dim=N_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            device=DEVICE,
            filter_type="sobel",
            neighborhood_size=nb_size,
        )

        model_mix = ExtendedMixtureNCA(
            update_nets=standard_update_net,
            state_dim=N_CHANNELS,
            num_rules=N_RULES,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            device=DEVICE,
            temperature=1.0,
            neighborhood_size=nb_size,
        )

        model_gmix = ExtendedMixtureNCANoise(
            update_nets=standard_update_net,
            state_dim=N_CHANNELS,
            num_rules=N_RULES,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            device=DEVICE,
            temperature=1.0,
            neighborhood_size=nb_size,
        )

        print("Training standard model...")
        loss_nca = train_nca(
            model,
            target,
            device=DEVICE,
            num_steps=NUM_STEPS,
            milestones=MILESTONES,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            decay=DECAY,
            total_steps=TOTAL_STEPS,
            print_every=PRINT_EVERY,
            batch_size=BATCH_SIZE,
            state_dim=N_CHANNELS,
            seed_loc=SEED_LOC,
            pool_size=POOL_SIZE,
        )

        print("Training mixture model...")
        loss_mix_nca = train_nca(
            model_mix,
            target,
            device=DEVICE,
            num_steps=NUM_STEPS,
            milestones=MILESTONES,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            decay=DECAY,
            total_steps=TOTAL_STEPS,
            print_every=PRINT_EVERY,
            batch_size=BATCH_SIZE,
            state_dim=N_CHANNELS,
            seed_loc=SEED_LOC,
            pool_size=POOL_SIZE,
            temperature=1,
            min_temperature=1,
            anneal_rate=0.001,
            straight_through=True,
        )

        print("Training stochastic mixture model...")
        loss_gmix = train_nca(
            model_gmix,
            target,
            device=DEVICE,
            num_steps=NUM_STEPS,
            milestones=MILESTONES,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            decay=DECAY,
            total_steps=TOTAL_STEPS,
            print_every=PRINT_EVERY,
            batch_size=BATCH_SIZE,
            state_dim=N_CHANNELS,
            seed_loc=SEED_LOC,
            pool_size=POOL_SIZE,
            temperature=1,
            min_temperature=1,
            anneal_rate=0.001,
            straight_through=True,
        )

        # Save models
        torch.save(model.state_dict(), os.path.join(exp_dir, 'standard_model.pt'))
        torch.save(model_mix.state_dict(), os.path.join(exp_dir, 'mixture_model.pt'))
        torch.save(model_gmix.state_dict(), os.path.join(exp_dir, 'mixture_model_noise.pt'))

        with open(os.path.join(exp_dir, 'loss_history.json'), 'w') as f:
            json.dump({
                'standard': loss_nca['loss'],
                'mixture': loss_mix_nca['loss'],
                'mixture_noise': loss_gmix['loss']
            }, f)

        plt.figure(figsize=(10, 5))
        plt.plot(loss_nca['loss'], label='Standard NCA')
        plt.plot(loss_mix_nca['loss'], label='Mixture NCA')
        plt.plot(loss_gmix['loss'], label='Stochastic Mixture NCA')
        plt.legend()
        plt.yscale('log')
        plt.title(f'Training Loss - nb {nb_size}')
        plt.savefig(os.path.join(exp_dir, 'training_loss.png'))
        plt.close()

        # Robustness analysis
        init_state = target.to(DEVICE)
        init_state[..., 3:, SEED_LOC[0], SEED_LOC[1]] = 1
        init_state_nca = torch.cat([
            init_state,
            torch.zeros(1, N_CHANNELS - 4, *init_state.shape[-2:]).to(DEVICE)
        ], dim=1)

        robustness_analysis = RobustnessAnalysis(
            standard_nca=model.to(DEVICE),
            mixture_nca=model_mix.to(DEVICE),
            stochastic_mixture_nca=model_gmix.to(DEVICE),
            device=DEVICE
        )

        deletion_sizes = [5, 10]
        deletion_results = {}
        print("\nTesting deletion robustness...")
        for size in deletion_sizes:
            deletion_results[size] = robustness_analysis.compute_robustness_metrics(
                init_state_nca,
                'deletion',
                n_runs=n_runs,
                size=size,
                steps=robust_steps,
                seed=SEED_LOC
            )
            with open(os.path.join(exp_dir, f'deletion_{size}.pkl'), 'wb') as f:
                pickle.dump(deletion_results[size], f)

            fig, ax = robustness_analysis.visualize_stored_results(
                results=deletion_results[size],
                plot_type='error',
                figsize=(10, 5)
            )
            fig.savefig(os.path.join(exp_dir, f'deletion_error_{size}.png'))
            plt.close()

            fig, ax = robustness_analysis.visualize_stored_results(
                results=deletion_results[size],
                plot_type='trajectories'
            )
            fig.savefig(os.path.join(exp_dir, f'deletion_trajectories_{size}.png'))
            plt.close()

        noise_levels = [0.1, 0.25]
        noise_results = {}
        print("\nTesting noise robustness...")
        for noise_level in noise_levels:
            noise_results[noise_level] = robustness_analysis.compute_robustness_metrics(
                init_state_nca,
                'noise',
                n_runs=n_runs,
                noise_level=noise_level,
                steps=robust_steps,
                seed=SEED_LOC
            )
            with open(os.path.join(exp_dir, f'noise_{noise_level}.pkl'), 'wb') as f:
                pickle.dump(noise_results[noise_level], f)

            fig, ax = robustness_analysis.visualize_stored_results(
                results=noise_results[noise_level],
                plot_type='error',
                figsize=(10, 5)
            )
            fig.savefig(os.path.join(exp_dir, f'noise_error_{noise_level}.png'))
            plt.close()

            fig, ax = robustness_analysis.visualize_stored_results(
                results=noise_results[noise_level],
                plot_type='trajectories'
            )
            fig.savefig(os.path.join(exp_dir, f'noise_trajectories_{noise_level}.png'))
            plt.close()

        pixel_counts = [100, 500]
        results = {}
        print("\nTesting pixel removal robustness...")
        for n_pixels in pixel_counts:
            results[n_pixels] = robustness_analysis.compute_robustness_metrics(
                init_state_nca,
                perturbation_type='random_pixels',
                steps=robust_steps,
                seed=SEED_LOC,
                n_runs=n_runs,
                n_pixels=n_pixels
            )
            with open(os.path.join(exp_dir, f'pixel_removal_{n_pixels}.pkl'), 'wb') as f:
                pickle.dump(results[n_pixels], f)

            fig, _ = robustness_analysis.visualize_stored_results(
                results=results[n_pixels],
                plot_type='error',
                figsize=(10, 5)
            )
            fig.savefig(os.path.join(exp_dir, f'pixel_removal_error_{n_pixels}.png'))
            plt.close()

            fig, _ = robustness_analysis.visualize_stored_results(
                results=results[n_pixels],
                plot_type='trajectories'
            )
            fig.savefig(os.path.join(exp_dir, f'pixel_removal_trajectories_{n_pixels}.png'))
            plt.close()

        print("\nCreating detailed summary CSV (per neighborhood size)...")
        csv_path = os.path.join(exp_dir, 'detailed_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write('Perturbation Type,Model Type,Replicate,Final Error\n')
            all_results = [
                *[(f'Deletion {size}', deletion_results[size]) for size in deletion_sizes],
                *[(f'Noise {level}', noise_results[level]) for level in noise_levels],
                *[(f'{n} Masked Pixels', results[n]) for n in pixel_counts]
            ]
            for perturb_name, result in all_results:
                for model_name, metrics in [
                    ('Standard', result['standard_metrics']),
                    ('Mixture', result['mixture_metrics']),
                    ('Stochastic', result['stochastic_metrics'])
                ]:
                    for rep_idx, metric in enumerate(metrics):
                        row = [
                            perturb_name,
                            model_name,
                            str(rep_idx),
                            f"{metric['final_error']:.4f}"
                        ]
                        f.write(','.join(row) + '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--emoji_code', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_extended')
    parser.add_argument('--neighborhood_sizes', type=str, default='3,5,7', help='Comma-separated list, e.g. 3,5,7')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--total_steps', type=int, default=None)
    parser.add_argument('--print_every', type=int, default=None)
    parser.add_argument('--n_runs', type=int, default=None)
    parser.add_argument('--robust_steps', type=int, default=None)
    parser.add_argument('--pool_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.neighborhood_sizes.split(',') if s.strip()]
    for s in sizes:
        if s not in (3, 5, 7):
            raise ValueError(f"Unsupported neighborhood size: {s}. Supported: 3,5,7")

    # Defaults
    total_steps = 8000
    print_every = 200
    n_runs = 50
    robust_steps = 100
    pool_size = 1000
    batch_size = 8

    # Quick profile
    if args.quick:
        total_steps = 500
        print_every = 100
        n_runs = 5
        robust_steps = 50
        pool_size = 256
        batch_size = 4

    # Overrides
    if args.total_steps is not None:
        total_steps = args.total_steps
    if args.print_every is not None:
        print_every = args.print_every
    if args.n_runs is not None:
        n_runs = args.n_runs
    if args.robust_steps is not None:
        robust_steps = args.robust_steps
    if args.pool_size is not None:
        pool_size = args.pool_size
    if args.batch_size is not None:
        batch_size = args.batch_size

    run_experiment(
        args.emoji_code,
        args.output_dir,
        sizes,
        total_steps=total_steps,
        print_every=print_every,
        n_runs=n_runs,
        robust_steps=robust_steps,
        pool_size=pool_size,
        batch_size=batch_size,
    )


