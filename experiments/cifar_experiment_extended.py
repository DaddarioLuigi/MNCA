import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import pickle
from PIL import Image
from torchvision import datasets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mix_NCA.utils_images import train_nca, standard_update_net
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.RobustnessAnalysis import RobustnessAnalysis


def load_samples(data_dir='../data', category_idx=0, seed=42):
    """Load one sample from a CIFAR-10 category as RGBA array."""
    np.random.seed(seed)

    dataset = datasets.CIFAR10(
        root=data_dir,
        download=True
    )

    categories = sorted(dataset.classes)
    if category_idx >= len(categories):
        raise ValueError(f"Category index {category_idx} out of range (0-{len(categories)-1})")

    selected_category = categories[category_idx]

    samples = []
    sample_indices = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if categories[label] == selected_category:
            img_np = np.array(img) / 255.0
            rgba = np.zeros((*img_np.shape[:2], 4))
            rgba[..., :3] = img_np
            rgba[..., 3] = 1.
            samples.append(rgba)
            sample_indices.append(idx)
            if len(samples) == 1:
                break

    print(f"Loaded {len(samples)} samples from category: {selected_category}")
    return samples, sample_indices, [selected_category] * len(samples)


def process_image(img_array, target_size=50, padding=6):
    """Convert numpy array to padded tensor."""
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    if img_tensor.shape[-2:] != (target_size, target_size):
        img_tensor = torch.nn.functional.interpolate(
            img_tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

    padded_tensor = torch.nn.functional.pad(
        img_tensor,
        (padding, padding, padding, padding),
        mode='constant',
        value=0
    )
    return padded_tensor


def run_experiment(sample_idx, samples, class_names, output_dir, neighborhood_sizes,
                   total_steps=8000, print_every=200, n_runs=50, robust_steps=200,
                   pool_size=1000, batch_size=8):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TARGET_SIZE = 32
    N_CHANNELS = 16
    BATCH_SIZE = batch_size
    POOL_SIZE = pool_size
    NUM_STEPS = [30, 50]
    SEED_LOC = (20, 20)
    GAMMA = 0.2
    DECAY = 3e-5
    DROPOUT = 0.2
    HIDDEN_DIM = 64
    TOTAL_STEPS = total_steps
    PRINT_EVERY = print_every
    SEED = 3
    MILESTONES = [4000, 6000, 7000]
    LEARNING_RATE = 1e-3
    N_RULES = 6

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    class_name = f"{class_names[0]}_{sample_idx}"
    base_dir = os.path.join(output_dir, f"experiment_{class_name}_robustness")
    os.makedirs(base_dir, exist_ok=True)

    target = process_image(samples[sample_idx], target_size=TARGET_SIZE).to(DEVICE)

    for nb_size in neighborhood_sizes:
        print(f"\n=== Neighborhood size: {nb_size}x{nb_size} ===")
        exp_dir = os.path.join(base_dir, f"nb_{nb_size}")
        os.makedirs(exp_dir, exist_ok=True)

        # Models
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

        # Training
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

        # Save loss history
        with open(os.path.join(exp_dir, 'loss_history.json'), 'w') as f:
            json.dump({
                'standard': loss_nca['loss'],
                'mixture': loss_mix_nca['loss'],
                'mixture_noise': loss_gmix['loss']
            }, f)

        # Plot training curves
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

        # Deletion tests
        deletion_sizes = [5, 10]
        deletion_results = {}
        for size in deletion_sizes:
            print(f"Testing deletion size {size} (nb {nb_size})...")
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

        # Noise tests
        noise_levels = [0.1, 0.25]
        noise_results = {}
        for noise_level in noise_levels:
            print(f"Testing noise level {noise_level} (nb {nb_size})...")
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

        # Random pixel removal tests
        pixel_counts = [100, 500]
        results = {}
        for n_pixels in pixel_counts:
            print(f"Testing pixel removal count {n_pixels} (nb {nb_size})...")
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

            fig, ax = robustness_analysis.visualize_stored_results(
                results=results[n_pixels],
                plot_type='error',
                figsize=(10, 5)
            )
            fig.savefig(os.path.join(exp_dir, f'pixel_removal_error_{n_pixels}.png'))
            plt.close()

            fig, ax = robustness_analysis.visualize_stored_results(
                results=results[n_pixels],
                plot_type='trajectories'
            )
            fig.savefig(os.path.join(exp_dir, f'pixel_removal_trajectories_{n_pixels}.png'))
            plt.close()

        # Detailed metrics CSV for this neighborhood size
        print("\nCreating summary CSV for nb {}...".format(nb_size))
        csv_path = os.path.join(exp_dir, 'detailed_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write('Perturbation Type,Model Type,Replicate,Final Error\n')
            all_results = [
                *[(f'Deletion {size}', deletion_results[size]) for size in deletion_results.keys()],
                *[(f'Noise {level}', noise_results[level]) for level in noise_results.keys()],
                *[(f'{n} Masked Pixels', results[n]) for n in results.keys()]
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
    parser.add_argument('--category', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='results_extended')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--neighborhood_sizes', type=str, default='4,5,6,7', help='Comma-separated list, e.g. 4,5,6,7')
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
        if s not in (4, 5, 6, 7):
            raise ValueError(f"Unsupported neighborhood size: {s}. Supported: 4,5,6,7")

    # Defaults
    total_steps = 8000
    print_every = 200
    n_runs = 50
    robust_steps = 200
    pool_size = 1000
    batch_size = 8

    if args.quick:
        total_steps = 500
        print_every = 100
        n_runs = 5
        robust_steps = 100
        pool_size = 256
        batch_size = 4

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

    samples, labels, class_names = load_samples(args.data_dir, args.category)
    for i in range(len(samples)):
        run_experiment(
            i, samples, class_names, args.output_dir, sizes,
            total_steps=total_steps,
            print_every=print_every,
            n_runs=n_runs,
            robust_steps=robust_steps,
            pool_size=pool_size,
            batch_size=batch_size,
        )


