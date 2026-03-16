"""Animation utilities for the Programmable Cubes challenge.

Provides a function to apply a chromosome step-by-step to a `ProgrammableCubes`
ensemble and save the resulting sequence as a GIF (or matplotlib animation).

Usage (from Python):
	from animation import generate_gif_from_chromosome
	generate_gif_from_chromosome(chromosome, problem_name='ISS', out_path='anim.gif')

"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
try:
	from tqdm import tqdm
except Exception:
	def tqdm(x, **kwargs):
		return x
from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes


def _coords_to_dense(positions, plot_dim, shift):
	"""Convert an array of (N,3) positions into a dense boolean voxel grid.

	`shift` is applied to all positions (positions + shift) so all frames share the
	same coordinate origin. Values outside the `plot_dim` box are ignored.
	"""
	grid = np.zeros((plot_dim, plot_dim, plot_dim), dtype=bool)
	if positions is None or len(positions) == 0:
		return grid
	for p in positions.astype(int):
		i, j, k = p + shift
		if 0 <= i < plot_dim and 0 <= j < plot_dim and 0 <= k < plot_dim:
			grid[i, j, k] = True
	return grid


def generate_gif_from_chromosome(chromosome, problem_name, out_path='animation.gif', interval=500, dpi=100, cube_types_to_plot=None, framecnt=100):
	"""Apply `chromosome` step-by-step and save a GIF of the ensemble reconfiguration.

	Args:
		chromosome: sequence-like of ints encoding [cube_id, move_id, ... , -1].
		problem_name: name of the problem (folder in `problems/`), e.g. 'ISS', 'WALL'.
		out_path: output filepath for the GIF.
		interval: delay between frames in milliseconds.
		dpi: resolution for saved GIF.
		cube_types_to_plot: list of cube types to render. If None, render all defined types.
	"""
	chrom = np.array(chromosome, dtype=int)
	udp = programmable_cubes_UDP(problem_name)

	# Load initial configuration explicitly (same used by UDP.fitness)
	init_config = np.load(f"{udp.setup['path']}/Initial_Config.npy")
	cubes = ProgrammableCubes(init_config)

	# Collect frames: start with initial placement
	frames = [cubes.cube_position.copy()]

	# Determine end of chromosome
	if -1 in chrom:
		chrom_end = int(np.where(chrom == -1)[0][0] // 2)
	else:
		chrom_end = int(len(chrom) // 2)

	skip = np.floor(chrom_end/framecnt)
	if skip == 0:
		skip = 1
	legal_count = 0
	for i in tqdm(range(chrom_end), desc='Applying moves', unit='move'):
		cube_id = int(chrom[2 * i])
		move = int(chrom[2 * i + 1])
		# apply single update; record only when legal (move applied)
		done = cubes.apply_single_update_step(cube_id, move, step=i, verbose=False)
		if done == 1:
			legal_count += 1
			if legal_count % skip == 0:
				frames.append(cubes.cube_position.copy())

	# Prepare plotting parameters
	plot_dim = udp.setup.get('plot_dim', 50)
	if cube_types_to_plot is None:
		cube_types_to_plot = list(np.unique(udp.initial_cube_types))

	colours = udp.setup.get('colours', ['yellow'])
	cube_types = np.array(udp.initial_cube_types)

	# If we have too many frames, downsample uniformly to max_frames
	#if max_frames is not None and len(frames) > max_frames:
	#	idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
	#	frames = [frames[i] for i in idx]

	# Build animation using FuncAnimation and save with PillowWriter (gif)
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(1, 1, 1, projection='3d')

	# Compute global shift so voxels don't jump between frames
	all_pos = np.vstack(frames)
	minp = all_pos.min(axis=0).astype(int)
	shift = -minp

	def _render_frame(positions):
		ax.clear()
		ax.set_facecolor('white')
		for t in cube_types_to_plot:
			tensor = _coords_to_dense(positions[cube_types == t], plot_dim, shift)
			if np.any(tensor):
				ax.voxels(tensor, facecolor=colours[int(t) % len(colours)], edgecolor='k', alpha=1)
		ax.axis('off')
		ax.set_aspect('equal')
		# fix axis limits for consistent view
		ax.set_xlim(0, plot_dim)
		ax.set_ylim(0, plot_dim)
		ax.set_zlim(0, plot_dim)

	def update(k):
		_render_frame(frames[k])

	ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)

	# Ensure output directory exists
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

	writer = PillowWriter(fps=max(1, int(1000 / interval)))
	ani.save(out_path, writer=writer, dpi=dpi)
	plt.close(fig)


if __name__ == '__main__':
	import argparse

	p = argparse.ArgumentParser(description='Generate GIF from chromosome for Programmable Cubes')
	p.add_argument('problem', help='Problem name (folder in problems/), e.g. ISS')
	p.add_argument('chromosome', help='Path to a .npy file containing the chromosome (or a comma-separated list)')
	p.add_argument('--framecnt', '-fcnt', default=100, help='How many frames to have')
	p.add_argument('--out', '-o', default='animation.gif', help='Output GIF path')
	p.add_argument('--interval', type=int, default=100, help='Frame interval in ms')
	args = p.parse_args()

	# Load chromosome: try .npy first, else parse comma-separated list
	chrom_input = args.chromosome
	if os.path.exists(chrom_input):
		chrom = np.load(chrom_input)
	else:
		chrom = np.array([x for x in chrom_input.split(',')], dtype=int)

	generate_gif_from_chromosome(chrom, args.problem, out_path=args.out, interval=args.interval)
