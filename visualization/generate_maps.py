#!/usr/bin/env python3
"""
Generate maps showing the physical layouts for Indoor and Outdoor datasets

Indoor: 4 rooms with 7 APs in a realistic office layout (T-shaped corridor)
Outdoor: 2D terrain map with buildings/height variations, split into 4x4 grid (16 cells)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import PatchCollection

# Config
WORKDIR = "./work_dir"
os.makedirs(WORKDIR, exist_ok=True)

# ========================================
# INDOOR MAP (4 Rooms, 7 APs, T-shaped corridor)
# ========================================

def generate_indoor_map():
    """
    Generate a realistic indoor office layout with T-shaped corridor
    4 rooms of different sizes + 7 APs + furniture
    """
    print("\n=== Generating Indoor Map ===")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Fill background
    ax.add_patch(Rectangle((0, 0), 17, 14, facecolor='white'))
    
    # Draw exterior walls
    exterior_walls = [
        [(0, 0), (17, 0)],      # Bottom
        [(17, 0), (17, 14)],    # Right
        [(17, 14), (0, 14)],    # Top
        [(0, 14), (0, 0)],      # Left
    ]
    for wall in exterior_walls:
        xs, ys = zip(*wall)
        ax.plot(xs, ys, 'k-', linewidth=3, solid_capstyle='round')
    
    # Draw interior walls (creating T-shaped corridor and 4 rooms)
    interior_walls = [
        # Horizontal corridor walls
        [(0, 6), (8, 6)],       # Bottom of upper-left room
        [(0, 9), (8, 9)],       # Top of lower-left room
        [(11, 6), (17, 6)],     # Bottom of upper-right room
        [(11, 9), (17, 9)],     # Top of lower-right room
        
        # Vertical corridor walls
        [(8, 0), (8, 6)],       # Left wall of lower corridor
        [(11, 0), (11, 6)],     # Right wall of lower corridor
        [(8, 9), (8, 14)],      # Left wall of upper corridor
        [(11, 9), (11, 14)],    # Right wall of upper corridor
    ]
    for wall in interior_walls:
        xs, ys = zip(*wall)
        ax.plot(xs, ys, 'k-', linewidth=2.5, solid_capstyle='round')
    
    # Draw doors (small gaps with door swing arcs)
    doors = [
        # (x, y, width, orientation: 'h' or 'v', swing direction)
        (3.5, 6, 1, 'h', 'up'),      # Room 0 to corridor
        (13.5, 6, 1, 'h', 'up'),     # Room 1 to corridor
        (3.5, 9, 1, 'h', 'down'),    # Room 2 to corridor
        (13.5, 9, 1, 'h', 'down'),   # Room 3 to corridor
    ]
    
    for dx, dy, dw, orient, swing in doors:
        if orient == 'h':
            # Draw door arc
            if swing == 'up':
                arc = patches.Arc((dx, dy), dw, dw, angle=0, theta1=0, theta2=90, 
                                 linewidth=1.5, edgecolor='gray', linestyle='--')
            else:
                arc = patches.Arc((dx, dy), dw, dw, angle=0, theta1=270, theta2=360, 
                                 linewidth=1.5, edgecolor='gray', linestyle='--')
            ax.add_patch(arc)
    
    # Room 0 - Conference Room (large, bottom-left)
    # Conference table (larger, more detailed)
    ax.add_patch(Rectangle((1.5, 1.8), 5, 2.8, facecolor='#A0522D', edgecolor='black', linewidth=1.5))
    # Chairs around table (rectangles with backs)
    chair_positions = [(1.2, 2.2), (1.2, 4.0), (2.8, 1.4), (4.2, 1.4), (5.6, 1.4), 
                       (6.8, 2.2), (6.8, 4.0), (4.2, 5.0)]
    for cx, cy in chair_positions:
        # Seat
        ax.add_patch(Rectangle((cx-0.3, cy-0.25), 0.6, 0.5, facecolor='#654321', edgecolor='black', linewidth=1))
        # Back
        ax.add_patch(Rectangle((cx-0.25, cy+0.25), 0.5, 0.15, facecolor='#654321', edgecolor='black', linewidth=1))
    
    # Room 1 - Office (medium, bottom-right)
    # L-shaped desk
    ax.add_patch(Rectangle((11.5, 1.2), 3, 1, facecolor='#A0522D', edgecolor='black', linewidth=1.5))
    ax.add_patch(Rectangle((11.5, 2.2), 1, 2, facecolor='#A0522D', edgecolor='black', linewidth=1.5))
    # Office chair (with wheels)
    ax.add_patch(Rectangle((13.5, 1.5), 0.7, 0.7, facecolor='#2F4F4F', edgecolor='black', linewidth=1))
    ax.add_patch(Circle((13.5, 1.5), 0.15, facecolor='black'))
    ax.add_patch(Circle((14.2, 1.5), 0.15, facecolor='black'))
    # Filing cabinet (with drawers)
    ax.add_patch(Rectangle((15.2, 4), 1.4, 1.2, facecolor='#696969', edgecolor='black', linewidth=1.5))
    for dy in [0.3, 0.6, 0.9]:
        ax.plot([15.3, 16.5], [4+dy, 4+dy], 'k-', linewidth=0.8)
    # Plant in corner
    ax.add_patch(Circle((12, 5), 0.35, facecolor='#228B22', edgecolor='#006400', linewidth=1.5))
    ax.add_patch(Rectangle((11.85, 4.6), 0.3, 0.4, facecolor='#8B4513', edgecolor='black', linewidth=1))
    
    # Room 2 - Office (small, top-left)
    # Desk with computer
    ax.add_patch(Rectangle((1.2, 10.5), 2.5, 1.2, facecolor='#A0522D', edgecolor='black', linewidth=1.5))
    # Monitor on desk
    ax.add_patch(Rectangle((2.2, 11.2), 0.5, 0.4, facecolor='#1C1C1C', edgecolor='black', linewidth=1.5))
    # Chair
    ax.add_patch(Rectangle((2.2, 9.8), 0.6, 0.6, facecolor='#2F4F4F', edgecolor='black', linewidth=1))
    # Bookshelf with shelves
    ax.add_patch(Rectangle((5, 9.8), 0.7, 3.5, facecolor='#8B4513', edgecolor='black', linewidth=1.5))
    for shelf_y in [10.5, 11.2, 11.9, 12.6]:
        ax.plot([5, 5.7], [shelf_y, shelf_y], 'k-', linewidth=1.2)
    
    # Room 3 - Lab (large, top-right)
    # Lab benches with detail
    ax.add_patch(Rectangle((11.5, 9.8), 5, 1.2, facecolor='#708090', edgecolor='black', linewidth=1.5))
    ax.add_patch(Rectangle((11.5, 12.5), 5, 1.2, facecolor='#708090', edgecolor='black', linewidth=1.5))
    # Equipment on benches (microscope, computer, etc)
    # Microscope
    ax.add_patch(Rectangle((12, 10.1), 0.6, 0.7, facecolor='#4169E1', edgecolor='black', linewidth=1.2))
    ax.add_patch(Circle((12.3, 10.9), 0.15, facecolor='#87CEEB', edgecolor='black', linewidth=1))
    # Computer
    ax.add_patch(Rectangle((13.5, 10.1), 0.5, 0.7, facecolor='#2F4F4F', edgecolor='black', linewidth=1.2))
    # Lab equipment boxes
    for ex in [14.5, 15.5]:
        ax.add_patch(Rectangle((ex, 10.2), 0.5, 0.5, facecolor='#FFD700', edgecolor='black', linewidth=1))
    # Storage cabinets under bench
    ax.add_patch(Rectangle((12, 12.7), 2, 0.8, facecolor='#696969', edgecolor='black', linewidth=1.5))
    
    # AP positions (strategically placed on ceiling)
    aps = [
        (9.5, 3, 'AP 0'),       # Lower corridor
        (9.5, 11, 'AP 1'),      # Upper corridor
        (4, 3, 'AP 2'),         # Conference room
        (13.5, 3, 'AP 3'),      # Office (Room 1)
        (3, 11.5, 'AP 4'),      # Office (Room 2)
        (14, 11.5, 'AP 5'),     # Lab (Room 3)
        (4, 7.5, 'AP 6'),       # Central corridor
    ]
    
    # Draw APs as ceiling-mounted access points (realistic shape)
    for x, y, label in aps:
        # Main AP body (round)
        circle = Circle((x, y), 0.25, facecolor='white', edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        # Center LED indicator
        ax.add_patch(Circle((x, y), 0.08, facecolor='#00FF00', edgecolor='black', linewidth=0.5, zorder=11))
        # Mounting bracket (small lines showing it's attached to ceiling)
        ax.plot([x-0.15, x-0.3], [y+0.2, y+0.35], 'k-', linewidth=1.5)
        ax.plot([x+0.15, x+0.3], [y+0.2, y+0.35], 'k-', linewidth=1.5)
        # Antenna indicators (small lines)
        ax.plot([x-0.2, x-0.2], [y-0.25, y-0.45], 'k-', linewidth=1.2)
        ax.plot([x+0.2, x+0.2], [y-0.25, y-0.45], 'k-', linewidth=1.2)
    
    # Add room labels
    room_labels = [
        (4, 5.3, 'Room 0'),      # Conference room
        (14, 5.3, 'Room 1'),     # Office
        (3, 13.3, 'Room 2'),     # Office
        (14, 13.3, 'Room 3'),    # Lab
    ]
    
    for rx, ry, rlabel in room_labels:
        ax.text(rx, ry, rlabel, ha='center', va='center',
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='black', linewidth=1.5, alpha=0.9))
    
    # Styling
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, 14.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(WORKDIR, "indoor_map.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved indoor map to: {output_path}")
    plt.close()
    
    return output_path

# ========================================
# OUTDOOR MAP (Terrain with buildings, 4x4 grid)
# ========================================

def generate_outdoor_map():
    """
    Generate outdoor terrain map with 1 base station and 4x4 grid
    Clean map only - no captions or legends
    """
    print("\n=== Generating Outdoor Map ===")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create terrain height map
    np.random.seed(42)
    terrain_size = 100
    x = np.linspace(0, 100, terrain_size)
    y = np.linspace(0, 100, terrain_size)
    X, Y = np.meshgrid(x, y)
    
    # Generate smooth terrain
    Z = (10 * np.sin(X / 15) * np.cos(Y / 15) + 
         5 * np.sin(X / 8) + 
         3 * np.cos(Y / 12) +
         np.random.randn(terrain_size, terrain_size) * 1.5)
    
    # Plot terrain
    contour = ax.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.7)
    
    # Add buildings with labels
    buildings = [
        # (x, y, width, height, label)
        (10, 15, 12, 15, 'Building A'),
        (30, 10, 15, 10, 'Building B'),
        (58, 20, 20, 18, 'Building C'),
        (15, 55, 10, 12, 'Building D'),
        (42, 60, 18, 15, 'Building E'),
        (68, 65, 15, 20, 'Building F'),
        (20, 80, 12, 10, 'Building G'),
        (60, 83, 15, 12, 'Building H'),
        (82, 15, 10, 20, 'Building I'),
    ]
    
    for x, y, w, h, label in buildings:
        rect = Rectangle((x, y), w, h, linewidth=1.5, 
                         edgecolor='black', facecolor='gray', alpha=0.85, zorder=5)
        ax.add_patch(rect)
        # Add label inside building
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=6)
    
    # Add 4x4 grid
    grid_lines_x = np.linspace(0, 100, 5)
    grid_lines_y = np.linspace(0, 100, 5)
    
    for gx in grid_lines_x:
        ax.axvline(gx, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=10)
    for gy in grid_lines_y:
        ax.axhline(gy, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=10)
    
    # Add ONE base station - away from buildings, in open area
    bs_x, bs_y = 8, 42  # Open area in left-middle grid
    
    # Draw as tall triangle (simple cell tower representation)
    tower_width = 3
    tower_height = 12
    
    # Triangle points: base left, base right, top center
    triangle = plt.Polygon([
        (bs_x - tower_width/2, bs_y),
        (bs_x + tower_width/2, bs_y),
        (bs_x, bs_y + tower_height)
    ], facecolor='red', edgecolor='black', linewidth=2.5, zorder=15)
    ax.add_patch(triangle)
    
    # Add label
    ax.text(bs_x, bs_y - 2, 'Base Station', ha='center', va='top',
            fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='black', linewidth=1.5, alpha=0.95), zorder=16)
    
    # Styling
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    
    # Save
    output_path = os.path.join(WORKDIR, "outdoor_map.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0)
    print(f"✓ Saved outdoor map to: {output_path}")
    plt.close()
    
    return output_path

# ========================================
# MAIN
# ========================================

def main():
    print("="*70)
    print("GENERATING DATASET MAPS")
    print("="*70)
    
    # Generate maps
    indoor_path = generate_indoor_map()
    outdoor_path = generate_outdoor_map()
    
    print("\n" + "="*70)
    print("MAP GENERATION COMPLETE")
    print("="*70)
    print(f"Indoor map: {indoor_path}")
    print(f"Outdoor map: {outdoor_path}")
    print("="*70)

if __name__ == "__main__":
    main()
