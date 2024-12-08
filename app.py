import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN

# System Parameters
sphere_radius = 50
pyramid_height = 100
num_agents = 50
grid_size = 10
time_steps = 50
energy_threshold = 2
num_cycles = 10

# Fractal Space Energy Dynamics
class FractalEnergySpace:
    def __init__(self, base_energy_threshold=2, fractal_dimension=2.5):
        """
        Initialize fractal space with energy regeneration properties
        
        Args:
        - base_energy_threshold: Minimum energy for agent survival
        - fractal_dimension: Fractal dimension influencing energy dynamics
        """
        self.base_energy_threshold = base_energy_threshold
        self.fractal_dimension = fractal_dimension
        self.energy_memory = {}
    
    def fractal_energy_transform(self, agent_energy, agent_position):
        """
        Natural energy regeneration based on fractal space dynamics
        
        Principles:
        1. Lower energy agents have higher regeneration potential
        2. Position in fractal space affects energy transformation
        3. Uses fractal dimension to modulate energy exchange
        """
        # Energy regeneration based on current energy deficit
        energy_deficit = self.base_energy_threshold - agent_energy
        
        # Fractal-based energy amplification
        # Key idea: Energy can be generated from spatial information
        position_hash = hash(tuple(agent_position)) % 1000
        
        # Complex energy regeneration formula
        regeneration_factor = (
            np.abs(energy_deficit) ** (1 / self.fractal_dimension) *
            np.sin(position_hash / 100) *
            (1 + np.log(1 + np.abs(energy_deficit)))
        )
        
        # Ensure some randomness and natural variation
        noise = np.random.normal(0, 0.1)
        
        # Final energy transformation
        new_energy = agent_energy + regeneration_factor + noise
        
        return max(new_energy, self.base_energy_threshold)

def initialize_agents(num_agents, sphere_radius):
    """Initialize agents uniformly distributed on a sphere"""
    theta = np.random.uniform(0, 2 * np.pi, num_agents)
    phi = np.random.uniform(0, np.pi, num_agents)
    r = sphere_radius * np.cbrt(np.random.uniform(0, 1, num_agents))
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    positions = np.vstack((x, y, z)).T
    energies = np.random.uniform(5, 10, num_agents)
    return positions, energies

def simulate_fractal_energy_system(num_cycles=10):
    """Main simulation loop with fractal energy dynamics"""
    # Initialize fractal energy space
    fractal_space = FractalEnergySpace()
    
    # Initial agent setup
    positions, energies = initialize_agents(num_agents, sphere_radius)
    
    # Tracking metrics
    cycle_metrics = []
    collapsed_positions_per_cycle = []
    
    for cycle in range(num_cycles):
        collapsed_cycle = []
        
        # Simulate pyramid descent and energy loss
        for t in range(time_steps):
            # Gradual descent through pyramid
            positions[:, 2] -= pyramid_height / time_steps / 2
            positions[:, 2] = np.maximum(positions[:, 2], 0)
            
            # Selective agent interactions
            indices = np.random.choice(len(positions), min(len(positions), 20), replace=True)
            for idx in indices:
                # Energy decay
                energies[idx] -= np.random.uniform(0, 1)
                
                # Collapse condition
                if energies[idx] < fractal_space.base_energy_threshold:
                    collapsed_cycle.append(positions[idx].copy())
                    energies[idx] = 0
            
            # Spatial perturbation
            positions += np.random.normal(scale=0.5, size=positions.shape)
        
        # Store collapsed positions
        collapsed_positions_per_cycle.append(np.array(collapsed_cycle))
        
        # Fractal Energy Regeneration
        if len(collapsed_cycle) > 0:
            regenerated_positions = []
            regenerated_energies = []
            
            for pos in collapsed_cycle:
                # Natural energy regeneration in fractal space
                new_energy = fractal_space.fractal_energy_transform(0, pos)
                
                if new_energy > fractal_space.base_energy_threshold:
                    regenerated_positions.append(pos)
                    regenerated_energies.append(new_energy)
            
            # Reintegrate regenerated agents
            if regenerated_positions:
                positions = np.vstack((positions, np.array(regenerated_positions)))
                energies = np.concatenate((energies, np.array(regenerated_energies)))
        
        # Metrics calculation
        total_energy = np.sum(energies)
        entropy = calculate_entropy(positions)
        
        cycle_metrics.append({
            'Cycle': cycle + 1,
            'Total Energy': total_energy,
            'Entropy': entropy,
            'Collapsed Agents': len(collapsed_cycle),
            'Regenerated Agents': len(regenerated_positions) if 'regenerated_positions' in locals() else 0
        })
    
    return collapsed_positions_per_cycle, cycle_metrics

# Existing helper functions (entropy, etc.) remain the same as in previous code

# Run simulation
collapsed_positions, metrics = simulate_fractal_energy_system(num_cycles)

# Visualization
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting collapsed agents
for i, cycle_positions in enumerate(collapsed_positions):
    if len(cycle_positions) > 0:
        ax.scatter(cycle_positions[:, 0], cycle_positions[:, 1], cycle_positions[:, 2],
                   label=f'Cycle {i + 1}', alpha=0.7)

# Plotting pyramid
pyramid_x = [0, sphere_radius, -sphere_radius, 0, 0]
pyramid_y = [0, -sphere_radius, sphere_radius, 0, 0]
pyramid_z = [0, 0, 0, pyramid_height, 0]
ax.plot(pyramid_x, pyramid_y, pyramid_z, c='green', label='Pyramid')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fractal Energy Regeneration System')
ax.legend()
plt.show()

# Display metrics
metrics_df = pd.DataFrame(metrics)
print(metrics_df)
