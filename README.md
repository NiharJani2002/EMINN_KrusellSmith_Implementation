# EMINN_KrusellSmith_Implementation

EMINN Krusell-Smith Implementation

This repository contains a Python implementation of the EMINN (Econometric Modeling with Implicit Neural Networks) method applied to the Krusell-Smith model, as described in the research paper "Global Solutions to Master Equations for Continuous Time Heterogeneous Agent Macroeconomic Models" by Huo, Rios-Rull, and Zhang (2024). The implementation is written in a Jupyter Notebook (EMINN_KrusellSmith_Implementation.ipynb) and uses PyTorch for neural network training, designed to run in Google Colab with inline plotting and results saved to the ./eminn_results/ directory.
Project Overview
The code implements three numerical methods for solving the Krusell-Smith model with heterogeneous agents:

Finite Agent Method: Simulates a finite number of agents to approximate the wealth distribution.
Discrete State Method: Uses a histogram-based approach to represent the wealth distribution.
Projection Method: Represents the distribution using low-order moments (mean, variance, skewness).

The implementation reproduces the paper's key outputs, including:

Value and Policy Functions: Plots for low, mid, and high aggregate shock values (Figures 6–8).
Wealth Distribution Evolution: Histograms for finite_agent and moment trajectories for projection (Figure 9).
PDE Residuals: Visualizations of residual errors (Figure 10).
Relative Errors: Errors for value and policy functions (Figure 11).
Error Tables: Quantitative metrics for PDE residuals and moment errors (Tables in the paper).

Requirements
To run the notebook, ensure you have the following dependencies installed in your Google Colab environment or local Python setup:

Python 3.8+
PyTorch (torch)
NumPy (numpy)
Pandas (pandas)
Matplotlib (matplotlib)

For Google Colab, these dependencies are typically pre-installed. If running locally, install them using:
pip install torch numpy pandas matplotlib

Setup

Google Colab:

Upload EMINN_KrusellSmith_Implementation.ipynb to your Google Colab environment.
Ensure %matplotlib inline is included at the top of the notebook for inline plotting.
Create a directory ./eminn_results/ in your Colab environment to store output files (plots and CSV tables):import os
os.makedirs("./eminn_results", exist_ok=True)




Local Environment:

Clone this repository or download EMINN_KrusellSmith_Implementation.ipynb.
Install dependencies (see above).
Create the ./eminn_results/ directory in the same directory as the notebook:mkdir eminn_results





Usage

Open the Notebook:

In Google Colab, open EMINN_KrusellSmith_Implementation.ipynb.
Alternatively, run locally using Jupyter:jupyter notebook EMINN_KrusellSmith_Implementation.ipynb




Run the Code:

Execute all cells in the notebook. The main script is structured as follows:if __name__ == "__main__":
    params = {
        'beta': 0.98, 'r': 0.03, 'wage': 1.0, 'rho': 0.95, 'sigma_z': 0.02,
        'z0': 0.0, 'w_min': 0.0, 'w_max': 5.0, 'sigma': 2.0, 'sigma_w': 0.01,
        'lambda_pen': 10.0
    }
    env = EconomicEnvironment(params)
    methods = ['finite_agent', 'discrete_state', 'projection']
    error_dfs = []
    output_dir = "./eminn_results/"
    for method in methods:
        V_net = ValueNet(input_dim=1 + 1 + 3).to(device)
        print(f"Starting training using {method} method...")
        V_net, history = train_eminn(V_net, env, epochs=1000, batch_size=2048, lr=1e-3, method=method, verbose=True)
        plot_training_loss(history, output_dir=output_dir, method=method)
        policy_fn = get_policy_function(V_net, env)
        wealth_paths, z_path, phi_path = simulate_economy(env, policy_fn, method=method, T=100, dt=0.01)
        plot_value_and_policy(V_net, env, method=method, output_dir=output_dir)
        plot_wealth_distribution(wealth_paths, method=method, output_dir=output_dir)
        plot_pde_residuals(V_net, env, method=method, output_dir=output_dir)
        df_errors = generate_error_tables(V_net, env, method=method, output_dir=output_dir)
        error_dfs.append(df_errors)
        print(f"Results for {method}:\n", df_errors)
    combined_df = pd.concat(error_dfs, ignore_index=True)
    combined_df.to_csv(output_dir + "combined_error_table.csv", index=False)
    print(f"\n✅ Full EMINN pipeline completed for all methods!")
    print("Combined error table saved at:", output_dir + "combined_error_table.csv")
    print(combined_df)




Key Parameters:

The economic parameters (params) are set to match the Krusell-Smith model in the paper.
Simulation parameters: T=100 (time horizon), dt=0.01 (time step), N_agents=50, N_grid=100, N_basis=3.
Training parameters: epochs=1000, batch_size=2048, lr=1e-3.


Output:

Plots: Saved in ./eminn_results/ with filenames like training_loss_{method}.png, value_function_{Low z/Mid z/High z}_{method}.png, etc. Plots are also displayed inline in Colab.
Error Tables: Saved as CSV files (error_table_{method}.csv and combined_error_table.csv) in ./eminn_results/. Tables include:
PDE residuals (mean, std, max).
Moment errors (mean, variance, skewness).


Check the console for training progress and error metrics for each method.



Output Details
The notebook generates the following outputs, aligned with the paper’s Figures 6–12 and error tables:

Training Loss (training_loss_{method}.png): Loss curves for each method, ensuring convergence and no nan values.
Value and Policy Functions (value_function_{Low z/Mid z/High z}_{method}.png, policy_function_{Low z/Mid z/High z}_{method}.png): Plots for low, mid, and high aggregate shock values, showing increasing and concave value functions and reasonable consumption curves (Figures 6–8).
Wealth Distribution Evolution (wealth_distribution_evolution_{method}.png): Histograms for finite_agent and moment trajectories for projection (Figure 9).
PDE Residuals (pde_residuals_{method}.png): Visualizations of PDE residual errors, expected to be low (Figure 10).
Relative Errors (relative_error_{method}.png): Relative errors for value and policy functions, expected to be < 1% (Figure 11).
Error Tables (error_table_{method}.csv, combined_error_table.csv): Quantitative metrics, including:
Mean Residual: Should be < 0.1.
Mean Moment Error: Should be significantly less than 1.0 (previously 1.030928 for finite_agent).
Variance Moment Error and Skewness Moment Error: Should be close to analytical steady-state values.



Troubleshooting

Numerical Stability:

If projection produces nan, reduce the learning rate:V_net, history = train_eminn(V_net, env, epochs=1000, batch_size=2048, lr=1e-4, method=method)


Check for nan or inf in phi_path or wealth_paths by adding debug prints in simulate_economy.


High Moment Errors:

If Mean Moment Error is high, verify phi_final and mean_approx in generate_error_tables:print("phi_final:", phi_final)
print("mean_approx:", mean_approx)


Increase T (e.g., T=200) in simulate_economy for better steady-state convergence.


Shape Issues:

Ensure phi_path has shape [T_steps, 3] in generate_error_tables:print("phi_path shape:", phi_path.shape)





References

Huo, Z., Rios-Rull, J.-V., & Zhang, S. (2024). Global Solutions to Master Equations for Continuous Time Heterogeneous Agent Macroeconomic Models. arXiv:2406.13726.
Krusell, P., & Smith, A. A. (1998). Income and Wealth Heterogeneity in the Macroeconomy. Journal of Political Economy, 106(5), 867–896.

License
This project is licensed under the MIT License. See the LICENSE file for details.
