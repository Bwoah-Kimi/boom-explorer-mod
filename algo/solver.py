# Author: baichen318@gmail.com


import os
import json
import tqdm
import torch
import gpytorch
import numpy as np
from datetime import datetime
from typing import NoReturn
from initialize import micro_al
# Import custom rescaling if available
try:
    from custom_dataset import rescale_dataset as rescale_custom
    USE_CUSTOM_RESCALE = True
except ImportError:
    from dataset import rescale_dataset
    USE_CUSTOM_RESCALE = False
    rescale_custom = rescale_dataset
from model import initialize_dkl_gp
from visualize import plot_pareto_set
from problem import DesignSpaceProblem
from utils import assert_error, info, mkdir, write_txt
from metric import calc_adrs, get_pareto_frontier, get_pareto_optimal_solutions
from design_space_decoder import decode_design_vector, get_design_space_dict
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning


class BOOMExplorerSolver(object):
    def __init__(self, problem: object):
        super(BOOMExplorerSolver, self).__init__()
        self.problem = problem
        self.iteration_results = []
        self.start_timestamp = datetime.now().isoformat()
        
    def initialize(self) -> NoReturn:
        # Microarctecture-aware active learning
        self.visited_x, self.visited_y = micro_al(self.problem)
        self.adrs = []
        self.adrs.append(calc_adrs(
                get_pareto_frontier(self.problem.total_y, reverse=False),
                get_pareto_frontier(self.visited_y, reverse=False)
            )
        )
        # Log initial samples from MicroAL (iteration 0)
        self.log_iteration(self.visited_x, self.visited_y, iteration=0)

    def set_optimizer(self) -> torch.optim.Adam:
        parameters = [
            {"params": self.model.mlp.parameters()},
            {"params": self.model.gp.covar_module.parameters()},
            {"params": self.model.gp.mean_module.parameters()},
            {"params": self.model.gp.likelihood.parameters()}
        ]
        return torch.optim.Adam(
            parameters, lr=self.problem.configs["dkl-gp"]["learning-rate"]
        )

    def fit_dkl_gp(self) -> NoReturn:
        self.model = initialize_dkl_gp(
            self.visited_x,
            self.visited_y,
            self.problem.configs["dkl-gp"]["mlp-output-dim"]
        )
        self.model.set_train()
        optimizer = self.set_optimizer()

        iterator = tqdm.trange(
            self.problem.configs["dkl-gp"]["max-traininig-epoch"],
            desc="Training DKL-GP"
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.gp.likelihood,
            self.model.gp
        )
        y = self.model.transform_ylayout(self.visited_y).squeeze(1)
        for i in iterator:
            optimizer.zero_grad()
            _y = self.model.train(self.visited_x)
            loss = -mll(_y, y)
            loss.backward()
            optimizer.step()
            iterator.set_postfix(loss=loss.item())
        self.model.set_eval()

    def eipv_suggest(self, batch: int = 1, iteration: int = 0) -> NoReturn:
        # --- START: Candidate Sub-sampling to prevent OOM ---
        # Define how many random candidates to evaluate.
        # This is the most important parameter to tune for memory.
        num_candidates = 4096  # Use 4096 random points instead of all 60k+

        # Get indices of points we have already visited
        # This is slow, but necessary to avoid re-sampling visited points.
        # A more optimized version would use a boolean mask.
        visited_indices = {tuple(row.tolist()) for row in self.visited_x.cpu()}
        unevaluated_mask = [tuple(row.tolist()) not in visited_indices for row in self.problem.x.cpu()]
        unevaluated_indices = np.where(unevaluated_mask)[0]

        if len(unevaluated_indices) < num_candidates:
            candidate_indices = unevaluated_indices
        else:
            candidate_indices = np.random.choice(unevaluated_indices, num_candidates, replace=False)
        
        # The subset of candidates to evaluate on the GPU
        candidates = self.problem.x[candidate_indices].to(torch.float).to(self.model.device)
        info(f"Evaluating acquisition function on {len(candidates)} random candidates (out of {len(unevaluated_indices)}).")
        # --- END: Candidate Sub-sampling ---

        partitioning = NondominatedPartitioning(
            ref_point=self.problem._ref_point.to(self.model.device),
            Y=self.visited_y.to(self.model.device)
        )

        acq_func = ExpectedHypervolumeImprovement(
            model=self.model.gp,
            ref_point=self.problem._ref_point.tolist(),
            partitioning=partitioning
        ).to(self.model.device)

        # --- MODIFIED: Compute acq_val only on the candidate subset ---
        acq_val = acq_func(
            self.model.forward_mlp(
                candidates  # Use the smaller 'candidates' tensor
            ).unsqueeze(1).to(self.model.device)
        ).to(self.model.device)
        
        # Find the best candidate from the SUBSET
        top_acq_val, top_indices_in_subset = torch.topk(acq_val, k=batch)
        
        # Map the subset indices back to the original problem.x indices
        original_indices = torch.tensor(candidate_indices[top_indices_in_subset.cpu().numpy()])

        new_x = self.problem.x[original_indices.cpu()].to(torch.float32).reshape(-1, self.problem.n_dim)
        new_y = self.problem.evaluate_true(new_x).unsqueeze(0)
        
        self.visited_x = torch.cat((self.visited_x, new_x), 0)
        self.visited_y = torch.cat((self.visited_y, new_y), 0)
        
        # Log this iteration
        self.log_iteration(new_x, new_y, iteration)

    def log_iteration(self, new_x: torch.Tensor, new_y: torch.Tensor, iteration: int) -> NoReturn:
        """
        Log iteration results for JSON export.
        
        Args:
            new_x: Encoded design vector(s) sampled in this iteration
            new_y: Scaled objective values for the sampled design(s)
            iteration: Current iteration number
        """
        # Use custom rescaling if available
        rescale_func = rescale_custom if USE_CUSTOM_RESCALE else rescale_dataset
        
        # Handle batch sampling (new_x might have multiple rows)
        if new_x.dim() == 1:
            new_x = new_x.unsqueeze(0)
        if new_y.dim() == 1:
            new_y = new_y.unsqueeze(0)
        
        # Log each sampled design
        for i in range(new_x.shape[0]):
            # Decode design parameters
            dse_config = decode_design_vector(new_x[i])
            
            # Unscale objectives (convert back to original values)
            objectives_unscaled = rescale_func(new_y[i].cpu().numpy(), cycles_idx=0, cost_idx=1)
            
            # Create iteration result entry
            iteration_entry = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "dse_config": dse_config,
                "objectives": objectives_unscaled.tolist() if isinstance(objectives_unscaled, np.ndarray) else list(objectives_unscaled)
            }
            
            self.iteration_results.append(iteration_entry)
    
    def save_json(self, output_path: str) -> NoReturn:
        """
        Save all iteration results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        json_data = {
            "dse_start_timestamp": self.start_timestamp,
            "design_space": get_design_space_dict(),
            "optimizer_type": "BOOM-Explorer",
            "sampling_algo": "MicroAL + DKL-GP + EHVI",
            "misc_info": {
                "max_bo_steps": self.problem.configs["bo"]["max-bo-steps"],
                "mlp_output_dim": self.problem.configs["dkl-gp"]["mlp-output-dim"],
                "total_samples": len(self.iteration_results),
                "dataset_path": self.problem.configs["dataset"]["path"]
            },
            "iteration_results": self.iteration_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        info(f"Saved {len(self.iteration_results)} iteration results to {output_path}")

    def report(self):
        gt = get_pareto_frontier(self.problem.total_y, reverse=False)
        pred = get_pareto_frontier(self.visited_y, reverse=False)
        self.adrs.append(calc_adrs(gt, pred))
        optimal_solution = get_pareto_optimal_solutions(self.visited_x, self.visited_y)
        
        # Use custom rescaling if available
        rescale_func = rescale_custom if USE_CUSTOM_RESCALE else rescale_dataset
        
        # Determine indices based on number of objectives
        if pred.shape[1] == 2:
            # Custom dataset: 2 objectives (cycles, cost)
            pred_rescaled = rescale_func(pred, cycles_idx=-2, cost_idx=-1)
        else:
            # Original BOOM: 3 objectives (perf, power, time)
            pred_rescaled = rescale_func(pred, perf_idx=-2, power_idx=-1)
        
        info("pareto set: {}, size: {}, ADRS: {}".format(
                str(pred_rescaled),
                len(pred),
                self.adrs[-1]
            )
        )
        p = self.problem.configs["report"]["path"]
        mkdir(p)

        plot_pareto_set(
            rescale_func(pred) if pred.shape[1] == 2 else rescale_dataset(pred),
            gt=rescale_func(gt) if gt.shape[1] == 2 else rescale_dataset(gt),
            design_space=self.problem.configs["dataset"]["path"],
            output=os.path.join(p, "report.pdf")
        )
        write_txt(
            os.path.join(
                p,
                "adrs.rpt"
            ),
            np.array(self.adrs),
            fmt="%f"
        )
        write_txt(
            os.path.join(
                p,
                "pareto-frontier.rpt"
            ),
            np.array(pred),
            fmt="%f"
        )
        write_txt(
            os.path.join(
                p,
                "pareto-optimal-solutions.rpt"
            ),
            np.array(optimal_solution),
            fmt="%f"
        )
        self.model.save(
            os.path.join(
                p,
                "dkl-gp.mdl"
            )
        )
        # Save iteration results as JSON
        self.save_json(
            os.path.join(
                p,
                "iteration_results.json"
            )
        )


def create_solver(problem: DesignSpaceProblem) ->BOOMExplorerSolver:
    return BOOMExplorerSolver(problem)
