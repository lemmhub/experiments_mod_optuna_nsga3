#run line
#source ~/.optuna_workflow/bin/activate && python3 optimization.py best_model.pkl --algo NSGA2 --pm_prob=0.4 --sbx_prob=0.7 --sbx_eta=20 --seed=1234


import pickle
import numpy as np
from datetime import datetime
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
#from pymoo.factory import get_reference_directions, get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.nsga3 import NSGA3
import dill
import argparse
import os
import csv
from tqdm import tqdm

# ---------------------------
# CLI ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description="Run a single optimization job with a specific seed")

parser.add_argument("model_path", type=str, help="Path to XGBoost .pkl model")
parser.add_argument("--algo", type=str, required=True,
                    choices=["NSGA2", "RNSGA2", "MOEAD", "SMSEMOA", "NSGA3"],
                    help="Which algorithm to run")
parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
parser.add_argument("--pm_eta", type=float, default=20)
parser.add_argument("--pm_prob", type=float, required=True)
parser.add_argument("--sbx_eta", type=float, required=True)
parser.add_argument("--sbx_prob", type=float, required=True)

args = parser.parse_args()

pm_eta = args.pm_eta
pm_prob = args.pm_prob
sbx_eta = args.sbx_eta
sbx_prob = args.sbx_prob
model_path = args.model_path
algorithm_name = args.algo
seed = args.seed

model_name = os.path.splitext(os.path.basename(model_path))[0]
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------
# Setup logging
# ---------------------------
os.makedirs("logs", exist_ok=True)
log_file = "logs/optimization.log"
status_file = "logs/status.csv"

def log(msg):
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")

def update_status(status="running"):
    # Log run status in a CSV
    header = ["algorithm", "seed", "pm_prob", "sbx_prob", "sbx_eta", "status"]
    row = [algorithm_name, seed, pm_prob, sbx_prob, sbx_eta, status]
    exists = os.path.exists(status_file)

    with open(status_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

# ---------------------------
# Min-max bounds
# ---------------------------
mins = np.array([0.60, 1.2, 250, 0.8, 1.3, 1.0, 0.001, 0.00001, 0.05, 0.15])
maxs = np.array([1.8, 3.2, 700, 2.5, 3.5, 2.5, 0.01, 0.0002, 0.15, 0.5])

# ---------------------------
# Load Model
# ---------------------------
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Cost Function
# ---------------------------
def FC_costs(SH2, delta_mem, delta_io, LC):
    cost = 100
    if SH2 <= 0.3: cost += SH2 * 830
    elif SH2 <= 0.5: cost += SH2 * 810
    elif SH2 <= 0.7: cost += SH2 * 790
    elif SH2 <= 0.8: cost += SH2 * 785
    elif SH2 <= 0.85: cost += SH2 * 780
    else: cost += SH2 * 777
    cost += delta_mem * 350 + delta_io * 500
    if LC <= 0.4: cost += LC * 7000
    elif LC <= 0.45: cost += LC * 6800
    elif LC <= 0.50: cost += LC * 6500
    elif LC <= 0.55: cost += LC * 6200
    elif LC <= 0.6: cost += LC * 6000
    elif LC <= 0.7: cost += LC * 5900
    elif LC <= 0.85: cost += LC * 5700
    else: cost += LC * 5650
    return cost

# ---------------------------
# Problem Definition
# ---------------------------
class Problem_XGB_MOP(ElementwiseProblem):
    def __init__(self, model):
        super().__init__(n_var=10, n_obj=2, n_ieq_constr=0, xl=mins, xu=maxs)
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        pred = self.model.predict(x.reshape(1, -1))[0]
        f1 = -pred
        f2 = FC_costs(x[0], x[6], x[7], x[9])
        out["F"] = [f1, f2]

# ---------------------------
# Get Algorithm
# ---------------------------
def get_algorithm(name):
    if name == "NSGA2":
        return NSGA2(
            pop_size=500,
            n_offsprings=100,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=sbx_prob, eta=sbx_eta),
            mutation=PM(prob=pm_prob, eta=pm_eta),
            eliminate_duplicates=True,
        )
    elif name == "RNSGA2":
        return RNSGA2(
            ref_points=np.array([[-1.54, 0.025], [-1.4, 0.001]]),
            n_offsprings=100,
            pop_size=500,
            epsilon=0.01,
            crossover=SBX(prob=sbx_prob, eta=sbx_eta),
            mutation=PM(prob=pm_prob, eta=pm_eta),
            eliminate_duplicates=True,
        )
    elif name == "MOEAD":
        return MOEAD(
            ref_dirs=get_reference_directions("uniform", 2, n_partitions=80),
            n_neighbors=15,
            prob_neighbor_mating=0.3
        )
    elif name == "SMSEMOA":
        return SMSEMOA(
            pop_size=500,
            n_offsprings=100,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=sbx_prob, eta=sbx_eta),
            mutation=PM(prob=pm_prob, eta=pm_eta),
            eliminate_duplicates=True,
        )
    elif name == "NSGA3":
        return NSGA3(
            pop_size=500,
            ref_dirs=get_reference_directions("uniform", 2, n_partitions=80),
            crossover=SBX(prob=sbx_prob, eta=sbx_eta),
            mutation=PM(prob=pm_prob, eta=pm_eta),
            eliminate_duplicates=True,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {name}")


from pymoo.core.callback import Callback

class ProgressBarCallback(Callback):
    def __init__(self, total_gens):
        super().__init__()
        self.pbar = tqdm(total=total_gens, desc=f"{algorithm_name}-seed{seed}", position=0, leave=True)

    def notify(self, algorithm):
        self.pbar.update(1)
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()



# ---------------------------
# Run
# ---------------------------
def run():
    log(f"START: {algorithm_name} | seed={seed} | pm_prob={pm_prob} | sbx_prob={sbx_prob} | sbx_eta={sbx_eta}")
    update_status("running")

    model = load_model(model_path)
    problem = Problem_XGB_MOP(model)
    algo = get_algorithm(algorithm_name)

    # tqdm visual progress (for local debug or attached tmux)
    callback = ProgressBarCallback(total_gens=500)
    res = minimize(problem,
               algo,
               termination=get_termination("n_gen", 500),
               seed=seed,
               save_history=True,
               verbose=False,
               callback=callback)

    # Save result
    filename = f"{model_name}_M{pm_prob}_C{sbx_prob}-{sbx_eta}_{algorithm_name}_seed{seed}_{now}.pkl"
    with open(filename, "wb") as f:
        dill.dump(res, f)

    log(f"END: {algorithm_name} seed={seed} saved as {filename}")
    update_status("completed")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    run()
