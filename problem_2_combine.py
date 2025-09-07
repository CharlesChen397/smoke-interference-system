from problem_2_DE import solve_problem_2_de
from problem_2_CMA import solve_problem_2_cma

if __name__ == '__main__':
    best = solve_problem_2_de(desired_NP_total=200,
                              maxiter=200, workers=-1, do_local_refine=True)
    print(
        "\nDifferential Evolution End. Best params (theta [rad], v [m/s], t_drop [s], tau [s]):")
    print(best)
    solve_problem_2_cma(best)
