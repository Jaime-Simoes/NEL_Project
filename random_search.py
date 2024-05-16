import time
import random
import logging

import torch

from gpolnel.utils.solution import Solution
from gpolnel.algorithms.search_algorithm import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    """Random Search (RS) Algorithm.

    Random Search (RS) can be seen as the very first and rudimentary
    stochastic iterative solve algorithm (SISA) for problem-solving.
    Its strategy, far away from being intelligent, consists of
    randomly sampling S for a given number of iterations. RS is
    frequently used in benchmarks as the baseline for assessing
    algorithms' performance. Following this rationale, one can
    conceptualize RS at the root of the hierarchy of intelligent
    SISAs; under this perspective, it is meaningful to assume that the
    SISAs donated with intelligence, like Hill Climbing and Genetic
    Algorithms, might be seen as improvements upon RS, thus branching
    from it.

    An instance of a RS can be characterized by the following features:
        1) a PI (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point of the
         solve space (𝑆);
        3) the best solution found by the ISA;
        4) a random state for random numbers generation;
        5) the processing device (CPU or GPU).

    To solve a PI, the RS:
        1) initializes the solve at a given point in 𝑆 (normally, by
         sampling candidate solution(s) at random);
        2) searches throughout 𝑆, in iterative manner, for the best
         possible solution by randomly sampling candidate solutions
         from it. Traditionally, the termination condition for an ISA
         is the number of iterations, the default stopping criteria in
         this library.

    Attributes
    ----------
    pi : Problem (inherited from SearchAlgorithm)
        An instance of an OP.
    best_sol : Solution (inherited from SearchAlgorithm)
        The best solution found.
    initializer : function (inherited)
        The initialization procedure.
    seed : int
        The seed for random numbers generators.
    device : str (inherited from SearchAlgorithm)
        Specification of the processing device.
    """
    __name__ = "RandomSearch"

    def __init__(self, pi, initializer, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            An instance of an OP.
        initializer : function
            The initialization procedure.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        SearchAlgorithm.__init__(self, pi, initializer, device)
        # Sets the random seed for torch and random
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def _initialize(self, start_at=None):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        if start_at:
            # Given the initial_seed, crates an instance of type Solution
            self.best_sol = Solution(start_at)
            # Evaluates the candidate solution
            self.pi.evaluate_sol(self.best_sol)
        else:
            # Generates an valid random initial solution (already evaluated)
            self.best_sol = self._get_random_sol()
            while not self.best_sol.valid:
                self.best_sol = self._get_random_sol()

    def _get_random_sol(self):
        """Generates one random initial solution.

        This method (1) generates a random representation of a
        candidate solution by means of the initializer function, (2)
        creates an instance of type Solution, (3) evaluates  instance's
        representation and (4) returns the evaluated object.
        Notice that the solution can be feasible under 𝑆's constraints
        or not.

        Returns
        -------
        Solution
            A random initial solution.
        """
        # 1)
        repr_ = self.initializer(sspace=self.pi.sspace, device=self.device)
        # 2)
        sol = Solution(repr_)
        # 3)
        self.pi.evaluate_sol(sol)
        # 4)
        return sol
