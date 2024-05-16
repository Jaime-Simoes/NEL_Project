import torch
import random

from gpolnel.utils.solution import Solution
from gpolnel.utils.population import Population
from gpolnel.algorithms.random_search import RandomSearch


class PopulationBased(RandomSearch):
    """Population-based ISA (PB-ISAs).

    Based on the number of candidate solutions they handle at each
    step, the optimization algorithms can be categorized into
    Single-Point (SP) and Population-Based (PB) approaches. The solve
    procedure in the SP algorithms is generally guided by the
    information provided by a single candidate solution from 𝑆,
    usually the best-so-far solution, that is gradually evolved in a
    well defined manner in hope to find the global optimum. The HC is
    an example of a SP algorithm as the solve is performed by
    exploring the neighborhood of the current best solution.
    Contrarily, the solve procedure in PB algorithms is generally
    guided by the information shared by a set of candidate solutions
    and the exploitation of its collective behavior of different ways.
    In abstract terms, one can say that PB algorithms share, at least,
    the following two features: an object representing the set of
    simultaneously exploited candidate solutions (i.e., the
    population), and a procedure to "move" them across 𝑆.

    An instance of a PB-ISA is characterized by the following features:
        1) a PI (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point of the
         solve space (𝑆);
        3) the best solution found by the PB-ISA;
        4) the number of simultaneously exploited solution (the
         population's size);
        6) a collection of candidate solutions - the population;
        7) a random state for random numbers generation;
        8) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from RandomSearch)
        An instance of OP.
    pop_size : int
        The population's size.
    best_sol : Solution (inherited from RandomSearch)
        The best solution found.
    pop_size : int
        Population's size.
    pop : Population
        Object of type Population which holds population's collective
        representation, feasibility states and fitness values.
    initializer : function (inherited from RandomSearch)
        The initialization procedure.
    mutator : function
        A function to move solutions across 𝑆.
    seed : int (inherited from RandomSearch)
        The seed for random numbers generators.
    device : str (inherited from RandomSearch)
        Specification of the processing device.
    """

    def __init__(self, pi, initializer, mutator, pop_size=100, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        mutator : function
            A function to move solutions across the solve space.
        pop_size : int (default=100)
            Population's size.
        seed : int str (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        RandomSearch.__init__(self, pi, initializer, seed, device)
        self.mutator = mutator
        self.pop_size = pop_size
        # Initializes the population's object (None by default)
        self.pop = None

    def _initialize(self, start_at=None):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        # Creates as empty list for the population's representation
        pop_size, pop_repr = self.pop_size, []
        # Recomputes populations' size and extends the list with user-specified initial seed, is such exists
        if start_at is not None:
            pop_size -= len(start_at)
            pop_repr.extend(start_at)
        # Initializes pop_size individuals by means of 'initializer' function
        # self.initializer(sspace=self.pi.sspace, n_sols=pop_size)
        pop_repr.extend(self.initializer(sspace=self.pi.sspace, n_sols=pop_size))
        # Stacks population's representation, if candidate solutions are objects of type torch.tensor
        if isinstance(pop_repr[0], torch.Tensor):
            pop_repr = torch.stack(pop_repr)
        # Set pop and best solution
        self._set_pop(pop_repr=pop_repr)

    def _set_pop(self, pop_repr):
        """Encapsulates the set method of the population attribute of PopulationBased algorithm.

        Parameters
        ----------
        pop_repr : list
            A list of solutions' representation.

        Returns
        -------
        None
        """
        # Creates an object of type 'Population', given the initial representation
        self.pop = Population(pop_repr)
        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)
        # Sets the best solution
        self._set_best_sol()

    def _set_best_sol(self):
        """Encapsulates the set method of the best_sol attribute of PopulationBased algorithm.

        Parameters
        ----------
            self
        Returns
        -------
            None
        """
        self.best_sol = self.pop.get_best_pop(min_=self.pi.min_)

