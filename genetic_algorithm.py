import os
import time
import random
import pickle
import logging
import torch
import pandas as pd
from copy import deepcopy

from gpolnel.algorithms.population_based import PopulationBased
from gpolnel.utils.population import PopulationTree
from gpolnel.utils.inductive_programming import _execute_tree, _get_tree_depth


class GeneticAlgorithm(PopulationBased):
    """Implements Genetic Algorithm (GA).

    Genetic Algorithm (GA) is a meta-heuristic introduced by John
    Holland, strongly inspired by Darwin's Theory of Evolution.
    Conceptually, the algorithm starts with a random-like population of
    candidate-solutions (called chromosomes). Then, by resembling the
    principles of natural selection and the genetically inspired
    variation operators, such as crossover and mutation, the algorithm
    breeds a population of next-generation candidate-solutions (called
    the offspring population), which replaces the previous population
    (a.k.a. the population of parents). The procedure is iterated until
    reaching some stopping criteria, like a predefined number of
    iterations (also called generations).

    An instance of GA can be characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) a function to initialize the solve at a given point in 𝑆;
        3) a function to select candidate solutions for variation phase;
        4) a function to mutate candidate solutions;
        5) the probability of applying mutation;
        6) a function to crossover two solutions (the parents);
        7) the probability of applying crossover;
        8) the population's size;
        9) the best solution found by the PB-ISA;
        10) a collection of candidate solutions - the population;
        11) a random state for random numbers generation;
        12) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from PopulationBased)
        An instance of OP.
    best_sol : Solution (inherited from PopulationBased))
        The best solution found.
    pop_size : int (inherited from PopulationBased)
        The population's size.
    pop : Population (inherited from PopulationBased)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from PopulationBased))
        The initialization procedure.
    selector : function
        The selection procedure.
    mutator : function (inherited from PopulationBased)
        The mutation procedure.
    p_m : float
        The probability of applying mutation.
    crossover : function
        The crossover procedure.
    p_c : float
        The probability of applying crossover.
    elitism : bool
        A flag which activates elitism during the evolutionary process.
    reproduction : bool
        A flag which states if reproduction should happen (reproduction
        is True), when the crossover is not applied. If reproduction is
        False, then either crossover or mutation will be applied.
    seed : int (inherited from PopulationBased)
        The seed for random numbers generators.
    device : str (inherited from PopulationBased)
        Specification of the processing device.
    """
    __name__ = "GeneticAlgorithm"

    def __init__(self, pi, initializer, selector, mutator, crossover, p_m=0.2, p_c=0.8, pop_size=100, elitism=True,
                 reproduction=False, seed=0, device="cpu"):
        """ Objects' constructor

        Following the main purpose of a PB-ISA, the constructor takes a
        problem instance (PI) to solve, the population's size and an
        initialization procedure. Moreover it receives the mutation and
        the crossover functions along with the respective probabilities.
        The constructor also takes two boolean values indicating whether
        elitism and reproduction should be applied. Finally, it takes
        some technical parameters like the random seed and the processing
        device.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        selector : function
            The selection procedure.
        mutator : function
            A function to move solutions across the solve space.
        crossover : function
            The crossover function.
        p_m : float (default=0.2)
            Probability of applying mutation.
        p_c : float (default=0.8)
            Probability of applying crossover.
        pop_size : int (default=100)
            Population's size.
        elitism : bool (default=True)
            A flag which activates elitism during the evolutionary process.
        reproduction : bool (default=False)
            A flag which states if reproduction should happen (reproduction
            is True), when the crossover is not applied.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        PopulationBased.__init__(self, pi, initializer, mutator, pop_size, seed, device)  # at this point, it has the pop attribute, but it is None
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.p_c = p_c
        self.elitism = elitism
        self.reproduction = reproduction

    def _set_pop(self, pop_repr):
        """Encapsulates the set method of the population attribute of GeneticAlgorithm algorithm.

        Parameters
        ----------
        pop_repr : list
            A list of solutions' representation.

        Returns
        -------
        None
        """
        # Creates an object of type 'PopulationTree', given the initial representation
        self.pop = PopulationTree(pop_repr)
        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)
        # Gets the best in the initial population
        self._set_best_sol()
