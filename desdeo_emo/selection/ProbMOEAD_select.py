import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_emo.othertools.ProbabilityWrong import Probability_wrong


class ProbMOEAD_select(SelectionBase):
    """The MOEAD selection operator. 

    Parameters
    ----------
    pop : Population
        The population of individuals
    SF_type : str
        The scalarizing function employed to evaluate the solutions

    """
    def __init__(
        self, pop: Population, SF_type: str
    ):
	 # initialize
        self.SF_type = SF_type

    def do(self, pop: Population, vectors: ReferenceVectors, ideal_point, current_neighborhood, offspring_fx, offspring_unc) -> List[int]:
        """Select the individuals that are kept in the neighborhood.

        Parameters
        ----------
        pop : Population
            The current population.
        vectors : ReferenceVectors
            Class instance containing reference vectors.
        ideal_point
            Ideal vector found so far
        current_neighborhood
            Neighborhood to be updated
        offspring_fx
            Offspring solution to be compared with the rest of the neighborhood

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        # Compute the value of the SF for each neighbor
        num_neighbors               = len(current_neighborhood)
        current_population          = pop.objectives[current_neighborhood,:]
        current_uncertainty          = pop.uncertainity[current_neighborhood,:]
        current_reference_vectors   = vectors.values[current_neighborhood,:]
        offspring_population        = np.array([offspring_fx]*num_neighbors)
        offspring_uncertainty       = np.array([offspring_unc]*num_neighbors)
        ideal_point_matrix          = np.array([ideal_point]*num_neighbors)

        pwrong_current = Probability_wrong(mean_values=current_population, stddev_values=current_uncertainty, n_samples=1000)
        pwrong_current.vect_sample_f()

        pwrong_offspring = Probability_wrong(mean_values=offspring_population.reshape(-1,pop.problem.n_of_objectives), stddev_values=offspring_uncertainty.reshape(-1,pop.problem.n_of_objectives), n_samples=1000)
        pwrong_offspring.vect_sample_f()

        values_SF           = self._evaluate_SF(current_population, current_reference_vectors, ideal_point_matrix, pwrong_current)
        values_SF_offspring = self._evaluate_SF(offspring_population, current_reference_vectors, ideal_point_matrix, pwrong_offspring)

        ##### KDE here and then compute probability
        #np.asarray([values_SF_offspring])
        #values_SF_offspring.reshape(20,1,1000)
        #pwrong_offspring.compute_pdf(values_SF_offspring.reshape(20,1,1000))
        #pwrong_offspring.plt_density(values_SF_offspring.reshape(20,1,1000))

        # Compare the offspring with the individuals in the neighborhood 
        # and replace the ones which are outperformed by it. 
        #selection = np.where(values_SF_offspring < values_SF)[0]

        selection = np.where(np.mean(values_SF_offspring, axis=1) < np.mean(values_SF, axis=1))[0]    
        #print(selection)
        return current_neighborhood[selection]


    def tchebycheff(self, objective_values:np.ndarray, weights:np.ndarray, ideal_point:np.ndarray):
        feval   = np.abs(objective_values - ideal_point) * weights
        max_fun = np.max(feval)
        return max_fun

    def weighted_sum(self, objective_values, weights):
        feval   = np.sum(objective_values * weights)
        return feval

    def pbi(self, objective_values, weights, ideal_point, pwrong_f_samples, theta = 5):

        norm_weights    = np.linalg.norm(weights)
        weights         = weights/norm_weights
        
        #fx_a            = objective_values - ideal_point
        fx_a            = pwrong_f_samples - ideal_point.reshape(-1,1)
        
        #d1              = np.inner(fx_a, weights)
        
        d1               = np.sum(np.transpose(fx_a)* np.tile(weights,(1000,1)), axis=1)
        
        #fx_b            = objective_values - (ideal_point + d1 * weights)

        fx_b             = np.transpose(pwrong_f_samples) - (np.tile(ideal_point,(1000,1)) + np.reshape(d1,(-1,1)) * np.tile(weights,(1000,1)))

        #d2              = np.linalg.norm(fx_b)
        
        d2               = np.linalg.norm(fx_b, axis=1)

        fvalue          = d1 + theta * d2

        return fvalue


    def _evaluate_SF(self, neighborhood, weights, ideal_point, pwrong):
        if self.SF_type == "TCH":
            SF_values = np.array(list(map(self.tchebycheff, neighborhood, weights, ideal_point)))
            return SF_values
        elif self.SF_type == "PBI":
            SF_values = np.array(list(map(self.pbi, neighborhood, weights, ideal_point, pwrong.f_samples)))
            return SF_values
        elif self.SF_type == "WS":
            SF_values = np.array(list(map(self.weighted_sum, neighborhood, weights)))
            return SF_values
        else:
            return []



    

    

    
    

