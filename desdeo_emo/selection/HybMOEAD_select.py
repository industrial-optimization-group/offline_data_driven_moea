import numpy as np
from typing import List
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_emo.othertools.ProbabilityWrong import Probability_wrong


class HybMOEAD_select(SelectionBase):
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

    def do(self, pop: Population, vectors: ReferenceVectors, ideal_point, current_neighborhood, offspring_fx, offspring_unc, theta_adaptive) -> List[int]:
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
        theta_adaptive_matrix       = np.array([theta_adaptive]*num_neighbors)
        n_samples = 1000
        # Performing Generic MOEA/D selection
        
        values_SF           = self._evaluate_SF(current_population, current_reference_vectors, ideal_point_matrix, theta_adaptive_matrix)
        values_SF_offspring = self._evaluate_SF(offspring_population, current_reference_vectors, ideal_point_matrix, theta_adaptive_matrix)

        # Compare the offspring with the individuals in the neighborhood 
        # and replace the ones which are outperformed by it.
        selection_generic = np.where(values_SF_offspring.reshape(-1,) < values_SF)[0]
        #print("Generic selction:",selection_generic)
        
        
        # Probabilistic MOEA/D selection
        pwrong_current = Probability_wrong(mean_values=current_population, stddev_values=current_uncertainty, n_samples=1000)
        pwrong_current.vect_sample_f()

        pwrong_offspring = Probability_wrong(mean_values=offspring_population.reshape(-1,pop.problem.n_of_objectives), stddev_values=offspring_uncertainty.reshape(-1,pop.problem.n_of_objectives), n_samples=1000)
        pwrong_offspring.vect_sample_f()

        values_SF_current_dist   = self._evaluate_SF_dist(current_population, current_reference_vectors, ideal_point_matrix, pwrong_current, theta_adaptive_matrix)
        values_SF_offspring_dist = self._evaluate_SF_dist(offspring_population, current_reference_vectors, ideal_point_matrix, pwrong_offspring, theta_adaptive_matrix)

        ##### KDE here and then compute probability
        pwrong_current.pdf_list = {}
        pwrong_current.ecdf_list = {}
        pwrong_offspring.pdf_list = {}
        pwrong_offspring.ecdf_list = {}
        values_SF_offspring_temp = np.asarray([values_SF_offspring_dist])
        values_SF_current_temp = np.asarray([values_SF_current_dist])
        pwrong_offspring.compute_pdf(values_SF_offspring_temp.reshape(num_neighbors,1,n_samples))
        pwrong_current.compute_pdf(values_SF_current_temp.reshape(num_neighbors,1,n_samples))
        #pwrong_offspring.plt_density(values_SF_offspring.reshape(20,1,1000))
        probabilities = np.zeros(num_neighbors)
        for i in range(num_neighbors):
            probabilities[i]=pwrong_current.compute_probability_wrong_PBI(pwrong_offspring, index=i)
        # Compare the offspring with the individuals in the neighborhood 
        # and replace the ones which are outperformed by it if P_{wrong}>0.5
        selection_probabilitic = np.where(probabilities>0.5)[0]
        #print("Selection prob:", selection_probabilitic)   
        selection = np.union1d(selection_generic,selection_probabilitic)
        #print("Overall selection:", selection)

        # considering mean 
        #selection_probabilitic_2 = np.where(np.mean(values_SF_offspring_dist, axis=1) < np.mean(values_SF_current_dist, axis=1))[0]
        #selection_2 = np.union1d(selection_generic,selection_probabilitic_2)
        #print("Overall selection 2:", selection_2)
        
        return current_neighborhood[selection]


    def tchebycheff(self, objective_values:np.ndarray, weights:np.ndarray, ideal_point:np.ndarray):
        feval   = np.abs(objective_values - ideal_point) * weights
        max_fun = np.max(feval)
        return max_fun

    def weighted_sum(self, objective_values, weights):
        feval   = np.sum(objective_values * weights)
        return feval

    def pbi_dist(self, objective_values, weights, ideal_point, pwrong_f_samples, theta):

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

    def pbi(self, objective_values, weights, ideal_point, theta):
        norm_weights    = np.linalg.norm(weights)
        weights         = weights/norm_weights
        fx_a            = objective_values - ideal_point
        d1              = np.inner(fx_a, weights)

        fx_b            = objective_values - (ideal_point + d1 * weights)
        d2              = np.linalg.norm(fx_b)
        
        fvalue          = d1 + theta * d2
        return fvalue


    def _evaluate_SF_dist(self, neighborhood, weights, ideal_point, pwrong, theta_adaptive):
        if self.SF_type == "TCH":
            SF_values = np.array(list(map(self.tchebycheff, neighborhood, weights, ideal_point)))
            return SF_values
        elif self.SF_type == "PBI":
            SF_values = np.array(list(map(self.pbi_dist, neighborhood, weights, ideal_point, pwrong.f_samples, theta_adaptive)))
            return SF_values
        elif self.SF_type == "WS":
            SF_values = np.array(list(map(self.weighted_sum, neighborhood, weights)))
            return SF_values
        else:
            return []

    def _evaluate_SF(self, neighborhood, weights, ideal_point, theta_adaptive):
        if self.SF_type == "TCH":
            SF_values = np.array(list(map(self.tchebycheff, neighborhood, weights, ideal_point)))
            return SF_values
        elif self.SF_type == "PBI":
            SF_values = np.array(list(map(self.pbi, neighborhood, weights, ideal_point, theta_adaptive)))
            return SF_values
        elif self.SF_type == "WS":
            SF_values = np.array(list(map(self.weighted_sum, neighborhood, weights)))
            return SF_values
        else:
            return []

    

    

    
    

