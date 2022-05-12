import numpy as np
from warnings import warn
from typing import List, Callable
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
import matplotlib.pyplot as plt
from matplotlib import rc

class APD_Select(SelectionBase):
    """
    The selection operator for the RVEA algorithm. Read the following paper for more
    details.
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016

    Parameters
    ----------
    pop : Population
        The population instance
    time_penalty_function : Callable
        A function that returns the time component in the penalty function.
    alpha : float, optional
        The RVEA alpha parameter, by default 2
    """

    def __init__(
        self,
        pop: Population,
        time_penalty_function: Callable,
        alpha: float = 2,
        selection_type: str = None,
    ):
        self.time_penalty_function = time_penalty_function
        if alpha is None:
            alpha = 2
        self.alpha = alpha
        self.n_of_objectives = pop.problem.n_of_objectives
        if selection_type is None:
            selection_type = "mean"
        self.selection_type = selection_type
        self.ideal: np.ndarray = pop.ideal_fitness_val

    def do(self, pop: Population, vectors: ReferenceVectors) -> List[int]:
        """Select individuals for mating on basis of Angle penalized distance.

        Parameters
        ----------
        pop : Population
            The current population.
        vectors : ReferenceVectors
            Class instance containing reference vectors.

        Returns
        -------
        List[int]
            List of indices of the selected individuals
        """
        partial_penalty_factor = self._partial_penalty_factor()
        refV = vectors.neighbouring_angles_current
        # Normalization - There may be problems here
        fitness = self._calculate_fitness(pop)
        """
        partial_penalty_factor = 2

        fitness = np.array([[1.2, 0.6],
                            [1.23, 0.65],
                            [0.2, 1.5],
                            [0.2, 1.5],
                            [0.6,1.2],
                            [0.7, 1],
                            [1.4, 0.1],
                            [1.38, 0.09]])

        uncertainty = np.array([[0.1, 0.2],
                                [0.03, 0.05],
                                [0.1, 0.1],
                                [0.05, 0.05],
                                [0.05,0.1],
                                [0.1,0.05],
                                [0.1, 0.05],
                                [0.1, 0.05]])
        """
        fmin = np.amin(fitness, axis=0)
        self.ideal = np.amin(
            np.vstack((self.ideal, fmin, pop.ideal_fitness_val)), axis=0
        )
        translated_fitness = fitness - self.ideal
        fitness_norm = np.linalg.norm(translated_fitness, axis=1)
        # TODO check if you need the next line
        # TODO changing the order of the following few operations might be efficient
        fitness_norm = np.repeat(fitness_norm, len(translated_fitness[0, :])).reshape(
            len(fitness), len(fitness[0, :])
        )
        # Convert zeros to eps to avoid divide by zero.
        # Has to be checked!
        fitness_norm[fitness_norm == 0] = np.finfo(float).eps
        normalized_fitness = np.divide(
            translated_fitness, fitness_norm
        )  # Checked, works.
        cosine = np.dot(normalized_fitness, np.transpose(vectors.values))
        if cosine[np.where(cosine > 1)].size:
            warn("RVEA.py line 60 cosine larger than 1 decreased to 1")
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            warn("RVEA.py line 64 cosine smaller than 0 increased to 0")
            cosine[np.where(cosine < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        # Reference vector assignment
        assigned_vectors = np.argmax(cosine, axis=1)
        selection = np.array([], dtype=int)
        # Selection
        # Convert zeros to eps to avoid divide by zero.
        # Has to be checked!
        refV[refV == 0] = np.finfo(float).eps
        for i in range(0, len(vectors.values)):
            sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors == i))
            )

            # Constraint check
            if len(sub_population_index) > 1 and pop.constraint is not None:
                violation_values = pop.constraint[sub_population_index]
                # violation_values = -violation_values
                violation_values = np.maximum(0, violation_values)
                # True if feasible
                feasible_bool = (violation_values == 0).all(axis=1)

                # Case when entire subpopulation is infeasible
                if (feasible_bool == False).all():
                    violation_values = violation_values.sum(axis=1)
                    sub_population_index = sub_population_index[
                        np.where(violation_values == violation_values.min())
                    ]
                # Case when only some are infeasible
                else:
                    sub_population_index = sub_population_index[feasible_bool]

            sub_population_fitness = translated_fitness[sub_population_index]
            # fast tracking singly selected individuals
            if len(sub_population_index) == 1:
                selx = sub_population_index
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
            elif len(sub_population_index) > 1:
                # APD Calculation
                angles = theta[sub_population_index, i]
                angles = np.divide(angles, refV[i])  # This is correct.
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude = np.sqrt(
                    np.sum(np.power(sub_population_fitness, 2), axis=1)
                )
                apd = np.multiply(
                    np.transpose(sub_pop_fitness_magnitude),
                    (1 + np.dot(partial_penalty_factor, angles)),
                )
                minidx = np.where(apd == np.nanmin(apd))
                if np.isnan(apd).all():
                    continue
                selx = sub_population_index[minidx]
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
        assigned_vectors = np.argmax(cosine, axis=1)
        #plot_selection(selection, fitness, uncertainty, assigned_vectors, vectors)
        return selection.squeeze()

    def _partial_penalty_factor(self) -> float:
        """Calculate and return the partial penalty factor for APD calculation.
            This calculation does not include the angle related terms, hence the name.
            If the calculated penalty is outside [0, 1], it will round it up/down to 0/1

        Returns
        -------
        float
            The partial penalty value
        """
        if self.time_penalty_function() < 0:
            px = 0
        elif self.time_penalty_function() > 1:
            px = 1
        else:
            px= self.time_penalty_function()
        penalty = ((px) ** self.alpha) * self.n_of_objectives
        return penalty

    def _calculate_fitness(self, pop) -> np.ndarray:
        if self.selection_type == "mean":
            return pop.fitness
        if self.selection_type == "optimistic":
            return pop.fitness - pop.uncertainity
        if self.selection_type == "robust":
            return pop.fitness + pop.uncertainity

def plot_selection(selection, fitness, uncertainty, assigned_vectors, vectors):
    rc('font',**{'family':'serif','serif':['Helvetica']})
    rc('text', usetex=True)
    plt.rcParams.update({'font.size': 17})
    #plt.rcParams["text.usetex"] = True
    vector_anno = np.arange(len(vectors.values))
    fig = plt.figure(1, figsize=(10, 10))
    fig.set_size_inches(4, 4)
    ax = fig.add_subplot(111)
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    plt.xlim(-0.02, 1.75)
    plt.ylim(-0.02, 1.75)

    #plt.scatter(vectors.values[:, 0], vectors.values[:, 1])
    sx= selection.squeeze()
    #plt.scatter(translated_fitness[sx,0],translated_fitness[sx,1])
    #plt.scatter(fitness[sx, 0], fitness[sx, 1])
    #for i, txt in enumerate(vector_anno):
    #    ax.annotate(vector_anno, (vectors.values[i, 0], vectors.values[i, 1]))

    [plt.arrow(0, 0, dx, dy, color='magenta', length_includes_head=True,
               head_width=0.02, head_length=0.04) for ((dx, dy)) in vectors.values]

    plt.errorbar(fitness[:, 0], fitness[:, 1], xerr=1.96 * uncertainty[:, 0], yerr=1.96 * uncertainty[:, 1],fmt='*', ecolor='r', c='r')
    for i in vector_anno:
        ax.annotate(vector_anno[i], ((vectors.values[i, 0]+0.01), (vectors.values[i, 1]+0.01)))


    plt.errorbar(fitness[sx,0],fitness[sx,1], xerr=1.96*uncertainty[sx, 0], yerr=1.96*uncertainty[sx, 1],fmt='o', ecolor='g',c='g')
    #for i in range(len(assigned_vectors)):
    #    ax.annotate(assigned_vectors[i], ((fitness[i, 0]+0.01), (fitness[i, 1]+0.01)))
        #ax.annotate(assigned_vectors[i], ((fitness[i, 0] + 0.01), (fitness[i, 1] + 0.01)))
    for i in range(len(assigned_vectors)):
        if np.isin(i, sx):
            ax.annotate(assigned_vectors[i], ((fitness[i, 0]-0.1), (fitness[i, 1]-0.1)))
        else:
            ax.annotate(assigned_vectors[i], ((fitness[i, 0]+0.01), (fitness[i, 1]+0.01)))
    plt.show()
    fig.savefig('t_select_gen.pdf',bbox_inches='tight')
    ax.cla()
    fig.clf()
    print("plotted!")