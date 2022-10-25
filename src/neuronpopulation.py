import numpy as np
from functools import lru_cache
import math

STIMULI_COUNT = 1000

COMPUTATIONAL_INVERSE_PRECISION_NR_POINTS = 1000

# ==== utility function ====

# calculates the inverses of a vector of functions
def inverse_calculator(f_vec, lower_bound, higher_bound):
    steps = COMPUTATIONAL_INVERSE_PRECISION_NR_POINTS
    
    y_vec_to_x = {}
    
    for i, x in enumerate(np.linspace(lower_bound, higher_bound, steps)):
        y_vec = [f(x) for f in f_vec]
        y_vec_to_x[tuple(y_vec)] = x
    
    return y_vec_to_x
    

# ==== completely functional tuning curve functions ====

# neuron_multiplier: which neuron's tuning curve this is, as a fraction of the total number of neurons
@lru_cache(maxsize=None)
def s(x, stimulus_range, frequency, avg_nr_peaks, invert, x_offset):
    maxAmplitude = 15
    peak_frequency = frequency
    
    amplitude = maxAmplitude
    y_offset = maxAmplitude

    result = amplitude * np.sin(2 * np.pi * peak_frequency * (x - x_offset)) + y_offset
    return result


# derivative of s
@lru_cache(maxsize=None)
def sp(x, stimulus_range, frequency, nr_peaks, invert, x_offset):
    maxAmplitude = 15
    peak_frequency = frequency
    
    amplitude = maxAmplitude
    result = 2 * np.pi * amplitude * peak_frequency * np.cos(2 * np.pi * peak_frequency * (x - x_offset))

    return result



# inverse of s
def si(y, inverses):
    y_vec = list(inverses.keys())
    closest_point = y_vec[0]
    closest_distance = np.linalg.norm(closest_point - y)
    
    for y_cur in y_vec:
        y_cur_distance = np.linalg.norm(y - y_cur)
        if y_cur_distance < closest_distance:
            closest_distance = y_cur_distance
            closest_point = y_cur
    
    return inverses[closest_point]



    
# ==== main neuron population class ====

class NeuronPopulation:
    def __init__(self, nr_neurons, nr_peaks, add_channel_noise = False):
        # population properties
        self.nr_neurons = nr_neurons
        self.nr_peaks = nr_peaks
        
        # noise added to stimulus encodings in the population
        self.set_noise(0, 0)
        self.add_channel_noise = add_channel_noise
        
        # bounds of the neuron's stimulus range
        self.stimulus_range_lower_bound = 0
        self.stimulus_range_higher_bound = 1
        self.stimulus_range = self.stimulus_range_higher_bound - self.stimulus_range_lower_bound
        
        # generate modules
        self.nr_modules = 4
        self.module_indices = np.zeros(self.nr_modules, dtype=int)
        self.generate_evenly_distributed_modules()
        self.frequencies = np.empty(self.nr_neurons)
        self.x_offsets = np.empty(self.nr_neurons)
        self.generate_frequencies_according_to_modules()
        
        
        # generate tuning curves
        self.generate_tuning_curves()

    
    def generate_frequencies_according_to_modules(self):
        peak_frequency = self.nr_peaks / self.stimulus_range
        
        # period 
        peak_frequency_interval = self.nr_modules / sum([i for i in range(1, self.nr_modules + 1)]) * self.nr_peaks
        
        frequency_per_module = np.empty(self.nr_modules)
        for i in range(self.nr_modules):
            frequency_per_module[i] = (i+1) * peak_frequency_interval

        curr_module = 0
        curr_neuron_in_module = 0
        total_neurons_per_module = self.nr_neurons / self.nr_modules
        frequency_increment = 0.4*peak_frequency_interval
        start_frequency = -0.1*peak_frequency_interval + frequency_per_module[curr_module]
        for neuron in range(self.nr_neurons):
            if neuron > self.module_indices[curr_module]:
                curr_module += 1
                curr_neuron_in_module = 0
                start_frequency = -0.1*peak_frequency_interval + frequency_per_module[curr_module]
            self.x_offsets[neuron] = (curr_neuron_in_module / total_neurons_per_module) * (1 / frequency_per_module[curr_module])
            curr_neuron_in_module += 1
            self.frequencies[neuron] = start_frequency + (curr_neuron_in_module/total_neurons_per_module) * frequency_increment
            
    def generate_evenly_distributed_modules(self):
        rest = self.nr_neurons % self.nr_modules
        curr_index = -1
        for curr_module in range(self.nr_modules):
            neurons_in_this_module = math.ceil(self.nr_neurons / self.nr_modules) if curr_module < rest else math.floor(self.nr_neurons / self.nr_modules)
            curr_index += neurons_in_this_module
            self.module_indices[curr_module] = int(curr_index)

    def set_noise(self, std, cor):
        self.noise_std = std
        self.noise_correlation = cor
        
        self.correlation_matrix = np.full(
            (self.nr_neurons, self.nr_neurons), 
            self.noise_correlation
        )
        
        # set diagonal of correlation matrix to 1
        for i in range(self.nr_neurons):
            self.correlation_matrix[i, i] = 1
    
    
    # generates tuning curves, their inverses and their derivatives
    def generate_tuning_curves(self):
        s_vec = []
        si_vec = []
        sp_vec = []
                
        def tuning_curve_factory(x_offset, frequency, invert):
            def s_wrapper(x):
                return s(x, self.stimulus_range, frequency, self.nr_peaks, invert, x_offset)
            def sp_wrapper(x):
                return sp(x, self.stimulus_range, frequency, self.nr_peaks, invert, x_offset)
            
            return [s_wrapper, sp_wrapper]
        
        curr_module = 1
        for i in range(self.nr_neurons):
            if i > self.module_indices[curr_module]:
                curr_module += 1
                
            tcurves = tuning_curve_factory(self.x_offsets[i], self.frequencies[i], False)
            s_vec.append(tcurves[0])
            sp_vec.append(tcurves[1])
        
        inverses = inverse_calculator(s_vec, 0, 1)
        def si_wrapper(y):
            return si(y, inverses)
        
        
        self.s_vec = s_vec
        self.sp_vec = sp_vec
        self.si_vec = si_wrapper

    
    # encode a stimulus value into an array of spike frequencies. Also adds observation uncertainty as specified by self.set_noise(std, cor)
    # stimuli: either one stimulus for all neurons, or an array of different stimulus for each respective neuron
    def encode(self, stimuli, noisy=True):
        encoded_frequencies = np.empty(self.nr_neurons)
        
        observation_uncertainties = np.random.multivariate_normal(
            np.zeros(self.nr_neurons), #mean
            self.noise_std ** 2  * self.correlation_matrix #covariance matrix
        )
        
        for i in range(self.nr_neurons):
            stimulus = stimuli if np.isscalar(stimuli) else stimuli[i]
            # add observation uncertainty
            if noisy:
                stimulus += observation_uncertainties[i]
            stimulus = min([self.stimulus_range_higher_bound, stimulus])
            stimulus = max([self.stimulus_range_lower_bound, stimulus])
            activity = self.s_vec[i](stimulus)
            # add channel noise
            if noisy and self.add_channel_noise:
                activity = np.random.poisson(activity)
            encoded_frequencies[i] = activity
        return encoded_frequencies

            
    # decode an array of spike frequencies to a stimulus value
    def decode(self, encoded_frequencies):
        return self.si_vec(encoded_frequencies)
    
    
    def calculate_sq_error(self, stimulus):
        return (stimulus - self.decode(self.encode(stimulus))) ** 2
    
    
    def approximate_sq_error(self, stimulus):
        # pre-calculate s' for all neurons
        precalc_sp = [self.sp_vec[i](stimulus) for i in range(self.nr_neurons)]
        
        inner_sum = 0
        for i in range(self.nr_neurons):
            for j in range(self.nr_neurons):
                pairwise_correlation = self.correlation_matrix[i, j]
                inner_sum += (pairwise_correlation * (precalc_sp[i] ** 2) * (precalc_sp[j] ** 2))
        error = (self.noise_std ** 2) * inner_sum / (np.linalg.norm(precalc_sp) ** 4) #vector -> 2norm = rätt
        
        if self.add_channel_noise:
            precalc_s = [self.s_vec[i](stimulus) for i in range(self.nr_neurons)]

            inner_sum = 0
            for i in range(self.nr_neurons):
                for j in range(self.nr_neurons):
                    pairwise_correlation = 1 if i == j else 0
                    inner_sum += pairwise_correlation * (np.sqrt(max([self.s_vec[i](stimulus) * self.s_vec[j](stimulus), 0])) / 1000) * precalc_sp[i] *  precalc_sp[j]
            error += inner_sum / (np.linalg.norm(precalc_sp) ** 4)

        return error
        
            
    def calculate_mse_range(self):
        tot = 0
        stimuli = np.linspace(self.stimulus_range_lower_bound, self.stimulus_range_higher_bound, STIMULI_COUNT)
        for stimulus in stimuli:
            tot += self.calculate_sq_error(stimulus)
        return tot/len(stimuli)
        
        
    def approximate_mse_range(self):
        tot = 0
        stimuli = np.linspace(0, self.stimulus_range_higher_bound, STIMULI_COUNT)
        for stimulus in stimuli:
            tot += self.approximate_sq_error(stimulus)
        return tot/len(stimuli)
    
    def approximate_threshold_distortion_relative(self, stimulus):
        return self.calculate_sq_error(stimulus) / self.approximate_sq_error(stimulus)
    
    def approximate_threshold_distortion_absolute(self, stimulus):
        return self.calculate_sq_error(stimulus) - self.approximate_sq_error(stimulus)
        

# ==== main simulator class ====

class Simulator:
    def __init__(self, neuron_population):
        self.neuron_population = neuron_population
        
    def simulation_and_approximation__noise_std_and_noise_correlation(self, stds, cors):
        simulation_errors = np.empty((len(stds), len(cors)))
        approximation_errors = np.empty((len(stds), len(cors)))

        for i, std in enumerate(stds):
            for j, cor in enumerate(cors):
                self.neuron_population.set_noise(std, cor)
                simulation_errors[i, j] = self.neuron_population.calculate_mse_range()
                approximation_errors[i, j] = self.neuron_population.approximate_mse_range()
        return simulation_errors, approximation_errors
    
    def simulation_and_approximation(self):
        return self.neuron_population.calculate_mse_range(), self.neuron_population.approximate_mse_range()
    