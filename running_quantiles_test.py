
import numpy
import matplotlib.pyplot as plt

"""
This experiment contains the code for a memory efficient StatsTracker class and a main function that
executes a simple test. The test consists of creating a random queue of data, feeding the queue to the
StatsTracker, and comparing the estimated statistics to the true statistics.
"""

def main():

    # Decide how long of a simulation to run
    queue_size = 3000
    num_quantiles = 4

    # Along with the queue, track some series to plot how well the estimator did
    real = numpy.zeros((queue_size, num_quantiles))
    estimated = numpy.zeros((queue_size, num_quantiles))
    queue = numpy.zeros((queue_size,))

    # Create a simulation where the queue data is drawn from a distribution which shifts a couple times
    queue[:1000] = numpy.random.normal(loc=0., scale=1., size=1000)
    queue[1000:2000] = numpy.random.normal(loc=1., scale=1., size=1000)
    queue[2000:3000] = numpy.random.normal(loc=-1., scale=2., size=1000)
    seed_data = numpy.random.normal(loc=0., scale=1., size=30)

    # Initialize the tracker object that will estimate statistics of the stream
    tracker = StatsTracker(num_quantiles=num_quantiles, simulated_queue_length=queue_size, seed_data=seed_data)

    # Execute the simulation
    run_queue(queue, real, estimated, tracker)

    # Plot the results from the simulation
    for i in range(num_quantiles):
        plt.plot(real[:, i], color=(i/num_quantiles, 0, (num_quantiles-i)/num_quantiles), label=f"Q{i+1}")
        plt.plot(estimated[:, i], color=(0.5 * (num_quantiles - i)/num_quantiles, 0.5 + 0.5 * i / num_quantiles, 0))
        plt.title("Quantile Track Accuracy")
        plt.ylabel("Datum value")
        plt.xlabel("Sequence in queue")
        plt.legend()
    plt.show()

    print("Done")

def run_queue(queue, real, estimated, tracker):
    # Iteratively perform the quantile update
    queue_size = queue.shape[0]
    num_quantiles = estimated.shape[1]
    for datum_index in range(queue_size):
        datum = queue[datum_index]
        tracker.update_estimates(datum)
        # We'll let the real quantiles be the 300 most recent members from the current distribution.
        # If there aren't 300 members of this distribution yet, then just take all of the members generated
        # from this distribution.
        real[datum_index, :] = numpy.quantile(queue[max(datum_index-299, 0, 1000 * int(datum_index / 1000)):datum_index+1], [i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)])
        estimated[datum_index, :] = tracker.quantiles

class StatsTracker:

    """
    A class that estimates the mean, variance, and quantiles of a queue of data by processing
    one element from the queue at a time.

    num_quantiles - the number of quantile divisions to track (e.g. for quartiles, this would be 3)
    simulated_queue_length - roughly speaking, the number of past data points that should influence the current estimates
    seed_data - at least two dissimilar data points to allow the statistics to be initialized in some reasonable way
    """

    def __init__(self, num_quantiles, simulated_queue_length, seed_data):

        # Store and cache useful values
        self.num_quantiles = num_quantiles
        self.simulated_queue_length = simulated_queue_length
        self.old_mean_weight = (simulated_queue_length - 1) / simulated_queue_length
        self.new_weight = 1 / simulated_queue_length

        # Ensure there is sufficient data for initializing stats
        if seed_data is None or not len(seed_data) > 2:
            raise Exception("Must provide some starting data (at least two points) when initializing a stats tracker.")
    
        # Initialize mean
        self.mean = seed_data[0]
        
        # Initialize variance
        self.variance = None
        for i in range(len(seed_data)):
            if seed_data[i] != self.mean:
                self.variance = abs(seed_data[i] - self.mean)
                break
        if self.variance is None:
            raise Exception("Seed data contained no variation. Couldn't estimate a non-zero variance.")
        
        # Initialize quantiles
        std_dev = self.variance**.5
        self.quantiles = [self.mean + std_dev * (-2 + 4 * (i + 0.5) / (num_quantiles + 1)) for i in range(num_quantiles)]

    def update_estimates(self, datum):

        old_mean = self._update_mean(datum)
        self._update_variance(datum, old_mean)
        self._update_quantiles(datum)
        return self.quantiles


    def _update_mean(self, datum):

        old_mean = self.mean
        self.mean = self.mean * self.old_mean_weight + datum * self.new_weight
        return old_mean


    def _update_variance(self, datum, old_mean):

        self.variance += ((datum - old_mean)**2 / self.simulated_queue_length + (datum - self.mean)**2 - (old_mean - self.mean)**2) / self.simulated_queue_length


    def _update_quantiles(self, datum):

        # Perform a binary search to find the index of the nearest quantile below the datum
        q_lower_index = 0
        q_upper_index = self.num_quantiles - 1
        if datum < self.quantiles[q_lower_index]:
            q_lower_index = -1
        elif datum > self.quantiles[q_upper_index]:
            q_lower_index = q_upper_index
        else:
            while q_upper_index > q_lower_index + 1:
                mean_index = int((q_lower_index + q_upper_index) / 2)
                mean_value = self.quantiles[mean_index]
                if datum > mean_value:
                    q_lower_index = mean_index
                else:
                    q_upper_index = mean_index

        # Shift the quantiles
        a = self.variance**.5 / 10 / (self.num_quantiles + 1)
        step = a
        for i in range(q_lower_index + 1):
            self.quantiles[i] += step
            step += a
        step = (self.num_quantiles - q_lower_index - 1) * a
        for i in range(q_lower_index + 1, self.num_quantiles):
            self.quantiles[i] -= step
            step -= a

if __name__ == "__main__":
    main()
