from matplotlib import pyplot as plt
import pandas as pd


def plot_c(c, alpha, threshold):
    """ A function to visualize the changing c value (vertical) and
    number of updates (horizontal)

    Args:
        c: temperature parameter
        alpha: decreasing factor
        threshold: threshold for termination condition
    """
    c_list = [c]
    while c > threshold:
        c = c * alpha
        c_list.append(c)
    plt.plot(c_list)
    plt.show()


def i_need_a_frame(population):
    all = {individual:individual.fitness for individual in population}
    pd_frame = pd.DataFrame.from_dict(all, orient="index", columns = ["Fitness"]).reset_index()
    return pd_frame