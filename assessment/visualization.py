import assessment.preprocessing as preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def variable_plot(variable_name: str, all_data_frame: pd.DataFrame):
    """
    Plotting a given variable
    """
    # Get the poi and non-poi data points
    poi_variable = all_data_frame.poi
    variable_poi = all_data_frame[variable_name][poi_variable]
    variable_no_poi = all_data_frame[variable_name][poi_variable == False]

    # Get variable statistics
    # (variable_poi_mean, variable_poi_std, variable_no_poi_mean, variable_no_poi_std, r, missing_values)
    variable_stats = preprocessing.get_statistics_description(all_data_frame[variable_name], poi_variable)

    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.tick_params(labelsize=14)
    ax.set_title('{}, {:.2f}% missing'.format(variable_name, variable_stats[-1]), fontsize=20)

    # Set up the y axis
    ax.set_ylabel(variable_name, fontsize=18)
    ax.set_ylim(all_data_frame[variable_name].min(), all_data_frame[variable_name].max())
    ax = set_log_scale(ax, all_data_frame[variable_name].min(), all_data_frame[variable_name].max())

    # Set up the x axis
    ax.set_xticks([0, 1])
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticklabels(['POI', 'Not POI'])

    # Plot all the POI points
    ax.scatter(np.zeros(variable_poi.size), variable_poi, c='tab:red', alpha=0.2, marker='o', s=120)
    # Plot all the non-POI points
    ax.scatter(np.ones(variable_no_poi.size), variable_no_poi, c='tab:blue', alpha=0.2, marker='o', s=120)

    # Plot the mean value ± the STD for each condition
    ax.errorbar(0, variable_stats[0], yerr=variable_stats[1], fmt='X', c='black', markersize=20,
                markerfacecolor="None", markeredgewidth=3)
    ax.errorbar(1, variable_stats[2], yerr=variable_stats[3], fmt='X', c='black', markersize=20,
                markerfacecolor="None", markeredgewidth=3)


def set_log_scale(ax, value_min, value_max):
    """
    If the values cover a wide range, we want to change the axis to a log axis so that we can visualize the data
    more clearly
    :param ax: axis object of the plot
    :param value_min: the lowest value to be plotted
    :param value_max: the largest value to be plotted

    return ax: the axis object
    """
    if np.abs(value_max - value_min) > 1000:
        if value_max > 0 and value_min > 0:
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog')
    return ax