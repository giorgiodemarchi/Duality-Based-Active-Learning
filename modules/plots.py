import matplotlib.pyplot as plt
import pandas as pd


def plot_result(accuracies_df, stats_sample_df, stats_full_df, baseline_accuracy, title, show_iqr_sample=False):
    plt.figure(figsize=(15,10))

    # Plot average accuracies
    plt.plot(accuracies_df.reset_index()['index'], accuracies_df['Sample Accuracy'], label='Sample Accuracy')
    plt.plot(accuracies_df.reset_index()['index'], accuracies_df['Full Accuracy'], label='Full Accuracy')

    # Optional: Plot IQR
    if show_iqr_sample:
        plt.fill_between(accuracies_df.reset_index()['index'], stats_sample_df['25%'], stats_sample_df['75%'], color='blue', alpha=0.1, label='Sample Acc IQR')
    plt.fill_between(accuracies_df.reset_index()['index'], stats_full_df['25%'], stats_full_df['75%'], color='orange', alpha=0.3, label='Full Acc IQR')

    # Baseline accuracy and other plot settings
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')
    plt.legend()
    plt.title(title)
    plt.xlabel('Labelled data points')
    plt.ylabel('Accuracy')
    plt.show()


def process_opti_factors(tot_layout_factor, tot_objective_factor):
    ## Build layout factor
    layout_factor = pd.DataFrame.from_dict(tot_layout_factor[0], orient='index').reset_index()
    layout_factor.columns = ['seed',f"value_{0}"]

    for seed in range(1, len(tot_layout_factor)):
        df1 = pd.DataFrame.from_dict(tot_layout_factor[seed], orient='index').reset_index()
        df1.columns = ['seed',f"value_{seed}"]
        layout_factor = pd.merge(layout_factor, df1, on='seed')

    layout_factor['average'] = layout_factor[['value_0','value_1','value_2','value_3','value_4']].mean(axis=1)
    layout_factor['third_qnt'] = layout_factor[['value_0','value_1','value_2','value_3','value_4']].quantile(0.75, axis=1)
    layout_factor['first_qnt'] = layout_factor[['value_0','value_1','value_2','value_3','value_4']].quantile(0.25, axis=1)

    layout_factor.drop(['value_0','value_1','value_2','value_3','value_4'], axis=1, inplace=True)

    # Build objective factor
    objective_factor = pd.DataFrame.from_dict(tot_objective_factor[0], orient='index').reset_index()
    objective_factor.columns = ['seed',f"value_{0}"]
    for seed in range(1, len(tot_objective_factor)):
        df1 = pd.DataFrame.from_dict(tot_objective_factor[seed], orient='index').reset_index()
        df1.columns = ['seed',f"value_{seed}"]
        objective_factor = pd.merge(objective_factor, df1, on='seed')

    objective_factor['average'] = objective_factor[['value_0','value_1','value_2','value_3','value_4']].mean(axis=1)
    objective_factor['third_qnt'] = objective_factor[['value_0','value_1','value_2','value_3','value_4']].quantile(0.75, axis=1)
    objective_factor['first_qnt'] = objective_factor[['value_0','value_1','value_2','value_3','value_4']].quantile(0.25, axis=1)

    objective_factor.drop(['value_0','value_1','value_2','value_3','value_4'], axis=1, inplace=True)

    layout_factor_processed = layout_factor
    layout_factor_processed.rename(columns={'seed':'Size'}, inplace=True)

    objective_factor_processed = objective_factor
    objective_factor_processed.rename(columns={'seed':'Size'}, inplace=True)

    return layout_factor_processed, objective_factor_processed

def compute_stats(df, key, element_index=0):
    max_vals = df[key].apply(lambda x: x[element_index]).max()
    min_vals = df[key].apply(lambda x: x[element_index]).min()
    q25_vals = df[key].apply(lambda x: x[element_index]).quantile(0.25)
    q75_vals = df[key].apply(lambda x: x[element_index]).quantile(0.75)
    return max_vals, min_vals, q25_vals, q75_vals

def process_accuracy(tot_accuracies):
    list_of_dicts = tot_accuracies
    sums = {key: (0, 0) for key in list_of_dicts[0].keys()}
    # Initialize a dictionary to hold the count of entries for each key
    counts = {key: 0 for key in list_of_dicts[0].keys()}

    # Iterate over each dictionary and then each key to accumulate sums and counts
    for d in list_of_dicts:
        for key, value_pair in d.items():
            # Sum the values for each key separately
            sums[key] = (sums[key][0] + value_pair[0], sums[key][1] + value_pair[1])
            # Increment the count for each key
            counts[key] += 1

    # Calculate the average for each key
    averages = {key: ((sums[key][0] / counts[key]), (sums[key][1] / counts[key])) for key in sums.keys()}

    accuracies_df = pd.DataFrame(averages).T
    accuracies_df.columns = ['Sample Accuracy', 'Full Accuracy']

    all_accuracies_df = pd.DataFrame.from_records(list_of_dicts)

    # Compute stats for sample and full accuracy
    stats_sample = {size: compute_stats(all_accuracies_df, size, element_index=0) for size in all_accuracies_df.columns}
    stats_full = {size: compute_stats(all_accuracies_df, size, element_index=1) for size in all_accuracies_df.columns}

    # Convert these stats into DataFrames
    stats_sample_df = pd.DataFrame(stats_sample, index=['Max', 'Min', '25%', '75%']).T
    stats_full_df = pd.DataFrame(stats_full, index=['Max', 'Min', '25%', '75%']).T

    return accuracies_df, stats_sample_df, stats_full_df