import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

similarity_metrics = ['cosine',
                      'braycurtis',
                      'canberra',
                      'chebyshev',
                      'cityblock',
                      'correlation',
                      'euclidean',
                      'sqeuclidean']


def handle_csv(file_csv, label):
    df = pd.read_csv(file_csv)
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df.set_index('index', inplace=True)
    df_trans = df.transpose()
    df_trans.insert(0, 'label', label)
    return df_trans


def plot_and_save_figure(df_couples, df_diff, category, metric, output):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    min_temp = min(min(df_couples['mean']), min(df_diff['mean'])) - min(min(df_couples['std']), min(df_diff['std']))
    max_temp = max(max(df_couples['mean']), max(df_diff['mean'])) + max(max(df_couples['std']), max(df_diff['std']))

    line_width = 1.5

    # Plot data
    df_couples['mean'].plot(ax=axes, c='blue', lw=line_width,
                            ylim=[min_temp, max_temp], grid=True)
    df_diff['mean'].plot(ax=axes, c='red', lw=line_width,
                         ylim=[min_temp, max_temp], grid=True)
    axes.legend(['Mean similarity value of the couple',
                 'Mean similarity value of the non-couple'])

    # Rotate the x-axis labels so they don't overlap
    plt.setp(axes.xaxis.get_majorticklabels(), rotation=20)

    # Axis labels
    axes.set_xlabel('Model')
    axes.set_ylabel('Value')
    plt.savefig(os.path.join(output, f'{metric}_{category}.png'))
    plt.close('all')


def main():
    individual_path = '../embedding_distance'
    ensemble_path = '../ensemble_model'
    output = '../outputs/results'
    output_figure = '../outputs/figure'

    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(output_figure):
        os.makedirs(output_figure)

    for path, category in zip([individual_path, ensemble_path], ['individual', 'ensemble']):
        for metric in tqdm(similarity_metrics):
            couples_list = []
            diff_list = []
            for r, d, files in os.walk(path):
                for file in tqdm(files):
                    file_path = os.path.join(r, file)
                    label = file_path.split(os.sep)[-3]
                    df = handle_csv(file_path, label)
                    if metric in file_path and 'couples' in file:
                        couples_list.append(df)
                    elif metric in file_path and 'diff' in file:
                        diff_list.append(df)
            # df_couples = pd.concat(couples_list, axis=0, ignore_index=True)
            # df_diff_couples = pd.concat(diff_list, axis=0, ignore_index=True)
            #
            # df_couples.to_csv(os.path.join(output, f'{metric}_{category}_couples.csv'))
            # df_diff_couples.to_csv(os.path.join(output, f'{metric}_{category}_diff.csv'))
            df_couples = pd.read_csv(os.path.join(output, f'{metric}_{category}_couples.csv'))
            df_diff_couples = pd.read_csv(os.path.join(output, f'{metric}_{category}_diff.csv'))

            plot_and_save_figure(df_couples, df_diff_couples, category, metric, output_figure)


if __name__ == "__main__":
    main()
