import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

def plot(tensor, motif_list, image_path=None):
    C = 2
    
    # Convert to numpy array
    array = tensor.numpy()

    # Define a color map using Tableau colors
    color_map = list(mcolors.TABLEAU_COLORS.values())

    # Get the top 10 and bottom 10 values and their indices for each column
    top_10_values = []
    top_10_labels = []
    top_colors = []

    bottom_10_values = []
    bottom_10_labels = []
    bottom_colors = []

    for col in range(C):
        top_10_indices = np.argsort(array[:, col])[-10:][::-1]
        bottom_10_indices = np.argsort(array[:, col])[:10]

        top_10_values.extend(array[top_10_indices, col])
        top_10_labels.extend([motif_list[idx] for idx in top_10_indices])
        top_colors.extend([color_map[col % len(color_map)]] * 10)

        bottom_10_values.extend(array[bottom_10_indices, col])
        bottom_10_labels.extend([motif_list[idx] for idx in bottom_10_indices])
        bottom_colors.extend([color_map[col % len(color_map)]] * 10)

    # Clear the previous plot
    plt.clf()

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 14))

    # Plot top 10 values
    axes[0].bar(range(len(top_10_values)), top_10_values, color=top_colors)
    axes[0].set_xticks(range(len(top_10_values)))
    axes[0].set_xticklabels(top_10_labels, rotation=90)
    axes[0].set_xlabel('Column and Row Index')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Top 10 Rows in Each Column')
    axes[0].set_ylim(ymin=np.min(top_10_values)-1)

    # Create a legend for top plot
    legend_handles = [plt.Line2D([0], [0], color=color_map[i % len(color_map)], lw=4) for i in range(C)]
    axes[0].legend(legend_handles, [f'Column {i}' for i in range(C)])

    # Plot bottom 10 values
    axes[1].bar(range(len(bottom_10_values)), bottom_10_values, color=bottom_colors)
    axes[1].set_xticks(range(len(bottom_10_values)))
    axes[1].set_xticklabels(bottom_10_labels, rotation=90)
    axes[1].set_xlabel('Column and Row Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title(f'Bottom 10 Rows in Each Column Dataset: {dataset_name}')
    axes[1].set_ylim(ymax=np.max(bottom_10_values)+1)

    # Create a legend for bottom plot
    axes[1].legend(legend_handles, [f'Column {i}' for i in range(C)])
    plt.tight_layout()
    
    if image_path is not None:
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    
def plot_losses(train_losses, val_losses, dataset_name, image_path=None, headers = None):
    plt.figure(figsize=(12, 5))
    
    if headers is None:
        p1_header = f"Training Loss Dataset: {dataset_name}"
        p2_header = f'Validation Loss  Dataset: {dataset_name}'
        y_label = 'Loss'
    else:
        p1_header = headers[0] + f"Dataset: {dataset_name}"
        p2_header = headers[1] + f"Dataset: {dataset_name}"
        y_label = 'ROC-AUC'
    
    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=p1_header, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(p1_header)
    plt.legend()
    
    # Plot validation losses
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label=p2_header, color='red')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(p2_header)
    plt.legend()
    if image_path is not None:
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()
        plt.close()
        
def plot_all_iteration_losses(plot_dict, dataset_name, image_path = None, header = None):
    
    # Plot each list as a line
    for key, values in plot_dict.items():
        plt.plot(range(len(values)), values, label=key)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel(header)
    plt.title(f'Iterations vs {header} vs Epochs')

    # Adding legend in the top right corner
    plt.legend(loc='upper right')

    if image_path is not None:
        image_path = f"{image_path}/{dataset_name}_{header}.png"
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()
        plt.close()
        
        
        

def plot_scatter_with_color(x_values, y_values, color_values, class_labels,title, xlabel, ylabel, cmap='viridis'):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_values, y_values, c=color_values, cmap=cmap, alpha=0.6)

    unique_classes = np.unique(color_values)
    # Unique class labels and corresponding colors
    unique_classes = np.unique(class_labels)
    handles = []
    for cls in unique_classes:
        # Filter for the color associated with the current class label
        index = class_labels.index(cls)
        color = scatter.cmap(scatter.norm(color_values[index]))
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                  markersize=10, linestyle='', label=f'Class {int(cls)}'))

    # plt.legend(handles=handles, title='Classes', loc='upper right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.colorbar(scatter, label='Difference in prediction probability')
    plt.show()
    
def process_data_and_plot(output_dir, dataset_name, vanilla_model):
    for type_split in ["test","validation","train"]:
        with open(f"{output_dir}/{dataset_name}_explanation_result_with_{type_split}.json", 'r') as fp:
            res = json.load(fp)

        x_values = []
        y_values = []
        diff_values = []
        exp_diff_values = []
        class_labels = []

        for key, value in res.items():
            x_values.append(float(vanilla_model.motif_params[value[3]][0].cpu().detach().numpy()))
            y_values.append(float(vanilla_model.motif_params[value[3]][1].cpu().detach().numpy()))
            diff_values.append(value[0] - value[1])
            exp_diff_values.append(np.exp(value[0]) - np.exp(value[1]))
            class_labels.append(value[2])  # Assuming value[2] is the class label


        # Separate class plots
        for class_label in np.unique(class_labels):
            class_x_values = [x for x, cls in zip(x_values, class_labels) if cls == class_label]
            class_y_values = [y for y, cls in zip(y_values, class_labels) if cls == class_label]
            class_diff_values = [diff for diff, cls in zip(diff_values, class_labels) if cls == class_label]
            class_exp_diff_values = [exp_diff for exp_diff, cls in zip(exp_diff_values, class_labels) if cls == class_label]


            plot_scatter_with_color(class_x_values, class_y_values, class_exp_diff_values, [class_label]*len(class_x_values),
                                    title=f'{dataset_name} Class {int(class_label)} Scatter plot colored by marginal difference in prediction prbability for {type_split}',
                                    xlabel='Class 0 Importance',
                                    ylabel='Class 1 importance',
                                    cmap='winter')