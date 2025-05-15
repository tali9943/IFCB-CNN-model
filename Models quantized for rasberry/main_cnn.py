import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import psutil
import time
import csv




def create_data_loaders(test_dir, batch_size, num_workers):
    """
    Crea i DataLoader per i set di addestramento e test.

    Args:
        train_dir (str): Percorso alla directory del set di addestramento.
        test_dir (str): Percorso alla directory del set di test.
        batch_size (int, opzionale): Dimensione del batch. Default è 32.
        num_workers (int, opzionale): Numero di processi per il caricamento dei dati. Default è 2.

    Returns:
        tuple: Contiene il DataLoader per il set di addestramento e quello per il set di test.
    """
    # Definisci le trasformazioni da applicare alle immagini
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Crea i dataset utilizzando ImageFolder
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # Crea i DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return  test_loader




def create_data_loaders2(dataset_path, batch_size=1, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Modify test_loader to include image paths
    def custom_loader():
        for path, label in test_dataset.samples:
            img = test_dataset.loader(path)
            if transform is not None:
                img = transform(img)
            yield img, label, path

    test_loader = torch.utils.data.DataLoader(
        list(custom_loader()), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader



def plot_per_class_accuracy_continuous(per_class_accuracy, class_names, color_scale='RdYlBu'):
    """
    Creates an interactive bar chart for per-class accuracy with a continuous color scale.

    Args:
        per_class_accuracy (list or array): Accuracy for each class.
        class_names (list): Names of the classes.
        color_scale (str or list, optional): Plotly color scale to use. Default is 'RdYlBu'.
    """
    # Create a DataFrame
    df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': per_class_accuracy
    })

    fig = px.bar(
        df,
        x='Class',
        y='Accuracy',
        color='Accuracy',
        color_continuous_scale=color_scale,
        hover_data={'Class': True, 'Accuracy': ':.2f'},
        labels={'Accuracy': 'Accuracy'},
        title='Per-Class Accuracy',
        range_y=[0, 1]
    )

    fig.update_layout(
        xaxis_title='Classes',
        yaxis_title='Accuracy',
        coloraxis_colorbar=dict(
            title="Accuracy",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        )
    )

    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.2f}<extra></extra>'
    )

    fig.show()


def group_classes_by_recall(per_class_recall, class_names, bins=None, plot=True):
    """
    Groups the number of classes based on their recall into specified ranges.

    Args:
        per_class_recall (list or array): Recall values for each class.
        class_names (list): List of class names corresponding to the recall values.
        bins (list of tuples, optional): List of (min, max) tuples defining recall ranges.
                                         Defaults to predefined ranges:
                                         [(0.9, 1.0), (0.8, 0.89), (0.7, 0.79), (0.6, 0.69), (0.0, 0.59)]
        plot (bool, optional): Whether to plot the group counts. Defaults to True.

    Returns:
        dict: A dictionary with bin labels as keys and counts as values.
    """
    if bins is None:
        bins = [
            (0.9, 1.0),
            (0.8, 0.899),
            (0.7, 0.799),
            (0.6, 0.699),
            (0.5, 0.599),
            (0.4, 0.499),
            (0.3, 0.399),
            (0.2, 0.299),
            (0.1, 0.199),
            (0.0, 0.1)
        ]

    # Define bin labels
    bin_labels = [
        '0.9 - 1.0',
        '0.8 - 0.899',
        '0.7 - 0.799',
        '0.6 - 0.699',
        '0.5 - 0.599',
        '0.4 - 0.499',
        '0.3 - 0.399',
        '0.2 - 0.299',
        '0.1 - 0.199',
        '< 0.1'
    ]

    # Initialize counts
    bin_counts = {label: 0 for label in bin_labels}
    bin_class_names = {label: [] for label in bin_labels}

    # Assign each class to a bin
    for recall, class_name in zip(per_class_recall, class_names):
        placed = False
        for (min_val, max_val), label in zip(bins, bin_labels):
            if min_val <= recall <= max_val:
                bin_counts[label] += 1
                bin_class_names[label].append(class_name)
                placed = True
                break
        if not placed:
            print(f"Recall value {recall} for class '{class_name}' does not fit into any bin.")

    # Print the counts
    print("\nNumber of classes in each recall range:")
    for label in bin_labels:
        print(f"{label}: {bin_counts[label]}")

    return bin_counts, bin_class_names


def plot_confusion_matrix(conf_matrix, class_names, save_path=None):

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved at: {save_path}")

    plt.show()

    plt.close()


def plot_true_false_positive(conf_matrix, class_names, save_path=None):

    total_set = conf_matrix.sum(axis=1)  # Total images per class
    true_positive = conf_matrix.diagonal()  # True positives
    false_positive = conf_matrix.sum(axis=0) - conf_matrix.diagonal()  # False positives

    plt.figure(figsize=(16, 8))
    x = np.arange(len(class_names))  # Indices for classes
    width = 0.3  # Width of each bar

    plt.bar(x - width, total_set, width=width, label='Total Set', color='blue')
    plt.bar(x, true_positive, width=width, label='True Positives (TP)', color='green')
    plt.bar(x + width, false_positive, width=width, label='False Positives (FP)', color='red')

    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.xlabel('Classes')
    plt.title('True Positives and False Positives per Class')
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"True/False Positives chart saved at: {save_path}")

    plt.show()

    plt.close()


def plot_per_class_recall(per_class_recall, class_names, accuracy_threshold=0.8, save_path=None):

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, per_class_recall, color='c')
    plt.axhline(y=accuracy_threshold, color='r', linestyle='--', label='Threshold')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Classes')
    plt.ylabel('Recall')
    plt.title('Recall per Class')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Per-class recall chart saved at: {save_path}")


    plt.show()

    plt.close()



def write_results_to_txt(save_path: str,model_name: str, model_path: str, device: torch.device, overall_accuracy: float, precision: float, recall: float, f1: float, class_report: str,
    per_class_accuracy: list, class_names: list, zero_pred_classes: list, memory_used: float, execution_time: float, resources):

    txt_path = os.path.join(save_path, f"{model_name}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Device: {device}\n\n")

        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write(f"Weighted Recall: {recall:.4f}\n")
        f.write(f"Weighted F1-Score: {f1:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(class_report + "\n")

        f.write("Per-Class Accuracy:\n")
        for idx, class_name in enumerate(class_names):
            f.write(f" - {class_name}: {per_class_accuracy[idx]:.4f}\n")

        if zero_pred_classes:
            f.write("\nClasses with no predictions:\n")
            for class_name in zero_pred_classes:
                f.write(f" - {class_name}\n")
        else:
            f.write("\nAll classes have at least one prediction.\n")

        f.write(f"\nGPU memory used during the test: {memory_used:.2f} MB\n")
        f.write(f"Execution time: {execution_time:.2f} seconds\n")
        f.write(f"Resource used(execution_time / memory_used): {resources:.2f}\n")

    print(f"\nResults saved in: {txt_path}")





def test_model(model_path,
              model_architecture=None,
              test_loader=None,
              device=None,
              class_names=None,
              accuracy_threshold=0.8,
              save_path=None):


    if model_architecture is None:
        raise ValueError("model_architecture must be provided.")

    if test_loader is None:
        raise ValueError("test_loader must be provided.")

    if class_names is None:
        raise ValueError("class_names must be provided.")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The specified model_path does not exist or is not a file: {model_path}")

    before_memory = psutil.Process().memory_info().rss  # Memoria RSS in byte
    start_time = time.time()

    device = torch.device('cpu')
    model_architecture.to(device)
    model_architecture.eval()


    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_architecture(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    end_time = time.time()
    after_memory = psutil.Process().memory_info().rss  # Memoria RSS in byte

    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nOverall Accuracy: {overall_accuracy:.4f}')

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    print(f'Weighted Precision: {precision:.4f}')
    print(f'Weighted Recall: {recall:.4f}')
    print(f'Weighted F1-Score: {f1:.4f}')

    conf_matrix = confusion_matrix(all_labels, all_preds)

    class_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print('\nClassification Report:')
    print(class_report)

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    per_class_recall = [report_dict[class_name]['recall'] for class_name in class_names]

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)

    print('Per-Class Accuracy:')
    for idx, class_name in enumerate(class_names):
        print(f' - {class_name}: {per_class_accuracy[idx]:.4f}')

    predictions_per_class = conf_matrix.sum(axis=0)
    zero_pred_classes = [class_names[i] for i, count in enumerate(predictions_per_class) if count == 0]

    if zero_pred_classes:
        print(f"\nClasses with no predictions: {zero_pred_classes}")
    else:
        print("\nAll classes have at least one prediction.")


    memory_used = (after_memory - before_memory) / (1024 ** 2)
    execution_time = end_time - start_time
    resources = execution_time * memory_used



    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        #txt_path = os.path.join(save_path, f"{model_name}.txt")
        #conf_matrix_path = os.path.join(save_path, f"{model_name}_conf_matrix.png")
        #bar_plot_path = os.path.join(save_path, f"{model_name}_barplot.png")
        #recall_plot_path = os.path.join(save_path, f"{model_name}_recall_plot.png")


        #write_results_to_txt(save_path=save_path, model_name=model_name, model_path=model_path, device=device, overall_accuracy=overall_accuracy,
        #    precision=precision, recall=recall, f1=f1, class_report=class_report, per_class_accuracy=per_class_accuracy, class_names=class_names,
        #    zero_pred_classes=zero_pred_classes, memory_used=memory_used, execution_time=execution_time, resources=resources )


        #plot_confusion_matrix(conf_matrix, class_names, save_path=conf_matrix_path)
        #plot_true_false_positive(conf_matrix, class_names, save_path=bar_plot_path)
        #plot_per_class_recall(per_class_recall, class_names, accuracy_threshold=accuracy_threshold, save_path=recall_plot_path)


    results = {
        'overall_accuracy': overall_accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'per_class_recall': per_class_recall,
        'per_class_accuracy': per_class_accuracy,
        'zero_pred_classes': zero_pred_classes
    }


    #print(f"\nCPU memory used during the test: {memory_used:.2f} MB")  # GPU memory used during the test in MB
    #print(f"Execution time: {execution_time:.2f} seconds")  # Total execution time in seconds
    #print(f"Resource used(execution_time / memory_used): {resources:.2f}")  # Execution time to memory usage ratio



    return results






def test_model2(model_path,
               model_architecture=None,
               test_loader=None,
               device=None,
               class_names=None,
               accuracy_threshold=0.8,
               save_path=None):

    if model_architecture is None:
        raise ValueError("model_architecture must be provided.")

    if test_loader is None:
        raise ValueError("test_loader must be provided.")

    if class_names is None:
        raise ValueError("class_names must be provided.")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The specified model_path does not exist or is not a file: {model_path}")

    if save_path is None:
        raise ValueError("save_path must be provided to save the results.")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open the CSV file for writing
    with open(save_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_name", "predicted_class", "true_class", "is_correct"])

        # Set the device and model to evaluation mode
        device = torch.device('cpu') if device is None else torch.device(device)
        model_architecture.to(device)
        model_architecture.eval()

        all_preds = []
        all_labels = []

        # Process each image individually
        with torch.no_grad():
            for inputs, labels, paths in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model_architecture(inputs)
                _, preds = torch.max(outputs, 1)

                # Collect predictions and true labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Write results to the CSV file
                for img_path, pred, true_label in zip(paths, preds, labels):
                    image_name = os.path.basename(img_path)
                    predicted_class = class_names[pred.item()]
                    true_class = class_names[true_label.item()]
                    is_correct = pred.item() == true_label.item()

                    csv_writer.writerow([image_name, predicted_class, true_class, is_correct])

    print(f"Results saved to {save_path}")

    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f'\nOverall Accuracy: {overall_accuracy:.4f}')

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    print(f'Weighted Precision: {precision:.4f}')
    print(f'Weighted Recall: {recall:.4f}')
    print(f'Weighted F1-Score: {f1:.4f}')

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print('\nClassification Report:')
    print(class_report)

    return {
        'overall_accuracy': overall_accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }












def load_quantized_vgg16_model(model_path,dataset):
  class_names = dataset.classes

  quant_model = torch.jit.load(model_path)
  quant_model.eval()
  print("Model Loaded!")

  return quant_model, class_names





if __name__ == "__main__":

    #test_path = '/test'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = './test'
    test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    test_loader = create_data_loaders2(dataset_path, batch_size=1 ,num_workers=0)


    quantized_model_path = './model/vgg16_qat_model_01_18_2025_11_29_0.9527.pth'

    # Carica il modello quantizzato
    quantized_model,class_names = load_quantized_vgg16_model(quantized_model_path, test_dataset)
    print("Modello quantizzato caricato correttamente!")

    save_path= './results/res.csv'


    results = test_model2(model_path = quantized_model_path,
              model_architecture=quantized_model,
              test_loader= test_loader,
              device='cpu',
              class_names=class_names,
              accuracy_threshold=0.8,
              save_path=save_path)