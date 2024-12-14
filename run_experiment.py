import argparse
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import torch.nn.utils.prune as prune
import pickle

torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", help="Model name from torchvision.models (e.g., resnet18)")
    parser.add_argument("--dataset", type=str, default="CIFAR100", help="Dataset name: CIFAR10 or CIFAR100")
    parser.add_argument("--data_ratio", type=float, default=0.1, help="Fraction of dataset to select based on forgetting scores")
    parser.add_argument("--prune_ratio", type=float, default=0.4, help="Pruning ratio")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for initial training")
    parser.add_argument("--post_epochs", type=int, default=10, help="Number of epochs for post-pruning training")
    args = parser.parse_args()
    return args

def get_dataset_and_out_features(dataset_name):
    if dataset_name.upper() == "CIFAR10":
        DATA = datasets.CIFAR10
        OUT_FEATURES = 10
    else:
        DATA = datasets.CIFAR100
        OUT_FEATURES = 100
    return DATA, OUT_FEATURES

def get_model(model_name, out_features):
    model = getattr(models, model_name)(pretrained=False)
    # Replace the final layer
    if hasattr(model, 'fc') and model.fc is not None:
        model.fc = nn.Linear(model.fc.in_features, out_features)
    elif hasattr(model, 'classifier') and model.classifier is not None:
        # For models like MobileNet or VGG
        if isinstance(model.classifier, nn.Sequential):
            last_layer_idx = -1
            in_features = model.classifier[last_layer_idx].in_features
            model.classifier[last_layer_idx] = nn.Linear(in_features, out_features)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, out_features)
    return model

def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return np.array(predictions)

def train_with_forgetting_tracking(model, trainloader, trainset, criterion, optimizer, num_epochs, device):
    print(f"Starting baseline training for {num_epochs} epochs with forgetting tracking...")
    forgetting_scores = {i: [] for i in range(len(trainset))}
    initial_predictions = get_predictions(model, trainloader, device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_predictions = get_predictions(model, trainloader, device)

        for i in range(len(trainset)):
            if epoch == 0:
                forgetting_scores[i].append(0)
            else:
                forgetting_score = int(initial_predictions[i] != epoch_predictions[i])
                forgetting_scores[i].append(forgetting_score)

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Training Loss: {running_loss/len(trainloader):.4f}, Accuracy: {epoch_acc:.2f}%")

        initial_predictions = epoch_predictions

    print("Baseline training with forgetting tracking completed.")
    return forgetting_scores

def plot_forgetting_scores(forgetting_scores, args, filename='forgetting_scores.png'):
    # Compute the total forgetting score for each sample
    total_forgetting_scores = [sum(scores) for scores in forgetting_scores.values()]

    plt.figure(figsize=(10,6))
    plt.hist(total_forgetting_scores, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Forgetting Score (# of forget events)')
    plt.ylabel('Number of Examples')
    plt.title('Distribution of Forgetting Scores')
    plt.suptitle(f"Model: {args.model}, Dataset: {args.dataset}, Data Ratio: {args.data_ratio}, Prune Ratio: {args.prune_ratio}", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_forgettability(forgetting_scores, args, filename='forgettability.png'):
    # Calculate total forgettability scores for each sample
    total_forgetting_scores = {i: sum(forgetting_scores[i]) for i in forgetting_scores}
    
    # Sort samples by their total forgetting score in descending order
    sorted_indices = sorted(total_forgetting_scores, key=total_forgetting_scores.get, reverse=True)
    sorted_forgetting_vals = [total_forgetting_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_indices)), sorted_forgetting_vals, color='b', alpha=0.7)
    plt.xlabel('Sample Index (Sorted by Forgettability)', fontsize=12)
    plt.ylabel('Total Forgettability Score', fontsize=12)
    plt.title('Total Forgettability Scores (All Samples, Sorted)')
    plt.suptitle(f"Model: {args.model}, Dataset: {args.dataset}, Data Ratio: {args.data_ratio}, Prune Ratio: {args.prune_ratio}", fontsize=10)
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def get_highest_and_lowest_forgetting_scores(forgetting_scores, fraction=0.5):
    total_forgetting_scores = {i: sum(forgetting_scores[i]) for i in forgetting_scores}
    sorted_indices_high = sorted(total_forgetting_scores, key=total_forgetting_scores.get, reverse=True)
    sorted_indices_low = sorted(total_forgetting_scores, key=total_forgetting_scores.get)
    top_n = int(fraction * len(forgetting_scores))
    highest_forgetting_indices = sorted_indices_high[:top_n]
    lowest_forgetting_indices = sorted_indices_low[:top_n]
    print(f"Selected top {top_n} highest-forgetting and top {top_n} lowest-forgetting samples.")
    return highest_forgetting_indices, lowest_forgetting_indices

def train_model(model, trainloader, criterion, optimizer, num_epochs, device):
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] complete: Loss: {running_loss/len(trainloader):.4f}, Accuracy: {epoch_acc:.2f}%")
    print("Training completed.")

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy

def train_and_test_model(model, trainloader, test_loader, criterion, optimizer, num_epochs, device):
    test_accuracies = []
    test_eval = evaluate_model(model, test_loader, device=device) # initial eval
    test_accuracies.append(test_eval)
    print(f"Initial Test Accuracy: {test_eval:.2f}%")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_eval = evaluate_model(model, test_loader, device=device)
        test_accuracies.append(test_eval)
        epoch_acc = 100.0 * correct / total
        print(f"Post-Pruning Training Epoch [{epoch+1}/{num_epochs}]: Loss: {running_loss/len(trainloader):.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_eval:.2f}%")

    return test_accuracies

def apply_pruning(model, pruning_percentage=0.4):
    print(f"Applying pruning with ratio {pruning_percentage}...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=pruning_percentage)
            prune.remove(module, 'weight')
    print("Pruning completed.")
    return model

def plot_pruning_accuracy_comparison(baseline_accuracy, pruned_baseline_accuracy,
                                     low_scores_accuracy, pruned_low_scores_accuracy,
                                     high_scores_accuracy, pruned_high_scores_accuracy,
                                     args, output_dir):
    accuracies = [
        baseline_accuracy, pruned_baseline_accuracy,
        low_scores_accuracy, pruned_low_scores_accuracy,
        high_scores_accuracy, pruned_high_scores_accuracy
    ]

    model_names = [
        'Baseline Model', 'Baseline Model (Pruned)',
        'Low Forgetting Model', 'LF Model (Pruned)',
        'High Forgetting Model', 'HF Model (Pruned)'
    ]

    # Number of bars in each group (we have three groups: baseline, low forgetting, high forgetting)
    n_groups = 3

    # Bar widths
    bar_width = 0.35

    # Define the positions for the bars
    index = np.arange(n_groups)

    # Arrange the accuracies for each group
    accuracies_grouped = [
        [accuracies[0], accuracies[1]],  # Baseline and pruned baseline
        [accuracies[2], accuracies[3]],  # Low forgetting and pruned low forgetting
        [accuracies[4], accuracies[5]]   # High forgetting and pruned high forgetting
    ]

    fig, ax = plt.subplots(figsize=(6, 6))

    bar1 = ax.bar(index - bar_width / 2, [group[0] for group in accuracies_grouped], bar_width, label='Before Pruning', color='blue')
    bar2 = ax.bar(index + bar_width / 2, [group[1] for group in accuracies_grouped], bar_width, label='After Pruning', color='lightblue')

    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Pruning Test Accuracy Comparison\n', fontsize=14, ha='center')
    plt.figtext(0.5, 0.915, f'(Epochs: {args.epochs}, Pruning: {args.prune_ratio * 100:.0f}%)', ha='center')
    ax.set_xticks(index)
    ax.set_xticklabels(['Baseline Model', 'Low Forgetting Model', 'High Forgetting Model'], rotation=15, ha='center', fontsize=10)
    ax.set_ylim(0, 1.2 * max(accuracies))

    # Add value labels on top of bars
    for bars in [bar1, bar2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

    # Add legend
    ax.legend()

    plt.tight_layout()
    bar_plot_filename = f"{output_dir}/pruning_accuracy_comparison.png"
    plt.savefig(bar_plot_filename)
    plt.close()
    print(f"Pruning accuracy comparison bar plot saved to {bar_plot_filename}.")

if __name__ == "__main__":
    args = parse_args()

    # Directory structure:
    # results/<model>/<dataset>/
    #     baseline_model.pth
    #     forgetting_scores.pkl
    #     dr<data_ratio>/
    #         pr<prune_ratio>/
    #             forgetting_scores.png
    #             forgettability.png
    #             accuracies.csv
    #             post_pruning_accuracy.png

    print("=============================================================================================================")

    base_model_dataset_dir = f"results/{args.model}/{args.dataset}"
    dr_dir = f"{base_model_dataset_dir}/dr{args.data_ratio}"
    current_run_dir = f"{dr_dir}/pr{args.prune_ratio}"

    os.makedirs(dr_dir, exist_ok=True)
    os.makedirs(current_run_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA, OUT_FEATURES = get_dataset_and_out_features(args.dataset.upper())
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = DATA(root="data", train=True, download=True, transform=transform)
    testset = DATA(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)

    forgetting_scores_path = f"{base_model_dataset_dir}/forgetting_scores.pkl"
    baseline_model_path = f"{base_model_dataset_dir}/baseline_model.pth"

    if os.path.exists(forgetting_scores_path) and os.path.exists(baseline_model_path):
        print("Loading precomputed forgetting scores and baseline model...")
        with open(forgetting_scores_path, "rb") as f:
            forgetting_scores = pickle.load(f)
        baseline_model = get_model(args.model, OUT_FEATURES).to(device)
        baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=device))
        print("Baseline model and forgetting scores loaded.")
    else:
        print("Forgetting scores and baseline model not found. Computing from scratch...")
        baseline_model = get_model(args.model, OUT_FEATURES)
        baseline_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

        forgetting_scores = train_with_forgetting_tracking(
            baseline_model, train_loader, trainset, criterion, optimizer,
            num_epochs=args.epochs, device=device
        )

        with open(forgetting_scores_path, "wb") as f:
            pickle.dump(forgetting_scores, f)
        torch.save(baseline_model.state_dict(), baseline_model_path)
        print("Forgetting scores and baseline model saved for future runs.")

        # Plot forgetting scores & forgettability
        fs_filename = f'{base_model_dataset_dir}/forgetting_scores.png'
        fg_filename = f'{base_model_dataset_dir}/forgettability.png'
        plot_forgetting_scores(forgetting_scores, args, filename=fs_filename)
        plot_forgettability(forgetting_scores, args, filename=fg_filename)

 

    # Data selection
    highest_forgetting_indices, lowest_forgetting_indices = get_highest_and_lowest_forgetting_scores(forgetting_scores, fraction=args.data_ratio)
    print(f"Highest forgetting subset size: {len(highest_forgetting_indices)}, Lowest forgetting subset size: {len(lowest_forgetting_indices)}")

    high_subset_trainset = Subset(trainset, highest_forgetting_indices)
    high_trainloader = DataLoader(high_subset_trainset, batch_size=64, shuffle=True, num_workers=4)

    low_subset_trainset = Subset(trainset, lowest_forgetting_indices)
    low_trainloader = DataLoader(low_subset_trainset, batch_size=64, shuffle=True, num_workers=4)

    # Train models on subsets
    print("Training model on high-forgetting subset...")
    model_high = get_model(args.model, OUT_FEATURES).to(device)
    optimizer_high = optim.Adam(model_high.parameters(), lr=0.001)
    train_model(model_high, high_trainloader, nn.CrossEntropyLoss(), optimizer_high, num_epochs=args.epochs, device=device)

    print("Training model on low-forgetting subset...")
    model_low = get_model(args.model, OUT_FEATURES).to(device)
    optimizer_low = optim.Adam(model_low.parameters(), lr=0.001)
    train_model(model_low, low_trainloader, nn.CrossEntropyLoss(), optimizer_low, num_epochs=args.epochs, device=device)

    criterion = nn.CrossEntropyLoss()
    baseline_accuracy = evaluate_model(baseline_model, test_loader, device=device)
    low_scores_accuracy = evaluate_model(model_low, test_loader, device=device)
    high_scores_accuracy = evaluate_model(model_high, test_loader, device=device)

    print(f"Baseline Model Accuracy: {baseline_accuracy:.2f}%")
    print(f"Low Forgetting Model Accuracy: {low_scores_accuracy:.2f}%")
    print(f"High Forgetting Model Accuracy: {high_scores_accuracy:.2f}%")

    # Apply pruning
    baseline_model_pruned = apply_pruning(copy.deepcopy(baseline_model), args.prune_ratio)
    low_model_pruned = apply_pruning(copy.deepcopy(model_low), args.prune_ratio)
    high_model_pruned = apply_pruning(copy.deepcopy(model_high), args.prune_ratio)

    pruned_baseline_accuracy = evaluate_model(baseline_model_pruned, test_loader, device=device)
    pruned_low_scores_accuracy = evaluate_model(low_model_pruned, test_loader, device=device)
    pruned_high_scores_accuracy = evaluate_model(high_model_pruned, test_loader, device=device)

    plot_pruning_accuracy_comparison(
    baseline_accuracy, pruned_baseline_accuracy,
    low_scores_accuracy, pruned_low_scores_accuracy,
    high_scores_accuracy, pruned_high_scores_accuracy,
    args, current_run_dir
)

    print(f"Pruned Baseline Model Accuracy: {pruned_baseline_accuracy:.2f}%")
    print(f"Pruned Low Forgetting Model Accuracy: {pruned_low_scores_accuracy:.2f}%")
    print(f"Pruned High Forgetting Model Accuracy: {pruned_high_scores_accuracy:.2f}%")

    # Post pruning retraining
    print("Post-Pruning Retraining...")
    post_pruned_baseline = copy.deepcopy(baseline_model_pruned)
    post_pruned_low = copy.deepcopy(low_model_pruned)
    post_pruned_high = copy.deepcopy(high_model_pruned)

    post_base_optim = optim.Adam(post_pruned_baseline.parameters(), lr=0.001)
    post_low_optim = optim.Adam(post_pruned_low.parameters(), lr=0.001)
    post_high_optim = optim.Adam(post_pruned_high.parameters(), lr=0.001)

    base_acc_by_epoch = train_and_test_model(post_pruned_baseline, train_loader, test_loader, criterion, post_base_optim, num_epochs=args.post_epochs, device=device)
    low_acc_by_epoch = train_and_test_model(post_pruned_low, train_loader, test_loader, criterion, post_low_optim, num_epochs=args.post_epochs, device=device)
    high_acc_by_epoch = train_and_test_model(post_pruned_high, train_loader, test_loader, criterion, post_high_optim, num_epochs=args.post_epochs, device=device)

    post_pruned_baseline_accuracy = evaluate_model(post_pruned_baseline, test_loader, device=device)
    post_pruned_low_accuracy = evaluate_model(post_pruned_low, test_loader, device=device)
    post_pruned_high_accuracy = evaluate_model(post_pruned_high, test_loader, device=device)

    print(f"Post-Pruned Baseline Model Accuracy: {post_pruned_baseline_accuracy:.2f}%")
    print(f"Post-Pruned Low Forgetting Model Accuracy: {post_pruned_low_accuracy:.2f}%")
    print(f"Post-Pruned High Forgetting Model Accuracy: {post_pruned_high_accuracy:.2f}%")

    # Save accuracy results
    accuracies_filename = f"{current_run_dir}/accuracies.csv"
    first_line = not os.path.exists(accuracies_filename)
    with open(accuracies_filename, 'a') as f:
        if first_line:
            f.write("model,dataset,data_ratio,prune_ratio,epochs,post_epochs,baseline_accuracy,low_scores_accuracy,high_scores_accuracy,pruned_baseline_accuracy,pruned_low_scores_accuracy,pruned_high_scores_accuracy,post_pruned_baseline_accuracy,post_pruned_low_accuracy,post_pruned_high_accuracy\n")
        line = f"{args.model},{args.dataset},{args.data_ratio},{args.prune_ratio},{args.epochs},{args.post_epochs}," \
               f"{baseline_accuracy},{low_scores_accuracy},{high_scores_accuracy}," \
               f"{pruned_baseline_accuracy},{pruned_low_scores_accuracy},{pruned_high_scores_accuracy}," \
               f"{post_pruned_baseline_accuracy},{post_pruned_low_accuracy},{post_pruned_high_accuracy}\n"
        f.write(line)
    print(f"Accuracy results saved to {accuracies_filename}.")

    # Plot Post-Pruning Accuracy over Additional Training Epochs
    post_pruning_plot = f'{current_run_dir}/post_pruning_accuracy.png'
    plt.figure(figsize=(8, 6))
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xlabel('Additional Epochs Trained', fontsize=12)
    plt.title('Post-Pruning Test Accuracy')
    plt.suptitle(f"Model: {args.model}, Dataset: {args.dataset}, Data Ratio: {args.data_ratio}, Prune Ratio: {args.prune_ratio}", fontsize=10)
    epochs_range = list(range(1, args.post_epochs + 1))
    plt.plot(epochs_range, base_acc_by_epoch[1:], 'b-', label="Baseline Model")
    plt.plot(epochs_range, low_acc_by_epoch[1:], 'g-', label="Low Forgetting Score Model")
    plt.plot(epochs_range, high_acc_by_epoch[1:], 'r-', label="High Forgetting Score Model")

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(post_pruning_plot)
    plt.close()
    print(f"Post-pruning accuracy plot saved to {post_pruning_plot}.")
