import matplotlib.pyplot as plt


import seaborn as sns

def plot_loss():
    plt.figure(1)
    plt.figure(figsize=(15,5))
    sns.set_style(style="darkgrid")
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1,181), y=total_train_loss, label="Train Loss")
    sns.lineplot(x=range(1,181), y=total_valid_loss, label="Valid Loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("DiceLoss")

    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1,181), y=total_train_score, label="Train Score")
    sns.lineplot(x=range(1,181), y=total_valid_score, label="Valid Score")
    plt.title("Score (IoU)")
    plt.xlabel("epochs")
    plt.ylabel("IoU")
    plt.show()