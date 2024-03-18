import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(cumulative_reward_list, episode_length_list):
    epochs = np.arange(1, len(cumulative_reward_list) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, cumulative_reward_list, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, episode_length_list, marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Episode Length')
    plt.title('Episode Length per Epoch')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Cargar los datos de las métricas desde los archivos .npy
    cumulative_reward_list = np.load('cumulative_reward_list.npy')
    episode_length_list = np.load('episode_length_list.npy')

    # Llama a la función para graficar las métricas
    plot_metrics(cumulative_reward_list, episode_length_list)
