import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(cumulative_reward_list, episode_length_list, step=10):
    epochs = np.arange(1, len(cumulative_reward_list) + 1)

    # Tomar cada "step" elemento para mostrar en el gráfico
    epochs = epochs[::step]
    cumulative_reward_list = cumulative_reward_list[::step]
    episode_length_list = episode_length_list[::step]

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
    cumulative_reward_list = np.load('Baseline_results/cumulative_reward_list.npy')
    episode_length_list = np.load('Baseline_results/episode_length_list.npy')

    # Llama a la función para graficar las métricas
    plot_metrics(cumulative_reward_list, episode_length_list, step=100)
