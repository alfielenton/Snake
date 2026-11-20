from game import Snake
from agent.player import Agent

from matplotlib import pyplot as plt
from matplotlib import animation

import json

N_GAMES = 500
grid_size = 10
num_food = 4

gamma = .99
batch_size = 128
capacity = int(1e5)
lr = 0.01

eps = 0.9
eps_decay = 1 - 1e-3
eps_min = 0.05

flute = Agent(grid_size, num_food, 5, eps, eps_decay, eps_min, gamma, batch_size, capacity, lr)

game_rewards = []
game_scores = []

total_steps = 0
for game in range(N_GAMES):

    terminal = False
    frames = []
    snake = Snake(grid_size, num_food)
    state = snake.initial_state()

    steps = 0
    total_reward = 0
    total_score = 0
    while not terminal:

        frame = snake.display_grid(snake.build_snake(), show = False)
        frames.append(frame)

        action = flute.select_action(state)
        next_state, reward, score, terminal = snake.step(action)
        flute.remember(state, action, reward, next_state, terminal)

        error = flute.learn()

        state = next_state
        steps += 1
        total_reward += reward
        total_score += score

        if terminal:
            frame = snake.display_grid(snake.build_snake(), show=False)
            frames.append(frame)

        print(f"\tGame {game + 1} | " \
              f"Steps {steps} | " \
                f"Reward {reward} | " \
                f"Learning error {error:.3f} | " \
                f"Epsilon {flute.eps:.3f}")

    total_steps += steps

    if game == 0:
        best_score = total_score
        best_game = game
    elif total_score >= best_score:
        best_score = total_score
        best_game = game
    print(f"\nGame {game + 1} | Best game {best_game} | Total steps {total_steps} | Total reward {total_reward:.3f} | Total score {total_score} | Learn count {flute.learn_count}\n\n")

    game_rewards.append(total_reward)
    game_scores.append(total_score)

    fig, ax = plt.subplots()
    ax.axis("off")

    im = ax.imshow(frames[0])

    def update(frame_idx, im, frames):
        im.set_data(frames[frame_idx])
        return im

    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames), 
        fargs=(im, frames),
        interval=33, 
        blit=False
    )

    video_name = f"clips//gameplay-{game}.gif"
    fps = 15

    writer = animation.PillowWriter(fps=fps)
    ani.save(video_name, writer=writer)

    plt.close(fig)

with open("agent-results.json", "w") as f:

    data = {"rewards": game_rewards, 
            "scores": game_scores}

    json.dump(data, f)