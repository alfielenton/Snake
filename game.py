import numpy as np
from matplotlib import pyplot as plt
import cv2

class Grid:

    def __init__(self, size, num_foods):

        self.size = size
        self.grid = np.zeros((size, size))
        self.color_map = {0 : [0, 0, 255], 
                          -1 : [255, 0, 0], 
                          1 : [0, 255, 0]}

        self.num_foods = num_foods
        self.foods = []

    def build_grid(self, snake):

        self.grid = np.zeros((self.size, self.size))
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[0]):

                if (i, j) in snake:
                    self.grid[i, j] = 1
                elif (i, j) in self.foods:
                    self.grid[i, j] = -1

        return self.grid

    def place_food(self, snake):

        if len(self.foods) < self.num_foods:
            curr_num_foods = len(self.foods)
            while curr_num_foods < self.num_foods:

                food_placed = False
                while not food_placed:

                    food = tuple(np.random.randint(0, self.grid.shape[0], 2).tolist())
                    if food not in snake and food not in self.foods:
                        curr_num_foods += 1
                        self.foods.append(food)
                        food_placed = True
                    else:
                        pass

    def display_grid(self, snake):

        self.build_grid(snake)
        h, w = self.grid.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        for num, rgb in self.color_map.items():
            rgb_image[self.grid == num] = rgb

        plt.imshow(rgb_image)
        plt.axis('on')
        plt.show()
        return rgb_image

class Snake(Grid):

    def __init__(self, size, num_foods, init_snake_size = 4):

        super().__init__(size, num_foods)

        self.actions = ['forward', 'back', 'left', 'right', None]
        self.snake = {"head" : {"position" : np.array([0, init_snake_size - 1]), "velocity" : np.array([0, 1])}, 
                      "corners" : [], 
                      "length" : init_snake_size}

        self.turn_left = np.array([[0, -1],
                                   [1, 0]])
        self.turn_right = np.array([[0, 1], 
                                    [-1, 0]])

        self.place_food(self.build_snake())

    def build_snake(self):

        snake = [tuple(self.snake["head"]["position"].tolist())]
        body_direction = self.snake["head"]["velocity"]
        s = np.abs(body_direction[body_direction != 0])
        body_direction = -body_direction / s

        corner_reached = False
        for _ in range(self.snake["length"] - 1):

            curr_snake_pos = snake[-1]

            if corner_reached:
                if turn_direction == "right":
                    body_direction = self.turn_left @ body_direction
                else:
                    body_direction = self.turn_right @ body_direction

            new_snake_pos = curr_snake_pos + body_direction
            corners_taken = np.zeros(len(self.snake["corners"])).astype(np.bool).tolist()
            for corner, turn in self.snake["corners"]:

                if corner == tuple(new_snake_pos.tolist()):
                    corner_reached = True
                    turn_direction = turn
                    corners_taken[self.snake["corners"].index((corner, turn))] = True
                    break
                else:
                    corner_reached = False
                    turn_direction = None

            snake.append(tuple(new_snake_pos.astype('i').tolist()))

        return snake

    def step(self, action):

        action = self.actions[int(action)]
        curr_snake_vel = self.snake["head"]["velocity"]

        if action == "forward":
            s = np.abs(curr_snake_vel[curr_snake_vel != 0])[0]
            self.snake["head"]["velocity"] = self.snake["head"]["velocity"] + curr_snake_vel / s
            self.snake["head"]["velocity"] = self.snake["head"]["velocity"].astype("i")

        elif action == "back":
            s = np.abs(curr_snake_vel[curr_snake_vel != 0])[0]

            if not (curr_snake_vel == 0).all():
                curr_snake_vel = curr_snake_vel - curr_snake_vel / s

            self.snake["head"]["velocity"] = curr_snake_vel
            self.snake["head"]["velocity"] = self.snake["head"]["velocity"].astype("i")

        elif action == "left":
            self.snake["head"]["velocity"] = self.turn_left @ curr_snake_vel
            self.snake["corners"].append((tuple(self.snake["head"]["position"].tolist()), "left"))

        elif action == "right":
            self.snake["head"]["velocity"] = self.turn_right @ curr_snake_vel
            self.snake["corners"].append((tuple(self.snake["head"]["position"].tolist()), "right"))

        else:
            pass

        curr_snake_vel = self.snake["head"]["velocity"]

        s = np.abs(curr_snake_vel[curr_snake_vel != 0])[0]
        orig_head = self.snake["head"]["position"]
        curr_head = orig_head

        reward = 0
        for _ in range(int(s)):

            terminal = False
            curr_head = curr_head + curr_snake_vel / s
            if tuple(curr_head.tolist()) in self.foods:

                self.foods.remove(tuple(curr_head.tolist()))
                self.snake["length"] += 1
                reward += 10

            self.snake["head"]["position"] = curr_head
            curr_snake = self.build_snake()
            self.snake["head"]["position"] = orig_head
            if len(set(curr_snake)) < len(curr_snake):
                terminal = True
                reward -= 10
                break

        if not terminal:
            self.snake["head"]["position"] += self.snake["head"]["velocity"]

            if (self.snake["head"]["position"] < 0).any() or (self.snake["head"]["position"] > self.size - 1).any():
                reward -= 10
                terminal = True
            else:
                terminal = False

        reward -= .1
        snake = self.build_snake()
        head_vel = self.snake["head"]["velocity"].tolist()
        self.place_food(snake)

        state = (snake, head_vel, self.foods)
        return state, reward, terminal


def record_video(frames, video_filename, frame_size, fps=30):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(video_filename, fourcc, fps, (frame_size, frame_size))
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_output.write(bgr_frame)

    video_output.release()

snake = Snake(10, 5)
s = 0
terminal = False
frames = []

while not terminal:

    s += 1
    frame = snake.display_grid(snake.build_snake())
    frames.append(frame)

    print("\nAvailable actions:")
    for i, action in enumerate(snake.actions):
        print(f"{i}: {action}")
    print("5: Terminate game")

    action = input("\nChoose an action: ")
    if int(action) == 5:
        break
    state, reward, terminal = snake.step(action)
    snake_body, head_vel, foods = state

    print("State: ")
    print("\n\tSnake: ")
    for body in snake_body:
        print(f"\t{body}")
    print("\n\n\tFood:")
    for food in foods:
        print(f"\t{food}")
    print(f"\n\tHead velocity: {head_vel}")
    print(f"\nReward: {reward}")
    print(f"Terminal: {terminal}")
    print(f"Steps taken: {s + 1}")

    if terminal:
        print("Game finished")

record_video(frames, "game_recording.mp4", snake.size)