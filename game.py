import numpy as np
from matplotlib import pyplot as plt

class Grid:

    def __init__(self, size, num_foods, init_snake_size):

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
                    if food not in snake:
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
        plt.axis('off')
        plt.show()

class Snake(Grid):

    def __init__(self, size, num_foods, init_snake_size = 4):

        super().__init__(size, num_foods, init_snake_size)

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
            for corner, turn in self.snake["corners"]:
                if corner == new_snake_pos:
                    corner_reached = True
                    turn_direction = turn
                    break
                else:
                    corner_reached = False
                    turn_direction = None

            snake.append(tuple(new_snake_pos.astype('i').tolist()))

        return snake

    def step(self, action):

        action = self.actions[action]
        curr_snake_vel = self.snake["head"]["velocity"]

        if action == "forward":
            s = np.abs(curr_snake_vel[curr_snake_vel != 0])
            self.snake["head"]["velocity"] += curr_snake_vel + curr_snake_vel / s

        elif action == "back":
            s = np.abs(curr_snake_vel[curr_snake_vel != 0])
            curr_snake_vel -= curr_snake_vel / s
            self.snake["head"]["velocity"] = np.clip(curr_snake_vel, 0, None)

        elif action == "left":
            self.snake["head"]["velocity"] = self.turn_left @ curr_snake_vel
            self.snake["corners"].append((self.snake["head"]["position"], "left"))

        elif action == "right":
            self.snake["head"]["velocity"] = self.turn_right @ curr_snake_vel
            self.snake["corners"].append((self.snake["head"]["position"], "right"))

        else:
            pass

        curr_snake_vel = self.snake["head"]["velocity"]
        s = np.abs(curr_snake_vel[curr_snake_vel != 0])
        curr_head = self.snake["head"]["position"]

        reward = 0
        for _ in range(int(s)):

            curr_head += curr_snake_vel / s
            if curr_head in self.foods:

                self.foods.pop(curr_head)
                reward += 10

        self.snake["head"]["position"] += self.snake["head"]["velocity"]
        if self.snake["head"]["position"] != np.clip(self.snake["head"]["position"], 0, self.grid.shape[0]):
            reward -= 10
            terminal = True
        else:
            terminal = False

        reward -= .1
        snake = self.build_snake()
        self.place_food(snake)

        state = (snake, self.foods, terminal)
        return state, reward

snake = Snake(10, 5)
STEPS = 100

for s in range(STEPS):
    snake.display_grid(snake.build_snake())