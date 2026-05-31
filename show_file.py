import json
import matplotlib.pyplot as plt

with open('dataset/jsons/trajectory_351.json', 'r') as f:
    points = json.load(f)
# with open('src/generated_trajectory.json', 'r') as f:
#     points = json.load(f)

xs, ys = [], []
for x, y, state in points:
    if state == 0:
        xs.append(x)
        ys.append(y)
    elif state == 1:
        # Отрисовываем накопленный штрих
        if len(xs) == 1:
            # Одиночная точка – рисуем маркером
            plt.plot(xs, ys, 'o', markersize=1, color="black")
        elif len(xs) > 1:
            # Линия
            plt.plot(xs, ys, 'k-', linewidth=1)
        xs, ys = [], []

plt.gca().invert_yaxis()  # так как y растёт вниз на экране
plt.axis('equal')
plt.show()