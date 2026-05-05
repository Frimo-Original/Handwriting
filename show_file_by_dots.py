import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def main():
    # Загрузка данных
    with open('src/generated_trajectory.json', 'r') as f:
        points = json.load(f)  # каждый элемент: (x, y, state)

    # Состояние для пошагового просмотра
    completed_segments = []   # список завершённых штрихов, каждый: {'xs': list, 'ys': list}
    current_segment = {'xs': [], 'ys': []}  # текущий строящийся штрих
    index = 0                 # индекс следующей точки в points

    # Настройка графика
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.15)   # место под кнопку
    ax.set_title("Пошаговый просмотр точек датасета")
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Информационная строка (вместо заголовка или текста)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    def redraw():
        """Очищает оси и перерисовывает все штрихи: завершённые (чёрные) и текущий (красный)"""
        ax.cla()
        ax.set_aspect('equal')
        ax.invert_yaxis()
        # Рисуем завершённые штрихи
        for seg in completed_segments:
            xs, ys = seg['xs'], seg['ys']
            if len(xs) > 1:
                ax.plot(xs, ys, 'k-', linewidth=1)
            elif len(xs) == 1:
                ax.plot(xs, ys, 'o', markersize=1, color="black")
        # Рисуем текущий (незавершённый) штрих
        xs, ys = current_segment['xs'], current_segment['ys']
        if len(xs) > 1:
            ax.plot(xs, ys, 'r-', linewidth=1)
        elif len(xs) == 1:
            ax.plot(xs, ys, 'ro', markersize=2)   # красная точка побольше, чтобы было заметно
        # Обновляем информационную строку
        info_text.set_text(f"Обработано точек: {index} / {len(points)}")
        # Автомасштабируем оси, чтобы всё поместилось
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def next_point(event):
        nonlocal index, current_segment, completed_segments
        if index >= len(points):
            # Все точки обработаны – можно ничего не делать либо отключить кнопку
            ax.set_title("Все точки просмотрены!")
            fig.canvas.draw_idle()
            return
        x, y, state = points[index]
        index += 1
        if state == 0:
            # Добавляем точку в текущий штрих
            current_segment['xs'].append(x)
            current_segment['ys'].append(y)
        elif state == 1:
            # Завершаем текущий штрих (если в нём что-то есть)
            if current_segment['xs']:
                # Сохраняем копию текущего штриха в завершённые
                completed_segments.append({
                    'xs': current_segment['xs'].copy(),
                    'ys': current_segment['ys'].copy()
                })
                # Очищаем текущий штрих для следующего
                current_segment = {'xs': [], 'ys': []}
        # Перерисовываем график
        redraw()

    # Создаём кнопку
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    button = Button(ax_button, 'Следующая точка')
    button.on_clicked(next_point)

    # Начальная отрисовка (пустой график)
    redraw()
    plt.show()

if __name__ == "__main__":
    main()