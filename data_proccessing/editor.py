import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
import json
from collections import deque

class HandwritingAnnotator:
    def __init__(self, root, image_path, smooth_window=7, smooth_method='savitzky_golay',
                 min_component_size=20, show_progress=True):
        self.root = root
        self.root.title("Аннотатор почерка (сглаженный скелет)")

        self.smooth_window = smooth_window
        self.smooth_method = smooth_method
        self.min_component_size = min_component_size
        self.show_progress = show_progress

        # Загружаем и бинаризуем изображение
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить {image_path}")
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

        # Улучшенная бинаризация (адаптивная, чтобы убрать шум)
        # Если изображение уже почти бинарное, можно оставить порог 127,
        # но для надёжности используем адаптивный порог
        binary = cv2.adaptiveThreshold(self.img_gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 2)
        # Удаляем мелкие шумовые компоненты (по площади)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_component_size:
                cleaned[labels == i] = 255
        binary = cleaned

        # Скелет
        raw_skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        # Сглаживание скелета (с прогресс-баром)
        self.skeleton_points, self.skeleton_image = self.smooth_skeleton(
            raw_skeleton, window=self.smooth_window, method=self.smooth_method
        )

        # KDTree
        if len(self.skeleton_points) > 0:
            self.kdtree = cKDTree(self.skeleton_points)
        else:
            self.kdtree = None
            print("Предупреждение: скелет пуст")

        # Отображение (скелет красным)
        display = self.img_bgr.copy()
        for (x, y) in self.skeleton_points:
            cv2.circle(display, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
        self.display_img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(self.display_img)
        self.tk_img = ImageTk.PhotoImage(self.pil_img)

        # Canvas с прокруткой
        self.canvas = tk.Canvas(root, scrollregion=(0, 0, self.pil_img.width, self.pil_img.height))
        self.hbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Данные для рисования
        self.points = []
        self.current_stroke = []
        self.drawing = False
        self.last_added_point = None

        # Привязка событий
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.root.bind("<s>", self.save)
        self.root.bind("<c>", self.clear)

    # ---------- Функции сглаживания скелета ----------
    def get_neighbors(self, y, x, mask):
        h, w = mask.shape
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 255:
                    neighbors.append((ny, nx))
        return neighbors

    def order_curve_points(self, component_mask):
        pts = np.column_stack(np.where(component_mask == 255))  # (y, x)
        if len(pts) == 0:
            return []
        pt_set = {tuple(pt): i for i, pt in enumerate(pts)}
        degree = {}
        for (y, x) in pt_set:
            neighbors = self.get_neighbors(y, x, component_mask)
            degree[(y, x)] = len(neighbors)

        endpoints = [p for p in pt_set if degree[p] == 1]
        if len(endpoints) == 0:
            start = tuple(pts[0])
        else:
            start = endpoints[0]

        ordered = []
        visited = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            y, x = cur
            ordered.append((x, y))
            neighbors = self.get_neighbors(y, x, component_mask)
            for ny, nx in neighbors:
                if (ny, nx) not in visited:
                    stack.append((ny, nx))
        return ordered

    def smooth_skeleton(self, skeleton_img, window=7, method='savitzky_golay'):
        num_labels, labels = cv2.connectedComponents(skeleton_img, connectivity=8)
        total = num_labels - 1
        if self.show_progress:
            # Создаём окно прогресса
            progress_win = tk.Toplevel(self.root)
            progress_win.title("Обработка скелета")
            progress_win.geometry("300x80")
            label = tk.Label(progress_win, text="Сглаживание компонент...")
            label.pack(pady=5)
            progress_bar = ttk.Progressbar(progress_win, length=250, mode='determinate', maximum=total)
            progress_bar.pack(pady=5)
            progress_win.update()

        all_smoothed_points = []
        smooth_img = np.zeros_like(skeleton_img, dtype=np.uint8)

        for label_id in range(1, num_labels):
            component_mask = (labels == label_id).astype(np.uint8) * 255
            # Пропускаем слишком маленькие компоненты (шум)
            if np.sum(component_mask == 255) < self.min_component_size:
                if self.show_progress:
                    progress_bar['value'] = label_id
                    progress_win.update()
                continue

            ordered_pts = self.order_curve_points(component_mask)
            if len(ordered_pts) < window:
                all_smoothed_points.extend(ordered_pts)
                for (x, y) in ordered_pts:
                    cv2.circle(smooth_img, (int(round(x)), int(round(y))), 1, 255, -1)
                if self.show_progress:
                    progress_bar['value'] = label_id
                    progress_win.update()
                continue

            pts_arr = np.array(ordered_pts, dtype=np.float32)
            if method == 'savitzky_golay':
                if window % 2 == 0:
                    window += 1
                smoothed_x = savgol_filter(pts_arr[:, 0], window_length=window, polyorder=2)
                smoothed_y = savgol_filter(pts_arr[:, 1], window_length=window, polyorder=2)
            else:
                kernel = np.ones(window) / window
                smoothed_x = np.convolve(pts_arr[:, 0], kernel, mode='same')
                smoothed_y = np.convolve(pts_arr[:, 1], kernel, mode='same')
            smoothed_pts = np.column_stack((smoothed_x, smoothed_y))
            all_smoothed_points.extend(smoothed_pts.tolist())
            for (x, y) in smoothed_pts:
                cv2.circle(smooth_img, (int(round(x)), int(round(y))), 1, 255, -1)

            if self.show_progress:
                progress_bar['value'] = label_id
                progress_win.update()

        if self.show_progress:
            progress_win.destroy()
        return np.array(all_smoothed_points, dtype=np.float32), smooth_img

    # ---------- Методы редактора ----------
    def find_nearest_skeleton_point(self, x, y, max_dist=15):
        if self.kdtree is None:
            return None
        dist, idx = self.kdtree.query([x, y])
        if dist <= max_dist:
            return self.skeleton_points[idx]
        return None

    def start_draw(self, event):
        self.drawing = True
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        pt = self.find_nearest_skeleton_point(x, y)
        if pt is not None:
            sx, sy = pt
            self.current_stroke = [(sx, sy)]
            self.last_added_point = (sx, sy)
            self.draw_point(sx, sy, "red")
        else:
            self.current_stroke = []
            self.last_added_point = None

    def draw(self, event):
        if not self.drawing:
            return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        pt = self.find_nearest_skeleton_point(x, y, max_dist=20)
        if pt is None:
            return
        sx, sy = pt
        if self.last_added_point is not None:
            dist = np.hypot(sx - self.last_added_point[0], sy - self.last_added_point[1])
            if dist < 2.0:
                return
        self.current_stroke.append((sx, sy))
        self.last_added_point = (sx, sy)
        if len(self.current_stroke) >= 2:
            prev = self.current_stroke[-2]
            self.canvas.create_line(prev[0], prev[1], sx, sy, fill="orange", width=2)
        self.draw_point(sx, sy, "orange")

    def stop_draw(self, event):
        self.drawing = False
        if len(self.current_stroke) < 2:
            self.current_stroke = []
            self.last_added_point = None
            return

        for (x, y) in self.current_stroke:
            self.points.append([x, y, 0])
            self.draw_point(x, y, "red")
        last_x, last_y = self.current_stroke[-1]
        self.points.append([last_x, last_y, 1])
        self.draw_point(last_x, last_y, "blue")

        if len(self.current_stroke) >= 2:
            for i in range(1, len(self.current_stroke)):
                x1, y1 = self.current_stroke[i-1]
                x2, y2 = self.current_stroke[i]
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)

        self.current_stroke = []
        self.last_added_point = None

    def draw_point(self, x, y, color):
        r = 3
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=color)

    def save(self, event):
        if not self.points:
            print("Нет данных")
            return
        with open("trajectory.json", "w") as f:
            json.dump(self.points, f)
        print(f"Сохранено {len(self.points)} точек")

    def clear(self, event):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.points = []
        self.current_stroke = []
        self.last_added_point = None
        print("Очищено")

if __name__ == "__main__":
    root = tk.Tk()
    # Укажите свой файл (можно .png или .jpeg)
    app = HandwritingAnnotator(root, "my_page_0.jpeg",
                               smooth_window=7,
                               smooth_method='savitzky_golay',
                               min_component_size=20,
                               show_progress=True)
    root.mainloop()