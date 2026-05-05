import json
from pathlib import Path

from kivy.app import App
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget


DATASET_DIR = Path(__file__).resolve().parent
JSON_DIR = DATASET_DIR / "jsons"
TEXT_DIR = DATASET_DIR / "texts"


class DrawWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []
        self.strokes = []
        self.current_stroke = None
        self.min_point_distance2 = 25
        self.bind(pos=self.redraw, size=self.redraw)

    def redraw(self, *args):
        self.canvas.clear()
        self._draw_background()
        for stroke in self.strokes:
            self._draw_stroke(stroke)
        if self.current_stroke:
            self._draw_stroke(self.current_stroke)

    def _draw_background(self):
        with self.canvas:
            Color(0, 0, 0, 1)
            Rectangle(pos=self.pos, size=self.size)
            Color(0.5, 0.5, 0.5, 0.45)
            left, bottom = self.pos
            right = left + self.width
            top = bottom + self.height
            x = left + 50
            while x < right:
                Line(points=[x, bottom, x, top], width=1)
                x += 50
            y = bottom + 50
            while y < top:
                Line(points=[left, y, right, y], width=1)
                y += 50

    def _draw_stroke(self, stroke):
        draw_points = [(p[0], p[1]) for p in stroke if p[2] == 0]
        with self.canvas:
            Color(1, 0, 0, 1)
            if len(draw_points) >= 2:
                flat = []
                for x, y in draw_points:
                    flat.extend([x, y])
                Line(points=flat, width=2)
            elif len(draw_points) == 1:
                x, y = draw_points[0]
                Ellipse(pos=(x - 3, y - 3), size=(1, 1))

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        touch.grab(self)
        x, y = touch.pos
        self.current_stroke = [[float(x), float(y), 0]]
        touch.ud["last_pos"] = (x, y)
        touch.ud["moved"] = False
        self.redraw()
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self or self.current_stroke is None:
            return False
        x, y = touch.pos
        last_x, last_y = touch.ud["last_pos"]
        distance2 = (x - last_x) ** 2 + (y - last_y) ** 2
        if distance2 < self.min_point_distance2:
            return True
        self.current_stroke.append([float(x), float(y), 0])
        touch.ud["last_pos"] = (x, y)
        touch.ud["moved"] = True
        self.redraw()
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self or self.current_stroke is None:
            return False
        touch.ungrab(self)
        x, y = touch.pos

        if not touch.ud.get("moved", False):
            self.current_stroke = [[float(x), float(y), 0]]
        else:
            last_x, last_y, _ = self.current_stroke[-1]
            if (x - last_x) ** 2 + (y - last_y) ** 2 >= self.min_point_distance2:
                self.current_stroke.append([float(x), float(y), 0])

        self.current_stroke.append([float(x), float(y), 1])
        self.strokes.append(self.current_stroke)
        self.points.extend(self.current_stroke)
        self.current_stroke = None
        self.redraw()
        return True

    def undo_last_stroke(self):
        if not self.strokes:
            return False
        stroke = self.strokes.pop()
        del self.points[-len(stroke) :]
        self.redraw()
        return True

    def clear_all(self):
        self.points.clear()
        self.strokes.clear()
        self.current_stroke = None
        self.redraw()


class HandwritingRoot(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        top = BoxLayout(size_hint=(1, 0.1), spacing=8, padding=6)
        self.text_input = TextInput(hint_text="Текст для этого образца", multiline=False)
        top.add_widget(self.text_input)

        save_btn = Button(text="Save")
        save_btn.bind(on_press=self.save_data)
        top.add_widget(save_btn)

        undo_btn = Button(text="Undo")
        undo_btn.bind(on_press=self.undo_last_stroke)
        top.add_widget(undo_btn)

        clear_btn = Button(text="Clear")
        clear_btn.bind(on_press=self.clear_all)
        top.add_widget(clear_btn)

        build_btn = Button(text="Build NPZ")
        build_btn.bind(on_press=self.build_dataset)
        top.add_widget(build_btn)

        self.add_widget(top)
        self.draw_widget = DrawWidget()
        self.add_widget(self.draw_widget)
        self.status = Label(size_hint=(1, 0.06), text="Ready")
        self.add_widget(self.status)

    def _next_index(self):
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        TEXT_DIR.mkdir(parents=True, exist_ok=True)
        idx = 1
        while (JSON_DIR / f"trajectory_{idx}.json").exists() or (
            TEXT_DIR / f"trajectory_{idx}.txt"
        ).exists():
            idx += 1
        return idx

    def save_data(self, instance):
        text = self.text_input.text.strip()
        if not self.draw_widget.points:
            self.status.text = "Нет точек для сохранения"
            return
        if not text:
            self.status.text = "Введите текст образца перед сохранением"
            return

        idx = self._next_index()
        json_path = JSON_DIR / f"trajectory_{idx}.json"
        text_path = TEXT_DIR / f"trajectory_{idx}.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.draw_widget.points, f, ensure_ascii=False)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        saved_points = len(self.draw_widget.points)
        self.draw_widget.clear_all()
        self.text_input.text = ""
        self.status.text = f"Saved trajectory_{idx}: {saved_points} points"

    def undo_last_stroke(self, instance):
        if self.draw_widget.undo_last_stroke():
            self.status.text = "Последний штрих удален"
        else:
            self.status.text = "Нет штрихов для отмены"

    def clear_all(self, instance):
        self.draw_widget.clear_all()
        self.status.text = "Очищено"

    def build_dataset(self, instance):
        try:
            from converter import main as build_npz

            build_npz()
            self.status.text = "Dataset rebuilt: all_trajectories.npz"
        except Exception as exc:
            self.status.text = f"Build failed: {exc}"


class TabletEditorApp(App):
    def build(self):
        self.title = "Handwriting Tablet Dataset Editor"
        return HandwritingRoot()


if __name__ == "__main__":
    TabletEditorApp().run()
