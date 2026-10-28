from manim import *

class Exemplo(Scene):
    def construct(self):
        titulo = Text("Testando", font_size=48)
        titulo.to_edge(LEFT)
        self.play(Write(titulo))

