import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os, sys
sys.path.append(os.environ['PWD'])
from commune.gradio.module import GradioClient


class HelloWorld_2_0:

    @register(inputs=["text", "text", gr.Radio(["morning", "evening", "night"])], outputs="text")
    def Hello(self, Lname : str, Fname : str, day : 'list[any]'=["morning", "evening", "night"]) -> str:
        return "Hello, {} {}".format(Fname, Lname)  

    @register(inputs=["text", "text"], outputs="text")
    def goodbye(self, Fname : str, Lname : str) -> str:
        return "Goodbye, {} {}".format(Fname, Lname)  
    
    @register(inputs=["text", gr.Checkbox() , gr.Slider(0, 60)], outputs=["text", "number"])
    def greet(self, name, is_morning, temperature):
        salutation = "Good morning" if is_morning else "Good evening"
        greeting = "%s %s. It is %s degrees today" % (salutation, name, temperature)
        celsius = (temperature - 32) * 5 / 9
        return (greeting, round(celsius, 2))



@gradio_compile
class FSD:

    def get_new_val(self, old_val, nc):
        return np.round(old_val * (nc - 1)) / (nc - 1)


    def palette_reduce(self, img : PIL.Image.Image, nc : 'tuple[float, float, float]'=(0.0000, 0, 16)):
        pixels = np.array(img, dtype=float) / 255
        pixels = self.get_new_val(pixels, nc)

        carr = np.array(pixels / np.max(pixels) * 255, dtype=np.uint8)
        return PIL.Image.fromarray(carr)

    @register(inputs=[gr.Image(), gr.Slider(0.00, 16)], outputs=gr.Gallery())
    def Floyd_Steinberg_dithering(self, img, nc : 'tuple[float, float, float]'=(0.0000, 0, 16) ) -> 'list[PIL.Image.Image]':
        pixels = np.array(img, dtype=float) / 255
        new_height, new_width, _ = img.shape 
        for row in range(new_height):
            for col in range(new_width):
                old_val = pixels[row, col].copy()
                new_val = self.get_new_val(old_val, nc)
                pixels[row, col] = new_val
                err = old_val - new_val
                if col < new_width - 1:
                    pixels[row, col + 1] += err * 7 / 16
                if row < new_height - 1:
                    if col > 0:
                        pixels[row + 1, col - 1] += err * 3 / 16
                    pixels[row + 1, col] += err * 5 / 16
                    if col < new_width - 1:
                        pixels[row + 1, col + 1] += err * 1 / 16
        carr = np.array(pixels / np.max(pixels, axis=(0, 1)) * 255, dtype=np.uint8)
        return [PIL.Image.fromarray(carr), self.palette_reduce(img, nc) ]



@gradio_compile
class C:

    def Hello(self):
        return "Hello"
    
    @register(inputs="text", outputs="text")
    def Greeting(self, name):
        return self.Hello() + " " + name

@gradio_compile
class stock_forecast:
    
    def __init__(self):
        matplotlib.use('Agg')

    @register(inputs=[gr.Checkbox(label="legend"), gr.Radio([2025, 2030, 2035, 2040], label="projct"), gr.CheckboxGroup(["Google", "Microsoft", "Gradio"], label="company"), gr.Slider(label="noise"), gr.Radio(["cross", "line", "circle"], label="style")], outputs=[gr.Plot()])
    def plot_forcast(self, legend, project, companies , noise , styles)-> matplotlib.figure.Figure:
        start_year = 2022
        x = np.arange(start_year, project + 1)
        year_count = x.shape[0]
        plt_format = ({"cross": "X", "line": "-", "circle": "o--"})[styles]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, company in enumerate(companies):
            series = np.arange(0, year_count, dtype=float)
            series = series**2 * (i + 1)
            series += np.random.rand(year_count) * noise
            ax.plot(x, series, plt_format)
        if legend:
            plt.legend(companies)
        print(type(fig))
        return fig 
