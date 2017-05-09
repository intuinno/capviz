# hello.py

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.widgets import TextInput, Button, Paragraph, Select

# create some widgets
button = Button(label="Say Hi")
input = TextInput(value="Bokeh")
output = Paragraph()
select = Select(title="Salute:", value='Dr.',
                options=['Sir,','Dr.','Mr.','Ms.'])

# add a callback to a widget
def update():
    output.text = "Hello, " + input.value + select.value
button.on_click(update)

#create a layout for everything
layout = column(button, select, input, output)

# add the layout to curdoc
curdoc().add_root(layout) 

