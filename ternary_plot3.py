import plotly.graph_objects as go
import numpy as np
import PySimpleGUI as sg

files = sg.popup_get_file('Select spectra for plotting',
                no_window=True, multiple_files=True,
                file_types=(('Logfiles', '*.log'), ('ALL Files', '*.*'),), )
mgiarr = []
feiarr = []
naiarr = []
labelarr = []
for file in files:
    with open(file) as f:
        while True:
            line = f.readline()
            x = line.find(' fei: ')
            if x != -1:
                print(line[x: x + 42])
                y = line.find(' mgi: ')
                z = line.find(' nai: ')
                label = line[z + 5:].split(',')[1]
                feiarr.append(line[x + 5:].split(',')[0])
                mgiarr.append(line[y + 5:].split(',')[0])
                naiarr.append(line[z + 5:].split(',')[0])
                labelarr.append(label)
                print(label)
            if len(line) == 0:
                break


def makeAxis(title, tickangle):
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 5,
      'showline': True,
      'showgrid': True
    }

fig = go.Figure(go.Scatterternary({
    'mode': 'markers',
    'a': feiarr,
    'b': mgiarr,
    'c': naiarr,
    'text': labelarr,
    'marker': {
        'symbol': 'triangle-up-open',
        'color': '#FF0000',
        'size': 10,
        'line': { 'width': 2 }
    }
}))

fig.update_layout({
    'ternary': {
        'sum': 100,
        'aaxis': makeAxis('Fe I', 0),
        'baxis': makeAxis('<br>Mg I', 60),
        'caxis': makeAxis('<br>Na I', -60)
    },
    'annotations': [{
      'showarrow': False,
      'text': 'Simple Ternary Plot with Markers',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
})

fig.show()