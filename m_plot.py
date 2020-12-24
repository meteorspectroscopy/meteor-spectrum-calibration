# -------------------------------------------------------------------
# m_plot functions for m_spec
# Author: Martin Dubs, 2020
# -------------------------------------------------------------------
import logging
import os.path as path
import time

import PySimpleGUI as sg
import numpy as np
from PIL import ImageGrab

import m_specfun as m_fun

version = '0.9.21'


def graph_calibrated_spectrum(llist, lmin=0, lmax=720, imin=0, imax=1, autoscale=True, gridlines=True,
                              canvas_size=(800, 400), plot_title='Spectrum', multi_plot=False, offset=0):
    """
    displays calibrated spectrum llist in separate window
    allows change of intensity scale and saving of resulting plot
    if no filename is given, result is saved as llist + '_plot.png'
    :param llist: filename of calibrated spectrum with extension .dat
    :param lmin, lmax: wavelength range, can be inverse
    :param imin, imax: intensity range
    :param autoscale: if True, range is determined automatically
    :param gridlines: if True, grid lines are shown
    :param canvas_size: size of image
    :param plot_title: title displayed at the top
    :param multi_plot: if True, multiple spectra can be selected and displayed
    :param offset: spacing between spectra in plot
    :return: p, imin, imax, caltext
    """

    # --------------------------------------------------------------
    def draw_spectrum(lcal, ical, lmin, lmax, color='blue'):
        for l0 in range(0, len(lcal)):
            if (lmax > lmin and lmin <= lcal[l0] <= lmax) or (lmax < lmin and lmin >= lcal[l0] >= lmax):
                if l0:
                    graph.DrawLine((lcal[l0 - 1], ical[l0 - 1]), (lcal[l0], ical[l0]), color, 2)

    # --------------------------------------------------------------
    if llist:
        lcal, ical = np.loadtxt(llist, unpack=True, ndmin=2)
    mod_file = ''
    caltext = ''
    x = y = 0
    c_array = ['blue', 'green', 'red', 'black', 'grey', 'brown',
               'blue', 'green', 'red', 'black', 'grey', 'brown']
    if multi_plot:
        index = 0
        l_array = []
        i_array = []
        f_array = []
        spec_list = sg.popup_get_file('Select spectra for plotting',
                          no_window=True, multiple_files=True,
                          file_types=(('Spectra', '*.dat'), ('ALL Files', '*.*'),), )
        if spec_list:
            imin = 0
            imax = 0
            for spec in spec_list:
                lcal, ical = np.loadtxt(spec, unpack=True, ndmin=2)
                ical = ical + index * offset
                imin = min(imin, min(ical))
                imax = max(imax, max(ical))
                l_array.append(lcal)
                i_array.append(ical)
                f_array.append(spec)
                index += 1
            idelta = 0.05 * (imax - imin)
            imin -= idelta
            imax += idelta
    elif autoscale:
        lmin = lcal[0]
        lmax = lcal[len(lcal) - 1]
        imin = min(ical)
        imax = max(ical)
        idelta = 0.05 * (imax - imin)
        imin -= idelta
        imax += idelta
    points = []
    for l0 in range(len(lcal) - 1):
        points.append((lcal[l0], ical[l0]))
    # y coordinate autoscale
    # plotscale pixel/unit
    lscale = canvas_size[0] / (lmax - lmin)
    iscale = canvas_size[1] / (imax - imin)
    # layout with border for scales, legends
    layout = [[sg.Graph(canvas_size=canvas_size,
                        graph_bottom_left=(lmin - 40 / lscale, imin - 40 / iscale),
                        graph_top_right=(lmax + 10 / lscale, imax + 30 / iscale),
                        enable_events=True, float_values=True, background_color='white', key='graph')],
              [sg.Button('Save', key='Save', bind_return_key=True), sg.Button('Close Window', key='Close'),
               sg.Text('Imin:'), sg.InputText('', key='imin', size=(8, 1)),
               sg.Text('Imax:'), sg.InputText('', key='imax', size=(8, 1)),
               sg.Button('Scale I', key='scaleI'), sg.Text('Cursor Position: '),
               sg.InputText('', size=(26, 1), key='cursor', disabled=True),
               sg.Text('Scale Factor'), sg.InputText('1.0', key='factor', size=(8, 1))]]

    right_click_menu = ['unused', ['Multiply spectrum by factor', 'Divide Spectrum by factor',
                                   'Save modified spectrum', 'Normalize to peak value',
                                   'Compare with spectrum', 'Label Peak']]

    window = sg.Window(llist, layout, keep_on_top=True, right_click_menu=right_click_menu).Finalize()
    graph = window['graph']
    label_str, lam_calib = m_fun.create_line_list_combo('m_linelist', window, combo=False)

    # draw x-axis
    if lcal[0]:  # for raw spectrum lcal[0] = 0, otherwise lmin
        x_label = u'\u03BB' + ' [nm]'
    else:
        x_label = 'Pixel'
    # lamda = u'\u03BB'
    graph.DrawText(x_label, ((lmax + lmin) / 2, imin - 30 / iscale), font='Arial 12')
    graph.DrawText(plot_title, ((lmax + lmin) / 2, imax + 15 / iscale), font='Arial 12')
    # calculate spacing
    deltax = round((lmax - lmin) / 250) * 50
    const = 1
    while not deltax:
        const *= 10
        deltax = int(const * (lmax - lmin) / 250) * 50
    deltax /= const
    dmax = int(lmax / deltax) + 1
    dmin = int(lmin / deltax)
    for x in range(dmin, dmax):
        graph.DrawLine((x * deltax, imin - 3 / iscale), (x * deltax, imin))
        if gridlines:
            graph.DrawLine((x * deltax, imin), (x * deltax, imax), 'grey')
        graph.DrawText(x * deltax, (x * deltax, imin - 5 / iscale), text_location=sg.TEXT_LOCATION_TOP, font='Arial 10')

    # draw y-axis
    graph.DrawText('I', (lmin - 30 / lscale, (imin + imax) / 2), font='Arial 12')
    # calculate spacing
    deltay = round((imax - imin) / 5)
    const = 1
    while not deltay:
        const *= 10
        deltay = int(const * (imax - imin) / 5)
    deltay /= const
    dmax = int(imax / deltay) + 1
    dmin = int(imin / deltay)
    for d in range(dmin, dmax):
        graph.DrawLine((lmin - 3 / lscale, d * deltay), (lmin, d * deltay))
        if gridlines:
            graph.DrawLine((lmin, d * deltay), (lmax, d * deltay), 'grey')
        graph.DrawText(d * deltay, (lmin - 5 / lscale, d * deltay), text_location=sg.TEXT_LOCATION_RIGHT,
                       font='Arial 10')

    graph.DrawRectangle((lmin, imin), (lmax, imax), line_width=2)
    # draw graph
    if multi_plot:
        if index:
            for ind in range(index):
                if offset <= 0:
                    pos_y = 25 * (ind + 1)
                else:
                    pos_y = 25 * (index - ind)
                draw_spectrum(l_array[ind], i_array[ind], lmin, lmax, color=c_array[ind])
                graph.DrawText(f_array[ind], (lmax - 20 / lscale, imax - pos_y / iscale),
                   text_location=sg.TEXT_LOCATION_RIGHT, font='Arial 12', color=c_array[ind])
    else:
        draw_spectrum(lcal, ical, lmin, lmax)
    while True:
        event, values = window.read()
        if event in (None, 'Close'):
            window.close()
            return mod_file, imin, imax, caltext

        elif event == 'graph':  # if there's a "Graph" event, then it's a mouse
            x, y = (values['graph'])
            window['cursor'].update(f'Lambda:{x:8.2f}  Int:{y:8.2f}')

        elif event == 'Save':
            window.Minimize()
            p, ext = path.splitext(llist)
            p += '_plot.png'
            filename, info = m_fun.my_get_file(p, save_as=True,
                                               file_types=(('Image Files', '*.png'), ('ALL Files', '*.*')),
                                               title='Save spectrum plot (.PNG)', default_extension='*.png', )
            window.Normal()
            time.sleep(1.0)
            if filename:
                p, ext = path.splitext(filename)
                p += '.png'
            save_element_as_file(window['graph'], p)
            info = f'spectrum {llist} plot saved as {str(p)}'
            logging.info(info)
            caltext += info + '\n'
            window.close()
            return mod_file, imin, imax, caltext
        elif event == 'scaleI':
            try:
                imin = float(values['imin'])
                imax = float(values['imax'])
                iscale = canvas_size[1] / (imax - imin)
                graph.change_coordinates((lmin - 40 / lscale, imin - 40 / iscale),
                                         (lmax + 10 / lscale, imax + 30 / iscale))
                draw_spectrum(lcal, ical, lmin, lmax, color='red')
                graph.update()
            except Exception as e:
                sg.PopupError(f'invalid values for Imin, Imax, try again\n{e}', keep_on_top=True)
        elif event in ('Multiply spectrum by factor', 'Divide Spectrum by factor'):
            try:
                factor = float(values['factor'])
                if event == 'Multiply spectrum by factor':
                    ical = ical * factor
                    info = f'spectrum {llist} multiplied by factor {factor}'
                else:
                    ical = ical / factor
                    info = f'spectrum {llist} divided by factor {factor}'
            except Exception as e:
                sg.PopupError(f'invalid value for Factor, try again\n{e}', keep_on_top=True)
                info = 'invalid factor'
            caltext += info + '\n'
            logging.info(info)
            draw_spectrum(lcal, ical, lmin, lmax, color='red')
            graph.update()
        elif event == 'Save modified spectrum':
            window.Minimize()
            mod_file, info = m_fun.my_get_file(llist, title='Save modified spectrum', save_as=True,
                                file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*'),))
            if mod_file:
                mod_file = m_fun.change_extension(mod_file, '.dat')
                np.savetxt(mod_file, np.transpose([lcal, ical]), fmt='%8.3f %8.5f')
                info = f'modified spectrum {llist} saved as {mod_file}'
                logging.info(info)
                caltext += info + '\n'
            window.Normal()
        elif event == 'Normalize to peak value':
            peak_int = max(ical)
            ical = ical / peak_int
            imin = -.1
            imax = 1.1
            mod_file = m_fun.change_extension(llist, 'N.dat')
            np.savetxt(mod_file, np.transpose([lcal, ical]), fmt='%8.3f %8.5f')
            info = f'spectrum normalized to peak intensity = {peak_int}\n' \
                   f'saved as {mod_file}'
            caltext += info
            logging.info(info)
            draw_spectrum(lcal, ical, lmin, lmax, color='red')
        elif event == 'Compare with spectrum':
            window.Minimize()
            comp_file, info = m_fun.my_get_file(llist, title='Compare with spectrum', save_as=False,
                        file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*'),))
            if comp_file:
                window.Normal()
                caltext += f'File {comp_file} loaded\n'
                lcomp, icomp = np.loadtxt(comp_file, unpack=True, ndmin=2)
                draw_spectrum(lcomp, icomp, lmin, lmax, color='red')
                graph.DrawText(llist, (lmax - 20 / lscale, imax - 15 / iscale),
                               text_location=sg.TEXT_LOCATION_RIGHT,
                               font='Arial 12', color='blue')
                graph.DrawText(comp_file, (lmax - 20 / lscale, imax - 40 / iscale),
                               text_location=sg.TEXT_LOCATION_RIGHT,
                               font='Arial 12', color='red')
        elif event == 'Label Peak':
            layout_label = [[sg.InputText('Cursor', size=(40, 1), key='cursor', disabled=True)],
                            [sg.InputText('', size=(40, 1), key='label')],
                            [sg.Button('Apply'), sg.Button('Cancel')]]
            window_label = sg.Window('Label Peak', layout_label, keep_on_top=True).Finalize()
            for k in range(len(lam_calib)):
                if label_str[k][0] < x:
                    kk = k
            if kk < len(lam_calib):
                if abs(label_str[kk][0] - x) > abs(label_str[kk + 1][0] - x):
                    kk += 1
            window_label['label'].update(lam_calib[kk])
            klam = 0
            for k in range(len(lcal)):
                if lcal[k] < x:
                    klam = k
            i_peak = 0
            for k in range(max(0, klam - 10), min(klam + 10, len(lcal))):
                i_peak = max(i_peak, ical[k])
                lam_peak = label_str[kk][0]
            window_label['cursor'].update(f'Lambda:{lcal[klam]:8.2f}  Peak:{i_peak:8.2f}')
            while True:
                event, values = window_label.read()
                if event in 'Apply':
                    # check if label changed
                    new_label = values['label']
                    if new_label != lam_calib[kk]:
                        x = new_label.lstrip()
                        if len(x.split(' ', 1)) == 2:
                            (lam_peak, name) = x.split(' ', 1)
                        else:
                            window.Minimize()
                            window_label.close()
                            sg.PopupError(' type: wavelength (space) description, two values required ')
                            window.Normal()
                            break
                        lam_peak = float(lam_peak)
                    if y > i_peak:
                        graph.DrawLine((lam_peak, i_peak + 20 / iscale), (lam_peak, y - 20 / iscale), 'black', 2)
                    else:
                        graph.DrawLine((lam_peak, i_peak - 20 / iscale), (lam_peak, y + 20 / iscale), 'black', 2)
                    graph.DrawText(new_label, location=(lam_peak, y),
                                   text_location=sg.TEXT_LOCATION_CENTER,
                                   font='Arial 12', color='black')
                if event in ('Cancel', None):
                    pass
                window_label.close()
                break


# -------------------------------------------------------------------

def save_element_as_file(element, filename):
    """
    Saves any element as an image file.
    Element needs to have an underlying Widget available (almost if not all of them do)
    : param element: The element to save
    : param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif, ?)
    """
    widget = element.Widget
    box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(),
           widget.winfo_rooty() + widget.winfo_height())
    grab = ImageGrab.grab(bbox=box)
    grab.save(filename)


# -------------------------------------------------------------------


def plot_raw_spectrum(rawspec, graph, canvasx):
    """
    plots  a raw (uncalibrated)spectrum for selection of calibration lines
    :param rawspec: filename of uncalibrated spectrum with extension .dat
    :param graph: window to display spectrum
    :param canvasx: width of graph (needed to size points in graph)
    :return:
    lmin, lmax: pixel range
    imin, imax: intensity range
    lcal, ical: pixel, intensity array
    """
    lcal, ical = np.loadtxt(rawspec, unpack=True, ndmin=2)
    lmin = lcal[0]
    lmax = lcal[len(lcal) - 1]
    # y coordinate autoscale
    imin = min(ical)
    imax = max(ical)
    idelta = 0.05 * (imax - imin)
    imin -= idelta
    imax += idelta
    points = []
    for l0 in range(len(lcal)):
        points.append((lcal[l0], ical[l0]))
    graph.change_coordinates((lmin, imin), (lmax, imax))
    # erase graph with rectangle
    graph.DrawRectangle((lmin, imin), (lmax, imax), fill_color='white', line_width=1)
    graph.DrawText(rawspec, (0.5 * (lmax - lmin), imax - 0.05 * (imax - imin)))
    # draw graph
    for l0 in range(0, len(lcal)):
        if lmin <= lcal[l0] < lmax:
            graph.DrawCircle(points[l0], 2 / canvasx, line_color='red', fill_color='red')
            if l0:
                graph.DrawLine(points[l0 - 1], points[l0], 'red', 1)
    return lmin, lmax, imin, imax, lcal, ical

# -------------------------------------------------------------------
