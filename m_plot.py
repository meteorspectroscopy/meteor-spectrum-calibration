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
from scipy import interpolate
from skimage.filters import gaussian
from scipy import optimize

import m_specfun as m_fun

version = '0.9.25'


def graph_calibrated_spectrum(spec_file, line_list, meteor_lines='meteor_lines', lmin=0, lmax=720,
                              imin=0, imax=1, autoscale=True, gridlines=True,
                              canvas_size=(800, 400), plot_title='Spectrum', multi_plot=False, offset=0):
    """
    displays calibrated spectrum spec_file in separate window
    allows change of intensity scale and saving of resulting plot
    if no filename is given, result is saved as spec_file + '_plot.png'
    :param spec_file: filename of calibrated spectrum with extension .dat
    :param line_list: filename of calibration lines with extension .txt
    :param meteor_lines: filename of meteor spectral lines with extension .txt
    :param lmin: wavelength range, can be inverse
    :param lmax: wavelength range, can be inverse
    :param imin: intensity range
    :param imax: intensity range
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
        id_list = []
        for l0 in range(0, len(lcal)):
            if (lmax > lmin and lmin <= lcal[l0] <= lmax) or (lmax < lmin and lmin >= lcal[l0] >= lmax):
                if l0:
                    idg = graph.DrawLine((lcal[l0 - 1], ical[l0 - 1]), (lcal[l0], ical[l0]), color, 2)
                    id_list.append(idg)
        return id_list

    # --------------------------------------------------------------
    def gauss(x, *p):
        a, mu, sigma = p
        return a * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    # --------------------------------------------------------------
    def linewidth_results(start, end, coeff, fwg, result):
        result += '\n'
        layout = [[sg.Text('measured line parameters')],
                  [sg.Multiline(result, size=(45, 10), k='log', autoscroll=True)],
                  [sg.B('OK', bind_return_key=True, tooltip='log results and go to next line'),
                   sg.B('Retry'), sg.B('Exit', button_color=('white', 'red'), tooltip='finish linewidth tool')]]
        window_lw = sg.Window('Linewidth results', layout, keep_on_top=True).Finalize()
        tool_enabled = True
        info_base = f'baseline ({start[0]:8.1f}, {start[1]:8.3f}) to ({end[0]:8.1f}, {end[1]:8.3f})'
        info_coeff = f'c = [{coeff[0]:8.3f}, {coeff[1]:8.2f}, {coeff[2]:8.3f}]'
        info_line = f'Linewidth:{fwg:6.2f}  lambda:{coeff[1]:8.2f}'
        result = result + info_base + '\n' + info_coeff + '\n' + info_line + '\n'
        window_lw['log'].update(result)
        while True:
            ev, vals = window_lw.read()
            if ev in (None, 'Exit'):
                tool_enabled = False
                result = ''
                break
            if ev in ('OK', 'Retry'):
                if ev == 'OK':
                    logging.info(info_base)
                    logging.info(info_coeff)
                    logging.info(info_line)
                break
        window_lw.close()
        return tool_enabled, result

    lcal = []
    mod_file = ''
    caltext = ''
    text_cursor_position = 'Cursor Position:'
    x = y = 0
    c_array = ['blue', 'green', 'red', 'black', 'grey', 'brown',
               'blue', 'green', 'red', 'black', 'grey', 'brown']
    id_list = []
    id_list_comp = []
    id_label = []
    id_line = []
    if spec_file:
        lcal, ical = np.loadtxt(spec_file, unpack=True, ndmin=2)
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
    if lcal != []:
        plot_pixel_increment = int(abs((lcal[1] - lcal[0]) / lscale) + 1)
    else:
        return mod_file, imin, imax, caltext
    # layout with border for scales, legends
    right_click_menu = ['&Tools', ['Plot Tools', '---', '&Multiply spectrum by factor',
                                   '&Divide Spectrum by factor', '&Save modified spectrum',
                                   '&Normalize to peak value', 'Clip wavelength range',
                                   '&Compare with spectrum', '&Label calibration lines',
                                   'Label meteor lines', '&Remove label', 'Line&width tool']]
    layout = [[sg.Menu([right_click_menu])],[sg.Graph(canvas_size=canvas_size, drag_submits=True,
                        graph_bottom_left=(lmin - 40 / lscale, imin - 40 / iscale),
                        graph_top_right=(lmax + 10 / lscale, imax + 30 / iscale),
                        enable_events=True, float_values=True, background_color='white', key='graph')],
              [sg.Button('Save', key='Save', bind_return_key=True, tooltip='Save the actual plot'),
               sg.Button('Close Window', key='Close'),
               sg.Text('Imin:'), sg.InputText('', key='imin', size=(8, 1)),
               sg.Text('Imax:'), sg.InputText('', key='imax', size=(8, 1)),
               sg.Button('Scale I', key='scaleI'),
               sg.InputText(text_cursor_position, size=(30, 1), key='cursor', disabled=True),
               sg.Button('FWHM', disabled=True),
               sg.Text('Scale Factor'), sg.InputText('1.0', key='factor', size=(8, 1))]]


    window = sg.Window(spec_file, layout, keep_on_top=True, right_click_menu=right_click_menu).Finalize()
    graph = window['graph']
    linewidth_tool_enabled = False
    label_str, lam_calib = m_fun.create_line_list_combo(line_list, window, combo=False)
    label_str_meteor, lam_meteor = m_fun.create_line_list_combo(meteor_lines, window, combo=False)

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
    try:
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
    except Exception as e:
        info = f'invalid intensities found in {spec_file}'
        sg.PopupError(info + f'\n{e}', keep_on_top=True)
        logging.error('Error, ' + info)
        logging.error(e)
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
        id_list = draw_spectrum(lcal, ical, lmin, lmax)
    while True:
        event, values = window.read()
        if event in (None, 'Close'):
            window.close()
            return mod_file, imin, imax, caltext

        elif event == 'graph' and not linewidth_tool_enabled:
            x, y = (values['graph'])
            window['cursor'].update(f'Cursor Lambda:{x:8.2f}  Int:{y:8.2f}')

        elif event == 'Save':
            window.Minimize()
            p, ext = path.splitext(spec_file)
            p += '_plot.png'
            filename, info = m_fun.my_get_file(p, save_as=True,
                                               file_types=(('Image Files', '*.png'), ('ALL Files', '*.*')),
                                               title='Save spectrum plot (.PNG)', default_extension='*.png', )
            window.Normal()
            window.refresh()
            time.sleep(1.0)
            if filename:
                p, ext = path.splitext(filename)
                p += '.png'
            save_element_as_file(window['graph'], p)
            info = f'spectrum {spec_file} plot saved as {str(p)}'
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
                delete_curve(id_list, graph)
                id_list = draw_spectrum(lcal, ical, lmin, lmax, color='red')
                graph.update()
            except Exception as e:
                sg.PopupError(f'invalid values for Imin, Imax, try again\n{e}',
                              title='Input Error', keep_on_top=True)
        elif event in ('Multiply spectrum by factor', 'Divide Spectrum by factor'):
            try:
                factor = float(values['factor'])
                if event == 'Multiply spectrum by factor':
                    ical = ical * factor
                    info = f'spectrum {spec_file} multiplied by factor {factor}'
                else:
                    ical = ical / factor
                    info = f'spectrum {spec_file} divided by factor {factor}'
                    if abs(factor) <= 1.e-12:
                        raise Exception('division by zero not allowed')
            except Exception as e:
                sg.PopupError(f'invalid value for Factor, try again\n{e}',
                              title='Input Error', keep_on_top=True)
                info = f'invalid arithmetic factor, {e}'
            caltext += info + '\n'
            logging.info(info)
            delete_curve(id_list, graph)
            id_list = draw_spectrum(lcal, ical, lmin, lmax, color='red')
            graph.update()
        elif event == 'Save modified spectrum':
            window.Minimize()
            mod_file, info = m_fun.my_get_file(spec_file, title='Save modified spectrum', save_as=True,
                                               file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*'),))
            if mod_file:
                mod_file = m_fun.change_extension(mod_file, '.dat')
                np.savetxt(mod_file, np.transpose([lcal, ical]), fmt='%8.3f %8.5f')
                info = f'modified spectrum {spec_file} saved as {mod_file}'
                logging.info(info)
                caltext += info + '\n'
                spec_file = mod_file
                window.set_title(spec_file)
            window.Normal()
        elif event == 'Normalize to peak value':
            peak_int = max(ical)
            ical = ical / peak_int
            imin = -.1
            imax = 1.1
            mod_file = m_fun.change_extension(spec_file, 'N.dat')
            np.savetxt(mod_file, np.transpose([lcal, ical]), fmt='%8.3f %8.5f')
            spec_file = mod_file
            window.set_title(spec_file)
            info = f'spectrum normalized to peak intensity = {peak_int}\n' \
                   f'                               saved as {mod_file}'
            caltext += info
            logging.info(info)
            delete_curve(id_list, graph)
            id_list = draw_spectrum(lcal, ical, lmin, lmax, color='red')
        elif event == 'Clip wavelength range':
            lclip = []
            iclip = []
            for l0 in range(0, len(lcal)):
                if (lmin <= lcal[l0] <= lmax) or (lmin >= lcal[l0] >= lmax):
                    lclip.append(lcal[l0])
                    iclip.append(ical[l0])
            mod_file = m_fun.change_extension(spec_file, 'C.dat')
            np.savetxt(mod_file, np.transpose([lclip, iclip]), fmt='%8.3f %8.5f')
            lcal, ical = np.loadtxt(mod_file, unpack=True, ndmin=2)
            spec_file = mod_file
            window.set_title(spec_file)
            info = f'spectrum clipped to  ({lmin}, {lmax}) nm\n' \
                   f'                               saved as {mod_file}'
            caltext += info
            logging.info(info)
        elif event == 'Compare with spectrum':
            window.Minimize()
            comp_file, info = m_fun.my_get_file(spec_file, title='Compare with spectrum', save_as=False,
                                                file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*'),))
            if comp_file:
                caltext += f'File {comp_file} loaded\n'
                lcomp, icomp = np.loadtxt(comp_file, unpack=True, ndmin=2)
                delete_curve(id_list_comp, graph)
                id_list_comp = draw_spectrum(lcomp, icomp, lmin, lmax, color='red')
                graph.DrawText(spec_file, (lmax - 20 / lscale, imax - 15 / iscale),
                               text_location=sg.TEXT_LOCATION_RIGHT,
                               font='Arial 12', color='blue')
                graph.DrawText(comp_file, (lmax - 20 / lscale, imax - 40 / iscale),
                               text_location=sg.TEXT_LOCATION_RIGHT,
                               font='Arial 12', color='red')
            window.Normal()
        elif event in ('Label calibration lines', 'Label meteor lines'):
            window.Disable()
            calib_lines = True if event == 'Label calibration lines' else False
            layout_label = [[sg.InputText('Cursor', size=(40, 1), key='cursor', disabled=True)],
                            [sg.InputText('', size=(40, 1), key='label')],
                            [sg.Button('Apply'), sg.Button('Cancel')]]
            window_label = sg.Window('Label Peak', layout_label, keep_on_top=True).Finalize()
            offset_calibration = 1000
            lam_peak = 0.0
            if calib_lines:
                for k in range(len(lam_calib)-1):
                    new_offset = abs(label_str[k][0] - x)
                    if new_offset < offset_calibration:
                        offset_calibration = new_offset
                        kk = k
                lam_peak = label_str[kk][0]
                window_label['label'].update(lam_calib[kk])
            else:
                kk = -1
                for k in range(len(lam_meteor)):
                    lambda_list = float(label_str_meteor[k][0])
                    new_offset = abs(lambda_list - x)
                    if new_offset < offset_calibration:
                        offset_calibration = new_offset
                        kk = k
                if kk >= 0:
                    lam_peak = float(label_str_meteor[kk][0])
                    window_label['label'].update(lam_meteor[kk])
            klam = 0
            for k in range(len(lcal)):
                if lcal[k] < x:
                    klam = k
            i_peak = 0
            for k in range(max(0, klam - 20 * plot_pixel_increment), min(klam + 20 * plot_pixel_increment, len(lcal))):
                i_peak = max(i_peak, ical[k])
            window_label['cursor'].update(f'Lambda:{lcal[klam]:8.2f}  Peak:{i_peak:8.2f}')
            while True:
                event, values = window_label.read()
                if event in 'Apply':
                    new_label = values['label']
                    if calib_lines:
                        # check if label changed
                        if new_label != lam_calib[kk]:
                            x = new_label.lstrip()
                            if len(x.split(' ', 1)) == 2:
                                (lam_peak, name) = x.split(' ', 1)
                            else:
                                lam_peak = x
                            try:
                                lam_peak = float(lam_peak)
                            except Exception as e:
                                lam_peak = 0.0
                                sg.PopupError(f'invalid value for wavelength, try again\n{e}',
                                              title='Input Error', keep_on_top=True)
                        if y > i_peak:
                            id_line = graph.DrawLine((lam_peak, i_peak + 20 / iscale),
                                                      (lam_peak, y - 20 / iscale), 'black', 2)
                        else:
                            id_line = graph.DrawLine((lam_peak, i_peak - 20 / iscale),
                                                     (lam_peak, y + 20 / iscale), 'black', 2)
                        id_label = graph.DrawText(new_label, location=(lam_peak, y),
                                       text_location=sg.TEXT_LOCATION_CENTER,
                                       font='Arial 12', color='black')
                    else:
                        id_line = graph.DrawLine((lam_peak, y - 20 / iscale), (lam_peak, imin), 'green', 1)
                        id_label = graph.DrawText(new_label, location=(lam_peak, y),
                                       text_location=sg.TEXT_LOCATION_CENTER,
                                       font='Arial 12', color='green')
                if event in ('Cancel', None):
                    pass
                window_label.close()
                break
            window.Enable()
        elif event == 'Remove label':
            graph.delete_figure(id_line)
            graph.delete_figure(id_label)
            window.refresh()
        elif event == 'Linewidth tool':
            linewidth_tool_enabled = True
            dragging = False
            start_point = end_point = prior_rect = None
            prior_gauss = []
            result = 'Linewidth measurement:'
        elif event == 'FWHM':
            # calculate FWHM
            if not linewidth_tool_enabled:
                sg.PopupError('select linewidth tool and select wavelength range in spectrum')
            else:
                window.Disable()
                linewidth_tool_enabled, result = linewidth_results((lmin_lw, imin_lw),
                                                           (lmax_lw, imax_lw), coeffgauss, fwg, result)
                window.Enable()
                graph.delete_figure(prior_rect)
                delete_curve(prior_gauss, graph)
                window['FWHM'].update(disabled=not linewidth_tool_enabled)

        elif event == 'graph' and linewidth_tool_enabled:  # if there's a "graph" event, then it's a mouse
            x, y = (values['graph'])
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                xmin = min(start_point[0], end_point[0])
                xmax = max(start_point[0], end_point[0])
                prior_rect = graph.draw_rectangle((start_point[0], imin),
                                                  (end_point[0], imax), line_color='Red')
        elif str(event).endswith('+UP') and linewidth_tool_enabled and end_point:
            # The drawing has ended because mouse up
            # lcal_lw  = ical_lw = bias = polygon = []    does not work, makes multiple array!
            lcal_lw = []   # subarray of selected range
            ical_lw = []    # subarray of corresponding intensities
            bias = []
            gauss_fit = []
            # fwg = 0
            for l0 in range(0, len(lcal)):
                if xmin <= lcal[l0] <= xmax:
                    lcal_lw.append(lcal[l0])
                    ical_lw.append(ical[l0])
                    bias.append(0.0)
            if len(lcal_lw) > 4:
                # sort end_points of bias in increasing wavelength order
                # data points are always ordered or should be
                if start_point[0] > end_point[0]:
                    (start_point, end_point) = (end_point, start_point)
                lmin_lw = lcal_lw[0]
                lmax_lw = lcal_lw[-1]
                imin_lw = start_point[1]
                imax_lw = end_point[1]
                # correct baseline
                for l0 in range(0, len(lcal_lw)):
                    bias[l0] = imin_lw + l0 / (len(lcal_lw) - 1) * (imax_lw - imin_lw)
                    ical_lw[l0] -= bias[l0]
                delete_curve(prior_gauss, graph)
                # Gauss fit
                peak_int = np.max(ical_lw)
                for i in range(0, len(lcal_lw)):
                    if (ical_lw[i] - peak_int + 1.e-5) > 0: m = i
                peak0 = lcal_lw[m]
                coeff = [peak_int, peak0, (lmax_lw - lmin_lw) / 4]    # peak height, wavelength, sigma
                try:
                    coeffgauss, var_matrix = optimize.curve_fit(gauss, lcal_lw, ical_lw, p0=coeff)
                    for l0 in range(0, len(lcal_lw)):
                        gauss_fit.append((lcal_lw[l0], gauss(lcal_lw[l0], *coeffgauss) + bias[l0]))
                        # Gaussian lineshape with bias added
                    fwg = 2 * np.sqrt(2 * np.log(2)) * np.abs(coeffgauss[2])  # convert to FWHM
                    np.set_printoptions(precision=3, suppress=True)
                    gauss_fit.append(end_point)
                    gauss_fit.append(start_point)
                    prior_gauss = plot_curve(gauss_fit, graph, line_color='red')
                    window['cursor'].update(f'Linewidth:{fwg:6.2f}  lambda:{coeffgauss[1]:8.2f}')
                except Exception as e:
                    sg.PopupError(f'Error in Gaussfit\n{e}', title='Fit Error', keep_on_top=True)
            else:
                window['cursor'].update(f'not enough datapoints for linewidth')
            # cleanup
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            window['FWHM'].update(disabled=False, button_color=('white', 'red'))


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


def plot_raw_spectrum(rawspec, graph, canvasx, autoscale=True, plot_range=(0, 1000, -.2, 1),
                      plot_style=('red', 1, 1, -0.05)):
    """
    plots  a raw (uncalibrated)spectrum for selection of calibration lines
    :param rawspec: filename of uncalibrated spectrum with extension .dat
    :param graph: window to display spectrum
    :param canvasx: width of graph (needed to size points in graph)
    :param autoscale: if True fit scale to spectrum
    :param plot_range: lmin, lmax, imin, imax
    :param plot_style: defined in main: star_style, ref_style, raw_style, response_style
                        (color, circle_size, line_size, offset)
    :return:
    lmin, lmax: pixel range for autoscale
    imin, imax: intensity range
    lcal, ical: pixel, intensity array
    """
    lmin, lmax, imin, imax = plot_range
    color, circle_size, line_size, offset = plot_style
    lcal, ical = np.loadtxt(rawspec, unpack=True, ndmin=2)
    if autoscale:
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
        if (lmin <= lcal[l0] <= lmax) or (lmin >= lcal[l0] >= lmax):
            points.append((lcal[l0], ical[l0]))
    graph.change_coordinates((lmin, imin), (lmax, imax))
    graph.erase()
    graph.DrawText(rawspec, (0.5 * (lmax + lmin), imax - 0.05 * (imax - imin)), color=color)
    # draw graph
    plot_curve(points, graph, radius=circle_size*(lmax-lmin)/canvasx, line_color=color,
               fill_color=color, width=line_size)
    return (lmin, lmax, imin, imax), lcal, ical


# -------------------------------------------------------------------
def plot_reference_spectrum(rawspec, lcal, ical, graph, canvasx, plot_range=(0, 1000, -.1, 2),
                            plot_style=('blue', 0, 2, -.1)):
    """
    plots  a raw (uncalibrated)spectrum for selection of calibration lines
    :param rawspec: filename of reference spectrum
    :param lcal: wavelength array of spectrum
    :param ical: intensity array of spectrum
    :param graph: window to display spectrum
    :param canvasx: width of graph (needed to size points in graph)
    :param plot_range: lmin, lmax, imin, imax
    :param plot_style: defined in main: star_style, ref_style, raw_style, response_style
                        (color, circle_size, line_size, offset)
    :return id_list: list of graph elements
    """
    # lcal, ical = np.loadtxt(rawspec, unpack=True, ndmin=2)
    color, circle_size, line_size, offset = plot_style
    points = []
    lmin, lmax, imin, imax = plot_range
    for l0 in range(len(lcal)):
        if (lmin <= lcal[l0] <= lmax) or (lmin >= lcal[l0] >= lmax):
            points.append((lcal[l0], ical[l0]))
    graph.change_coordinates((lmin, imin), (lmax, imax))
    # draw graph
    id_list = plot_curve(points, graph, radius=circle_size*(lmax-lmin)/canvasx, line_color=color,
                         fill_color=color, width=line_size)
    id = graph.DrawText(rawspec, (0.5 * (lmax + lmin), imax + offset * (imax - imin)), color=color)
    id_list.append(id)
    id = graph.DrawLine((lmin, 0), (lmax, 0), 'grey', 1)
    id_list.append(id)
    return id_list


def plot_curve(points, graph, radius=0.5, line_color='', fill_color='', width=1):
    """
    :param points: list of coordinates tuple
    :param graph: PySimpleGUI Graph
    :param radius: radius of circle for points
    :param line_color: color of line between points, '' if no line desired
    :param fill_color: color of filled circles at points, '' if no points desired
    :param width: width of connecting line between points
    :return id_list: list of graph id's, used for removing curve with
        for id in id_list:
            graph.delete_figure(id)
    """
    id_list = []
    for x in range(0, len(points)):
        if fill_color and radius:
            idg = graph.DrawCircle(points[x], radius=radius, line_color='', fill_color=fill_color)
            id_list.append(idg)
        if x:
            idg = graph.DrawLine(points[x-1], points[x], color=line_color, width=width)
            id_list.append(idg)
    return id_list


def delete_curve(id_list, graph):
    """
    :param id_list: list of graph elements produced by plot_curve
    :param graph: PySimpleGUI Graph
    """
    for idg in id_list:
        graph.delete_figure(idg)


def wavelength_tools(sigma_nm, file='', _type='reference'):
    """
    several tools for wavelength manipulation
    convolute with Gaussian with width sigma (standard deviation)
    convert A to nm
    convert nm to A
    convert negative and higher orders to wavelength (spectrum is scaled in wavelength*order)
        therefore the wavelength scale is divided by order
    :param sigma_nm: width of the Gaussian in units nm for a spectrum calibrated in nm
    :param file: input file for operations
    :param _type: 'star' for order conversion,
                 'reference' for gaussian and A <--> nm
                 this is used to set the output file in the correct place in the main window
    :return: file, new_file, l_new, i_new, info, _type
    """
    new_file = ''
    l_new = i_new = []
    layout = [[sg.Text('Input File'), sg.InputText(file, key='file', size=(33, 1)),
               sg.Button('Load File')],
              [sg.Frame('Gaussian Filter', [[sg.Text('Sigma Gaussian:'),
                        sg.InputText(sigma_nm, size=(19, 1), key='sigma'), sg.Button('Apply Gaussian')]])],
              [sg.Frame('Wavelength conversion', [[sg.Button('Convert A -> nm'), sg.T('', size=(17, 1)),
                                                   sg.Button('Convert nm -> A')]])],
              [sg.Frame('Order conversion', [[sg.Combo(list(range(-5, 0)) + list(range(2, 6)),
                                                       key='order', enable_events=True, default_value=-1),
               sg.Text('Order --> 1st order', size=(20, 1)), sg.Button('Convert order')]]), sg.Button('Cancel')]]
    window = sg.Window('Convolute with Gaussian, convert wavelength', layout, keep_on_top=True).Finalize()
    while True:
        info = ''
        event, values = window.read()
        if event in (None, 'Cancel'):
            window.close()
            return file, new_file, l_new, i_new, info, _type

        elif event == 'Load File':
            window.Minimize()
            file, info = m_fun.my_get_file(values['file'], title='Spectrum file',
                                     file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*')),
                                     default_extension='*.dat')
            if file:
                window.Normal()
                window['file'].update(file)

        elif event in ('Convert A -> nm', 'Convert nm -> A', 'Convert order'):
            try:
                l_ori, i_ori = np.loadtxt(file, unpack=True, ndmin=2)
                p, ext = path.splitext(file)
                _type = 'reference'
                if event == 'Convert A -> nm':
                    l_new = l_ori / 10.0
                    p += '_nm.dat'
                    info = f'spectrum {p} converted to nm'
                elif event == 'Convert nm -> A':
                    l_new = l_ori * 10.0
                    p += '_A.dat'
                    info = f'spectrum {p} converted to A'
                else:
                    _type = 'star'
                    order = int(values['order'])
                    l_new = l_ori/order
                    if order < 0:
                        l_new = list(reversed(l_new))  # best solution
                        i_ori = list(reversed(i_ori))
                        # l_new = l_new[::-1]  # also good
                        # i_new = i_new[::-1]
                        # l_swap = list(l_new)  # complicated 3 lines
                        # l_swap.reverse()
                        # l_new = l_swap ...
                    p += f'_O{order}.dat'
                    info = f'spectrum {p} converted from order {order} to wavelength'
                np.savetxt(p, np.transpose([l_new, i_ori]), fmt='%8.3f %8.5f')
                logging.info(info)
                window.close()
                return file, p, l_new, i_ori, info, _type
            except Exception as e:
                sg.PopupError(f'invalid value for wavelength, try again\n{e}',
                              title='Input Error', keep_on_top=True)
                info = 'invalid wavelength conversion'
                logging.info(info)

        elif event == 'Apply Gaussian':
            try:
                l_ori, i_ori = np.loadtxt(file, unpack=True, ndmin=2)
                delta = (l_ori[-1] - l_ori[0]) / (len(l_ori) - 1)
                sigma_nm = float(values['sigma'])
                sigma = sigma_nm / delta
                for k in range(len(l_ori)):
                    l_new.append(l_ori[0] + k * delta)
                i_iso = interpolate.interp1d(l_ori, i_ori, kind='quadratic')(l_new)
                i_new = gaussian(i_iso, sigma=sigma)
                window.Minimize()
                p, ext = path.splitext(file)
                p += '_gauss.dat'
                filename, info = m_fun.my_get_file(p, save_as=True,
                                                   file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*')),
                                                   title='Save convoluted spectrum', default_extension='*.dat', )
                window.Normal()
                if len(l_new) > 1:
                    p, ext = path.splitext(filename)
                    if not filename:
                        p = f'filtered{sigma_nm}nm'
                    p += '.dat'
                    np.savetxt(p, np.transpose([l_new, i_new]), fmt='%8.3f %8.5f')
                    info = f'spectrum {p} saved with sigma = {sigma_nm}'
                    logging.info(info)
                    new_file = p
                window.close()

                return file, new_file, l_new, i_new, info, _type
            except Exception as e:
                sg.PopupError(f'invalid value for file or sigma_nm, try again\n{e}',
                              title='Input Error', keep_on_top=True)
                info = 'invalid Gaussian smoothing'
                logging.info(info)

