# -------------------------------------------------------------------
# m_image_tools functions for m_spec
# Author: Martin Dubs, 2021
# -------------------------------------------------------------------
import logging
import warnings
import copy
import PySimpleGUI as sg
import numpy as np
from skimage import io as ios
from skimage.filters import gaussian

import m_specfun as m_fun

version = '0.9.27'


def image_tools(file, opt_dict, fits_dict, res_dict,
                sigma_gauss=5, f_pix=1000):
    """
    image tool for creation of flat
    possibility to flip image and average
    convolute with Gaussian with width sigma (standard deviation)
    :param file: image input file (*.fit)
    :param opt_dict: options dictionary
    :param fits_dict: fits keywords dictionary
    :param res_dict: results dictionary
    :param sigma_gauss: sigma for Gaussian smoothing
    :param f_pix: focal length expressed in pixels focal length / pixel_size / binning

    :return: file, new_file, l_new, i_new, info, _type
    """

    def get_inverse_polynomial(r_max, rp_max, r_poly, deg=9, get_max=False):
        rp_lin = np.linspace(-rp_max, rp_max, 21)
        r_calc = np.poly1d(r_poly)
        y = np.poly1d(r_calc(rp_lin))
        rp_poly = np.polyfit(y, rp_lin, deg)
        rp_fit = np.poly1d(rp_poly)
        rp_max = rp_fit(r_max)
        # print(f'rp_max, rp_poly, {rp_max:6.1f}, {rp_poly[:-1:2]}')
        if get_max:
            return rp_max, rp_poly
        else:
            return rp_poly

    np.set_printoptions(precision=3, suppress=False)
    new_file = ''
    image = []
    edge = 5
    bob_doubler = opt_dict['bob']
    wloc_image = (opt_dict['win_x'] + opt_dict['calc_off_x'], opt_dict['win_y'] + opt_dict['calc_off_y'])
    graph_size = opt_dict['graph_size']
    image_file = 'tmp.png'      # scaled image,used for new window size
    operator_dict = {'': ' ', '+': 'plus', '-': 'minus', '*': 'mult', '/': 'div', 'N': 'N'}
    imrescale = np.flipud(ios.imread(image_file))  # get shape
    (canvasy, canvasx) = imrescale.shape[:2]
    image_elem_flat = sg.Graph(canvas_size=(canvasx, canvasy), graph_bottom_left=(0, 0),
        graph_top_right=(graph_size, graph_size), key='-GRAPH-', change_submits=True, drag_submits=True)
    layout = [[sg.Text('Input File'), sg.InputText(file, key='file', size=(33, 1)),
               sg.Button('Load File')],
              [sg.Frame('Flat processing',
                        [[sg.Checkbox('Uniform', key='cb_const')],
                         [sg.Checkbox('Mirror x', key='cb_x')],
                         [sg.Checkbox('Mirror y', key='cb_y')],
                         [sg.Checkbox('Invert', key='cb_inv')],
                         [sg.Checkbox('Average', key='cb_ave')],
                         [sg.Checkbox('Edge_excl.', key='cb_edge'), sg.InputText(edge, key='edge', size=(3, 1))],
                         [sg.Checkbox('Gaussian', key='cb_gauss')],
                         [sg.Text('Sigma'), sg.InputText(sigma_gauss, key='sigma_g', size=(3, 1))],
                         [sg.Checkbox('Corr. dist.', key='cb_corr_dist')],
                         [sg.Checkbox('Bob doubler', key='cb_bob', default=bob_doubler)],
                         [sg.Combo(list(operator_dict.keys()), key='-OPER-', enable_events=True),
                          sg.InputText('', key='operand', size=(15, 1))],
                         [sg.Text('Image manip.'), sg.Button('Sel. Im.')],
                         [sg.Button('Apply Flat')], ]), image_elem_flat],
              [sg.Text('Output File'), sg.InputText(new_file, key='new_file', size=(33, 1)),
               sg.Button('Save File'), sg.Button('Cancel')]]
    window = sg.Window('Image Tools', layout, keep_on_top=True, location=wloc_image).Finalize()
    graph = window['-GRAPH-']  # type: sg.Graph
    while True:
        all_info = ''
        event, values = window.read()
        if event in (None, 'Cancel'):
            window.close()
            return new_file, all_info

        elif event == 'Load File':
            window.Minimize()
            file, info = m_fun.my_get_file(values['file'], title='Image file',
                                    file_types=(('FIT Files', '*.fit'), ('PNG Files', '*.png'),
                                                ('BMP-File', '*.bmp')),
                                    default_extension='*.fit')
            if file.lower().endswith(('.fit', '.png', '.bmp')):
                window['file'].update(file)
                # original graph size for loading image in correct position
                graph.change_coordinates((0, 0), (graph_size, graph_size))
                data, file, image = m_fun.draw_scaled_image(file, graph, opt_dict, get_array=True)
                multichannel = True if len(image.shape) == 3 else False
                new_file = m_fun.change_extension(file, '')
                if file.lower().endswith('.fit'):
                    image, header = m_fun.get_fits_image(new_file)
                    if 'M_BOB' in header.keys():
                        bob_doubler = int(header['M_BOB'])
                        window['cb_bob'].update(bob_doubler)
                else:
                    image = m_fun.get_png_image(file, multichannel)
                (imy, imx) = image.shape[0:2]
                # change to image coordinates
                graph.change_coordinates((0, 0), (imx, imy))
                window.refresh()
            else:
                sg.PopupError('wrong file format, only fit, png or bmp allowed')
            window.Normal()

        if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = (values["-GRAPH-"])
            if 0 <= x < imx and 0 <= y < imy:
                if multichannel:
                    window.set_title(f'Image Tools, x: {x}, y: {y}, I: {image[y, x, :]}')
                else:
                    window.set_title(f'Image Tools, x: {x}, y: {y}, I: {image[y, x]:7.5f}')
                window.refresh()

        elif event == 'Apply Flat':
            if image != []:
                cb_dist = values['cb_corr_dist']
                cb_average = values['cb_ave']
                all_info = new_file
                if values['cb_const']:
                    # create uniform flat from actual image for correction of distortion
                    image = 0 * image + 1.0
                    new_file = 'uniform'
                    all_info = new_file
                if values['cb_x']:  # mirror horizontally
                    new_image = image[:, ::-1]
                    image = (image + new_image) / 2 if cb_average else new_image
                    new_file += '_mx'
                    all_info += ' mx'
                if values['cb_y']:  # mirror vertically
                    new_image = image[::-1, :]
                    image = (image + new_image) / 2 if cb_average else new_image
                    new_file += '_my'
                    all_info += ' my'
                if values['cb_inv']:  # rotate 180° around center
                    new_image = image[::-1, ::-1]
                    image = (image + new_image) / 2 if cb_average else new_image
                    new_file += '_inv'
                    all_info += ' inv'
                if cb_average:  # average new and previous images, update info
                    new_file += '_ave'
                    all_info += ' average'
                if values['cb_edge']:
                    edge = int(float(values['edge']))
                    for x in range(edge):
                        image[:, x] = image[:, edge]
                        image[:, -x] = image[:, -edge]
                        image[x, :] = image[edge, :]
                        image[-x, :] = image[-edge, :]
                if values['cb_gauss']:  # convolute with Gaussian with width sigma
                    sigma_gauss = int(float(values['sigma_g']))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        image = gaussian(image, sigma=sigma_gauss)
                    new_file += f'_g{sigma_gauss}'
                    all_info += f' gauss sigma={sigma_gauss}'

                if cb_dist:  # correct area change from distortion
                    if multichannel:
                        window.Minimize()
                        sg.PopupError('color flat not implemented, converted to b/w')
                        window.Normal()
                        image = np.sum(image, axis=2) / 3
                        multichannel = False
                    (imy, imx) = image.shape[:2]
                    [scalxy, x00, y00, rot, disp0, a3, a5] = res_dict.values()
                    bob_doubler = values['cb_bob']
                    fits_dict['M_BOB'] = bob_doubler
                    if bob_doubler:  # center measured before scaling in y scalxy*2 compensates for half image height
                        scalxy *= 2.0
                        y00 /= 2.0
                        fits_dict['M_BOB'] = 1
                    # get range of radius function and inverse
                    xmax = max(x00, imx - x00)
                    ymax = max(y00, imy - y00)
                    r_max = np.sqrt(xmax**2 + (ymax * scalxy) ** 2)
                    print(f'r_max, {r_max:6.1f}')
                    # select approximate value for rp_max
                    rp_max = r_max
                    # distortion polynomial r = f(rp):
                    r_poly = np.array([a5, 0, a3, 0, 1.0, 0.])
                    rp_new = 0.0
                    while abs(rp_max - rp_new) > 1.0:
                        rp_new, rp_poly = get_inverse_polynomial(r_max, rp_max, r_poly, get_max=True)
                        rp_max = rp_new
                    # inverse polynomial rp = fp(r):
                    rp_poly = get_inverse_polynomial(r_max, rp_max, r_poly)
                    print(f'rp_max, rp_poly, {rp_max:6.1f}, {rp_poly[:-1:2]}')
                    logging.info(f'rp_max, rp_poly, {rp_max:6.1f}, {rp_poly[:-1:2]}')
                    rp_fit = np.poly1d(rp_poly)

                    def cos_rho(scalxy, x00, y00):
                        # returns cos(rho) = sqrt(1 - (r’/f_pix)**2)
                        return lambda y, x: np.sqrt(1.0 - (rp_fit(np.sqrt((x - x00)**2 +
                                                    (((y - y00) * scalxy)**2))) / f_pix)**2)

                    # create the image coordinates array:
                    yin, xin = np.mgrid[0:imy, 0:imx]
                    try:
                        new_file += '_corr'
                        all_info += ' correct distortion'
                        s_data = cos_rho(scalxy, x00, y00)(yin, xin)
                        image = image * s_data
                        image = image / np.max(image)
                        image = np.maximum(image, -1.0)
                    except Exception as e:
                        window.Minimize()
                        image = 0.0 * image
                        sg.PopupError(f'Flat error\n{e}')
                        window.Normal()

                if values['-OPER-']:  # arithmetic operations
                    window.Minimize()
                    operator = values['-OPER-']
                    operand_string = values['operand']
                    try:
                        operand = float(operand_string)  # number
                        constant = True
                    except Exception:
                        operand = operand_string  # filename
                        constant = False
                        if operand:
                            try:
                                if operand.lower().endswith('.fit'):
                                    image2, header2 = m_fun.get_fits_image(operand)
                                else:
                                    image2 = m_fun.get_png_image(operand)  # get b/w image
                            except Exception:
                                sg.PopupError(f'cannot read image {operand}')
                                image2 = np.array([])
                            operand = m_fun.change_extension(operand, '')
                            if (imy, imx) != image2.shape[0:2]:
                                sg.PopupError(f'wrong image size: {image2.shape[0:2]} != {(imy, imx)}')
                    try:
                        if operator == 'N':
                            image = image / np.max(image)
                            operand = ''
                            window['operand'].update(operand)
                        else:
                            if constant:
                                image2 = np.array([operand])
                            if operator == '+':
                                image = image + image2
                            elif operator == '-':
                                image = image - image2
                            elif operator == '*':
                                image = image * image2
                            elif operator == '/':
                                image = image / image2
                        new_file += ('_' + operator_dict[operator] + '_' + str(operand).replace('.', ''))
                        all_info += (' ' + operator_dict[operator] + ' ' + str(operand))

                    except Exception as e:
                        sg.PopupError(f'Illegal arithmetic operation:\n{e}')
                    window.Normal()

                # write results to file and log
                new_file = new_file.replace('.fit', '')
                new_file += '.fit'
                all_info += ' written to _flat.fit'
                logging.info(all_info)
                m_fun.write_fits_image(image, '_flat.fit', fits_dict, dist=cb_dist)
                # temporary change of graph_size
                opt_dict_temp = copy.deepcopy(opt_dict)
                opt_dict_temp['graph_size'] = imy
                m_fun.draw_scaled_image('_flat.fit', graph, opt_dict_temp)
                window['new_file'].update(new_file)

        elif event == 'Sel. Im.':
            window.Minimize()
            operand, info = m_fun.my_get_file(values['operand'], title='Image file',
                                    file_types=(('FIT Files', '*.fit'), ('PNG Files', '*.png'),
                                                ('BMP-File', '*.bmp')),
                                    default_extension='*.fit')
            window['operand'].update(operand)
            window.refresh()
            window.Normal()

        elif event == 'Save File':
            window.Minimize()
            new_file, info = m_fun.my_get_file(new_file, title='Save image',
                                               save_as=True, default_extension='.fit',
                                               file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),))
            if new_file:
                m_fun.write_fits_image(image, new_file, fits_dict)
            window.Normal()
