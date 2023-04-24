import numpy as np
from skimage import io as ios
import PySimpleGUI as sg
import warnings

import m_specfun as m_fun

version = '0.9.30'


def select_lines(infile, contrast, lines, res_dict, fits_dict, wloc, outfil):
    """
    displays new window with image infile + start + 'fit
    a rectangle around the selected line can be selected with dragging the mouse
    :param infile: filebase of image
    :param contrast: brightness of image
    :param lines: list of calibration wavelengths
    :param res_dict: dictionary
    :param fits_dict: "
    :param wloc: location of displayed window for selection
    :param outfil: filename without extension (.txt) with results of line selection
    :return:
    x0, y0: center coordinates of selected rectangle (int)
    dx, dy: half width and height of selected rectangle (int)
    """

    def fitgaussimage(image, xy0, dxy, lam):
        x0 = xy0[0]
        y0 = xy0[1]
        dx = dxy[0]
        dy = dxy[1]
        print(x0, y0, dx, dy)
        data = image[y0 - dy:y0 + dy, x0 - dx:x0 + dx]  # x - y reversed
        params, success = m_fun.fit_gaussian_2d(data)
        if success in [1, 2, 3, 4]:
            (height, x, y, width_x, width_y) = params  # x and y reversed
            width_x = 2 * np.sqrt(2 * np.log(2)) * np.abs(width_x)  # FWHM
            width_y = 2 * np.sqrt(2 * np.log(2)) * np.abs(width_y)  # FWHM
            x = x + y0 - dy  # y and x reversed
            y = y + x0 - dx
            xyw = (y, x, width_y, width_x, lam)  # x - y changed back
            return xyw
        else:
            return 0, 0, 0, 0, 0

    xyl = []
    dxy = [10, 10]
    i = i_plot = 0
    im, header = m_fun.get_fits_image(infile)
    if len(im.shape) == 3:
        imbw = np.sum(im, axis=2)  # used for fitgaussian(data)
    else:
        imbw = im
    # (ymax, xmax) = im.shape
    # print (xmax,ymax)
    m_fun.get_fits_keys(header, fits_dict, res_dict, keyprint=False)
    # #===================================================================
    # new rect_plt
    # first get size of graph from tmp.png and size of image
    # graph coordinates are in image pixels!
    (imy, imx) = im.shape[:2]
    image_file = 'tmp.png'      # scaled image
    imrescale = np.flipud(ios.imread(image_file))  # get shape
    (canvasy, canvasx) = imrescale.shape[:2]
    wlocw = (wloc[0], wloc[1])
    image_elem_sel = [sg.Graph(canvas_size=(canvasx, canvasy), graph_bottom_left=(0, 0),
                               graph_top_right=(imx, imy), key='-GRAPH-',
                               change_submits=True, drag_submits=True)]
    layout_select = [[sg.Ok(), sg.Cancel(), sg.Button('Skip Line'), sg.Button('Finish'),
                      sg.Button('I'), sg.Button('D'), sg.Text(infile, size=(30, 1)),
                      sg.Text(key='info', size=(40, 1))], image_elem_sel]
    winselect = sg.Window('select rectangle  for fit size, click lines',
                          layout_select, finalize=True, location=wlocw,
                          keep_on_top=True, no_titlebar=False, resizable=True,
                          disable_close=False, disable_minimize=True, element_padding=(2, 2))
    # get the graph element for ease of use later
    graph = winselect['-GRAPH-']  # type: sg.Graph
    # initialize interactive graphics
    winselect_active = True
    img = graph.draw_image(image_file, location=(0, imy))
    dragging = False
    start_point = end_point = prior_rect = None
    index = 0
    icircle = itext = None
    color = 'yellow'
    while winselect_active:
        event, values = winselect.read()
        if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = (values["-GRAPH-"])
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = graph.draw_rectangle(start_point,
                                                  end_point, line_color='red')
        elif event is not None and event.endswith('+UP'):
            # The drawing has ended because mouse up
            xy0 = [int(0.5 * (start_point[0] + end_point[0])),
                   int(0.5 * (start_point[1] + end_point[1]))]
            size = (abs(start_point[0] - end_point[0]),
                    abs(start_point[1] - end_point[1]))
            info = winselect["info"]
            info.update(value=f"grabbed rectangle at {xy0} with size {size}")
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            if min(size[0], size[1]) > 2:  # rectangle
                info.update(value=f"rectangle at {xy0} with size {size}")
                dxy = size
            elif i < len(lines):
                if prior_rect:
                    graph.delete_figure(prior_rect)
                print(xy0, lines[i])
                xyw = (fitgaussimage(imbw, xy0, dxy, lines[i]))
                if xyw[0]:  # successful fit
                    if 0 < xyw[0] < imx and 0 < xyw[1] < imy:
                        print(np.float16(xyw))
                        xyl.append(np.float32(xyw))
                        # Draw the click just made
                        r = (xyw[2] + xyw[3]) / 4
                        icircle = graph.DrawCircle((xyw[0], xyw[1]), r, line_color=color, line_width=3)
                        itext = graph.DrawText(
                            '  ' + str(lines[i]), location=(xyw[0], xyw[1]), color=color,
                            font=('Arial', 12), angle=45, text_location=sg.TEXT_LOCATION_BOTTOM_LEFT)
                        info.update(value=f"line {lines[i]} at {np.float16(xyw)}")
                        graph.update()
                        i += 1
                        i_plot += 1
                    else:
                        info.update(value='bad fit, try again')
                        print('bad fit, try again')
                else:
                    info.update(value='Fit not successful, try again')
                    print('Fit not successful, try again')
            else:
                info.update(value='all lines measured, press OK or Cancel')

        elif event == 'Ok':
            if np.array(xyl).shape[0] > 1:
                # minimum of two lines needed for fit
                xyl = np.array(xyl, dtype=np.float32)  # for ordered output
                with open(m_fun.change_extension(outfil, '.txt'), 'ab+') as f:
                    np.savetxt(f, xyl, fmt='%8.2f', header=str(index) + ' ' + str(infile) + '.fit')
                    np.savetxt(f, np.zeros((1, 5)), fmt='%8.2f')
                index += 1
                color = 'red' if color == 'yellow' else 'yellow'  # alternate colors for spectra
            elif icircle:
                graph.delete_figure(icircle)  # last point
                graph.delete_figure(itext)
                graph.update()
            xyl = []
            i = i_plot = 0

        elif event == 'Cancel':
            for ind in range(i_plot):
                xyl = np.array(xyl, dtype=np.float32)  # for ordered output
                rsq2 = (xyl[ind, 2] + xyl[ind, 3]) / 5.6
                drag_figures = graph.get_figures_at_location((xyl[ind, 0] + rsq2, xyl[ind, 1] + rsq2))
                for figure in drag_figures:
                    if figure != img:
                        graph.delete_figure(figure)
            graph.update()
            xyl = []
            i = i_plot = 0

        elif event == 'Skip Line':
            i += 1  # do not increment iplot!

        elif event in ('I', 'D'):
            if event == 'I':
                contrast *= 2
            else:
                contrast /= 2
            im_tmp = imrescale / np.max(imrescale) * 255 * contrast
            im_tmp = np.clip(im_tmp, 0.0, 255)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ios.imsave(image_file, np.flipud(im_tmp).astype(np.uint8))
            graph.delete_figure(img)
            img = graph.draw_image(image_file, location=(0, imy))
            graph.send_figure_to_back(img)

        elif event in ('Finish', None):
            if event == 'Finish':
                with open(outfil + '.txt', 'ab+') as f:
                    np.savetxt(f, np.zeros((1, 5)), fmt='%8.2f')
                (x, y) = winselect.current_location()
                wlocw = (x, y)
            winselect.close()
            return wlocw
