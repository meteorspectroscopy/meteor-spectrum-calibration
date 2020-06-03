# -------------------------------------------------------------------
# m_spec, processing of meteor spectra
# Author: Martin Dubs, 2020
# -------------------------------------------------------------------
import numpy as np
import logging
import time
import warnings
from skimage import io  # , img_as_float # needed for reading
import os.path as path
import os
import PySimpleGUI as sg
import m_specfun as m_fun

version = '0.9.18'


# -------------------------------------------------------------------
# main program
# -------------------------------------------------------------------
def main():
    def t_zoom(z):
        """
        if zoom is close to 1.0, speed of loading images can be increased
        by loading images directly
        :param z: zoom factor
        :return: True if rescaling necessary, else no image rescaling
        """
        if abs(z - 1.0) > 0.1:
            result = True
        else:
            result = False
        return result

    # start with default inifile, if inifile not found, a default configuration is loaded
    bc_enabled = ('white', 'green')
    bc_disabled = (None, 'darkblue')
    sg.SetGlobalIcon('Koji.ico')
    sg_ver = sg.version.split(' ')[0]
    print('PySimpleGUI', sg_ver)
    if int(sg_ver.split('.')[0]) >= 4 and int(sg_ver.split('.')[1]) >= 9:
        sg.change_look_and_feel('SystemDefault')  # suppresses message in PySimpleGUI >= 4.9.0
    print('version m_spec, m_specfun', version, m_fun.version)
    logging.info('M_SPEC START +++++++++++++++++++++++++++++++++++++++++++++')
    logging.info(f'M_SPEC version {version}, {m_fun.version} ++++++++++++++++++++++++++++')
    ini_file = 'm_set.ini'
    par_text, par_dict, res_dict, fits_dict, opt_dict = m_fun.read_configuration(ini_file,
                                            m_fun.par_dict, m_fun.res_dict, m_fun.opt_dict)
    res_key = list(res_dict.keys())
    res_v = list(res_dict.values())
    fits_dict['VERSION'] = version
    fits_v = list(fits_dict.values())
    [zoom, wsx, wsy, wlocx, wlocy, xoff_calc, yoff_calc, xoff_setup, yoff_setup,
        debug, fit_report, win2ima, opt_comment, outpath, mdist, colorflag] = list(opt_dict.values())
    if par_text == '':
        sg.PopupError(f'no valid configuration found, default {ini_file} created')
    # default values for video
    dat_tim = ''
    sta = ''
    binning = par_dict['i_binning']
    maxim = 200
    im_scale = zoom
    pngdir = 'tmp'
    png_name = 'm_'
    result_text = ''
    video_list_length = 20
    # default values for distortion
    n_back = 20
    # default values for registration
    first = 25
    nm = 0
    nim = 0
    nmp = 0
    i_reg = 1
    contrast = 1
    reg_text = ''
    reg_file = 'r'
    out_fil = ''
    outfile = ''
    last_file_sum = False
    # default values for calibration
    _image = np.flipud(io.imread('tmp.png'))  # get shape of screen image
    (canvasy, canvasx) = _image.shape[:2]
    raw_spec_file = ''
    select_line_enabled = False
    table = []
    spec_file = ''
    llist = ''
    plot_w = 1000
    plot_h = 500
    i_min = -0.5
    i_max = 5
    cal_dat_file = ''
    cal_text_file = ''
    c = []
    graph_enabled = False
    line_list = 'm_linelist'
    table_edited = False

    # -------------------------------------------------------------------
    # definition of GUI
    # -------------------------------------------------------------------

    # setup tab---------------------------------------------------------------------
    #     parameter_element = sg.Multiline('partext', size=(40,8), autoscroll=True)
    result_elem = sg.Multiline(par_text, size=(50, 30), disabled=True, autoscroll=True)
    setup_file_display_elem = sg.InputText('m_set.ini', size=(60, 1), key='-INI_FILE-')

    menu_def = [
        ['File', ['Exit']],
        ['View', ['Logfile', 'Edit Text File', 'Fits-Header', 'Set Zoom Factor']],
        ['Tools', ['Add Images']],
        ['Help', 'About...'], ]

    setup_file_element = sg.Frame('Configuration File',
                                  [[sg.Text('File'), setup_file_display_elem,
                                    sg.Button('Load Setup', key='-LOAD_SETUP-'),
                                    sg.Button('Save Setup', key='-SAVE_SETUP-'),
                                    sg.Button('Save Default', key='-SAVE_DEFAULT-')]])

    row_layout = [[sg.Text('Distortion')]]
    for k in range(7):
        kk = f'k{k}'
        row_layout += [sg.Text(res_key[k], size=(10, 1)),
                       sg.Input(res_v[k], size=(45, 1), key=kk)],
    for k in range(7):
        kk = f'k{k + 7}'
        row_layout += [sg.Text(list(fits_dict.keys())[k], size=(10, 1)),
                       sg.Input(fits_v[k], size=(45, 1), key=kk)],

    # Options
    zoom_elem = sg.Input(zoom, key='-ZOOM-', size=(7, 1))
    cb_elem_debug = sg.Checkbox('debug', default=debug, pad=(10, 0), key='-DEBUG-')
    cb_elem_fitreport = sg.Checkbox('fit-report', default=fit_report,
                                    pad=(10, 0), key='-FIT_REPORT-')
    cb_elem_w2i = sg.Checkbox('scale_win2ima', default=win2ima,
                              pad=(10, 0), key='-W2I-')
    input_rows = [[sg.Text('Zoom', size=(6, 1)), zoom_elem],
                  [cb_elem_debug], [cb_elem_fitreport], [cb_elem_w2i],
                  [sg.Text('Comment', size=(10, 1))],
                  [sg.InputText(opt_comment, size=(16, 1), key='-COMMENT-')],
                  [sg.Button('Apply', key='-APPLY_OPT-')]]
    layout_options = sg.Frame('Options', input_rows)

    # Parameters
    layout_setup = [[sg.Frame('Settings', [[setup_file_element], [result_elem,
                    sg.Frame('Setup Parameters', row_layout), layout_options]])]]

    # Video tab---------------------------------------------------------------------
    image_element_video = sg.Image(filename='tmp.png', key='-V_IMAGE-')  # do not reuse
    filename_display_elem = sg.InputText('', size=(60, 1), key='-VIDEO-')

    image_options_element = [[sg.Text('Temporary Image Folder')],
                             [sg.InputText(pngdir, size=(38, 1), key='-TMP-'),
                              sg.FolderBrowse('...')], [sg.Text('PNG Image Base:'),
                              sg.InputText(png_name, size=(25, 1), key='-PNG_BASE-')],
                             [sg.Checkbox('Bob Doubler', default=False, pad=(10, 0), key='-BOB-')],
                             [sg.Checkbox('Bottom Field First', default=True, pad=(10, 0), key='-BFF-')],
                             [sg.Combo([1, 2, 3, 4], key='-BIN-', enable_events=True,
                                       default_value=binning), sg.Text(' Binning')],
                             [sg.Text('Max number of images:'),
                              sg.InputText(str(maxim), size=(8, 1), key='-MAXIM-')],
                             [sg.Text('_' * 44)], [sg.Text('Results')],
                             [sg.Multiline('', size=(42, 5), disabled=True,
                                           key='-RESULT-', autoscroll=True)]]

    sub_frame_element = sg.Frame('Video File', [[sg.Text('File'),
                                filename_display_elem, sg.Button('Load_Video',
                                    file_types=(('AVI-File', '*.avi'), ('ALL Files', '*.*'))),
                                sg.Button('Previous', key='-PREVIOUS-', disabled=True),
                                sg.Button('Next', key='-NEXT-', disabled=True),
                                sg.Button('Continue', key='-GOTO_DIST-', disabled=True,
                                    button_color=bc_disabled, bind_return_key=True)]])

    video_options_element = sg.Frame('Options', image_options_element)


    # Distortion tab--------------------------------------------------------------------
    dist_elem = sg.Frame('Parameters',
                [[sg.Text('temporary image folder')],
                [sg.InputText('tmp', size=(38, 1), key='-TMP_D-', enable_events=True),
                    sg.FolderBrowse('...')], [sg.Text('PNG Image Base:'),
                    sg.InputText('m_', size=(26, 1), key='-PNG_BASED-', enable_events=True)],
                [sg.Text('Process folder')],
                [sg.InputText(outpath, size=(34, 1), key='-OUT-', disabled=True),
                 sg.Button('Select', key='-SEL_OUT-')], [sg.Text('Distorted Image Base:'),
                    sg.InputText(mdist, size=(22, 1), key='-M_DIST-')],
                [sg.Checkbox('Apply distortion', default=True, pad=(10, 0), key='-DIST-')],
                [sg.Checkbox('Background subtraction', default=True,
                    pad=(10, 0), key='-BACK-')],
                [sg.Checkbox('Color processing', default=False,
                    pad=(10, 0), key='-COLOR-')],
                [sg.Checkbox('Bob Doubler', default=False, pad=(10, 0), key='-BOB_D-')],
                [sg.Text('Number of background images:'),
                    sg.InputText(n_back, size=(15, 1), key='-N_BACK-')],
                [sg.Text('Index of start image:'),
                    sg.InputText(first, size=(24, 1), key='-N_START-')],
                [sg.Text('Number of distorted images:'),
                    sg.InputText(nm, size=(17, 1), key='-N_IMAGE-')], [sg.Text('_' * 34)],
                [sg.Button('Apply Distortion', key='-APPLY_DIST-'),
                    sg.Button('Continue', key='-GOTO_REG-', disabled=True)],
                [sg.Text('Results')], [sg.Multiline('Result', size=(42, 8),
                    disabled=True, key='-RESULT2-', autoscroll=True)]])

    image_element_distortion = sg.Image(filename='tmp.png', key='-D_IMAGE-')

    # Registration tab--------------------------------------------------------------------
    image_element_registration = sg.Image(filename='tmp.png', key='-R_IMAGE-')
    register_elem = [[sg.Frame('Registration', [
        [sg.Text('Process folder'),
         sg.InputText(outpath, size=(30, 1), key='-OUT_R-', disabled=True),
         sg.Button('Select', key='-SEL_OUT_R-'), sg.Button('Previous', key='-PREV_R-'),
         sg.Button('Next', key='-NEXT_R-'), sg.Text('Current Image:'),
         sg.InputText(mdist + str(i_reg), size=(20, 1), key='-INDEX_R-', disabled=True),
         sg.Text('Max Images:'),
         sg.InputText(nm, size=(4, 1), key='-N_MAX_R-'),
         sg.Button('Darker', key='-LOW_C-'), sg.Button('Brighter', key='-HIGH_C-')]])],
           [sg.Frame('Parameters', [[sg.Text('Distorted Image Base:'),
                sg.InputText('mdist', size=(24, 1), key='-M_DIST_R-', enable_events=True)],
            [sg.Text('Registered Image Base:'),
                sg.InputText(reg_file, size=(22, 1), key='-REG_BASE-')],
            [sg.Text('Index of start image:'),
                sg.InputText('1', size=(25, 1), key='-N_START_R-')],
            [sg.Text('Number of registered images:'),
                sg.InputText(nm, size=(18, 1), key='-N_REG-')],
            [sg.Text('_' * 44)], [sg.Button('Sel Start', key='-SEL_START-'),
                sg.Button('Sel Last', key='-SEL_LAST-')],
            [sg.Button('Register', key='-REGISTER-'),
                sg.Button('Show Sum', key='-SHOW_SUM_R-', disabled=True),
                sg.Checkbox('show registered', default=False, pad=(10, 0),
                            key='-SHOW_REG-')],
            [sg.InputText('r_add', size=(32, 1), key='-RADD-'),
                sg.Button('Load Radd', key='-LOAD_R-')],
            [sg.Button('Add Rows', disabled=True, key='-ADD_ROWS-'),
                sg.Button('Save raw spectrum', disabled=True, key='-SAVE_RAW-'),
                sg.Button('Calibrate', disabled=True, key='-CAL_R-')],
            [sg.Text('Results')],
            [sg.Multiline('Result', size=(42, 15), disabled=True, key='-RESULT3-',
                            autoscroll=True)]]), image_element_registration]]

    column = [[sg.Graph(canvas_size=(canvasx, canvasy), graph_bottom_left=(0.0, 0.0),
                        graph_top_right=(1.0, 1.0), background_color='white', key='graph',
                        enable_events=True, drag_submits=True, float_values=True)], ]

    plot_elem = [sg.Frame('Plot Spectrum',
                          [[sg.InputText(cal_dat_file, size=(40, 1), key='-PLOT_SPEC-')],
                           [sg.Button('Load Spectrum', key='-LOADS-'),
                            sg.Button('Plot Spectrum', key='-PLOTS-', disabled=True,
                                button_color=bc_disabled)],
                           [sg.Checkbox('Grid lines', default=True, key='-GRID-'),
                            sg.Checkbox('Auto scale', default=False, key='-AUTO_SCALE-'),
                            sg.Checkbox('Norm scale', default=False, key='-NORM_SCALE-')],
                           [sg.T('lambda min:'), sg.In('', key='l_min', size=(8, 1)),
                            sg.T('max:'), sg.In('', key='l_max', size=(8, 1))],
                           [sg.T('Title    Plot width'), sg.In(plot_w, key='plot_w', size=(7, 1)),
                            sg.T(' height'), sg.In(plot_h, key='plot_h', size=(7, 1))],
                           [sg.Combo('', size=(38, 1), key='-PLOT_TITLE-')]], )]

    calibrate_elem = [[sg.Frame('Calibration', [
        [sg.Text('Process folder'),
         sg.InputText(outpath, size=(28, 1), disabled=True, key='-OUT_C-')],
        [sg.InputText(outfile, size=(31, 1), key='-SPEC_R-'),
         sg.Button('Load Raw', key='-LOAD_RAW-')],
        [sg.Button('Select Lines', key='-S_LINES-', disabled=True, button_color=bc_disabled)],
        [sg.Text('Pos          Wavelength')],
        [sg.InputText('0', size=(9, 1), justification='r', key='-POS-', disabled=True),
         sg.Combo(['           ', '0 zero', '517.5 Mg I', '589 Na I', '777.4 O I'], key='-LAMBDA-',
                  enable_events=True, disabled=True),
         sg.Button('Sel. Line', key='-S_LINE-', disabled=True, button_color=bc_disabled),
         sg.Button('Save table', key='-SAVE_T-')],
        [sg.Button('Load Table', key='-LOAD_TABLE-', disabled=True, button_color=bc_disabled),
         sg.Button('Calibration', key='-CALI-', disabled=True, button_color=bc_disabled),
         sg.Text('Polynomial degree:'),
         sg.Combo([0, 1, 2, 3, 4, 5], key='-POLY-', enable_events=True, default_value=1), ],
        plot_elem,
        [sg.Multiline('Result', size=(40, 15), disabled=True, key='-RESULT4-', autoscroll=True)]]),
                   sg.Frame('Raw spectrum', column, key='-COLUMN-')]]

    # ==============================================================================
    # Tabs and window
    setup_tab_element = sg.Tab('Setup', layout_setup, tooltip='tip', key='-T_SETUP-')
    video_tab_element = sg.Tab('Video conversion', [[sub_frame_element],
                               [video_options_element, image_element_video]], key='-T_VIDEO-')
    dist_tab_element = sg.Tab('Distortion', [[dist_elem, image_element_distortion]], tooltip='tip', key='-T_DIST-')
    reg_tab_element = sg.Tab('Registration', register_elem, tooltip='tip', key='-T_REG-')
    cal_tab_element = sg.Tab('Calibration', calibrate_elem, tooltip='tip', key='-T_CAL-')
    tabs_element = sg.TabGroup([[setup_tab_element], [video_tab_element],
                            [dist_tab_element], [reg_tab_element], [cal_tab_element]],
                             enable_events=True)
    window = sg.Window('M_SPEC, Version: ' + version, [[sg.Menu(menu_def, tearoff=True)],
                        [tabs_element]], location=(wlocx, wlocy), size=(wsx, wsy), resizable=True)

    # ==============================================================================
    # Main loop
    # ==============================================================================
    while True:
        event, values = window.read()
        if event is None:  # always give a way out!
            break

        # ==============================================================================
        # Menu
        # ==============================================================================
        if event is 'Logfile':
            m_fun.log_window(m_fun.logfile)
        if event is 'Edit Text File':
            m_fun.edit_text_window(llist)
        if event is 'Fits-Header':
            m_fun.view_fits_header(outfile)
        if event is 'About...':
            m_fun.about(version)
        if event is 'Add Images':
            sum_file, nim = m_fun.add_images(im_scale, contrast=1, average=True)
            window['-RADD-'].update(sum_file)
        if event is 'Set Zoom Factor':
            file = sg.PopupGetFile('Select Fits-Image', no_window=True,
                                   file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),), )
            if file:
                image, header = m_fun.get_fits_image(file)
                (opt_dict['win_width'], opt_dict['win_height']) = window.Size
                im_scale = m_fun.set_image_scale(image, opt_dict)
                m_fun.show_image_array(image, im_scale, image_element_video)
                m_fun.show_image_array(image, im_scale, image_element_distortion)
                m_fun.show_image_array(image, im_scale, image_element_registration)
        # ==============================================================================
        # Setup Tab
        # ==============================================================================
        elif event == '-LOAD_SETUP-':
            ini_file = sg.PopupGetFile('Get Setup File', no_window=True,
                        file_types=(('Setup Files', '*.ini'), ('ALL Files', '*.*'),), )
            if ini_file:
                setup_file_display_elem.update(ini_file)
                par_text, par_dict, res_dict, fits_dict, opt_dict = m_fun.read_configuration(ini_file,
                                                                                             par_dict, res_dict, opt_dict)
                result_elem.update(par_text)
                # update version to current script
                fits_dict['VERSION'] = version
                window['-BIN-'].update(value=par_dict['i_binning'])
                res_v = list(res_dict.values())
                fits_v = list(fits_dict.values())
                for k in range(7):
                    kk = f'k{k}'
                    window[kk].Update(res_v[k])
                for k in range(7):
                    kk = f'k{k + 7}'
                    window[kk].Update(fits_v[k])
                if list(opt_dict.values()) != []:
                    [zoom, wsx, wsy, wlocx, wlocy, xoff_calc, yoff_calc,
                     xoff_setup, yoff_setup, debug, fit_report, win2ima,
                     opt_comment, outpath, mdist, colorflag] = list(opt_dict.values())
                zoom_elem.Update(zoom)
                im_scale = zoom
                cb_elem_debug.Update(debug)
                cb_elem_fitreport.Update(fit_report)
                cb_elem_w2i.Update(win2ima)
                window['-COMMENT-'].Update(opt_comment)
                window['-OUT-'].Update(outpath)
                window['-OUT_R-'].Update(outpath)
                window['-OUT_C-'].Update(outpath)
                window.Move(wlocx, wlocy)

        elif event in ('-SAVE_SETUP-', '-SAVE_DEFAULT-', '-APPLY_OPT-', 'Exit'):
            if event == '-SAVE_SETUP-':
                ini_file = sg.PopupGetFile('Get Setup File', no_window=True,
                    default_extension='*.ini', default_path=ini_file, save_as=True,
                    file_types=(('Setup Files', '*.ini'), ('ALL Files', '*.*'),), )
            else:
                ini_file = 'm_set.ini'
            setup_file_display_elem.update(ini_file)
            par_dict['i_binning'] = int(values['-BIN-'])  # from video tab
            # update res_dict and fits_dict with new values
            for k in range(7):
                kk = f'k{k}'
                print(k, res_key[k], values[kk])
                res_v[k] = float(values[kk])
            res_dict = dict(list(zip(res_key, res_v)))  # update res_dict
            for k in range(7):
                kk = f'k{k + 7}'
                fits_v[k] = values[kk]
            for k in res_dict.keys():
                fkey = 'D_' + k.upper()
                fits_dict[fkey] = np.float32(res_dict[k])
                if k[0] == 'a':
                    logging.info(f'{k} = {res_dict[k]:9.3e}')
                else:
                    logging.info(f'{k} = {res_dict[k]:9.3f}')
            logging.info(f"'DATE-OBS' = {dat_tim}")
            logging.info(f"'M-STATIO' = {sta}")
            # update options
            opt_dict['zoom'] = float(zoom_elem.Get())
            opt_dict['debug'] = cb_elem_debug.Get()
            opt_dict['fit-report'] = cb_elem_fitreport.Get()
            opt_dict['scale_win2ima'] = cb_elem_w2i.Get()
            opt_dict['comment'] = values['-COMMENT-']
            opt_dict['outpath'] = values['-OUT-']
            opt_dict['mdist'] = values['-M_DIST-']
            opt_dict['colorflag'] = values['-COLOR-']  # from register_images tab
            (wsx, wsy) = window.Size
            opt_dict['win_width'] = wsx
            opt_dict['win_height'] = wsy
            # optvar = list(opt_dict.values())
            [zoom, wsx, wsy, wlocx, wlocy, xoff_calc, yoff_calc,
             xoff_setup, yoff_setup, debug, fit_report, win2ima,
             opt_comment, outpath, mdist, colorflag] = list(opt_dict.values())
            if ini_file and event != '-APPLY_OPT-':
                m_fun.write_configuration(ini_file, par_dict, res_dict, fits_dict, opt_dict)
            if event == 'Exit':
                window.close()
                break

        # ==============================================================================
        # Video Tab
        # ==============================================================================
        elif event == 'Load_Video':
            window['-GOTO_DIST-'].update(disabled=True, button_color=bc_disabled)
            avifile = sg.PopupGetFile('Get Setup File', no_window=True,
                        file_types=(('Video Files', '*.avi'), ('ALL Files', '*.*'),),)
            if avifile:
                filename_display_elem.update(avifile)
                pngdir = values['-TMP-']
                png_name = values['-PNG_BASE-']
                bob_doubler = values['-BOB-']
                binning = int(values['-BIN-'])
                bff = values['-BFF-']
                # check previous PNG images
                oldfiles, deleted, answer = m_fun.delete_old_files(pngdir + '/' + png_name, maxim)
                if answer != 'Cancel':
                    nim, dat_tim, sta, out = m_fun.extract_video_images(avifile, pngdir, png_name,
                        bob_doubler, binning, bff, int(values['-MAXIM-']))
                    if nim:
                        window['-PREVIOUS-'].update(disabled=False)
                        window['-NEXT-'].update(disabled=False)
                        window['-GOTO_DIST-'].update(disabled=False, button_color=bc_enabled)
                    fits_dict['DATE-OBS'] = dat_tim
                    fits_dict['M_STATIO'] = sta
                    fits_v = list(fits_dict.values())
                    for k in range(7):
                        kk = f'k{k + 7}'
                        window[kk].Update(fits_v[k])
                    if nim:
                        (opt_dict['win_width'], opt_dict['win_height']) = window.Size
                        wsx = opt_dict['win_width']
                        wsy = opt_dict['win_height']
                        setup_tab_element.set_size(size=(wsx - 40, wsy - 40))  # works
                        image = m_fun.get_png_image(out + '1.png', colorflag=True)
                        im_scale = m_fun.set_image_scale(image, opt_dict)
                        m_fun.show_image_array(image, im_scale, window['-V_IMAGE-'])
                        m_fun.show_image_array(image, im_scale, window['-D_IMAGE-'])
                        # add avifile to video_list
                        video_list = m_fun.read_video_list('videolist.txt')
                        # video_name = Path(avifile).with_suffix('').name
                        video_name, ext = path.splitext(path.basename(avifile))
                        # for UFO Capture videos, replace M by S:
                        if video_name[0] == 'M':
                            video_name = 'S' + video_name[1:]
                        for v in video_list:
                            if v in (video_name, ' '):
                                video_list.remove(v)
                        if len(video_list) >= video_list_length:
                            del video_list[-1:]
                        video_list.insert(0, video_name)
                        with open('videolist.txt', 'w') as f:
                            for v in video_list:
                                print(v, file=f)
                    logging.info(f'converted {avifile} {nim} images')
                    logging.info(f'Station = {sta} Time = {dat_tim}')
                    result_text = f'Station = {sta}\nTime = {dat_tim}\n'
                    result_text += opt_comment + f'\nNumber converted images = {str(nim)}\n'
                    result_text += f'Image Scale = {im_scale:6.3f}\n'
                    window['-RESULT2-'].update(result_text)
                    window['-TMP_D-'].update(pngdir)
                    window['-PNG_BASED-'].update(png_name)
                    window['-BOB_D-'].update(bob_doubler)
                    if bob_doubler:
                        i = 50  # jump to 1st image after background
                        n_back = 40
                        first = 50
                    else:
                        i = 25
                        n_back = 20
                        first = 25
                    nm = nim - first + 1
                    window['-N_BACK-'].update(value=str(n_back))
                    window['-N_START-'].update(value=str(first))
                    if nm < 1:
                        nm = 0
                    window['-N_IMAGE-'].update(value=str(nm))
                else:
                    result_text = 'no video converted'
                window['-RESULT-'].update(result_text)

        elif event in ('-NEXT-', '-PREVIOUS-'):
            if 1 < i < nim:
                if event == '-NEXT-':
                    i += 1
                if event == '-PREVIOUS-':
                    i -= 1
            if opt_dict['scale_win2ima'] and not t_zoom(zoom):
                im_scale = 1.0
                window['-V_IMAGE-'].update(filename=out + str(i) + '.png')
            else:
                image = m_fun.get_png_image(out + str(i) + '.png', colorflag=True)
                m_fun.show_image_array(image, im_scale, window['-V_IMAGE-'])

        if event is '-GOTO_DIST-':
            window['-T_DIST-'].select()  # works

        # ==============================================================================
        # Distortion Tab
        # ==============================================================================
        elif event in ('-TMP_D-', '-PNG_BASED-'):
            # get new number of images
            inpath = values['-TMP_D-'] + '/' + values['-PNG_BASED-']
            nmfound = m_fun.check_files(inpath, maxim, ext='.png')
            nm = nmfound - first + 1
            print('inpath, nm, nmfound', inpath, nm, nmfound)
            window['-N_IMAGE-'].update(nm)

        elif event in ('-SEL_OUT-', '-SEL_OUT_R-'):
            outpath = sg.PopupGetFolder('Select Process Folder', no_window=True)
            window['-OUT-'].update(outpath)
            window['-OUT_C-'].update(outpath)
            window['-OUT_R-'].update(outpath)

        elif event is '-APPLY_DIST-':
            window['-GOTO_REG-'].update(disabled=False, button_color=bc_disabled)
            pngdir = values['-TMP_D-']
            png_name = values['-PNG_BASED-']
            mdist = values['-M_DIST-']
            dist = values['-DIST-']
            background = values['-BACK-']
            bob_doubler = values['-BOB_D-']
            colorflag = values['-COLOR-']
            n_back = int(values['-N_BACK-'])
            first = int(values['-N_START-'])
            nm = int(values['-N_IMAGE-'])
            # check number of  tmp\*.png   <--
            inpath = pngdir + '/' + png_name
            nmfound = m_fun.check_files(inpath, maxim, ext='.png')
            if nm <= 0 or nm > nmfound - first + 1:
                sg.PopupError(
                    f'not enough meteor images, check data\n nim = {nmfound}, '
                    f'maximum processed: {str(nmfound - first + 1)}')
                nm = nmfound - first + 1
                window['-N_IMAGE-'].update(nm)
            else:
                if not path.exists(outpath):
                    os.mkdir(outpath)
                [scalxy, x00, y00, rot, disp0, a3, a5] = res_dict.values()
                if bob_doubler:  # center measured before scaling in y scalxy*2 compensates for half image height
                    scalxy *= 2.0
                    y00 /= 2.0
                # ---------------------------------------------------------------
                # check previous images mdist
                distfile = outpath + '/' + mdist
                oldfiles, deleted, answer = m_fun.delete_old_files(distfile, maxim, ext='.fit')
                disttext = f' {deleted} files deleted of {oldfiles}\n' \
                           f'start background image\nwait for background image\n'
                # ---------------------------------------------------------------
                if answer is not 'Cancel':

                    # make background image
                    inpath = pngdir + '/' + png_name
                    t0 = time.time()  # start timer
                    back = m_fun.create_background_image(inpath, n_back, colorflag)
                    # save background image as png and fit
                    # remove unnecessary fits header items before saving fits-images
                    fits_dict.pop('M_NIM', None)
                    fits_dict.pop('M_STARTI', None)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        io.imsave(outpath + '/m_back.png', np.flipud(back * 255).astype(np.uint8))
                    if debug:
                        print("process time background %8.2f sec" % (time.time() - t0))
                    m_fun.write_fits_image(back, outpath + '/m_back.fit', fits_dict)
                    disttext += f'background created of {n_back} images\n'
                    disttext += f'process time background {time.time() - t0:8.2f} sec\n'
                    window['-RESULT2-'].update(disttext)
                    window.refresh()
                    # apply distortion
                    if dist:  # with distortion, add parameters to fits-header
                        for key in res_dict.keys():
                            fkey = 'D_' + key.upper()
                            fits_dict[fkey] = np.float32(res_dict[key])
                            if key[0] == 'a':
                                logging.info(f'{key} = {res_dict[key]:9.3e}')
                            else:
                                logging.info(f'{key} = {res_dict[key]:9.3f}')
                    print('wait for processing')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        (nmp, sum_image, peak_image, disttext) = m_fun.apply_dark_distortion(inpath,
                                                                                             outpath + '/m_back', outpath, mdist, first, nm, window, dist,
                                                                                             background, (x00, y00), a3, a5, rot, scalxy, colorflag, fits_dict=fits_dict)
                    m_fun.show_image_array(peak_image, im_scale, window['-D_IMAGE-'])
                    print(nmp, ' images processed of ', nm)
                    logging.info(f'{nmp} images processed of {nm}')
                    t2 = time.time() - t0
                    if dist:
                        print(f'process time background, dark and dist {t2:8.2f} sec')
                    else:
                        print(f'process time background, dark {t2:8.2f} sec')
                    #     disttext += f'M{dat_tim}_{sta}\nprocess time background, dark {t2:8.2f} sec'
                    window['-RESULT2-'].update(result_text + disttext)
                    window['-GOTO_REG-'].update(disabled=False, button_color=bc_enabled)
                    last_file_sum = False
                else:
                    disttext = 'no files deleted'
                window['-RESULT2-'].update(disttext)
                window.refresh()

        if event is '-GOTO_REG-':
            window['-T_REG-'].select()  # works
            window['-M_DIST_R-'].update(mdist)
            window['-N_MAX_R-'].update(nmp)
            window['-R_IMAGE-'].update(filename='tmp.png')
            window['-SHOW_SUM_R-'].update(disabled=True, button_color=bc_disabled)

        # ==============================================================================
        # Registration Tab
        # ==============================================================================
        elif event is '-M_DIST_R-':
            mdist = values['-M_DIST_R-']
            infile = outpath + '/' + mdist
            nmfound = m_fun.check_files(infile, maxim, ext='.fit')
            window['-N_MAX_R-'].update(nmfound)

        # image browser---------------------------------------------------------
        elif event in ('-NEXT_R-', '-PREV_R-'):
            mdist = values['-M_DIST_R-']
            reg_file = values['-REG_BASE-']
            infile = outpath + '/' + mdist
            out_fil = outpath + '/' + reg_file
            nmfound = m_fun.check_files(infile, maxim, ext='.fit')
            nmp = int(values['-N_MAX_R-'])
            if nmfound < nmp or nmp <= 0:
                nmp = nmfound
                window['-N_MAX_R-'].update(nmp)
            last_file_sum = False
            if opt_dict['scale_win2ima'] and not t_zoom(zoom):
                im_scale = 1.0
            if i_reg < nmp and event is '-NEXT_R-':
                i_reg += 1
            if i_reg > 1 and event is '-PREV_R-':
                i_reg -= 1
                if i_reg > nmp:
                    i_reg = nmp
            if values['-SHOW_REG-']:
                nim = m_fun.check_files(out_fil, maxim, ext='.fit')
                i_reg = min(nim, i_reg)
                if i_reg <= nim and path.exists(out_fil + str(i_reg) + '.fit'):
                    m_fun.show_fits_image(out_fil + str(i_reg), im_scale, window['-R_IMAGE-'], contr=contrast)
                    window['-INDEX_R-'].update(reg_file + str(i_reg))
                elif nim > 0:
                    i_reg -= 1
            else:
                if path.exists(infile + str(i_reg) + '.fit'):
                    m_fun.show_fits_image(infile + str(i_reg), im_scale, window['-R_IMAGE-'], contr=contrast)
                    window['-INDEX_R-'].update(mdist + str(i_reg))
                else:
                    sg.PopupError(f'File {infile + str(i_reg)}.fit not found')
            result_text = ('Station = ' + fits_dict['M_STATIO'] + '\n'
                       + 'Time = ' + fits_dict['DATE-OBS'] + '\n'
                       + opt_comment + '\n'
                       + 'Number Images = ' + str(nim) + '\n'
                       + 'Image Scale = ' + str('%6.3f' % im_scale) + '\n')

        # image contrast--------------------------------------------------------
        elif event in ('-LOW_C-', '-HIGH_C-'):
            mdist = values['-M_DIST_R-']
            infile = path.join(outpath, mdist)
            if event is '-LOW_C-':
                contrast = 0.5 * contrast
            else:
                contrast = 2.0 * contrast
            if last_file_sum:
                m_fun.show_fits_image(outfile, im_scale, window['-R_IMAGE-'], contr=contrast)
            else:
                if values['-SHOW_REG-']:
                    if path.exists(out_fil + str(i_reg) + '.fit'):
                        m_fun.show_fits_image(out_fil + str(i_reg), im_scale, window['-R_IMAGE-'], contr=contrast)
                else:
                    m_fun.show_fits_image(infile + str(i_reg), im_scale, window['-R_IMAGE-'], contr=contrast)

        # image selection-------------------------------------------------------
        elif event is '-SHOW_SUM_R-':
            m_fun.show_fits_image(outfile, im_scale, window['-R_IMAGE-'], contr=contrast)
            last_file_sum = True

        elif event is '-SEL_START-':
            start = i_reg
            window['-N_START_R-'].update(start)

        elif event is '-SEL_LAST-':
            nsel = i_reg
            start = int(values['-N_START_R-'])
            nim = nsel - start + 1
            window['-N_REG-'].update(nim)

        if event is '-GOTO_CAL-':
            window['-T_CAL-'].select()  # works
            window['-CALI-'].update(disabled=True, button_color=bc_disabled)
            window['-SHOW_REG-'].update(False)
        # ==============================================================================
        # Registration Tab
        # ==============================================================================
        elif event is '-REGISTER-':
            window['-SHOW_SUM_R-'].update(disabled=True, button_color=bc_disabled)
            window['-CAL_R-'].update(disabled=True, button_color=bc_disabled)
            window['-SHOW_REG-'].update(False)
            mdist = values['-M_DIST_R-']
            infile = path.join(outpath, mdist)
            reg_file = values['-REG_BASE-']
            start = int(values['-N_START_R-'])
            nim = int(values['-N_REG-'])
            nmp = int(values['-N_MAX_R-'])
            out_fil = path.join(outpath, reg_file)
            im, header = m_fun.get_fits_image(infile + str(start))
            (opt_dict['win_width'], opt_dict['win_height']) = window.Size
            im_scale = m_fun.set_image_scale(im, opt_dict)
            m_fun.show_fits_image(infile + str(start), im_scale, window['-R_IMAGE-'], contrast)
            if not sta:
                sta = header['M_STATIO']
                dat_tim = header['DATE-OBS']
            # ===================================================================
            # select rectangle for registration
            select_event, x0, y0, dx, dy = m_fun.select_rectangle(infile, start, par_dict,
                                                                  res_dict, fits_dict, (wlocx, wlocy), out_fil, maxim)
            if select_event == 'Ok':
                nsel = start + nim - 1  # nsel index of last selected image, nim number of images
                if nsel > nmp:
                    nim = max(nmp - start + 1, 0)
                    window['-N_REG-'].update(nim)
                t0 = time.time()
                index, sum_image, reg_text, dist, outfile, fits_dict = m_fun.register_images(start, nim, x0,
                                                                                  y0, dx, dy, infile, out_fil, window, fits_dict=fits_dict)
                t3 = time.time() - t0
                nim = index - start + 1
                if nim > 1:
                    print(f'time for register one image : {t3 / nim:6.2f} sec')
                    logging.info(f'time for register one image : {t3 / nim:6.2f} sec')
                    result_text = (f'Station = {sta}\nTime = {dat_tim}\n'
                               + opt_comment + f'\nStart image = {str(start)}\n'
                               + f'Number registered images: {nim}\nof total images: {nmp}\n'
                               + f'time for register one image: {t3 / nim:6.2f} sec\n'
                               + 'Image Scale = ' + str('%6.3f' % im_scale) + '\n')
                    m_fun.show_fits_image(outfile, im_scale, window['-R_IMAGE-'])
                    window['-SHOW_REG-'].update(True)
                    window['-RADD-'].update(outfile)
                    window['-SHOW_SUM_R-'].update(disabled=False, button_color=bc_enabled)
                    window['-ADD_ROWS-'].update(disabled=False, button_color=bc_enabled)
                else:
                    result_text = (f'Number registered images: {nim}\n'
                               + f'of total images: {nmp}\nnot enough images\n')
                    sg.PopupError(f'register did not work with last image, try again!')
                    logging.info(f'register did not work with last image, try again!')
                window['-RESULT3-'].update(reg_text + result_text)

        # =======================================================================
        # convert 2-D spectrum to 1-D spectrum
        elif event is '-ADD_ROWS-':
            if outfile:
                m_fun.show_fits_image(outfile, im_scale, window['-R_IMAGE-'], contrast)
                event, tilt, slant = m_fun.add_rows_apply_tilt_slant(outfile, par_dict,
                    res_dict, fits_dict, im_scale, contrast, (wlocx, wlocy), result_text, reg_text, window)
                m_fun.show_fits_image(outfile + 'st', im_scale, window['-R_IMAGE-'], contrast)

        # =======================================================================
        elif event is '-LOAD_R-':
            # load existing file for adding rows and apply tilt and slant
            result_text = ''
            window['-SAVE_RAW-'].update(disabled=True, button_color=bc_disabled)
            window['-CAL_R-'].update(disabled=True, button_color=bc_disabled)
            outfile = sg.PopupGetFile('Get Registered File', no_window=True,
                                      file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),),)
            if outfile:
                # remove fits header items not present in mdist files, load actual values below
                try:
                    fits_dict['M_NIM'] = '1'
                    fits_dict['M_STARTI'] = '1'
                except KeyError:
                    print('M_STARTI not in fits_dict')
                outfile, ext = path.splitext(outfile)
                window['-RADD-'].update(outfile)
                last_file_sum = True
                im, header = m_fun.get_fits_image(outfile)  # for im_scale
                (opt_dict['win_width'], opt_dict['win_height']) = window.Size
                im_scale = m_fun.set_image_scale(im, opt_dict)
                m_fun.show_fits_image(outfile, im_scale, window['-R_IMAGE-'], contrast)
                if 'D_X00' in header.keys():
                    dist = True
                else:
                    dist = False
                m_fun.get_fits_keys(header, fits_dict, res_dict, keyprint=debug)
                window['-RADD-'].update(outfile)
                window['-ADD_ROWS-'].update(disabled=False, button_color=bc_enabled)
                window['-SHOW_SUM_R-'].update(disabled=False, button_color=bc_enabled)
                result_text = ('Load File: ' + outfile + '\n'
                           + 'Station = ' + str(fits_dict['M_STATIO']) + '\n'
                           + 'Time = ' + str(fits_dict['DATE-OBS']) + '\n'
                           + 'comment = ' + str(fits_dict['COMMENT']) + '\n')
                try:
                    result_text += 'Start image = ' + str(fits_dict['M_STARTI']) + '\n'
                    result_text += f'Number registered images: {str(fits_dict["M_NIM"])}\n'
                except KeyError:
                    print('M_STARTI not in fits_dict')
                result_text += 'Image Scale = ' + str('%6.3f' % im_scale) + '\n'
                window['-RESULT3-'].update(reg_text + result_text)
            else:
                window['-ADD_ROWS-'].update(disabled=True, button_color=bc_disabled)

        elif event is '-SAVE_RAW-':
            imtilt, header = m_fun.get_fits_image(outfile + 'st')
            lcal, ical = np.loadtxt(outfile + '.dat', unpack=True, ndmin=2)
            # default values for fits_dict:
            fits_dict['M_TILT'] = 0.0
            fits_dict['M_SLANT'] = 0.0
            fits_dict['M_ROWMIN'] = 0
            fits_dict['M_ROWMAX'] = 0
            # update fits_dict
            m_fun.get_fits_keys(header, fits_dict, res_dict, keyprint=debug)
            new_outfile = sg.PopupGetFile('Save image and raw spectrum as',
                          no_window=True, save_as=True, file_types=(('Spectrum Files', '*.dat'),
                                                                    ('ALL Files', '*.*')))
            if new_outfile:
                # outfile = str(Path(new_outfile).with_suffix(''))
                outfile, ext = path.splitext(new_outfile)
                window['-RADD-'].update(outfile)
                m_fun.write_fits_image(imtilt, outfile + '.fit', fits_dict, dist=dist)
                m_fun.show_fits_image(outfile, im_scale, window['-R_IMAGE-'], contrast)
                np.savetxt(outfile + '.dat', np.transpose([lcal, ical]), fmt='%6i %8.5f')
                result_text += 'File saved as :' + outfile + '.dat .fit\n'
                window['-RESULT3-'].update(reg_text + result_text)

        # =======================================================================
        # load uncalibrated raw spectrum
        elif event in ('-LOAD_RAW-', '-CAL_R-'):
            if event is '-CAL_R-':
                # start calibration
                raw_spec_file = m_fun.change_extension(values['-RADD-'], '.dat')
                window['-T_CAL-'].select()  # works
            else:
                raw_spec_file = sg.PopupGetFile('Load raw spectrum',
                                          no_window=True, save_as=False,
                                          file_types=(('Spectrum Files', '*.dat'),
                                                      ('ALL Files', '*.*'),))
            window['-S_LINES-'].update(disabled=True, button_color=bc_disabled)
            window['-SPEC_R-'].update(raw_spec_file)
            if raw_spec_file:
                m_fun.create_line_list_combo(line_list, window)
                result_text += f'File {raw_spec_file} loaded\n'
                graph = window['graph']
                # plot raw spectrum
                lmin, lmax, i_min, i_max, lcal, ical = m_fun.plot_raw_spectrum(raw_spec_file, graph, canvasx)
                window['-S_LINES-'].update(disabled=False, button_color=bc_enabled)
                window['-LOAD_TABLE-'].update(disabled=False, button_color=bc_enabled)
                llist = m_fun.change_extension(raw_spec_file, '.txt')
                graph_enabled = True
                if lmin not in (0.0, 1.0):
                    sg.PopupError('raw files only, load uncalibrated file or Load Spectrum',
                                  title='Wavelength calibration', line_width=60)

        # ==============================================================================
        # select spectral lines for calibration
        elif event is '-S_LINES-':
            table = []
            select_line_enabled = True
            table_edited = False
            dragging = False
            start_point = end_point = prior_rect = None
            cal_text_file = ' Pixel    width  lambda    fit    delta\n'

        elif event is 'graph' and graph_enabled:  # if there's a "Graph" event, then it's a mouse
            if select_line_enabled:
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
                    prior_rect = graph.draw_rectangle((xmin, i_min),
                                                      (xmax, i_max), line_color='LightGreen')
        elif str(event).endswith('+UP') and graph_enabled:
            if select_line_enabled:
                # The drawing has ended because mouse up
                x0 = 0.5 * (start_point[0] + end_point[0])
                w = abs(0.5 * (start_point[0] - end_point[0]))
                window['-POS-'].update(f'{x0:7.1f}')
                window['-LAMBDA-'].update(disabled=False)
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False

        # ==============================================================================
        # select single calibration line
        elif event is '-S_LINE-':
            # set focus? to lambda if wavelength entered ok
            window['-S_LINE-'].update(disabled=True, button_color=bc_disabled)
            window['-LAMBDA-'].update(disabled=True)
            (l0, name) = values['-LAMBDA-'].split(' ', 1)
            lam = float(l0)
            print(f'x0, lambda, w:  {x0:8.2f} {lam} {w:6.2f}  {name}')
            x0, fwp, cal_text_file = m_fun.select_calibration_line(x0, w, lam, name, lcal, ical, graph, table, cal_text_file)
            result_text += cal_text_file
            window['-RESULT4-'].update(result_text, disabled=True)

        # ==============================================================================
        # select calibration wavelength from list
        elif event is '-LAMBDA-':
            window['-S_LINE-'].update(disabled=False, button_color=bc_enabled)

        # ==============================================================================
        # save calibrated spectrum
        elif event is '-SAVE_T-':
            if table_edited:
                # table in result window has been edited, save new values
                result_text = values['-RESULT4-']
                with open(llist, 'w') as f:
                    f.write(result_text)
                ev = sg.PopupOKCancel(f'table saved as {llist}')
                if ev in 'OK':
                    xcalib, lcalib = np.loadtxt(llist, unpack=True, ndmin=2)
            elif table and raw_spec_file:
                llist = m_fun.change_extension(raw_spec_file, '.txt')
                with open(llist, 'w+') as f:
                    np.savetxt(f, table, fmt='%8.2f', header='    x     lambda')
                xcalib, lcalib = np.loadtxt(llist, unpack=True, ndmin=2)
                print(f'table \n {table} \nsaved as {outfile}.txt, \n'
                      f'edit if necessary before proceeding with calibration')
                window['-CALI-'].update(disabled=False, button_color=bc_enabled)
                graph_enabled = False
                select_line_enabled = False
            else:
                sg.PopupError('no table to save, select lines first')

        # ==============================================================================
        # load table, enable editing table
        elif event is '-LOAD_TABLE-':
            llist = sg.PopupGetFile('Load calibration table',
                                    no_window=True, save_as=False,
                                    file_types=(('Calibration Files', '*.txt'), ('ALL Files', '*.*'),),
                                    default_path=llist)
            if llist:
                window['-CALI-'].update(disabled=False, button_color=bc_enabled)
                llist = m_fun.change_extension(llist, '.txt')
                xcalib, lcalib = np.loadtxt(llist, unpack=True, ndmin=2)
                # table is displayed in result window and made editable
                with open(llist, 'r') as f:
                    result_text = f.read()
                window['-RESULT4-'].update(result_text, disabled=False)
                table_edited = True

        # ==============================================================================
        # calculate calibration polynom, calibrate aw spectrum
        elif event is '-CALI-':
            deg = int(values['-POLY-'])
            if deg >= 1:
                if len(xcalib) <= deg:
                    sg.PopupError(f'not enough data points for polynomial degree {deg}, chooose lower value',
                                  title='Polynom degree')
                else:
                    c = np.polyfit(xcalib, lcalib, deg)  # do the polynomial fit (order=2)
            else:  # use disp0 for conversion to wavelength
                deg = 1
                disp0 = fits_dict['D_DISP0']
                disp = float(sg.PopupGetText('select value for disp0:',
                                             title='linear dispersion [nm/Pixel]', default_text=str(disp0)))
                c = [disp, lcalib[0] - disp * xcalib[0]]
            if len(c):
                cal_dat_file, spec_file, lmin, lmax, cal_text_file = m_fun.calibrate_raw_spectrum(raw_spec_file, xcalib, lcalib, deg, c)
                logging.info(f'spectrum {spec_file} saved')
                window['-RESULT4-'].update(result_text + cal_text_file, disabled=True)
                window['-PLOT_SPEC-'].update(spec_file)
                window['-PLOTS-'].update(disabled=False, button_color=bc_enabled)
                window['l_min'].update(str(int(lmin)))
                window['l_max'].update(str(int(lmax)))
                video_list = m_fun.read_video_list('videolist.txt')
                video_list.insert(0, spec_file)
                video_list.insert(0, ' ')
                window['-PLOT_TITLE-'].update(values=video_list)

        # ==============================================================================
        # load (un)calibrated spectrum
        if event is '-LOADS-':
            window['-S_LINES-'].update(disabled=True, button_color=bc_disabled)
            spec_file = sg.PopupGetFile('Load spectrum',
                                   no_window=True, save_as=False,
                                   file_types=(('Spectrum Files', '*.dat'),
                                               ('ALL Files', '*.*'),))
            if spec_file:
                result_text += f'File {spec_file} loaded\n'
                lspec, ispec = np.loadtxt(spec_file, unpack=True, ndmin=2)
                logging.info(f'spectrum {spec_file} loaded')
                lmin = lspec[0]
                lmax = lspec[len(lspec) - 1]
                video_list = m_fun.read_video_list('videolist.txt')
                video_list.insert(0, spec_file)
                video_list.insert(0, ' ')
                window['-PLOT_SPEC-'].update(spec_file)
                window['-PLOT_TITLE-'].update(values=video_list)
                window['-PLOTS-'].update(disabled=False, button_color=bc_enabled)
                window['l_min'].update(str(int(lmin)))
                window['l_max'].update(str(int(lmax)))
                window['-RESULT4-'].update(result_text)

        # ==============================================================================
        # plot spectrum
        if event is '-PLOTS-' and spec_file:
            window['-PLOTS-'].update(disabled=True, button_color=bc_disabled)
            gridlines = values['-GRID-']
            autoscale = values['-AUTO_SCALE-']
            if values['-NORM_SCALE-']:
                autoscale = False
                i_min = -0.1
                i_max = 1.1
            try:
                lmin = float(values['l_min'])
                lmax = float(values['l_max'])
                plot_w = int(values['plot_w'])
                plot_h = int(values['plot_h'])
            except:
                sg.PopupError('bad value for plot range, try again', title='Plot Graph')
            plot_title = values['-PLOT_TITLE-']
            mod_file, i_min, i_max, cal_text_file = m_fun.graph_calibrated_spectrum(spec_file, lmin=lmin, lmax=lmax,
                                     imin=i_min, imax=i_max, autoscale=autoscale, gridlines=gridlines,
                                     canvas_size=(plot_w, plot_h), plot_title=plot_title)
            window['-PLOTS-'].update(disabled=False, button_color=bc_enabled)
            result_text += cal_text_file
            if mod_file:
                window['-PLOT_SPEC-'].update(mod_file)
                spec_file = mod_file
            window['-RESULT4-'].update(result_text)

        # other stuff, open issues ---------------------------------------------
        else:
            (wlocx, wlocy) = window.current_location()
            opt_dict['win_x'] = wlocx
            opt_dict['win_y'] = wlocy
            # check change of tabs
            # print(event, tabs_element.get())
            if tabs_element.get() is '-T_REG-':
                window['-M_DIST_R-'].update(mdist)
                window['-N_REG-'].update(nim)
                window['-N_MAX_R-'].update(nmp)
                window['-R_IMAGE-'].update(filename='tmp.png')
                # window['-SHOW_REG-'].update(False)
                out_fil = outpath + '/' + reg_file
                # window['-SHOW_SUM_R-'].update(disabled=True, button_color=bc_disabled)
            # if tabs_element.get() == '-T_DIST-'
            # unused code, maybe use later
            # tabs_element.set_focus('Tab 1') #does not work
            # print('size = ',image_element_video.get_size(), setup_tab_element.get_size(),
            # window.Size) #works only first time
    window.close()
    # end of main()

if __name__ == '__main__':
    main()
