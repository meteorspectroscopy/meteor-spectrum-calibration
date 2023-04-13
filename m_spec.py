# -------------------------------------------------------------------
# m_spec, processing of meteor spectra
# Author: Martin Dubs, 2020
# -------------------------------------------------------------------
import logging
import os
import os.path as path
import time
from pathlib import Path

import numpy as np
from skimage import io as ios
from skimage.filters import gaussian
from scipy import interpolate

import PySimpleGUI as sg
import m_calfun as lfun
import m_plot
import m_specfun as m_fun
import myselect as sel
import m_image_tools as m_image
import nist
try:
    from csaps import csaps
    csaps_installed = True
except Exception as ex:
    sg.PopupError(f'missing module\ninstall csaps, if you want to use response function\n'
                  f'or continue without response\n{ex}', no_titlebar=True, keep_on_top=True)
    csaps_installed = False


# -------------------------------------------------------------------
# main program
# -------------------------------------------------------------------
def main():
    # start with default inifile, if inifile not found, a default configuration is loaded
    version = '0.9.29'
    # GUI settings:
    bc_enabled = ('white', 'green')
    bc_disabled = (None, 'darkblue')
    sg.SetGlobalIcon('Koji.ico')
    sg.DEFAULT_TOOLTIP_OFFSET = (0, -50)     
    # Default offset (dx, dy) relative to the position when mouse enter the element
    sg_ver = sg.version.split(' ')[0]
    print('PySimpleGUI', sg_ver)
    if int(sg_ver.split('.')[0]) >= 4 and int(sg_ver.split('.')[1]) >= 9:
        sg.change_look_and_feel('SystemDefault')  # suppresses message in PySimpleGUI >= 4.9.0
    print('version m_spec, m_specfun, m_plot', version, m_fun.version, m_plot.version)
    logging.info('M_SPEC START +++++++++++++++++++++++++++++++++++++++++++++')
    info = f'M_SPEC version {version}, lfun version {lfun.version}, m_fun.version {m_fun.version}, ' \
           f'PySimpleGUI {sg_ver}'
    logging.info(info)
    ini_file = 'm_set.ini'
    par_text, par_dict, res_dict, fits_dict, opt_dict = m_fun.read_configuration(ini_file,
                                            m_fun.par_dict, m_fun.res_dict, m_fun.opt_dict)
    res_key = list(res_dict.keys())
    res_v = list(res_dict.values())
    fits_dict['VERSION'] = version
    fits_v = list(fits_dict.values())
    fits_k = list(fits_dict.keys())
    [zoom, wsx, wsy, wlocx, wlocy, xoff_calc, yoff_calc, xoff_setup, yoff_setup,
        debug, fit_report, win2ima, opt_comment, png_name, outpath, mdist, colorflag,
        bob_doubler, plot_w, plot_h, i_min, i_max, graph_size,
        show_images, meteor_lines, flat_flag, flat_file] = list(opt_dict.values())
    if par_text == '':
        sg.PopupError(f'no valid configuration found, default {ini_file} created')
    # default values for video
    par_text = info + '\n\n' + par_text
    maxim = 200
    result_text = ''
    video_list = m_fun.read_video_list('videolist.txt')
    # default values for distortion
    n_back = 20
    dat_tim = ''
    sta = ''
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
    spectrum_files = (('Spectrum Files', '*.dat'), ('ALL Files', '*.*'),)
    # default values for calibration
    _image = np.flipud(ios.imread('tmp.png'))  # get shape of screen image
    (canvasy, canvasx) = _image.shape[:2]
    raw_spec_file = ''
    select_line_enabled = False
    table = []
    spec_file = ''
    llist = ''
    cal_dat_file = ''
    c = []
    graph_enabled = False
    line_list = 'm_linelist'
    table_edited = False
    graph_s2 = (graph_size, graph_size)
    idg = None
    # new from m_calib
    pngdir = '_tmp_/cal_'
    outfil = par_dict['s_outfil']
    notok, linelist, lines = lfun.get_linelist(par_dict['s_linelist'], par_dict['f_lam0'])
    if linelist:
        par_dict['s_linelist'] = linelist
    infile = par_dict['s_infile']
    imbw = []
    logtext = ''
    # initial values for response
    old_cal_tab = False
    old_ir_tab = False
    lcal = lref = []
    lmin = 350
    lmax = 1000
    i_min = -.1
    i_max = 2
    graph_ir_enabled = False
    autoscale = True
    l_spec = l_smooth = l_resp = []
    smooth_parameter = 0.01
    response_ok = False
    response_folder = 'response'
    # plot_style=(color, circle_size, line_size, offset):
    star_style = ('red', 1, 1, -0.05)
    ref_style = ('blue', 0, 2, -.1)
    raw_style = ('green', 2, 0, -.15)
    response_style = ('darkred', 0, 2, -0.2)
    trans_style = ('lightblue', 0, 2, -0.15)
    flux_style = ref_style
    flux_file = ''
    sigma = 2.0
    tool_type = 'reference'
    idg_ref = []
    idg_resp = []
    idg_smooth = []
    idg_trans = []
    flux_flag = False
    elevation_deg = 90.0
    h = 0.0
    aod = 0.1
    trans_flag = True
    resp_flag = False
    # spectrum_analysis, initial and default values
    spec_file_analysis = 'analysis/'
    resp_file_analysis = 'response/'
    graph_analysis_enabled = False
    idg_spec = []
    idg_resp_a = []
    idg_lines = []
    idg_res = []
    idg_fit = []
    lresp = []
    iresp = []
    iclip = []
    i_fit = []
    i_residue = []
    t_cont = 3000
    t_el = 3000
    threshold = 1.0e-4
    sigma_nist = 1
    sigma0 = 0.7
    sigma_fit = 0
    delta = 1
    dragging = False
    start_point = end_point = prior_rect = None
    zoom_window = []
    t_n2i = 'NIST/N2I4000'
    # sel_elements = []
    show_residuals = False
    residual_offset = 0.3
    p0 = (0, 0)  # used for padding
    p2 = (2, 0)
    # result_a = ''
    fit_enabled = False
    result_fit = ''
    # -------------------------------------------------------------------
    element_names = ['Cont', 'FeI', 'MgI', 'NaI', 'CaI', 'CaII', 'CrI', 'MnI', 'OI', 'NI', 'N2I']
    elements = []
    for ele in element_names:
        # needed, because config uses all lower case variables
        elements.append(ele.lower())
    elements_dict = dict(list(zip(elements, element_names)))
    ele_color = {'cont': 'Purple', 'fei': 'red', 'mgi': 'green', 'nai': 'orange',
                 'cai': 'Chartreuse', 'caii': 'purple', 'cri': 'yellow', 'mni': 'magenta',
                 'oi': 'red', 'ni': 'cyan', 'n2i': 'magenta'}
    # "white", "black", "red", "green", "blue", "cyan", "yellow", and "magenta" 'Purple', 'Chartreuse'
    # -------------------------------------------------------------------
    # introduce class nist.Element
    all_ele = []
    for ele in elements:
        all_ele.append(nist.Element(ele, 1.0, 1.0, 0, True, color=ele_color[ele]))
    for ele in all_ele:
        print(ele.name, ele.mult, ele.scale, ele.color, ele.index, ele.fit)
        if ele.name == 'fei':
            ele.range_low = 523.0
            ele.range_high = 550.0
        elif ele.name == 'mgi':
            ele.range_low = 512.0
            ele.range_high = 523.0
        elif ele.name == 'nai':
            ele.range_low = 584.0
            ele.range_high = 594.0

    # initialize configuration for spectrum fit
    lmin_a = 350
    lmax_a = 900
    lsqf_keys = ['spectrum', 'response', 'lmin', 'lmax', 'sigma', 'sigma0', 't_cont', 't_el', 'threshold', 't_n2i']
    lsqf_var = ['', '', lmin, lmax, sigma_nist, sigma0, t_cont, t_el, threshold, t_n2i]
    lsqf_dict = dict(list(zip(lsqf_keys, lsqf_var)))
    rms = ''
    # -------------------------------------------------------------------
    # definition of GUI
    # -------------------------------------------------------------------

    # setup tab---------------------------------------------------------------------
    result_elem = sg.Multiline(par_text, size=(50, 30), disabled=True, autoscroll=True)
    setup_file_display_elem = sg.InputText('m_set.ini', size=(60, 1), key='-INI_FILE-')

    menu_def = [
        ['&File', ['Save calibration &Image', 'E&xit']],
        ['&View', ['&Fits-Header']],
        ['&Tools', ['!&Offset', 'Edit &Text File', 'Edit &Log File', 'Edit &Spectrum',
         '&Add Images', '!&Gaussian, wavelength tools', '!Atmospheric correction', 'Image Tools']],
        ['&Options', ['&Setup Options', '&Meteor lines']],
        ['&Help', '&About...'], ]

    setup_file_element = sg.Frame('Configuration File',
                                  [[sg.Text('File'), setup_file_display_elem,
                                    sg.Button('Load Setup', key='-LOAD_SETUP-'),
                                    sg.Button('Save Setup', key='-SAVE_SETUP-'),
                                    sg.Button('Save Default', key='-SAVE_DEFAULT-',
                                              tooltip='Save m_set.ini as default')]])
    row_layout = [[sg.Text('Distortion')]]
    for k in range(7):
        kk = f'k{k}'
        row_layout += [sg.Text(res_key[k], size=(10, 1)),
                       sg.Input(res_v[k], size=(45, 1), key=kk, tooltip='Distortion parameters')],
    for k in range(7):
        kk = f'k{k + 7}'
        row_layout += [sg.Text(list(fits_dict.keys())[k], size=(10, 1)),
                       sg.Input(fits_v[k], size=(45, 1), key=kk, tooltip='Fits-header default values')],

    # Parameters
    layout_setup = [[sg.Frame('Settings', [[setup_file_element], [result_elem,
                    sg.Frame('Setup Parameters', row_layout, tooltip='Edit parameters')]])]]

    # Video tab---------------------------------------------------------------------
    image_element_video = sg.Graph(canvas_size=graph_s2, graph_bottom_left=(0, 0),
                                   graph_top_right=graph_s2, key='-V_IMAGE-')
    filename_display_elem = sg.InputText('', size=(60, 1), key='-VIDEO-')
    image_options_element = [[sg.Text('Temporary Image Folder')],
                             [sg.Text('PNG Image Base:'),
                              sg.InputText(png_name, size=(25, 1), key='-PNG_BASE-')],
                             [sg.Checkbox('Bob Doubler', default=False, pad=(10, 0), key='-BOB-',
                                          tooltip='if checked, create images from fields for interlaced video')],
                             [sg.Checkbox('Bottom Field First', default=True, pad=(10, 0), key='-BFF-')],
                             [sg.Combo([1, 2, 3, 4], key='-BIN-', enable_events=True,
                                       default_value=par_dict['i_binning'],
                                       tooltip='reduce image size for faster processing'),
                              sg.Text(' Binning')],
                             [sg.Text('Max number of images:'),
                              sg.InputText(str(maxim), size=(8, 1), key='-MAXIM-',
                                           tooltip='limit output to number of images')],
                             [sg.Text('_' * 44)], [sg.Text('Results')],
                             [sg.Multiline('', size=(42, 20), disabled=True,
                                           key='-RESULT-', autoscroll=True)]]
    sub_frame_element = sg.Frame('Video File', [[sg.Text('File'),
                                                filename_display_elem, sg.Button('Load_Video',
                                                file_types=(('AVI-File', '*.avi'), ('ALL Files', '*.*')),
                                                tooltip='Load video and convert to images'),
                                                sg.Button('Previous', key='-PREVIOUS-', disabled=True),
                                                sg.Button('Next', key='-NEXT-', disabled=True),
                                                sg.Button('Continue', key='-GOTO_DIST-', disabled=True,
                                                          button_color=bc_disabled, bind_return_key=False,
                                                          tooltip='Go to next tab for applying distortion')]])
    video_options_element = sg.Frame('Options', image_options_element)

    # Distortion tab--------------------------------------------------------------------
    dist_elem = sg.Frame('Parameters',
                    [[sg.Text('temporary image folder')],
                     [sg.InputText(png_name, size=(26, 1), key='-PNG_BASED-', enable_events=True,
                                   tooltip='Path and filebase of extracted images, e.g. "tmp/m_"')],
                    [sg.Text('Process folder')],
                    [sg.InputText(outpath, size=(34, 1), key='-OUT-', disabled=False),
                     sg.Button('Select', key='-SEL_OUT-', tooltip='select process folder for output')],
                    [sg.Text('Distorted Image Base:'),
                     sg.InputText(mdist, size=(22, 1), key='-M_DIST-')],
                    [sg.Checkbox('Apply distortion', default=True, pad=(10, 0), key='-DIST-')],
                    [sg.Checkbox('Background subtraction', default=True,
                                 pad=(10, 0), key='-BACK-')],
                    [sg.Checkbox('Color processing', default=False,
                                 pad=(10, 0), key='-COLOR-')],
                    [sg.Checkbox('Bob Doubler', default=False, pad=(10, 0), key='-BOB_D-')],
                    [sg.Checkbox('Flat corr.', default=flat_flag, pad=(10, 0), key='-FLAT_C-'),
                     sg.InputText(flat_file, size=(20, 1), key='-FLAT-', enable_events=True,
                                  tooltip='Filename of flat if Flat corr. checked'),
                     sg.Button('Sel Flat', key='-SEL_FLAT-', tooltip='select flat file')],
                     [sg.Text('Number of background images:'),
                     sg.InputText(str(n_back), size=(15, 1), key='-N_BACK-')],
                    [sg.Text('Index of start image:'),
                     sg.InputText(str(first), size=(24, 1), key='-N_START-')],
                    [sg.Text('Number of distorted images:'),
                     sg.InputText(str(nm), size=(17, 1), key='-N_IMAGE-')], [sg.Text('_' * 34)],
                    [sg.Button('Apply Dist.', key='-APPLY_DIST-', tooltip='background subtraction and distortion'),
                     sg.Button('Stop', key='-STOP-', tooltip='finish processing', button_color=('white', 'red')),
                     sg.Checkbox('Show Images', default=show_images, key='-SHOW_IM-'),
                     sg.Button('Continue', key='-GOTO_REG-', disabled=True,
                               tooltip='go to registration tab for next processing step')],
                    [sg.Text('Results')], [sg.Multiline('Result', size=(42, 8), disabled=True,
                                                        key='-RESULT2-', autoscroll=True)]])
    image_element_distortion = sg.Graph(canvas_size=graph_s2, graph_bottom_left=(0, 0),
                                        graph_top_right=graph_s2, key='-D_IMAGE-')

    # Registration tab--------------------------------------------------------------------
    image_element_registration = sg.Graph(canvas_size=graph_s2, graph_bottom_left=(0, 0),
                                          graph_top_right=graph_s2, key='-R_IMAGE-')
    register_elem = [[sg.Frame('Registration', [
        [sg.Text('Process folder'),
         sg.InputText(outpath, size=(30, 1), key='-OUT_R-', disabled=False),
         sg.Button('Select', key='-SEL_OUT_R-', tooltip='Select process folder'),
         sg.Button('Previous', key='-PREV_R-'),
         sg.Button('Next', key='-NEXT_R-'), sg.Text('Current Image:'),
         sg.InputText(mdist + str(i_reg), size=(20, 1), key='-INDEX_R-', disabled=True),
         sg.Text('Max Images:'),
         sg.InputText(str(nm), size=(4, 1), key='-N_MAX_R-', tooltip='Limit number of images to register, \n' +
                      'if register fails, limit is set automatically'),
         sg.Button('Darker', key='-LOW_C-'), sg.Button('Brighter', key='-HIGH_C-')]])],
        [sg.Frame('Parameters', [[sg.Text('Distorted Image Base:'),
                     sg.InputText('mdist', size=(24, 1), key='-M_DIST_R-', enable_events=True)],
                    [sg.Text('Registered Image Base:'),
                     sg.InputText(reg_file, size=(22, 1), key='-REG_BASE-')],
                    [sg.Text('Index of start image:'),
                     sg.InputText('1', size=(25, 1), key='-N_START_R-')],
                    [sg.Text('Number of registered images:'),
                     sg.InputText(str(nm), size=(18, 1), key='-N_REG-',
                                  tooltip='Limit number of images to register, \n' +
                                  'if register fails, limit is set automatically')],
                    [sg.Text('_' * 44)],
                    [sg.Button('Sel Start', key='-SEL_START-', tooltip='set actual image as start image'),
                     sg.Button('Sel Last', key='-SEL_LAST-', tooltip='set actual image as last image', size=(12, 1)),
                     sg.Checkbox('Gaussian', default=False, pad=(10, 0), key='-GAUSSIAN-',
                                 tooltip='use Gaussian fit, default is cross correlation method')],
                    [sg.Button('Register', key='-REGISTER-'),
                     sg.Button('Show Sum', key='-SHOW_SUM_R-', disabled=True, size=(12, 1)),
                     sg.Checkbox('show registered', default=False, pad=(10, 0), key='-SHOW_REG-')],
                    [sg.InputText('r_add', size=(32, 1), key='-RADD-', tooltip='File for spectrum extraction'),
                     sg.Button('Load Radd', key='-LOAD_R-', tooltip='Load file for spectrum extraction')],
                    [sg.Button('Add Rows', disabled=True, key='-ADD_ROWS-', tooltip='Convert to 1d spectrum *.dat'),
                     sg.Button('Save raw spectrum', disabled=True, key='-SAVE_RAW-'),
                     sg.Button('Calibrate', disabled=True, key='-CAL_R-', tooltip='Continue with calibration')],
                    [sg.Text('Results')],
                    [sg.Multiline('Result', size=(42, 15), disabled=True, key='-RESULT3-',
                                  autoscroll=True)]]), image_element_registration]]

    # Calibration tab--------------------------------------------------------------------
    column = [[sg.Graph(canvas_size=(canvasx, canvasy), graph_bottom_left=(0.0, 0.0),
                        graph_top_right=(1.0, 1.0), background_color='white', key='graph',
                        enable_events=True, drag_submits=True, float_values=True,
                        tooltip='Uncalibrated (raw) spectrum, select calibration lines with mouse draw')], ]
    plot_elem = [sg.Frame('Plot Spectrum',
                          [[sg.InputText(cal_dat_file, size=(40, 1), key='-PLOT_SPEC-')],
                           [sg.Button('Load Spectrum', key='-LOADS-'),
                            sg.Button('Plot Spectrum', key='-PLOTS-', disabled=True,
                                      button_color=bc_disabled, tooltip='Plot calibrated spectrum'),
                            sg.Button('Save Spectrum', key='-SAVES-', disabled=True,
                                      button_color=bc_disabled, tooltip='Plot calibrated spectrum')],
                           [sg.Checkbox('Grid lines', default=True, key='-GRID-'),
                            sg.Checkbox('Auto scale', default=False, key='-AUTO_SCALE-'),
                            sg.Checkbox('Norm scale', default=False, key='-NORM_SCALE-')],
                           [sg.T('lambda min:'), sg.In(lmin, key='l_min', size=(8, 1)),
                            sg.T('max:'), sg.In(lmax, key='l_max', size=(8, 1))],
                           [sg.T('Title    Plot width'), sg.In(plot_w, key='plot_w', size=(7, 1)),
                            sg.T(' height'), sg.In(plot_h, key='plot_h', size=(7, 1))],
                           [sg.Combo(('',), size=(38, 1), key='-PLOT_TITLE-')],
                           [sg.Button('Multiplot', tooltip='Plot multiple spectra with vertical offset'),
                            sg.T('offset'), sg.In('1.0', size=(8, 1), key='-OFFSET-',
                                                  tooltip='Vertical offset for multiplot')]])]
    calibrate_elem = [[sg.Frame('Calibration', [
        [sg.Text('Process folder'),
         sg.InputText(outpath, size=(28, 1), disabled=True, key='-OUT_C-')],
        [sg.InputText(outfile, size=(31, 1), key='-SPEC_R-'),
         sg.Button('Load Raw', key='-LOAD_RAW-', tooltip='load uncalibrated 1d spectrum r_addxx.dat')],
        [sg.Button('Select Lines', key='-S_LINES-', disabled=True, button_color=bc_disabled,
                   tooltip='Click to start new calibration, finish with "Save table"'),
         sg.Checkbox('Absorption line', default=False, key='-ABSORPTION-')],
        [sg.Text('Pos          Wavelength')],
        [sg.InputText('0', size=(7, 1), justification='r', key='-POS-', disabled=True),
         sg.Combo(['             ', '0 zero', '517.5 Mg I', '589 Na I', '777.4 O I'], key='-LAMBDA-',
                  enable_events=True, disabled=True, tooltip='click Box to enable selection, then select, ' +
                  'click Button Sel. Line to confirm'),
         sg.Button('Sel. Line', key='-S_LINE-', disabled=True, button_color=bc_disabled),
         sg.Button('Save table', key='-SAVE_T-', tooltip='Save table when finished selection of lines')],
        [sg.Button('Load Table', key='-LOAD_TABLE-', disabled=True, button_color=bc_disabled),
         sg.Button('Calibration', key='-CALI-', disabled=True, button_color=bc_disabled),
         sg.Text('Polynomial degree:'),
         sg.Combo([0, 1, 2, 3, 4, 5], key='-POLY-', enable_events=True, default_value=1,
                  tooltip='for single line calibration select 0, otherwise select degree of polynomial')],
        plot_elem,
        [sg.Multiline('Result', size=(40, 15), disabled=True, key='-RESULT4-', autoscroll=True)]]),
                   sg.Frame('Raw spectrum', column, key='-COLUMN-')]]

    # Instrument response tab--------------------------------------------------------------------
    column = [[sg.Graph(canvas_size=(canvasx, canvasy), graph_bottom_left=(0.0, 0.0),
                        graph_top_right=(1.0, 1.0), background_color='white', key='graph_ir',
                        enable_events=True, drag_submits=True, float_values=True,
                        tooltip='erase points with mouse draw')]]
    ir_elem = [[sg.Frame('Instrument response', [
        [sg.Frame('Reference', [[sg.InputText('', size=(31, 1), key='-STAR-'),
         sg.Button('Load Spectrum', key='-LOAD_STAR-', tooltip='load calibrated star spectrum *.dat')],
        [sg.InputText('', size=(31, 1), key='-REF_I-'),
         sg.Button('Load Reference', key='-LOAD_REF-', tooltip='load calibrated reference spectrum *.dat')],
        [sg.Button('Divide spectra', key='raw_response',
                   tooltip='divide star spectrum by calibration reference'),
         sg.Button('Smooth response', key='smooth_response'),
         sg.In(smooth_parameter, size=(7, 1), key='smooth_parameter', tooltip='Smooth parameter'),
         sg.Button('Reset', key='response_reset')],
        [sg.InputText('response', size=(31, 1), key='-RESPONSE-'),
         sg.Button('Save response', key='-SAVE_RESPONSE-')], ]), ],
        [sg.InputText(response_folder, size=(28, 1), disabled=False, key='-RESPONSE_FOLDER-'),
         sg.Button('Response Folder', key='-SEL_RESP_FOLDER-', tooltip='select response folder')],
        [sg.Text('Lambda min, max', size=(15, 1)), sg.In(lmin, size=(6, 1), key='lmin', enable_events=True),
         sg.In(lmax, size=(6, 1), key='lmax', enable_events=True)],
        [sg.Text('     I min, max', size=(15, 1)), sg.In(i_min, size=(6, 1), key='imin', enable_events=True),
         sg.In(i_max, size=(6, 1), key='imax', enable_events=True), sg.Checkbox('Autoscale', key='autoscale')],
        [sg.Frame('Response correction', [
        [sg.InputText('', size=(31, 1), key='-RESPONSE2-'),
         sg.Button('Load Response', key='-LOAD_RESPONSE-')],
        [sg.InputText(outpath + '/', size=(31, 1), key='-SPECTRUM-'),
         sg.Button('Load Spectrum', key='-LOAD_SPECTRUM-', tooltip='load calibrated spectrum *.dat')],
        [sg.Button('Apply', key='apply_response', tooltip='divide meteor spectrum by response'),
         sg.InputText('', size=(28, 1), key='-FLUX-'),
         sg.Button('Save flux', key='-SAVE_FLUX-')],
        ]), ],
        [sg.Multiline('Result', size=(40, 15), disabled=True, key='-RESULT5-', autoscroll=True)]]),
                   sg.Frame('Instrument response', column, key='-COLUMN_IR-')], ]

    # laser calibration tab--------------------------------------------------------------------
    image_elem_calib = sg.Graph(canvas_size=graph_s2, graph_bottom_left=(0, 0),
                                graph_top_right=graph_s2, key='calib_image')
    log_elem = sg.Multiline('Log', size=(38, 12), autoscroll=True)

    # layout laser calibration window
    layout_parameters = sg.Frame('', [[sg.Frame('Setup',
                  [[sg.Input(ini_file, size=(40, 1), key='setup_file')],
                   [sg.Button('Load Setup'),
                    sg.Button('Edit Setup', tooltip='edit specific laser calibration parameters')]])],
                  [sg.Frame('Video Extraction',
                  [[sg.Input('', size=(40, 1), key='avi_file')],
                   [sg.Button('Load Avi', tooltip='convert avi-file to average background image'),
                    sg.Text('Calibration image:'), sg.Button('Save Image', tooltip='save image before using it')],
                   [sg.Input('', size=(40, 1), key='image_file')]])],
                  [sg.Frame('Select Lines',
                  [[sg.Input(infile, size=(40, 1), key='input_file')],
                   [sg.Button('Load Image', tooltip='select 1 or multiple images (load again) for calibration')],
                   [sg.Text('Calibration data, ".txt":')],
                   [sg.Input(outfil, size=(40, 1), key='output_file',
                             tooltip='select file for calibration data')],
                   [sg.Button('Select File'), sg.Button('Edit File'),
                    sg.Button('Select Lines', tooltip='open window for spectral line selection')],
                   [sg.Text('Linelist'), sg.Input(linelist, size=(20, 1), key='linelist'),
                    sg.Button('Load L_list', tooltip='select file with ordered list of calibration wavelengths, '
                                                     '\ntype "l" for laser calibration')]])],
                  [sg.Frame('Calibration',
                  [[sg.Button('LSQF', tooltip='determine transformation parameters '
                                              '\nwith least square fit of observed line positions'),
                    sg.Checkbox('SQRT-Fit', default=par_dict['b_sqrt'], key='SQRT-Fit'),
                    sg.Checkbox('Fit-xy', default=par_dict['b_fitxy'], key='fitxy')]])],
                   [sg.Text('Results:')], [log_elem]])
    laser_calib_elem = [[layout_parameters,
                         sg.Column([[sg.Text(infile, size=(100, 1), key='image_filename')],
                                   [image_elem_calib]])]]

    # Spectral analysis tab--------------------------------------------------------------------
    element_layout = [[]]
    for ele in elements:
        element_layout += [sg.Checkbox(elements_dict[ele], enable_events=True, key=ele,
                                       default=True, size=(5, 1), pad=p0),
                         sg.T('Scale', key=ele + '_scale', size=(8, 1), pad=p0),
                         sg.I('1.0', key=ele + '_mult', size=(8, 1), enable_events=True, pad=p0),
                         sg.Checkbox('', enable_events=True, key=ele + '_fit', default=True,
                                     size=(0, 1), pad=p0)],
    element_layout += [sg.InputText(t_n2i, size=(20, 1), key='-N2I-FILE-', pad=p0),
                            sg.Button('Sel N2I', key='-SEL_N2I-', tooltip='Select N2I spectrum', pad=p0)],
    element_layout += [sg.B('NIST', tooltip='calculate NIST spectrum', pad=p2),
                       sg.B('Fit', key='-LSQF_SPEC-', disabled=True, button_color=bc_disabled, pad=p2),
                       sg.B('Load Fit', key='-LOAD_FIT-', pad=p2),
                       sg.B('Save Fit', key='-SAVE_FIT-', disabled=True, button_color=bc_disabled, pad=p2)],
    element_layout += [sg.Multiline('Result', size=(30, 10), disabled=True, key='-RESULT_A-',
                                    autoscroll=True, font='Fixedsys 8')],
    spectrum_element = [[sg.Frame('', [[sg.Frame('Spectrum', [[sg.InputText(spec_file_analysis,
                                                               size=(30, 1), key='-SPECTRUM_A-', pad=p0),
                            sg.Button('Select', key='-SEL_SPECTRUM-', tooltip='Select spectrum', pad=p0)],
                            [sg.T('Response', pad=p0),
                             sg.InputText(resp_file_analysis, size=(20, 1), key='-RESPONSE_A-', pad=p0),
                             sg.Button('Select', key='-SEL_RESPONSE_A-', tooltip='Select response', pad=p0)]])],
                            [sg.Frame('Parameters', [[sg.T('Wavelength range', pad=p2),
                             sg.I(lmin_a, size=(6, 1), key='-LMIN_A-', enable_events=True, pad=p2),
                             sg.I(lmax_a, size=(6, 1), key='-LMAX_A-', enable_events=True, pad=p2)],
                            [sg.T('Sigma', pad=p2), sg.I(sigma_nist, size=(6, 1), key='-SIGMA_NIST-',
                                                         enable_events=True, pad=p2),
                             sg.Checkbox('fit', enable_events=True, key='-SIGMA_FIT-',
                                         default=True, size=(0, 1), pad=p2),
                             sg.T('Sigma0', pad=p2),
                             sg.I(sigma0, size=(6, 1), key='-SIGMA0-', enable_events=True, pad=p2)],
                            [sg.T('Continuum Temp.', pad=p2), sg.I('3000', size=(6, 1), key='-T_CONT-',
                                                                   enable_events=True, pad=p2),
                             sg.Checkbox('fit', enable_events=True, key='-T_CONT_FIT-',
                                         default=True, size=(0, 1), pad=p2)],
                            [sg.T('Plasma Temp.', pad=p2), sg.I('3000', size=(6, 1), key='-T_ELECTRON-',
                                                                enable_events=True, pad=p0),
                             sg.Checkbox('fit', enable_events=True, key='-T_EL_FIT-', size=(0, 1), pad=p0),
                             sg.Checkbox('Show sum', enable_events=True, key='-SUM-', pad=p0)],
                            [sg.T('Threshold', pad=p2), sg.I(f'{threshold:5.1e}', size=(6, 1),
                                                             key='-THRESHOLD-', enable_events=True, pad=p2),
                             sg.Checkbox('Show residual', default=show_residuals,
                                         enable_events=True, key='-RESIDUALS-', pad=p2)]], pad=p0)],
                            [sg.Frame('Elements' + 24*' ' + 'factor' + 4*' ' + 'fit', element_layout, pad=p0)]],
                             pad=p0),
                         sg.Graph(canvas_size=(canvasx, canvasy), graph_bottom_left=(0.0, 0.0),
                                  graph_top_right=(1.0, 1.0), background_color='white', key='graph_analysis',
                                  enable_events=True, drag_submits=True, float_values=True,
                                  tooltip='Uncalibrated (raw) spectrum, select calibration lines with mouse draw')]]

    # ==============================================================================
    # Tabs and window
    setup_tab_element = sg.Tab('Setup', layout_setup, key='-T_SETUP-',
                               tooltip='Edit configuration parameters')
    video_tab_element = sg.Tab('Video conversion', [[sub_frame_element],
                               [video_options_element, image_element_video]], key='-T_VIDEO-',
                               tooltip='Convert video to images')
    dist_tab_element = sg.Tab('Distortion', [[dist_elem, image_element_distortion]], key='-T_DIST-',
                              tooltip='Apply orthographic transformation to image series')
    reg_tab_element = sg.Tab('Registration', register_elem, key='-T_REG-',
                             tooltip='Register, add images, apply tilt, slant')
    cal_tab_element = sg.Tab('Calibration', calibrate_elem, key='-T_CAL-',
                             tooltip='Calibrate wavelength scale')
    ir_tab_element = sg.Tab('Instrument Response', ir_elem, key='-T_IR-')
    laser_tab_element = sg.Tab('Laser Calibration', laser_calib_elem, key='-T_LASER_CAL-')
    analysis_tab_element = sg.Tab('Spectral analysis', spectrum_element, key='-T_ANALYSIS-')

    tabs_element = sg.TabGroup([[setup_tab_element], [video_tab_element],
                                [dist_tab_element], [reg_tab_element], [cal_tab_element],
                                [ir_tab_element], [laser_tab_element],
                                [analysis_tab_element]], enable_events=True)
    current_dir = path.abspath('')
    window_title = f'M_SPEC, Version: {version}, {current_dir} , Image: '
    window = sg.Window(window_title, [[sg.Menu(menu_def, tearoff=True, key='menu')],
                       [tabs_element]], location=(wlocx, wlocy), size=(wsx, wsy), resizable=True)
    window.read()
    image_data, actual_file = m_fun.draw_scaled_image('tmp.png', window['-V_IMAGE-'],
                                                      opt_dict, resize=False)
    graph_ir = window['graph_ir']
    graph_an = window['graph_analysis']
    # window['-T_ANALYSIS-'].select()

    # ==============================================================================
    # Main loop
    # ==============================================================================
    while True:
        event, values = window.read(timeout=100)
        if event is None:  # always give a way out!
            break
        event = str(event)  # to catch integer events from ?
        # TODO: why event = 2? noname elements numbered
        # adjust image size if window size, position changed
        if (wsx, wsy) != window.Size or (wlocx, wlocy) != window.current_location():
            if tabs_element.get() != '-T_ANALYSIS-':
                graph_analysis_enabled = False
            if tabs_element.get() == '-T_VIDEO-':
                actual_image = window['-V_IMAGE-']
            elif tabs_element.get() == '-T_DIST-':
                actual_image = window['-D_IMAGE-']
            elif tabs_element.get() == '-T_LASER_CAL-':
                actual_image = window['calib_image']
            else:
                actual_image = window['-R_IMAGE-']
            (wsx, wsy) = window.Size
            (wlocx, wlocy) = window.current_location()
            opt_dict['win_width'] = wsx
            opt_dict['win_height'] = wsy
            opt_dict['win_x'] = wlocx
            opt_dict['win_y'] = wlocy
            image_data, actual_file = m_fun.draw_scaled_image(actual_file, actual_image,
                                                        opt_dict, resize=True, tmp_image=True)
        if not graph_analysis_enabled:
            window.set_title(window_title + str(actual_file))
        if tabs_element.get() == '-T_LASER_CAL-':
            log_elem.Update(logtext)
            cal_tab = True
        elif tabs_element.get() == '-T_IR-':
            ir_tab = True
        else:
            cal_tab = False
            ir_tab = False

        # enable, disable 'Offset', only if tab '-T_LASER_CAL-' switched
        if cal_tab != old_cal_tab:
            menu_def[2][1][0] = '&Offset' if cal_tab else '!%Offset'
            window['menu'].update(menu_def)
            old_cal_tab = cal_tab
        elif ir_tab != old_ir_tab:
            menu_def[2][1][5] = '&Gaussian, wavelength tools' if ir_tab else '!Gaussian, wavelength tools'
            menu_def[2][1][6] = 'Atmospheric correction' if ir_tab else '!Atmospheric correction'
            window['menu'].update(menu_def)
            old_ir_tab = ir_tab

        # ==============================================================================
        # Menu
        # ==============================================================================
        # submenus
        if event == 'Fits-Header':
            m_fun.view_fits_header(outfile)
        if event == 'Edit Text File':
            m_fun.edit_text_window(llist)
        if event == 'Edit Log File':
            m_fun.edit_text_window(m_fun.logfile, select=False, size=(90, 30))
        if event == 'Edit Spectrum':
            m_fun.edit_text_window('', default_extension='*.dat')
        if event == 'Add Images':
            sum_file, nim = m_fun.add_images(graph_s2, contrast=1, average=True)
            window['-RADD-'].update(sum_file)

        if event == 'Gaussian, wavelength tools':
            if tool_type == 'reference':
                file = values['-REF_I-']
            else:
                file = values['-STAR-']
            file, new_file, lref, iref, info, tool_type = m_plot.wavelength_tools(sigma, file=file)
            if len(lref) and file:
                if tool_type == 'reference':
                    reference_file = new_file  # use as new reference
                    plot_range, l_spec, i_spec = m_plot.plot_raw_spectrum(file, graph_ir, canvasx,
                                                                    autoscale=values['autoscale'])
                    if reference_file:
                        idg_ref = m_plot.plot_reference_spectrum(reference_file, lref, iref, graph_ir, canvasx,
                                                                 plot_style=ref_style)
                    window['-REF_I-'].update(reference_file)
                else:
                    if new_file:  # order conversion, leave filename as is for new order conversion
                        plot_range, l_spec, i_spec = m_plot.plot_raw_spectrum(new_file, graph_ir, canvasx,
                                                                            autoscale=values['autoscale'],
                                                                            plot_style=ref_style)
                    result_text += info + '\n'
                window['-RESULT5-'].update(result_text)
                window.refresh()

        if event == 'Atmospheric correction':
            spectrum_uncorrected = values['-SPECTRUM-']
            spectrum_am0, elevation_deg, h, aod, info, resp_flag, trans_flag = m_fun.extinction_tool(
                                            spectrum_uncorrected, elevation_deg, h, aod, resp_flag, trans_flag)
            if spectrum_am0:
                if resp_flag:
                    response_file = spectrum_am0
                    window['-RESPONSE2-'].update(response_file)
                    m_plot.delete_curve(idg_resp, graph_ir)
                    l_smooth, i_smooth = np.loadtxt(response_file, unpack=True, ndmin=2)
                    idg_resp = m_plot.plot_reference_spectrum(spectrum_am0, l_smooth, i_smooth,
                                                graph_ir, canvasx, plot_style=response_style)
                else:
                    spectrum_file = spectrum_am0
                    window['-SPECTRUM-'].update(spectrum_am0)
                    m_plot.delete_curve(idg_ref, graph_ir)
                    l_spec, i_spec = np.loadtxt(spectrum_am0, unpack=True, ndmin=2)
                    idg_ref = m_plot.plot_reference_spectrum(spectrum_am0, l_spec, i_spec, graph_ir, canvasx,
                                                             plot_style=ref_style)
                result_text += info + '\n'
                window['-RESULT5-'].update(result_text)
                window.refresh()
                if trans_flag:
                    l_trans, i_trans = np.loadtxt('transmission_atmos.dat', unpack=True, ndmin=2)
                    idg_trans = m_plot.plot_reference_spectrum('transmission_atmos.dat', l_trans, i_trans,
                                                               graph_ir, canvasx, idg_trans, plot_style=trans_style)

        if event == 'Image Tools':
            f_pix = m_fun.par_dict['f_f0'] / m_fun.par_dict['f_pix'] / m_fun.par_dict['i_binning']
            m_image.image_tools('back', opt_dict, fits_dict, res_dict, f_pix=f_pix)

        if event == 'Setup Options':
            opt_dict = m_fun.select_options(opt_dict, )
            debug = opt_dict['debug']

        if event == 'Meteor lines':
            meteor_lines, info = m_fun.my_get_file(meteor_lines, title='Meteor line catalog',
                                             default_extension='.txt',
                                             file_types=(('Calibration Files', '*.txt'),),
                                             error_message='no file loaded')
            if not meteor_lines:
                sg.PopupError('no file selected, use default list')
                meteor_lines = 'meteor_lines'
            opt_dict['meteor_lines'] = m_fun.m_join(meteor_lines)

        if event == 'About...':
            m_fun.about(version)

        if event == 'Offset':
            if infile:
                if Path(m_fun.change_extension(infile, '.fit')).exists():
                    image_data, actual_file, imbw = m_fun.draw_scaled_image(
                        m_fun.change_extension(infile, '.fit'),
                        image_elem_calib, opt_dict, get_array=True)
                    im = imbw - np.average(imbw)
                    im_std = np.std(im)
                    im_clip = np.clip(im, -2.0 * im_std, 2.0 * im_std)
                    offset = - np.average(imbw) - np.average(im_clip)
                    print('orig: min, max , ave, offset ', np.min(imbw), np.max(imbw),
                          np.average(imbw), offset)
                    imbw = np.clip(imbw + offset, 0.0, 1.0)
                    file_offset = m_fun.change_extension(infile, '_off.fit')
                    print('clip: min, max , ave ', np.min(imbw), np.max(imbw), np.average(imbw))
                    m_fun.write_fits_image(imbw, file_offset, fits_dict)
                    image_data, actual_file = m_fun.draw_scaled_image(file_offset,
                                        image_elem_calib, opt_dict, tmp_image=True)
                    infile = infile + '_off'
                    window['image_filename'].Update(infile)
                    window['input_file'].Update(infile)
                    logging.info(f'image with offset saved as: {file_offset}')
                    logtext += f'image with offset saved as: {file_offset}\n'
                    window.refresh()
                else:
                    sg.PopupError('file not found, load valid fits image')

        # ==============================================================================
        # Setup Tab
        # ==============================================================================
        elif event == '-LOAD_SETUP-':
            ini_file, info = m_fun.my_get_file(setup_file_display_elem.Get(),
                                               title='Get Setup File',
                                               file_types=(('Setup Files', '*.ini'), ('ALL Files', '*.*')),
                                               default_extension='*.ini')
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
                if list(opt_dict.values()):
                    [zoom, wsx, wsy, wlocx, wlocy, xoff_calc, yoff_calc,
                     xoff_setup, yoff_setup, debug, fit_report, win2ima,
                     opt_comment, png_name, outpath, mdist, colorflag, bob_doubler,
                     plot_w, plot_h, i_min, i_max, graph_size, show_images,
                     meteor_lines, flat_flag, flat_file] = list(opt_dict.values())
                window['-PNG_BASE-'].Update(png_name)
                window['-PNG_BASED-'].Update(png_name)
                window['-OUT-'].Update(outpath)
                window['-OUT_R-'].Update(outpath)
                window['-OUT_C-'].Update(outpath)
                window['-SPECTRUM-'].update(outpath + '/')
                window['-M_DIST-'].Update(mdist)
                window['-M_DIST_R-'].Update(mdist)
                window['-COLOR-'].Update(colorflag)
                window['-BOB-'].Update(bob_doubler)
                window['-BOB_D-'].Update(bob_doubler)
                window['-SHOW_IM-'].Update(show_images)
                window['-FLAT_C-'].Update(flat_flag)
                window['-FLAT-'].Update(flat_file)
                window.Move(wlocx, wlocy)

        elif event in ('-SAVE_SETUP-', '-SAVE_DEFAULT-', 'Exit'):
            if event == '-SAVE_SETUP-':
                ini_file, info = m_fun.my_get_file(setup_file_display_elem.Get(), save_as=True,
                                                   file_types=(('Setup Files', '*.ini'), ('ALL Files', '*.*')),
                                                   title='Save Setup File', default_extension='*.ini', )
            else:
                ini_file = 'm_set.ini'
            setup_file_display_elem.update(ini_file)
            par_dict['i_binning'] = int(values['-BIN-'])  # from video tab
            # update res_dict and fits_dict with new values
            for k in range(7):
                kk = f'k{k}'
                res_v[k] = float(values[kk])
            res_dict = dict(list(zip(res_key, res_v)))  # update res_dict
            for k in range(7):
                kk = f'k{k + 7}'
                fits_v[k] = values[kk]
                fits_dict[fits_k[k]] = fits_v[k]
            for k in res_dict.keys():
                fkey = 'D_' + k.upper()
                fits_dict[fkey] = np.float32(res_dict[k])
                logging.info(f'{k} = {res_dict[k]:9.3e}') if k[0] == 'a' else logging.info(f'{k} = {res_dict[k]:9.3f}')
            logging.info(f"'DATE-OBS' = {dat_tim}")
            logging.info(f"'M-STATIO' = {sta}")
            opt_dict['png_name'] = values['-PNG_BASE-']
            opt_dict['outpath'] = values['-OUT-']
            opt_dict['mdist'] = values['-M_DIST-']
            opt_dict['colorflag'] = values['-COLOR-']  # from register_images tab
            opt_dict['bob'] = values['-BOB-']
            opt_dict['plot_w'] = plot_w
            opt_dict['plot_h'] = plot_h
            opt_dict['i_min'] = i_min
            opt_dict['i_max'] = i_max
            opt_dict['show_images'] = values['-SHOW_IM-']
            opt_dict['flat_flag'] = values['-FLAT_C-']
            opt_dict['flat_file'] = values['-FLAT-']
            [zoom, wsx, wsy, wlocx, wlocy, xoff_calc, yoff_calc,
                xoff_setup, yoff_setup, debug, fit_report, win2ima,
                opt_comment, png_name, outpath, mdist, colorflag, bob_doubler,
                plot_w, plot_h, i_min, i_max, graph_size, show_images,
                meteor_lines, flat_flag, flat_file] = list(opt_dict.values())
            if ini_file and event != '-APPLY_OPT-':
                m_fun.write_configuration(ini_file, par_dict, res_dict, fits_dict, opt_dict)
            try:
                # finish with meaningful image for next start, if not, use existing image
                image_data, actual_file = m_fun.draw_scaled_image(m_fun.m_join(outpath, mdist) + '_peak.fit',
                                                  window['-D_IMAGE-'], opt_dict, contr=1, tmp_image=True)
            except Exception as e:
                print(f'save tmp.png error, outpath: {outpath},\n{e}')
            finally:
                if event == 'Exit':
                    window.close()
                    break

        # ==============================================================================
        # Video Tab
        # ==============================================================================
        elif event == 'Load_Video':
            window['-GOTO_DIST-'].update(disabled=True, button_color=bc_disabled)
            avifile, info = m_fun.my_get_file(filename_display_elem.Get(), title='Get Video File',
                                default_extension='.avi', file_types=(('Video Files', '*.avi'),
                                                                      ('ALL Files', '*.*')))
            if avifile:
                filename_display_elem.update(avifile)
                png_name = values['-PNG_BASE-']
                bob_doubler = values['-BOB-']
                par_dict['i_binning'] = int(values['-BIN-'])
                bff = values['-BFF-']
                # check previous PNG images
                old_files, deleted, answer = m_fun.delete_old_files(png_name, max(maxim, int(values['-MAXIM-'])))
                maxim = int(values['-MAXIM-'])
                if answer != 'Cancel':
                    window['-RESULT-'].update('Start video conversion\n')
                    window.refresh()
                    nim, dat_tim, sta, out = m_fun.extract_video_images(avifile, png_name,
                                                bob_doubler, par_dict['i_binning'], bff, maxim)
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
                        image_data, actual_file = m_fun.draw_scaled_image(out + '1.png', window['-V_IMAGE-'],
                                                                          opt_dict, idg)
                        # add avifile to video_list
                        m_fun.update_video_list('videolist.txt', avifile)
                    logging.info(f'converted {avifile} {nim} images')
                    logging.info(f'Station = {sta} Time = {dat_tim}')
                    result_text = f'Station = {sta}\nTime = {dat_tim}\n'
                    result_text += f'\nNumber converted images = {str(nim)}\n'
                    window['-RESULT2-'].update(result_text)
                    window['-PNG_BASED-'].update(png_name)
                    window['-BOB_D-'].update(bob_doubler)
                    if bob_doubler:
                        i = 50  # jump to 1st image after background
                        n_back = 40
                        first = 50
                        fits_dict['M_BOB'] = 1
                    else:
                        i = 25
                        n_back = 20
                        first = 25
                        fits_dict['M_BOB'] = 0
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
            if event == '-NEXT-' and i < nim:
                i += 1
            if event == '-PREVIOUS-' and i > 1:
                i -= 1
            image_data, actual_file = m_fun.draw_scaled_image(out + str(i) + '.png',
                                                              window['-V_IMAGE-'], opt_dict, idg)

        if event == '-GOTO_DIST-':
            image_data, actual_file = m_fun.draw_scaled_image(out + str(i) + '.png',
                                                              window['-D_IMAGE-'], opt_dict, idg)
            window['-T_DIST-'].select()  # works

        # ==============================================================================
        # Distortion Tab
        # ==============================================================================
        elif event in '-PNG_BASED-':
            # remove fits-header entries from earlier runs, no longer valid in new directory
            fits_dict.pop('DATE-OBS', None)
            fits_dict.pop('M_STATIO', None)
            dat_tim = ''
            sta = ''
            # get new number of images
            inpath = png_name
            nm_found = m_fun.check_files(inpath, maxim, ext='.png')
            nm = nm_found - first + 1
            window['-N_IMAGE-'].update(nm)

        elif event in ('-SEL_OUT-', '-SEL_OUT_R-'):
            outpath = sg.PopupGetFolder('', title='Select Process Folder',
                                        initial_folder=outpath, no_window=True)
            outpath = m_fun.m_join(outpath)
            print('outpath', str(outpath))
            # if outpath == '.':
            #     outpath = values['-OUT-']
            window['-OUT-'].update(outpath)
            window['-OUT_C-'].update(outpath)
            window['-OUT_R-'].update(outpath)
            window['-SPECTRUM-'].update(outpath + '/')

        elif event == '-SEL_FLAT-':
            flat_file = values['-FLAT-']
            flat_file, info = m_fun.my_get_file(flat_file, title='Get Flat File',
                                                file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),),
                                                default_extension='*.fit')
            flat_file = m_fun.m_join(flat_file)
            window['-FLAT-'].update(flat_file)

        elif event == '-APPLY_DIST-':
            m_fun._go = True
            window['-GOTO_REG-'].update(disabled=False, button_color=bc_disabled)
            png_name = values['-PNG_BASED-']
            outpath = values['-OUT-']
            mdist = values['-M_DIST-']
            infile = m_fun.m_join(outpath, mdist)
            dist = values['-DIST-']
            background = values['-BACK-']
            bob_doubler = values['-BOB_D-']
            colorflag = values['-COLOR-']
            flat_flag = values['-FLAT_C-']
            flat_file = values['-FLAT-']
            flat = flat_file if flat_flag else ''
            n_back = int(values['-N_BACK-'])
            first = int(values['-N_START-'])
            nm = int(values['-N_IMAGE-'])
            show_images = values['-SHOW_IM-']
            inpath = path.normpath(png_name)
            # check number of  tmp\*.png   <--
            nm_found = m_fun.check_files(inpath, int(values['-MAXIM-']), ext='.png')
            if nm <= 0 or nm > nm_found - first + 1:
                sg.PopupError(
                    f'not enough meteor images, check data\n nim = {nm_found}, '
                    f'maximum processed: {str(nm_found - first + 1)}')
                nm = nm_found - first + 1
                window['-N_IMAGE-'].update(nm)
            else:
                if not path.exists(outpath):
                    os.mkdir(outpath)
                [scalxy, x00, y00, rot, disp0, a3, a5] = res_dict.values()
                fits_dict['M_BOB'] = 0
                if bob_doubler:  # center measured before scaling in y scalxy*2 compensates for half image height
                    scalxy *= 2.0
                    y00 /= 2.0
                    fits_dict['M_BOB'] = 1
                # ---------------------------------------------------------------
                # check previous images mdist
                distfile = path.normpath(path.join(outpath, mdist))  # 'D:/Daten/Python/out\\mdist'
                old_files, deleted, answer = m_fun.delete_old_files(distfile, maxim, ext='.fit')
                disttext = f' {deleted} files deleted of {old_files}\n'
                if answer != 'Cancel' and scalxy > 0.0:
                    # make background image
                    t0 = time.time()  # start timer
                    back = m_fun.create_background_image(inpath, n_back, colorflag)
                    # save background image as png and fit
                    # remove unnecessary fits header items before saving fits-images
                    fits_dict.pop('M_NIM', None)
                    fits_dict.pop('M_STARTI', None)
                    temp_dict = {'COMMENT': ''}
                    disttext += 'start background image\nwait for background image\n'
                    window['-RESULT2-'].update(value=result_text + disttext, append=True)
                    window.refresh()
                    m_fun.save_fit_png(m_fun.m_join(outpath, 'm_back'), back, temp_dict, dist=False)
                    image_data, actual_file = m_fun.draw_scaled_image(m_fun.m_join(outpath, 'm_back.fit'),
                                                        window['-D_IMAGE-'], opt_dict, tmp_image=True)
                    disttext += f'background created of {n_back} images\n'
                    disttext += f'process time background {time.time() - t0:8.2f} sec\n'
                    window['-RESULT2-'].update(result_text + disttext)
                    window.refresh()
                    # apply distortion
                    if dist:  # with distortion, add parameters to fits-header
                        for key in res_dict.keys():
                            fkey = 'D_' + key.upper()
                            fits_dict[fkey] = np.float32(res_dict[key])
                            logging.info(f'{key} = {res_dict[key]:9.3e}') if key[0] == 'a' \
                                else logging.info(f'{key} = {res_dict[key]:9.3f}')
                    else:
                        for key in fits_dict.keys():
                            if key in ('D_A3', 'D_A5'):
                                fits_dict[key] = 0.0
                        logging.info('no distortion applied')
                    cval = 0.001
                    center = (x00, y00)
                    m_fun.distortion_long_function(inpath, outpath, mdist, first, nm, window, dist, background,
                                                   center, a3, a5, rot, scalxy, colorflag, show_images, cval, flat)
                    while True:
                        event, values = window.read()
                        if event == '-STOP-':
                            m_fun._go = False
                        elif event == '-THREAD PROGRESS-':
                            fileout = values['-THREAD PROGRESS-']
                            if show_images:
                                image_data, actual_file = m_fun.draw_scaled_image(fileout +
                                            '.fit', window['-D_IMAGE-'], opt_dict, resize=True)
                        elif event == '-THREAD DONE-':
                            (a, disttext) = (values['-THREAD DONE-'])
                            break
                    m_fun.draw_scaled_image(outpath + '/' + mdist + '_peak' + '.fit',
                                            window['-D_IMAGE-'], opt_dict, resize=True)
                    # window.refresh()
                elif abs(scalxy) < 1.e-3:
                    sg.PopupError('Load valid calibration parameters before applying distortion',)
                else:
                    disttext = 'no files deleted'
                window['-RESULT2-'].update(result_text + disttext)
                window.refresh()

        if event == '-GOTO_REG-':
            window['-T_REG-'].select()
            window['-M_DIST_R-'].update(mdist)
            window['-N_MAX_R-'].update(m_fun.check_files(infile, maxim, ext='.fit'))
            m_fun.refresh_image(image_data, window['-R_IMAGE-'], opt_dict, idg)
            window['-SHOW_SUM_R-'].update(disabled=True, button_color=bc_disabled)
            window['-SHOW_REG-'].update(False)

        # ==============================================================================
        # Registration Tab
        # ==============================================================================
        elif event == '-M_DIST_R-':
            mdist = values['-M_DIST_R-']
            infile = m_fun.m_join(outpath, mdist)
            nm_found = m_fun.check_files(infile, maxim, ext='.fit')
            window['-N_MAX_R-'].update(nm_found)
            result_text = ''

        # image browser---------------------------------------------------------
        elif event in ('-NEXT_R-', '-PREV_R-'):
            mdist = values['-M_DIST_R-']
            reg_file = values['-REG_BASE-']
            infile = m_fun.m_join(outpath, mdist)
            out_fil = m_fun.m_join(outpath, reg_file)
            nm_found = m_fun.check_files(infile, maxim, ext='.fit')
            nmp = int(values['-N_MAX_R-'])
            if nm_found < nmp or nmp <= 0:
                nmp = nm_found
                window['-N_MAX_R-'].update(nmp)
            last_file_sum = False
            if i_reg < nmp and event == '-NEXT_R-':
                i_reg += 1
            if i_reg > 1 and event == '-PREV_R-':
                i_reg -= 1
                i_reg = min(nmp, i_reg)
            if values['-SHOW_REG-']:
                nim = m_fun.check_files(out_fil, maxim, ext='.fit')
                i_reg = min(nim, i_reg)
                if i_reg <= nim and path.exists(out_fil + str(i_reg) + '.fit'):
                    image_data, actual_file = m_fun.draw_scaled_image(out_fil + str(i_reg) + '.fit',
                                                        window['-R_IMAGE-'], opt_dict, contr=contrast)
                    window['-INDEX_R-'].update(reg_file + str(i_reg))
                elif i_reg > 1:
                    i_reg -= 1
            else:
                if path.exists(infile + str(i_reg) + '.fit'):
                    image_data, actual_file = m_fun.draw_scaled_image(infile + str(i_reg) + '.fit',
                                                        window['-R_IMAGE-'], opt_dict, contr=contrast)
                    window['-INDEX_R-'].update(mdist + str(i_reg))
                else:
                    sg.PopupError(f'File {infile + str(i_reg)}.fit not found')
            if 'DATE-OBS' in fits_dict.keys():
                dat_tim = fits_dict['DATE-OBS']
                sta = fits_dict['M_STATIO']
            else:
                logging.info('no fits-header DATE-OBS, M-STATIO')
                result_text = '\n!!!no fits-header DATE-OBS, M-STATIO!!!\n'
        # image contrast--------------------------------------------------------
        elif event in ('-LOW_C-', '-HIGH_C-'):
            contrast = 0.5 * contrast if event == '-LOW_C-' else 2.0 * contrast
            try:
                if last_file_sum:
                    image_data, actual_file = m_fun.draw_scaled_image(outfile + '.fit', window['-R_IMAGE-'],
                                                                      opt_dict, contr=contrast)
                else:
                    if values['-SHOW_REG-']:
                        if path.exists(out_fil + str(i_reg) + '.fit'):
                            image_data, actual_file = m_fun.draw_scaled_image(out_fil + str(i_reg) + '.fit',
                                                                window['-R_IMAGE-'], opt_dict, contr=contrast)
                    else:
                        image_data, actual_file = m_fun.draw_scaled_image(infile + str(i_reg) + '.fit',
                                                                window['-R_IMAGE-'], opt_dict, contr=contrast)
            except Exception as e:
                print(f'{e}')
        # image selection-------------------------------------------------------
        elif event == '-SHOW_SUM_R-':
            image_data, actual_file = m_fun.draw_scaled_image(outfile + '.fit', window['-R_IMAGE-'],
                                                              opt_dict, contr=contrast)
            last_file_sum = True

        elif event == '-SEL_START-':
            start = i_reg
            window['-N_START_R-'].update(start)

        elif event == '-SEL_LAST-':
            nsel = i_reg
            start = int(values['-N_START_R-'])
            nim = nsel - start + 1
            window['-N_REG-'].update(nim)

        if event == '-GOTO_CAL-':
            window['-T_CAL-'].select()
            window['-CALI-'].update(disabled=True, button_color=bc_disabled)
            window['-SHOW_REG-'].update(False)
        # ==============================================================================
        # Registration
        elif event == '-REGISTER-':
            window['-SHOW_SUM_R-'].update(disabled=True, button_color=bc_disabled)
            window['-CAL_R-'].update(disabled=True, button_color=bc_disabled)
            mdist = values['-M_DIST_R-']
            infile = m_fun.m_join(outpath, mdist)
            reg_file = values['-REG_BASE-']
            start = int(values['-N_START_R-'])
            nim = int(values['-N_REG-'])
            nmp = int(values['-N_MAX_R-'])
            out_fil = m_fun.m_join(outpath, reg_file)
            if start > nmp:
                sg.PopupError('check image folder and number of distorted images')
            else:
                im, header = m_fun.get_fits_image(infile + str(start))
                if 'M_FLAT' in header.keys():
                    fits_dict['M_FLAT'] = header['M_FLAT']
                # 'tmp.png' needed for select_rectangle:
                image_data, actual_file = m_fun.draw_scaled_image(infile + str(start) + '.fit', window['-R_IMAGE-'],
                                                                  opt_dict, contr=contrast, tmp_image=True)
                if not sta and 'M_STATIO' in header.keys():
                    sta = header['M_STATIO']
                    dat_tim = header['DATE-OBS']
                # ===================================================================
                # select rectangle for registration
                window.Disable()
                select_event, x0, y0, dx, dy = m_fun.select_rectangle(infile, start, res_dict, fits_dict,
                                                                      (wlocx, wlocy), out_fil, maxim)
                window.Enable()
                window.bring_to_front()
                if select_event == 'Ok':
                    nsel = start + nim - 1  # nsel index of last selected image, nim number of images
                    if nsel > nmp:
                        nim = max(nmp - start + 1, 0)
                        window['-N_REG-'].update(nim)
                    t0 = time.time()
                    fits_dict['M_STARTI'] = start
                    reg_type = 'Gaussian' if values['-GAUSSIAN-'] else 'cross-correlation'
                    index, sum_image, reg_text, dist, outfile, fits_dict = m_fun.register_images(start, nim, x0,
                                                    y0, dx, dy, infile, out_fil, window, fits_dict, contrast,
                                                    values['-GAUSSIAN-'], values['-SHOW_REG-'], debug)
                    t3 = time.time() - t0
                    nim = index - start + 1
                    if nim > 1:
                        logging.info(f'time for register one image : {t3 / nim:6.2f} sec, {reg_type} ')
                        result_text += (f'Station = {sta}\nTime = {dat_tim}\n'
                                        + f'\nStart image = {str(start)} register {reg_type}\n'
                                        + f'Number registered images: {nim}\nof total images: {nmp}\n'
                                        + f'time for register one image: {t3 / nim:6.2f} sec\n')
                        image_data, actual_file = m_fun.draw_scaled_image(outfile + '.fit', window['-R_IMAGE-'],
                                                                          opt_dict, contr=contrast)
                        window['-SHOW_REG-'].update(True)
                        window['-RADD-'].update(outfile)
                        window['-SHOW_SUM_R-'].update(disabled=False, button_color=bc_enabled)
                        window['-ADD_ROWS-'].update(disabled=False, button_color=bc_enabled)
                    else:
                        result_text = (f'Number registered images: {nim}\n'
                                       + f'of total images: {nmp}\nnot enough images\n')
                        info = f'{reg_type} register did not work with last image, try again!'
                        sg.PopupError(info)
                        logging.error(info)
                    window['-RESULT3-'].update(reg_text + result_text)
        # =======================================================================
        # convert 2-D spectrum to 1-D spectrum
        elif event == '-ADD_ROWS-':
            if outfile:
                image_data, actual_file = m_fun.draw_scaled_image(outfile + '.fit', window['-R_IMAGE-'],
                                                                  opt_dict, contr=contrast, tmp_image=True)
                wloc_image = (opt_dict['win_x'] + opt_dict['calc_off_x'], opt_dict['win_y'] + opt_dict['calc_off_y'])
                window.Disable()
                ev, tilt, slant, wloc_image = m_fun.add_rows_apply_tilt_slant(outfile, par_dict,
                        res_dict, fits_dict, opt_dict, contrast, wloc_image, result_text, reg_text, window)
                opt_dict['calc_off_x'] = wloc_image[0] - opt_dict['win_x']
                opt_dict['calc_off_y'] = wloc_image[1] - opt_dict['win_y']
                window.Enable()
                window.BringToFront()
                image_data, actual_file = m_fun.draw_scaled_image(outfile + 'st.fit', window['-R_IMAGE-'],
                                                                  opt_dict, contr=contrast)
                window['-RADD-'].update(outfile)

        # =======================================================================
        elif event == '-LOAD_R-':
            # load existing file for adding rows and apply tilt and slant
            result_text = ''
            outfile = values['-RADD-']
            window['-SAVE_RAW-'].update(disabled=True, button_color=bc_disabled)
            window['-CAL_R-'].update(disabled=True, button_color=bc_disabled)
            outfile, info = m_fun.my_get_file(outfile, title='Get Registered File',
                                              file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),),
                                              default_extension='*.fit')
            if outfile:
                # remove fits header items not present in mdist files, load actual values below
                fits_dict['M_NIM'] = '1'
                fits_dict['M_STARTI'] = '0'
                outfile = m_fun.change_extension(outfile, '')
                window['-RADD-'].update(outfile)
                last_file_sum = True
                im, header = m_fun.get_fits_image(outfile)
                image_data, actual_file = m_fun.draw_scaled_image(outfile + '.fit', window['-R_IMAGE-'],
                                                                  opt_dict, contr=contrast, tmp_image=True)
                dist = True if 'D_X00' in header.keys() else False
                fits_dict = m_fun.get_fits_keys(header, fits_dict, res_dict, keyprint=debug)
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
                    pass
                window['-RESULT3-'].update(reg_text + result_text)
            else:
                window['-ADD_ROWS-'].update(disabled=True, button_color=bc_disabled)

        elif event == '-SAVE_RAW-':
            imtilt, header = m_fun.get_fits_image(outfile + 'st')
            lcal, ical = np.loadtxt(outfile + '.dat', unpack=True, ndmin=2)
            # default values for fits_dict:
            fits_dict['M_TILT'] = 0.0
            fits_dict['M_SLANT'] = 0.0
            fits_dict['M_ROWMIN'] = 0
            fits_dict['M_ROWMAX'] = 0
            # update fits_dict
            m_fun.get_fits_keys(header, fits_dict, res_dict, keyprint=debug)
            new_outfile, info = m_fun.my_get_file(outfile, title='Save image and raw spectrum as',
                                                  save_as=True, file_types=spectrum_files)
            if new_outfile:
                outfile, ext = path.splitext(new_outfile)
                window['-RADD-'].update(outfile)
                m_fun.write_fits_image(imtilt, outfile + '.fit', fits_dict, dist=dist)
                image_data, actual_file = m_fun.draw_scaled_image(outfile + '.fit', window['-R_IMAGE-'],
                                                                  opt_dict, contr=contrast)
                np.savetxt(outfile + '.dat', np.transpose([lcal, ical]), fmt='%6i %8.5f')
                result_text += 'File saved as :' + outfile + '.dat (.fit)\n'
                window['-RESULT3-'].update(reg_text + result_text)

        # =======================================================================
        # Calibration Tab
        # =======================================================================
        # load uncalibrated raw spectrum
        elif event in ('-LOAD_RAW-', '-CAL_R-'):
            if event == '-CAL_R-':
                # start calibration
                raw_spec_file = m_fun.change_extension(values['-RADD-'], '.dat')
                window['-T_CAL-'].select()
            else:
                raw_spec_file, info = m_fun.my_get_file(raw_spec_file, title='Load raw spectrum',
                                        file_types=spectrum_files, default_extension='*.dat')
            window['-S_LINES-'].update(disabled=True, button_color=bc_disabled)
            if raw_spec_file:
                m_fun.create_line_list_combo(line_list, window)
                result_text += f'File {raw_spec_file} loaded\n'
                graph = window['graph']
                # plot raw spectrum
                plot_range, lcal, ical = m_plot.plot_raw_spectrum(raw_spec_file, graph, canvasx, plot_style=star_style)
                base, ext = path.splitext(raw_spec_file)
                if ext != '.dat':  # .dat files uploaded to github as .txt, .dat not allowed, converted here
                    raw_spec_file = m_fun.change_extension(base, '.dat')
                    np.savetxt(raw_spec_file, np.transpose([lcal, ical]), fmt='%6i %8.5f')
                    result_text += f'File saved as :{raw_spec_file}\n'
                    window['-RESULT4-'].update(reg_text + result_text)
                window['-SPEC_R-'].update(raw_spec_file)
                ((lmin, i_min), (lmax, i_max)) = plot_range
                window['-S_LINES-'].update(disabled=False, button_color=bc_enabled)
                window['-LOAD_TABLE-'].update(disabled=False, button_color=bc_enabled)
                llist = m_fun.change_extension(raw_spec_file, '.txt')
                graph_enabled = True
                if lmin not in (0.0, 1.0):
                    sg.PopupError('raw files only, load uncalibrated file or Load Spectrum',
                                  title='Wavelength calibration', line_width=60)

        # ==============================================================================
        # select spectral lines for calibration
        elif event == '-S_LINES-':
            table = []
            select_line_enabled = True
            table_edited = False
            dragging = False
            start_point = end_point = prior_rect = None
            window['-CALI-'].update(disabled=True, button_color=bc_disabled)
            cal_text_file = ' Pixel    width  lambda    fit    delta\n'

        elif event == 'graph' and graph_enabled:  # if there's a "graph" event, then it's a mouse
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
        elif event == '-S_LINE-':
            # set focus? to lambda if wavelength entered ok
            window['-S_LINE-'].update(disabled=True, button_color=bc_disabled)
            window['-LAMBDA-'].update(disabled=True)
            try:
                (l0, name) = values['-LAMBDA-'].split(' ', 1)
            except:  # name label missing
                l0 = values['-LAMBDA-']
                name = ''
            try:
                lam = float(l0)
                abs_sign = -1 if values['-ABSORPTION-'] else 1
                x0, fwp, cal_text_file = m_fun.select_calibration_line(x0, w, lam, name, lcal, ical,
                                                                       graph, table, abs_sign=abs_sign)
                result_text += cal_text_file
                window['-RESULT4-'].update(result_text, disabled=True)
            except Exception as e:
                sg.PopupError(f'invalid calibration wavelength\n{e}')

        # ==============================================================================
        # select calibration wavelength from list
        elif event == '-LAMBDA-':
            window['-S_LINE-'].update(disabled=False, button_color=bc_enabled)

        # ==============================================================================
        # save calibration data
        elif event == '-SAVE_T-':
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
                logging.info(f'table saved as {llist}')
                window['-CALI-'].update(disabled=False, button_color=bc_enabled)
                graph_enabled = False
                select_line_enabled = False
            else:
                sg.PopupError('no table to save, select lines first')

        # ==============================================================================
        # load table, enable editing table
        elif event == '-LOAD_TABLE-':
            llist, info = m_fun.my_get_file(llist, title='Load calibration table',
                                            file_types=(('Calibration Files', '*.txt'), ('ALL Files', '*.*'),),
                                            default_extension='*.txt')
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
        # calculate calibration polynomial, calibrate raw spectrum
        elif event == '-CALI-':
            deg = int(values['-POLY-'])
            if deg >= 1:
                if len(xcalib) <= deg:
                    sg.PopupError(f'not enough data points for polynomial degree {deg}, chooose lower value',
                                  title='Polynom degree')
                else:
                    c = np.polyfit(xcalib, lcalib, deg)  # do the polynomial fit (order=2)
            else:  # use disp0 for conversion to wavelength
                deg = 1
                disp0 = float(fits_dict['D_DISP0'])
                try:
                    disp = float(sg.PopupGetText('select value for disp0:',
                                                 title='linear dispersion [nm/Pixel]', default_text=str(disp0)))
                except Exception as e:
                    sg.PopupError(f'invalid value for dispersion, try again\n{e}')
                    disp = disp0
                c = [disp, lcalib[0] - disp * xcalib[0]]
            if len(c):
                cal_dat_file, spec_file, lmin, lmax, cal_text_file = m_fun.calibrate_raw_spectrum(raw_spec_file,
                                                                            xcalib, lcalib, deg, c)
                logging.info(f'spectrum {spec_file} saved')
                window['-RESULT4-'].update(result_text + cal_text_file, disabled=True)
                window['-PLOT_SPEC-'].update(spec_file)
                window['-PLOTS-'].update(disabled=False, button_color=bc_enabled)
                window['-SAVES-'].update(disabled=False, button_color=bc_enabled)
                window['l_min'].update(str(int(lmin)))
                window['l_max'].update(str(int(lmax)))
                video_list = m_fun.read_video_list('videolist.txt')
                video_list.insert(0, spec_file)
                video_list.insert(0, ' ')
                window['-PLOT_TITLE-'].update(values=video_list, value='')

        # ==============================================================================
        # load (un)calibrated spectrum
        if event == '-LOADS-':
            window['-S_LINES-'].update(disabled=True, button_color=bc_disabled)
            spec_file, info = m_fun.my_get_file(spec_file, title='Load spectrum', save_as=False,
                                                file_types=spectrum_files)
            if spec_file:
                try:
                    lspec, ispec = np.loadtxt(spec_file, unpack=True, ndmin=2)
                    base, ext = path.splitext(spec_file)
                    if ext != '.dat':  # .dat files uploaded to github as .txt, .dat not allowed, converted here
                        spec_file = m_fun.change_extension(base, '.dat')
                        np.savetxt(spec_file, np.transpose([lspec, ispec]), fmt='%8.3f %8.5f')
                        result_text += f'File saved as :{spec_file}\n'
                        window['-RESULT4-'].update(reg_text + result_text)
                    lmin = lspec[0]
                    lmax = lspec[len(lspec) - 1]
                    video_list = m_fun.read_video_list('videolist.txt')
                    video_list.insert(0, spec_file)
                    video_list.insert(0, ' ')
                    window['-PLOT_SPEC-'].update(spec_file)
                    window['-PLOT_TITLE-'].update(values=video_list, value=spec_file)
                    window['-PLOTS-'].update(disabled=False, button_color=bc_enabled)
                    window['-SAVES-'].update(disabled=False, button_color=bc_enabled)
                    window['l_min'].update(str(int(lmin)))
                    window['l_max'].update(str(int(lmax)))
                    window['-RESULT4-'].update(result_text)
                except Exception as e:
                    sg.PopupError(f'Cannot read file\n{e}')

        # ==============================================================================
        # plot spectrum
        if (event == '-PLOTS-' and spec_file) or event == 'Multiplot':
            if event == 'Multiplot':
                multi_plot = True
            else:
                multi_plot = False
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
                offset = float(values['-OFFSET-'])
            except Exception as e:
                sg.PopupError(f'bad value for plot range or offset, try again\n{e}', title='Plot Graph')
            else:
                plot_title = values['-PLOT_TITLE-']
                window.disable()
                mod_file, i_min, i_max, cal_text_file = m_plot.graph_calibrated_spectrum(spec_file, line_list,
                                         meteor_lines=meteor_lines, lmin=lmin,
                                         lmax=lmax, imin=i_min, imax=i_max, autoscale=autoscale, gridlines=gridlines,
                                         canvas_size=(plot_w, plot_h), plot_title=plot_title,
                                         multi_plot=multi_plot, offset=offset)
                window.enable()
                window.bring_to_front()
                window['-PLOTS-'].update(disabled=False, button_color=bc_enabled)
                result_text += cal_text_file
                if mod_file:
                    window['-PLOT_SPEC-'].update(mod_file)
                    spec_file = mod_file
                window['-RESULT4-'].update(result_text)

        if event == '-SAVES-':
            new_file, info = m_fun.my_get_file(spec_file, title='Save spectrum', save_as=True,
                                               file_types=spectrum_files)
            if new_file:
                lam, ical = np.loadtxt(spec_file, unpack=True, ndmin=2)
                np.savetxt(new_file, np.transpose([lam, ical]), fmt='%8.3f %8.5f')
                result_text += info + '\n'
                window['-RESULT4-'].update(result_text)

        # =======================================================================
        # Instrument response Tab
        # =======================================================================
        # load uncalibrated raw spectrum
        elif event == '-LOAD_STAR-':
            flux_flag = False
            response_ok = False
            star_file, info = m_fun.my_get_file(values['-STAR-'], title='Load star spectrum',
                                                file_types=spectrum_files, default_extension='*.dat')
            if star_file:
                window['-STAR-'].update(star_file)
                result_text = info + '\n'
                window['-RESULT5-'].update(result_text)
                # plot raw spectrum
                autoscale = values['autoscale']
                try:
                    lmin = float(values['lmin'])
                    lmax = float(values['lmax'])
                    i_min = float(values['imin'])
                    i_max = float(values['imax'])
                    graph_ir.change_coordinates((lmin, i_min), (lmax, i_max))
                except Exception as e:
                    sg.PopupError(f'bad value for plot range or offset, try again\n{e}', title='Plot Graph')
                (graph_ir_ll, graph_ir_ur), lcal, ical = m_plot.plot_raw_spectrum(star_file, graph_ir, canvasx,
                                                    autoscale=autoscale)

        elif event == '-LOAD_REF-':
            if not Path(response_folder).exists():
                sg.PopupError('Response folder does not exist, select correct folder')
            else:
                reference_file = m_fun.m_join(response_folder, path.basename(values['-REF_I-']))
                reference_file, info = m_fun.my_get_file(reference_file, title='Load reference spectrum',
                                      file_types=spectrum_files, default_extension='*.dat')
                if reference_file:
                    window['-REF_I-'].update(reference_file)
                    result_text += info + '\n'
                    window['-RESULT5-'].update(result_text)
                    graph_ir = window['graph_ir']
                    # plot raw spectrum
                    lref, iref = np.loadtxt(reference_file, unpack=True, ndmin=2)
                    idg_ref = m_plot.plot_reference_spectrum(reference_file, lref, iref, graph_ir, canvasx, idg_ref,
                                                             plot_style=ref_style)

        elif event == 'raw_response' and len(lcal) and len(lref):
            l_response = []
            response_ok = False
            for l2 in lcal:
                if lmin <= l2 <= lmax and lref[0] <= l2 <= lref[-1]:
                    l_response.append(l2)
            iref2 = interpolate.interp1d(lref, iref, kind='quadratic')(l_response)
            ical2 = interpolate.interp1d(lcal, ical, kind='quadratic')(l_response)
            i_response = np.zeros([len(iref2)])
            for k in range(len(iref2)):
                if iref2[k] > 0.0 and ical2[k] > 0.0:
                    i_response[k] = (ical2[k]/iref2[k])
            np.savetxt(f'{response_folder}/raw_response.txt', np.transpose([l_response, i_response]),
                       fmt='%8.3f %8.5f')
            info = f'{response_folder}/raw_response.txt saved\n'
            result_text += info + '\n'
            logging.info(info)
            window['-RESULT5-'].update(result_text)
            idg_resp = m_plot.plot_reference_spectrum('raw_response.txt', l_response, i_response,
                                                      graph_ir, canvasx, idg_resp, plot_style=raw_style)
            graph_ir_enabled = True
            select_line_enabled = True
            dragging = False
            start_point = end_point = prior_rect = None
            l_resp = list(l_response)
            i_resp = list(i_response)

        #  remove points from raw_response list
        elif event == 'graph_ir' and graph_ir_enabled:  # if there's a "graph" event, then it's a mouse
            if select_line_enabled:
                x, y = (values['graph_ir'])
                if not dragging:
                    start_point = (x, y)
                    dragging = True
                else:
                    end_point = (x, y)
                if prior_rect:
                    graph_ir.delete_figure(prior_rect)
                if None not in (start_point, end_point):
                    xmin = min(start_point[0], end_point[0])
                    xmax = max(start_point[0], end_point[0])
                    prior_rect = graph_ir.draw_rectangle((xmin, i_min), (xmax, i_max), line_color='LightGreen')
        elif str(event).endswith('+UP') and graph_ir_enabled:
            if select_line_enabled:
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False
                # remove points between xmin and xmax from l_resp, i_resp
                n = len(l_resp)
                # start from top to get correct index
                for k in range(n - 1, -1, -1):
                    if xmin < l_resp[k] < xmax:
                        l_resp.pop(k)
                        i_resp.pop(k)
                logging.info(f'points in [{xmin:6.1f}, {xmax:6.1f}] deleted from raw_response')
                idg_resp = m_plot.plot_reference_spectrum('filter_response', l_resp, i_resp,
                                                          graph_ir, canvasx, idg_resp, plot_style=raw_style)

        elif event == 'smooth_response' and len(l_resp) > 1:
            flux_flag = False
            smooth_parameter = float(values['smooth_parameter'])
            l_smooth = np.linspace(l_resp[0], l_resp[-1], len(l_response))
            if csaps_installed:
                i_smooth = csaps(l_resp, i_resp, l_smooth, smooth=smooth_parameter)
                response_ok = True
                idg_smooth = m_plot.plot_reference_spectrum('smooth_response.txt', l_smooth, i_smooth,
                                                            graph_ir, canvasx, idg_smooth, plot_style=response_style)
                logging.info(f'smoothing with smooth parameter = {smooth_parameter}')
            else:
                sg.PopupError('csaps not installed')

        elif event == '-SEL_RESP_FOLDER-':
            response_folder = values['-RESPONSE_FOLDER-']
            response_folder = sg.PopupGetFolder('', title='Select Response Folder',
                                                initial_folder=response_folder, no_window=True)
            window['-RESPONSE_FOLDER-'].update(m_fun.m_join(response_folder))

        elif event in ('lmin', 'lmax', 'imin', 'imax', 'response_reset'):
            try:
                lmin = float(values['lmin'])
                lmax = float(values['lmax'])
                i_min = float(values['imin'])
                i_max = float(values['imax'])
                graph_ir.change_coordinates((lmin, i_min), (lmax, i_max))
                graph_ir.erase()
                # logging.info('reset response')
                graph_ir_enabled = True
                if flux_flag:
                    (graph_ir_ll, graph_ir_ur), lcal, ical = m_plot.plot_raw_spectrum(spectrum_file,
                                                graph_ir, canvasx, autoscale=autoscale)
                    m_plot.plot_reference_spectrum(flux_file, l_flux, i_flux, graph_ir,
                                                   canvasx, plot_style=flux_style)
                else:
                    (graph_ir_ll, graph_ir_ur), lcal, ical = m_plot.plot_raw_spectrum(star_file,
                                                graph_ir, canvasx, autoscale=autoscale)
                    idg_ref = m_plot.plot_reference_spectrum(reference_file, lref, iref, graph_ir,
                                                             canvasx, plot_style=ref_style)
                    idg_resp = m_plot.plot_reference_spectrum('raw_response.txt', l_response, i_response,
                                                              graph_ir, canvasx, plot_style=raw_style)
            except:
                pass

        elif event == '-SAVE_RESPONSE-':
            response_file = m_fun.m_join(response_folder, path.basename(values['-RESPONSE-']))
            new_file, info = m_fun.my_get_file(response_file, title='Save response', save_as=True,
                                               file_types=spectrum_files)
            if new_file:
                if response_ok:
                    np.savetxt(new_file, np.transpose([l_smooth, i_smooth]), fmt='%8.3f %8.5f')
                    result_text += info + '\n'
                    window['-RESULT5-'].update(result_text)
                    window['-RESPONSE-'].update(new_file)
                    window['-RESPONSE2-'].update(new_file)

        # =======================================================================
        #  Response correction
        elif event == '-LOAD_RESPONSE-':
            response_file, info = m_fun.my_get_file(values['-RESPONSE2-'], title='Load response',
                                                    file_types=spectrum_files, default_extension='*.dat')
            if response_file:
                window['-RESPONSE2-'].update(response_file)
                result_text += info + '\n'
                window['-RESULT5-'].update(result_text)
                l_smooth, i_smooth = np.loadtxt(response_file, unpack=True, ndmin=2)
                idg_smooth = m_plot.plot_reference_spectrum(response_file, l_smooth, i_smooth, graph_ir,
                                                            canvasx, idg_smooth, plot_style=response_style)
                response_ok = True

        elif event == '-LOAD_SPECTRUM-':
            spectrum_file, info = m_fun.my_get_file(values['-SPECTRUM-'], title='Load spectrum',
                                                    file_types=spectrum_files, default_extension='*.dat')
            if spectrum_file:
                window['-SPECTRUM-'].update(spectrum_file)
                result_text += info + '\n'
                window['-RESULT5-'].update(result_text)
                # autoscale = values['autoscale']
                graph_ir.change_coordinates((lmin, i_min), (lmax, i_max))
                plot_range, l_spec, i_spec = m_plot.plot_raw_spectrum(spectrum_file, graph_ir,
                                                                      canvasx, autoscale=False)

        elif event == 'apply_response' and len(l_smooth) and len(l_spec):
            flux_flag = True
            l_flux = []
            for l2 in l_spec:
                if lmin <= l2 <= lmax and l_smooth[0] <= l2 <= l_smooth[-1]:
                    l_flux.append(l2)
            iresp2 = interpolate.interp1d(l_smooth, i_smooth, kind='quadratic')(l_flux)
            ispec2 = interpolate.interp1d(l_spec, i_spec, kind='quadratic')(l_flux)
            i_flux = np.zeros([len(iresp2)])
            for k in range(len(iresp2)):
                if iresp2[k] > 0.0:
                    i_flux[k] = (ispec2[k]/iresp2[k])
            flux_file = m_fun.change_extension(spectrum_file, '_F.dat')
            np.savetxt(flux_file, np.transpose([l_flux, i_flux]), fmt='%8.3f %8.5f')
            result_text += f'{flux_file} saved\n'
            logging.info(f'{flux_file} saved')
            window['-RESULT5-'].update(result_text)
            window['-FLUX-'].update(flux_file)
            # graph_ir.change_coordinates((lmin, i_min), (lmax, i_max))
            plot_range, l_spec, i_spec = m_plot.plot_raw_spectrum(spectrum_file, graph_ir, canvasx,
                                                                  autoscale=False)
            m_plot.plot_reference_spectrum(flux_file, l_flux, i_flux, graph_ir, canvasx, plot_style=flux_style)

        elif event == '-SAVE_FLUX-':
            new_file, info = m_fun.my_get_file(values['-FLUX-'], title='Save flux', save_as=True,
                                               file_types=spectrum_files)
            if new_file:
                np.savetxt(new_file, np.transpose([l_flux, i_flux]), fmt='%8.3f %8.5f')
                result_text += info + '\n'
                window['-RESULT5-'].update(result_text)
                window['-FLUX-'].update(new_file)

        # =======================================================================
        #  Laser calibration tab
        # =======================================================================
        #  Laser calibration Video conversion
        if event == 'Load Avi':
            avifile, info = m_fun.my_get_file(window['avi_file'].Get(), title='Get Video File',
                                              file_types=(('Video Files', '*.avi'), ('ALL Files', '*.*')),
                                              default_extension='.avi')
            logtext += info + '\n'
            if avifile:
                logging.info(f'start video conversion: {str(avifile)}')
                logtext += 'start video conversion: WAIT!\n'
                window['avi_file'].Update(avifile)
                window.refresh()
                nim, dattim, sta, out = m_fun.extract_video_images(avifile, pngdir, bobdoubler=False,
                                                                   binning=par_dict['i_binning'], bff=False,
                                                                   maxim=20)
                fits_dict['DATE-OBS'] = dattim
                fits_dict['M_STATIO'] = sta
                fits_dict['M_NIM'] = '1'
                fits_dict['M_STARTI'] = '0'
                logging.info(f'finished video conversion: {str(avifile)}')
                logging.info(f'nim: {str(nim)} date time: {dattim} station: {sta}')
                logtext += ('finished video conversion: ' + avifile + '\n' +
                            f'nim: {str(nim)} date time: {dattim} ' + '\n' +
                            f'station: {sta}' + '\n')
                print('nim:', nim, dattim, sta)
                imbw = m_fun.create_background_image(pngdir, nim)
                # save average image as png and fit
                m_fun.save_fit_png('avi.png', imbw, fits_dict, dist=False)
                image_data, actual_file = m_fun.draw_scaled_image('avi.fit', image_elem_calib, opt_dict)

        if event in ('Save Image', 'Save calibration Image'):
            imfilename, info = m_fun.my_get_file(window['image_file'].Get(), title='Save image',
                                                 file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*')),
                                                 save_as=True, default_extension='*.fit', )
            if imfilename:
                imfilename = m_fun.change_extension(imfilename, '')
                try:
                    m_fun.save_fit_png(imfilename, imbw, fits_dict, dist=False)
                    logtext += info + '\n'
                    window['image_file'].Update(imfilename)
                    window['image_filename'].Update(imfilename)
                except Exception as e:
                    sg.PopupError(f'no video converted or image saved,\n{e}')
            else:
                'no image saved, missing filename'

        # -------------------------------------------------------------------------------
        #       Load, save image
        # -------------------------------------------------------------------------------
        if event == 'Load Image':
            files, info = m_fun.my_get_file(window['input_file'].Get(), title='Load image',
                                            file_types=(('Image Files', '*.fit'), ('PNG-File', '*.png'),
                                                        ('BMP-File', '*.bmp'), ('ALL Files', '*.*')),
                                            default_extension='*.fit', multiple_files=True)
            nim = len(files)
            new_infile = ''
            if nim == 0:
                sg.Popup('No file selected, keep last image')
                image_data, actual_file, imbw = m_fun.draw_scaled_image('tmp.png', image_elem_calib,
                                                                        opt_dict, get_array=True)
            else:
                image_data, actual_file, imbw = m_fun.draw_scaled_image(files[0], image_elem_calib, opt_dict,
                                                                        tmp_image=True, get_array=True)
                infile = m_fun.m_join(files[0])
                if nim == 1:
                    if len(imbw):
                        if not files[0].lower().endswith('.fit'):  # further processing is with fits-images
                            error = m_fun.write_fits_image(imbw, m_fun.change_extension(infile, '.fit'),
                                                           fits_dict, dist=False)
                            if error:
                                infile = ''
                            else:
                                logging.info(f'Load_Image: {infile} size: {str(imbw.shape)}')
                                logtext += 'Load_Image: ' + infile + ' size: ' + str(imbw.shape) + '\n'
                        new_infile = m_fun.change_extension(infile, '')
                    else:
                        sg.PopupError(' File not found or not read')
                elif nim > 1:
                    error = False
                    shape0 = imbw.shape
                    for file in files:
                        # load images to compare shape
                        # TODO: shorter version of load array only, no need to draw image
                        image_data, actual_file, imbw = m_fun.draw_scaled_image(file, image_elem_calib,
                                                                                opt_dict, get_array=True)

                        if imbw.shape != shape0:
                            sg.PopupError('all files must have the same format, try again!', keep_on_top=True)
                            error = True
                            break
                    if not error:
                        for f in range(nim):
                            files[f] = path.relpath(files[f])
                            # TODO: shorter version of load array only, no need to draw image
                            image_data, actual_file, im = m_fun.draw_scaled_image(files[f], image_elem_calib,
                                                                                  opt_dict,
                                                                                  get_array=True)
                            imbw = im if f == 0 else np.maximum(imbw, im)
                        new_infile = m_fun.change_extension(infile, '') + '_peak_' + str(nim)
                        m_fun.save_fit_png(new_infile, imbw, fits_dict, dist=False)
                        image_data, actual_file = m_fun.draw_scaled_image(m_fun.change_extension(new_infile,
                                                                          '.fit'), image_elem_calib, opt_dict,
                                                                          tmp_image=True)
                        logging.info(f'image saved as: {new_infile} (.fit, .png)')
                        logtext += 'Load_Images:' + '\n'
                        for f in range(nim):
                            logtext += files[f] + '\n'
                        logtext += f'image saved as: {new_infile}, .png)\n'
            if new_infile:
                infile = m_fun.m_join(new_infile)
                window['image_filename'].Update(infile)
                par_dict['s_infile'] = infile
                (imy, imx) = imbw.shape[:2]
                par_dict['i_imx'] = imx
                par_dict['i_imy'] = imy
            window['input_file'].Update(new_infile)

        # -------------------------------------------------------------------------------
        #       Select Lines
        # ------------------------------------------------------------------------------
        if event == 'Select File':
            old_outfil = window['output_file'].Get()
            outfil, info = m_fun.my_get_file(old_outfil,
                                             title='Load measured calibration lines file',
                                             file_types=(('Calibration Files', '*.txt'), ('ALL Files', '*.*')),
                                             save_as=False, default_extension='*.txt')
            if not outfil:
                outfil = old_outfil
                if not outfil:
                    sg.PopupError('no File selected, try again')
            if outfil:
                outfil = m_fun.change_extension(outfil, '')
                window['output_file'].Update(outfil)

        if event == 'Load L_list':
            linelist, info = m_fun.my_get_file(window['linelist'].Get(),
                                               title='Get Linelist',
                                               file_types=(('Linelist', '*.txt'), ('ALL Files', '*.*')),
                                               default_extension='*.txt')
            if not linelist or linelist[:-1] == 'l':
                linelist = 'l'
            linelist = m_fun.change_extension(linelist, '')
            window['linelist'].Update(linelist)
            par_dict['s_linelist'] = linelist

        if event == 'Select Lines':
            infile = window['input_file'].Get()
            outfil = window['output_file'].Get()
            if infile:
                if not Path(infile + '.fit').exists():
                    imbw, opt_dict = lfun.load_image(infile, opt_dict)
                # 'tmp.png' created with tmp_image=True, needed for sel.select_lines
                image_data, actual_file = m_fun.draw_scaled_image(m_fun.change_extension(infile, '.fit'),
                                                                  image_elem_calib, opt_dict,
                                                                  tmp_image=True, get_array=False)
                p = Path(m_fun.change_extension(outfil, '.txt'))
                if not Path(p).exists():
                    if not p.parent.exists():
                        Path.mkdir(p.parent, exist_ok=True)
                        p = m_fun.m_join(Path.joinpath(p.parent, p.name), '.txt')
                    outfil = str(Path(p).with_suffix(''))
                    par_dict['s_outfil'] = outfil
                notok, linelist, lines = lfun.get_linelist(par_dict['s_linelist'], par_dict['f_lam0'])
                window.Disable()
                wloc_image = (opt_dict['win_x'] + opt_dict['calc_off_x'], opt_dict['win_y'] + opt_dict['calc_off_y'])
                wloc_image = sel.select_lines(infile, contrast, lines,
                                              res_dict, fits_dict, wloc_image, outfil)
                opt_dict['calc_off_x'] = wloc_image[0] - opt_dict['win_x']
                opt_dict['calc_off_y'] = wloc_image[1] - opt_dict['win_y']
                window.Enable()
                window.BringToFront()
                logging.info(f'Finished, saved {outfil}.txt')
                logtext += ('Finished, saved ' + outfil + '.txt\n')
            else:
                sg.PopupError('no image selected, try again')

        # ------------------------------------------------------------------------------
        #       LSQ-Fit
        # ------------------------------------------------------------------------------
        if event == 'LSQF':
            outfil = window['output_file'].Get()
            if Path(m_fun.change_extension(outfil, '.txt')).exists():
                window['output_file'].update(outfil)
                par_dict['s_outfil'] = outfil
                par_dict['b_sqrt'] = window['SQRT-Fit'].Get()
                par_dict['b_fitxy'] = window['fitxy'].Get()
                parv = list(par_dict.values())
                logging.info(f'outfil: {outfil} START LSQF')
                logtext += 'outfil: ' + outfil + '\n'
                logtext += 'START LSQF ...\n'
                window.refresh()
                try:
                    par, result = lfun.lsqf(parv, debug=debug, fit_report=opt_dict['fit-report'])
                    (scalxy, x00, y00, rotdeg, disp0, a3, a5, errorx, errory) = par
                    image_data, actual_file = m_fun.draw_scaled_image(outfil + '_lsfit.png', image_elem_calib,
                                                                      opt_dict)
                    rot = rotdeg * np.pi / 180
                    resv = np.float32([scalxy, x00, y00, rot, disp0, a3, a5])
                    reskey = ['scalxy', 'x00', 'y00', 'rot', 'disp0', 'a3', 'a5']
                    reszip = zip(reskey, resv)
                    res_dict = dict(list(reszip))
                    # write default configuration as actual configuration
                    # m_fun.write_configuration('m_cal.ini', par_dict, res_dict, fits_dict, opt_dict)
                    logging.info('Result LSQF:')
                    logging.info(result)  # more detailed info
                    logtext += 'Result LSQF saved as m_cal.ini:\n'
                    logtext += result
                    print('END OF LSQ-Fit!!!')
                    logtext += 'END OF LSQ-Fit!\n'
                except Exception as e:
                    sg.PopupError(f'Error in LSQ-fit, wrong {outfil} ?\n{e}')
                    result = 'Error with: ' + str(outfil) + '.txt'
                    logging.error(result + ', {e}')
                    result += '\n----------------------------------------\n'
                    logtext += result
            else:
                sg.PopupError('no such file: ' + outfil + '.txt')

        # ------------------------------------------------------------------------------
        #       Setup
        # ------------------------------------------------------------------------------
        if event in ('Load Setup', 'Edit Setup'):
            if event == 'Load Setup':
                ini_file, info = m_fun.my_get_file(window['setup_file'].Get(),
                                     title='Get Configuration File',
                                     file_types=(('Configuration Files', '*.ini'), ('ALL Files', '*.*')),
                                     default_extension='*.ini')
                par_text, par_dict, res_dict, fits_dict, opt_dict = m_fun.read_configuration(ini_file,
                                                        m_fun.par_dict, m_fun.res_dict, m_fun.opt_dict)
                fits_dict['VERSION'] = version
                if par_text == '':
                    sg.PopupError(f'no valid configuration found, use current configuration',
                                  keep_on_top=True)
            else:  # edit conf, update values from main menu
                par_dict['s_infile'] = window['input_file'].Get()
                par_dict['s_outfil'] = window['output_file'].Get()
                par_dict['s_linelist'] = window['linelist'].Get()
                par_dict['b_sqrt'] = window['SQRT-Fit'].Get()
                par_dict['b_fitxy'] = window['fitxy'].Get()
            window.Disable()
            ini_file, par_dict, res_dict, fits_dict, opt_dict, logtext\
                = lfun.calib_setup(ini_file, par_dict, res_dict, fits_dict, opt_dict, logtext)

            infile = par_dict['s_infile']
            if infile:
                image_data, actual_file, imbw = m_fun.draw_scaled_image(m_fun.change_extension(infile, '.fit'),
                                                        image_elem_calib, opt_dict, tmp_image=True, get_array=True)
                if not image_data:
                    imbw = []
            else:
                imbw = []
            if not len(imbw):
                sg.PopupError(f'Image {infile}.fit not found, load {infile}.png instead:', keep_on_top=True)
                infile, info = m_fun.my_get_file(infile, title='Load image',
                                                 file_types=(('PNG-File', '*.png'), ('Image Files', '*.fit'),
                                                             ('BMP-File', '*.bmp'), ('ALL Files', '*.*')),
                                                            default_extension='*.png')
                if infile:
                    lfun.load_image(infile, opt_dict)  # creates fit image
                    image_data, actual_file, imbw = m_fun.draw_scaled_image(infile, image_elem_calib,
                                                                            opt_dict, get_array=True)
                    infile = m_fun.m_join(m_fun.change_extension(infile))
            if len(imbw):
                window['input_file'].Update(infile)
                window['image_filename'].Update(infile)
                par_dict['s_infile'] = infile
            window['setup_file'].Update(ini_file)
            window['output_file'].Update(par_dict['s_outfil'])
            window['SQRT-Fit'].Update(par_dict['b_sqrt'])
            window['fitxy'].Update(par_dict['b_fitxy'])
            window['linelist'].Update(par_dict['s_linelist'])
            window.Enable()
            window.BringToFront()

        if event == 'Edit File':
            outfil = window['output_file'].Get()
            m_fun.edit_text_window(outfil, size=(50, 30))

        # ------------------------------------------------------------------------------
        #       Spectrum analysis
        # ------------------------------------------------------------------------------
        if event == '-SEL_SPECTRUM-':
            i_residue = []
            sel_ele = []
            m_plot.delete_curve(idg_spec, graph_an)
            spec_file_analysis, info = m_fun.my_get_file(spec_file_analysis, title='Load spectrum',
                                                         save_as=False, file_types=spectrum_files)
            if spec_file_analysis:
                fit_enabled = nist.lsqf_fit_disable(window, disabled=True)
                delta, lclip, iclip, idg_spec = nist.load_spectrum_analysis(spec_file_analysis, lmin_a, lmax_a,
                                                                            graph_an, canvasx, ref_style)
                graph_analysis_enabled = True  # used for switching title, show image or coordinates
            window['-SPECTRUM_A-'].update(spec_file_analysis)
            window.set_title(window_title + str(spec_file_analysis))
            result_fit = rms = ''  # reset lsqf results with new file
            logging.info(f'{spec_file_analysis} loaded for analysis')

        if event == '-SEL_RESPONSE_A-':
            resp_file_analysis, info = m_fun.my_get_file(resp_file_analysis, title='Load response',
                                                         save_as=False, file_types=spectrum_files)
            fit_enabled = nist.lsqf_fit_disable(window, disabled=True)
            m_plot.delete_curve(idg_resp_a, graph_an)
            if resp_file_analysis:
                lresp, iresp, idg_resp_a = nist.load_response_analysis(resp_file_analysis,
                                                                       graph_an, canvasx, response_style)
            else:
                lresp = []
            window['-RESPONSE_A-'].update(resp_file_analysis)
            logging.info(f'{resp_file_analysis} loaded as response')

        if event == '-SEL_N2I-':
            t_n2i, info = m_fun.my_get_file(t_n2i, title='Load N2I spectrum', save_as=False,
                                            file_types=spectrum_files)
            if path.exists(t_n2i):
                t_n2i = m_fun.change_extension(t_n2i, '')
            fit_enabled = nist.lsqf_fit_disable(window, disabled=True)
            window['-N2I-FILE-'].update(t_n2i)

        if event == 'NIST':
            t0 = time.time()
            i_residue = []
            try:
                # load actual parameters for fit
                lmin_a, lmax_a, t_el, t_cont, threshold, sigma_nist, sigma0 = nist.get_fit_parameters(window)
                t_n2i = window['-N2I-FILE-'].get()
                n_gauss = int(3 * sigma0 / delta)
                if n_gauss < 3:
                    sigma0 = delta
                    n_gauss = 3
                    window['-SIGMA0-'].update(sigma0)
                if sigma0 > sigma_nist:
                    sg.popup_error('sigma_nist >= sigma0 required, changed value')
                    sigma_nist = sigma0
                    window['-SIGMA_NIST-'].update(f'{sigma_nist:6.3f}')
                # fit t_cont, sigma, used for calculate_spectrum:
                sigma_fit = np.sqrt(sigma_nist ** 2 - sigma0 ** 2)
                par_ele = nist.par_ele_create(sigma_fit, t_cont, t_el, window)
                # replot actual spectrum and response
                graph_an.erase()
                idg_spec = m_plot.plot_reference_spectrum(spec_file_analysis, lclip, iclip, graph_an, canvasx,
                                                          plot_style=ref_style)
                if len(lresp):
                    idg_resp_a = m_plot.plot_reference_spectrum(resp_file_analysis, lresp, iresp, graph_an,
                                                                canvasx, plot_style=response_style)
                sel_ele = []
                for ele in all_ele:  # update element list
                    # name = ele.name
                    # print('ele.name', name)
                    if window[ele.name].Get():
                        sel_ele.append(ele)
                    else:
                        window[ele.name + '_scale'].update('')
                element_index = 0
                t1 = time.time() - t0
                for ele in sel_ele:
                    # load elements with parameters and calculate gaussian smoothed spectrum, plot
                    ok = nist.get_element(ele, lmin_a, lmax_a, t_el, t_cont, t_n2i, threshold,
                                          sigma0, n_gauss, lresp, iresp, lclip, debug=debug)
                    if ok:
                        ele.mult = float(window[ele.name + '_mult'].Get())
                        ele.fit = window[ele.name + '_fit'].Get()
                        window[ele.name + '_scale'].update(f'{ele.scale:8.2e}')
                        ele.index = element_index
                        if ele.scale > 0.0:
                            ele.gspec = gaussian(ele.ele_spec, sigma=sigma_fit / delta) * ele.mult
                            idg_lines = m_plot.plot_reference_spectrum(ele.name, lclip, ele.gspec,
                                                                       graph_an, canvasx,
                                                plot_style=(ele.color, 0, 1, -0.23 - 0.03 * element_index))
                        element_index += 1
                    window.refresh()
                # calculate sum_spectrum
                t2 = time.time() - t0
                if element_index:
                    fit_enabled = nist.lsqf_fit_disable(window, disabled=False)
                    i_fit, i_residue, rms_error, result_fit = nist.calculate_spectrum(sel_ele, par_ele, iclip,
                                                        result_fit, lclip, lresp, iresp, sigma0, n_gauss)
                    if window['-SUM-'].Get():
                        idg_fit = m_plot.plot_reference_spectrum('Sum', lclip, i_fit, graph_an, canvasx, idg_fit,
                                                                 plot_style=('black', 0, 2, -0.15))
                    if window['-RESIDUALS-'].Get() and len(i_residue):
                        idg_res = m_plot.plot_reference_spectrum('', lclip, i_residue - residual_offset,
                                                                 graph_an, canvasx, idg_res,
                                                                 plot_style=('green', 0, 2, -0.15))
                    window['-RESULT_A-'].update(result_fit)
                t3 = time.time() - t0
                print(f'({t1:5.3f}, {t2:5.3f}, {t3:5.3f})')
            except Exception as e:
                sg.PopupError(f'Load spectrum first or other error\n{e}')

        if event in ('-LMIN_A-', '-LMAX_A-', '-SIGMA0-', '-T_ELECTRON-', '-THRESHOLD-'):
            graph_an.erase()  # new graph required, with new wavelength scale or pre-calculated parameters
            fit_enabled = nist.lsqf_fit_disable(window, disabled=True)
            lmin_a = float(window['-LMIN_A-'].Get())
            lmax_a = float(window['-LMAX_A-'].Get())
            off = -2 * residual_offset if window['-RESIDUALS-'].get() else -.1
            graph_an.change_coordinates((lmin_a, off), (lmax_a, 1.2))

        if event[: -5] in elements and event[-5:] == '_mult':
            try:
                mult = float(window[event].Get())
                element = event[: -5]  # ele_mult.key = (ele) + '_mult'
                nist.set_ele(element, all_ele, mult=mult)
                ele = nist.get_ele(element, all_ele)
                delta_spec = (lclip[-1] - lclip[0]) / (len(lclip) - 1)
                ele.gspec = gaussian(ele.ele_spec * ele.mult, sigma=sigma_fit / delta_spec)
                if fit_enabled:
                    i_fit, i_residue, rms_error, result_fit = nist.calculate_spectrum(sel_ele, par_ele, iclip,
                                                    result_fit, lclip, lresp, iresp, sigma0, n_gauss)
                    window['-RESULT_A-'].update(result_fit)
                    if ele in sel_ele:
                        nist.plot_analysis(window, i_residue, lclip, iclip, i_fit,
                                           spec_file_analysis, lmin_a, lmax_a, residual_offset, graph_an, canvasx,
                                           ref_style, sel_ele, zoom_window=zoom_window)
            except:
                pass

        if event in elements:
            ele = event
            fit_enabled = nist.lsqf_fit_disable(window, disabled=True)
            try:
                nist.set_ele(ele, sel_ele, mult=float(window[ele + '_mult'].Get()))
            except Exception as e:
                sg.PopupError(f'bad input value\n{e}')

        if event == 'graph_analysis':  # if there's a "graph_analysis" event, then it's a mouse
            x, y = (values['graph_analysis'])
            window.set_title(window_title[: -7] + str(spec_file_analysis) + f'  lambda: {x:8.1f}, Int: {y:6.3f}')
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                graph_an.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = graph_an.draw_rectangle(start_point, end_point, line_color='red')

        if event.endswith('+UP') and graph_analysis_enabled:  # The drawing has ended because mouse up
            if None not in (start_point, end_point):
                list_end = list(end_point)
                list_start = list(start_point)
                size = np.array(list_end) - np.array(list_start)
                if abs(size[0]) >= 2.0 or abs(size[1]) >= 0.2:  # ignore small cursor movements
                    if size[0] < 0.0:  # dragging left resets zoom
                        zoom_window = []
                    else:  # zoom window
                        if size[1] < 0.0:
                            list_end[1], list_start[1] = list_start[1], list_end[1]
                        zoom_window = np.array([list_start, list_end])
                    nist.plot_analysis(window, i_residue, lclip, iclip, i_fit,
                                       spec_file_analysis, lmin_a, lmax_a, residual_offset, graph_an, canvasx,
                                       ref_style, sel_ele, zoom_window=zoom_window)
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False

        if event == '-LSQF_SPEC-':
            # update latest values for new fit
            lmin, lmax, t_el, t_cont, threshold, sigma_nist, sigma0 = nist.get_fit_parameters(window)
            sigma_fit = np.sqrt(sigma_nist ** 2 - sigma0 ** 2)
            par_ele = nist.par_ele_create(sigma_fit, t_cont, t_el, window)
            for ele in sel_ele:
                try:  # check fit variables and initial values
                    mult = float(window[ele.name + '_mult'].Get())
                    ele.mult = mult
                    ele.fit = window[ele.name + '_fit'].Get()
                except Exception as e:
                    sg.PopupError(f'bad input value\n{e}')
            # least square fit
            i_fit, i_residue, sel_ele, par_ele, time_fit, result = nist.lsq_fit(iclip, sel_ele, par_ele, lclip,
                                                                        lresp, iresp, sigma0, n_gauss, debug=debug)
            for ele in sel_ele:
                window[ele.name + '_mult'].update(f"{ele.mult:8.3e}")
                window[ele.name + '_scale'].update(f'{ele.scale:8.2e}')
            t_cont = nist.get_ele('t_cont', par_ele).mult
            window['-T_CONT-'].update(f'{t_cont:6.0f}')
            t_el = nist.get_ele('t_el', par_ele).mult
            window['-T_ELECTRON-'].update(f'{t_el:6.0f}')
            sigma_fit = nist.get_ele('sigma_fit', par_ele).mult
            sigma_nist = np.sqrt(sigma_fit ** 2 + sigma0 ** 2)
            window['-SIGMA_NIST-'].update(f'{sigma_nist:6.3f}')
            rms_error = np.std(i_residue)
            m_plot.delete_curve(idg_lines, graph_an)
            rms = f'#  rms: {rms_error:8.6f}, t:{time_fit:6.2f}sec'
            if debug:
                print(f't_el, t_cont, t_fit: {t_el:6.0f}, {t_cont:6.0f} {rms}')  # no C/R needed
            logging.info('LSQF error: ' + rms)
            result_fit += result + rms + '\n'
            window['-RESULT_A-'].update(result_fit)
            window['-SUM-'].update(True)
            nist.plot_analysis(window, i_residue, lclip, iclip, i_fit, spec_file_analysis,
                               lmin_a, lmax_a, residual_offset, graph_an, canvasx,
                               ref_style, sel_ele, zoom_window=zoom_window)

        if event == '-SAVE_FIT-':
            new_file, info = m_fun.my_get_file(m_fun.change_extension(spec_file_analysis, '_fit.dat'),
                                    title='Save calculated spectrum', save_as=True, file_types=spectrum_files)
            if new_file:
                lmin_a, lmax_a, _el, t_cont, threshold, sigma_nist, sigma0 = nist.get_fit_parameters(window)
                lsqf_var = [m_fun.change_extension(spec_file_analysis, ''),
                            m_fun.change_extension(resp_file_analysis, ''),
                            lmin_a, lmax_a, sigma_nist, sigma0, t_cont, t_el, threshold, t_n2i]
                lsqf_dict = dict(list(zip(lsqf_keys, lsqf_var)))
                np.savetxt(new_file, np.transpose([lclip, i_fit]), fmt='%8.3f %8.5f')
                if rms:
                    logging.info(new_file + rms)
                with open(new_file, 'a') as f:
                    f.write(result_fit)
                result_text += info + '\n'
                for ele in sel_ele:
                    ele.fit = window[ele.name + '_fit'].Get()
                if nist.line_strength(all_ele, lclip, lresp, iresp):
                    sg.popup_error('incorrect or missing instrument response')
                    logging.info('incorrect or missing instrument response')
                fei_strength = nist.get_ele("fei", all_ele).strength
                mgi_strength = nist.get_ele("mgi", all_ele).strength
                nai_strength = nist.get_ele("nai", all_ele).strength
                info = f'fei:{fei_strength:8.3f}, mgi:{mgi_strength:8.3f}, nai:{nai_strength:8.3f}, ' \
                       f'{spec_file_analysis}, {new_file}'
                logging.info(info)
                print(info + '\n')
                window.set_title(window_title[: -7] + str(m_fun.change_extension(new_file, '.inf')))
                nist.write_config_fit(m_fun.change_extension(new_file, '.inf'), sel_ele, lsqf_dict)

        if event == '-LOAD_FIT-':
            i_residue = []
            result_fit = ''
            conf, info = m_fun.my_get_file('', title='Load fit configuration', save_as=False,
                                           file_types=(('Fit configuration files', '*.inf'), ('ALL Files', '*.*'),))
            if conf:
                for element in elements:
                    window[element].update(value=False)
                lsqf_var, sel_ele = nist.read_configuration(conf, lsqf_dict, all_ele)
                nist.set_fit_parameters(window, lsqf_var)
                window['-RESIDUALS-'].update(value=False)
                [lmin_a, lmax_a] = lsqf_var[2:4]
                graph_an.change_coordinates((lmin_a, -.1), (lmax_a, 1.2))
                for ele in sel_ele:
                    window[ele.name].update(value=True)
                    window[ele.name + '_mult'].update(ele.mult)
                    window[ele.name + '_fit'].update(value=ele.fit)
                graph_an.erase()
                spec_file_analysis = m_fun.change_extension(lsqf_var[0], '.dat')
                if path.exists(spec_file_analysis):
                    delta, lclip, iclip, idg_spec = nist.load_spectrum_analysis(spec_file_analysis, lmin_a, lmax_a,
                                                                                graph_an, canvasx, ref_style)
                    graph_analysis_enabled = True
                    window.set_title(window_title[: -7] + str(spec_file_analysis) + ', ' + str(conf))
                    logging.info(f'{spec_file_analysis} loaded for analysis')
                else:
                    spec_file_analysis = ''
                window['-SPECTRUM_A-'].update(spec_file_analysis)
                resp_file_analysis = m_fun.change_extension(lsqf_var[1], '.dat')
                if path.exists(resp_file_analysis):
                    lresp, iresp, idg_resp_a = nist.load_response_analysis(resp_file_analysis,
                                                                           graph_an, canvasx, response_style)
                    logging.info(f'{resp_file_analysis} loaded as response')
                else:
                    resp_file_analysis = ''
                window['-RESPONSE_A-'].update(resp_file_analysis)

        if event in ('-SUM-', '-RESIDUALS-') and graph_analysis_enabled:
            nist.plot_analysis(window, i_residue, lclip, iclip, i_fit,
                               spec_file_analysis, lmin_a, lmax_a, residual_offset, graph_an, canvasx,
                               ref_style, sel_ele, zoom_window=zoom_window)

        window.refresh()

        # other stuff, open issues ---------------------------------------------
        # tabs_element.set_focus('Tab 1') #does not work
        # window.Size) #works only first time
    window.close()
    # end of main()


if __name__ == '__main__':
    main()
