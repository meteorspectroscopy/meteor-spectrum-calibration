import numpy as np
import configparser
import os.path as path
import time
import logging
from scipy import interpolate
from skimage.filters import gaussian
from lmfit import minimize, Parameters, report_fit
import m_specfun as m_fun
import m_plot

version = '0.9.28'


class Element:
    def __init__(self, name, mult, scale, index, fit, color=None, lines=None):
        self.name = name
        self.mult = mult
        self.scale = scale
        self.index = index
        self.fit = fit
        self.color = color
        self.lines = lines
        self.ele_spec = []
        self.gspec = []


class Line:
    def __init__(self, wave, aki_gk, ek, intensity=0, agr=0.0):
        self.wave = wave
        self.aki_gk = aki_gk
        self.ek = ek
        self.intensity = intensity
        self.agr = agr


def get_ele(name, e):
    for ele in e:
        if ele.name == name:
            return ele
    return Element('None', 0, 0, 0, False)


def set_ele(name, e, mult=0.):
    for ele in e:
        if ele.name == name:
            ele.mult = mult


def sfloat(x):
    return float(x[1: -1])  # clip extra ", used for reading special NIST table format


def line_intensity(aki, e_k, g_k, t_el):
    k_ev = 1/11605.465
    return aki * g_k * np.exp(-e_k/k_ev/t_el)


def planck_radiation(lamda, temp):
    """
    gets intensity of Planck radiation I(lambda, dl_lambda)
    as a function of wavelength and temperature
    hc/k = 0.014388 m = 14.388e6 nm
    dU(lamda) = (8* pi *h *c / lamda**5) * 1 / (exp(h*c/(lamda*k*temp)) - 1)
    :param lamda: wavelength [nm]
    :param temp: blackbody temperature [K]
    :return: Planck spectral density (arbitrary units)
    """
    const = 14.3866e6
    return 1/(lamda**5 * (np.exp(const/lamda/temp) - 1))


def planck(t_pl, lclip, lresp, iresp):
    """
    modified intensity of Planck radiation I(lambda, dl_lambda)
    corrected for instrument response
    :param lclip: wavelength array[nm]
    :param t_pl: blackbody temperature [K]
    :param lclip: wavelength array[nm]
    :param lresp: wavelength array[nm] for response
    :param iresp: spectral response array
    :return: Planck spectral density * response (arbitrary units)
    """
    igauss = np.zeros(len(lclip))
    for k in range(len(lclip)):
        lam = lclip[k]
        igauss[k] = planck_radiation(lam, t_pl)
        if len(lresp):
            if lresp[0] <= lam <= lresp[len(lresp) - 1]:
                igauss[k] = igauss[k] * interpolate.interp1d(lresp, iresp, kind='linear')(lam)
            else:
                igauss[k] = 0.0
    return np.array(igauss / np.max(igauss))


def read_nist_lines(element, lmin, lmax, t_el):
    with open(element) as f:
        s = f.readline()
        s0 = 'ritz_wl_air(nm)	Aki(s^-1)	Acc	Ei(eV)	Ek(eV)	g_i	g_k	Type	\n'
        assert s == s0, f'wrong data table, should be\n{s0}'
    a = np.loadtxt(element, dtype=str, skiprows=1)
    np.set_printoptions(precision=5, suppress=False)
    row, col = a.shape
    # just in case, sort array by increasing wavelength
    a = a[a[:, 0].argsort()]
    lines = []
    for j in range(row):
        lam = sfloat(a[j, 0])
        if lmin <= lam <= lmax:
            aki = sfloat(a[j, 1])
            e_k = sfloat(a[j, 4])
            g_k = int(a[j, 6])
            intens = line_intensity(aki, e_k, g_k, t_el) / lam  # energy flux, relative intensities
            # b.append([lam, aki, e_k, g_k, intens])
            lines.append(Line(lam, aki * g_k, e_k, intens))
    return lines


def get_element(ele, lmin, lmax, t_el, t_pl, t_n2i, threshold, sigma0,
                n_gauss, lresp, iresp, lclip, debug=False):
    if len(lclip):
        #  initialize intensity array
        i_line_spec = np.zeros(len(lclip))
        if ele.name == 'n2i':
            ln2, in2 = np.loadtxt(t_n2i + '.dat', unpack=True, ndmin=2)
            if len(ln2) == 0:
                print('NIST/N2I' + t_n2i + '.dat' + 'not found')
            delta_n2 = ln2[1] - ln2[0]
            if delta_n2 < 0.0:
                ln2 = np.flip(ln2)
                in2 = np.flip(in2)
                delta_n2 = - delta_n2
            in2 = gaussian(in2, sigma=sigma0 / delta_n2)
            for k in range(len(lclip)):
                lam = lclip[k]
                if ln2[0] <= lam <= ln2[-1]:
                    i_line_spec[k] = interpolate.interp1d(ln2, in2, kind='linear')(lam)
                else:
                    i_line_spec[k] = 0.0
                if len(lresp):
                    if lresp[0] <= lam <= lresp[len(lresp) - 1]:
                        i_line_spec[k] = i_line_spec[k] * interpolate.interp1d(lresp, iresp, kind='linear')(lam)
                    else:
                        i_line_spec[k] = 0.0
            ele.ele_spec = i_line_spec/np.max(i_line_spec)
        elif ele.name == 'cont':
            ele.ele_spec = planck(t_pl, lclip, lresp, iresp)
        else:
            lines = read_nist_lines('NIST/' + ele.name + '.txt', lmin, lmax, t_el)
            l_line = []
            i_line = []
            for j in range(len(lines)):
                lam = lines[j].wave
                if len(lresp):
                    if lresp[0] <= lam <= lresp[len(lresp) - 1]:
                        intens = lines[j].intensity * interpolate.interp1d(lresp, iresp, kind='linear')(lam)
                        l_line.append(lam)
                        i_line.append(intens)
                        agr = lines[j].aki_gk * interpolate.interp1d(lresp, iresp, kind='linear')(lam)
                    else:
                        l_line.append(lam)
                        intens = lines[j].intensity
                        i_line.append(intens)
                        agr = lines[j].aki_gk
                    lines[j].agr = agr / lam  # energy flux, relative intensities
            maxi = max(i_line)
            len_ini = len(l_line)
            for k in range(len(l_line) - 1, 1, -1):
                if i_line[k] < threshold * maxi:
                    i_line.pop(k)
                    l_line.pop(k)
                    lines[k].agr = 0.0
            if debug:
                print(ele.name, len(l_line), 'lines of', len_ini)
            ele.lines = lines
            try:
                element_spectrum(ele, t_el, sigma0, n_gauss, lclip)
            except Exception as e:
                print(e, '\n', k, j, lam, e)
        if debug:
            np.savetxt('NIST/' + m_fun.change_extension(ele.name, '.lst'),
                       np.transpose(np.array([lclip, i_line_spec])), fmt='%8.3e')
        return True
    else:
        return False


def element_spectrum(ele, t_el, sigma0, n_gauss, lclip):
    # old version
    index_start = 0
    i_line_spec = np.zeros(len(lclip))
    for k in range(len(ele.lines) - 1):
        lam = ele.lines[k].wave
        while lam - lclip[int(index_start)] > 3.0 * sigma0 and index_start < len(lclip) - 2:
            index_start += 1
        if index_start < len(lclip) - 2 and ele.lines[k].agr > 0.0:
            i_line = line_intensity(ele.lines[k].agr, ele.lines[k].ek, 1, t_el)
            for j in range(2 * n_gauss - 1):
                if index_start + j < len(lclip) - 1:
                    lam_index = lclip[int(index_start + j)]
                    i_line_spec[int(index_start) + j] += i_line * np.exp(-((lam_index - lam) / sigma0) ** 2
                                                                         / 2) / sigma0
                    ele.scale = max(i_line_spec)
    ele.ele_spec = np.array(i_line_spec / ele.scale)


def get_fit_parameters(window):
    # load actual parameters for fit
    lmin = float(window['-LMIN_A-'].Get())
    lmax = float(window['-LMAX_A-'].Get())
    t_el = float(window['-T_ELECTRON-'].Get())
    t_cont = float(window['-T_CONT-'].Get())
    threshold = float(window['-THRESHOLD-'].Get())
    sigma_nist = float(window['-SIGMA_NIST-'].Get())
    sigma0 = float(window['-SIGMA0-'].Get())
    return lmin, lmax, t_el, t_cont, threshold, sigma_nist, sigma0


def set_fit_parameters(window, lsqf_var, bc_disabled=(None, 'darkblue')):
    [_spec, _resp, lmin_a, lmax_a, sigma_nist, sigma0, t_cont, t_el, threshold, _t_n2i] = lsqf_var
    window['-LMIN_A-'].update(lmin_a)
    window['-LMAX_A-'].update(lmax_a)
    window['-SAVE_FIT-'].update(disabled=True, button_color=bc_disabled)
    window['-LSQF_SPEC-'].update(disabled=True, button_color=bc_disabled)
    window['-T_ELECTRON-'].update(t_el)
    window['-T_CONT-'].update(t_cont)
    window['-THRESHOLD-'].update(threshold)
    window['-SIGMA_NIST-'].update(sigma_nist)
    window['-SIGMA0-'].update(sigma0)
    window.refresh()


def par_ele_create(sigma_fit, t_cont, t_el, window):
    par_ele = []
    par_ele.append(Element('sigma_fit', sigma_fit, 1, 0, window['-SIGMA_FIT-'].get()))
    par_ele.append(Element('t_cont', t_cont, 1, 0, window['-T_CONT_FIT-'].get()))
    par_ele.append(Element('t_el', t_el, 1, 0, window['-T_EL_FIT-'].get()))
    return par_ele


def write_config_fit(conf, sel_ele, lsqf_dict):
    """
    writes configuration to conf
    :param conf: filename with ext .inf
    :param sel_ele: class Element, list of atomic and molecular species plus continuum
    :param lsqf_dict: dictionary of parameters for spectrum calculation
    :return: None
    """
    def configsetbool(section, option, boolean):
        if boolean:
            config.set(section, option, '1')
        else:
            config.set(section, option, '0')

    config = configparser.ConfigParser()
    cfgfile = open(conf, 'w')
    config.add_section('Parameter')
    config.add_section('Elements')
    for key in lsqf_dict.keys():
        if key in ('spectrum', 'response', 't_n2i'):
            config.set('Parameter', key, lsqf_dict[key])
        else:
            config.set('Parameter', key, str(np.float32(lsqf_dict[key])))
    for ele in sel_ele:
        config.set('Elements', ele.name, '1')
        config.set('Elements', ele.name + '_mult', str(np.float32(ele.mult)))
        config.set('Elements', ele.name + '_scale', str(np.float32(ele.scale)))
        config.set('Elements', ele.name + '_index', str(ele.index))
        configsetbool('Elements', ele.name + '_fit', ele.fit)
    config.write(cfgfile)
    logging.info(f' configuration saved as {conf}')
    cfgfile.close()


# -------------------------------------------------------------------
def read_configuration(conf, lsqf_dict, all_ele):
    """
    read configuration file for m_spec analysis from conf
    :param conf: filename of configuration with extension .inf
    :param all_ele: class Element, list of atomic and molecular species plus continuum
    :param lsqf_dict: dictionary of parameters for spectrum calculation
    :return: list of lsqf_dict values, selected elements class Element with chosen fit parameters
    """
    partext = ''
    sel_ele = []
    if path.exists(conf):
        config = configparser.ConfigParser()
        config.read(conf)
        for section in config.sections():
            partext += f'[{section}]\n'
            for key in config[section]:
                partext += f'- [{key}] = {config[section][key]}\n'
        for key in config['Parameter'].keys():
            if key in ('spectrum', 'response', 't_n2i'):
                lsqf_dict[key] = config['Parameter'][key]
            elif key in lsqf_dict.keys():
                lsqf_dict[key] = float(config['Parameter'][key])
            else:
                print('unknown key in readconf: ', key)
        elements = []
        for ele in all_ele:  # update element list
            el = ele.name
            elements.append(el)
        sel_elements = []
        for key in config['Elements']:
            if key in elements:
                sel_elements.append(key)
        for ele in sel_elements:
            for key in config['Elements']:
                if key == str(ele + '_mult'):
                    mult = float(config['Elements'][key])
            for key in config['Elements']:
                if key == str(ele + '_max') or key == str(ele + '_scale'):
                    scale = float(config['Elements'][key])
            for key in config['Elements']:
                if key == str(ele + '_index'):
                    index = int(config['Elements'][key])
            for key in config['Elements']:
                if key == str(ele + '_fit'):
                    fit = config['Elements'][key]
            color = get_ele(ele, all_ele).color
            sel_ele.append(Element(ele, mult, scale, index, fit, color=color))
        logging.info(f'configuration {conf} loaded')
    return list(lsqf_dict.values()), sel_ele


def load_spectrum_analysis(spec_file_analysis, lmin, lmax, graph_an, canvasx, ref_style):
    lspec, ispec = np.loadtxt(spec_file_analysis, unpack=True, ndmin=2)
    # select desired wavelength range and normalize spectrum
    lclip = []
    iclip = []
    delta = (lspec[1] - lspec[0])
    if delta < 0:
        lspec = np.flip(lspec)
        ispec = np.flip(ispec)
        delta = (lspec[1] - lspec[0])
    for l0 in range(0, len(lspec)):
        if (lmin <= lspec[l0] <= lmax) or (lmin >= lspec[l0] >= lmax):
            lclip.append(lspec[l0])
            iclip.append(ispec[l0])
    peak_int = max(iclip)
    iclip = np.array(iclip) / peak_int
    graph_an_ll, graph_an_ur = (lmin, -.1), (lmax, 1.2)
    graph_an.change_coordinates(graph_an_ll, graph_an_ur)
    idg_spec = m_plot.plot_reference_spectrum(spec_file_analysis, lclip, iclip, graph_an, canvasx,
                                              plot_style=ref_style)
    return delta, lclip, iclip, idg_spec


def load_response_analysis(resp_file_analysis, graph_an, canvasx, response_style):
    if resp_file_analysis:
        lresp, iresp = np.loadtxt(resp_file_analysis, unpack=True, ndmin=2)
        iresp = iresp / max(iresp)
        idg_resp_a = m_plot.plot_reference_spectrum(resp_file_analysis, lresp, iresp, graph_an, canvasx,
                                                    plot_style=response_style)
    else:
        lresp = []
        iresp = []
    return lresp, iresp, idg_resp_a


def lsqf_fit_disable(window, disabled=True):
    """
    switches button colors and sets enabled flag
    :param window: main window where buttons are located
    :param disabled: if True, sets button color to disabled
    :return: enabled, True, if disabled = False and else False
    """
    bc_enabled = ('white', 'green')
    bc_disabled = (None, 'darkblue')
    bc = bc_disabled if disabled else bc_enabled
    window['-SAVE_FIT-'].update(disabled=disabled, button_color=bc)
    window['-LSQF_SPEC-'].update(disabled=disabled, button_color=bc)
    return not disabled


# ------------------------------------------------------------------------------
#       LSQF
# ------------------------------------------------------------------------------
global t_cont_old, t_el_old


def errorsum(params, sel_ele, i_clip, lclip, lresp, iresp, sigma0, n_gauss):
    global t_cont_old, t_el_old
    par = []
    delta_spec = (lclip[-1] - lclip[0]) / (len(lclip) - 1)
    for ele in sel_ele:
        par.append(params[ele.name].value)
    # if params['t_cont'].vary:
    t_cont = params['t_cont'].value
    if t_cont != t_cont_old:
        get_ele('cont', sel_ele).ele_spec = planck(t_cont, lclip, lresp, iresp)
        t_cont_old = t_cont
    if params['t_el'].vary:
        t_el = params['t_el'].value
        if t_el != t_el_old:
            for ele in sel_ele:
                if ele.name not in ('cont', 'n2i') and ele.fit:
                    element_spectrum(ele, t_el, sigma0, n_gauss, lclip)
            t_el_old = t_el
    # calculate synthetic spectrum
    i_fit = np.zeros(len(i_clip))
    for ele in sel_ele:
        i_fit = i_fit + ele.ele_spec * par[ele.index]
    sigma_fit = params['sigma_fit'].value
    i_fit = gaussian(i_fit, sigma_fit / delta_spec)
    # calculate errors
    residual = i_clip - i_fit
    return np.ravel(residual)


def lsq_fit(iclip, sel_ele, par_ele, lclip, lresp, iresp, sigma0, n_gauss,
            no_fit=False, debug=False):
    """
    can also be used to calculate spectrum without fit
    :param iclip intensities of measured spectrum
    :param sel_ele: list of class Element with fixed or variable intensity
    :param par_ele: list of class Element with fixed or variable intensity
    :param lclip: wavelength list
    :param lresp: response wavelength list
    :param iresp: response intensity list
    :param sigma0: gaussian broadening for initial spectrum
    :param n_gauss: number of data points for gaussian filtering
    :param no_fit: True if no fit required, only spectrum
    :param debug: detailed info if set
    :return: result of fit, parameters, spectrum and residuals

    """
    global t_cont_old, t_el_old
    t0 = time.time()
    t_cont_old = 0
    t_el = get_ele('t_el', par_ele).mult
    t_el_old = t_el
    sigma_fit = get_ele('sigma_fit', par_ele).mult
    delta_spec = (lclip[-1] - lclip[0]) / (len(lclip) - 1)
    args = (sel_ele, iclip, lclip, lresp, iresp, sigma0, n_gauss)
    params = Parameters()
    n_fit = 0
    for ele in sel_ele:
        fit = True if ele.fit else False
        params.add(ele.name, value=ele.mult, vary=fit)
        if ele.fit:
            n_fit += 1
    for ele in par_ele:
        fit = True if ele.fit else False
        if ele.name == 'sigma_fit':
            params.add(ele.name, value=ele.mult, vary=fit, min=0.0)
        else:
            params.add(ele.name, value=ele.mult, vary=fit, min=1000, max=10000)
    # ------------------------------------------------------------------------------
    # LEAST SQUARE FIT
    # ------------------------------------------------------------------------------
    if no_fit or n_fit == 0:
        i_residual = errorsum(params, sel_ele, iclip, lclip, lresp, iresp, sigma0, n_gauss)
        result_fit = ''
    else:
        max_nfev = 10 * n_fit
        out = minimize(errorsum, params, args=args, max_nfev= max_nfev)
        if debug:
            report_fit(out)
        result_fit = '#Parameter Value Stderr  %\n'
        for name, param in out.params.items():
            try:
                percent = 100.0 * param.stderr / param.value
                result_fit += f'#{name:4s} {param.value:8.4f} {param.stderr:7.4f} ({percent:4.1f}%)\n'
            except:
                if param.stderr:
                    result_fit += f'#{name:4s} {param.value:8.4f} {param.stderr:7.4f}\n'
                else:
                    result_fit += f'#{name:4s} {param.value:8.4f} \n'

        # ------------------------------------------------------------------------------
        # after fit set new values
        i_residual = errorsum(out.params, sel_ele, iclip, lclip, lresp, iresp, sigma0, n_gauss)
        for ele in sel_ele:
            ele.mult = out.params[ele.name].value
        for ele in par_ele:
            ele.mult = out.params[ele.name].value
        sigma_fit = get_ele('sigma_fit', par_ele).mult
    i_fit = np.zeros(len(iclip))
    for ele in sel_ele:
        i_fit = i_fit + ele.ele_spec * ele.mult
        ele.gspec = gaussian(ele.ele_spec * ele.mult, sigma=sigma_fit / delta_spec)
    i_fit = gaussian(i_fit, sigma=sigma_fit / delta_spec)
    time_fit = time.time() - t0
    # if debug:
    print(f'time lsqf: {time_fit:5.3f} rms:{np.std(i_residual):8.6f}')
    return i_fit, i_residual, sel_ele, par_ele, time_fit, result_fit


def calculate_spectrum(sel_ele, par_ele, iclip, result_fit, lclip, lresp, iresp, sigma0, n_gauss):
    i_fit, i_residue, _s_e, _p_e, _t, _rf = lsq_fit(iclip, sel_ele, par_ele, lclip,
                                                    lresp, iresp, sigma0, n_gauss, no_fit=True)
    rms_error = np.std(i_residue)
    result_fit += f'#  rms: {rms_error:8.6f}\n'
    return i_fit, i_residue, rms_error, result_fit


def plot_analysis(window, i_residue, lclip, iclip, i_fit, spec_file_analysis, lmin_a, lmax_a, residual_offset,
                  graph_an, canvasx, ref_style, sel_ele, zoom_window=None):
    try:
        graph_an.erase()
        show_res = window['-RESIDUALS-'].Get()
        show_sum = window['-SUM-'].Get()
        off = -2 * residual_offset if show_res else -0.1
        graph_an_ll = (lmin_a, off)
        graph_an_ur = (lmax_a, 1.2)
        graph_an.change_coordinates(graph_an_ll, graph_an_ur)
        if zoom_window != []:
            graph_an.change_coordinates((zoom_window[0, 0], zoom_window[0, 1]), (zoom_window[1, 0], zoom_window[1, 1]))
        if show_res:
            graph_an.DrawLine((lmin_a, -residual_offset), (lmax_a, -residual_offset), 'grey', 1)
            if len(i_residue):
                m_plot.plot_reference_spectrum('', lclip, i_residue - residual_offset,
                                               graph_an, canvasx, [], plot_style=('green', 0, 2, -0.15))
        m_plot.plot_reference_spectrum(spec_file_analysis, lclip, iclip, graph_an, canvasx, [], plot_style=ref_style)
        for ele in sel_ele:
            if ele.ele_spec != []:
                m_plot.plot_reference_spectrum(ele.name, lclip, ele.gspec, graph_an, canvasx,
                                               plot_style=(ele.color, 0, 1, -0.23 - 0.03 * ele.index))

        if len(i_fit) == len(lclip) and show_sum:
            m_plot.plot_reference_spectrum('', lclip, i_fit, graph_an, canvasx, [], plot_style=('black', 0, 2, -0.1))
    except Exception as e:
        print(f'error in plot_analysis: {e}')
    return

def line_strength(all_ele, lclip, lresp, iresp):
    for ele in all_ele:
        ele.strength = 0.0
        warning = False
        if ele.name in ('fei', 'mgi', 'nai'):
            for k in range(len(lclip)):
                if ele.range_low <= lclip[k] <= ele.range_high:
                    ele.strength += ele.ele_spec[k] * ele.mult
            if lresp != [] and lresp[0] < 512 and lresp[-1] > 594:
                resp = interpolate.interp1d(lresp, iresp, kind='linear')((ele.range_low + ele.range_high) / 2)
                ele.strength /= resp
            else:
                warning = True

    return warning

