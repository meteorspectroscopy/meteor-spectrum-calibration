
# -------------------------------------------------------------------
# m_calfun8 functions for m_pipe0, revised for m_specall
# -------------------------------------------------------------------
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from lmfit import minimize, Parameters, report_fit
from skimage import img_as_float  # , img_as_ubyte,
from skimage import io as ios
from skimage.transform import rescale

import PySimpleGUI as sg
import m_specfun as m_fun

version = '0.8.8'
fitsv = ['' for x in range(7)]
fitskey = ['DATE-OBS', 'OBSERVER', 'VERSION', 'INSTRUME', 'TELESCOP', 'M_STATIO', 'COMMENT']
fits_dict = dict(list(zip(fitskey, fitsv)))
debug = False


# -------------------------------------------------------------------
# from l_calib
# -------------------------------------------------------------------
def dist(x0s, y0s, scalxy, lam, x00, y00, rot, disp0, a3=0, a5=0):
    """
    calculates coordinates x, y in original image from transformed image (x', y')
    for given
    x0s: array[maxs] x'-coordinate of zero (leftmost) order for each spectrum
    y0s: array[maxs] x'-coordinate of zero (leftmost) order for each spectrum
    lam0: float, wavelength of calibration laser *order
    rot: float, image rotation in rad
    disp0: float, dispersion in [pixel/nm] in the center of the image
    returns (x,y)
    a3, a5: coefficients of the radial transformation matrix
    r = r'*(1 + a3*r'**2 + a5*r'**4)

    the function also allows to fit a sqrt for the radial function, approximated by
    r = r'*(1 + 0.5*(r'/feff)**2 + 0.375*(r'/feff)**4) or
    r = r'*(1 + a3**2 + 1.5*a3**4

    the desired function is selected by the value of a5:
    a5 < 1: polynom
    a5 == 1: sqrt-fit
    this requires minimal changes of the code
    """
    # select polynom or sqrt-fit
    _a5 = 1.5 * a3 * a3 if a5 == 1 else a5
    xy00 = [x00, y00]
    xyl = [x0s + lam / disp0, y0s] - np.array(xy00)  # coordinates of laser lines wrt, optical axis
    r = np.sqrt(xyl[0]**2 + xyl[1]**2)  # polar coordinates
    phi = np.arctan2(xyl[1], xyl[0])    # polar coordinates
    phi += rot                        # apply rotation
    r = r * (1 + a3 * r**2 + _a5 * r**4)    # transform radial coordinate
    xyr = np.multiply(r, [np.cos(phi), np.sin(phi) / scalxy])  # return to cartesian coordinates
    return xyr + xy00


# -------------------------------------------------------------------------------
def errorsum(params, maxs, maxl, x, y, lam):
    (x0, y0, scalxy, x00, y00, rot, disp0, a3, a5) = get_param(params, maxs)
    # calculate positions
    xf = np.zeros([maxs, maxl])
    yf = np.zeros([maxs, maxl])
    lf = np.zeros([maxs, maxl])
    for s in range(0, maxs):
        for l0 in range(0, maxl):
            if x[s, l0] > 0:
                (xf[s, l0], yf[s, l0]) = dist(x0[s], y0[s], scalxy, lam[s, l0],
                                              x00, y00, rot, disp0, a3, a5)
    # calculate errors
    xf = xf - x
    yf = yf - y
    return np.ravel(xf), np.ravel(yf), np.ravel(lf)
    # TODO: is np.ravel(lf) needed


# -------------------------------------------------------------------------------
def get_param(params, maxs):  # from least square fit
    x0 = np.zeros(maxs)
    y0 = np.zeros(maxs)
    for i in range(maxs):
        strx = 'x0_' + str(i)
        x0[i] = params[strx].value
        stry = 'y0_' + str(i)
        y0[i] = params[stry].value

    # other variable parameters
    scalxy = params['scalxy'].value
    x00 = params['x00'].value
    y00 = params['y00'].value
    rot = params['rot'].value
    disp0 = params['disp0'].value
    a3 = params['a3'].value
    a5 = params['a5'].value
    return x0, y0, scalxy, x00, y00, rot, disp0, a3, a5


# -------------------------------------------------------------------------------
def set_params(param, maxs, params):  # for least square fit
    (x0, y0, scalxy, x00, y00, rot, disp0, a3, a5) = param
    for i in range(maxs):
        strx = 'x0_' + str(i)
        stry = 'y0_' + str(i)
        params.add(strx, value=x0[i])
        params.add(stry, value=y0[i])
    params.add('scalxy', value=scalxy)
    params.add('x00', value=x00)
    params.add('y00', value=y00)
    params.add('rot', value=rot)
    params.add('disp0', value=disp0)
    params.add('a3', value=a3)
    params.add('a5', value=a5)
    return params


# -------------------------------------------------------------------------------
def select_lines(xl, yl, wxl, wyl, laml, maxs, maxl):
    sizear = [maxs, maxl]
    x0 = np.zeros(maxs)
    y0 = np.zeros(maxs)
    l0 = 0
    s = 0
    maxl = 0
    x = np.zeros(sizear)
    y = np.zeros(sizear)
    w = np.zeros(sizear)
    lam = np.zeros(sizear)
    sl = np.zeros(sizear, dtype=np.int8)  # used for plotting results
    k = 0  # index input series
    # convert data from txt file into arrays
    while k < np.shape(xl)[0]:
        while xl[k] > 0 and l0 < sizear[1]:
            x[s, l0] = xl[k]
            y[s, l0] = yl[k]
            w[s, l0] = (wxl[k] + wyl[k]) / 2
            lam[s, l0] = laml[k]
            sl[s, l0] = s + 1
            k += 1
            l0 += 1
            maxl = max(maxl, l0)
        if l0 > 0:
            s += 1
            l0 = 0
            k += 1
        else:
            k += 1
    maxs = s
    # reduce matrix size:
    x = x[:maxs, :maxl]
    y = y[:maxs, :maxl]
    w = w[:maxs, :maxl]
    lam = lam[:maxs, :maxl]
    sl = sl[:maxs, :maxl]
    x0 = x0[:maxs]
    y0 = y0[:maxs]
    return x, y, w, lam, sl, x0, y0, maxs, maxl


# -------------------------------------------------------------------------------
def plot_laser(x, y, w, sl, xf, yf, title):
    dotsize = 5  # size of dots in scatterplot
    plt.rc('text', usetex=False)  # to not use Latex in python graph!!!
    plt.rc('font', size=10)  # set the dimension of the figure
    fig, ax = plt.subplots(1)
    fig.set_size_inches(7, 5)
    ax.scatter(x, y, s=dotsize * w, c=sl, marker='o')  # size corresponds to FWHM
    ax.scatter(xf, yf, s=dotsize * w, c='white', marker='x')
    ax.scatter(xf, yf, s=dotsize * w, c='black', marker='+')
    ax.set_title(title)  # set the title
    ax.set_ylabel('y')  # set the label on the y axis
    ax.set_xlabel('x')  # set the label on the x axis
    ax.grid()  # grid
    return


# -------------------------------------------------------------------------------
def mreadbmp(filename, colflag='bb'):
    image = np.flipud(img_as_float(ios.imread(filename)))
    if (colflag[0] == 'b') and (len(image.shape)) == 3:
        image = np.sum(image, axis=2) / 3
    return image


# -------------------------------------------------------------------------------
def load_image(infil, opt_dict, imagesave=True):
    # debug = opt_dict['debug']
    p = Path(infil).with_suffix('.fit')
    if p.exists():
        im = np.array(fits.getdata(p))
        if len(im.shape) == 3:
            im = np.transpose(im, (1, 2, 0))
            imbw = np.sum(im, axis=2)  # used for fitgaussian(data)
        else:
            imbw = im
    else:
        print(f'{p} not found, try {infil}')
        try:
            imbw = mreadbmp(infil)  # with extension
            imbw = imbw / np.max(imbw)
            # we do not want tmp.fit, it overwrites tmp.png (default image type)!
            if str(Path(infil).with_suffix('')) != 'tmp':
                hdu = fits.PrimaryHDU(imbw.astype(np.float32))
                hdu.header['BSCALE'] = 32767
                hdu.header['BZERO'] = 0
                hdu.writeto(p, overwrite=True)
        except Exception:
            imbw = []
            return imbw, opt_dict
    imbw = imbw / np.max(imbw)
    # save image as png for display with sg and adjust image scale or window size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (imy, imx) = imbw.shape
        if opt_dict['scale_win2ima']:   # fit window to image size
            zoom = opt_dict['zoom']
            im = rescale(imbw, zoom)
            opt_dict['win_width'] = max(int(imx * zoom), 600) + 400
            opt_dict['win_height'] = max(int(imy * zoom), 540) + 90
        else:                           # fit image size to window
            max_width = max(opt_dict['win_width'] - 400, 50)  # avoid non-negative values
            max_height = max(opt_dict['win_height'] - 90, 50)
            imscale = min(max_width / imx, max_height / imy)
            im = rescale(imbw, imscale)
        if imagesave:
            ios.imsave('tmp.png', np.flipud(im * 255).astype(np.uint8))
    return imbw, opt_dict


def lsqf(parv, debug=False, fit_report=False):
    """
    input:
        text file with laser line positions (x,y)
        spectra separated with zeroes
        end of file: double zero row

    result: array of x, y for each spectrum in a row

    ready for fit
    """
    # -------------------------------------------------------------------------------

    [lam0, scalxy, fitxy, imx, imy, f0, pix, grat, rotdeg, binning, comment, infile, outfil, linelist, typesqrt] = parv
    xl, yl, wxl, wyl, ll = np.loadtxt(outfil + '.txt', unpack=True)
    # debug = False
    # spectra with zeros for separator
    # print(np.transpose(np.array([xl,yl,wxl,wyl])))
    # select max size of arrays, to be reduced later, initialize parameters
    # -------------------------------------------------------------------------------
    maxs = 20
    maxl = 20
    # select lines and arrange them in array, each spectrum one row
    (x, y, w, lam, sl, x0, y0, maxs, maxl) = select_lines(xl, yl, wxl, wyl, ll, maxs, maxl)
    if debug:
        print('l s = ', maxl, maxs, '\nx\n', x, '\ny\n', y)
    # start values for fit
    disp0 = 1e6 / f0 / grat * pix * binning  # dispersion at image center
    x00 = imx / 2   # image center
    y00 = imy / 2
    # start values for distortion
    a3 = 0.5 * (pix * binning / f0)**2  # calculate from tang to ortho distortion
    a5 = 0.37 * (pix * binning / f0)**4
    rot = rotdeg * np.pi / 180
    s = 0
    while s < maxs:
        x0[s] = x[s, 0] - lam[s, 0] / disp0
        y0[s] = y[s, 0]
        s += 1
    #  -------------------------------------------------------------------------------
    # plot data before fit with default values for parameters
    xf = np.zeros([maxs, maxl])
    yf = np.zeros([maxs, maxl])
    for s in range(0, maxs):
        for l0 in range(0, maxl):
            if x[s, l0] > 0:
                (xf[s, l0], yf[s, l0]) = dist(x0[s], y0[s], scalxy, lam[s, l0],
                                              x00, y00, rot, disp0, a3, a5)
    # if debug:
    #     # show data and initial fit
    #     print('l s = ', maxl, maxs,'\nxf\n',xf,'\n','yf\n',yf)
    #     plot_laser(x,y,w,sl,xf,yf,maxs,maxl, f'{filename}.txt  o: data points, + before fit  ')
    #     plt.show() #show on screen
    #     # print('delta x\n',(x[:,:]-xf[:,:]),'\ndelta y\n', (y[:,:]-yf[:,:]))
    if typesqrt:
        a5 = 1.0
    param = (x0, y0, scalxy, x00, y00, rot, disp0, a3, a5)
    args = (maxs, maxl, x, y, lam)
    params = Parameters()
    set_params(param, maxs, params)  # select variable parameters
    params['scalxy'].vary = fitxy
    params['x00'].vary = True
    params['y00'].vary = True
    params['rot'].vary = True
    params['disp0'].vary = True
    params['a3'].vary = True
    params['a5'].vary = False if typesqrt else True
    time0 = time.time()
    # ------------------------------------------------------------------------------
    # LEAST SQUARE FIT
    # ------------------------------------------------------------------------------
    out = minimize(errorsum, params, args=args)
    if fit_report:
        report_fit(out)
    # after fit set new values
    param = get_param(out.params, maxs)
    (x0, y0, scalxy, x00, y00, rot, disp0, a3, a5) = param
    fl = 1.0e6 / grat * pix * binning / disp0
    if typesqrt:  # put back correct a5
        a5 = 1.5 * a3 * a3
        feff = pix * binning / (np.sqrt(2 * a3))
        print(f'for sqrt fit: feff = {feff:10.2f}')
    for s in range(0, maxs):
        for l0 in range(0, maxl):
            if x[s, l0] > 0:
                (xf[s, l0], yf[s, l0]) = dist(x0[s], y0[s], scalxy, lam[s, l0],
                                              x00, y00, rot, disp0, a3, a5)
    tim = time.time() - time0
    timestr = "{:10.4f}".format(tim)
    print('fit time = ', timestr, ' sec')
    np.set_printoptions(precision=2, suppress=True)
    if debug:
        print('delta x\n', (x[:, :] - xf[:, :]), '\ndelta y\n', (y[:, :] - yf[:, :]))
    np.set_printoptions()
    errorx = np.sqrt(np.average(np.square(x - xf)[:, :]))  # missing lines also counted
    errory = np.sqrt(np.average(np.square(y - yf)[:, :]))
    if debug:
        print(f'\n{comment}\n{outfil}.txt')
        print(f'rms_x ={errorx:8.4f}')
        print(f'rms_y ={errory:8.4f}')
    plot_laser(x, y, w, sl, xf, yf, f'{outfil}.txt  o: data points, + afterfit  ')
    plt.savefig(outfil + '_lsfit.png')  # and if we want to save the figure
    rotdeg = rot * 180 / np.pi
    par = np.float32([scalxy, x00, y00, rotdeg, disp0, a3, a5, errorx, errory])
    parstr = ['scalxy', 'x00', 'y00', 'rotdeg', 'disp0']
    # ------------------------------------------------------------------------------
    if debug:
        print('parameters after fit:')
    result = f'{comment}\n{outfil}.txt\n'
    result += 'Parameters after fit:\n'
    result += f"a3     ={a3:12.4e}\n"
    result += f"a5     ={a5:12.4e}\n"
    for i in range(5):
        if debug:
            print(f"{parstr[i]:6s} ={par[i]:12.4f}")
        result += f"{parstr[i]:6s} ={par[i]:12.4f}\n"
    result += f'rms_x ={errorx:8.4f}\n'
    result += f'rms_y ={errory:8.4f}\n'
    result += f'focal length from disp0 = {fl:8.4}\n'
    if typesqrt:
        result += f'for sqrt fit: feff = {feff:10.2f}\n'
    else:
        result += 'polynomial fit\n'
    if debug:
        print(f"a3     ={a3:12.4e}")
        print(f"a5     ={a5:12.4e}")
    return par, result
# --------------------------------------------------------------------------


def get_linelist(linelist, lam0):
    notok = True
    lines = np.zeros(20)
    while notok:
        if linelist == 'l':
            for i in range(20):
                lines[i] = i * float(lam0)
            notok = False
        else:
            if linelist:
                p = Path(linelist).with_suffix('.txt')
                if p.exists():
                    lines = np.loadtxt(p, usecols=0, unpack=True)
            notok = False
    notok = not linelist
    return notok, linelist, lines
# -------------------------------------------------------------------------------


def calib_setup(ini_file, par_dict, res_dict, fits_dict, opt_dict, logtext):
    winsetup_active = True
    parv = list(par_dict.values())
    parkey = list(par_dict.keys())
    wloc_setup = (opt_dict['win_x'] + opt_dict['setup_off_x'],
                  opt_dict['win_y'] + opt_dict['setup_off_y'])
    # update values of setup window
    input_row = []
    input_elem = []
    info = ''
    for k in range(11):
        input_elem.append(sg.Input(parv[k], size=(30, 1)))
        input_row.append([sg.Text(parkey[k], size=(10, 1)), input_elem[k]])
    filename_ini_in_elem = sg.InputText(ini_file, size=(34, 1))
    # layout of setup window
    layout_setup = [[sg.Frame('Settings',
                              [[sg.Frame('Lasercal',
                                         [[sg.Text('Lasercal')],
                                          # [[sg.Text(ki[k], size=(5,1)), sg.Input(kval[k])] for k in range(15)],
                                          input_row[0], input_row[1], input_row[2], input_row[3],
                                          input_row[4], input_row[5], input_row[6], input_row[7],
                                          input_row[8], input_row[9], input_row[10],
                                          # [input_row[k] for k in range(15)], does not work
                                          [filename_ini_in_elem],
                                          [sg.Button('SaveC', size=(6, 1)),
                                           sg.Button('Apply', size=(6, 1)),
                                           sg.Button('Cancel', size=(6, 1))]])]])]]

    winsetup = sg.Window('Parameters', layout_setup, disable_close=True,
                         disable_minimize=True, location=wloc_setup, keep_on_top=True,
                         no_titlebar=False, resizable=True)

    while winsetup_active:
        evsetup, valset = winsetup.Read(timeout=100)
        if evsetup == 'Cancel':
            winsetup_active = False
            winsetup.Close()

        if evsetup in ('Apply', 'SaveC'):
            for k in range(11):
                key = parkey[k]
                if key[0] == 'b':
                    if valset[k] == '0':
                        par_dict[key] = False
                    else:
                        par_dict[key] = True
                elif key[0] == 'i':
                    par_dict[key] = int(valset[k])
                elif key[0] == 'f':
                    par_dict[key] = float(valset[k])
                else:
                    par_dict[key] = valset[k]
                input_elem[k].Update(valset[k])
            (x, y) = winsetup.current_location()
            opt_dict['setup_off_x'] = x - opt_dict['win_x']
            opt_dict['setup_off_y'] = y - opt_dict['win_y']
            notok = True
            while notok:
                notok, linelist, lines = get_linelist(par_dict['s_linelist'], par_dict['f_lam0'])
                if notok:
                    winsetup.Hide()
                    linelist, info = m_fun.my_get_file('', title='Get Linelist',
                                                       file_types=(('Linelist', '*.txt'), ('ALL Files', '*.*')),
                                                       default_extension='*.txt')
                    winsetup.UnHide()
                    if not linelist:
                        linelist = 'l'
                        notok = False
            par_dict['s_linelist'] = str(Path(linelist).with_suffix(''))

            if evsetup in 'SaveC':
                winsetup.Hide()
                ini_file, info = m_fun.my_get_file(filename_ini_in_elem.Get(),
                                                   title='Save Configuration File', save_as=True,
                                                   file_types=(('Configuration Files', '*.ini'), ('ALL Files', '*.*')),
                                                   default_extension='*.ini', error_message='no configuration saved: ')
                if ini_file:
                    m_fun.write_configuration(ini_file, par_dict, res_dict, fits_dict, opt_dict)
                else:
                    sg.Popup('No file saved', keep_on_top=True)
            logtext += info + '\n'
            winsetup_active = False
            winsetup.Close()
    return ini_file, par_dict, res_dict, fits_dict, opt_dict, logtext
    # -------------------------------------------------------------------------
