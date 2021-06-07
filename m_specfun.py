# -------------------------------------------------------------------
# m_specfun functions for m_spec
# Author: Martin Dubs, 2020
# -------------------------------------------------------------------
import configparser
import ctypes
import io
import logging
import os
import os.path as path
import platform
import subprocess
import time
import warnings
from datetime import datetime, date

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.time import Time
from scipy import optimize, interpolate
from scipy.ndimage import map_coordinates
from skimage import img_as_float
from skimage import io as ios
from skimage import transform as tf

import PySimpleGUI as sg

if platform.system() == 'Windows':
    ctypes.windll.user32.SetProcessDPIAware()  # Set unit of GUI to pixels

version = '0.9.25'
# today = date.today()
logfile = 'm_spec' + date.today().strftime("%y%m%d") + '.log'
# turn off other loggers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logfile, format='%(asctime)s %(levelno)s %(lineno)d %(message)s', level=logging.INFO)
# -------------------------------------------------------------------
# initialize dictionaries for configuration
parv = ['1' for x in range(10)] + ['' for x in range(10, 15)]
parv[14] = 'l'
parkey = ['f_lam0', 'f_scalxy', 'b_fitxy', 'i_imx', 'i_imy', 'f_f0', 'f_pix', 'f_grat', 'f_rotdeg', 'i_binning',
          's_comment', 's_infile', 's_outfil', 's_linelist', 'b_sqrt']
par_dict = dict(list(zip(parkey, parv)))

resv = [0.0 for x in range(7)]
reskey = ['scalxy', 'x00', 'y00', 'rot', 'disp0', 'a3', 'a5']
res_dict = dict(list(zip(reskey, resv)))
fitsv = ['' for x in range(7)]
fitskey = ['DATE-OBS', 'OBSERVER', 'VERSION', 'INSTRUME', 'TELESCOP', 'M_STATIO', 'COMMENT']
fits_dict = dict(list(zip(fitskey, fitsv)))

bc_enabled = ('white', 'green')  # button_color
bc_disabled = (None, 'darkblue')
# default values for setup, if no m_set.ini
debug = False
fit_report = False
win2ima = False
zoom = 1.0
wsize = (1060, 660)
wloc = (50, 20)
(xoff_calc, yoff_calc) = (355, 50)
(xoff_setup, yoff_setup) = (250, 120)
wloc_calc = (wloc[0] + xoff_calc, wloc[1] + yoff_calc)
wloc_setup = (wloc[0] + xoff_setup, wloc[1] + yoff_setup)
(max_width, max_height) = (700, 500)
opt_comment = ''
pngdir = ''
png_name = 'tmp/m_'
outpath = 'out'
mdist = 'mdist'
colorflag = False
bob_doubler = False
plot_w = 1000
plot_h = 500
i_min = -0.5
i_max = 5
graph_size = 2000
show_images = True
meteor_lines = 'meteor_lines'
video_list_length = 50
optkey = ['zoom', 'win_width', 'win_height', 'win_x', 'win_y', 'calc_off_x',
          'calc_off_y', 'setup_off_x', 'setup_off_y', 'debug', 'fit-report',
          'scale_win2ima', 'comment', 'png_name', 'outpath', 'mdist', 'colorflag', 'bob',
          'plot_w', 'plot_h', 'i_min', 'i_max', 'graph_size', 'show_images', 'meteor_lines']
optvar = [zoom, wsize[0], wsize[1], wloc[0], wloc[1], xoff_calc, yoff_calc,
          xoff_setup, yoff_setup, debug, fit_report, win2ima, opt_comment, png_name,
          outpath, mdist, colorflag, bob_doubler, plot_w, plot_h, i_min, i_max, graph_size, show_images, meteor_lines]
opt_dict = dict(list(zip(optkey, optvar)))

# -------------------------------------------------------------------
def my_rescale(im, imscale, multichannel=False):
    try:
        ima = tf.rescale(im, imscale, multichannel=multichannel)
    except:
        # for older versions of skimage
        ima = tf.rescale(im, imscale)
    return ima

# -------------------------------------------------------------------
def read_configuration(conf, par_dict, res_dict, opt_dict):
    """
    read configuration file for m_calib and m-spec
    :param conf: filename of configuration with extension .ini
    :param par_dict: parameters for m_calib, partly used in m-spec
    :param res_dict: results of m_calib, used for distortion
    :param opt_dict: options, used in m_calib and m_spec
    :return:
    partext: multiline text of configuration
    updated values of par_dict, res_dict, fits_dict, opt_dict
     # from readconf in m_config3.py
    """
    partext = ''
    pngdir = ''
    if path.exists(conf):
        config = configparser.ConfigParser()
        config.read(conf)
        for section in config.sections():
            partext += f'[{section}]\n'
            for key in config[section]:
                partext += f'- [{key}] = {config[section][key]}\n'

        for key in config['Lasercal'].keys():
            k = key[0]
            if k == 'b':
                if config['Lasercal'][key] == '0':
                    par_dict[key] = False
                else:
                    par_dict[key] = True
            elif k == 'f':
                par_dict[key] = float(config['Lasercal'][key])
            elif k == 'i':
                par_dict[key] = int(config['Lasercal'][key])
            elif k == 's':
                par_dict[key] = config['Lasercal'][key]
            else:
                print('unknown key in readconf: ', key)

        if 'Calib' in config.sections():
            for key in config['Calib']:
                res_dict[key] = float(config['Calib'][key])

        if 'Fits' in config.sections():
            for key in config['Fits']:
                fits_dict[key.upper()] = config['Fits'][key]
        if 'Options' in config.sections():
            for key in config['Options'].keys():
                if key in (
                        'win_width', 'win_height', 'win_x', 'win_y', 'calc_off_x', 'calc_off_y', 'setup_off_x',
                        'setup_off_y', 'graph_size'):
                    opt_dict[key] = int(config['Options'][key])
                elif key in ('debug', 'fit-report', 'scale_win2ima', 'scale_ima2win',
                             'colorflag', 'bob', 'show_images'):
                    opt_dict[key] = bool(int(config['Options'][key]))
                elif key in ('zoom', 'i_min', 'i_max'):
                    opt_dict[key] = float(config['Options'][key])
                else:
                    if key == 'pngdir':
                        pngdir = config['Options'][key]
                    else:
                        opt_dict[key] = config['Options'][key]
        opt_dict['png_name'] = path.join(pngdir, opt_dict['png_name'])  # used for compatibility with old inifile
        logging.info(f'configuration {conf} loaded')
    return partext, par_dict, res_dict, fits_dict, opt_dict


# ------------------------------------------------------------------------------
def write_configuration(conf, par_dict, res_dict, fits_dict, opt_dict):
    """
    writes configuration to conf
    :param conf: filename with ext .ini
    :param par_dict: parameters for m_calib, partly used in m-spec
    :param res_dict: results of m_calib, used for distortion
    :param fits_dict: content of fits header
    :param opt_dict: options, used in m_calib and m_spec
    :return: None
    """

    def configsetbool(section, option, boolean):
        if boolean:
            config.set(section, option, '1')
        else:
            config.set(section, option, '0')

    # for compatibility with old versions
    pngdir, opt_dict['png_name'] = path.split(opt_dict['png_name'])
    config = configparser.ConfigParser()
    cfgfile = open(conf, 'w')
    if 'Lasercal' not in config: config.add_section('Lasercal')
    if 'Calib' not in config: config.add_section('Calib')
    if 'Fits' not in config: config.add_section('Fits')
    if 'Options' not in config: config.add_section('Options')
    for key in par_dict.keys():
        k = key[0]
        if k == 'b':
            configsetbool('Lasercal', key, par_dict[key])
        elif k == 'i':
            config.set('Lasercal', key, str(par_dict[key]))
        elif k == 'f':
            config.set('Lasercal', key, str(par_dict[key]))
        elif k == 's':
            config.set('Lasercal', key, par_dict[key])
        else:
            print('unknown key in writeconf: ', key)
    for key in res_dict.keys():
        config.set('Calib', key, str(np.float32(res_dict[key])))
    for key in fits_dict.keys():
        config.set('Fits', key.upper(), str(fits_dict[key]))
    for key in opt_dict.keys():
        if key in ('debug', 'fit-report', 'scale_win2ima', 'scale_ima2win', 'colorflag', 'bob', 'show_images'):
            configsetbool('Options', key, opt_dict[key])
        else:
            config.set('Options', key, str(opt_dict[key]))
    config.set('Options', 'pngdir', str(pngdir))
    config.write(cfgfile)
    logging.info(f' configuration saved as {conf}')
    cfgfile.close()


# -------------------------------------------------------------------
def write_fits_image(image, filename, fits_dict, dist=True):
    """
    writes image as 32-bit float array into fits-file
    :param image: np.array with image data, scaled to +/- 1.0, b/w or color
    :param filename: filename with extension .fit
    :param fits_dict: content of fits header
    :param dist: True: distorted image; False: undistorted image
    :return: 1 if error, else 0
    """
    if len(image.shape) == 3:
        if image.shape[2] > 3:  # cannot read plots with multiple image planes
            sg.PopupError('cannot convert png image, try bmp or fit', title='write_fits_image')
            return 1
        image = np.transpose(image, (2, 0, 1))
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    hdul = fits.HDUList([hdu])
    fits_dict['BSCALE'] = 32767
    fits_dict['BZERO'] = 0
    fits_dict['COMMENT'] = str(fits_dict['COMMENT'])  # [:20]
    for key in fits_dict.keys():
        if dist:
            hdu.header[key] = fits_dict[key]
        else:
            if key not in ('D_SCALXY', 'D_X00', 'D_Y00', 'D_ROT', 'D_DISP0', 'D_A3', 'D_A5'):
                hdu.header[key] = fits_dict[key]
    hdul.writeto(filename, overwrite=True)
    hdul.close()
    return 0


# -------------------------------------------------------------------
def get_png_image(filename, colorflag=False):
    """
    reads png image and converts to np.array
    :param filename: with extension 'png
    :param colorflag: True: colour image, False: image converted to b/w
    :return: image as 2 or 3-D array
    """
    image = np.flipud(img_as_float(ios.imread(filename)))
    if not colorflag and len(image.shape) == 3:
        image = np.sum(image, axis=2) / 3
    return image


# -------------------------------------------------------------------
def extract_video_images(avifile, pngname, bobdoubler=False, binning=1, bff=True, maxim=200):
    """
    creates png images from AVI file
    :param avifile: filename of avi file (full path, with extension)
    :param pngname: filebase of png images, e.g. tmp/m for series m1.png, m2.png,...
    :param bobdoubler: if True: interlaced frames are separated into fields of half height,
                       default: False, frames are read
    :param binning: integer [1]
    :param bff: if True: bottom field first read for interlaced video, else top field first
    :param maxim: integer, limit for converting images
    :return:
    nim: number of converted images, starting with index 1
    dattim: date and time of video, extracted from filename created in UFO Capture
    sta: station name, extracted from filename created in UFO Capture
    out: full path filebase of extracted images, e.g. data/out/mdist
     """

    # extract dattim and station from filename (for files from UFO capture)
    def tfits(p):
        # f = Path(p).name
        f, ext = path.splitext(path.basename(p))
        if f[0] != 'M':
            print('f:', f)
            f = f[1:]
        t = Time(datetime(int(f[1:5]), int(f[5:7]), int(f[7:9]), int(f[10:12]), int(f[12:14]), int(f[14:16]))).fits
        sta = f[17:22]
        return t, sta

    # -------------------------------------------------------------------
    # subprocess is os specific
    sys = platform.system()
    if sys == 'Windows':
        cshell = False
    else:
        cshell = True
    logging.info(f'Platform: {sys}')
    out = pngname
    pngdir, tmp = path.split(pngname)
    nim = 0
    dattim = ''
    sta = ''
    if avifile:
        avifile = '"' + avifile + '"'  # double quotes needed for filenames containing white spaces
        # path name for png images
        if pngdir:
            if not path.exists(pngdir):
                os.mkdir(pngdir)
        try:
            if bobdoubler:
                # read bottom and top fields
                command = f"ffmpeg -i {avifile} -frames {maxim / 2} -vf field=top {pngdir}/top%d.png -loglevel quiet"
                subprocess.call(command, shell=cshell)
                command = f"ffmpeg -i {avifile} -frames {maxim / 2} -vf field=bottom {pngdir}/bot%d.png -loglevel quiet"
                subprocess.call(command, shell=cshell)
                nfr = 0
                n = 0
                end = False
                # sort and rename fields
                while not end:
                    try:
                        n += 1
                        nfr += 1
                        if bff:
                            os.rename(f'{pngdir}/bot' + str(nfr) + '.png', out + str(n) + '.png')
                            n += 1
                            os.rename(f'{pngdir}/top' + str(nfr) + '.png', out + str(n) + '.png')
                        else:
                            os.rename(f'{pngdir}/top' + str(nfr) + '.png', out + str(n) + '.png')
                            n += 1
                            os.rename(f'{pngdir}/bot' + str(nfr) + '.png', out + str(n) + '.png')
                    except:
                        end = True
                nim = n - 1

            elif binning > 1:
                # binning bin*bin for reducing file size
                command = f"ffmpeg -i {avifile} -frames {maxim} -vf scale=iw/{binning}:-1  {out}%d.png -loglevel quiet"
                subprocess.call(command, shell=cshell)
                nim = check_files(out, maxim)

            else:
                # regular processing of frames
                command = f"ffmpeg -i {avifile} -frames {maxim} {out}%d.png -loglevel quiet"
                subprocess.call(command, shell=cshell)
                nim = check_files(out, maxim)

            if debug:
                print(f'last file written: {out}' + str(nim) + '.png')
            # get dattim from filename
            dattim, sta = tfits(avifile)
        except Exception as e:
            info = 'problem with ffmpeg, no images converted'
            sg.PopupError(info + f'\n{e}', title='AVI conversion')
            logging.error(info)
            logging.error({e})
    return nim, dattim, sta, out


# -------------------------------------------------------------------
def create_file_list(file, n, ext='.png', start=1):
    """
    create a file series according to IRIS convention
    :param file: filebase
    :param n: number of files
    :param ext: extension, default = .png
    :param start: index of first file
    :return: file_list
    """
    result = []
    for a in range(start, start + n):
        filen = file + str(a) + ext
        result.append(filen)
    return result


# -------------------------------------------------------------------
def check_files(file, n, ext='.png'):
    """
    check if files in file series file+index+ext exist, starting with index 1
    :param file: filebase
    :param n: last index to check
    :param ext: file extension, default = .png
    :return: number of files found, 0 if no file exists
    """
    filelist = create_file_list(file, n, ext=ext)
    index = 0
    for i in range(len(filelist)):
        if path.exists(file + str(i + 1) + ext):
            index = i + 1
        else:
            index = i
            return index
    return index


# -------------------------------------------------------------------
def delete_old_files(file, n, ext='.png'):
    """
    delete files in order to clean up directory before new calculation
    :param file: filebase
    :param n: last index to check
    :param ext: file extension, default = .png
    :return:
    number of files found
    number of deleted files
    """
    oldfiles = check_files(file, n, ext)
    deleted = 0
    answer = ''
    if oldfiles:
        answer = sg.PopupOKCancel(f'delete {oldfiles} existing files {file}, \nARE YOU SURE?', title='Delete old Files')
        if answer == 'OK':
            for index in range(oldfiles):
                os.remove(file + str(index + 1) + ext)
            deleted = oldfiles
    return oldfiles, deleted, answer


# -------------------------------------------------------------------
def create_background_image(im, nb, colorflag=False):  # returns background image
    """
    creates background image from first nb png images extracted from
    video with VirtualDub
    Parameters:
    im: filebase of image without number and .png extension
        e.g. m_ for series m_1.png, m_2.png,...
    nb: number of images, starting with index 1,
        for calculation of background image
        n = 0: zero intensity background image
    colorflag: True: color image, False: b/w image output
    Return:
    background image, average of input images, as image array
    """
    if nb > 0:
        # create list of filenames
        image_list = create_file_list(im, nb)
        # open a series of files and store the data in a list,  image_concat
        first = True
        for image in image_list:
            ima = get_png_image(image, colorflag)
            if first:
                image_sum = ima
                first = False
            else:
                image_sum += ima
        ave_image = image_sum / nb
    else:
        # zero background
        ave_image = 0 * get_png_image(im + '1.png', colorflag)
    return ave_image


# -------------------------------------------------------------------
def apply_dark_distortion(im, backfile, outpath, mdist, first, nm, window, fits_dict, dist=False,
                          background=False, center=None, a3=0, a5=0, rotation=0, yscale=1, colorflag=False,
                          show_images=True, cval=0):
    # subtracts background and transforms images in a single step
    """
    subtracts background image from png images and stores the result
    as fit-images
    (peak image from series, sum image from series)
    Perform a dist transformation

    Parameters:
    im: filebase of image without number and .bmp extension
        e.g. m_ for series m_1.bmp, m_2.bmp,...
    backfile: background fit-file created in previous step without extension
    outpath: path to mdist (output files)
    mdist: file base of output files, appended with number starting from 1
    (IRIS convention) and .fit
    first: index of first image converted
    nm: number of images created (if exist)
    dist: flag, if True the distortion is calculated,
        with additional parameters
    background: flag, if True the background image (backfile) is subtracted
    center : (column, row) tuple or (2,) ndarray, optional
        Center coordinate of transformation, corresponds to optical axis.
        If None, the image center is assumed
    a3 : float, optional
        The cubic coefficient of radial transformation
    a5 : float, optional
        The quintic coefficient of radial transformation
        (the linear coefficient is set equal 1 to preserve image scale
        at center, even order coefficients are equal zero due to the
        symmetry of the transformation
    rotation : float, optional
        Additional rotation applied to the image.
    yscale : float, optional
        scales image by a factor in y-direction to compensate for non-square
        pixels. The center coordinate y0 is scaled as well
    colorflag: True for colour images, False for b/w images
    fits_dict: dictionary with fits-header info
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Return:
    actual number of images created
    peak image from series, sum image from series
    disttext: multiline info about success

    The distortion was adapted from skimage.transform.swirl.
    Instead of a swirl transformation a rotation symmetric radial transformation
    for converting tangential projection to orthographic projection and/or to
    correct lens distorsion  described by
    r =rp*(1+a3*rp^2 +a5*rp^4)

    Other parameters, as used in swirl
    ----------------
    # output_shape : tuple (rows, cols), optional
    #     Shape of the output image generated. By default the shape of the input
    #     image is preserved.
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        # 3: Bi-cubic
        # 4: Bi-quartic
        # 5: Bi-quintic
mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode, with 'constant' used as the default.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    # preserve_range : bool, optional
    #     Whether to keep the original range of values. Otherwise, the input
    #     image is converted according to the conventions of `img_as_float`.
        Also see
        http://scikit-image.org/docs/dev/user_guide/data_types.html
"""

    def _distortion_mapping(xy, center, rotation, a3, a5, yscale=1.0):
        """
        the original images are converted to square pixels by scaling y
        with factor yscale
        if yscale is omitted, square pixels are assumed
        Calculate shifted coordinates:  xs,ys =x',y' – x0,y0
        Calculate r', phi':             r' =sqrt(xs^2+ys^2)
                                        phi' =phi = arctan2(ys,xs)
        Calculate r:                    r =r'*(1+a3*(r'/f)^2 +...)
        Calculate x,y:                  x=x0+r*cos(phi)
                                        y= y0 + r*sin(phi)
        (Pixel value at x',y':           I'(x',y') = I(x,y) in the original image)
        """
        x, y = xy.T
        x0, y0 = center
        y0 = y0 * yscale  # the center in the original image has to be scaled as well
        # y has been scaled in a previous step with resize image
        rp = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        phi = np.arctan2(y - y0, x - x0) + rotation
        r = rp * (1 + rp ** 2 * (a3 + a5 * rp ** 2))  # 8sec, 2.9217, 2.906 for single image png
        xy[..., 0] = x0 + r * np.cos(phi)
        xy[..., 1] = y0 + r * np.sin(phi)
        return xy

    idg = None
    dattim = ''
    sta = ''
    # scale image
    back, header = get_fits_image(backfile)
    # notice order of coordinates in rescale
    if center is None:
        center = np.array(back.shape)[:2][::-1] / 2
    warp_args = {'center': center,
                 'a3': a3,
                 'a5': a5,
                 'rotation': rotation,
                 'yscale': yscale}
    # warnings.filterwarnings('ignore') # ignore warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    # create list of filenames
    image_list = create_file_list(im, nm, start=first)
    a = 0
    if dist:
        if len(back.shape) == 3:
            multichannel = True
        else:
            multichannel = False
        ima = my_rescale(back, (yscale, 1), multichannel=multichannel)  # scale sum and peak image start
        if debug:
            print('imy imx , x00 y00: ', ima.shape, center)
    else:
        ima = back
    imsum = 0 * ima
    impeak = imsum
    t1 = time.time()
    fullmdist = outpath + '/' + mdist
    for image in image_list:
        if path.exists(image):
            a += 1  # create output filename suffix
            fileout = fullmdist + str(a)
            idist = get_png_image(image, colorflag)
            if background:
                idist = idist - back  # subtract background
            # calculate distortion
            if dist:
                if abs(yscale - 1.0) > 1.0e-3:  # scale image if yscale <> 1.0
                    idist = my_rescale(idist, (yscale, 1), multichannel=multichannel)
                if len(idist.shape) == 3:
                    for c in [0, 1, 2]:  # separate color planes for faster processing
                        idist2 = idist[:, :, c]
                        # use bi-quadratic interpolation (order = 2) for reduced fringing
                        idist2 = tf.warp(idist2, _distortion_mapping, map_args=warp_args, order=2,
                                         mode='constant', cval=cval)
                        idist[:, :, c] = idist2
                else:
                    idist = tf.warp(idist, _distortion_mapping, map_args=warp_args, order=2,
                                    mode='constant', cval=cval)
            write_fits_image(idist, fileout + '.fit', fits_dict, dist=dist)
            if show_images:
                image_data, idg, actual_file = draw_scaled_image(fileout + '.fit',
                                    window['-D_IMAGE-'], opt_dict, idg, resize=True)
            # create sum and peak image
            imsum = imsum + idist
            file = path.basename(fileout + '.fit')
            impeak = np.maximum(impeak, idist)
            disttext = f'{file} of {nm} done\n'
            window['-RESULT2-'].update(value=disttext, append=True)
            window.refresh()
    # write sum and peak fit-file
    write_fits_image(imsum, fullmdist + '_sum.fit', fits_dict, dist=dist)
    save_fit_png(fullmdist + '_peak', impeak, fits_dict)
    nmp = a
    # print(nmp, ' images processed of ', nm)
    logging.info(f'{nmp} images processed of {nm}')
    tdist = (time.time() - t1) / nmp
    disttext = f'{nmp} images processed of {nm}\n'
    if dist:
        info = f'process time for single distortion: {tdist:8.2f} sec'
        logging.info(info)
        disttext += info + '\n'
        # print(f'process time background, dark and dist {t2:8.2f} sec')
        if 'DATE-OBS' in fits_dict.keys():
            dattim = fits_dict['DATE-OBS']
            sta = fits_dict['M_STATIO']
        else:
            logging.info('no fits-header DATE-OBS, M-STATIO')
            disttext += '\n!!!no fits-header DATE-OBS, M-STATIO!!!\n'
        logging.info(f"'DATE-OBS' = {dattim}")
        logging.info(f"'M-STATIO' = {sta}")
        if not background:
            logging.info('no background applied')
        info = f'Bobdoubler, start image = {im}{first}'
        if int(fits_dict['M_BOB']):
            logging.info(f'with ' + info)
        else:
            logging.info('without ' + info)
    return a, imsum, impeak, disttext


# -------------------------------------------------------------------
def register_images(start, nim, x0, y0, dx, dy, infile, outfil, window, fits_dict, contr=1, idg=0, show_reg=False):
    """
    :param start: index of first image (reference) for registering
    :param nim: number of images to register_images
    :param x0: x-coordinate of reference pixel (int)
    :param y0: y-coordinate of reference pixel (int)
    :param dx: half width of selected rectangle
    :param dy: half height of selected rectangle
    :param infile: full filebase of images e.g. out/mdist
    :param outfil: filebase of registered files, e.g. out/mdist
    :param window: GUI window for displaying results of registered files
    :param fits_dict: content of fits-header
    :param contr: image brightness
    :param idg: graph number, used to delete previous graph
    :param show_reg: if True show registered images, otherwise mdist images
    if the procedure stops early, nim = index - start + 1
    :return:
    index: last processed image
    sum_image: !average! of registered images
    regtext: multiline text of results
    dist: if True, distorted images, else False
    outfile: filename of sum-image, e.g. for sum of 20 added images: out/r_add20
    fits_dict: updated values of fits-header
    """

    def _shift(xy):
        return xy - np.array(dxy)[None, :]

    index = start
    sum_image = []
    outfile = ''
    dist = False
    fits_dict.pop('M_NIM', None)  # M_NIM only defined for added images
    info = f'start x y, dx dy, file: {x0} {y0},{2 * dx} {2 * dy}, {infile}'
    regtext = info + '\n'
    logging.info(info)
    image_list = create_file_list(infile, nim, ext='', start=start)
    info = f'file          peak    x         y    wx   wy'
    regtext += '        ' + info + '\n'
    logging.info(info)
    try:
        for image_file in image_list:
            im, header = get_fits_image(image_file)
            if 'D_X00' in header.keys():
                dist = True
            if 'M_BOB' in header.keys():
                fits_dict['M_BOB'] = header['M_BOB']
            if len(im.shape) == 3:
                imbw = np.sum(im, axis=2)  # used for fit_gaussian_2d(data)
                data = imbw[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
                shifted = im
            # selected area
            else:
                data = im[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
            params, success = fit_gaussian_2d(data)
            (height, x, y, width_x, width_y) = params  # x and y reversed
            width_x = 2 * np.sqrt(2 * np.log(2)) * np.abs(width_x)  # FWHM
            width_y = 2 * np.sqrt(2 * np.log(2)) * np.abs(width_y)  # FWHM
            # full image
            x = x + y0 - dy  # y and x reversed
            y = y + x0 - dx
            imagename = os.path.basename(image_file)
            info = f'{imagename:12s} {height:7.3f} {y:6.1f} {x:6.1f} {width_y:5.2f} {width_x:5.2f}'
            regtext += info + '\n'
            window['-RESULT3-'].update(regtext)
            window.refresh()
            logging.info(info)
            if index == start:  # reference position for register_images
                x00 = y
                y00 = x
            # register_images
            dxy = [x00 - y, y00 - x]
            if len(im.shape) == 3:
                for c in [0, 1, 2]:  # separate color planes for faster processing
                    im2 = im[:, :, c]
                    coords = tf.warp_coords(_shift, im2.shape)
                    sh2 = map_coordinates(im2, coords)  # / 255
                    shifted[:, :, c] = sh2
            else:
                coords = tf.warp_coords(_shift, im.shape)
                shifted = map_coordinates(im, coords)  # / 255
            if index == start:  # reference position for register_images
                sum_image = shifted

            else:
                sum_image += shifted
            # write image as fit-file
            write_fits_image(shifted, outfil + str(index - start + 1) + '.fit', fits_dict, dist=dist)
            if show_reg:
                image_data, idg, actual_file = draw_scaled_image(outfil + str(index - start + 1) + '.fit',
                                                                 window['-R_IMAGE-'], opt_dict, idg, contr=contr,
                                                                 resize=True, tmp_image=True)
                window.set_title('Register: ' + str(actual_file))
                window.refresh()
            # set new start value
            x0 = int(y)
            y0 = int(x)
            index += 1  # next image
        index += -1
    except Exception as e:
        # Exception, delete last image with error
        if path.exists(outfil + str(index - start + 1) + '.fit'):
            os.remove(outfil + str(index - start + 1) + '.fit')
        index += -1
        info = f'problem with register_images, last image: {image_file}, number of images: {index}'
        logging.error(info)
        logging.error({e})
        regtext += f'{info}\n{e}\n'
    nim = index - start + 1
    if nim > 1:
        if index == nim + start - 1:
            outfile = outfil + '_add' + str(nim)
            sum_image = sum_image / nim  # averaging
            fits_dict['M_STARTI'] = str(start)
            fits_dict['M_NIM'] = str(nim)
            write_fits_image(sum_image, outfile + '.fit', fits_dict, dist=dist)
    return index, sum_image, regtext, dist, outfile, fits_dict


# -------------------------------------------------------------------
def _gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a _gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


# -------------------------------------------------------------------
def _moments(data):
    """Returns (height, x, y, width_x, width_y)
    the _gaussian parameters of a 2D distribution by calculating its
    _moments """
    height = x = y = width_x = width_y = 0.0
    total = data.sum()
    if total > 0.0:
        xx, yy = np.indices(data.shape)
        x = (xx * data).sum() / total
        y = (yy * data).sum() / total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
        height = data.max()
        # print('h: %5.1f'%height,'x: %5.1f'%x, 'y: %5.1f'%y, 'wx: %5.1f'%width_x, 'wy: %5.1f'%width_y)
    return height, x, y, width_x, width_y


# -------------------------------------------------------------------
def fit_gaussian_2d(data):
    """Returns (height, x, y, width_x, width_y)
    the _gaussian parameters of a 2D distribution found by a fit
    (ravel makes 1-dim array)"""
    params = _moments(data)
    success = 0
    if params[0] > 0:
        errorfunction = lambda p: np.ravel(_gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p, success


# -------------------------------------------------------------------
def get_fits_keys(header, fits_dict, res_dict, keyprint=False):
    """
    gets fits-header from image and appends or overwrites the current fits-header
    dictionary. It also converts the specially coded D_ keys to res_dict keys
    and attributes floating type values
    updates res_dict
    :param header:
    :param fits_dict: dictionary of fits-header values
    :param res_dict: dictionary of distortion parameters (result of m-calib)
    :param keyprint:
    :return: updated fits_dict
    """
    for key in fits_dict.keys():
        if key in header.keys():
            fits_dict[key] = header[key]
            if keyprint:
                print(key, header[key])
    for key in res_dict.keys():
        fkey = 'D_' + key.upper()
        if fkey in header.keys():
            res_dict[key] = np.float32(header[fkey])
            fits_dict[fkey] = np.float32(header[fkey])
            if keyprint:
                print(key, fits_dict[fkey])
    return fits_dict


# -------------------------------------------------------------------
def get_fits_image(fimage):
    """
    reads fits image data and header
    fimage: filename with or without extension
    converts 32-bit floating values and 16-bit data to Python compatible values
    reads also color images and transposes matrix to correct order
    (normalizes images to +/- 1 range)
    returns: image as np array, header
    """
    fimage = change_extension(fimage, '.fit')
    im, header = fits.getdata(fimage, header=True)
    if int(header['BITPIX']) == -32:
        im = np.array(im) / 32767
    elif int(header['BITPIX']) == 16:
        im = np.array(im)
    else:
        print(f'unsupported data format BITPIX: {header["BITPIX"]}')
        exit()
    if len(im.shape) == 3:
        im = np.transpose(im, (1, 2, 0))
    return im, header


# -------------------------------------------------------------------
def show_fits_image(file, imscale, image_element, contr=1.0, show=True):
    """
    not needed at present, left in place for further use
    loads fits-image, adjusts contrast and scale and displays in GUI as tmp.png
    replaced by draw_scaled_image
    :param file: fits-file with extension
    :param imscale: scale for displayed image
    :param image_element: where to display image in GUI
    :param contr: image contrast
    :param show: if True, image_element isupdated, otherwise only 'tmp.png' is created
    :return:
    """
    imbw, header = get_fits_image(file)
    if len(imbw.shape) == 2:
        im = my_rescale(imbw, imscale, multichannel=False)
    else:
        im = my_rescale(imbw, imscale, multichannel=True)
    im = im / np.max(im) * 255 * contr
    im = np.clip(im, 0.0, 255)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ios.imsave('tmp.png', np.flipud(im).astype(np.uint8))
    if show:
        image_element.update(filename='tmp.png')
    return


# -------------------------------------------------------------------
def select_rectangle(infile, start, res_dict, fits_dict, wloc, outfil, maxim):
    """
    displays new window with image infile + start + 'fit
    a rectangle around the selected line can be selected with dragging the mouse
    :param infile: filebase of image
    :param start: index of selected image
    :param res_dict: dictionary
    :param fits_dict: "
    :param wloc: location of displayed window for selection
    :param outfil:
    :param maxim:
    :return:
    Ok if rectangle selected,
    x0, y0: center coordinates of selected rectangle (int)
    dx, dy: half width and height of selected rectangle (int)
    """
    im, header = get_fits_image(infile + str(start))
    im = im / np.max(im)
    get_fits_keys(header, fits_dict, res_dict, keyprint=False)
    # #===================================================================
    # new rect_plt
    # first get size of graph from tmp.png and size of image
    # graph coordinates are in image pixels!
    (imy, imx) = im.shape[:2]
    image_file = 'tmp.png'  # scaled image
    imbw = np.flipud(ios.imread(image_file))  # get shape
    (canvasy, canvasx) = imbw.shape[:2]
    wlocw = (wloc[0] + 300, wloc[1] + 50)
    # check for old files
    delete_old_files(outfil, maxim, ext='.fit')
    image_elem_sel = [sg.Graph(
        canvas_size=(canvasx, canvasy),
        graph_bottom_left=(0, 0),  # starts at top, set y-scale here
        graph_top_right=(imx, imy),  # set x-scale here
        key='-GRAPH-',
        change_submits=True,  # mouse click events
        drag_submits=True)]
    layout_select = [[sg.Text('Start File: ' + infile + str(start), size=(50, 1)), sg.Text(key='info', size=(40, 1)),
                      sg.Ok(), sg.Cancel()],
                     image_elem_sel]
    # ---------------------------------------------------------------------------
    winselect_active = True
    winselect = sg.Window(f'select zero order or spectral line',
                          layout_select, finalize=True, location=wlocw,
                          keep_on_top=True, no_titlebar=False,
                          disable_close=False, disable_minimize=True)
    # get the graph element for ease of use later
    graph = winselect['-GRAPH-']  # type: sg.Graph
    graph.draw_image(image_file, location=(0, imy)) if image_file else None
    winselect.refresh()
    dragging = False
    start_point = end_point = prior_rect = None
    x0 = y0 = dx = dy = 0
    while winselect_active:
        event, values = winselect.read()
        idg = graph.draw_rectangle((0, 0), (imx, imy), line_color='blue')
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
            if min(size[0], size[1]) > 1:  # rectangle
                info.update(value=f"rectangle at {xy0} with size {size}")
                x0 = xy0[0]
                y0 = xy0[1]
                dx = int((size[0] + 1) / 2)
                dy = int((size[1] + 1) / 2)

        elif event in ('Ok', 'Cancel'):
            graph.delete_figure(idg)
            winselect_active = False
            winselect.close()
    return event, x0, y0, dx, dy


# -------------------------------------------------------------------
def add_rows_apply_tilt_slant(outfile, par_dict, res_dict, fits_dict, opt_dict,
                              contr, wlocw, restext, regtext, window):
    """
    displays new window with image outfile.fit for selection of rows to be added
    allows adjustment of tilt and slant after selection of rows
    if Ok, images outfile + ['st.fit', 'st,png'] are saved
    :param outfile:
    :param par_dict:
    :param res_dict:
    :param fits_dict:
    :param opt_dict:
    # :param imscale:
    :param contr:
    :param wlocw:
    :param restext:
    :param regtext:
    :param window:
    :return:
    Ok if selection is accepted
    tilt, slant: selected values for image outfile + ['st.fit', 'st,png']
    """

    def _slant_tilt_mapping(xy, center, dx, dy):
        """
        Calculate shifted coordinates:  xs = x' - (y'-y0)*dx (slant)
                                        ys = y' - (x'-x0)*dy (tilt)
        (Pixel value at x',y':          I'(x',y') = I(x,y) in the original image)
        """
        x, y = xy.T
        x0, y0 = center
        xy[..., 0] = x - (y - y0) * dx
        xy[..., 1] = y - (x - x0) * dy
        return xy

    tilt = 0.0
    slant = 0.0
    ymin = 0
    ymax = 0
    idg = None
    im, header = get_fits_image(outfile)
    if 'D_X00' in header.keys():
        dist = True
    else:
        dist = False
    if debug:
        print(np.max(im))
    im = im / np.max(im)
    imtilt = im_ori = im
    fits_dict = get_fits_keys(header, fits_dict, res_dict, keyprint=False)
    write_fits_image(imtilt, outfile + 'st.fit', fits_dict, dist=dist)  # used for calibration, if no tilt, slant
    # new rect_plt
    (imy, imx) = im.shape[:2]
    imbw = np.flipud(ios.imread('tmp.png'))  # get shape
    (canvasy, canvasx) = imbw.shape[:2]
    # wlocw = (wloc[0] + 300, wloc[1] + 100)
    image_file = 'tmp.png'
    # -------------------------------------------------------------------
    par_dict['i_imx'] = imx
    par_dict['i_imy'] = imy
    image_elem_sel = [sg.Graph(
        canvas_size=(canvasx, canvasy),
        graph_bottom_left=(0, 0),
        graph_top_right=(imx, imy),  # set x- and y-scale here
        key='-GRAPH-',
        change_submits=True,  # mouse click events
        drag_submits=True)]
    layout_select = [[sg.Text('Start File: ' + outfile, size=(40, 1)),
                      sg.Checkbox('correct background', key='-BACK-'),
                      sg.Text('Tilt'), sg.InputText(tilt, size=(6, 1), key='-TILT-'),
                      sg.Text('Slant'), sg.InputText(slant, size=(6, 1), key='-SLANT-'),
                      sg.Button('Apply', key='-APPLY_TS-', bind_return_key=True),
                      sg.Ok(), sg.Cancel()],
                     image_elem_sel, [sg.Text(key='info', size=(60, 1))]]
    # ---------------------------------------------------------------------------
    winselect_active = True
    winselect = sg.Window(f'select rows for 1-D sum spectrum, apply tilt and slant',
                          layout_select, finalize=True, location=wlocw,
                          keep_on_top=True, no_titlebar=False,
                          disable_close=False, disable_minimize=True)
    # get the graph element for ease of use later
    graph = winselect['-GRAPH-']  # type: sg.Graph
    graph.draw_image(image_file, location=(0, imy)) if image_file else None
    dragging = False
    start_point = end_point = prior_rect = upper_back = lower_back = None

    while winselect_active:
        event, values = winselect.read()
        graph.draw_rectangle((0, 0), (imx, imy), line_color='blue')
        if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            background = values['-BACK-']
            x, y = (values["-GRAPH-"])
            if not dragging:
                start_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            if upper_back:
                graph.delete_figure(upper_back)
                graph.delete_figure(lower_back)
            if None not in (start_point, end_point):
                ymin = min(start_point[1], end_point[1])
                ymax = max(start_point[1], end_point[1])
                width = ymax - ymin
                if background:
                    upper_back = graph.draw_rectangle((0, ymax + width // 2),
                                                      (imx, ymax + width // 2 + width), line_color='green')
                    lower_back = graph.draw_rectangle((0, ymin - width // 2 - width),
                                                      (imx, ymin - width // 2), line_color='green')
                prior_rect = graph.draw_rectangle((0, ymin),
                                                  (imx, ymax), line_color='red')
        elif event is not None and event.endswith('+UP'):
            # The drawing has ended because mouse up
            y0 = int(0.5 * (start_point[1] + end_point[1]))
            info = f"selected lines from {ymin} to {ymax}"
            winselect["info"].update(value=info)
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            restext += info + '\n'
            window['-RESULT3-'].update(regtext + restext)

        elif event == '-APPLY_TS-':
            if ymax == 0:
                sg.PopupError('select rows first', keep_on_top=True)
            else:
                try:
                    tilt = float(values['-TILT-'])
                    slant = float(values['-SLANT-'])
                    image = im
                    center = (image.shape[1] / 2, y0)
                    warp_args = {'center': center,
                                 'dx': slant,
                                 'dy': tilt}
                    imtilt = tf.warp(image, _slant_tilt_mapping, map_args=warp_args,
                                     order=1, mode='constant', cval=0)
                    fits_dict['M_TILT'] = str(tilt)
                    fits_dict['M_SLANT'] = str(slant)
                    fits_dict['M_ROWMIN'] = str(ymin)
                    fits_dict['M_ROWMAX'] = str(ymax)
                    fits_dict['COMMENT'] = str(fits_dict['COMMENT'])  # [:20] # shorten to max size
                    restext += f'tilt = {tilt:8.4f}, slant = {slant:7.3f}' + '\n'
                    window['-RESULT3-'].update(regtext + restext, autoscroll=True)

                except Exception as e:
                    sg.PopupError(f'bad values for tilt or slant, try again\n{e}',
                                  title='apply_tilt_slant', keep_on_top=True)
                write_fits_image(imtilt, '_st.fit', fits_dict, dist=dist)
                image_data, idg, actual_file = draw_scaled_image('_st' + '.fit', window['-R_IMAGE-'],
                                                                 opt_dict, idg, contr=contr, tmp_image=True)
                graph.draw_image(data=image_data, location=(0, imy))
                # graph.draw_rectangle((0, ymin), (imx, ymax), line_color='red')
                for figure in (prior_rect, upper_back, lower_back):
                    if figure:
                        graph.BringFigureToFront(figure)
                graph.update()

        elif event == 'Ok':
            write_fits_image(imtilt, outfile + 'st.fit', fits_dict, dist=dist)
            image_data, idg, actual_file = draw_scaled_image(outfile + 'st.fit', window['-R_IMAGE-'],
                                                             opt_dict, idg, contr=contr, tmp_image=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im = imtilt / np.max(imtilt) * 255 * contr
                im = np.clip(im, 0.0, 255)
                ios.imsave(outfile + 'st.png', np.flipud(im.astype(np.uint8)))
            logging.info(f' file {outfile}.fit loaded for addition of rows')
            logging.info(f"start = {fits_dict['M_STARTI']}, nim = {fits_dict['M_NIM']}")
            logging.info(f'added from {ymin} to {ymax}, {(ymax - ymin + 1)} rows')
            logging.info(f'tilt = {tilt:8.4f}, slant = {slant:7.3f}')
            if len(imtilt.shape) == 3:
                imbw = np.sum(imtilt, axis=2)
            else:
                imbw = imtilt
            row_sum = np.sum(imbw[ymin:ymax, :], axis=0)  # Object spectrum extraction and flat
            i = np.arange(0, np.size(row_sum), 1)  # create pixels vector
            np.savetxt(outfile + '.dat', np.transpose([i, row_sum]), fmt='%6i %8.5f')
            # background correction for reference star images
            if background:
                row_sum -= 0.5*np.sum(imbw[ymax + width // 2:ymax + 3 * width // 2, :], axis=0)
                row_sum -= 0.5*np.sum(imbw[ymin - 3 * width // 2:ymin - width // 2, :], axis=0)
                np.savetxt(outfile + '_bg.dat', np.transpose([i, row_sum]), fmt='%6i %8.5f')
            fits_dict.pop('M_TILT', None)
            fits_dict.pop('M_SLANT', None)
            fits_dict.pop('M_ROWMIN', None)
            fits_dict.pop('M_ROWMAX', None)
            winselect_active = False
            (x, y) = winselect.current_location()
            wlocw = (x, y)
            if idg: graph.delete_figure(idg)
            winselect.close()
            window['-SAVE_RAW-'].update(disabled=False, button_color=bc_enabled)
            window['-CAL_R-'].update(disabled=False, button_color=bc_enabled)
            window['-RADD-'].update(outfile)

        elif event in ('Cancel', None):
            # save original image with 'st' added
            if event == 'Cancel':
                write_fits_image(im_ori, outfile + 'st.fit', fits_dict, dist=dist)
            if idg: graph.delete_figure(idg)
            winselect_active = False
            winselect.close()
    return event, tilt, slant, wlocw


# -------------------------------------------------------------------
def select_calibration_line(x0, w, lam, name, lcal, ical, graph, table, abs_sign=1):
    """
    fits a parabola to peak of selected line, determines peak position, intensity and width
    results are appended to table together with wavelength of selected line
    :param x0: peak position in pixel
    :param w:
    :param lam: wavelength of selected calibration line
    :param name: identifier for calibration line
    :param lcal: pixel array
    :param ical: intensity array
    :param graph: displayed window
    :param table: table with calibration results
    :param abs_sign: sign for emission = + 1, absorption = -1
    :return:
    coeff[1]: peak position of parabolic fit
    fwp: peak width (approximate)
    caltext: updated info
    """

    def parab(x, *p):
        aa, mu, b = p
        return aa * (1 - b * (x - mu) ** 2)

    coeff = [-1, 0, 0]
    fwp = 0
    lmin = lcal[0]
    ic = lcal
    lc = ical
    icleft = int(x0 - w)  # index of left border
    icright = int(x0 + w + 1)
    if lmin not in (0.0, 1.0):
        sg.PopupError('- raw files only,   load uncalibrated file!.......', title='Wavelength calibration',
                      line_width=60)
    else:
        try:
            lcr0 = abs_sign*lc[icleft:icright]
            lmax0 = np.max(lcr0)
            lmin0 = (lcr0[0] + lcr0[icright - icleft - 1]) / 2
            for i in range(icright - icleft):
                if (lcr0[i] - lmax0 + 1.e-5) > 0:
                    m = i
            peak0 = icleft + m  # index of max value, center for parabolic fit
            icr = ic[peak0 - 2:peak0 + 3]
            lcr = abs_sign*lc[peak0 - 2:peak0 + 3]
            coeff[0] = lmax0
            coeff[1] = peak0
            coeff[2] = 1 / (w + 1)
            coeff, var_matrix = optimize.curve_fit(parab, icr, lcr, p0=coeff)
            # lcp_fit = parab(icr, *coeff) # function for display results
            x0p = coeff[1]
            fwp = np.sqrt(1.5 / abs(coeff[2]))
            if abs_sign < 0:  # Absorption line, correct peak height
                fwp *= np.sqrt(abs((coeff[0]-lmin0) / coeff[0]))
            # parabolic fit
            if debug:
                print(f'x0p ={x0p:8.2f} FWHMP={fwp:8.3f}')
            points = []
            l0: int
            for l0 in range(peak0 - 2, peak0 + 3):
                points.append((lcal[l0], abs_sign*parab(lcal[l0], *coeff)))
            for l0 in range(1, 5):
                graph.DrawLine(points[l0 - 1], points[l0], 'blue', 1)
            table.append((coeff[1], lam))
            info = f'{coeff[1]:8.2f} {fwp:6.2f} {lam:8.2f} {name}'
            caltext = info + '\n'
            logging.info(info)
        except Exception as e:
            sg.PopupError(f'No peak found, try again\n{e}', title='Select line')
    return coeff[1], fwp, caltext


# -------------------------------------------------------------------
def create_line_list_combo(m_linelist, window, combo=True):
    """
    shows values of table create_line_list_combo in Combobox
    :param m_linelist: table with wavelength, line identifier (space separated)
    :param window: Combobox for selecting wavelength
    :param combo: if True: update Combo, else only create list
    :return: label_str, lam_calib
    """
    try:
        lam_calib = []
        label_str = []
        i = -1
        with open(change_extension(m_linelist, '.txt')) as f:
            for x in f:
                x = x.lstrip()
                (l, name) = x.split(' ', 1)
                i += 1
                lam_calib.append(x)
                label_str.append((float(l), name))
                if abs(float(l)) < 0.1:
                    index0 = i  # set default index for list
        if combo:
            window['-LAMBDA-'].update(values=lam_calib, set_to_index=index0)
    except Exception as e:
        sg.PopupError(f'error with calibration lines {m_linelist}.txt\n{e}', keep_on_top=True)
    return label_str, lam_calib


# -------------------------------------------------------------------
def read_video_list(file):
    """
    reads list of latest converted video files from table
    :param file: table of video files
    :return: list of video files
    """
    video_list = []
    if path.exists(file):
        with open(file, 'r') as f:
            for line in f:
                video_list.append(line[:-1])
    return video_list


# -------------------------------------------------------------------
def update_video_list(file, avifile):
    """
    updates list of latest converted video files from table
    :param file: filename of video file table, e.g. 'videolist.txt'
    :param avifile: filename to be added to video file table
    """
    video_list = read_video_list(file)
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


# -------------------------------------------------------------------
def calibrate_raw_spectrum(rawspec, xcalib, lcalib, deg, c):
    """
    calculates the fit for the calibration table with residuals
    from the polynomial fit
    and apply those to the pixels vector
    :param rawspec: uncalibrated spectrum
    :param xcalib: measured pixel positions
    :param lcalib: calibration wavelengths
    :param deg: degree of fit polynom
    :param c: fit polynom
    :return:
    caldat: calibrated spectrum with extension .dat
    cal2dat: calibrated spectrum with constant wavelength spacing with extension .dat
    lmin, lmax: wavelength range of calibrated spectrum
    caltext: calibration info
    """
    np.set_printoptions(precision=4, suppress=False)
    lcal, ical = np.loadtxt(rawspec, unpack=True, ndmin=2)
    logging.info(f'polynom for fit lambda c: {c}')
    i = np.arange(0, len(lcal), 1)  # create pixels vector for uncalibrated image
    lam = np.poly1d(c)(i)
    res = np.poly1d(c)(xcalib) - lcalib
    rms_x = np.sqrt(np.average(np.square(res)))
    logging.info('    pixel     lambda      fit        error')
    caltext = '   Pixel     lambda        fit    error\n'
    for i in range(0, len(xcalib)):
        logging.info(f'{xcalib[i]:10.2f},{lcalib[i]:10.2f},{(lcalib[i] + res[i]):10.2f}, {res[i]:10.4f}')
        caltext += f'{xcalib[i]:9.2f} {lcalib[i]:9.2f} {(lcalib[i] + res[i]):9.2f}  {res[i]:8.2f}\n'
    logging.info(f'rms_x = {rms_x:8.4f}')
    caldat = change_extension(rawspec, 'cal.dat')
    np.savetxt(caldat, np.transpose([lam, ical]), fmt='%8.3f %8.5f')
    logging.info(f'spectrum {caldat} saved')
    caltext += f'polynom degree: {deg}\npolynom for fit lambda c: {c}\n'
    caltext += f'rms_x = {rms_x:8.4f}\nspectrum {caldat} saved\n'
    # for compatibility save *.dat with linear spacing
    lmin = np.int(np.min(lam)) + 1
    lmax = np.int(np.max(lam)) - 1
    dell = abs(int(5 * c[deg - 1]) / 10)
    # wavelength spacing of interpolated linear array, about double of original
    llin = np.arange(lmin, lmax, dell)
    y2 = interpolate.interp1d(lam, ical, kind='quadratic')(llin)
    cal2dat = change_extension(rawspec, 'cal2.dat')
    np.savetxt(cal2dat, np.transpose([llin, y2]), fmt='%8.3f %8.5f')
    return caldat, cal2dat, lmin, lmax, caltext


# -------------------------------------------------------------------
def change_extension(file_name, extension=''):
    """
    if no extension is specified, it is stripped from the filename
    :param file_name: original filename (str)
    :param extension: new extension, (str), eg. '.txt'
    :return: filename with new extension
    """
    base, ext = path.splitext(file_name)
    return base + extension


# -------------------------------------------------------------------
def log_window(logfile):
    """
    displays logfile in new window (cannot be edited)
    :param logfile: filename with actual date, e.g. m_spec200430.log
    """
    with open(logfile, "r") as f:
        log = f.read()
    window = sg.Window('Logfile:' + logfile,
                       [[sg.Multiline(log, size=(120, 30), autoscroll=True,
                                      auto_size_text=True, key='-MLINE-')],
                        [sg.Button('End'), sg.Button('Exit')]], keep_on_top=True)
    while True:  # Event Loop
        event, values = window.read()
        if event == 'End':
            window['-MLINE-'].update(value='', append=True)
        if event in ('Exit', None):
            break
    window.close()


# -------------------------------------------------------------------
def edit_text_window(text_file, select=True, size=(100, 30), default_extension='*.txt'):
    """
    displays editor window, file is saved under the same name
    :param text_file: filename
    :param select: if True, open file open dialog
    :param size: (columns, rows)
    :param default_extension: of filename
    """
    tmp_file = path.basename(text_file)
    if select:
        if default_extension == '*.dat':
            files = ('Spectra', '*.dat')
        else:
            files = ('Text Files', '*.txt')
        file, info = my_get_file(tmp_file, title='Edit Text File',
                                 file_types=(files, ('ALL Files', '*.*'),),
                                 default_extension=default_extension)
    else:
        file = tmp_file
    if file:
        with open(file, 'r') as f:
            text = f.read()
        window = sg.Window('Edit Text File: ' + file,
                           [[sg.Multiline(text, size=size, autoscroll=True,
                                          key='-MLINE-', font='Courier')],
                            [sg.Button('Save'), sg.Button('Cancel')]], keep_on_top=True)
        while True:  # Event Loop
            event, values = window.read()
            if event == 'Save':
                with open(file, 'w') as f:
                    f.write(values['-MLINE-'])
                break
            if event in ('Cancel', None):
                break
        window.close()


# -------------------------------------------------------------------
def view_fits_header(fits_file):
    """
    shows window with fits-header keys and values, not editable
    :param fits_file: filename of fits-file
    """
    file, info = my_get_file(fits_file, title='View Fits-Header', default_extension='.fit',
                             file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),))
    if file:
        im, header = fits.getdata(file, header=True)
        text = ''
        for key in header:
            line = f'{key:>20}: {header[key]}\n'
            text += line
        sg.Window('View Fits-Header: ' + file,
                  [[sg.Multiline(text, size=(60, 30), autoscroll=True, key='-MLINE-', font='Courier')],
                   [sg.Button('Exit')]], keep_on_top=True).read(close=True)


# -------------------------------------------------------------------
def about(version, program='M_Spec'):
    """
    shows program information, author, copyright, version
    :param version: version of main script
    :param program: default: 'M_Spec', alternative: 'M_Calib'
    """
    font = ('Helvetica', 12)
    sg.Window(program, [[sg.Text(program, font=('Helvetica', 20))],
                        [sg.Text('Analysis of meteor spectra from Video files', font=font)],
                        [sg.Text(f'Version = {version}', font=font)],
                        [sg.Text('copyright M. Dubs, 2020', font=font)],
                        [sg.Image('Martin.png'), sg.Button('Ok', font=font)]],
              keep_on_top=True, disable_minimize=True).read(close=True)


# -------------------------------------------------------------------
def add_images(graph_size, contrast=1, average=True):
    """
    shows window for selection of images to add and resulting sum-image
    or by default the average of the images
    :param graph_size: canvas size of graph in pixel
    :param contrast: brightness of displayed images
    :param average: if True, calculate average
    :return: filename of sum-image, number of images or '', 0
    """
    files = []
    idg = None
    max_width = opt_dict['win_width'] - 350  # graph size as for main window
    max_height = opt_dict['win_height'] - 111
    graph_element = sg.Graph(
        canvas_size=(max_width, max_height), graph_bottom_left=(0, 0), graph_top_right=graph_size,
        key='graph', change_submits=True, drag_submits=True)
    window = sg.Window('Add registered images', [[sg.Input('', key='add_images', size=(80, 1)),
                              sg.Button('Load Files')],
                             [sg.Text('Number Images:'), sg.Input('0', size=(8, 1), key='nim'),
                              sg.Button('Darker'), sg.Button('Brighter')],
                             [graph_element], [sg.Button('Save'), sg.Button('Cancel')]])
    while True:  # Event Loop
        event, values = window.read()
        if event == 'Load Files':
            files, info = my_get_file('', title='Add images', default_extension='.fit',
                                multiple_files=True,
                                file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),))
            if files:
                sum_image = []
                number_images = 0
                short_files = path.dirname(files[0])
                try:
                    for file in files:
                        short_files += ' ' + path.basename(file)
                        image, header = get_fits_image(change_extension(file, ''))
                        if sum_image == []:
                            sum_image = image
                        else:
                            sum_image += image
                        number_images += 1
                    if average and number_images:
                        sum_image /= number_images
                    get_fits_keys(header, fits_dict, res_dict)
                    fits_dict['M_STARTI'] = '0'  # set value for special addition
                    dist = False
                    for key in header.keys():
                        if key == 'D_A3':
                            dist = True
                    fits_dict['M_NIM'] = str(number_images)
                    write_fits_image(sum_image, '_add.fit', fits_dict, dist=dist)
                    # show_fits_image('tmp', imscale, window['sum_image'], contr=contrast)
                    image_data, idg, actual_file = draw_scaled_image('_add.fit', window['graph'],
                                                                     opt_dict, idg, contr=contrast)
                    window['add_images'].update(short_files)
                    window['nim'].update(str(number_images))
                    window.refresh()
                except Exception as e:
                    sg.PopupError(f'Images cannot be added, different size?\n{e}', title='add_images')
        if event == 'Darker':
            contrast = 0.5 * contrast
            image_data, idg, actual_file = draw_scaled_image('_add.fit', window['graph'],
                                                             opt_dict, idg, contr=contrast)
            window.refresh()
        if event == 'Brighter':
            contrast = 2.0 * contrast
            image_data, idg, actual_file = draw_scaled_image('_add.fit', window['graph'],
                                                             opt_dict, idg, contr=contrast)
        window.refresh()
        if event == 'Save' and files:
            sum_file, info = my_get_file('', title='Save images', save_as=True, default_extension='.fit',
                                         file_types=(('Image Files', '*.fit'), ('ALL Files', '*.*'),))
            if sum_file:
                write_fits_image(sum_image, sum_file, fits_dict)
                if idg: window['graph'].delete_figure(idg)
                window.close()
                return change_extension(sum_file, ''), number_images
        if event in ('Cancel', None):
            if idg: window['graph'].delete_figure(idg)
            window.close()
            return '', 0


# -------------------------------------------------------------------
def set_image_scale(imx, imy, opt_dict):
    """
    sets image scale of displayed image, depending on options
    if scale_win2ima: imscale = zoom, window size adapted to image size
    else imscale adapted to window size
    :param imx: width of image
    :param imy: heigth of image
    :param opt_dict: options dictionary
    # :param window: main window of UI
    :return: imscale
    """
    if opt_dict['scale_win2ima']:  # fit window to image size
        imscale = opt_dict['zoom']
        opt_dict['win_width'] = max(int(imx * imscale), 600) + 390
        opt_dict['win_height'] = max(int(imy * imscale), 540) + 111
    else:  # fit image size to window
        max_width = opt_dict['win_width'] - 390
        max_height = opt_dict['win_height'] - 111
        imscale = min(max_width / imx, max_height / imy)
    return imscale


# -------------------------------------------------------------------
def draw_scaled_image(file, graph, opt_dict, idg, contr=1, tmp_image=False, resize=True, get_array=False):
    """
    main drawing routine, draws scaled image into graph window and stores image as BytesIO
    :param file: image file (.fit, .png, .jpg etc)
    :param graph: graph window to put graph
    :param opt_dict: setup parameters
    :param idg: graph number, used to delete previous graph
    :param contr: image brightness, default = 1
    :param tmp_image: if true, save scaled image as tmp.png
    :param resize: if true, resize image
    :param get_array: if true, returns numpy image array
    :return:
        data: ByteIO, for reuse with refresh_image
        idg: graph number
        file: name of image file, used for bookkeeping of displayed image (e.g. in window title)
        (image: numpy image array)
    """
    if not path.exists(file):
        sg.PopupError(f'file {file} not found', title='draw_scaled_image', keep_on_top=True)
        if get_array:
            return None, None, file, None
        else:
            return None, None, file
    im_scale = 1.0
    if file.lower().endswith('.fit'):
        image, header = get_fits_image(file)  # get numpy array and fits-header
        if np.max(image) > 0.0:
            image = image / np.max(image)
        ima = np.clip(image*contr, 0, 1)
        ima = np.flipud(np.uint8(255 * ima))  # converts floating point to int8-array
        # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
        # needed for imag.resize, converts numpy array to PIL format
        imag = Image.fromarray(np.array(ima))
    else:
        imag = Image.open(file)
        image = np.flipud(np.array(imag))
        if np.max(image) > 0.0:
            image = image / np.max(image)
    if resize:
        cur_width, cur_height = imag.size  # size of image
        im_scale = set_image_scale(cur_width, cur_height, opt_dict)
        imag = imag.resize((int(cur_width * im_scale), int(cur_height * im_scale)), Image.ANTIALIAS)
    bio = io.BytesIO()
    imag.save(bio, format="PNG")
    if tmp_image:
        imag.save('tmp.png')
    del imag
    data = bio.getvalue()
    # if idg: graph.delete_figure(idg)
    graph.Erase()
    idg = graph.draw_image(data=data, location=(0, opt_dict['graph_size']))
    graph.update()
    if get_array:
        return data, idg, file, image
    else:
        return data, idg, file


# -------------------------------------------------------------------
def refresh_image(data, graph, opt_dict, idg):
    """
    for redraw image from buffer data on different graph
    :param data: ByteIO buffer
    :param graph: graph window
    :param opt_dict: setup parameters
    :param idg: graph number, used to delete previous graph
    :return: idg
    """
    # if resize:
    #     data = get_img_data(data, resize=True)
    # TODO: see if refresh_image can be used for speed or multiple windows
    #   make resize work
    if idg:
        graph.delete_figure(idg)
    idg = graph.draw_image(data=data, location=(0, opt_dict['graph_size']))
    graph.update()
    return idg


# -------------------------------------------------------------------
def m_join(p, f=''):
    """
    make relative path if possible from directory and / or file
    :param p: directory
    :param f: file
    :return: relative norm path if possible, else abs norm path
    """
    n_path = path.join(p, f)
    try:
        n_path = path.relpath(n_path)
    except:
        pass
    return path.normpath(n_path)


# -------------------------------------------------------------------
def my_get_file(file_in, title='', file_types=(('ALL Files', '*.*'),), save_as=False,
                multiple_files=False, default_extension='', error_message='no file loaded'):
    tmp_file = path.basename(file_in)
    initial_folder = path.dirname(file_in)
    result_file = sg.PopupGetFile('', title=title, no_window=True, initial_folder=initial_folder,
                                  file_types=file_types, save_as=save_as, multiple_files=multiple_files,
                                  default_path=tmp_file, default_extension=default_extension, keep_on_top=True)
    if save_as and error_message == 'no file loaded':
        error_message = 'no file saved'
    if result_file:
        if multiple_files:
            result_file = list(result_file)
        else:
            result_file = m_join(result_file)
        info = f'{title}: {result_file}'
    else:
        info = f'{title}: {error_message}'
    print(info)
    logging.info(info)
    return result_file, info


# -------------------------------------------------------------------
def save_fit_png(imfilename, image, fits_dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ios.imsave(change_extension(imfilename, '.png'), np.flipud(image*255).astype(np.uint8))
    write_fits_image(image, str(change_extension(imfilename, '.fit')), fits_dict)


#-------------------------------------------------------------------
def select_options(opt_dict, ):
    zoom_elem = sg.Input(opt_dict['zoom'], key='-ZOOM-', size=(7, 1), tooltip='display image scale if scale_win2ima')
    cb_elem_debug = sg.Checkbox('debug', default=opt_dict['debug'], pad=(10, 0), key='-DEBUG-')
    cb_elem_fitreport = sg.Checkbox('fit-report', default=opt_dict['fit-report'],
                                    pad=(10, 0), key='-FIT_REPORT-')
    cb_elem_w2i = sg.Checkbox('scale_win2ima', default=opt_dict['scale_win2ima'],
                              pad=(10, 0), key='-W2I-', tooltip='if not checked scale image to window size')
    layout_options = [[sg.Text('Zoom', size=(6, 1)), zoom_elem],
                      [cb_elem_debug], [cb_elem_fitreport], [cb_elem_w2i],
                      [sg.Button('Apply', key='-APPLY_OPT-')]]
    options_window = sg.Window('Options', layout_options, keep_on_top=True, disable_minimize=True)
    while True:
        ev, vals = options_window.read(timeout=100)
        if ev is None:  # always give a way out!
            break
        if ev == '-APPLY_OPT-':
            opt_dict['zoom'] = float(zoom_elem.Get())
            opt_dict['debug'] = cb_elem_debug.Get()
            opt_dict['fit-report'] = cb_elem_fitreport.Get()
            opt_dict['scale_win2ima'] = cb_elem_w2i.Get()
            break
    options_window.close()
    return opt_dict
