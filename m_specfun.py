# -------------------------------------------------------------------
# m_specfun functions for m_spec
# -------------------------------------------------------------------
import configparser
# -------------------------------------------------------------------
# logging
import logging
import platform
import subprocess
import warnings
import os
import time
from datetime import datetime, date
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy import optimize, interpolate
from scipy.ndimage import map_coordinates
from skimage import io, img_as_float
from skimage import transform as tf
import PySimpleGUI as sg
from PIL import ImageGrab

version = '0.9.17'
today = date.today()
logfile = 'm_spec' + today.strftime("%y%m%d") + '.log'
# turn off other loggers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logfile, format='%(asctime)s %(message)s', level=logging.INFO)
# -------------------------------------------------------------------
# initialize dictionaries for configuration
parv = ['1' for x in range(10)]+['' for x in range(10,15)]
parkey = ['f_lam0', 'f_scalxy', 'b_fitxy', 'i_imx', 'i_imy', 'f_f0', 'f_pix', 'f_grat', 'f_rotdeg', 'i_binning',
          's_comment', 's_infile', 's_outfil', 's_linelist', 'b_sqrt']
# parzip = zip(parkey, parv)
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
outpath = 'out'
mdist = 'mdist'
colorflag = False
optkey = ['zoom', 'win_width', 'win_height', 'win_x', 'win_y', 'calc_off_x',
          'calc_off_y', 'setup_off_x', 'setup_off_y', 'debug', 'fit-report',
          'scale_win2ima', 'comment', 'outpath', 'mdist', 'colorflag']
optvar = [zoom, wsize[0], wsize[1], wloc[0], wloc[1], xoff_calc, yoff_calc,
          xoff_setup, yoff_setup, debug, fit_report, win2ima, opt_comment, outpath, mdist, colorflag]
opt_dict = dict(list(zip(optkey, optvar)))


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
    if Path(conf).exists():
        config = configparser.ConfigParser()
        config.read(conf)
        for section in config.sections():
            if debug: print(f'[{section}]')
            partext += f'[{section}]\n'
            for key in config[section]:
                if debug: print(f'- [{key}] = ', {config[section][key]})
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
                        'setup_off_y'):
                    opt_dict[key] = int(config['Options'][key])
                elif key in ('debug', 'fit-report', 'scale_win2ima', 'scale_ima2win'):
                    opt_dict[key] = bool(int(config['Options'][key]))
                elif key in 'zoom':
                    opt_dict[key] = float(config['Options'][key])
                else:
                    opt_dict[key] = config['Options'][key]
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
    def Configsetbool(section, option, boolean):
        if boolean:
            config.set(section, option, '1')
        else:
            config.set(section, option, '0')

    config = configparser.ConfigParser()
    cfgfile = open(conf, 'w')
    if 'Lasercal' not in config: config.add_section('Lasercal')
    if 'Calib' not in config: config.add_section('Calib')
    if 'Fits' not in config: config.add_section('Fits')
    if 'Options' not in config: config.add_section('Options')
    for key in par_dict.keys():
        k = key[0]
        if k == 'b':
            Configsetbool('Lasercal', key, par_dict[key])
        elif k == 'i':
            config.set('Lasercal', key, str(par_dict[key]))
        elif k == 'f':
            config.set('Lasercal', key, str(par_dict[key]))
        elif k == 's':
            config.set('Lasercal', key, par_dict[key])
        else:
            print('unknown key in writeconf: ', key)
    for key in res_dict.keys():
        config.set('Calib', key, str(res_dict[key]))
    for key in fits_dict.keys():
        config.set('Fits', key.upper(), str(fits_dict[key]))
    for key in opt_dict.keys():
        if key in ('debug', 'fit-report', 'scale_win2ima', 'scale_ima2win'):
            Configsetbool('Options', key, opt_dict[key])
        else:
            config.set('Options', key, str(opt_dict[key]))
    config.write(cfgfile)
    cfgfile.close()


# -------------------------------------------------------------------

def write_fits_image(image, filename, fits_dict, dist=True):
    """
    writes image as 32-bit float array into fits-file
    :param image: np.array with image data, scaled to +/- 1.0, b/w or color
    :param filename: filename with extension .fit
    :param fits_dict: content of fits header
    :param dist: True: distorted image; False: undistorted image
    :return: None
    """
    if len(image.shape) == 3:
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
            if not key in ('D_SCALXY', 'D_X00', 'D_Y00', 'D_ROT', 'D_DISP0', 'D_A3', 'D_A5'):
                hdu.header[key] = fits_dict[key]
    hdul.writeto(filename, overwrite=True)
    hdul.close()


# -------------------------------------------------------------------

def get_png_image(filename, colorflag=False):
    """
    reads png image and converts to np.array
    :param filename: with extension 'png
    :param colorflag: True: colour image, False: image converted to b/w
    :return: image as 2 or 3-D array
    """
    image = np.flipud(img_as_float(io.imread(filename)))
    if not colorflag:
        image = np.sum(image, axis=2) / 3
    return image


# -------------------------------------------------------------------

def extract_video_images(avifile, pngdir, pngname, bobdoubler, binning, bff, maxim):
    """
    creates png images from AVI file
    :param avifile: filename of avi file (full path, with extension)
    :param pngdir: path to png images, e.g. tmp (is emptied before use)
    :param pngname: filebase of png images, e.g. m for series m1.png, m2.png,...
    :param bobdoubler: if True: interlaced frames are separated into fields of half height, 
                       default: False, frames are read
    :param binning: integer: 
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
        f = Path(p).name
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
    # print('Platform: ', sys)
    logging.info(f'Platorm: {sys}')
    p = Path(pngdir)
    out = pngdir + '\\' + pngname
    nim = 0
    dattim = ''
    sta = ''
    if avifile:
        # filename for png images
        if not p.exists():
            Path.mkdir(p, exist_ok=True)
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
                            Path(f'{pngdir}/bot' + str(nfr) + '.png').rename(Path(out + str(n) + '.png'))
                            n += 1
                            Path(f'{pngdir}/top' + str(nfr) + '.png').rename(Path(out + str(n) + '.png'))
                        else:
                            Path(f'{pngdir}/top' + str(nfr) + '.png').rename(Path(out + str(n) + '.png'))
                            n += 1
                            Path(f'{pngdir}/bot' + str(nfr) + '.png').rename(Path(out + str(n) + '.png'))
                    except:
                        end = True
                nim = n - 1

            elif binning > 1:
                # binning bin*bin for reducing file size
                command = f"ffmpeg -i {avifile} -frames {maxim} -vf scale=iw/{binning}:-1  {out}%d.png -loglevel quiet"
                subprocess.call(command, shell=cshell)
                nim = check_files(out, maxim)
                # for child in p.iterdir(): nim += 1

            else:
                # regular processing of frames
                command = f"ffmpeg -i {avifile} -frames {maxim} {out}%d.png -loglevel quiet"
                subprocess.call(command, shell=cshell)
                nim = check_files(out, maxim)
                # for child in p.iterdir(): nim += 1

            print(f'last file written: {out}' + str(nim) + '.png')
            # get dattim from filename
            dattim, sta = tfits(avifile)
        except:
            sg.PopupError('problem with ffmpeg, no images converted', title='AVI conversion')
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
        if Path(file + str(i + 1) + ext).exists():
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
        if answer is 'OK':
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

def apply_dark_distortion(im, backfile, outpath, mdist, first, nm, window, dist=False, background=False,
            center=None, a3=0, a5=0, rotation=0, yscale=1, colorflag=False, fits_dict=fits_dict, cval=0):
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
        ima = tf.rescale(back, (yscale, 1), multichannel=multichannel)  # scale sum and peak image start
        print('imy imx , x00 y00: ', ima.shape, center)
    else:
        ima = back
    imsum = 0 * ima
    impeak = imsum
    t1 = time.time()
    fullmdist = outpath + '/' + mdist
    for image in image_list:
        if Path(image).exists():
            a += 1  # create output filename suffix
            fileout = fullmdist + str(a)
            idist = get_png_image(image, colorflag)
            if background:
                idist = idist - back  # subtract background
            # calculate distortion
            if dist:
                if abs(yscale - 1.0) > 1.0e-3:  # scale image if yscale <> 1.0
                    idist = tf.rescale(idist, (yscale, 1), multichannel=multichannel)
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
            # create sum and peak image
            imsum = imsum + idist
            file = str(Path(fileout + '.fit').name)
            impeak = np.maximum(impeak, idist)
            disttext = f'{file} of {nm} done\n'
            window['-RESULT2-'].update(value=disttext, append=True)
            window.refresh()
    # write sum and peak fit-file
    write_fits_image(imsum, fullmdist + '_sum.fit', fits_dict, dist=dist)
    write_fits_image(impeak, fullmdist + '_peak.fit', fits_dict, dist=dist)
    nmp = a
    print(nmp, ' images processed of ', nm)
    logging.info(f'{nmp} images processed of {nm}')
    # t2 = time.time() - t0
    tdist = (time.time() - t1) / nmp
    disttext = f'{nmp} images processed of {nm}\n'
    if dist:
        logging.info(f'process time for single distortion: {tdist:8.2f} sec')
        # print(f'process time background, dark and dist {t2:8.2f} sec')
        print(f'process time for single distortion: {tdist:8.2f} sec')
        dattim = fits_dict['DATE-OBS']
        sta = fits_dict['M_STATIO']
        disttext += (
            # f'M{dattim}_{sta}\ncheck time!\nprocess time {t2:8.2f} sec\n'
            f'for single distortion: {tdist:8.2f} sec')
        logging.info(f"'DATE-OBS' = {dattim}")
        logging.info(f"'M-STATIO' = {sta}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(outpath + '/' + mdist + '_peak.png', np.flipud(impeak * 255).astype(np.uint8))
        # load_image(outpath + '/' + mdist + '_peak.png', opt_dict,
        #                                           colorflag=colorflag)
    return a, imsum, impeak, disttext


# -------------------------------------------------------------------

def register_images(start, nim, x0, y0, dx, dy, infile, outfil, window, fits_dict=fits_dict):
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
    if the procedure stops early, nim = index - start + 1
    :return:
    index: last processed image
    sum_image: !average! of registered images
    regtext: multiline text of results
    dist: if True, distorted images, else False
    outfile: filename of sum-image, e.g. for sum of 20 added images: out/r_add20
    fits_dict: updated values of fits-header
    """
    # dx, dy half width of rectangle
    index = start
    sum_image = []
    outfile = ''
    dist = False
    print('start x y dx dy: ', x0, y0, 2 * dx, 2 * dy)
    logging.info(f'start x y, dx dy, file: {x0} {y0},{2 * dx} {2 * dy}, {infile}')
    regtext = f'start x y, dx dy, file: {x0} {y0},{2 * dx} {2 * dy}, {infile}' + '\n'
    image_list = create_file_list(infile, nim, ext='', start=start)
    #      123456789012345678901234567890123456789012345678901234567890
    regtext += f'        file        peak      x         y    wx   wy\n'
    print('          file         peak     x      y     wx     wy')

    def _shift(xy):
        return xy - np.array(dxy)[None, :]

    try:
        for image_file in image_list:
            im, header = get_fits_image(image_file)
            if 'D_X00' in header.keys():
                dist = True
            if len(im.shape) == 3:
                imbw = np.sum(im, axis=2)  # used for _fit_gaussian_2d(data)
                data = imbw[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
                shifted = im
            # selected area
            else:
                data = im[y0 - dy:y0 + dy, x0 - dx:x0 + dx]
            params = _fit_gaussian_2d(data)
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
            # set new start value 
            x0 = int(y)
            y0 = int(x)
            index += 1  # next image
        index += -1
    except:
        # Exception, delete last image with error
        if Path(outfil + str(index - start + 1) + '.fit').exists():            
            os.remove(outfil + str(index - start + 1) + '.fit')
        index += -1
        info = f'problem with register_images, last image: {image_file}, number of images: {index}'
        print(info)
        logging.info(info)
        regtext += info + '\n'
    nim = index - start + 1
    if nim > 1:
        if index == nim + start - 1:
            outfile = outfil + '_add' + str(nim)
            sum_image = sum_image / (nim)  # averaging
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
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    # print('h: %5.1f'%height,'x: %5.1f'%x, 'y: %5.1f'%y, 'wx: %5.1f'%width_x, 'wy: %5.1f'%width_y)
    return height, x, y, width_x, width_y


# -------------------------------------------------------------------

def _fit_gaussian_2d(data):
    """Returns (height, x, y, width_x, width_y)
    the _gaussian parameters of a 2D distribution found by a fit
    (ravel makes 1-dim array)"""
    params = _moments(data)
    errorfunction = lambda p: np.ravel(_gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


# -------------------------------------------------------------------

def get_fits_keys(header, fits_dict, res_dict, keyprint=True):
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
            if keyprint: print(key, header[key])
    for key in res_dict.keys():
        fkey = 'D_' + key.upper()
        if fkey in header.keys():
            res_dict[key] = np.float32(header[fkey])
            fits_dict[fkey] = np.float32(header[fkey])
            if keyprint: print(key, fits_dict[fkey])
    return fits_dict


# -------------------------------------------------------------------

def get_fits_image(fimage):
    """
    reads fits image data and header
    fimage: filename without extension
    converts 32-bit floating values and 16-bit data to Python compatible values
    reads also color images and transposes matrix to correct order 
    (normalizes images to +/- 1 range)
    returns: image as np array, header
    """
    im, header = fits.getdata(fimage + '.fit', header=True)
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

def set_image_scale(image, opt_dict):
    """
    sets image scale of displayed image, depending on options
    if scale_win2ima: imscale = zoom, window size adapted to image size
    else imscale adapted to window size
    :param image: image array selected for display
    :param opt_dict: options dictionary
    :return: imscale
    """
    (imy, imx) = image.shape[:2]
    if opt_dict['scale_win2ima']:  # fit window to image size
        imscale = opt_dict['zoom']
        opt_dict['win_width'] = max(int(imx * zoom), 600) + 350
        opt_dict['win_height'] = max(int(imy * zoom), 540) + 111
    else:  # fit image size to window
        max_width = opt_dict['win_width'] - 350
        max_height = opt_dict['win_height'] - 111
        imscale = min(max_width / imx, max_height / imy)
    # if debug: print(max_width,max_height, imx, imy, imscale)
    return imscale

# -------------------------------------------------------------------


def show_image_array(image, imscale, image_element):
    """
    displays image in GUI
    :param image: image to be displayed
    :param imscale: scale to fit image in GUI
    :param image_element: where to display image
    :return: None
    """
    if len(image.shape) == 3:
        multichannel = True
    else:
        multichannel = False
    im = tf.rescale(image, imscale, multichannel=multichannel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave('tmp.png', np.flipud(im * 255).astype(np.uint8))
        image_element.update(filename='tmp.png')
        # if debug: print('show_image_array',imscale)
    return


# -------------------------------------------------------------------

def show_fits_image(file, imscale, image_element, contr=1.0):
    """
    loads fits-image, adjusts contrast and scale and displays in GUI
    :param file: fits-file with extension
    :param imscale: scale for displayed image
    :param image_element: where to display image in GUI
    :param contr: image contrast
    :return:
    """
    imbw, header = get_fits_image(file)
    if len(imbw.shape) == 2:
        im = tf.rescale(imbw, imscale, multichannel=False)
    else:
        im = tf.rescale(imbw, imscale, multichannel=True)
    # im = np.maximum(0.0, im)
    im = im / np.max(im) * 255 * contr
    im = np.clip(im, 0.0, 255)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave('tmp.png', np.flipud(im).astype(np.uint8))
    image_element.update(filename='tmp.png')
    return


# -------------------------------------------------------------------

def select_rectangle(infile, start, par_dict, res_dict, fits_dict, wloc, outfil, maxim):
    """
    displays new window with image infile + start + 'fit
    a rectangle around the selected line can be selected with dragging the mouse
    :param infile: filebase of image
    :param start: index of selected image
    :param par_dict: dictionary
    :param res_dict: "
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
    (imy, imx) = im.shape[:2]
    imbw = np.flipud(io.imread('tmp.png'))  # get shape
    (canvasy, canvasx) = imbw.shape[:2]
    print('canvas:', canvasx, canvasy)
    wlocw = (wloc[0] + 300, wloc[1] + 50)
    image_file = 'tmp.png'
    # check for old files
    delete_old_files(outfil, maxim, ext='.fit')
    par_dict['i_imx'] = imx
    par_dict['i_imy'] = imy
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
    dragging = False
    start_point = end_point = prior_rect = None
    x0 = y0 = dx = dy = 0
    while winselect_active:
        event, values = winselect.read()
        graph.draw_rectangle((0, 0), (imx, imy), line_color='blue')
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
                print('x0, y0, newr dx, dy: ', x0, y0, 2 * dx, 2 * dy)

        elif event in ('Ok', 'Cancel'):
            winselect_active = False
            winselect.close()
    return event, x0, y0, dx, dy


# -------------------------------------------------------------------

def add_rows_apply_tilt_slant(outfile, par_dict, res_dict, fits_dict, imscale, contr, wloc, restext, regtext, window):
    """
    displays new window with image outfile.fit for selection of rows to be added
    allows adjustment of tilt and slant after selection of rows
    if Ok, images outfile + ['st.fit', 'st,png'] are saved
    :param outfile:
    :param par_dict:
    :param res_dict:
    :param fits_dict:
    :param imscale:
    :param contr:
    :param wloc:
    :param restext:
    :param regtext:
    :param window:
    :return:
    Ok if selection is accepted
    tilt, slant: selected values for image outfile + ['st.fit', 'st,png']
    """
    def _slant_tilt_mapping(xy, center, dx, dy):
        """
        -	Calculate shifted coordinates:  xs = x' - (y'-y0)*dx (slant)
                                            ys = y' - (x'-x0)*dy (tilt)
        -	(Pixel value at x',y':           I'(x',y') = I(x,y) in the original image)
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
    im, header = get_fits_image(outfile)
    if 'D_X00' in header.keys():
        dist = True
    else:
        dist = False
    print(np.max(im))
    im = im / np.max(im)
    imtilt = im
    fits_dict = get_fits_keys(header, fits_dict, res_dict, keyprint=False)
    write_fits_image(imtilt, outfile + 'st.fit', fits_dict, dist=dist)
    # new rect_plt
    (imy, imx) = im.shape[:2]
    imbw = np.flipud(io.imread('tmp.png'))  # get shape
    (canvasy, canvasx) = imbw.shape[:2]
    # print('canvas:', canvasx, canvasy)
    wlocw = (wloc[0] + 300, wloc[1] + 100)
    image_file = 'tmp.png'
    # -------------------------------------------------------------------
    par_dict['i_imx'] = imx
    par_dict['i_imy'] = imy
    image_elem_sel = [sg.Graph(
        canvas_size=(canvasx, canvasy),
        graph_bottom_left=(0, 0),  # starts at top, set y-scale here
        graph_top_right=(imx, imy),  # set x-scale here
        key='-GRAPH-',
        change_submits=True,  # mouse click events
        drag_submits=True)]
    layout_select = [[sg.Text('Start File: ' + outfile, size=(50, 1)),
                      sg.Text('Tilt'), sg.InputText(tilt, size=(8, 1), key='-TILT-'),
                      sg.Text('Slant'), sg.InputText(slant, size=(8, 1), key='-SLANT-'),
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
    start_point = end_point = prior_rect = None

    while winselect_active:
        event, values = winselect.read()
        graph.draw_rectangle((0, 0), (imx, imy), line_color='blue')
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
                ymin = min(start_point[1], end_point[1])
                ymax = max(start_point[1], end_point[1])
                prior_rect = graph.draw_rectangle((0, ymin),
                                                  (imx, ymax), line_color='red')
        elif event is not None and event.endswith('+UP'):
            # The drawing has ended because mouse up
            y0 = int(0.5 * (start_point[1] + end_point[1]))
            w = int(abs(0.5 * (start_point[1] - end_point[1])))
            info = winselect["info"]
            info.update(value=f"selected lines from {ymin} to {ymax}")
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            print('ymin, ymax, y0, w: ', ymin, ymax, y0, w)
            restext += f"selected lines from {ymin} to {ymax}" + '\n'
            window['-RESULT3-'].update(regtext + restext)

        elif event == '-APPLY_TS-':
            if ymax == 0:
                sg.PopupError('select rows first', keep_on_top=True)
            else:
                try:
                    tilt = float(values['-TILT-'])
                    slant = float(values['-SLANT-'])
                    # print('tilt,slant', tilt,slant)
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
                    print(f'ymin = {ymin}, ymax = {ymax}, {ymax - ymin + 1} rows')
                    print(f'tilt = {tilt:8.4f}, slant = {slant:7.3f}')
                    restext += f'tilt = {tilt:8.4f}, slant = {slant:7.3f}' + '\n'
                    window['-RESULT3-'].update(regtext + restext, autoscroll=True)

                except:
                    sg.PopupError('bad values for tilt or slant, try again',
                                  keep_on_top=True)
                write_fits_image(imtilt, outfile + 'st.fit', fits_dict, dist=dist)
                show_fits_image(outfile + 'st', imscale, window['-R_IMAGE-'], contr)
                graph.draw_image(image_file, location=(0, imy)) if image_file else None
                graph.draw_rectangle((0, ymin), (imx, ymax), line_color='red')
                graph.update()

        elif event == 'Ok':
            show_fits_image(outfile + 'st', imscale, window['-R_IMAGE-'], contr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # im = np.maximum(0.0, imtilt)
                im = imtilt / np.max(imtilt) * 255 * contr
                im = np.clip(im, 0.0, 255)
                io.imsave(outfile + 'st.png', np.flipud(im.astype(np.uint8)))
            logging.info(f"start = {fits_dict['M_STARTI']}, nim = {fits_dict['M_NIM']}")
            logging.info(f'added from {ymin} to {ymax}, {(ymax - ymin + 1)} rows')
            logging.info(f'tilt = {tilt:8.4f}, slant = {slant:7.3f}')
            if len(imtilt.shape) == 3:
                imbw = np.sum(imtilt, axis=2)
            else:
                imbw = imtilt
            F = np.sum(imbw[ymin:ymax, :], axis=0)  # Object spectrum extraction and flat
            i = np.arange(0, np.size(F), 1)  # create pixels vector
            np.savetxt(outfile + '.dat', np.transpose([i, F]), fmt='%6i %8.5f')
            try:
                del fits_dict['M_TILT']
                del fits_dict['M_SLANT']
                del fits_dict['M_ROWMIN']
                del fits_dict['M_ROWMAX']
            except KeyError:
                print("Keys 'M_TILT... M_ROWMAX' not found")
            winselect_active = False
            winselect.close()
            window['-SAVE_RAW-'].update(disabled=False, button_color=bc_enabled)
            window['-CAL_R-'].update(disabled=False, button_color=bc_enabled)
            window['-RADD-'].update(outfile)

        elif event == 'Cancel':
            winselect_active = False
            winselect.close()
    return event, tilt, slant


# -------------------------------------------------------------------

def select_calibration_line(x0, w, lam, name, lcal, ical, graph, table, caltext):
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
    :param caltext: multiline info
    :return:
    coeff[1]: peak position of parabolic fit
    fwp: peak width (approximate)
    caltext: updated info
    """
    def parab(x, *p):
        A, mu, b = p
        return A * (1 - b * (x - mu) ** 2)

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
            lcr0 = lc[icleft:icright]
            lmax0 = np.max(lcr0)
            for i in range(icright - icleft):
                if (lcr0[i] - lmax0 + 1.e-5) > 0: m = i
            peak0 = icleft + m  # index of max value, center for parabolic fit
            icr = ic[peak0 - 2:peak0 + 3]
            lcr = lc[peak0 - 2:peak0 + 3]
            coeff[0] = lmax0
            coeff[1] = peak0
            coeff[2] = 1 / (w + 1)
            coeff, var_matrix = optimize.curve_fit(parab, icr, lcr, p0=coeff)
            # lcp_fit = parab(icr, *coeff) # function for display results
            x0p = coeff[1]
            fwp = np.sqrt(1.5 / coeff[2])
            # parabolic fit
            print(f'x0p ={x0p:8.2f} FWHMP={fwp:8.3f}')
            points = []
            l0: int
            for l0 in range(peak0 - 2, peak0 + 3):
                points.append((lcal[l0], parab(lcal[l0], *coeff)))
            for l0 in range(1, 5):
                graph.DrawLine(points[l0 - 1], points[l0], 'blue', 1)
            table.append((coeff[1], lam))
            info = f'{coeff[1]:8.2f} {fwp:6.2f} {lam:8.2f} {name}'
            caltext += info + '\n'
            logging.info(info)
        except:
            sg.PopupError('Select Line: no peak found, try again', title='Select line')
    return coeff[1], fwp, caltext


# -------------------------------------------------------------------

def graph_calibrated_spectrum(llist, lmin=0, lmax=720, imin=0, imax=1, autoscale=True, gridlines=True,
                              canvas_size=(800, 400), plot_title='Spectrum'):
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
    :return: None
    """
    # --------------------------------------------------------------
    def draw_spectrum(lcal, ical, lmin, lmax, color='blue'):
        for l0 in range(0, len(lcal)):
            if lmax > lmin:
                if lmin <= lcal[l0] <= lmax:
                    if l0:
                        graph.DrawLine((lcal[l0 - 1], ical[l0 - 1]), (lcal[l0], ical[l0]), color, 2)
            else:  # reverse axis for negative orders
                if lmin >= lcal[l0] >= lmax:
                    if l0:
                        graph.DrawLine((lcal[l0 - 1], ical[l0 - 1]), (lcal[l0], ical[l0]), color, 2)
    # --------------------------------------------------------------
    lcal, ical = np.loadtxt(llist, unpack=True, ndmin=2)
    p = ''
    if autoscale:
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
               sg.Text('Imin:'), sg.InputText('', key='imin', size=(12, 1)),
               sg.Text('Imax:'), sg.InputText('', key='imax', size=(12, 1)),
               sg.Button('Scale I', key='scaleI'), sg.Text('Cursor Position: '),
               sg.InputText('', size=(30, 1), key='cursor', disabled=True)]]

    window = sg.Window(llist, layout, keep_on_top=True).Finalize()
    graph = window['graph']

    # draw x-axis
    lamda = u'\u03BB'
    graph.DrawText(lamda + ' [nm]', ((lmax + lmin) / 2, imin - 30 / iscale), font='Arial 12')
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
    draw_spectrum(lcal, ical, lmin, lmax)
    while True:
        event, values = window.read()

        if event in (None, 'Close'):
            window.close()
            return p, imin, imax

        elif event is 'graph':  # if there's a "Graph" event, then it's a mouse
            x, y = (values['graph'])
            window['cursor'].update(f'Lambda:{x:8.2f}  Int:{y:8.2f}')

        elif event is 'Save':
            filename = sg.popup_get_file('Choose filename (PNG) to save to', save_as=True, keep_on_top=True,
                                         default_path=llist, default_extension='.png', size=(80,1))
            if filename:
                p = str(Path(filename).with_suffix('.png'))
                save_element_as_file(window['graph'], p)
            else:
                p = str(Path(llist).with_suffix('')) + '_plot.png'
            save_element_as_file(window['graph'], p)
            window.close()
            return p, imin, imax
        elif event is 'scaleI':
            try:
                imin = float(values['imin'])
                imax = float(values['imax'])
                iscale = canvas_size[1] / (imax - imin)
                graph.change_coordinates((lmin - 40 / lscale, imin - 40 / iscale),
                                         (lmax + 10 / lscale, imax + 30 / iscale))
                draw_spectrum(lcal, ical, lmin, lmax, color='red')
                graph.update()
            except:
                sg.PopupError('invalid values for Imin, Imax, try again', keep_on_top=True)


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

def create_line_list_combo(m_linelist, window):
    """
    shows values of table create_line_list_combo in Combobox
    :param m_linelist: table with wavelength, line identifier (space separated)
    :param window: Combobox for selecting wavelength
    :return: None
    """
    try:
        lam_calib = []
        i = -1
        with open(m_linelist + '.txt')as f:
            for x in f:
                x = x.lstrip()
                (l, name) = x.split(' ', 1)
                i += 1
                lam_calib.append(x)
                if abs(float(l)) < 0.1:
                    index0 = i  # set default index for list
        window['-LAMBDA-'].update(values=lam_calib, set_to_index=index0)
    except:
        sg.PopupError(f'no calibration lines {m_linelist}.txt found, use default')


# -------------------------------------------------------------------

def read_video_list(file):
    """
    reads list of latest converted video files from table
    :param file: table of video files
    :return: list of video files
    """
    video_list = []
    if Path(file).exists():
        with open(file, 'r') as f:
            for line in f:
                video_list.append(line[:-1])
    return video_list


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
    # graph = window['graph']
    graph.change_coordinates((lmin, imin), (lmax, imax))
    # erase graph with rectangle
    graph.DrawRectangle((lmin, imin), (lmax, imax), fill_color='white', line_width=1)
    graph.DrawText(rawspec, (0.5 * (lmax - lmin), 0.95 * (imax - imin)))
    # draw graph
    for l0 in range(0, len(lcal)):
        if lmin <= lcal[l0] < lmax:
            graph.DrawCircle(points[l0], 2 / canvasx, line_color='red', fill_color='red')
            if l0:
                graph.DrawLine(points[l0 - 1], points[l0], 'red', 1)
    return lmin, lmax, imin, imax, lcal, ical


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
    print('c Peak =: ', c)
    logging.info(f'polynom for fit lambda c: {c}')
    i = np.arange(0, len(lcal), 1)  # create pixels vector for uncalibrated image
    lam = np.poly1d(c)(i)
    res = np.poly1d(c)(xcalib) - lcalib
    rms_x = np.sqrt(np.average(np.square(res)))
    print(f'rms_x = {rms_x:8.4f}')
    # np.set_printoptions(precision=3, suppress=True)
    print('    pixel    lambda    fit      error\n',
          np.transpose(np.array([xcalib, lcalib, lcalib + res, res])))
    logging.info('    pixel     lambda      fit        error')
    caltext = '   Pixel     lambda        fit    error\n'
    for i in range(0, len(xcalib)):
        logging.info(f'{xcalib[i]:10.2f},{lcalib[i]:10.2f},{(lcalib[i] + res[i]):10.2f}, {res[i]:10.4f}')
        caltext += f'{xcalib[i]:9.2f} {lcalib[i]:9.2f} {(lcalib[i] + res[i]):9.2f}  {res[i]:8.2f}\n'
    logging.info(f'rms_x = {rms_x:8.4f}')
    caldat = str(Path(rawspec).with_suffix('')) + 'cal.dat'
    np.savetxt(caldat, np.transpose([lam, ical]), fmt='%8.3f %8.5f')
    # x,y,z equal sized 1D arrays
    logging.info(f'spectrum {caldat} saved')
    caltext += f'polynom degree: {deg}\npolynom for fit lambda c: {c}\n'
    caltext += f'rms_x = {rms_x:8.4f}\nspectrum {caldat} saved\n'
    # for compatibility save *.dat with linear spacing
    lmin = np.int(np.min(lam)) + 1
    lmax = np.int(np.max(lam)) - 1
    dell = int(5 * c[deg - 1]) / 10
    # wavelength spacing of interpolated linear array, about double of original
    llin = np.arange(lmin, lmax, dell)
    y2 = interpolate.interp1d(lam, ical, kind='quadratic')(llin)
    cal2dat = str(Path(rawspec).with_suffix('')) + 'cal2.dat'
    np.savetxt(cal2dat, np.transpose([llin, y2]), fmt='%8.3f %8.5f')
    return caldat, cal2dat, lmin, lmax, caltext
