# meteor-spectrum-calibration
Analysis of meteor spectra with windows GUI.

Newest version with calculation and application of instrument response function.

The previous calibration of camera, lens and grating combination has been included in the script. Therefore only a single GUI allows this calibration and the subsequent processing of meteor spectra. For details of the calibration, see:
https://github.com/meteorspectroscopy/calibrate-spectrum
<img src= https://github.com/meteorspectroscopy/meteor-spectrum-calibration/blob/master/doc/m_spec%20calib%20calib.PNG>

After the sucessful instrument calibration, the processing of meteor spectra can be done easily. It starts with a video file containing a meteor spectrum. The video is converted to video frames. A background image is computed and subtracted from the meteor images. Meteor images are transformed to orthographic projection in order to linearize spectra. These spectra are registered and added. After correcting tilt and slant the 2-D spectra are converted to 1-D spectra by adding rows. These raw spectra are calibrated using known meteor lines for calibration. The result can be plotted.

Calibration page:
<img src= https://github.com/meteorspectroscopy/meteor-spectrum-calibration/blob/master/doc/m_spec%20calib%20spec.PNG>

Result: calibrated meteor spectrum as file wavelength vs. intensity .dat
<img src= https://github.com/meteorspectroscopy/meteor-spectrum/blob/master/doc/m_spec%20plot%20spectrum.PNG>

Instrument response page:
<img src = https://github.com/meteorspectroscopy/meteor-spectrum-calibration/blob/response/doc/response%20calculation.PNG>

For a description of the processing see: https://meteorspectroscopy.org/2020/03/27/meteor-spectra-analysis-new-version/
or the manual in the doc folder.

For a description on the Instrument response (theory and application) see the response manual in the doc folder: 
https://github.com/meteorspectroscopy/meteor-spectrum-calibration/blob/response/doc/instrument%20response.pdf

Further information about the theoretical background can be found at https://meteorspectroscopy.org/

