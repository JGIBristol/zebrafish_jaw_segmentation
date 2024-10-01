# zebrafish_jaw_segmentation
![Pylint](https://github.com/JGIBristol/zebrafish_jaw_segmentation/actions/workflows/pylint.yml/badge.svg?branch=main)
Segmentation of Zebrafish Jawbones from uCT scans using a U-Net

TODO
- [ ] README (including credit for Wahab)
- [x] environment.yml
- [ ] GitHub actions - tests, lint, typing
- [x] Model train script + example notebook
- [x] config files for hard-coded config, user config
- [ ] hyperparameter tuning script + example notebook
- [ ] mesh script + example notebook
- [ ] compare accuracy to a sensible baseline
- [ ] latex paper with build script
- [ ] full documentation

Scripts:
- [x] Create DICOMs
- [x] View DICOMs

### Config Files
There are two config files for this project - `config.yml` and `userconf.yml`.

 - You probably don't need to edit anything in `config.yml`.
 - `userconf.yml` contains things that you might want to edit - model parameters etc.
    The training script reads the parameters from this file before it begins training.
