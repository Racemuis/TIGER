# TIGER
The project repository for the TIGER challenge by group 9 of the course "NWI-IMC037 Intelligent Systems in Medical Imaging".

## Usage
* Install the `environment.yml` to work with our code.
  ```
  conda env create -f environment.yml
  ```

* To work with the high-resolution images, we used the `ASAP` reader that has been created by the _Radboud University Medical Center Computational Pathology Group_. The program can be downloaded from their [github repository](https://github.com/computationalpathologygroup/ASAP/releases) (please download `ASAP 2.0`).
  * Once the program has been installed, `bin` directory of the `ASAP` reader needs to be added to the pythonpath, for this, one can use <br>
  ```python
  import sys
  sys.path.append(r'C:\Program Files\ASAP 2.0\bin') # Fill in your own path here
  ```

Alternatively, you can permanently add the `*\ASAP 2.0\bin` folder to your python path under `system properties -> Advanced -> Environment variables`.


