# TIGER
The project repository for the TIGER challenge by group 9 of the course "NWI-IMC037 Intelligent Systems in Medical Imaging".

## Usage
Install the `environment.yml` to work with our code.

* To work with the high-resolution images, we used the `ASAP` reader that has been created by the _Radboud University Medical Center Computational Pathology Group_. The program can be downloaded from their [github repository](https://github.com/computationalpathologygroup/ASAP/releases).
  * Once the program has been installed, `bin` directory of the `ASAP` reader needs to be added to the pythonpath, for this, one can use <br>
  ```python
  import sys
  sys.path.append(r'C:\Program Files\ASAP 1.9\bin') # Fill in your own path here
  ```

  Thus far, we only got this to work with python versions < 3.7.x
