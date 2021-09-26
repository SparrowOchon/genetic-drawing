# Genetic Drawing
The following is a heavily modified version of the original [project](https://github.com/anopara/genetic-drawing) which focuses on the speed
of runtime and ease of use of the codebase. Changes were made to reduce runtime and code clutter. Additional changes could be made if additional 
speed is required specifically the use of `pandas` as well as `multiprocessing` pool focusing specifically on the DNA evolution and 
sampling from DNA will result in the greatest additional boost in performance as `numpy.choice` takes 1/3 of the remaining runtime.

Examples of generated images:

![](imgs/img1.gif) <img src="imgs/img2.gif" width="380">

It also supports user-created sampling masks, in case you'd like to specify regions where more brushstrokes are needed (for ex, to allocate more finer details)


<img src="imgs/img3.gif">


## Usage
- Run the main script `python3 main.py`

## Installation
The following fork makes of conda to avoid dirtying the environment

- Building conda environment `conda env create --name genetic-drawing --file environment.yml`
- Starting env `conda activate genetic-drawing`

