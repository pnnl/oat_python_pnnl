# Open Applied Topology

[Open applied topology (OAT)](https://openappliedtopology.github.io) is a library for fast, user-friendly algebra and topology. OAT has 

- a user-friendly frontend for Python users, called [oat_python](https://github.com/OpenAppliedTopology/oat_python)
- a fast backend written in Rust, called [oat_rust](https://github.com/OpenAppliedTopology/oat_rust) 
- a variety of tutorials published as [jupyter notebooks](https://openappliedtopology.github.io)

This package contains the source code for [oat_python](https://github.com/OpenAppliedTopology/oat_python).


# Caution: breaking changes

OAT is in early stages of develpoment, and it's evolving quickly. Code that you write today may not work tomorrow, due to these changes. We will do our very best to make sure that if/when this happens, you will only need to make small changes to your code to fix the problem (e.g., updating the name of a function). However, please do bear this in mind as you write your code!

# Install and explore

**Python users** Can install oat_python through [PyPI](https://pypi.org/project/oat_python/) using any package manager, for example `pip install oat_python`, `conda_install oat_python`, etc. Explore the [Jupyter notebook tutorials on GitHub](https://github.com/OpenAppliedTopology/oat)!


**Developers** Everyone is a developer, and everyone is invited to modify and extend the source code for this package! oat_python is a combination of Rust and Python code. We developed oat_python using `PyO3` and `maturin`. To download and modify oat_python, then install and use the modified version, check out the instructions for Python [Installation from source](#python-installation-from-source), below.

# Documentation

Documentation for OAT-Python is currently under development. 

**Python users** The best resources currently available are
- [Jupyter notebook tutorials](https://openappliedtopology.github.io) available on the OAT homepage
- Docstrings available through Python's `help()` function.

**Rust Developers** If you are interested in modifying the Rust code in OAT-Python, see the API documenation for oat_python available at [Crates.io](https://crates.io/crates/oat_python). You may also find it helpful to explore the documenation for PyO3 and Maturin, which are the packages used to link Rust and Python.

**Python Developers** Documentation for the Python API is not available at this time (we are working to address this). In the meantime we have done our best to document the code with docstrings, and encourage you to check out the source code directly. 

# Contributing

For information on **contributing**, see [`CONTRIBUTING.md`](https://github.com/OpenAppliedTopology/oat_python/blob/main/CONTRIBUTING.md).

# License

For inforamtion on copyright and licensing, see [`LICENSE`](https://github.com/OpenAppliedTopology/oat_python/blob/main/LICENSE).

# Attributions

OAT is an extension of the ExHACT library. See [`ATTRIBUTIONS.md`](https://github.com/OpenAppliedTopology/oat_python/blob/main/ATTRIBUTIONS.md) for details.

# Python Installation from source

1. Download and install the most recent version of [Rust](https://www.rust-lang.org/).  Make sure your installation is up to date by running `rustup update` in a command shell.

2. Create a virtual Python environment, e.g. using Anaconda, or open one that you already have.  In this example, let's assume the environment name is `myenv`.  Activate `myenv`, and run

    ```bash
    pip install maturin
    ```

    A number of warning messages may appear; this is normal. 

    **If you already have maturin installed, make sure it is up to date!**

3. [Clone](https://github.com/OpenAppliedTopology/oat_python) a copy of oat_python. Open a shell and CD into the oat_python folder.  Activate `myenv` and run

    ```bash
    maturin develop --release
    ```
    
5. oat_python should now be installed.  Try running the Jupyter notebooks with `myenv`!
