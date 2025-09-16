# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import oat_python


import plotly.io
import plotly.io._sg_scraper
plotly.io.renderers.default = 'sphinx_gallery_png'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))



project = 'oat_python'
copyright = '2025, The Open Applied Topology Developers'
author = 'The Open Applied Topology Developers'
release = '0.2.0'
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_gallery.gen_gallery',        
    'sphinx_autodoc_typehints',   # for some reason this has to come towards the end of the list; otherwise errors occur  
    'sphinx.ext.viewcode',
]

autosummary_generate = True
autosummary_imported_members = True
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 1


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for sphix-gallery extension ---------------------------------------
sphinx_gallery_conf = {
     'examples_dirs': ['../examples'],   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'image_scrapers': ('matplotlib', 'plotly.io._sg_scraper.plotly_sg_scraper',),  # NOTE: plotly requires
     'nested_sections': False,  # Setting this to False avoids certain undesired nesting behavior of sections in the sidebar,
     'within_subsection_order': 'ExampleTitleSortKey',  # Order examples by their title (instead of filename)
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_theme = "sphinx_rtd_theme" # 'alabaster'


# -- Increase the width of the sidebar -------------------------------------------------
def setup(app):
    app.add_css_file('sidebar_width.css')