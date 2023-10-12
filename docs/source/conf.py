# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Buscarpy'
copyright = '2023, Max Callaghan'
author = 'Max Callaghan'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix()+'/src/buscarpy')

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.apidoc'
]

autosummary_generate = True

apidoc_module_dir = '../../buscarpy/'
apidoc_output_dir = 'reference'
apidoc_excluded_paths = ['tests']
apidoc_separate_modules = True

sphinx_gallery_conf = {
    'examples_dirs': 'examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'reference_url': {
         # The module you locally document uses None
        'buscarpy': None,
    },
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module'          : ('buscarpy',),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
