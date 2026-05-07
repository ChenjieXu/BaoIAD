# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'BaoIAD'
copyright = '2025, BaoIAD Contributors'
author = 'BaoIAD Contributors'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
]

# NOTE: autodoc is enabled only so existing ``automodule`` directives parse.
# Heavy torch/mmcv-style dependencies stay mocked below for Read the Docs.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    # Superseded legacy tutorials are kept in the repo for compatibility but
    # are intentionally excluded from the Phase 1 docs IA.
    'user_guides/new_model.md',
    'user_guides/new_dataset.md',
]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# Prefer the Read the Docs theme on Read the Docs.  The optional
# ``pytorch_sphinx_theme`` currently fails under the Sphinx version resolved by
# RTD with ``UndefinedError("'style' is undefined")``.
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'
html_theme_options = {
    'logo_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files.
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------

autodoc_mock_imports = [
    'matplotlib',
    'torch',
    'torchvision',
    'mmcv',
    'mmengine',
    'timm',
    'sklearn',
    'scipy',
    'faiss',
    'FrEIA',
    'open_clip',
    'einops',
    'cv2',
    'numpy',
    'PIL',
    'geomloss',
    'groundingdino',
    'segment_anything',
    'mmpretrain',
    'imgaug',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mmengine': ('https://mmengine.readthedocs.io/en/latest/', None),
}

# -- Options for MyST parser -------------------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
]

myst_heading_anchors = 3

# BaoIAD docs intentionally link to many repo-root sources (configs/, tools/,
# baoiad/, docs/alignment/) that are not Sphinx source documents. Treat MyST
# markdown links as links instead of unresolved cross-reference targets. RST
# toctrees still own documentation reachability.
myst_all_links_external = True

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
