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
]

# NOTE: autodoc/napoleon/viewcode are disabled because baoiad has heavy
# torch/mmcv dependencies that hang during import even with mocks.
# Re-enable them in a CI environment with all deps installed.

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
try:
    import pytorch_sphinx_theme
    html_theme = 'pytorch_sphinx_theme'
    html_theme_options = {
        'logo': {
            'text': 'BaoIAD',
        },
        'menu': [
            {
                'name': 'GitHub',
                'url': 'https://github.com/xxx/BaoIAD',
            },
        ],
    }
except ImportError:
    try:
        import sphinx_rtd_theme
        html_theme = 'sphinx_rtd_theme'
    except ImportError:
        html_theme = 'alabaster'
    html_theme_options = {
        'logo_only': False,
        'display_version': True,
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

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
