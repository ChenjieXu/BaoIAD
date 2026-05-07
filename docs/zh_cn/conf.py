# Configuration file for the Sphinx documentation builder (Chinese).

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'BaoIAD'
copyright = '2025, BaoIAD Contributors'
author = 'BaoIAD Contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
]

# NOTE: autodoc is enabled only so existing ``automodule`` directives parse.
# Heavy torch/mmcv-style dependencies stay mocked below for Read the Docs.

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

# -- Language and locale -----------------------------------------------------

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------

try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'
html_theme_options = {
    'logo_only': False,
}

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
