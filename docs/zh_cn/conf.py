# Configuration file for the Sphinx documentation builder (Chinese).

import os
import sys
from importlib.util import find_spec

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "BaoIAD"
copyright = "2026, Baosight-xVue and BaoIAD Contributors"
author = "Baosight-xVue and BaoIAD Contributors"
release = "1.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
]

# NOTE: autodoc is enabled only so existing ``automodule`` directives parse.
# Heavy torch/mmcv-style dependencies stay mocked below for Read the Docs.

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# -- Language and locale -----------------------------------------------------

language = "zh_CN"

# -- Options for HTML output -------------------------------------------------

if find_spec("sphinx_rtd_theme") is not None:
    html_theme = "sphinx_rtd_theme"
else:
    html_theme = "alabaster"
html_theme_options = {
    "logo_only": False,
}

# The Chinese documentation currently has no locale-specific static assets.
html_static_path = []

# -- Options for autodoc -----------------------------------------------------

autodoc_mock_imports = [
    "matplotlib",
    "torch",
    "torchvision",
    "mmcv",
    "mmengine",
    "timm",
    "sklearn",
    "scipy",
    "faiss",
    "FrEIA",
    "open_clip",
    "einops",
    "cv2",
    "numpy",
    "PIL",
    "geomloss",
    "groundingdino",
    "segment_anything",
    "mmpretrain",
    "imgaug",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "mmengine": ("https://mmengine.readthedocs.io/en/latest/", None),
}

# -- Options for MyST parser -------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_heading_anchors = 3

# Many Markdown pages intentionally link to repository-root sources that are
# outside the Sphinx source tree. Render them as ordinary links rather than
# unresolved cross-reference targets.
myst_all_links_external = True

# -- Options for copybutton --------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
