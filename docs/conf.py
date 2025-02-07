import tonic

# Project information
project = "Tonic"
copyright = "2019-present, the neuromorphs of Telluride"
author = "Gregor Lenz"
master_doc = "index"

# Sphinx extensions
extensions = [
    "autoapi.extension",
    "myst_nb",
    "pbr.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

# Sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "gallery/",
    "gallery_dirs": "auto_examples",
    "backreferences_dir": None,
    "matplotlib_animations": True,
    "doc_module": ("tonic",),
    "download_all_examples": False,
    "ignore_pattern": r"utils\.py",
}

# AutoAPI configuration
autodoc_typehints = "both"
autoapi_type = "python"
autoapi_dirs = ["../tonic"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# Napoleon settings for docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST-NB settings
nb_execution_timeout = 300
nb_execution_show_tb = True
nb_execution_excludepatterns = ["large_datasets.ipynb"]
suppress_warnings = ["myst.header"]

# Paths for templates and static files
templates_path = ["_templates"]
html_static_path = ["_static"]

# Patterns to exclude from processing
exclude_patterns = [
    "auto_examples/**.ipynb",
    "auto_examples/**.py",
    "auto_examples/**.md5",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
    "README.rst",
]

# HTML output options
html_theme = "sphinx_book_theme"
html_title = tonic.__version__
html_logo = "_static/tonic-logo-black-bg.png"
html_favicon = "_static/tonic_favicon.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

# HTML theme options
html_theme_options = {
    "repository_url": "https://github.com/neuromorphs/tonic",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_fullscreen_button": True,
}
