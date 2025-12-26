# __init__.py for the HTSAT package

"""
This file makes the folder a Python package.

It re-exports the main HTSAT_Swin_Transformer class so you can do:
    from htsat import HTSAT_Swin_Transformer
OR
    from htsat.htsat import HTSAT_Swin_Transformer
"""

from .htsat import HTSAT_Swin_Transformer
