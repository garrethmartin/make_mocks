# make_mocks/__init__.py

from .make_mocks import *
from .smooth_3d import *
import sys as _sys
smooth_3d = _sys.modules[__name__ + '.smooth_3d']
del _sys
