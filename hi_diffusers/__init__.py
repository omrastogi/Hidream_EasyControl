# There is no problem with the import statements themselves—they are valid Python imports.
# However, if you are asking "what is the problem here" in the context of this code,
# there is no explicit problem unless:
#   - The modules or classes do not exist at the specified paths.
#   - The __init__.py file is not exposing all the intended public API.
#   - You want to add __all__ for explicit export control.
# Here is a version with __all__ for clarity:

from .models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from .pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline

__all__ = [
    "HiDreamImageTransformer2DModel",
    "HiDreamImagePipeline",
]
