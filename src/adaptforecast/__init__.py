"""AdaptForecast: auditable demand forecasting for small datasets."""

from .schema import CANONICAL_COLUMNS, DataValidationError, validate_dataframe

__all__ = ["CANONICAL_COLUMNS", "DataValidationError", "validate_dataframe"]
__version__ = "0.2.0"
