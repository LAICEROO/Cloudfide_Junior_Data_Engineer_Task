import logging
import pandas as pd
import re
from typing import Optional, Tuple

# Logger configuration 
logging.basicConfig(
    level=logging.WARNING,
    format="%(Y-%m-%d %H:%M:%S) %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Regex for validating labels: only letters and underscores
LABEL_REGEX = re.compile(r"^[A-Za-z_]+$")

def is_valid_label(label: str) -> bool:
    """
    Check if the label contains only letters and underscores.
    """
    return bool(LABEL_REGEX.fullmatch(label))


def parse_role(expression: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse an arithmetic expression of the form "col1 op col2".
    Returns a tuple (col1, operator, col2) or None if invalid.
    Supported operators: +, -, *, /.
    """
    pattern = re.compile(r"^\s*([A-Za-z_]+)\s*([+\-*/])\s*([A-Za-z_]+)\s*$")
    match = pattern.fullmatch(expression)
    return match.groups() if match else None


def add_virtual_column(
    df: pd.DataFrame,
    role: str,
    new_column: str
) -> pd.DataFrame:
    """
    Add a new column to the DataFrame based on an arithmetic expression.

    Parameters:
        df (pd.DataFrame): Source DataFrame.
        role (str): Arithmetic expression "col1 op col2".
        new_column (str): Name of the new column (letters and underscores only).

    Returns:
        pd.DataFrame: Copy of df with the new column, or an empty DataFrame on error.
    """
    # Validate new column name
    if not is_valid_label(new_column):
        logger.warning("Invalid new column name: %r", new_column)
        return pd.DataFrame([])

    # Parse the role expression
    parsed = parse_role(role)
    if not parsed:
        logger.warning("Invalid expression format: %r", role)
        return pd.DataFrame([])
    left_col, op, right_col = parsed

    # Validate operand labels and existence in DataFrame
    for col in (left_col, right_col):
        if not is_valid_label(col):
            logger.warning("Invalid operand label: %r", col)
            return pd.DataFrame([])
        if col not in df.columns:
            logger.warning("Column not found in DataFrame: %r", col)
            return pd.DataFrame([])

    # Map operators to functions
    operations = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b,
    }
    func = operations.get(op)
    if func is None:
        logger.warning("Unsupported operator: %r", op)
        return pd.DataFrame([])

    # Perform the calculation
    try:
        result = func(df[left_col], df[right_col])
    except Exception as e:
        logger.warning(
            "Error during calculation %r %s %r: %s", left_col, op, right_col, e
        )
        return pd.DataFrame([])

    # Return a new DataFrame with the virtual column
    df_copy = df.copy()
    df_copy[new_column] = result
    return df_copy
